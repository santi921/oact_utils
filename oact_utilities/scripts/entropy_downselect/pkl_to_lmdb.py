"""Convert architector output pickle files to LMDB for fairchem inference.

Each row's mol2string is parsed directly (no architector dependency) to extract
positions and elements, then stored as pickled dicts in raw LMDB.
Uses batch transactions for fast writes (~2.2M entries in minutes, not hours).
"""

from __future__ import annotations

import argparse
import gc
import os
import pickle
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms
from tqdm import tqdm

METADATA_COLUMNS = [
    "name",
    "gen_unique_name",
    "is_distorted",
    "distortion_variant_index",
    "metal",
    "met_chrg_spn",
]

KEEP_COLUMNS = [
    "mol2string",
    "total_charge",
    "calc_n_unpaired_electrons",
] + METADATA_COLUMNS


def debug_log(message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [pid={os.getpid()}] {message}", flush=True)


_ATOM_SECTION_RE = re.compile(r"@<TRIPOS>ATOM")
_ELEMENT_FROM_TYPE = re.compile(r"^([A-Z][a-z]?)")


def parse_mol2(mol2_str: str) -> tuple[list[str], np.ndarray]:
    """Parse a mol2 string into element symbols and positions.

    Returns:
        Tuple of (symbols list, positions array shape (N,3)).
    """
    lines = mol2_str.strip().split("\n")
    in_atom_block = False
    symbols = []
    positions = []
    for line in lines:
        if _ATOM_SECTION_RE.search(line):
            in_atom_block = True
            continue
        if in_atom_block and line.startswith("@"):
            break
        if not in_atom_block:
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        atom_type = parts[5]
        match = _ELEMENT_FROM_TYPE.match(atom_type)
        if not match:
            match = _ELEMENT_FROM_TYPE.match(parts[1])
        symbols.append(match.group(1))
        positions.append([float(parts[2]), float(parts[3]), float(parts[4])])
    return symbols, np.array(positions, dtype=np.float64)


def row_to_entry(
    idx: int,
    mol2_str: str,
    charge: int,
    n_unpaired: int,
    metadata: dict,
) -> dict | None:
    """Convert one row to a storable dict with ASE Atoms.

    Returns dict on success, None on failure (logged separately).
    """
    try:
        symbols, positions = parse_mol2(mol2_str)
        atoms = Atoms(symbols=symbols, positions=positions)
        atoms.info = {
            "charge": int(charge),
            "spin": int(n_unpaired) + 1,
            "orig_idx": idx,
            **metadata,
        }
        return {
            "atoms": atoms,
            "natoms": len(atoms),
        }
    except Exception:
        return None


def convert_pickle_to_lmdb(
    pkl_path: Path,
    output_path: Path,
    chunk_size: int = 50_000,
) -> tuple[Path, int, int]:
    """Load pickle, convert mol2 -> Atoms, write raw LMDB with batch transactions."""
    import lmdb

    debug_log(f"Loading pickle: {pkl_path}")
    t0 = time.time()
    df = pd.read_pickle(pkl_path)
    debug_log(f"Loaded {len(df)} rows in {time.time() - t0:.1f}s")

    missing = [c for c in KEEP_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[KEEP_COLUMNS].copy()
    gc.collect()
    debug_log(f"Trimmed to {len(KEEP_COLUMNS)} columns, memory freed")

    total_rows = len(df)

    # Resume: check existing LMDB
    existing = 0
    if output_path.exists():
        try:
            env = lmdb.open(str(output_path), readonly=True, lock=False)
            with env.begin() as txn:
                length_val = txn.get(b"length")
                if length_val is not None:
                    existing = pickle.loads(length_val)
            env.close()
        except Exception:
            existing = 0
    if existing > 0:
        debug_log(f"Resuming: {existing} entries already written, skipping")
        df = df.iloc[existing:]
        total_rows = len(df)

    debug_log(f"Processing {total_rows} rows, chunk_size={chunk_size}")

    map_size = 20 * (1 << 30)  # 20 GB
    env = lmdb.open(
        str(output_path),
        map_size=map_size,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    total_written = existing
    total_failed = 0
    natoms_list = []
    failures_log = output_path.with_suffix(".failures.log")

    for chunk_start in tqdm(range(0, total_rows, chunk_size), desc="Chunks"):
        chunk = df.iloc[chunk_start : chunk_start + chunk_size]
        entries = []
        chunk_failed = 0

        for i, (_, row) in enumerate(chunk.iterrows()):
            metadata = {}
            for col in METADATA_COLUMNS:
                if col in row.index:
                    v = row[col]
                    metadata[col] = v.item() if hasattr(v, "item") else v

            entry = row_to_entry(
                idx=existing + chunk_start + i,
                mol2_str=row["mol2string"],
                charge=int(row["total_charge"]),
                n_unpaired=int(row["calc_n_unpaired_electrons"]),
                metadata=metadata,
            )
            if entry is not None:
                entries.append(entry)
            else:
                chunk_failed += 1

        # Batch write entire chunk in one transaction
        txn = env.begin(write=True)
        for entry in entries:
            key = f"{total_written}".encode("ascii")
            txn.put(key, pickle.dumps(entry["atoms"], protocol=-1))
            natoms_list.append(entry["natoms"])
            total_written += 1
        txn.commit()

        if chunk_failed > 0:
            total_failed += chunk_failed
            with open(failures_log, "a") as f:
                f.write(f"chunk_start={chunk_start}: {chunk_failed} failures\n")

        del entries, chunk
        gc.collect()

    # Write length key
    txn = env.begin(write=True)
    txn.put(b"length", pickle.dumps(total_written, protocol=-1))
    txn.commit()

    env.sync()
    env.close()

    debug_log(f"Done: {total_written} written, {total_failed} failed")
    return output_path, total_written, total_failed, np.array(natoms_list, dtype=np.int32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert architector output pickle to LMDB for fairchem inference."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to input pickle file with mol2string column.",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path for output .lmdb file.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        help="Rows per processing chunk (default: 50000).",
    )
    parser.add_argument(
        "--create-metadata",
        action="store_true",
        help="Create metadata.npz after conversion.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pkl_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    debug_log(f"Converting {pkl_path} -> {output_path}")

    out, written, failed, natoms = convert_pickle_to_lmdb(
        pkl_path=pkl_path,
        output_path=output_path,
        chunk_size=args.chunk_size,
    )

    debug_log(f"Result: {written} entries in {out}, {failed} failures")

    if args.create_metadata:
        npz_path = output_path.parent / (output_path.stem + "_metadata.npz")
        np.savez_compressed(str(npz_path), natoms=natoms)
        debug_log(f"Created {npz_path}")

    sys.exit(0 if failed == 0 else 1)
