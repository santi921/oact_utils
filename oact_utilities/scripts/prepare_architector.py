"""Utilities to preprocess Architector CSV outputs into chunked XYZ files

The main utility `chunk_architector_csv` reads a CSV file produced by
Architector and writes chunked `.xyz` files suitable for downstream
high-throughput workflows. It also produces a manifest CSV linking
structure indices to chunk files and a list of element symbols.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pandas as pd


def parse_xyz_elements(xyz_str: str) -> list[str]:
    """Return the list of element symbols found in an XYZ string.

    This is a best-effort parser: it ignores the atom count/comment lines
    and extracts the first token of remaining lines.
    """
    lines = [ln for ln in xyz_str.splitlines() if ln.strip()]
    if len(lines) <= 2:
        return []
    atom_lines = lines[2:]
    elems = []
    for ln in atom_lines:
        parts = ln.split()
        if not parts:
            continue
        elems.append(parts[0])
    return elems


def chunk_architector_csv(
    csv_path: str | Path,
    output_dir: str | Path,
    chunk_size: int = 10000,
    column: str = "aligned_csd_core",
) -> Path:
    """Read an Architector CSV and write chunked XYZ files.

    Args:
        csv_path: path to the CSV file.
        output_dir: directory where chunk files and manifest will be written.
        chunk_size: number of structures per chunk file.
        column: name of the column containing XYZ strings.

    Returns:
        Path to the manifest CSV describing the generated files.
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in {csv_path}")

    total = len(df)
    n_chunks = math.ceil(total / float(chunk_size)) if total else 0

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as mf:
        writer = csv.DictWriter(
            mf, fieldnames=["orig_index", "chunk_file", "index_in_chunk", "elements"]
        )
        writer.writeheader()

        for ci in range(n_chunks):
            start = ci * chunk_size
            end = min((ci + 1) * chunk_size, total)
            chunk = df.iloc[start:end]
            chunk_file = output_dir / f"chunk_{ci}.xyz"
            with chunk_file.open("w") as cf:
                idx_in_chunk = 0
                for orig_idx, row in chunk.iterrows():
                    xyz_str = row.get(column)
                    if pd.isna(xyz_str):
                        continue
                    # write xyz string and ensure separation
                    cf.write(str(xyz_str).rstrip())
                    cf.write("\n")

                    # compute elements and write manifest row
                    elems = parse_xyz_elements(str(xyz_str))
                    writer.writerow(
                        {
                            "orig_index": int(orig_idx),
                            "chunk_file": str(chunk_file.name),
                            "index_in_chunk": idx_in_chunk,
                            "elements": ";".join(elems),
                        }
                    )
                    idx_in_chunk += 1

    return manifest_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Chunk Architector CSV into XYZ files and generate a manifest"
    )
    ap.add_argument("csv_path", help="Path to the Architector CSV file")
    ap.add_argument("output_dir", help="Directory to write chunks and manifest")
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Number of structures per chunk file",
    )
    ap.add_argument(
        "--column",
        default="aligned_csd_core",
        help="Column name containing XYZ strings",
    )
    args = ap.parse_args()

    manifest = chunk_architector_csv(
        args.csv_path, args.output_dir, chunk_size=args.chunk_size, column=args.column
    )
    print(f"Wrote manifest to: {manifest}")
