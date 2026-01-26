"""Architector preprocessing utilities.

Includes chunking CSVs into XYZ chunk files and optional SQLite DB logging
with a status column and basic metadata (natoms, elements, geometry, etc.).
"""

from __future__ import annotations

import csv
import math
import pickle
import sqlite3
from pathlib import Path

import pandas as pd


def chunk_architector_to_lmdb(
    csv_path: str | Path,
    lmdb_path: str | Path,
    chunk_size: int = 10000,
    column: str = "aligned_csd_core",
    status: str = "ready",
    map_size: int = 1 << 40,
) -> Path:
    """Chunk an Architector CSV and store structures and metadata into an LMDB.

    Each LMDB entry stores a pickled dict with keys: orig_index, chunk_file,
    index_in_chunk, elements, natoms, status, geometry.

    This function requires the `lmdb` package. If it is not installed,
    ImportError is raised.
    """
    try:
        import lmdb
    except Exception as exc:  # pragma: no cover - depends on env
        raise ImportError("lmdb package is required to write lmdb files") from exc

    csv_path = Path(csv_path)
    lmdb_path = Path(lmdb_path)
    lmdb_path.parent.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(str(lmdb_path), map_size=map_size)

    df = pd.read_csv(csv_path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in {csv_path}")

    total = len(df)
    n_chunks = math.ceil(total / float(chunk_size)) if total else 0

    with env.begin(write=True) as txn:
        for ci in range(n_chunks):
            start = ci * chunk_size
            end = min((ci + 1) * chunk_size, total)
            chunk = df.iloc[start:end]
            chunk_file = f"chunk_{ci}.xyz"
            idx_in_chunk = 0
            for orig_idx, row in chunk.iterrows():
                xyz_str = row.get(column)
                if pd.isna(xyz_str):
                    continue
                elems = parse_xyz_elements(str(xyz_str))
                natoms = len(elems)

                rec = {
                    "orig_index": int(orig_idx),
                    "chunk_file": chunk_file,
                    "index_in_chunk": idx_in_chunk,
                    "elements": ";".join(elems),
                    "natoms": natoms,
                    "status": status,
                    "geometry": str(xyz_str),
                }
                key = f"{orig_idx}".encode()
                txn.put(key, pickle.dumps(rec))
                idx_in_chunk += 1

    env.sync()
    env.close()
    return lmdb_path


def parse_xyz_elements(xyz_str: str) -> list[str]:
    """Return element symbols from an XYZ string (best-effort parser)."""
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


def _init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS structures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            orig_index INTEGER,
            chunk_file TEXT,
            index_in_chunk INTEGER,
            elements TEXT,
            natoms INTEGER,
            status TEXT,
            charge INTEGER,
            spin INTEGER,
            geometry TEXT
        )
        """
    )
    conn.commit()
    return conn


def _insert_row(
    conn: sqlite3.Connection,
    orig_index: int,
    chunk_file: str,
    index_in_chunk: int,
    elements: str,
    natoms: int,
    geometry: str,
    status: str = "ready",
    charge: int | None = None,
    spin: int | None = None,
):
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO structures (orig_index, chunk_file, index_in_chunk, elements, natoms, status, charge, spin, geometry)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            orig_index,
            chunk_file,
            index_in_chunk,
            elements,
            natoms,
            status,
            charge,
            spin,
            geometry,
        ),
    )


def chunk_architector_csv(
    csv_path: str | Path,
    output_dir: str | Path,
    chunk_size: int = 10000,
    column: str = "aligned_csd_core",
    db_path: str | Path | None = None,
) -> Path:
    """Chunk an Architector CSV into xyz files and (optionally) log entries to a sqlite DB.

    If `db_path` is provided, a sqlite DB is created and a row is added for each
    structure with default status `ready`. The manifest.csv is still written
    for quick inspection.
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = None
    if db_path:
        conn = _init_db(Path(db_path))

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
                    # Trim leading/trailing whitespace to avoid blank lines that
                    # break ASE's XYZ reader, then write a single newline after
                    # the frame (no trailing empty frame at EOF).
                    frame = str(xyz_str).strip() + "\n"
                    cf.write(frame)

                    elems = parse_xyz_elements(str(xyz_str))
                    writer.writerow(
                        {
                            "orig_index": int(orig_idx),
                            "chunk_file": str(chunk_file.name),
                            "index_in_chunk": idx_in_chunk,
                            "elements": ";".join(elems),
                        }
                    )

                    if conn is not None:
                        _insert_row(
                            conn,
                            int(orig_idx),
                            str(chunk_file.name),
                            idx_in_chunk,
                            ";".join(elems),
                            len(elems),
                            str(xyz_str),
                            status="ready",
                        )

                    idx_in_chunk += 1

    if conn is not None:
        conn.commit()
        conn.close()

    return manifest_path


__all__ = ["chunk_architector_csv", "parse_xyz_elements"]
