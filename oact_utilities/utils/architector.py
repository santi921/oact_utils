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
    """Return element symbols from an XYZ string (best-effort parser).

    Handles two formats:
    1. Standard XYZ: atom_count\\ncomment\\nelement x y z...
    2. Architector CSV format: element x y z... (no header)
    """
    lines = [ln for ln in xyz_str.splitlines() if ln.strip()]
    if not lines:
        return []

    elems = []

    # Try to parse first line as atom count (standard XYZ format)
    try:
        # atom_count = int(lines[0].strip())
        # If successful, skip first two lines (count + comment)
        atom_lines = lines[2:] if len(lines) > 2 else []
    except ValueError:
        # First line is not a number - assume no header (architector format)
        atom_lines = lines

    for ln in atom_lines:
        parts = ln.split()
        if not parts:
            continue
        # First token should be element symbol
        elems.append(parts[0])

    return elems


def _init_db(db_path: Path, timeout: float = 30.0) -> sqlite3.Connection:
    """Initialize SQLite database with WAL mode for better concurrency.

    Args:
        db_path: Path to database file.
        timeout: Timeout in seconds for database locks.

    Returns:
        SQLite connection object.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=timeout)

    # Enable WAL mode for better concurrent access
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes with WAL

    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS structures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            orig_index INTEGER,
            elements TEXT,
            natoms INTEGER,
            status TEXT,
            charge INTEGER,
            spin INTEGER,
            geometry TEXT,
            job_dir TEXT,
            max_forces REAL,
            scf_steps INTEGER,
            final_energy REAL,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # Create indexes for common queries
    cur.execute("CREATE INDEX IF NOT EXISTS idx_status ON structures(status)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_orig_index ON structures(orig_index)")

    conn.commit()
    return conn


def _insert_row(
    conn: sqlite3.Connection,
    orig_index: int,
    elements: str,
    natoms: int,
    geometry: str,
    status: str = "ready",
    charge: int | None = None,
    spin: int | None = None,
    job_dir: str | None = None,
    max_forces: float | None = None,
    scf_steps: int | None = None,
    final_energy: float | None = None,
    error_message: str | None = None,
):
    """Insert a structure row into the database.

    Args:
        conn: SQLite connection.
        orig_index: Original row index from CSV.
        elements: Semicolon-separated element symbols.
        natoms: Number of atoms.
        geometry: XYZ geometry string.
        status: Job status (default: "ready").
        charge: Molecular charge.
        spin: Spin multiplicity.
        job_dir: Path to job directory.
        max_forces: Maximum force from optimization.
        scf_steps: Number of SCF steps.
        final_energy: Final energy in Hartree.
        error_message: Error message if failed.
    """
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO structures (orig_index, elements, natoms, status, charge, spin, geometry, job_dir, max_forces, scf_steps, final_energy, error_message)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            orig_index,
            elements,
            natoms,
            status,
            charge,
            spin,
            geometry,
            job_dir,
            max_forces,
            scf_steps,
            final_energy,
            error_message,
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
                            orig_index=int(orig_idx),
                            elements=";".join(elems),
                            natoms=len(elems),
                            geometry=str(xyz_str),
                            status="ready",
                        )

                    idx_in_chunk += 1

    if conn is not None:
        conn.commit()
        conn.close()

    return manifest_path


def create_workflow_db(
    csv_path: str | Path,
    db_path: str | Path,
    geometry_column: str = "aligned_csd_core",
    charge_column: str | None = "charge",
    spin_column: str | None = "uhf",
    batch_size: int = 10000,
) -> Path:
    """Create a workflow database directly from an architector CSV.

    This function reads structures from a CSV file and populates a SQLite
    database for workflow tracking. No chunking is performed.

    Args:
        csv_path: Path to the architector CSV file.
        db_path: Path to the SQLite database file to create.
        geometry_column: Name of column containing XYZ geometry strings.
        charge_column: Name of column containing molecular charges (optional).
        spin_column: Name of column containing unpaired electrons for spin (optional).
        batch_size: Number of rows to process at a time (for memory efficiency).

    Returns:
        Path to the created database.
    """
    csv_path = Path(csv_path)
    db_path = Path(db_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Initialize database
    conn = _init_db(db_path)

    # Read CSV in chunks for memory efficiency
    total_inserted = 0

    try:
        for chunk in pd.read_csv(csv_path, chunksize=batch_size):
            if geometry_column not in chunk.columns:
                raise ValueError(f"Column '{geometry_column}' not found in CSV")

            for idx, row in chunk.iterrows():
                xyz_str = row.get(geometry_column)
                if pd.isna(xyz_str):
                    continue

                # Parse geometry
                elems = parse_xyz_elements(str(xyz_str))
                natoms = len(elems)

                # Get charge and spin if columns exist
                charge = None
                spin = None

                if charge_column and charge_column in chunk.columns:
                    charge_val = row.get(charge_column)
                    if not pd.isna(charge_val):
                        charge = int(charge_val)

                if spin_column and spin_column in chunk.columns:
                    spin_val = row.get(spin_column)
                    if not pd.isna(spin_val):
                        # Convert unpaired electrons to spin multiplicity (2S+1)
                        spin = int(spin_val) + 1

                # Insert into database
                _insert_row(
                    conn,
                    orig_index=int(idx),
                    elements=";".join(elems),
                    natoms=natoms,
                    geometry=str(xyz_str),
                    status="ready",
                    charge=charge,
                    spin=spin,
                )

                total_inserted += 1

            # Commit after each chunk
            conn.commit()

    finally:
        conn.close()

    print(f"Created workflow database with {total_inserted} structures at: {db_path}")
    return db_path


__all__ = ["chunk_architector_csv", "parse_xyz_elements", "create_workflow_db"]
