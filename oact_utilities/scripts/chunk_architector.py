"""CLI wrapper for Architector chunking utilities.

This script provides a thin CLI that delegates to
``oact_utilities.utils.architector.chunk_architector_csv`` which contains the
actual logic and (optional) DB logging.
"""

from __future__ import annotations

from pathlib import Path

from oact_utilities.utils.architector import chunk_architector_csv

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
    ap.add_argument(
        "--db",
        help="Optional path to sqlite DB to log structures (default: <output_dir>/manifest.db)",
    )
    args = ap.parse_args()

    db_path = args.db if args.db is not None else Path(args.output_dir) / "manifest.db"

    manifest = chunk_architector_csv(
        args.csv_path,
        args.output_dir,
        chunk_size=args.chunk_size,
        column=args.column,
        db_path=db_path,
    )
    print(f"Wrote manifest to: {manifest}")
