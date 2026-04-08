"""Prepare a Sella opt workflow DB seeded from completed SP calculations.

Reads a completed SP workflow DB, scans a directory for the actual SP job
folders (which may have moved from stored job_dir paths), validates seed
files, copies orca.engrad and orca.gbw to a staging directory, and writes
a new Sella opt workflow DB with sp_engrad_path and sp_gbw_path columns.

Note: sp_engrad_path and sp_gbw_path are not part of _LIGHT_COLS in
ArchitectorWorkflow and have no corresponding fields on JobRecord. Code
that consumes this DB (e.g. submit_jobs.py) must read these columns via
raw SQL or extend _LIGHT_COLS / JobRecord at integration time.

Usage:
    python -m oact_utilities.workflows.prepare_seeded_sella_db \\
        --sp-db workflow_sp.db \\
        --sp-scan-dir /storage/sp_jobs/ \\
        --out-db workflow_sella_opt.db \\
        --seed-dir seeds/ \\
        --execute
"""

from __future__ import annotations

import argparse
import re
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any

from oact_utilities.utils.analysis import get_engrad
from oact_utilities.utils.architector import _init_db, _insert_row


def scan_for_job_dir(scan_dir: Path, orig_index: int) -> Path | None:
    """Find the subdirectory for orig_index under scan_dir.

    Tries exact match ``job_{orig_index}`` first. Falls back to scanning one
    level deep for directories whose name ends with ``_{orig_index}`` or
    ``-{orig_index}``.

    Args:
        scan_dir: Directory to search within.
        orig_index: Original index of the molecule in the SP DB.

    Returns:
        Matching Path, or None if not found.

    Raises:
        ValueError: If more than one directory matches the fallback pattern.
    """
    exact = scan_dir / f"job_{orig_index}"
    if exact.is_dir():
        return exact

    pattern = re.compile(rf".*[_\-]{orig_index}$")
    matches = [
        d for d in scan_dir.iterdir() if d.is_dir() and pattern.fullmatch(d.name)
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Multiple directories match orig_index={orig_index} under {scan_dir}: "
            + ", ".join(d.name for d in matches)
        )
    return None


def validate_seed_files(job_dir: Path) -> tuple[bool, str]:
    """Check that orca.engrad is parseable and orca.gbw exists.

    Args:
        job_dir: Path to the SP job directory.

    Returns:
        Tuple of (ok, reason). reason is empty string when ok is True.
    """
    engrad = job_dir / "orca.engrad"
    gbw = job_dir / "orca.gbw"
    if not engrad.exists():
        return False, "missing orca.engrad"
    if not gbw.exists():
        return False, "missing orca.gbw"
    try:
        result = get_engrad(str(engrad))
        if not result.get("gradient_Eh_per_bohr"):
            return False, "empty gradient in engrad"
    except Exception as e:
        return False, f"engrad parse error: {e}"
    return True, ""


def _safe_copy(src: Path, dst: Path, job_dir: Path) -> None:
    """Copy src to dst, asserting src resolves within job_dir.

    Args:
        src: Source file path.
        dst: Destination file path.
        job_dir: Expected parent directory of src (symlink escape check).

    Raises:
        ValueError: If resolved src path escapes job_dir.
    """
    resolved_src = src.resolve()
    resolved_job = job_dir.resolve()
    if not str(resolved_src).startswith(str(resolved_job)):
        raise ValueError(
            f"Symlink escape: {src} resolves to {resolved_src}, "
            f"which is outside {resolved_job}"
        )
    shutil.copy2(str(resolved_src), str(dst))


def prepare_seeded_sella_db(
    sp_db: Path,
    sp_scan_dir: Path,
    out_db: Path,
    seed_dir: Path,
    execute: bool = False,
) -> None:
    """Scan SP jobs, validate seed files, copy them, and write a new Sella DB.

    Args:
        sp_db: Path to the completed SP workflow SQLite database.
        sp_scan_dir: Directory to scan for SP job folders.
        out_db: Path for the output Sella opt workflow database.
        seed_dir: Directory to copy seed files into.
        execute: If False (default), dry-run only - no files or DB written.
    """
    mode = "EXECUTE" if execute else "DRY RUN"
    print(f"[{mode}] prepare_seeded_sella_db")
    print(f"  sp-db:       {sp_db}")
    print(f"  sp-scan-dir: {sp_scan_dir}")
    print(f"  out-db:      {out_db}")
    print(f"  seed-dir:    {seed_dir}")
    print()

    # Load completed rows from SP DB
    conn_sp = sqlite3.connect(str(sp_db))
    conn_sp.row_factory = sqlite3.Row
    rows: list[dict[str, Any]] = [
        dict(r)
        for r in conn_sp.execute(
            "SELECT * FROM structures WHERE status='completed'"
        ).fetchall()
    ]
    conn_sp.close()
    print(f"Loaded {len(rows)} completed rows from SP DB")

    n_found = 0
    n_missing = 0
    n_parse_errors = 0
    validated: list[dict[str, Any]] = []

    for row in rows:
        orig_index: int = row["orig_index"]
        try:
            job_dir = scan_for_job_dir(sp_scan_dir, orig_index)
        except ValueError as e:
            print(f"  [SKIP] orig_index={orig_index}: {e}")
            n_missing += 1
            continue

        if job_dir is None:
            print(f"  [MISSING] orig_index={orig_index}: no directory found")
            n_missing += 1
            continue

        ok, reason = validate_seed_files(job_dir)
        if not ok:
            if "parse error" in reason:
                n_parse_errors += 1
            else:
                n_missing += 1
            print(f"  [INVALID] orig_index={orig_index} ({job_dir.name}): {reason}")
            continue

        n_found += 1
        dst_folder = seed_dir / f"orig_index_{orig_index}"
        engrad_dst = dst_folder / "orca.engrad"
        gbw_dst = dst_folder / "orca.gbw"

        print(f"  [OK] orig_index={orig_index} ({job_dir.name}) -> {dst_folder}")

        if execute:
            dst_folder.mkdir(parents=True, exist_ok=True)
            _safe_copy(job_dir / "orca.engrad", engrad_dst, job_dir)
            _safe_copy(job_dir / "orca.gbw", gbw_dst, job_dir)

        validated.append(
            {
                "row": row,
                "sp_engrad_path": (
                    str(engrad_dst.resolve()) if execute else str(engrad_dst)
                ),
                "sp_gbw_path": str(gbw_dst.resolve()) if execute else str(gbw_dst),
            }
        )

    print()
    print(
        f"Summary: {n_found} found, {n_missing} missing, "
        f"{n_parse_errors} parse errors"
    )

    if not execute:
        print(f"  (dry run) would write {len(validated)} rows to {out_db}")
        return

    if not validated:
        print("No valid jobs to write. Output DB not created.")
        return

    extra_columns = {"sp_engrad_path": "TEXT", "sp_gbw_path": "TEXT"}
    conn_out = _init_db(out_db, extra_columns=extra_columns)

    for entry in validated:
        row = entry["row"]
        _insert_row(
            conn_out,
            orig_index=row["orig_index"],
            elements=row["elements"],
            natoms=row["natoms"],
            geometry=row["geometry"],
            status="to_run",
            charge=row["charge"],
            spin=row["spin"],
            extra_values={
                "sp_engrad_path": entry["sp_engrad_path"],
                "sp_gbw_path": entry["sp_gbw_path"],
            },
        )

    conn_out.commit()
    conn_out.close()
    print(f"  Written {len(validated)} rows to {out_db}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sp-db", required=True, type=Path, help="SP workflow database"
    )
    parser.add_argument(
        "--sp-scan-dir",
        required=True,
        type=Path,
        help="Directory to scan for SP job folders",
    )
    parser.add_argument(
        "--out-db", required=True, type=Path, help="Output Sella opt workflow DB"
    )
    parser.add_argument(
        "--seed-dir",
        required=True,
        type=Path,
        help="Staging directory for copied seed files",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Actually copy files and write DB (default: dry run)",
    )
    args = parser.parse_args()

    if not args.sp_db.exists():
        print(f"Error: SP database not found: {args.sp_db}", file=sys.stderr)
        sys.exit(1)
    if not args.sp_scan_dir.exists():
        print(
            f"Error: SP scan directory not found: {args.sp_scan_dir}", file=sys.stderr
        )
        sys.exit(1)
    if not args.execute and args.out_db.exists():
        print(
            f"Warning: output DB already exists: {args.out_db} (dry run, not overwriting)"
        )

    prepare_seeded_sella_db(
        args.sp_db, args.sp_scan_dir, args.out_db, args.seed_dir, args.execute
    )


if __name__ == "__main__":
    main()
