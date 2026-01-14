#!/usr/bin/env python3
"""
Simple helper to traverse a folder tree up to a given depth and launch any
`flux_job.flux` files it finds using `flux batch`.

Usage:
    python run_multi_spin.py /path/to/root

The script accepts a single positional argument (root folder). Optional flags:
    --max-depth N    : maximum directory depth to traverse (default: 5)
    --dry-run        : do not actually execute flux, just print what would run
    --verbose        : print extra diagnostic info
"""
from __future__ import annotations

import argparse
import os
import sqlite3
from time import time
from typing import Iterator, Dict, Any
from datetime import datetime
from oact_utilities.utils.status import (
    check_job_termination,
    check_sella_complete,
)


def iter_dirs_limited(root: str, max_depth: int) -> Iterator[str]:
    """Yield directory paths under `root` up to `max_depth` levels deep.

    Depth 0 yields `root` itself. Depth 1 yields immediate subdirectories, etc.
    """
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        return

    for current_dir, dirs, files in os.walk(root):
        # compute relative depth
        rel = os.path.relpath(current_dir, root)
        if rel == ".":
            depth = 0
        else:
            depth = len(rel.split(os.sep))
        if depth > max_depth:
            # tell os.walk not to recurse deeper
            dirs[:] = []
            continue
        yield current_dir


def parse_info_from_path(path: str) -> Dict[str, Any]:
    """Parse information from the path string in a robust way.

    Attempts to locate a part beginning with 'spin' and extracts the spin
    identifier plus the molecule name (the directory immediately before the
    spin directory when possible), category, and lot when present.
    Returns keys: 'lot', 'cat', 'name', 'spin'. Missing values are ''.
    """
    parts = [p for p in path.split(os.sep) if p]
    info: Dict[str, Any] = {"lot": "", "cat": "", "name": "", "spin": ""}

    # find index of element that starts with 'spin'
    spin_idx = None
    for i, p in enumerate(parts[::-1]):
        if p.startswith("spin"):
            # convert reversed index to normal index
            spin_idx = len(parts) - 1 - i
            break

    if spin_idx is not None:
        info["spin"] = parts[spin_idx]
        if spin_idx - 1 >= 0:
            info["name"] = parts[spin_idx - 1]
        if spin_idx - 2 >= 0:
            info["cat"] = parts[spin_idx - 2]
        if spin_idx - 3 >= 0:
            info["lot"] = parts[spin_idx - 3]
    else:
        # fallback: use last 4 parts if available
        if len(parts) >= 1:
            info["name"] = parts[-1]
        if len(parts) >= 2:
            info["spin"] = parts[-2]
        if len(parts) >= 3:
            info["cat"] = parts[-3]
        if len(parts) >= 4:
            info["lot"] = parts[-4]

    return info


def _ensure_db(conn: sqlite3.Connection) -> None:
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE,
            lot TEXT,
            category TEXT,
            name TEXT,
            spin TEXT,
            status INTEGER,
            note TEXT,
            checked_at REAL
        )
        """
    )
    conn.commit()


def find_and_get_status(
    root: str,
    max_depth: int = 5,
    verbose: bool = False,
    *,
    print_table: bool = False,
    running_age_seconds: int = 3600,
) -> int:
    """Traverse `root` up to `max_depth` and launch flux jobs when `flux_job.flux` is found.

    Returns the number of jobs launched (or that would be launched in dry-run).
    """
    db_path = os.path.join(root, "multi_spin_jobs.sqlite3")
    conn = sqlite3.connect(db_path)
    _ensure_db(conn)

    processed = 0
    for d in iter_dirs_limited(root, max_depth=max_depth):
        flux_file = os.path.join(d, "flux_job.flux")
        status = 0
        note = ""
        if os.path.exists(flux_file):
            processed += 1
            # default is remaining (0). Determine status in order:
            # 1) failed (-1) 2) completed (1) 3) running (2 if recent flux-*.out) 4) remaining (0)
            
            term = check_job_termination(d)
            
            if term == -1:
                if verbose:
                    print(f"Found failed job at {d}")
                status = -1
                note = "job_failed"
            elif term:
                if verbose:
                    print(f"Found completed job at {d}")
                status = 1
                note = "job_completed"
            elif check_sella_complete(d):
                if verbose:
                    print(f"Skipping {d} because it has a completed job (sella) ")
                status = 1
                note = "sella_complete"
            else:
                # check for running indicator: recent flux-*.out read/write
                import glob

                out_files = glob.glob(os.path.join(d, "flux-*.out"))
                if out_files:
                    latest_out = max(out_files, key=os.path.getmtime)
                    latest_mtime = os.path.getmtime(latest_out)
                    if (time() - latest_mtime) < running_age_seconds:
                        if verbose:
                            print(f"Found recent flux output {latest_out}; marking as running")
                        status = 2
                        note = f"running_recent:{os.path.basename(latest_out)}"

            path_data = parse_info_from_path(d)

            # This script is a status checker only; always record status

            # insert/update into DB
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO jobs (path, lot, category, name, spin, status, note, checked_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    lot=excluded.lot,
                    category=excluded.category,
                    name=excluded.name,
                    spin=excluded.spin,
                    status=excluded.status,
                    note=excluded.note,
                    checked_at=excluded.checked_at
                """,
                (
                    d,
                    path_data.get("lot", ""),
                    path_data.get("cat", ""),
                    path_data.get("name", ""),
                    path_data.get("spin", ""),
                    status,
                    note,
                    time(),
                ),
            )
            conn.commit()

    # Print a clean summary per molecule/name and spin state
    c = conn.cursor()
    c.execute(
        """
        SELECT name, category, spin,
               SUM(CASE WHEN status = 2 THEN 1 ELSE 0 END) AS running,
               SUM(CASE WHEN status = 0 THEN 1 ELSE 0 END) AS remaining,
               SUM(CASE WHEN status = 1 THEN 1 ELSE 0 END) AS done,
               SUM(CASE WHEN status = -1 THEN 1 ELSE 0 END) AS failed
        FROM jobs
        GROUP BY name, category, spin
        ORDER BY name, category, spin
        """
    )
    rows = c.fetchall()
    if rows:
        print("\nSummary of multi-spin jobs:\n")
        current_name = None
        current_cat = None
        for name, category, spin, running, remaining, done, failed in rows:
            if name != current_name or category != current_cat:
                if current_name is not None:
                    print("")
                print(f"Molecule: {name} (category: {category})")
                current_name = name
                current_cat = category
            print(
                f"  {spin:<10} -> running: {running:3d}  remaining: {remaining:3d}  done: {done:3d}  failed: {failed:3d}"
            )
    else:
        print("No job records found in DB.")

    # Print legend for status codes
    print("\nStatus legend:\n  running: 2 (recent flux-*.out)\n  remaining: 0 (no run detected)\n  done: 1 (completed)\n  failed: -1 (failed)\n")

    # Optionally print the entire jobs table for debugging (pretty formatted)
    if print_table:
        print("\nFull jobs table:\n")
        # print legend above the table for clarity
        print("Status legend: running=2, remaining=0, done=1, failed=-1\n")
        c2 = conn.cursor()
        c2.execute(
            "SELECT id, path, lot, category, name, spin, status, note, checked_at FROM jobs ORDER BY name, spin"
        )
        all_rows = c2.fetchall()
        headers = []
        if c2.description:
            headers = [d[0] for d in c2.description]

        # prefer tabulate if available, otherwise do basic aligned columns
        try:
            from tabulate import tabulate

            print(tabulate(all_rows, headers=headers, tablefmt="github"))
        except Exception:
            # fallback simple table
            if headers:
                widths = [max(len(str(h)), max((len(str(r[i])) for r in all_rows), default=0)) for i, h in enumerate(headers)]
                header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
                sep = "-+-".join("".ljust(widths[i], "-") for i in range(len(widths)))
                print(header_line)
                print(sep)
                for r in all_rows:
                    print(" | ".join(str(r[i]).ljust(widths[i]) for i in range(len(r))))
            else:
                for r in all_rows:
                    print("\t".join([str(x) for x in r]))

    conn.close()
    return processed


def main() -> None:
    # TODO: add folder checking
    parser = argparse.ArgumentParser(
        description="Launch flux_job.flux files under a folder (depth-limited)"
    )
    parser.add_argument("root", help="Root folder to traverse")
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Max subdirectory depth to traverse (default: 5)",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--print-table", action="store_true", help="Print the full jobs table for debugging")
    parser.add_argument(
        "--running-age-seconds",
        type=int,
        default=3600,
        help="Seconds to treat a flux-*.out as 'running' (default: 3600)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        parser.error(f"Root path does not exist or is not a directory: {args.root}")

    n = find_and_get_status(
        args.root,
        max_depth=args.max_depth,
        verbose=args.verbose,
        print_table=args.print_table,
        running_age_seconds=args.running_age_seconds,
    )
    print(f"Total flux jobs launched/found: {n}")


if __name__ == "__main__":
    main()
