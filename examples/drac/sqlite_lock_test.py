#!/usr/bin/env python3
"""Concurrent-write stress test for SQLite across DRAC filesystems.

Mirrors the real access pattern in
``oact_utilities/workflows/architector_workflow.py`` (DELETE journal mode,
``BEGIN IMMEDIATE`` per write, exponential backoff with jitter on lock/busy
errors). Spawns N writer PROCESSES (not threads) so cross-process file locking
is exercised the same way separate Parsl workers exercise it.

Goal: decide where the workflow DB can live on a DRAC cluster. Run it pointed
at each candidate mount and compare throughput + failure counts:

    # On a login node (Lustre mounts):
    python sqlite_lock_test.py --db-dir "$HOME/scratch"        --workers 16
    python sqlite_lock_test.py --db-dir "$HOME/project/<id>"   --workers 16

    # Inside an salloc / batch job (node-local NVMe):
    python sqlite_lock_test.py --db-dir "$SLURM_TMPDIR"        --workers 16

By default it runs BOTH journal modes (delete, wal) so you can see whether WAL
even survives on each filesystem and how much DELETE-mode contention costs.

Pure standard library: runs under the system python or any module python; no
oact_utilities install required.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import sqlite3
import subprocess
import time
from pathlib import Path

# Mirror architector_workflow.py defaults for a Parsl coordinator.
BUSY_TIMEOUT_S = 30.0
MAX_RETRIES = 10
RETRY_DELAY_CAP_S = 10.0
N_ROWS = 1000  # rows pre-populated; workers UPDATE random rows (page contention)


def _fs_type(path: str) -> str:
    """Best-effort filesystem type via `stat -f`. Returns '?' on failure."""
    try:
        out = subprocess.run(
            ["stat", "-f", "-c", "%T", path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "?"


def _setup_db(db_path: Path, journal_mode: str) -> bool:
    """Create the DB and table, set journal mode, pre-populate rows.

    Returns True if setup succeeded. WAL setup can fail outright on networked
    filesystems (the whole point of the test) -- that failure is reported, not
    raised.
    """
    for suffix in ("", "-wal", "-shm", "-journal"):
        p = Path(str(db_path) + suffix)
        if p.exists():
            p.unlink()
    try:
        conn = sqlite3.connect(str(db_path), timeout=BUSY_TIMEOUT_S)
        got = conn.execute(f"PRAGMA journal_mode={journal_mode}").fetchone()
        if got and got[0].lower() != journal_mode.lower():
            print(
                f"  [setup] requested journal_mode={journal_mode} but DB reports "
                f"'{got[0]}' -- filesystem rejected it"
            )
            conn.close()
            return False
        conn.execute(
            "CREATE TABLE IF NOT EXISTS t ("
            "id INTEGER PRIMARY KEY, val INTEGER, worker TEXT, ts REAL)"
        )
        conn.executemany(
            "INSERT OR REPLACE INTO t (id, val, worker, ts) VALUES (?, 0, '', 0.0)",
            [(i,) for i in range(N_ROWS)],
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.OperationalError as e:
        print(f"  [setup] FAILED to initialize {journal_mode} DB: {e}")
        return False


def _worker(args: tuple[str, int, int]) -> dict:
    """One writer process: M UPDATE transactions with retry/backoff.

    Returns a stats dict. Replicates architector_workflow._execute_with_retry:
    BEGIN IMMEDIATE, catch lock/busy OperationalError, exponential backoff with
    jitter, give up after MAX_RETRIES.
    """
    db_path, wid, n_writes = args
    rng = random.Random(wid * 7919 + 1)
    conn = sqlite3.connect(db_path, timeout=BUSY_TIMEOUT_S)
    ok = 0
    hard_fail = 0
    total_retries = 0
    t0 = time.time()
    for _ in range(n_writes):
        target = rng.randrange(N_ROWS)
        for attempt in range(MAX_RETRIES):
            try:
                cur = conn.cursor()
                if not conn.in_transaction:
                    cur.execute("BEGIN IMMEDIATE")
                cur.execute(
                    "UPDATE t SET val = val + 1, worker = ?, ts = ? WHERE id = ?",
                    (str(wid), time.time(), target),
                )
                conn.commit()
                ok += 1
                break
            except sqlite3.OperationalError as e:
                try:
                    conn.rollback()
                except Exception:
                    pass
                msg = str(e).lower()
                retryable = "lock" in msg or "busy" in msg
                if retryable and attempt < MAX_RETRIES - 1:
                    total_retries += 1
                    delay = min(0.1 * (2**attempt), RETRY_DELAY_CAP_S)
                    time.sleep(delay + rng.uniform(0, delay * 0.2))
                    continue
                hard_fail += 1
                break
    conn.close()
    return {
        "wid": wid,
        "ok": ok,
        "hard_fail": hard_fail,
        "retries": total_retries,
        "wall": time.time() - t0,
    }


def _run_mode(db_dir: Path, journal_mode: str, workers: int, writes: int) -> None:
    db_path = db_dir / f"locktest_{journal_mode}.db"
    print(
        f"\n--- journal_mode={journal_mode}  workers={workers}  "
        f"writes/worker={writes} ---"
    )
    if not _setup_db(db_path, journal_mode):
        print(f"  RESULT: {journal_mode} unusable on this filesystem.")
        return

    t0 = time.time()
    with mp.Pool(processes=workers) as pool:
        results = pool.map(_worker, [(str(db_path), w, writes) for w in range(workers)])
    wall = time.time() - t0

    total_ok = sum(r["ok"] for r in results)
    total_fail = sum(r["hard_fail"] for r in results)
    total_retries = sum(r["retries"] for r in results)
    attempted = workers * writes
    rate = total_ok / wall if wall > 0 else 0.0

    print(f"  committed   : {total_ok}/{attempted}")
    print(f"  hard fails  : {total_fail}")
    print(
        f"  retries     : {total_retries} "
        f"({total_retries / max(total_ok, 1):.2f} per commit)"
    )
    print(f"  wall time   : {wall:.2f} s")
    print(f"  throughput  : {rate:.1f} commits/s")

    # Cleanup artifacts so reruns start clean.
    for suffix in ("", "-wal", "-shm", "-journal"):
        p = Path(str(db_path) + suffix)
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--db-dir",
        required=True,
        help="Directory to create the test DB in (a mount to probe)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Concurrent writer processes (default: 16)",
    )
    ap.add_argument(
        "--writes",
        type=int,
        default=200,
        help="Write transactions per worker (default: 200)",
    )
    ap.add_argument(
        "--journal",
        choices=["wal", "delete", "both"],
        default="both",
        help="Journal mode(s) to test (default: both)",
    )
    args = ap.parse_args()

    db_dir = Path(os.path.expandvars(args.db_dir)).expanduser()
    if not db_dir.is_dir():
        raise SystemExit(f"--db-dir does not exist or is not a directory: {db_dir}")

    print(f"target dir   : {db_dir}")
    print(f"filesystem   : {_fs_type(str(db_dir))}")
    print(f"python sqlite: {sqlite3.sqlite_version}")

    modes = ["delete", "wal"] if args.journal == "both" else [args.journal]
    for mode in modes:
        _run_mode(db_dir, mode, args.workers, args.writes)

    print("\nInterpretation:")
    print(
        "  - DELETE that commits 100% with low retries  -> safe for the workflow DB here."
    )
    print(
        "  - WAL 'unusable' on a Lustre mount           -> expected; confirms DELETE-only."
    )
    print("  - $SLURM_TMPDIR should show the highest throughput (node-local NVMe).")
    print(
        "  - Any hard fails on DELETE                   -> raise --workers contention is"
    )
    print("    exceeding retry budget; that mount is risky for the live DB.")


if __name__ == "__main__":
    # 'spawn' avoids inheriting an open sqlite connection across fork.
    mp.set_start_method("spawn", force=True)
    main()
