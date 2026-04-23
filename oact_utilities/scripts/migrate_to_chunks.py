"""Migrate job state from an unsplit workflow DB to chunked workflow DBs.

Matches rows by orig_index. Transfers status, job_dir, metrics, error info.
Running jobs are reset to to_run. Failed jobs stay as failed.
Optionally moves job directories into per-chunk roots and updates job_dir paths.

Usage:
    # Dry run (default) - shows what would be migrated
    python -m oact_utilities.scripts.migrate_to_chunks \
        /path/to/old.db /path/to/chunks/

    # Dry run with directory splitting
    python -m oact_utilities.scripts.migrate_to_chunks \
        /path/to/old.db /path/to/chunks/ \
        --old-root /p/vast1/vargas58/oact/act_4_06/jobs_parsl \
        --new-root-template /p/vast1/vargas58/oact/act_4_06_chunk_{i}/jobs_parsl

    # Execute everything
    python -m oact_utilities.scripts.migrate_to_chunks \
        /path/to/old.db /path/to/chunks/ \
        --old-root /p/vast1/vargas58/oact/act_4_06/jobs_parsl \
        --new-root-template /p/vast1/vargas58/oact/act_4_06_chunk_{i}/jobs_parsl \
        --execute
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
from pathlib import Path

# Columns to migrate from old DB to chunk DBs
_MIGRATE_COLS = [
    "job_dir",
    "max_forces",
    "scf_steps",
    "final_energy",
    "wall_time",
    "n_cores",
    "error_message",
    "fail_count",
    "worker_id",
]


def _load_old_jobs(old_db: Path) -> dict[int, dict]:
    """Load all non-to_run jobs from the old DB, keyed by orig_index."""
    conn = sqlite3.connect(str(old_db))
    conn.row_factory = sqlite3.Row
    cols = ", ".join(["orig_index", "status"] + _MIGRATE_COLS)
    rows = conn.execute(
        f"SELECT {cols} FROM structures WHERE status != 'to_run'"
    ).fetchall()
    conn.close()

    jobs: dict[int, dict] = {}
    for row in rows:
        d = dict(row)
        orig_idx = d.pop("orig_index")
        jobs[orig_idx] = d
    return jobs


def _find_chunk_dbs(chunk_dir: Path) -> list[Path]:
    """Find all .db files in the chunk directory."""
    dbs = sorted(chunk_dir.glob("*.db"))
    if not dbs:
        raise FileNotFoundError(f"No .db files found in {chunk_dir}")
    return dbs


def _load_chunk_orig_indices(chunk_db: Path) -> dict[int, int]:
    """Load orig_index -> id mapping for a chunk DB."""
    conn = sqlite3.connect(str(chunk_db))
    rows = conn.execute("SELECT id, orig_index FROM structures").fetchall()
    conn.close()
    return {orig_index: row_id for row_id, orig_index in rows}


def _migrate_status(old_status: str) -> str:
    """Map old status to new status. Running -> to_run, others preserved."""
    if old_status == "running":
        return "to_run"
    return old_status


def _chunk_index_from_db(chunk_db: Path, chunk_dbs: list[Path]) -> int:
    """Return the 0-based index of a chunk DB in the sorted list."""
    return chunk_dbs.index(chunk_db)


def _new_job_dir(
    old_job_dir: str, old_root: str, new_root_template: str, chunk_idx: int
) -> str:
    """Compute the new job_dir path by swapping the root prefix."""
    # e.g. /p/vast1/.../act_4_06/jobs_parsl/job_12345
    #   -> /p/vast1/.../act_4_06_chunk_0/jobs_parsl/job_12345
    relative = old_job_dir.replace(old_root, "", 1).lstrip("/")
    new_root = new_root_template.format(i=chunk_idx)
    return f"{new_root}/{relative}"


def migrate(
    old_db: Path,
    chunk_dir: Path,
    old_root: str | None = None,
    new_root_template: str | None = None,
    execute: bool = False,
) -> None:
    """Migrate job state from old_db to chunk DBs in chunk_dir.

    Args:
        old_db: Path to the old unsplit DB.
        chunk_dir: Directory containing chunk .db files.
        old_root: Old job directory root (e.g. /p/vast1/.../act_4_06/jobs_parsl).
        new_root_template: Template for new roots with {i} for chunk index
            (e.g. /p/vast1/.../act_4_06_chunk_{i}/jobs_parsl).
        execute: If False, dry run only.
    """
    move_dirs = old_root is not None and new_root_template is not None

    old_jobs = _load_old_jobs(old_db)
    print(f"Old DB: {old_db}")
    print(f"  Jobs with state to migrate: {len(old_jobs)}")

    # Count by status
    status_counts: dict[str, int] = {}
    for job in old_jobs.values():
        s = job["status"]
        status_counts[s] = status_counts.get(s, 0) + 1
    for s, c in sorted(status_counts.items()):
        print(f"    {s}: {c}")

    chunk_dbs = _find_chunk_dbs(chunk_dir)
    print(f"\nChunk DBs in {chunk_dir}: {len(chunk_dbs)}")
    for db in chunk_dbs:
        print(f"  {db.name}")

    if move_dirs:
        print("\nDirectory migration:")
        print(f"  Old root: {old_root}")
        print(f"  New root template: {new_root_template}")

    # Build orig_index -> (chunk_db, row_id) mapping
    index_to_chunk: dict[int, tuple[Path, int]] = {}
    for chunk_db in chunk_dbs:
        mapping = _load_chunk_orig_indices(chunk_db)
        for orig_index, row_id in mapping.items():
            index_to_chunk[orig_index] = (chunk_db, row_id)

    # Match old jobs to chunks
    matched = 0
    unmatched = 0
    updates_by_chunk: dict[Path, list[tuple[int, dict]]] = {db: [] for db in chunk_dbs}
    # Track directory moves: (old_path, new_path)
    moves_by_chunk: dict[Path, list[tuple[str, str]]] = {db: [] for db in chunk_dbs}

    for orig_index, job_data in old_jobs.items():
        if orig_index not in index_to_chunk:
            unmatched += 1
            continue
        chunk_db, row_id = index_to_chunk[orig_index]
        new_status = _migrate_status(job_data["status"])
        update = {"status": new_status}
        for col in _MIGRATE_COLS:
            if job_data[col] is not None:
                # Clear worker_id for jobs being reset to to_run
                if col == "worker_id" and new_status == "to_run":
                    continue
                update[col] = job_data[col]

        # Rewrite job_dir if moving directories
        old_job_dir = job_data.get("job_dir")
        if move_dirs and old_job_dir is not None:
            chunk_idx = _chunk_index_from_db(chunk_db, chunk_dbs)
            new_dir = _new_job_dir(old_job_dir, old_root, new_root_template, chunk_idx)
            update["job_dir"] = new_dir
            moves_by_chunk[chunk_db].append((old_job_dir, new_dir))

        updates_by_chunk[chunk_db].append((row_id, update))
        matched += 1

    print(f"\nMatched: {matched}")
    if unmatched:
        print(f"Unmatched (orig_index not in any chunk): {unmatched}")

    # Summary per chunk
    print("\nPer-chunk breakdown:")
    for chunk_db in chunk_dbs:
        updates = updates_by_chunk[chunk_db]
        if not updates:
            print(f"  {chunk_db.name}: 0 updates, 0 dirs to move")
            continue
        chunk_statuses: dict[str, int] = {}
        for _, upd in updates:
            s = upd["status"]
            chunk_statuses[s] = chunk_statuses.get(s, 0) + 1
        parts = ", ".join(f"{s}: {c}" for s, c in sorted(chunk_statuses.items()))
        n_moves = len(moves_by_chunk[chunk_db])
        move_info = f", {n_moves} dirs to move" if move_dirs else ""
        print(f"  {chunk_db.name}: {len(updates)} updates ({parts}){move_info}")

    if move_dirs:
        total_moves = sum(len(m) for m in moves_by_chunk.values())
        print(f"\nTotal directories to move: {total_moves}")
        # Show a few examples
        examples_shown = 0
        for chunk_db in chunk_dbs:
            for old_path, new_path in moves_by_chunk[chunk_db][:1]:
                print(f"  Example: {old_path}")
                print(f"       ->  {new_path}")
                examples_shown += 1
            if examples_shown >= 2:
                break

    if not execute:
        print("\n** DRY RUN -- no changes written. Pass --execute to apply. **")
        return

    # Move directories first (before updating DB paths)
    if move_dirs:
        print("\nMoving job directories...")
        moved = 0
        skipped = 0
        errors = 0
        for chunk_db in chunk_dbs:
            for old_path, new_path in moves_by_chunk[chunk_db]:
                old_p = Path(old_path)
                new_p = Path(new_path)
                if not old_p.exists():
                    skipped += 1
                    continue
                new_p.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.move(str(old_p), str(new_p))
                    moved += 1
                except OSError as e:
                    print(f"  ERROR moving {old_path}: {e}")
                    errors += 1
        print(f"  Moved: {moved}, Skipped (not found): {skipped}, Errors: {errors}")
        if errors:
            print("  WARNING: Some moves failed. DB updates will still proceed.")

    # Apply DB updates
    print("\nApplying DB updates...")
    for chunk_db in chunk_dbs:
        updates = updates_by_chunk[chunk_db]
        if not updates:
            continue
        conn = sqlite3.connect(str(chunk_db))
        try:
            for row_id, update in updates:
                set_clause = ", ".join(f"{col} = ?" for col in update)
                values = list(update.values()) + [row_id]
                conn.execute(
                    f"UPDATE structures SET {set_clause} WHERE id = ?",
                    values,
                )
            conn.commit()
            print(f"  {chunk_db.name}: {len(updates)} rows updated")
        finally:
            conn.close()

    print("\nMigration complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate job state from unsplit DB to chunked DBs."
    )
    parser.add_argument("old_db", type=Path, help="Path to the old unsplit DB")
    parser.add_argument(
        "chunk_dir", type=Path, help="Directory containing chunk .db files"
    )
    parser.add_argument(
        "--old-root",
        type=str,
        default=None,
        help="Old job directory root to move from",
    )
    parser.add_argument(
        "--new-root-template",
        type=str,
        default=None,
        help="New root template with {i} for chunk index "
        "(e.g. /path/to/act_4_06_chunk_{i}/jobs_parsl)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually apply changes (default: dry run)",
    )
    args = parser.parse_args()

    if not args.old_db.exists():
        raise FileNotFoundError(f"Old DB not found: {args.old_db}")
    if not args.chunk_dir.is_dir():
        raise FileNotFoundError(f"Chunk directory not found: {args.chunk_dir}")

    if (args.old_root is None) != (args.new_root_template is None):
        parser.error("--old-root and --new-root-template must be used together")

    migrate(
        args.old_db,
        args.chunk_dir,
        old_root=args.old_root,
        new_root_template=args.new_root_template,
        execute=args.execute,
    )


if __name__ == "__main__":
    main()
