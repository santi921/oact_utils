"""One-time merge of two ORCA job-directory roots: SRC (A) into DEST (B).

Walks every job directory (immediate subdirectory) under both roots,
classifies each from on-disk content alone (no database) using the same
checks as inventory.py / the dashboard, then merges A into B by directory
name. B is the final home: after --execute, every kept job lives under B.

Rules:

* Final-state jobs (completed / failed) are always kept.
  - failed jobs receive ``.do_not_rerun.json`` (same file clean.py writes)
    once they land in B. This blocks resubmission by submit_jobs and makes
    dashboard --update mark the row FAILED, which is exactly the desired end
    state. Contents are KEPT.
  - completed jobs get NO marker: a normally-terminated orca.out is its own
    proof of completion. (``.do_not_rerun.json`` in particular must never be
    written to a completed dir -- the dashboard treats it as an override and
    would flip the row to FAILED on the next ``--update``.)
* Incomplete jobs (to_run / running / timeout) are kept by default; pass
  ``--drop-incomplete`` to delete them from BOTH roots instead.
* Name collisions (same job dir name in A and B) are cross-checked with
  overlap.py's canonical orca.inp structure hash (charge + spin + geometry)
  before anything is deleted -- same name is NOT assumed to mean same
  molecule. Only when both hashes parse and match is the collision resolved:
  completed beats failed beats incomplete; within the same tier the copy
  with the newer files (max mtime inside the dir) wins; exact ties keep B.
  The losing copy is deleted on --execute. When hashes differ or cannot be
  verified, BOTH copies are left untouched and reported for manual
  resolution (exception: with --drop-incomplete, two incomplete copies are
  still dropped -- that removal is status-based, not dedup).

Merging is by directory NAME (e.g. ``barfoot_BN4C6H8O2_q2_m1_idx8761``).
The same structure submitted under two different idx values is NOT collapsed
here -- use workflows/overlap.py to find those.

DRY RUN is the default: nothing is moved, deleted, or markered without
--execute. The report shows, per root, how many jobs would be kept / dropped
and how every collision would resolve.

Usage:
    python -m oact_utilities.scripts.merge_job_roots SRC_ROOT DEST_ROOT
    python -m oact_utilities.scripts.merge_job_roots SRC_ROOT DEST_ROOT \\
        --drop-incomplete --csv plan.csv
    python -m oact_utilities.scripts.merge_job_roots SRC_ROOT DEST_ROOT --execute
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from oact_utilities.workflows.clean import (
    MARKER_FILENAME,
    _extract_failure_info,
    is_marker_blocked,
)
from oact_utilities.workflows.inventory import (
    _STATUS_COMPLETED,
    _STATUS_FAILED,
    _STATUS_ORDER,
    _classify_job_status,
)
from oact_utilities.workflows.overlap import ParsedStructure, parse_job_dir

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]

_FINAL_STATES = frozenset({_STATUS_COMPLETED, _STATUS_FAILED})

# Plan actions.
ACT_MOVE = "move_to_dest"  # A only: move A -> B
ACT_KEEP_DEST = "keep_dest"  # B only (or B wins): B stays
ACT_REPLACE_DEST = "replace_dest"  # collision, A wins: delete B copy, move A
ACT_DISCARD_SRC = "discard_src"  # collision, B wins: delete A copy
ACT_DROP_SRC = "drop_src"  # incomplete, dropped from A
ACT_DROP_DEST = "drop_dest"  # incomplete, dropped from B
ACT_DROP_BOTH = "drop_both"  # collision, both incomplete, dropped
ACT_SKIP_MISMATCH = "skip_hash_mismatch"  # collision, different/unverifiable
# structures under one name: BOTH copies left untouched, resolve manually


@dataclass
class JobInfo:
    """One job directory: path, on-disk status, newest file mtime, input hash.

    ``geom_key`` is overlap.py's canonical structure hash of the orca.inp
    (charge + spin + centered/rounded/sorted geometry), or None when the dir
    has no parseable input.
    """

    path: Path
    status: str
    newest_mtime: float
    geom_key: str | None


@dataclass
class PlanEntry:
    """Planned outcome for one job name across the two roots."""

    name: str
    info_a: JobInfo | None
    info_b: JobInfo | None
    action: str
    reason: str

    @property
    def survivor_status(self) -> str | None:
        """Status of the copy that ends up in DEST (None if dropped)."""
        if self.action in (ACT_MOVE, ACT_REPLACE_DEST):
            assert self.info_a is not None
            return self.info_a.status
        if self.action in (ACT_KEEP_DEST, ACT_DISCARD_SRC):
            assert self.info_b is not None
            return self.info_b.status
        return None


def _newest_mtime(job_dir: Path) -> float:
    """Max mtime of any file inside ``job_dir`` (recursive, symlinks skipped).

    Falls back to the directory's own mtime when it holds no stat-able files.
    """
    newest = 0.0
    for dirpath, _dirnames, filenames in os.walk(job_dir, followlinks=False):
        for name in filenames:
            fp = os.path.join(dirpath, name)
            try:
                if not os.path.islink(fp):
                    newest = max(newest, os.stat(fp).st_mtime)
            except OSError:
                continue
    if newest == 0.0:
        try:
            newest = job_dir.stat().st_mtime
        except OSError:
            pass
    return newest


def scan_root(
    root: Path, hours_cutoff: int, workers: int, label: str, decimals: int = 3
) -> dict[str, JobInfo]:
    """Classify every job directory (immediate subdirectory) under ``root``."""
    job_dirs = sorted(p for p in root.iterdir() if p.is_dir() and not p.is_symlink())

    def scan_one(jd: Path) -> JobInfo:
        status, _code = _classify_job_status(jd, hours_cutoff)
        parsed = parse_job_dir(jd, decimals)
        geom_key = parsed.geom_key if isinstance(parsed, ParsedStructure) else None
        return JobInfo(
            path=jd,
            status=status,
            newest_mtime=_newest_mtime(jd),
            geom_key=geom_key,
        )

    out: dict[str, JobInfo] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(scan_one, jd): jd for jd in job_dirs}
        completed = as_completed(futures)
        if tqdm is not None:
            completed = tqdm(
                completed, total=len(futures), desc=f"Classifying {label}", unit="job"
            )
        for fut in completed:
            info = fut.result()
            out[info.path.name] = info
    return out


def _tier(status: str) -> int:
    """Keep-priority tier: completed > failed > incomplete."""
    if status == _STATUS_COMPLETED:
        return 3
    if status == _STATUS_FAILED:
        return 2
    return 1


def _decide(
    info_a: JobInfo | None, info_b: JobInfo | None, drop_incomplete: bool
) -> tuple[str, str]:
    """Return ``(action, reason)`` for one job name."""
    if info_a is not None and info_b is None:
        if drop_incomplete and info_a.status not in _FINAL_STATES:
            return ACT_DROP_SRC, f"{info_a.status} in A only, --drop-incomplete"
        return ACT_MOVE, f"{info_a.status} in A only"
    if info_b is not None and info_a is None:
        if drop_incomplete and info_b.status not in _FINAL_STATES:
            return ACT_DROP_DEST, f"{info_b.status} in B only, --drop-incomplete"
        return ACT_KEEP_DEST, f"{info_b.status} in B only"

    assert info_a is not None and info_b is not None

    # Collision. Never assume same name means same molecule: only resolve
    # (i.e. delete one copy) when both orca.inp structure hashes parse AND
    # match. Exception: with --drop-incomplete, two incomplete copies are
    # both dropped regardless -- that removal is status-based, not dedup.
    same_hash = info_a.geom_key is not None and info_a.geom_key == info_b.geom_key
    both_incomplete = (
        info_a.status not in _FINAL_STATES and info_b.status not in _FINAL_STATES
    )
    if not same_hash:
        if drop_incomplete and both_incomplete:
            return (
                ACT_DROP_BOTH,
                f"A={info_a.status} vs B={info_b.status}, both incomplete, "
                "--drop-incomplete (hashes differ but both dropped on status)",
            )
        hash_note = (
            "input hashes DIFFER"
            if info_a.geom_key is not None and info_b.geom_key is not None
            else "input hash unverifiable "
            f"(A={'ok' if info_a.geom_key else 'missing'}, "
            f"B={'ok' if info_b.geom_key else 'missing'})"
        )
        return (
            ACT_SKIP_MISMATCH,
            f"A={info_a.status} vs B={info_b.status}, {hash_note}; "
            "both copies kept, resolve manually",
        )

    ta, tb = _tier(info_a.status), _tier(info_b.status)
    a_wins = (ta, info_a.newest_mtime) > (tb, info_b.newest_mtime)
    winner = info_a if a_wins else info_b
    reason = (
        f"A={info_a.status} vs B={info_b.status}, same input hash, "
        f"{'A' if a_wins else 'B'} wins "
        f"({'higher tier' if ta != tb else 'newer files' if info_a.newest_mtime != info_b.newest_mtime else 'tie, B kept'})"
    )
    if drop_incomplete and winner.status not in _FINAL_STATES:
        return ACT_DROP_BOTH, reason + "; winner incomplete, --drop-incomplete"
    return (ACT_REPLACE_DEST if a_wins else ACT_DISCARD_SRC), reason


def build_plan(
    jobs_a: dict[str, JobInfo],
    jobs_b: dict[str, JobInfo],
    drop_incomplete: bool,
) -> list[PlanEntry]:
    """Build the per-job merge plan across the union of names in A and B."""
    plan: list[PlanEntry] = []
    for name in sorted(set(jobs_a) | set(jobs_b)):
        info_a, info_b = jobs_a.get(name), jobs_b.get(name)
        action, reason = _decide(info_a, info_b, drop_incomplete)
        plan.append(
            PlanEntry(
                name=name, info_a=info_a, info_b=info_b, action=action, reason=reason
            )
        )
    return plan


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------


def _marker_needed(entry: PlanEntry) -> tuple[Path, str] | None:
    """Return ``(dir_to_check, marker_filename)`` if the survivor needs one.

    Only FAILED survivors are markered -- a completed job's normally-terminated
    orca.out is its own proof of completion, so it gets no marker. The check
    runs against the CURRENT location of the surviving copy (A's dir when it
    has not moved yet), since the marker file travels with the directory on
    move.
    """
    if entry.survivor_status != _STATUS_FAILED:
        return None
    src_info = (
        entry.info_a if entry.action in (ACT_MOVE, ACT_REPLACE_DEST) else entry.info_b
    )
    assert src_info is not None
    if is_marker_blocked(src_info.path):
        return None
    return src_info.path, MARKER_FILENAME


def _write_failed_marker(job_dir: Path, source_root: Path) -> None:
    """Write ``.do_not_rerun.json`` into a failed ``job_dir`` (contents kept)."""
    data: dict[str, str | int | None] = {
        "generated_by": "python -m oact_utilities.scripts.merge_job_roots",
        "date": datetime.now(tz=timezone.utc).isoformat(),
        "detected_status": _STATUS_FAILED,
        "merged_from": str(source_root),
        **_extract_failure_info(job_dir),
    }
    (job_dir / MARKER_FILENAME).write_text(json.dumps(data, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def execute_plan(
    plan: list[PlanEntry], dest_root: Path, src_root: Path, execute: bool
) -> tuple[Counter, int, list[str]]:
    """Apply (or preview) the plan. Returns (action counts, markers, errors)."""
    counts: Counter = Counter()
    markers_written = 0
    errors: list[str] = []

    for entry in plan:
        counts[entry.action] += 1
        marker_info = _marker_needed(entry)

        if execute:
            try:
                if entry.action == ACT_MOVE:
                    assert entry.info_a is not None
                    shutil.move(str(entry.info_a.path), str(dest_root / entry.name))
                elif entry.action == ACT_REPLACE_DEST:
                    assert entry.info_a is not None and entry.info_b is not None
                    shutil.rmtree(str(entry.info_b.path))
                    shutil.move(str(entry.info_a.path), str(dest_root / entry.name))
                elif entry.action == ACT_DISCARD_SRC:
                    assert entry.info_a is not None
                    shutil.rmtree(str(entry.info_a.path))
                elif entry.action == ACT_DROP_SRC:
                    assert entry.info_a is not None
                    shutil.rmtree(str(entry.info_a.path))
                elif entry.action == ACT_DROP_DEST:
                    assert entry.info_b is not None
                    shutil.rmtree(str(entry.info_b.path))
                elif entry.action == ACT_DROP_BOTH:
                    assert entry.info_a is not None and entry.info_b is not None
                    shutil.rmtree(str(entry.info_a.path))
                    shutil.rmtree(str(entry.info_b.path))
            except (OSError, shutil.Error) as e:
                errors.append(f"{entry.name}: {entry.action} failed: {e}")
                continue

        if marker_info is not None:
            markers_written += 1
            if execute:
                origin = (
                    src_root
                    if entry.action in (ACT_MOVE, ACT_REPLACE_DEST)
                    else dest_root
                )
                try:
                    _write_failed_marker(dest_root / entry.name, origin)
                except (OSError, PermissionError) as e:
                    errors.append(f"{entry.name}: marker write failed: {e}")

    return counts, markers_written, errors


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _status_counts(jobs: dict[str, JobInfo]) -> Counter:
    """Status -> count for one scanned root."""
    return Counter(info.status for info in jobs.values())


def print_report(
    src_root: Path,
    dest_root: Path,
    jobs_a: dict[str, JobInfo],
    jobs_b: dict[str, JobInfo],
    plan: list[PlanEntry],
    counts: Counter,
    markers: int,
    errors: list[str],
    execute: bool,
    top: int,
) -> None:
    """Print the merge summary."""
    verb = "" if execute else "would be "
    print(f"\n{'=' * 70}")
    print("  Job Root Merge Report (A -> B)")
    print(f"{'=' * 70}")
    print(f"  Folder A (source): {src_root}  ({len(jobs_a)} job dirs)")
    print(f"  Folder B (dest):   {dest_root}  ({len(jobs_b)} job dirs)")

    print("\n--- On-disk status ---")
    print("  status | A | B")
    ca, cb = _status_counts(jobs_a), _status_counts(jobs_b)
    for status in _STATUS_ORDER:
        print(f"  {status} | {ca.get(status, 0)} | {cb.get(status, 0)}")

    # Hash cross-check over ALL name collisions, independent of the action
    # taken: same name does not imply same molecule.
    collisions = [e for e in plan if e.info_a is not None and e.info_b is not None]
    n_same_hash = n_diff_hash = n_unverifiable = 0
    for e in collisions:
        assert e.info_a is not None and e.info_b is not None
        if e.info_a.geom_key is None or e.info_b.geom_key is None:
            n_unverifiable += 1
        elif e.info_a.geom_key == e.info_b.geom_key:
            n_same_hash += 1
        else:
            n_diff_hash += 1

    kept_from_a = counts[ACT_MOVE] + counts[ACT_REPLACE_DEST]
    kept_from_b = counts[ACT_KEEP_DEST] + counts[ACT_DISCARD_SRC]
    print("\n--- Plan ---")
    print(f"  Jobs {verb}kept from A (moved into B): {kept_from_a}")
    print(f"  Jobs {verb}kept from B (stay in place): {kept_from_b}")
    print(
        f"  Name collisions: {len(collisions)} "
        f"(A wins {counts[ACT_REPLACE_DEST]}, B wins {counts[ACT_DISCARD_SRC]}, "
        f"both dropped {counts[ACT_DROP_BOTH]}, "
        f"skipped {counts[ACT_SKIP_MISMATCH]})"
    )
    print("\n--- Collision input-hash cross-check (same name != same molecule) ---")
    print(f"  Same name, same structure hash:      {n_same_hash}")
    print(f"  Same name, DIFFERENT structure hash: {n_diff_hash}")
    print(f"  Same name, hash unverifiable:        {n_unverifiable}")
    if counts[ACT_SKIP_MISMATCH]:
        print(
            f"  -> {counts[ACT_SKIP_MISMATCH]} collision(s) left untouched in "
            "both roots (no deletion without a verified hash match)."
        )
    print(
        f"\n  Incomplete {verb}dropped from A: "
        f"{counts[ACT_DROP_SRC] + counts[ACT_DROP_BOTH]}"
    )
    print(
        f"  Incomplete {verb}dropped from B: "
        f"{counts[ACT_DROP_DEST] + counts[ACT_DROP_BOTH]}"
    )
    print(f"  Failed-job .do_not_rerun.json markers {verb}written: {markers}")

    if collisions:
        print(f"\n--- Collisions (showing up to {top}) ---")
        print("  name | action | reason")
        shown = sorted(collisions, key=lambda e: e.action != ACT_SKIP_MISMATCH)
        for e in shown[:top]:
            print(f"  {e.name} | {e.action} | {e.reason}")
        if len(collisions) > top:
            print(f"  ... and {len(collisions) - top} more (use --csv for all)")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for err in errors[:20]:
            print(f"  - {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")

    if not execute:
        print(
            "\n  DRY RUN -- nothing was moved, deleted, or markered. "
            "Re-run with --execute to apply."
        )


def write_csv(plan: list[PlanEntry], csv_path: Path) -> None:
    """Write the full per-job plan to CSV."""
    header = [
        "name",
        "status_a",
        "status_b",
        "newest_mtime_a",
        "newest_mtime_b",
        "geom_key_a",
        "geom_key_b",
        "action",
        "reason",
    ]
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for e in plan:
            writer.writerow(
                [
                    e.name,
                    e.info_a.status if e.info_a else "",
                    e.info_b.status if e.info_b else "",
                    f"{e.info_a.newest_mtime:.0f}" if e.info_a else "",
                    f"{e.info_b.newest_mtime:.0f}" if e.info_b else "",
                    (e.info_a.geom_key or "") if e.info_a else "",
                    (e.info_b.geom_key or "") if e.info_b else "",
                    e.action,
                    e.reason,
                ]
            )
    print(f"\nPer-job plan written to {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the one-time A -> B job-root merge."""
    parser = argparse.ArgumentParser(
        prog="python -m oact_utilities.scripts.merge_job_roots",
        description=(
            "Merge job root A into job root B (B is the final home). "
            "Final-state jobs (completed/failed) are kept, failed ones are "
            "markered with .do_not_rerun.json; "
            "collisions keep the better/newer copy. DB-blind, name-based. "
            "Dry run by default."
        ),
    )
    parser.add_argument("src_root", type=Path, help="Root A: jobs move FROM here")
    parser.add_argument("dest_root", type=Path, help="Root B: jobs live HERE after")
    parser.add_argument(
        "--drop-incomplete",
        action="store_true",
        help="Delete to_run/running/timeout jobs from BOTH roots instead of "
        "keeping them. CANNOT tell a live ORCA run from a dead one -- only "
        "use when nothing is executing.",
    )
    parser.add_argument(
        "--hours-cutoff",
        type=int,
        default=24,
        metavar="H",
        help="Hours of inactivity before a job with output but no termination "
        "is classified timeout instead of running (default: 24)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write the full per-job plan to this CSV",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        metavar="N",
        help="Max collisions to print (default: 20)",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Parallel scan workers (default: 8)"
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        metavar="N",
        help="Coordinate rounding for the collision input-hash cross-check, "
        "in decimal places of Angstrom (default: 3, matches overlap.py)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move/delete/marker (default: dry-run preview only)",
    )
    args = parser.parse_args()

    for root in (args.src_root, args.dest_root):
        if not root.is_dir():
            print(f"Error: root directory not found: {root}", file=sys.stderr)
            sys.exit(1)
    src_root = args.src_root.resolve()
    dest_root = args.dest_root.resolve()
    if src_root == dest_root:
        print("Error: SRC and DEST are the same directory", file=sys.stderr)
        sys.exit(1)

    mode = "EXECUTE" if args.execute else "DRY RUN (preview only)"
    print("!" * 70)
    print(f"  JOB ROOT MERGE [{mode}] -- DB-BLIND, name-based")
    print(f"  A (source): {src_root}")
    print(f"  B (dest):   {dest_root}")
    print("  Collision losers are DELETED on --execute, but only when both")
    print("  copies' orca.inp structure hashes match; mismatches are kept and")
    print("  reported. --drop-incomplete also deletes to_run/running/timeout")
    print("  jobs from both roots and cannot tell a live ORCA process from a")
    print("  dead one.")
    print("!" * 70)

    jobs_a = scan_root(src_root, args.hours_cutoff, args.workers, "A", args.decimals)
    jobs_b = scan_root(dest_root, args.hours_cutoff, args.workers, "B", args.decimals)
    plan = build_plan(jobs_a, jobs_b, args.drop_incomplete)
    counts, markers, errors = execute_plan(plan, dest_root, src_root, args.execute)
    print_report(
        src_root,
        dest_root,
        jobs_a,
        jobs_b,
        plan,
        counts,
        markers,
        errors,
        args.execute,
        args.top,
    )
    if args.csv is not None:
        write_csv(plan, args.csv)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
