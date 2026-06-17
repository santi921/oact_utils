"""Diagnose why clean.py touches far fewer directories than exist on disk.

Read-only. clean.py is DB-driven: it iterates DB rows, resolves each row's
stored ``job_dir`` (via :func:`oact_utilities.workflows.clean._resolve_job_dir`),
and only acts on directories that resolve AND exist under ``root_dir``. Any
on-disk directory not referenced by a resolvable DB row is invisible to cleanup
-- never cleaned, never purged. When the DB's ``job_dir`` paths do not line up
with the directories actually on disk (e.g. the corpus was transferred to a new
root after submission), most rows are silently skipped and most disk space is
never reclaimed.

This tool reproduces clean.py's resolution exactly -- it imports the same
``_resolve_job_dir`` and uses the same clean-set status list -- then bins every
candidate row by why it would (or would not) be processed, and cross-checks the
DB against the directories actually present under ``root_dir``:

  - exists_processed : stored job_dir resolves under root AND the dir exists
                       (clean.py would process it)
  - dir_missing      : stored job_dir resolves under root but the dir is absent
  - escapes_root     : stored job_dir resolves OUTSIDE root_dir (rejected)
  - null_job_dir     : the DB row has no job_dir at all

  - reroot_recoverable : a dir_missing/escapes_root row whose basename DOES exist
                         as ``root_dir/<basename>`` (classic prefix mismatch -- a
                         basename reroot would recover these)
  - orphan dirs        : directories on disk that NO processed row maps to (the
                         space clean.py cannot currently see)

It writes nothing -- not the DB, not the filesystem.

Usage:
    python -m oact_utilities.workflows.diagnose_coverage <db_path> <root_dir>
    python -m oact_utilities.workflows.diagnose_coverage <db> <root> --workers 16
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .architector_workflow import ArchitectorWorkflow, JobRecord, JobStatus
from .clean import _resolve_job_dir

# The exact status set clean.py processes in its scratch-cleanup phase with
# --include-failed (clean.py:919-924). "running" is intentionally excluded:
# clean.py never cleans running jobs, so neither does this diagnostic.
_CLEAN_STATUSES = [
    JobStatus.COMPLETED,
    JobStatus.FAILED,
    JobStatus.TIMEOUT,
    JobStatus.TO_RUN,
]

# Bin labels.
_EXISTS = "exists_processed"
_DIR_MISSING = "dir_missing"
_ESCAPES = "escapes_root"
_NULL = "null_job_dir"

_MAX_SAMPLES = 8


@dataclass
class CoverageReport:
    """Aggregate result of the clean.py coverage diagnostic."""

    total_rows: int
    bins: Counter  # bin label -> count
    reroot_recoverable: int  # rows whose basename exists under root_dir
    dirs_on_disk: int
    dirs_referenced: int  # distinct on-disk dirs a processed row maps to
    orphan_dirs: int  # on disk, no processed row maps to them
    orphans_reroot_recoverable: int  # orphan dirs whose basename a DB row carries
    unresolved_samples: list[str] = field(default_factory=list)
    orphan_samples: list[str] = field(default_factory=list)


def _classify_row(
    job: JobRecord, root_dir: Path
) -> tuple[str, bool, str | None, str | None]:
    """Classify one DB row by how clean.py would treat its job_dir.

    Args:
        job: The DB row.
        root_dir: Resolved cleanup root directory.

    Returns:
        ``(bin_label, reroot_recoverable, referenced_name, sample)``.
        ``referenced_name`` is the on-disk basename a processed row maps to
        (None otherwise). ``sample`` is a short human-readable string for
        unresolved rows (None otherwise).
    """
    if job.job_dir is None:
        return _NULL, False, None, None

    basename = Path(job.job_dir).name
    # A basename reroot only makes sense if the basename is non-empty and the
    # directory is actually present directly under root_dir.
    reroot_ok = bool(basename) and (root_dir / basename).is_dir()

    resolved = _resolve_job_dir(job.job_dir, root_dir)
    if resolved is None:
        return _ESCAPES, reroot_ok, None, f"[escapes] {job.job_dir}"
    if not resolved.is_dir():
        return _DIR_MISSING, reroot_ok, None, f"[missing] {job.job_dir}"
    return _EXISTS, False, resolved.name, None


def diagnose_coverage(
    db_path: Path, root_dir: Path, workers: int = 8
) -> CoverageReport:
    """Reproduce clean.py's row resolution and bin the coverage gap.

    Args:
        db_path: Path to the workflow SQLite database.
        root_dir: Root directory containing job subdirectories.
        workers: Parallel threads for on-disk existence probes (network FS).

    Returns:
        A :class:`CoverageReport`.
    """
    root_dir = root_dir.resolve()

    with ArchitectorWorkflow(db_path) as wf:
        jobs = wf.get_jobs_by_status(_CLEAN_STATUSES, include_geometry=False)

    bins: Counter = Counter()
    reroot_recoverable = 0
    referenced_names: set[str] = set()
    unresolved_samples: list[str] = []

    workers = max(1, workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(lambda j: _classify_row(j, root_dir), jobs)
        for label, reroot_ok, ref_name, sample in results:
            bins[label] += 1
            if reroot_ok:
                reroot_recoverable += 1
            if ref_name is not None:
                referenced_names.add(ref_name)
            if sample is not None and len(unresolved_samples) < _MAX_SAMPLES:
                unresolved_samples.append(sample)

    # On-disk inventory via scandir (DirEntry.is_dir caches the type -- no extra
    # stat per entry, which matters for tens of thousands of dirs on Lustre).
    on_disk: set[str] = set()
    try:
        with os.scandir(root_dir) as it:
            for entry in it:
                try:
                    if entry.is_dir():
                        on_disk.add(entry.name)
                except OSError:
                    continue
    except OSError as e:
        print(f"Warning: cannot scan root dir {root_dir}: {e}", file=sys.stderr)

    orphans = on_disk - referenced_names
    # All non-null job_dir basenames the DB knows about -- an orphan dir whose
    # name matches one of these is recoverable by a basename reroot.
    db_basenames = {
        Path(j.job_dir).name for j in jobs if j.job_dir is not None and j.job_dir
    }
    orphans_reroot = len(orphans & db_basenames)
    orphan_samples = sorted(orphans)[:_MAX_SAMPLES]

    return CoverageReport(
        total_rows=len(jobs),
        bins=bins,
        reroot_recoverable=reroot_recoverable,
        dirs_on_disk=len(on_disk),
        dirs_referenced=len(referenced_names),
        orphan_dirs=len(orphans),
        orphans_reroot_recoverable=orphans_reroot,
        unresolved_samples=unresolved_samples,
        orphan_samples=orphan_samples,
    )


def print_report(report: CoverageReport, db_path: Path, root_dir: Path) -> None:
    """Print the coverage report in clean.py's reporting style."""
    print(f"\n{'=' * 60}")
    print("  clean.py Coverage Diagnostic (READ-ONLY)")
    print(f"{'=' * 60}")
    print(f"  Database:   {db_path}")
    print(f"  Root dir:   {root_dir}")
    print(
        "  Clean set:  completed + failed + timeout + to_run "
        "(running excluded, as in clean.py)"
    )
    print(f"  DB rows in clean set: {report.total_rows}")

    print("\n--- Why each row is / isn't processed by clean.py ---")
    print(f"  {_EXISTS:18s}: {report.bins.get(_EXISTS, 0)}  (clean.py processes these)")
    print(f"  {_DIR_MISSING:18s}: {report.bins.get(_DIR_MISSING, 0)}")
    print(f"  {_ESCAPES:18s}: {report.bins.get(_ESCAPES, 0)}")
    print(f"  {_NULL:18s}: {report.bins.get(_NULL, 0)}")
    skipped = (
        report.bins.get(_DIR_MISSING, 0)
        + report.bins.get(_ESCAPES, 0)
        + report.bins.get(_NULL, 0)
    )
    print(f"  -> skipped (not processed): {skipped}")
    print(
        f"\n  reroot-recoverable (stored dir absent, but root_dir/<basename> "
        f"exists): {report.reroot_recoverable}"
    )

    print("\n--- On-disk vs DB ---")
    print(f"  dirs on disk under root:        {report.dirs_on_disk}")
    print(f"  dirs a processed row maps to:   {report.dirs_referenced}")
    print(f"  ORPHAN dirs (no processed row): {report.orphan_dirs}")
    print(
        f"    of which reroot-recoverable:  {report.orphans_reroot_recoverable} "
        "(a DB row carries this basename)"
    )

    if report.unresolved_samples:
        print("\n--- Sample stored job_dir values that don't resolve ---")
        for s in report.unresolved_samples:
            print(f"  {s}")
    if report.orphan_samples:
        print("\n--- Sample orphan dir names (on disk, unreferenced) ---")
        for name in report.orphan_samples:
            print(f"  {name}")

    # Interpretation hint -- which fix path the numbers point to.
    print("\n--- Interpretation ---")
    null_n = report.bins.get(_NULL, 0)
    missing_n = report.bins.get(_DIR_MISSING, 0) + report.bins.get(_ESCAPES, 0)
    if skipped == 0:
        print("  No gap: clean.py resolves every clean-set row to an existing dir.")
    elif null_n >= missing_n and null_n > 0:
        print(
            "  Gap dominated by NULL job_dir. dashboard --fix-unlinked "
            "(with a matching --job-dir-pattern) can relink these, then re-run "
            "clean."
        )
    elif report.reroot_recoverable > 0:
        print(
            "  Gap dominated by non-null paths that don't resolve, and many are "
            "reroot-recoverable: the DB stores a different root than this one "
            "(prefix mismatch). A basename reroot of job_dir would recover them."
        )
    else:
        print(
            "  Gap dominated by non-null paths that don't resolve and are NOT "
            "reroot-recoverable: on-disk dir names differ from the DB basenames "
            "(different job_dir_pattern/prefix). A name-based relink keyed on "
            "orig_index/id is needed."
        )


def main() -> None:
    """CLI entry point for the coverage diagnostic."""
    parser = argparse.ArgumentParser(
        prog="python -m oact_utilities.workflows.diagnose_coverage",
        description=(
            "Read-only diagnostic: explain why clean.py processes fewer "
            "directories than exist on disk."
        ),
    )
    parser.add_argument("db_path", type=Path, help="Path to workflow SQLite database")
    parser.add_argument(
        "root_dir", type=Path, help="Root directory containing job subdirectories"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel threads for on-disk existence probes (default: 8)",
    )
    args = parser.parse_args()

    if not args.db_path.exists():
        print(f"Error: database not found: {args.db_path}", file=sys.stderr)
        sys.exit(1)
    if not args.root_dir.is_dir():
        print(f"Error: root directory not found: {args.root_dir}", file=sys.stderr)
        sys.exit(1)

    report = diagnose_coverage(args.db_path, args.root_dir, workers=args.workers)
    print_report(report, args.db_path, args.root_dir.resolve())


if __name__ == "__main__":
    main()
