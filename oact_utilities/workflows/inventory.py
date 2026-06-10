"""On-disk file inventory for ORCA job-directory corpora (no database needed).

Walks every job directory under a root and reports, per directory and in
aggregate, how many files/subdirectories it holds, the cumulative byte size,
the breakdown by file type, and whether it carries nested scratch directories
(``orca_tmp_*``). This is a pure-filesystem audit -- it never opens the workflow
DB and never deletes anything. Use it to find the directories whose size is
"quite frankly ridiculous" and to see exactly what is sitting in them before
deciding how to clean.

The scratch-vs-essential classification mirrors clean.py exactly (the patterns
are imported from it), so the "reclaimable" totals here match what
``clean --clean-all`` / ``--purge-*`` would actually remove. Aligning the
inventory with the DB (job status, orig_index) is intentionally out of scope --
do that later with clean.

Usage:
    python -m oact_utilities.workflows.inventory jobs/
    python -m oact_utilities.workflows.inventory jobs/ --top 40
    python -m oact_utilities.workflows.inventory jobs/ --csv inventory.csv
"""

from __future__ import annotations

import argparse
import csv
import heapq
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

# Single source of truth for what counts as scratch vs. an essential corpus
# file. Imported from clean.py so this audit and the actual cleanup never drift.
from .clean import (
    _BAS_FILE_PATTERNS,
    _ORCA_TMP_DIR_RE,
    _TMP_FILE_PATTERNS,
    _format_size,
    _is_excluded,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# File-type and classification helpers
# ---------------------------------------------------------------------------

# Numbered scratch families collapse to a single bucket so the type breakdown
# does not explode into ".tmp.0", ".tmp.1", ".bas12", "core.4711", ...
_TMP_EXT_RE = re.compile(r"\.tmp(\.\d+)?$")
_BAS_EXT_RE = re.compile(r"\.bas\d*$")
_CORE_EXT_RE = re.compile(r"^core(\.\d+)?$|\.core$")

# Classification bucket labels (also the column order in reports).
_CLASS_ESSENTIAL = "essential"
_CLASS_SCRATCH_TMP = "scratch_tmp"
_CLASS_SCRATCH_BAS = "scratch_bas"
_CLASS_OTHER = "other"
_CLASS_ORDER = (_CLASS_ESSENTIAL, _CLASS_SCRATCH_TMP, _CLASS_SCRATCH_BAS, _CLASS_OTHER)

# How many largest files each job keeps for the global top-files list.
_PER_JOB_TOP_FILES = 5


def _ext_key(name: str) -> str:
    """Return a normalized file-type key for a filename.

    Numbered scratch variants (``.tmp.3``, ``.bas12``, ``core.4711``) collapse to
    ``.tmp`` / ``.bas`` / ``core``. Gzipped outputs keep their double suffix
    (``orca.out.gz`` -> ``.out.gz``) so they stay distinguishable from plain
    ``.gz``. Files with no extension map to ``(no ext)``.
    """
    if _TMP_EXT_RE.search(name):
        return ".tmp"
    if _BAS_EXT_RE.search(name):
        return ".bas"
    if _CORE_EXT_RE.search(name):
        return "core"
    suffixes = Path(name).suffixes
    if not suffixes:
        return "(no ext)"
    if suffixes[-1].lower() == ".gz" and len(suffixes) >= 2:
        return "".join(suffixes[-2:]).lower()
    return suffixes[-1].lower()


def _classify_file(name: str, in_tmp_dir: bool) -> str:
    """Classify a file as essential / scratch_tmp / scratch_bas / other.

    Anything living inside an ``orca_tmp_*`` directory is scratch regardless of
    its own name (clean removes the whole directory). Otherwise the essential
    exclusion set wins, then the tmp/bas scratch patterns, else ``other``.
    """
    if in_tmp_dir:
        return _CLASS_SCRATCH_TMP
    if _is_excluded(name):
        return _CLASS_ESSENTIAL
    for pattern in _TMP_FILE_PATTERNS:
        if pattern.search(name):
            return _CLASS_SCRATCH_TMP
    for pattern in _BAS_FILE_PATTERNS:
        if pattern.search(name):
            return _CLASS_SCRATCH_BAS
    return _CLASS_OTHER


# ---------------------------------------------------------------------------
# Per-job scan
# ---------------------------------------------------------------------------


@dataclass
class JobInventory:
    """Inventory of a single job directory (recursive, symlinks not followed)."""

    path: Path
    n_files: int = 0
    n_subdirs: int = 0
    total_bytes: int = 0
    ext_bytes: Counter = field(default_factory=Counter)
    ext_counts: Counter = field(default_factory=Counter)
    class_bytes: Counter = field(default_factory=Counter)
    class_counts: Counter = field(default_factory=Counter)
    n_nested_tmp_dirs: int = 0
    nested_tmp_bytes: int = 0
    # (job-relative path, size) for the largest files in this job.
    largest_files: list[tuple[str, int]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def reclaimable_bytes(self) -> int:
        """Bytes clean would remove (everything that is not essential)."""
        return self.total_bytes - self.class_bytes.get(_CLASS_ESSENTIAL, 0)


def scan_job_dir(job_dir: Path) -> JobInventory:
    """Recursively inventory one job directory without following symlinks.

    Symlinked files contribute 0 bytes and symlinked directories are not
    descended into (mirrors clean.py's size accounting so a link into the final
    corpus is never counted as local weight).
    """
    inv = JobInventory(path=job_dir)
    files_seen: list[tuple[str, int]] = []

    for dirpath, dirnames, filenames in os.walk(job_dir, followlinks=False):
        inv.n_subdirs += len(dirnames)
        for d in dirnames:
            if _ORCA_TMP_DIR_RE.match(d):
                inv.n_nested_tmp_dirs += 1

        rel = os.path.relpath(dirpath, job_dir)
        in_tmp_dir = rel != "." and any(
            _ORCA_TMP_DIR_RE.match(part) for part in rel.split(os.sep)
        )

        for name in filenames:
            fp = os.path.join(dirpath, name)
            try:
                size = 0 if os.path.islink(fp) else os.path.getsize(fp)
            except OSError as e:
                inv.errors.append(f"{fp}: stat failed: {e}")
                size = 0

            inv.n_files += 1
            inv.total_bytes += size

            ext = _ext_key(name)
            inv.ext_bytes[ext] += size
            inv.ext_counts[ext] += 1

            cls = _classify_file(name, in_tmp_dir)
            inv.class_bytes[cls] += size
            inv.class_counts[cls] += 1

            if in_tmp_dir:
                inv.nested_tmp_bytes += size

            files_seen.append((os.path.relpath(fp, job_dir), size))

    inv.largest_files = heapq.nlargest(
        _PER_JOB_TOP_FILES, files_seen, key=lambda t: t[1]
    )
    return inv


# ---------------------------------------------------------------------------
# Aggregate over all job directories
# ---------------------------------------------------------------------------


@dataclass
class Corpus:
    """Aggregate inventory across every job directory under a root."""

    root: Path
    jobs: list[JobInventory] = field(default_factory=list)
    loose_files: int = 0
    loose_bytes: int = 0
    ext_bytes: Counter = field(default_factory=Counter)
    ext_counts: Counter = field(default_factory=Counter)
    class_bytes: Counter = field(default_factory=Counter)
    class_counts: Counter = field(default_factory=Counter)
    total_files: int = 0
    total_subdirs: int = 0
    total_bytes: int = 0
    jobs_with_nested_tmp: int = 0
    total_nested_tmp_dirs: int = 0
    nested_tmp_bytes: int = 0
    errors: list[str] = field(default_factory=list)


def inventory_root(root: Path, workers: int = 8, limit: int | None = None) -> Corpus:
    """Scan every immediate subdirectory of ``root`` as a job directory.

    Files sitting directly under ``root`` (not inside a job subdirectory) are
    tallied separately as "loose" and not classified.
    """
    root = root.resolve()
    corpus = Corpus(root=root)

    job_dirs: list[Path] = []
    for entry in sorted(root.iterdir()):
        if entry.is_symlink():
            # Do not traverse a symlink at the top level (could point into the
            # corpus); note it but skip.
            continue
        if entry.is_dir():
            job_dirs.append(entry)
        elif entry.is_file():
            corpus.loose_files += 1
            try:
                corpus.loose_bytes += entry.stat().st_size
            except OSError:
                pass

    if limit is not None:
        job_dirs = job_dirs[:limit]

    print(f"Scanning {len(job_dirs)} job directories under {root} ...")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(scan_job_dir, jd): jd for jd in job_dirs}
        completed_iter = as_completed(futures)
        if tqdm is not None:
            completed_iter = tqdm(
                completed_iter, total=len(futures), desc="Scanning", unit="job"
            )
        for fut in completed_iter:
            inv = fut.result()
            corpus.jobs.append(inv)
            corpus.total_files += inv.n_files
            corpus.total_subdirs += inv.n_subdirs
            corpus.total_bytes += inv.total_bytes
            corpus.ext_bytes.update(inv.ext_bytes)
            corpus.ext_counts.update(inv.ext_counts)
            corpus.class_bytes.update(inv.class_bytes)
            corpus.class_counts.update(inv.class_counts)
            if inv.n_nested_tmp_dirs:
                corpus.jobs_with_nested_tmp += 1
                corpus.total_nested_tmp_dirs += inv.n_nested_tmp_dirs
                corpus.nested_tmp_bytes += inv.nested_tmp_bytes
            corpus.errors.extend(inv.errors)

    corpus.jobs.sort(key=lambda j: j.total_bytes, reverse=True)
    return corpus


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _pct(part: int, whole: int) -> str:
    """Format ``part`` as a percentage of ``whole`` (0.0% when whole is 0)."""
    return f"{(100.0 * part / whole):.1f}%" if whole else "0.0%"


def _print_report(corpus: Corpus, top: int) -> None:
    """Print the full console report for a scanned corpus."""
    n_jobs = len(corpus.jobs)
    print(f"\n{'=' * 70}")
    print("  Job Directory File Inventory")
    print(f"{'=' * 70}")
    print(f"  Root:            {corpus.root}")
    print(f"  Job directories: {n_jobs}")
    print(f"  Total files:     {corpus.total_files}")
    print(f"  Total subdirs:   {corpus.total_subdirs}")
    print(f"  Total size:      {_format_size(corpus.total_bytes)}")
    if n_jobs:
        print(
            f"  Mean per job:    {_format_size(corpus.total_bytes // n_jobs)} "
            f"({corpus.total_files // n_jobs} files)"
        )
    if corpus.loose_files:
        print(
            f"  Loose files at root (not in a job dir): {corpus.loose_files} "
            f"({_format_size(corpus.loose_bytes)})"
        )

    # --- Classification (reclaimable vs essential) ---
    print("\n--- By classification (matches clean.py delete rules) ---")
    print("  class | files | size | % of size")
    for cls in _CLASS_ORDER:
        cnt = corpus.class_counts.get(cls, 0)
        size = corpus.class_bytes.get(cls, 0)
        print(
            f"  {cls} | {cnt} | {_format_size(size)} | "
            f"{_pct(size, corpus.total_bytes)}"
        )
    reclaimable = corpus.total_bytes - corpus.class_bytes.get(_CLASS_ESSENTIAL, 0)
    print(
        f"  -> reclaimable (non-essential): {_format_size(reclaimable)} "
        f"({_pct(reclaimable, corpus.total_bytes)})"
    )

    # --- By file type ---
    print(f"\n--- By file type (top {top} by size) ---")
    print("  type | files | size | % of size")
    for ext, size in corpus.ext_bytes.most_common(top):
        print(
            f"  {ext} | {corpus.ext_counts[ext]} | {_format_size(size)} | "
            f"{_pct(size, corpus.total_bytes)}"
        )

    # --- Nested scratch directories ---
    print("\n--- Nested scratch dirs (orca_tmp_*) ---")
    print(f"  Job dirs with nested tmp dirs: {corpus.jobs_with_nested_tmp} of {n_jobs}")
    print(f"  Total nested tmp dirs:         {corpus.total_nested_tmp_dirs}")
    print(f"  Bytes inside nested tmp dirs:  {_format_size(corpus.nested_tmp_bytes)}")

    # --- Largest job directories ---
    print(f"\n--- Largest {top} job directories ---")
    print("  size | files | subdirs | nested_tmp | job_dir")
    for inv in corpus.jobs[:top]:
        rel = _rel(inv.path, corpus.root)
        print(
            f"  {_format_size(inv.total_bytes)} | {inv.n_files} | {inv.n_subdirs} | "
            f"{inv.n_nested_tmp_dirs} | {rel}"
        )

    # --- Largest individual files ---
    global_files: list[tuple[int, str]] = []
    for inv in corpus.jobs:
        job_rel = _rel(inv.path, corpus.root)
        for file_rel, size in inv.largest_files:
            global_files.append((size, f"{job_rel}/{file_rel}"))
    top_files = heapq.nlargest(top, global_files, key=lambda t: t[0])
    print(f"\n--- Largest {top} individual files ---")
    print("  size | path")
    for size, path in top_files:
        print(f"  {_format_size(size)} | {path}")

    if corpus.errors:
        print(f"\nWarnings ({len(corpus.errors)}):")
        for err in corpus.errors[:20]:
            print(f"  - {err}")
        if len(corpus.errors) > 20:
            print(f"  ... and {len(corpus.errors) - 20} more")


def _rel(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root`` as a string (fall back to name)."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def _write_csv(corpus: Corpus, csv_path: Path) -> None:
    """Write one spreadsheet row per job directory.

    The ``top_exts`` column lists the largest file types in that job
    (``.gbw:1.2 GB;.tmp:300.0 MB``) so per-folder composition is visible without
    a second pass.
    """
    header = [
        "job_dir",
        "n_files",
        "n_subdirs",
        "total_bytes",
        "total_size",
        "n_nested_tmp_dirs",
        "nested_tmp_bytes",
        "essential_bytes",
        "scratch_tmp_bytes",
        "scratch_bas_bytes",
        "other_bytes",
        "reclaimable_bytes",
        "largest_file",
        "largest_file_bytes",
        "top_exts",
    ]
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for inv in corpus.jobs:
            top_exts = ";".join(
                f"{ext}:{_format_size(size)}"
                for ext, size in inv.ext_bytes.most_common(5)
            )
            largest_name, largest_bytes = (
                inv.largest_files[0] if inv.largest_files else ("", 0)
            )
            writer.writerow(
                [
                    _rel(inv.path, corpus.root),
                    inv.n_files,
                    inv.n_subdirs,
                    inv.total_bytes,
                    _format_size(inv.total_bytes),
                    inv.n_nested_tmp_dirs,
                    inv.nested_tmp_bytes,
                    inv.class_bytes.get(_CLASS_ESSENTIAL, 0),
                    inv.class_bytes.get(_CLASS_SCRATCH_TMP, 0),
                    inv.class_bytes.get(_CLASS_SCRATCH_BAS, 0),
                    inv.class_bytes.get(_CLASS_OTHER, 0),
                    inv.reclaimable_bytes,
                    largest_name,
                    largest_bytes,
                    top_exts,
                ]
            )
    print(f"\nPer-job inventory written to {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the on-disk job-directory inventory."""
    parser = argparse.ArgumentParser(
        prog="python -m oact_utilities.workflows.inventory",
        description=(
            "Inventory files in every job directory under a root (file counts, "
            "cumulative sizes, type breakdown, nested scratch dirs). Pure "
            "filesystem audit -- no database, no deletion."
        ),
    )
    parser.add_argument(
        "root_dir",
        type=Path,
        help="Root directory whose immediate subdirectories are the job dirs",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write a per-job-directory CSV (one row per folder) to this path",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        metavar="N",
        help="How many entries to show in the top lists (default: 20)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel scan workers (default: 8)",
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=None,
        metavar="N",
        help="Limit to first N job directories for testing",
    )
    args = parser.parse_args()

    if not args.root_dir.is_dir():
        print(f"Error: root directory not found: {args.root_dir}", file=sys.stderr)
        sys.exit(1)

    corpus = inventory_root(args.root_dir, workers=args.workers, limit=args.debug)
    _print_report(corpus, top=args.top)
    if args.csv is not None:
        _write_csv(corpus, args.csv)


if __name__ == "__main__":
    main()
