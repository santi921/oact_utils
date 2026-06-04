"""Job directory cleanup utility for ORCA workflow campaigns.

Removes superfluous scratch files from completed job directories, optionally
purges failed job directories, and (for final-home reclamation) purges
never-finished job directories that the on-disk content check confirms are
incomplete. Purges leave a .do_not_rerun.json marker to prevent resubmission.

WARNING: --purge-incomplete (and --validate-db) are for FINAL cleanup of a
completed/transferred corpus, NOT for an ongoing campaign. They delete dirs for
jobs the DB still calls running/to_run/timeout -- during an active campaign
those statuses mean "in flight" and purging them destroys live or pending work.
Only run --purge-incomplete when the campaign is finished and nothing is running
(e.g. after transferring the dataset and DB to their final home).

Usage:
    python -m oact_utilities.workflows.clean workflow.db jobs/ --clean-tmp
    python -m oact_utilities.workflows.clean workflow.db jobs/ --clean-all --execute
    python -m oact_utilities.workflows.clean workflow.db jobs/ --purge-failed --execute
    python -m oact_utilities.workflows.clean workflow.db jobs/ --validate-db
    python -m oact_utilities.workflows.clean workflow.db jobs/ --purge-incomplete --execute
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from ..utils.analysis import parse_scf_steps
from ..utils.create import read_geom_from_inp_file
from ..utils.status import check_job_termination, parse_failure_reason, pull_log_file
from .architector_workflow import ArchitectorWorkflow, JobRecord, JobStatus

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Marker file written by --purge-failed. Its presence blocks the job from
# ever being resubmitted by submit_jobs and flips the row to FAILED on
# dashboard --update. Shared by clean.py (writer), submit_jobs.py, and
# dashboard.py (readers).
MARKER_FILENAME = ".do_not_rerun.json"
MARKER_ERROR_MESSAGE = "Blocked by .do_not_rerun.json marker"

# Marker discriminator distinguishing --purge-incomplete from --purge-failed.
MARKER_PURGE_TYPE_INCOMPLETE = "incomplete_archive"

# DB statuses eligible for --purge-incomplete (the non-corpus, never-finished set).
_INCOMPLETE_STATUSES = frozenset(
    {JobStatus.RUNNING.value, JobStatus.TO_RUN.value, JobStatus.TIMEOUT.value}
)

# --validate-db internals. Constants, not CLI knobs: a tunable abort threshold on
# a destructive op is a footgun. Sample is stratified to include the purge
# population; the gate aborts on any real mismatch and fails closed when too few
# rows could be verified.
_VALIDATE_SAMPLE_SIZE = 100
_VALIDATE_MIN_VERIFIED_FRACTION = 0.5


def is_marker_blocked(job_dir: Path) -> bool:
    """Return True if job_dir carries the do-not-rerun marker."""
    return (job_dir / MARKER_FILENAME).exists()


# Files that must NEVER be deleted by any cleanup pattern.
_EXCLUSION_SET = frozenset(
    {
        "orca.out",
        "orca.out.gz",
        "orca.inp",
        "orca.inp.gz",
        "orca.engrad",
        "orca.engrad.gz",
        "orca.xyz",
        "orca_metrics.json",
        "generator_metrics.json",
        "orca.property.txt",
        "orca.property.txt.gz",
        "orca.gbw",
        "orca.densities",
        "flux_job.flux",
        "slurm_job.sh",
        "sella_status.txt",
        "sella.log",
        "opt.traj",
        "sella_config.json",
        "run_sella.py",
        "results.pkl",
        "sella_driver.log",
        MARKER_FILENAME,
    }
)

# Regex for orca_atom*.out files (also excluded)
_ORCA_ATOM_RE = re.compile(r"^orca_atom\d+\.out$")

# Regex for trajectory files (also excluded)
_TRJ_RE = re.compile(r"_trj\.xyz$")

# Regex for orca_tmp_* directory names (must be direct child of job dir)
_ORCA_TMP_DIR_RE = re.compile(r"^orca_tmp_[a-zA-Z0-9_]+$")

# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------

# Compiled regexes for cleanup categories
_TMP_FILE_PATTERNS = [
    re.compile(r"\.tmp$"),
    re.compile(r"\.tmp\.\d+$"),
    re.compile(r"^core$"),
    re.compile(r"^core\.\d+$"),
    re.compile(r"\.core$"),
]

_BAS_FILE_PATTERNS = [
    re.compile(r"\.bas$"),
    re.compile(r"\.bas\d+$"),
]


def _is_excluded(name: str) -> bool:
    """Check if a filename is in the exclusion set."""
    if name in _EXCLUSION_SET:
        return True
    if _ORCA_ATOM_RE.match(name):
        return True
    if _TRJ_RE.search(name):
        return True
    return False


def _match_cleanup_patterns(name: str, is_dir: bool, categories: set[str]) -> bool:
    """Check if a filename/dirname matches active cleanup categories.

    Args:
        name: The file or directory name (not full path).
        is_dir: True if this entry is a directory.
        categories: Set of active category strings ("tmp", "bas").

    Returns:
        True if the name matches a pattern in the active categories.
    """
    if _is_excluded(name):
        return False

    if "tmp" in categories:
        if is_dir and _ORCA_TMP_DIR_RE.match(name):
            return True
        if not is_dir:
            for pattern in _TMP_FILE_PATTERNS:
                if pattern.search(name):
                    return True

    if "bas" in categories:
        if not is_dir:
            for pattern in _BAS_FILE_PATTERNS:
                if pattern.search(name):
                    return True

    return False


# ---------------------------------------------------------------------------
# Size helpers
# ---------------------------------------------------------------------------


def _format_size(size_bytes: int) -> str:
    """Format a byte count as a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def _get_dir_size(path: Path) -> int:
    """Recursively sum file sizes in a directory without following symlinks.

    Uses os.walk(followlinks=False) and skips symlinked files so a symlink
    into the final corpus is never traversed and counted as reclaimable.
    """
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path, followlinks=False):
        for name in filenames:
            fp = os.path.join(dirpath, name)
            try:
                if not os.path.islink(fp):
                    total += os.path.getsize(fp)
            except OSError:
                pass
    return total


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


def _resolve_job_dir(job_dir_value: str, root_dir: Path) -> Path | None:
    """Resolve a job_dir DB value to a safe absolute path.

    Returns None if the resolved path escapes root_dir.
    """
    raw = Path(job_dir_value)
    if not raw.is_absolute():
        raw = root_dir / raw
    resolved = raw.resolve()
    root_resolved = root_dir.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        return None
    return resolved


# ---------------------------------------------------------------------------
# DB <-> folder validation (--validate-db)
# ---------------------------------------------------------------------------


class ValidationOutcome(str, Enum):
    """Per-row result of the DB<->folder alignment check."""

    MATCH = "match"  # parsed; element multiset + atom count equal the DB row
    MISMATCH = "mismatch"  # parsed but elements/natoms differ
    UNVERIFIABLE = "unverifiable"  # missing dir/inp, parse error, external coords


@dataclass(frozen=True)
class ValidationResult:
    """Aggregate outcome of the DB<->folder alignment sanity check."""

    match_count: int
    mismatch_count: int
    unverifiable_count: int
    mismatches: list[tuple[int, str, str]]  # (orig_index, db_elements, inp_elements)
    passed: bool

    @property
    def sampled(self) -> int:
        return self.match_count + self.mismatch_count + self.unverifiable_count

    @property
    def verifiable(self) -> int:
        return self.match_count + self.mismatch_count


def _check_row_alignment(
    job_record: JobRecord, root_dir: Path
) -> tuple[ValidationOutcome, str | None]:
    """Compare one DB row's structure against its on-disk orca.inp.

    Reuses read_geom_from_inp_file (no new parser, no ASE Atoms construction).
    Compares the element multiset and atom count only -- charge/spin/coordinates
    are not checked (the inp is round-tripped from the DB geometry, so they prove
    linkage at best; elements + natoms already catch the gross-error case the gate
    exists for).

    Returns:
        (outcome, inp_elements_str). inp_elements_str is the semicolon-joined
        parsed elements for MATCH/MISMATCH reporting, or None when UNVERIFIABLE.
    """
    if job_record.job_dir is None:
        return ValidationOutcome.UNVERIFIABLE, None
    resolved = _resolve_job_dir(job_record.job_dir, root_dir)
    if resolved is None or not resolved.is_dir():
        return ValidationOutcome.UNVERIFIABLE, None

    inp_file = resolved / "orca.inp"
    if not inp_file.is_file():
        return ValidationOutcome.UNVERIFIABLE, None

    try:
        atoms = read_geom_from_inp_file(str(inp_file), ase_format_tf=False)
        inp_elements = [str(a["element"]) for a in atoms]  # type: ignore[union-attr]
    except (OSError, ValueError, IndexError, KeyError, TypeError):
        return ValidationOutcome.UNVERIFIABLE, None

    if not inp_elements:
        # e.g. "* xyzfile" external coordinates -- nothing to compare.
        return ValidationOutcome.UNVERIFIABLE, None

    inp_elements_str = ";".join(inp_elements)
    db_elements = [e for e in (job_record.elements or "").split(";") if e]

    if (
        sorted(inp_elements) == sorted(db_elements)
        and len(inp_elements) == job_record.natoms
    ):
        return ValidationOutcome.MATCH, inp_elements_str
    return ValidationOutcome.MISMATCH, inp_elements_str


def validate_db_folder_alignment(
    wf: ArchitectorWorkflow, root_dir: Path, workers: int = 4
) -> ValidationResult:
    """Sample DB rows and verify they map to the on-disk job directories.

    A cheap fast-fail that the DB handed to the tool plausibly corresponds to the
    directories on disk, catching gross operator error (wrong root/db) before any
    destructive action. NOT the primary safety mechanism -- the per-job content
    check in the purge worker is. The sample is stratified to include the purge
    population (running/to_run/timeout) so the data actually at risk is checked.

    Gate (hardcoded, fail-closed): abort if nothing verifiable, or if the
    verifiable fraction is below _VALIDATE_MIN_VERIFIED_FRACTION, or on any
    MISMATCH.

    Returns:
        ValidationResult with bucket counts, the mismatching rows, and passed.
    """
    incomplete = wf.get_jobs_by_status(
        [JobStatus.RUNNING, JobStatus.TO_RUN, JobStatus.TIMEOUT],
        include_geometry=False,
    )
    other = wf.get_jobs_by_status(
        [JobStatus.COMPLETED, JobStatus.FAILED],
        include_geometry=False,
    )

    half = _VALIDATE_SAMPLE_SIZE // 2
    n_incomplete = min(len(incomplete), half)
    sample = random.sample(incomplete, n_incomplete) if n_incomplete else []
    remaining = _VALIDATE_SAMPLE_SIZE - len(sample)
    n_other = min(len(other), remaining)
    if n_other:
        sample += random.sample(other, n_other)

    match_count = 0
    mismatch_count = 0
    unverifiable_count = 0
    mismatches: list[tuple[int, str, str]] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_check_row_alignment, job, root_dir): job for job in sample
        }
        for fut in as_completed(futures):
            job = futures[fut]
            outcome, inp_elements = fut.result()
            if outcome is ValidationOutcome.MATCH:
                match_count += 1
            elif outcome is ValidationOutcome.MISMATCH:
                mismatch_count += 1
                mismatches.append(
                    (job.orig_index, job.elements or "", inp_elements or "")
                )
            else:
                unverifiable_count += 1

    sampled = match_count + mismatch_count + unverifiable_count
    verifiable = match_count + mismatch_count
    if sampled == 0 or verifiable == 0:
        passed = False
    elif verifiable / sampled < _VALIDATE_MIN_VERIFIED_FRACTION:
        passed = False
    else:
        passed = mismatch_count == 0

    return ValidationResult(
        match_count=match_count,
        mismatch_count=mismatch_count,
        unverifiable_count=unverifiable_count,
        mismatches=mismatches,
        passed=passed,
    )


def _print_validation_report(result: ValidationResult) -> None:
    """Print the DB<->folder validation bucket breakdown and verdict."""
    print("\n--- DB <-> Folder Validation ---")
    print(f"  Sampled:      {result.sampled}")
    print(f"  Match:        {result.match_count}")
    print(f"  Mismatch:     {result.mismatch_count}")
    print(f"  Unverifiable: {result.unverifiable_count}")
    if result.mismatches:
        print("  Mismatching rows (orig_index: db_elements != inp_elements):")
        for orig_index, db_el, inp_el in result.mismatches[:20]:
            print(f"    - {orig_index}: {db_el} != {inp_el}")
        if len(result.mismatches) > 20:
            print(f"    ... and {len(result.mismatches) - 20} more")
    print(f"  Verdict:      {'PASS' if result.passed else 'FAIL'}")


# ---------------------------------------------------------------------------
# Failure info extraction
# ---------------------------------------------------------------------------


def _extract_failure_info(job_dir: Path) -> dict[str, str | int | None]:
    """Parse SCF steps and failure reason from the ORCA output in job_dir.

    Returns:
        Dict with "scf_steps" (int or None) and "failure_reason" (str or None).
    """
    result: dict[str, str | int | None] = {
        "scf_steps": None,
        "failure_reason": None,
    }
    try:
        log_file = pull_log_file(str(job_dir))
    except FileNotFoundError:
        return result

    try:
        result["scf_steps"] = parse_scf_steps(log_file)
    except Exception:
        pass

    try:
        result["failure_reason"] = parse_failure_reason(log_file)
    except Exception:
        pass

    return result


def _write_marker_file(job_dir: Path, metadata: dict[str, str | int | None]) -> None:
    """Write .do_not_rerun.json marker with job metadata."""
    marker_path = job_dir / MARKER_FILENAME
    marker_data = {
        "generated_by": "python -m oact_utilities.workflows.clean",
        "date": datetime.now(tz=timezone.utc).isoformat(),
        **metadata,
    }
    marker_path.write_text(json.dumps(marker_data, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Per-job workers
# ---------------------------------------------------------------------------


def _process_job(
    job_dir: Path,
    root_dir: Path,
    categories: set[str],
    execute: bool,
    hours_cutoff: int,
    optimizer: str | None,
    skip_revalidation: bool = False,
) -> tuple[list[tuple[Path, int, bool]], int, list[str]]:
    """Scan a job directory for cleanup targets.

    Pure I/O worker -- no DB access, no shared mutable state.

    Args:
        skip_revalidation: If True, skip the on-disk completion check.
            Used when cleaning failed/to_run jobs where we know the job
            is not completed but still want to remove scratch files.

    Returns:
        (matched_files, bytes_freed, errors)
        matched_files: list of (path, size_bytes, is_directory)
        bytes_freed: total bytes actually freed (0 in dry-run)
        errors: list of error messages
    """
    matched: list[tuple[Path, int, bool]] = []
    bytes_freed = 0
    errors: list[str] = []

    # Revalidate job completion on disk (skip for non-completed jobs)
    if not skip_revalidation:
        try:
            status = check_job_termination(
                str(job_dir), hours_cutoff=hours_cutoff, optimizer=optimizer
            )
        except Exception as e:
            errors.append(f"{job_dir}: revalidation error: {e}")
            return matched, bytes_freed, errors

        if status != 1:
            errors.append(
                f"{job_dir}: revalidation returned {status} (expected 1=completed), skipping"
            )
            return matched, bytes_freed, errors

    # Scan direct children of job directory
    try:
        entries = list(job_dir.iterdir())
    except OSError as e:
        errors.append(f"{job_dir}: cannot list directory: {e}")
        return matched, bytes_freed, errors

    for entry in entries:
        is_dir = entry.is_dir()
        if not _match_cleanup_patterns(entry.name, is_dir, categories):
            continue

        try:
            if is_dir:
                size = _get_dir_size(entry)
            else:
                size = entry.stat().st_size
        except OSError:
            size = 0

        matched.append((entry, size, is_dir))

        if execute:
            try:
                if is_dir:
                    if entry.is_symlink():
                        entry.unlink()  # Remove symlink, don't follow it
                    else:
                        shutil.rmtree(str(entry), ignore_errors=True)
                else:
                    entry.unlink()
                bytes_freed += size
            except (OSError, PermissionError) as e:
                errors.append(f"{entry}: deletion failed: {e}")

    return matched, bytes_freed, errors


def _purge_failed_job(
    job_dir: Path,
    root_dir: Path,
    db_path: Path,
    job_id: int,
    execute: bool,
    job_metadata: dict[str, str | int | None],
) -> tuple[list[tuple[Path, int, bool]], int, list[str]]:
    """Purge a failed job directory: extract metadata, write marker, delete contents.

    Opens a short-lived read-only DB connection for TOCTOU re-check.

    Returns:
        (matched_files, bytes_freed, errors)
    """
    matched: list[tuple[Path, int, bool]] = []
    bytes_freed = 0
    errors: list[str] = []

    if not job_dir.is_dir():
        errors.append(f"{job_dir}: directory not found, skipping purge")
        return matched, bytes_freed, errors

    # TOCTOU re-check: confirm job is still failed in DB.
    # Uses a short-lived connection with DELETE mode (never WAL) to avoid
    # poisoning the DB with stale -wal/-shm files on Lustre/VAST. Reads the
    # current journal_mode first (shared lock) and only writes if it differs;
    # the write-form PRAGMA takes a reserved lock that contends badly with
    # active Parsl writers when N workers call this concurrently.
    try:
        with sqlite3.connect(str(db_path), timeout=5.0) as conn:
            conn.execute("PRAGMA busy_timeout=5000")
            current = conn.execute("PRAGMA journal_mode").fetchone()
            if not current or current[0].lower() != "delete":
                conn.execute("PRAGMA journal_mode=DELETE")
            cur = conn.execute("SELECT status FROM structures WHERE id = ?", (job_id,))
            row = cur.fetchone()
    except Exception as e:
        errors.append(f"{job_dir}: TOCTOU DB check failed: {e}, skipping purge")
        return matched, bytes_freed, errors

    if row is None:
        errors.append(f"{job_dir}: job_id={job_id} not found in DB, skipping purge")
        return matched, bytes_freed, errors

    current_status = row[0]
    if current_status != JobStatus.FAILED.value:
        errors.append(
            f"{job_dir}: status changed to '{current_status}' (expected 'failed'), "
            f"skipping purge"
        )
        return matched, bytes_freed, errors

    # Extract failure info from ORCA output before deletion
    failure_info = _extract_failure_info(job_dir)
    metadata = {**job_metadata, **failure_info}

    # Inventory all contents
    try:
        entries = list(job_dir.iterdir())
    except OSError as e:
        errors.append(f"{job_dir}: cannot list directory: {e}")
        return matched, bytes_freed, errors

    for entry in entries:
        # Skip the marker file if it already exists
        if entry.name == MARKER_FILENAME:
            continue
        is_dir = entry.is_dir()
        try:
            size = _get_dir_size(entry) if is_dir else entry.stat().st_size
        except OSError:
            size = 0
        matched.append((entry, size, is_dir))

    if execute:
        # Write marker first, then delete everything else
        try:
            _write_marker_file(job_dir, metadata)
        except (OSError, PermissionError) as e:
            errors.append(f"{job_dir}: marker write failed: {e}, skipping purge")
            return matched, bytes_freed, errors
        for entry, size, is_dir in matched:
            try:
                if is_dir:
                    if entry.is_symlink():
                        entry.unlink()  # Remove symlink, don't follow it
                    else:
                        shutil.rmtree(str(entry), ignore_errors=True)
                else:
                    entry.unlink()
                bytes_freed += size
            except (OSError, PermissionError) as e:
                errors.append(f"{entry}: deletion failed: {e}")

    return matched, bytes_freed, errors


def _purge_incomplete_job(
    job_dir: Path,
    root_dir: Path,
    db_path: Path,
    job_id: int,
    execute: bool,
    hours_cutoff: int,
    optimizer: str | None,
    job_metadata: dict[str, str | int | None],
) -> tuple[str, list[tuple[Path, int, bool]], int, list[str]]:
    """Purge a non-corpus incomplete job directory after a dual safety check.

    Candidate set is the DB running/to_run/timeout population. The on-disk
    content check decides the action and is evaluated BEFORE the marker write,
    so a job that terminated normally between the candidate query and this worker
    is protected rather than marked do-not-rerun.

    Returns:
        (classification, matched_files, bytes_freed, errors). classification is
        one of "purged", "protected", "looks_failed", "skipped".
    """
    matched: list[tuple[Path, int, bool]] = []
    bytes_freed = 0
    errors: list[str] = []

    if not job_dir.is_dir():
        errors.append(f"{job_dir}: directory not found, skipping purge")
        return "skipped", matched, bytes_freed, errors

    # TOCTOU re-check: confirm the job is still in the incomplete population.
    # Short-lived DELETE-mode connection (never WAL) to avoid stale -wal/-shm on
    # Lustre/VAST; read journal_mode first and only write the PRAGMA if needed.
    try:
        with sqlite3.connect(str(db_path), timeout=5.0) as conn:
            conn.execute("PRAGMA busy_timeout=5000")
            current = conn.execute("PRAGMA journal_mode").fetchone()
            if not current or current[0].lower() != "delete":
                conn.execute("PRAGMA journal_mode=DELETE")
            cur = conn.execute("SELECT status FROM structures WHERE id = ?", (job_id,))
            row = cur.fetchone()
    except Exception as e:
        errors.append(f"{job_dir}: TOCTOU DB check failed: {e}, skipping purge")
        return "skipped", matched, bytes_freed, errors

    if row is None:
        errors.append(f"{job_dir}: job_id={job_id} not found in DB, skipping purge")
        return "skipped", matched, bytes_freed, errors

    current_status = row[0]
    if current_status not in _INCOMPLETE_STATUSES:
        errors.append(
            f"{job_dir}: status changed to '{current_status}' "
            f"(expected running/to_run/timeout), skipping purge"
        )
        return "skipped", matched, bytes_freed, errors

    # Content precondition -- the real safety net. Runs BEFORE the marker write.
    try:
        disk_status = check_job_termination(
            str(job_dir), hours_cutoff=hours_cutoff, optimizer=optimizer
        )
    except Exception as e:
        errors.append(f"{job_dir}: content check failed: {e}, skipping purge")
        return "skipped", matched, bytes_freed, errors

    if disk_status == 1:
        # On disk this job actually completed -- it belongs in the corpus.
        return "protected", matched, bytes_freed, errors
    if disk_status == -1:
        # On disk this job failed -- preserve provenance via --purge-failed.
        return "looks_failed", matched, bytes_freed, errors
    # disk_status in (0, -2): no successful termination in content -> purge.

    # Inventory all contents (keep an existing marker so re-runs are idempotent).
    try:
        entries = list(job_dir.iterdir())
    except OSError as e:
        errors.append(f"{job_dir}: cannot list directory: {e}")
        return "skipped", matched, bytes_freed, errors

    for entry in entries:
        if entry.name == MARKER_FILENAME:
            continue
        is_dir = entry.is_dir()
        try:
            size = _get_dir_size(entry) if is_dir else entry.stat().st_size
        except OSError:
            size = 0
        matched.append((entry, size, is_dir))

    if execute:
        # Write marker first; abort the purge if it fails so nothing is deleted
        # without a traceable do-not-rerun record.
        metadata = {
            **job_metadata,
            "purge_type": MARKER_PURGE_TYPE_INCOMPLETE,
            "db_status_at_purge": current_status,
            "disk_status_code": disk_status,
        }
        try:
            _write_marker_file(job_dir, metadata)
        except (OSError, PermissionError) as e:
            errors.append(f"{job_dir}: marker write failed: {e}, skipping purge")
            return "skipped", matched, bytes_freed, errors
        for entry, size, is_dir in matched:
            try:
                if is_dir:
                    if entry.is_symlink():
                        entry.unlink()  # Remove symlink, don't follow it
                    else:
                        shutil.rmtree(str(entry), ignore_errors=True)
                else:
                    entry.unlink()
                bytes_freed += size
            except (OSError, PermissionError) as e:
                errors.append(f"{entry}: deletion failed: {e}")

    return "purged", matched, bytes_freed, errors


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def clean_job_directories(
    db_path: Path,
    root_dir: Path,
    categories: set[str],
    purge_failed: bool = False,
    execute: bool = False,
    workers: int = 4,
    hours_cutoff: int = 24,
    limit: int | None = None,
    verbose: bool = False,
    include_failed: bool = False,
    purge_incomplete: bool = False,
    validate_db: bool = False,
    skip_validation: bool = False,
) -> bool:
    """Orchestrate cleanup of job directories.

    Args:
        db_path: Path to workflow SQLite database.
        root_dir: Root directory containing job subdirectories.
        categories: Set of cleanup categories ("tmp", "bas").
        purge_failed: If True, purge failed job directories.
        execute: If True, actually delete files. Otherwise dry-run.
        workers: Number of parallel workers.
        hours_cutoff: Hours before revalidation considers a job timed out.
        limit: If set, process at most this many jobs.
        verbose: If True, show per-file listings.
        include_failed: If True, also clean scratch files from failed,
            timeout, and to_run jobs (not just completed).
        purge_incomplete: If True, full-purge running/to_run/timeout job
            directories that the on-disk content check confirms are incomplete.
        validate_db: If True, run the DB<->folder alignment sanity check.
        skip_validation: If True, bypass the validation gate for
            --purge-incomplete (a loud warning is printed).

    Returns:
        True on success, False if the validation gate failed (so the caller
        can exit non-zero). A standalone --validate-db run returns its verdict.
    """
    mode_label = "EXECUTING" if execute else "DRY RUN"
    print(f"\n{'=' * 60}")
    print(f"  Job Directory Cleanup ({mode_label})")
    print(f"{'=' * 60}")
    print(f"  Database:   {db_path}")
    print(f"  Root dir:   {root_dir}")
    if categories:
        print(f"  Categories: {', '.join(sorted(categories))}")
    if include_failed:
        print("  Include failed/timeout/to_run: yes")
    if purge_failed:
        print("  Purge failed: yes")
    if purge_incomplete:
        print("  Purge incomplete (running/to_run/timeout): yes")
        print(
            "  NOTE: FINAL cleanup only -- not for an active campaign. This "
            "deletes dirs the DB still calls running/to_run/timeout; only run "
            "when the campaign is finished and nothing is executing."
        )
        print(
            "  Tip: run 'dashboard --update <root> --recheck-completed --unzip' "
            "first so completed jobs are reconciled out of the purge set."
        )
    print(f"  Workers:    {workers}")
    print()

    root_dir = root_dir.resolve()

    with ArchitectorWorkflow(db_path) as wf:
        # --- Phase 0: DB <-> folder validation gate ---
        run_validation = validate_db or (purge_incomplete and not skip_validation)
        if run_validation:
            validation = validate_db_folder_alignment(wf, root_dir, workers=workers)
            _print_validation_report(validation)
            if not validation.passed:
                if purge_incomplete:
                    print(
                        "\nValidation FAILED. Refusing to purge. Re-run with "
                        "--skip-validation to override (only if you are certain "
                        "the DB matches these folders)."
                    )
                return False
            # Standalone audit: no destructive actions requested.
            if not categories and not purge_failed and not purge_incomplete:
                return validation.passed
        elif purge_incomplete and skip_validation:
            print(
                "\n" + "!" * 60 + "\n"
                "  WARNING: --skip-validation bypasses the DB<->folder gate.\n"
                "  The only safeguard against a wrong DB/root is the per-job\n"
                "  content check. Proceed only if you trust this mapping.\n" + "!" * 60
            )

        # --- Phase 1: Clean completed jobs ---
        total_matched = 0
        total_bytes = 0
        total_freed = 0
        total_errors: list[str] = []
        clean_job_count = 0

        if categories:
            if include_failed:
                statuses = [
                    JobStatus.COMPLETED,
                    JobStatus.FAILED,
                    JobStatus.TIMEOUT,
                    JobStatus.TO_RUN,
                ]
                clean_jobs = wf.get_jobs_by_status(
                    statuses,
                    limit=limit,
                    include_geometry=False,
                )
                print(
                    f"Found {len(clean_jobs)} jobs "
                    f"(completed + failed + timeout + to_run)"
                )
            else:
                clean_jobs = wf.get_jobs_by_status(
                    JobStatus.COMPLETED,
                    limit=limit,
                    include_geometry=False,
                )
                print(f"Found {len(clean_jobs)} completed jobs")

            # Build work items: (path, optimizer, skip_revalidation)
            work_items: list[tuple[Path, str | None, bool]] = []
            for job in clean_jobs:
                if job.job_dir is None:
                    total_errors.append(f"job id={job.id}: NULL job_dir, skipping")
                    continue
                resolved = _resolve_job_dir(job.job_dir, root_dir)
                if resolved is None:
                    total_errors.append(
                        f"job id={job.id}: path escapes root_dir ({job.job_dir}), skipping"
                    )
                    continue
                if not resolved.is_dir():
                    continue
                # Skip revalidation for non-completed jobs -- we already
                # know they are failed/timeout/to_run from the DB query.
                skip_reval = job.status != JobStatus.COMPLETED
                work_items.append((resolved, job.optimizer, skip_reval))

            print(f"Processing {len(work_items)} job directories...")

            # Parallel scan/delete
            futures = {}
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for idx, (job_dir, optimizer, skip_reval) in enumerate(work_items):
                    fut = pool.submit(
                        _process_job,
                        job_dir,
                        root_dir,
                        categories,
                        execute,
                        hours_cutoff,
                        optimizer,
                        skip_revalidation=skip_reval,
                    )
                    futures[fut] = idx

                completed_iter = as_completed(futures)
                if tqdm is not None:
                    completed_iter = tqdm(
                        completed_iter,
                        total=len(futures),
                        desc="Cleaning",
                        unit="job",
                    )
                for fut in completed_iter:
                    matched, freed, errs = fut.result()
                    if matched:
                        total_matched += len(matched)
                        total_bytes += sum(s for _, s, _ in matched)
                        total_freed += freed
                        clean_job_count += 1
                        if verbose:
                            for path, size, is_d in matched:
                                dtype = "DIR " if is_d else "FILE"
                                print(
                                    f"  {dtype} {path.relative_to(root_dir)} "
                                    f"({_format_size(size)})"
                                )
                    total_errors.extend(errs)

            print("\n--- Completed Jobs Cleanup ---")
            print(f"  Jobs with matches:   {clean_job_count}")
            print(f"  Files/dirs matched:  {total_matched}")
            action = "Freed" if execute else "Would free"
            print(f"  {action}:  {_format_size(total_bytes)}")

        # --- Phase 2: Purge failed jobs ---
        purge_matched = 0
        purge_bytes = 0
        purge_freed = 0
        purge_job_count = 0

        if purge_failed:
            failed_jobs = wf.get_jobs_by_status(
                JobStatus.FAILED,
                limit=limit,
                include_geometry=False,
            )
            print(f"\nFound {len(failed_jobs)} failed jobs")

            purge_items: list[tuple[Path, int, dict[str, str | int | None]]] = []
            for job in failed_jobs:
                if job.job_dir is None:
                    total_errors.append(
                        f"job id={job.id}: NULL job_dir, skipping purge"
                    )
                    continue
                resolved = _resolve_job_dir(job.job_dir, root_dir)
                if resolved is None:
                    total_errors.append(
                        f"job id={job.id}: path escapes root_dir ({job.job_dir}), "
                        f"skipping purge"
                    )
                    continue
                if not resolved.is_dir():
                    continue
                metadata: dict[str, str | int | None] = {
                    "orig_index": job.orig_index,
                    "elements": job.elements,
                    "charge": job.charge,
                    "spin": job.spin,
                    "fail_count": job.fail_count,
                    "error_message": job.error_message,
                }
                purge_items.append((resolved, job.id, metadata))

            print(f"Processing {len(purge_items)} failed job directories...")

            with ThreadPoolExecutor(max_workers=workers) as pool:
                purge_futures = {}
                for job_dir, job_id, meta in purge_items:
                    fut = pool.submit(
                        _purge_failed_job,
                        job_dir,
                        root_dir,
                        db_path,
                        job_id,
                        execute,
                        meta,
                    )
                    purge_futures[fut] = job_dir

                for fut in as_completed(purge_futures):
                    matched, freed, errs = fut.result()
                    if matched:
                        purge_matched += len(matched)
                        purge_bytes += sum(s for _, s, _ in matched)
                        purge_freed += freed
                        purge_job_count += 1
                        if verbose:
                            for path, size, is_d in matched:
                                dtype = "DIR " if is_d else "FILE"
                                print(
                                    f"  {dtype} {path.relative_to(root_dir)} "
                                    f"({_format_size(size)})"
                                )
                    total_errors.extend(errs)

            print("\n--- Failed Jobs Purge ---")
            print(f"  Jobs purged:         {purge_job_count}")
            print(f"  Files/dirs removed:  {purge_matched}")
            action = "Freed" if execute else "Would free"
            print(f"  {action}:  {_format_size(purge_bytes)}")

        # --- Phase 3: Purge incomplete jobs (running / to_run / timeout) ---
        inc_matched = 0
        inc_bytes = 0
        inc_freed = 0
        inc_counts = {"purged": 0, "protected": 0, "looks_failed": 0, "skipped": 0}

        if purge_incomplete:
            incomplete_jobs = wf.get_jobs_by_status(
                [JobStatus.RUNNING, JobStatus.TO_RUN, JobStatus.TIMEOUT],
                limit=limit,
                include_geometry=False,
            )
            print(
                f"\nFound {len(incomplete_jobs)} incomplete jobs "
                f"(running + to_run + timeout)"
            )

            inc_items: list[
                tuple[Path, int, str | None, dict[str, str | int | None]]
            ] = []
            for job in incomplete_jobs:
                if job.job_dir is None:
                    total_errors.append(
                        f"job id={job.id}: NULL job_dir, skipping purge"
                    )
                    continue
                resolved = _resolve_job_dir(job.job_dir, root_dir)
                if resolved is None:
                    total_errors.append(
                        f"job id={job.id}: path escapes root_dir ({job.job_dir}), "
                        f"skipping purge"
                    )
                    continue
                if not resolved.is_dir():
                    continue
                inc_meta: dict[str, str | int | None] = {
                    "orig_index": job.orig_index,
                    "elements": job.elements,
                    "charge": job.charge,
                    "spin": job.spin,
                }
                inc_items.append((resolved, job.id, job.optimizer, inc_meta))

            print(f"Processing {len(inc_items)} incomplete job directories...")

            with ThreadPoolExecutor(max_workers=workers) as pool:
                inc_futures = {}
                for job_dir, job_id, optimizer, meta in inc_items:
                    inc_fut = pool.submit(
                        _purge_incomplete_job,
                        job_dir,
                        root_dir,
                        db_path,
                        job_id,
                        execute,
                        hours_cutoff,
                        optimizer,
                        meta,
                    )
                    inc_futures[inc_fut] = job_dir

                inc_iter = as_completed(inc_futures)
                if tqdm is not None:
                    inc_iter = tqdm(
                        inc_iter,
                        total=len(inc_futures),
                        desc="Purging incomplete",
                        unit="job",
                    )
                for inc_fut in inc_iter:
                    classification, matched, freed, errs = inc_fut.result()
                    inc_counts[classification] = inc_counts.get(classification, 0) + 1
                    if classification == "purged" and matched:
                        inc_matched += len(matched)
                        inc_bytes += sum(s for _, s, _ in matched)
                        inc_freed += freed
                        if verbose:
                            for path, size, is_d in matched:
                                dtype = "DIR " if is_d else "FILE"
                                print(
                                    f"  {dtype} {path.relative_to(root_dir)} "
                                    f"({_format_size(size)})"
                                )
                    total_errors.extend(errs)

            print("\n--- Incomplete Jobs Purge ---")
            print(f"  Protected (corpus, on-disk completed): {inc_counts['protected']}")
            print(
                f"  Looks-failed (run --purge-failed):     "
                f"{inc_counts['looks_failed']}"
            )
            print(f"  Skipped (TOCTOU/missing/unreadable):   {inc_counts['skipped']}")
            print(f"  Purged:                                {inc_counts['purged']}")
            print(f"  Files/dirs removed:                    {inc_matched}")
            action = "Freed" if execute else "Would free"
            print(f"  {action}:  {_format_size(inc_bytes)}")

        # --- Summary ---
        grand_bytes = total_bytes + purge_bytes + inc_bytes
        grand_files = total_matched + purge_matched + inc_matched
        if grand_files > 0:
            print(f"\n{'=' * 60}")
            action = "Total freed" if execute else "Total would free"
            print(
                f"  {action}: {_format_size(grand_bytes)} across {grand_files} entries"
            )
            print(f"{'=' * 60}")

        if total_errors:
            print(f"\nWarnings ({len(total_errors)}):")
            for err in total_errors[:20]:
                print(f"  - {err}")
            if len(total_errors) > 20:
                print(f"  ... and {len(total_errors) - 20} more")

        if not execute and grand_files > 0:
            print("\nThis was a dry run. Use --execute to actually delete files.")

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for job directory cleanup."""
    parser = argparse.ArgumentParser(
        prog="python -m oact_utilities.workflows.clean",
        description="Clean superfluous files from ORCA job directories.",
    )
    parser.add_argument("db_path", type=Path, help="Path to workflow SQLite database")
    parser.add_argument(
        "root_dir", type=Path, help="Root directory containing job subdirectories"
    )

    # Action flags
    action = parser.add_argument_group("action flags (at least one required)")
    action.add_argument(
        "--clean-tmp",
        action="store_true",
        help="Remove scratch/temp files (.tmp, .core, orca_tmp_*/)",
    )
    action.add_argument(
        "--clean-bas",
        action="store_true",
        help="Remove basis set files (.bas, .bas[N])",
    )
    action.add_argument(
        "--clean-all",
        action="store_true",
        help="Remove all scratch categories (equivalent to --clean-tmp --clean-bas)",
    )
    action.add_argument(
        "--purge-failed",
        action="store_true",
        help="Purge failed job directories (write .do_not_rerun.json marker, delete contents)",
    )
    action.add_argument(
        "--include-failed",
        action="store_true",
        help="Also clean scratch files from failed, timeout, and to_run jobs (not just completed)",
    )
    action.add_argument(
        "--purge-incomplete",
        action="store_true",
        help="FINAL cleanup only (not for an active campaign): full-purge "
        "running/to_run/timeout dirs confirmed incomplete by content; writes "
        ".do_not_rerun.json marker (runs --validate-db first)",
    )
    action.add_argument(
        "--validate-db",
        action="store_true",
        help="Run the DB<->folder alignment sanity check (elements + atom count)",
    )

    # Execution control
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete files (default: dry-run preview only)",
    )
    parser.add_argument(
        "--skip-validation",
        "--force",
        dest="skip_validation",
        action="store_true",
        help="Bypass the --purge-incomplete validation gate (prints a loud warning)",
    )

    # Performance
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel workers for scanning (default: 4)",
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=None,
        metavar="N",
        help="Limit to first N jobs for testing",
    )

    # Output
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show per-file listing instead of per-job summary",
    )

    # Revalidation
    parser.add_argument(
        "--hours-cutoff",
        type=int,
        default=24,
        help="Hours before revalidation considers a job timed out (default: 24)",
    )

    args = parser.parse_args()

    # Validate: at least one action flag
    has_category = args.clean_tmp or args.clean_bas or args.clean_all
    if not (
        has_category or args.purge_failed or args.purge_incomplete or args.validate_db
    ):
        parser.error(
            "At least one action flag is required: "
            "--clean-tmp, --clean-bas, --clean-all, --purge-failed, "
            "--purge-incomplete, or --validate-db"
        )

    # Build categories set
    categories: set[str] = set()
    if args.clean_tmp or args.clean_all:
        categories.add("tmp")
    if args.clean_bas or args.clean_all:
        categories.add("bas")

    # Validate paths
    if not args.db_path.exists():
        print(f"Error: database not found: {args.db_path}", file=sys.stderr)
        sys.exit(1)
    if not args.root_dir.is_dir():
        print(f"Error: root directory not found: {args.root_dir}", file=sys.stderr)
        sys.exit(1)

    ok = clean_job_directories(
        db_path=args.db_path,
        root_dir=args.root_dir,
        categories=categories,
        purge_failed=args.purge_failed,
        execute=args.execute,
        workers=args.workers,
        hours_cutoff=args.hours_cutoff,
        limit=args.debug,
        verbose=args.verbose,
        include_failed=args.include_failed,
        purge_incomplete=args.purge_incomplete,
        validate_db=args.validate_db,
        skip_validation=args.skip_validation,
    )
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
