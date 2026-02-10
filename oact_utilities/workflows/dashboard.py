"""Dashboard for monitoring architector workflow status.

This script provides a command-line dashboard to monitor the status of
high-throughput architector calculations. It can be run on HPC systems
to check job progress.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

from .architector_workflow import ArchitectorWorkflow, JobStatus


def print_header(text: str, width: int = 80):
    """Print a formatted header."""
    print("\n" + "=" * width)
    print(text.center(width))
    print("=" * width)


def print_summary(workflow: ArchitectorWorkflow):
    """Print summary statistics table."""
    print_header("Workflow Status Summary")

    counts = workflow.count_by_status()
    total = sum(counts.values())

    # Create formatted table
    print(f"\n{'Status':<15} {'Count':>10} {'Percent':>10}")
    print("-" * 40)

    for status in JobStatus:
        count = counts.get(status, 0)
        pct = 100 * count / total if total > 0 else 0
        print(f"{status.value:<15} {count:>10} {pct:>9.1f}%")

    print("-" * 40)
    print(f"{'TOTAL':<15} {total:>10} {100.0:>9.1f}%\n")


def print_metrics_summary(workflow: ArchitectorWorkflow):
    """Print summary of computational metrics (forces, SCF steps, timing)."""
    import pandas as pd

    # Query completed jobs with metrics
    cur = workflow._execute_with_retry(
        """
        SELECT max_forces, scf_steps, wall_time, n_cores, final_energy
        FROM structures
        WHERE status = 'completed' AND max_forces IS NOT NULL
        """
    )
    rows = cur.fetchall()

    if not rows:
        print("\nNo metrics data available yet.")
        return

    df = pd.DataFrame(
        rows,
        columns=["max_forces", "scf_steps", "wall_time", "n_cores", "final_energy"],
    )

    print_header("Computational Metrics (Completed Jobs)")

    # Energy statistics
    energy_data = df["final_energy"].dropna()
    if len(energy_data) > 0:
        print("\nFinal Energy (Eh):")
        print(f"  Mean:   {energy_data.mean():.6f}")
        print(f"  Median: {energy_data.median():.6f}")
        print(f"  Min:    {energy_data.min():.6f}")
        print(f"  Max:    {energy_data.max():.6f}")

    # Forces statistics
    print("\nMax Forces (Eh/Bohr):")
    print(f"  Mean:   {df['max_forces'].mean():.6f}")
    print(f"  Median: {df['max_forces'].median():.6f}")
    print(f"  Min:    {df['max_forces'].min():.6f}")
    print(f"  Max:    {df['max_forces'].max():.6f}")

    # SCF steps statistics
    scf_data = df["scf_steps"].dropna()
    if len(scf_data) > 0:
        print("\nSCF Steps:")
        print(f"  Mean:   {scf_data.mean():.1f}")
        print(f"  Median: {scf_data.median():.0f}")
        print(f"  Min:    {scf_data.min():.0f}")
        print(f"  Max:    {scf_data.max():.0f}")

    # Timing statistics
    time_data = df["wall_time"].dropna()
    if len(time_data) > 0:
        print("\nWall Time (seconds):")
        print(f"  Mean:   {time_data.mean():.1f}")
        print(f"  Median: {time_data.median():.1f}")
        print(f"  Min:    {time_data.min():.1f}")
        print(f"  Max:    {time_data.max():.1f}")
        print(f"  Total:  {time_data.sum():.1f} ({time_data.sum() / 3600:.2f} hours)")

    # Cores statistics
    cores_data = df["n_cores"].dropna()
    if len(cores_data) > 0:
        print("\nCores Used:")
        print(f"  Mean:   {cores_data.mean():.1f}")
        print(f"  Min:    {cores_data.min():.0f}")
        print(f"  Max:    {cores_data.max():.0f}")

        # Calculate total core-hours
        if len(time_data) > 0:
            # Only compute for jobs that have both timing and cores data
            valid_rows = df.dropna(subset=["wall_time", "n_cores"])
            if len(valid_rows) > 0:
                core_seconds = (valid_rows["wall_time"] * valid_rows["n_cores"]).sum()
                print(f"  Total core-hours: {core_seconds / 3600:.2f}")

    print(f"\nTotal jobs with metrics: {len(df)}")
    if len(time_data) > 0:
        print(f"Jobs with timing data: {len(time_data)}\n")
    else:
        print()


def print_progress_bar(
    completed: int, total: int, width: int = 50, label: str = "Progress"
):
    """Print a text-based progress bar."""
    if total == 0:
        fraction = 0.0
    else:
        fraction = completed / total

    filled = int(width * fraction)
    bar = "█" * filled + "░" * (width - filled)
    pct = 100 * fraction
    print(f"{label}: [{bar}] {pct:.1f}% ({completed}/{total})")


def _extract_metrics_from_dir(
    job_dir: Path,
    unzip: bool = False,
) -> dict:
    """Extract metrics from a job directory (pure I/O, no DB writes).

    This function is safe to call from worker threads.

    Args:
        job_dir: Path to job directory containing ORCA output.
        unzip: If True, handle gzipped output files (quacc).

    Returns:
        Dictionary with extracted metrics and an 'error' key (None on success).
    """
    from ..utils.analysis import find_timings_and_cores, parse_job_metrics
    from ..utils.status import pull_log_file

    try:
        metrics = parse_job_metrics(job_dir, unzip=unzip)

        wall_time = None
        n_cores = None
        timing_warning = None
        try:
            log_file = pull_log_file(str(job_dir))
            n_cores_val, time_dict = find_timings_and_cores(log_file)
            if time_dict and "Total" in time_dict:
                wall_time = time_dict["Total"]
            n_cores = n_cores_val
        except Exception as e:
            timing_warning = f"Timing extraction failed: {e}"

        return {
            "max_forces": metrics.get("max_forces"),
            "scf_steps": metrics.get("scf_steps"),
            "final_energy": metrics.get("final_energy"),
            "wall_time": wall_time,
            "n_cores": n_cores,
            "error": None,
            "timing_warning": timing_warning,
        }

    except Exception as e:
        return {"error": str(e)}


def _parallel_extract_metrics(
    workflow: ArchitectorWorkflow,
    work_items: list[tuple[int, Path]],
    unzip: bool = False,
    verbose: bool = False,
    workers: int = 4,
    mark_failed_on_error: bool = True,
) -> tuple[int, int]:
    """Extract metrics in parallel, then batch-write results to DB.

    File I/O happens in a thread pool; DB writes happen in a single
    transaction after all extractions complete.

    Args:
        workflow: ArchitectorWorkflow instance.
        work_items: List of (job_id, job_dir) tuples.
        unzip: If True, handle gzipped output files (quacc).
        verbose: Print detailed progress messages.
        workers: Number of parallel worker threads.
        mark_failed_on_error: If True, mark jobs as FAILED when parsing fails.

    Returns:
        Tuple of (extracted_count, failed_count).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not work_items:
        return 0, 0

    # Phase 1: extract metrics in parallel (pure I/O, no DB)
    success_metrics: list[dict] = []
    failed_jobs: list[tuple[int, str]] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_job = {
            executor.submit(_extract_metrics_from_dir, job_dir, unzip): (
                job_id,
                job_dir,
            )
            for job_id, job_dir in work_items
        }

        for future in as_completed(future_to_job):
            job_id, job_dir = future_to_job[future]
            result = future.result()

            if result.get("error") is None:
                success_metrics.append(
                    {
                        "job_id": job_id,
                        "job_dir": str(job_dir),
                        "max_forces": result["max_forces"],
                        "scf_steps": result["scf_steps"],
                        "final_energy": result["final_energy"],
                        "wall_time": result["wall_time"],
                        "n_cores": result["n_cores"],
                    }
                )
                if verbose:
                    print(
                        f"    Job {job_id}: forces={result['max_forces']}, "
                        f"scf={result['scf_steps']}, "
                        f"energy={result['final_energy']}"
                    )
                    if result.get("timing_warning"):
                        print(f"      Warning: {result['timing_warning']}")
            else:
                failed_jobs.append((job_id, result["error"]))
                if verbose:
                    print(
                        f"    Job {job_id}: failed to extract metrics: "
                        f"{result['error']}"
                    )

    # Phase 2: batch-write all successful metrics in one transaction
    if success_metrics:
        workflow.update_job_metrics_bulk(success_metrics)

    # Phase 3: mark failed jobs
    if mark_failed_on_error:
        for job_id, error_msg in failed_jobs:
            workflow.update_status(
                job_id,
                JobStatus.FAILED,
                error_message=f"Metrics parse error: {error_msg}",
                increment_fail_count=True,
            )

    return len(success_metrics), len(failed_jobs)


def backfill_metrics(
    workflow: ArchitectorWorkflow,
    root_dir: str | Path,
    job_dir_pattern: str = "job_{orig_index}",
    unzip: bool = False,
    verbose: bool = False,
    workers: int = 4,
):
    """Extract metrics for completed jobs that don't have them yet.

    Args:
        workflow: ArchitectorWorkflow instance.
        root_dir: Root directory containing job subdirectories.
        job_dir_pattern: Pattern for job directory names.
        unzip: If True, handle gzipped output files (quacc).
        verbose: Print detailed progress messages.
        workers: Number of parallel worker threads for extraction.
    """
    root_dir = Path(root_dir)

    cur = workflow._execute_with_retry(
        """
        SELECT id, orig_index
        FROM structures
        WHERE status = 'completed' AND max_forces IS NULL
        """
    )
    rows = cur.fetchall()

    if not rows:
        print("\nAll completed jobs already have metrics.")
        return

    # Build work items, filtering out missing directories
    work_items = []
    skipped = 0
    for job_id, orig_index in rows:
        job_dir_name = job_dir_pattern.format(orig_index=orig_index, id=job_id)
        job_dir = root_dir / job_dir_name

        if not job_dir.exists():
            skipped += 1
            if verbose:
                print(f"  Job {job_id}: directory {job_dir} not found, skipping")
            continue

        work_items.append((job_id, job_dir))

    print(
        f"\nBackfilling metrics for {len(work_items)} completed jobs "
        f"({workers} workers)..."
    )

    extracted, failed = _parallel_extract_metrics(
        workflow, work_items, unzip=unzip, verbose=verbose, workers=workers
    )

    print(
        f"Backfill metrics: {extracted} extracted, "
        f"{failed} failed, {skipped} skipped (no directory)"
    )


def update_all_statuses(
    workflow: ArchitectorWorkflow,
    root_dir: str | Path,
    job_dir_pattern: str = "job_{orig_index}",
    check_func: Callable | None = None,
    verbose: bool = False,
    extract_metrics: bool = False,
    unzip: bool = False,
    workers: int = 4,
    recheck_completed: bool = False,
):
    """Scan job directories and update statuses in bulk.

    Args:
        workflow: ArchitectorWorkflow instance.
        root_dir: Root directory containing job subdirectories.
        job_dir_pattern: Pattern for job directory names. Use {orig_index} or {id}.
        check_func: Optional custom status checking function.
        verbose: Print detailed progress messages.
        extract_metrics: If True, extract computational metrics for newly completed jobs.
        unzip: If True, handle gzipped output files (quacc).
        workers: Number of parallel worker threads for metrics extraction.
        recheck_completed: If True, also re-verify jobs marked as completed.
    """
    from ..utils.status import check_job_termination

    if check_func is None:
        check_func = check_job_termination

    root_dir = Path(root_dir)
    if not root_dir.exists():
        print(f"Error: root directory {root_dir} does not exist")
        return

    # Get jobs to check — optionally include completed for re-verification
    statuses_to_check = [
        JobStatus.RUNNING,
        JobStatus.READY,
        JobStatus.FAILED,
        JobStatus.TO_RUN,
    ]
    if recheck_completed:
        statuses_to_check.append(JobStatus.COMPLETED)

    jobs = workflow.get_jobs_by_status(statuses_to_check)

    if verbose:
        print(f"\nScanning {len(jobs)} jobs for status updates...")

    updated_counts = {
        JobStatus.COMPLETED: 0,
        JobStatus.FAILED: 0,
        JobStatus.TIMEOUT: 0,
        JobStatus.RUNNING: 0,
    }
    completed_for_metrics: list[tuple[int, Path]] = []

    for i, job in enumerate(jobs):
        # Format job directory name
        job_dir_name = job_dir_pattern.format(
            orig_index=job.orig_index,
            id=job.id,
        )
        job_dir = root_dir / job_dir_name

        if not job_dir.exists():
            # Job directory doesn't exist yet, keep as ready
            continue

        # Check status
        result = check_func(str(job_dir))

        if result == 1:
            new_status = JobStatus.COMPLETED
        elif result == -2:
            new_status = JobStatus.TIMEOUT
        elif result == -1:
            new_status = JobStatus.FAILED
        else:
            new_status = JobStatus.RUNNING

        # Update if changed
        if new_status != job.status:
            workflow.update_status(job.id, new_status)
            updated_counts[new_status] += 1

            if verbose:
                print(
                    f"  [{i+1}/{len(jobs)}] Job {job.id}: {job.status.value} -> {new_status.value}"
                )

            # Collect newly completed jobs for parallel metrics extraction
            if extract_metrics and new_status == JobStatus.COMPLETED:
                completed_for_metrics.append((job.id, job_dir))

    # Print update summary
    print(
        f"\nStatus updates: {updated_counts[JobStatus.COMPLETED]} completed, "
        f"{updated_counts[JobStatus.FAILED]} failed, "
        f"{updated_counts[JobStatus.TIMEOUT]} timeout, "
        f"{updated_counts[JobStatus.RUNNING]} running"
    )

    # Extract metrics in parallel for newly completed jobs
    if completed_for_metrics:
        print(
            f"Extracting metrics for {len(completed_for_metrics)} newly completed "
            f"jobs ({workers} workers)..."
        )
        metrics_extracted, metrics_failed = _parallel_extract_metrics(
            workflow,
            completed_for_metrics,
            unzip=unzip,
            verbose=verbose,
            workers=workers,
        )
        print(
            f"Metrics extraction: {metrics_extracted} succeeded, "
            f"{metrics_failed} failed"
        )


def show_failed_jobs(workflow: ArchitectorWorkflow, limit: int = 20):
    """Display details of failed jobs."""
    failed = workflow.get_jobs_by_status(JobStatus.FAILED)

    if not failed:
        print("\nNo failed jobs found.")
        return

    print_header(f"Failed Jobs (showing up to {limit})")
    print(f"\n{'ID':<8} {'Orig Index':<12} {'Fails':<6} {'Job Dir':<25} {'Error':<25}")
    print("-" * 90)

    for job in failed[:limit]:
        job_dir = (
            (job.job_dir[:22] + "...")
            if job.job_dir and len(job.job_dir) > 25
            else (job.job_dir or "N/A")
        )
        error = (
            (job.error_message[:22] + "...")
            if job.error_message and len(job.error_message) > 25
            else (job.error_message or "N/A")
        )
        print(
            f"{job.id:<8} {job.orig_index:<12} {job.fail_count:<6} {job_dir:<25} {error:<25}"
        )

    if len(failed) > limit:
        print(f"\n... and {len(failed) - limit} more failed jobs")


def show_timeout_jobs(workflow: ArchitectorWorkflow, limit: int = 20):
    """Display details of timed out jobs."""
    timeout = workflow.get_jobs_by_status(JobStatus.TIMEOUT)

    if not timeout:
        print("\nNo timeout jobs found.")
        return

    print_header(f"Timeout Jobs (showing up to {limit})")
    print(f"\n{'ID':<8} {'Orig Index':<12} {'Fails':<6} {'Job Dir':<25} {'Error':<25}")
    print("-" * 90)

    for job in timeout[:limit]:
        job_dir = (
            (job.job_dir[:22] + "...")
            if job.job_dir and len(job.job_dir) > 25
            else (job.job_dir or "N/A")
        )
        error = (
            (job.error_message[:22] + "...")
            if job.error_message and len(job.error_message) > 25
            else (job.error_message or "N/A")
        )
        print(
            f"{job.id:<8} {job.orig_index:<12} {job.fail_count:<6} {job_dir:<25} {error:<25}"
        )

    if len(timeout) > limit:
        print(f"\n... and {len(timeout) - limit} more timeout jobs")


def show_ready_jobs(workflow: ArchitectorWorkflow, limit: int = 20):
    """Display jobs that are ready to run."""
    ready = workflow.get_jobs_by_status(JobStatus.READY)

    if not ready:
        print("\nNo jobs ready to run.")
        return

    print_header(f"Ready Jobs (showing up to {limit})")
    print(f"\n{'ID':<8} {'Orig Index':<12} {'N Atoms':<10}")
    print("-" * 60)

    for job in ready[:limit]:
        print(f"{job.id:<8} {job.orig_index:<12} {job.natoms:<10}")

    if len(ready) > limit:
        print(f"\n... and {len(ready) - limit} more ready jobs")


def show_running_jobs(workflow: ArchitectorWorkflow, limit: int = 20):
    """Display jobs that are currently running."""
    running = workflow.get_jobs_by_status(JobStatus.RUNNING)

    if not running:
        print("\nNo jobs currently running.")
        return

    print_header(f"Running Jobs (showing up to {limit})")
    print(f"\n{'ID':<8} {'Orig Index':<12} {'N Atoms':<10}")
    print("-" * 60)

    for job in running[:limit]:
        print(f"{job.id:<8} {job.orig_index:<12} {job.natoms:<10}")

    if len(running) > limit:
        print(f"\n... and {len(running) - limit} more running jobs")


def main():
    """Main dashboard entry point."""
    parser = argparse.ArgumentParser(
        description="Dashboard for monitoring architector workflow status"
    )
    parser.add_argument("db_path", help="Path to workflow SQLite database")
    parser.add_argument(
        "--update",
        metavar="ROOT_DIR",
        help="Scan job directories and update statuses",
    )
    parser.add_argument(
        "--job-dir-pattern",
        default="job_{orig_index}",
        help="Pattern for job directory names (default: job_{orig_index})",
    )
    parser.add_argument(
        "--show-failed",
        action="store_true",
        help="Show details of failed jobs",
    )
    parser.add_argument(
        "--show-timeout",
        action="store_true",
        help="Show details of timeout jobs",
    )
    parser.add_argument(
        "--show-ready",
        action="store_true",
        help="Show jobs ready to run",
    )
    parser.add_argument(
        "--show-running",
        action="store_true",
        help="Show jobs currently running",
    )
    parser.add_argument(
        "--reset-failed",
        action="store_true",
        help="Reset all failed jobs to ready status",
    )
    parser.add_argument(
        "--reset-timeout",
        action="store_true",
        help="Reset all timeout jobs to ready status",
    )
    parser.add_argument(
        "--include-timeout-in-reset",
        action="store_true",
        help="When using --reset-failed, also reset timeout jobs",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="When resetting failed jobs, only reset those with fail_count < this value",
    )
    parser.add_argument(
        "--show-chronic-failures",
        type=int,
        metavar="N",
        help="Show jobs that have failed at least N times",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Limit number of jobs to display (default: 20)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output during updates",
    )
    parser.add_argument(
        "--show-metrics",
        action="store_true",
        help="Show computational metrics (forces, SCF steps)",
    )
    parser.add_argument(
        "--extract-metrics",
        action="store_true",
        help="Extract metrics (forces, SCF steps, energy, timing) for completed jobs during --update",
    )
    parser.add_argument(
        "--unzip",
        action="store_true",
        help="Handle gzipped output files (e.g., from quacc)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for metrics extraction (default: 4)",
    )
    parser.add_argument(
        "--recheck-completed",
        action="store_true",
        help="Re-verify completed jobs during --update (catches tampered outputs or status checker changes)",
    )

    args = parser.parse_args()

    # Open workflow database
    try:
        workflow = ArchitectorWorkflow(args.db_path)
    except FileNotFoundError:
        print(f"Error: Database not found at {args.db_path}")
        sys.exit(1)

    # Update statuses if requested
    if args.update:
        update_all_statuses(
            workflow,
            args.update,
            job_dir_pattern=args.job_dir_pattern,
            verbose=args.verbose,
            extract_metrics=args.extract_metrics,
            unzip=args.unzip,
            workers=args.workers,
            recheck_completed=args.recheck_completed,
        )

        # Backfill metrics for previously completed jobs missing them
        if args.extract_metrics:
            backfill_metrics(
                workflow,
                args.update,
                job_dir_pattern=args.job_dir_pattern,
                unzip=args.unzip,
                verbose=args.verbose,
                workers=args.workers,
            )

    # Reset failed jobs if requested
    if args.reset_failed:
        failed_jobs = workflow.get_jobs_by_status(JobStatus.FAILED)
        if args.max_retries is not None:
            eligible = [j for j in failed_jobs if j.fail_count < args.max_retries]
            skipped = len(failed_jobs) - len(eligible)
            workflow.reset_failed_jobs(
                max_fail_count=args.max_retries,
                include_timeout=args.include_timeout_in_reset,
            )
            status_msg = "failed/timeout" if args.include_timeout_in_reset else "failed"
            print(
                f"\nReset {len(eligible)} {status_msg} jobs to ready status "
                f"(skipped {skipped} jobs that have already failed {args.max_retries}+ times)"
            )
        else:
            workflow.reset_failed_jobs(include_timeout=args.include_timeout_in_reset)
            status_msg = "failed/timeout" if args.include_timeout_in_reset else "failed"
            print(f"\nReset {len(failed_jobs)} {status_msg} jobs to ready status")

    # Reset timeout jobs if requested
    if args.reset_timeout:
        timeout_jobs = workflow.get_jobs_by_status(JobStatus.TIMEOUT)
        if args.max_retries is not None:
            eligible = [j for j in timeout_jobs if j.fail_count < args.max_retries]
            skipped = len(timeout_jobs) - len(eligible)
            workflow.reset_timeout_jobs(max_fail_count=args.max_retries)
            print(
                f"\nReset {len(eligible)} timeout jobs to ready status "
                f"(skipped {skipped} jobs that have already timed out {args.max_retries}+ times)"
            )
        else:
            workflow.reset_timeout_jobs()
            print(f"\nReset {len(timeout_jobs)} timeout jobs to ready status")

    # Always show summary
    print_summary(workflow)

    # Show progress bar
    counts = workflow.count_by_status()
    total = sum(counts.values())
    completed = counts.get(JobStatus.COMPLETED, 0)
    print_progress_bar(completed, total, width=60, label="Completion")

    # Show failed jobs if requested
    if args.show_failed:
        show_failed_jobs(workflow, limit=args.limit)

    # Show timeout jobs if requested
    if args.show_timeout:
        show_timeout_jobs(workflow, limit=args.limit)

    if args.show_running:
        show_running_jobs(workflow, limit=args.limit)

    # Show ready jobs if requested
    if args.show_ready:
        show_ready_jobs(workflow, limit=args.limit)

    # Show metrics if requested
    if args.show_metrics:
        print_metrics_summary(workflow)

    # Show chronic failures if requested
    if args.show_chronic_failures:
        chronic = workflow.get_jobs_by_fail_count(args.show_chronic_failures)
        if chronic:
            print_header(f"Jobs Failed {args.show_chronic_failures}+ Times")
            print(
                f"\n{'ID':<8} {'Orig Index':<12} {'Fails':<6} {'Status':<12} {'Error':<30}"
            )
            print("-" * 80)
            for job in chronic[: args.limit]:
                error = (
                    (job.error_message[:27] + "...")
                    if job.error_message and len(job.error_message) > 30
                    else (job.error_message or "N/A")
                )
                print(
                    f"{job.id:<8} {job.orig_index:<12} {job.fail_count:<6} "
                    f"{job.status.value:<12} {error:<30}"
                )
            if len(chronic) > args.limit:
                print(f"\n... and {len(chronic) - args.limit} more chronic failures")
        else:
            print(f"\nNo jobs have failed {args.show_chronic_failures}+ times.")

    workflow.close()


if __name__ == "__main__":
    main()
