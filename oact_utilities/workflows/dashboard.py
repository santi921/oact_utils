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
        SELECT max_forces, scf_steps, wall_time, n_cores
        FROM structures
        WHERE status = 'completed' AND max_forces IS NOT NULL
        """
    )
    rows = cur.fetchall()

    if not rows:
        print("\nNo metrics data available yet.")
        return

    df = pd.DataFrame(rows, columns=["max_forces", "scf_steps", "wall_time", "n_cores"])

    print_header("Computational Metrics (Completed Jobs)")

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


def update_all_statuses(
    workflow: ArchitectorWorkflow,
    root_dir: str | Path,
    job_dir_pattern: str = "job_{orig_index}",
    check_func: Callable | None = None,
    verbose: bool = False,
):
    """Scan job directories and update statuses in bulk.

    Args:
        workflow: ArchitectorWorkflow instance.
        root_dir: Root directory containing job subdirectories.
        job_dir_pattern: Pattern for job directory names. Use {orig_index} or {id}.
        check_func: Optional custom status checking function.
        verbose: Print detailed progress messages.
    """
    from ..utils.status import check_job_termination

    if check_func is None:
        check_func = check_job_termination

    root_dir = Path(root_dir)
    if not root_dir.exists():
        print(f"Error: root directory {root_dir} does not exist")
        return

    # Get all running and ready jobs (these might have changed)
    jobs = workflow.get_jobs_by_status(
        [JobStatus.RUNNING, JobStatus.READY, JobStatus.FAILED, JobStatus.TO_RUN]
    )

    if verbose:
        print(f"\nScanning {len(jobs)} jobs for status updates...")

    updated_counts = {
        JobStatus.COMPLETED: 0,
        JobStatus.FAILED: 0,
        JobStatus.TIMEOUT: 0,
        JobStatus.RUNNING: 0,
    }

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

    # Print update summary
    print(
        f"\nStatus updates: {updated_counts[JobStatus.COMPLETED]} completed, "
        f"{updated_counts[JobStatus.FAILED]} failed, "
        f"{updated_counts[JobStatus.TIMEOUT]} timeout, "
        f"{updated_counts[JobStatus.RUNNING]} running"
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
