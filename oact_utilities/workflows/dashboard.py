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
    """Print summary of computational metrics (forces, SCF steps)."""
    import pandas as pd

    # Query completed jobs with metrics
    cur = workflow.conn.cursor()
    cur.execute(
        """
        SELECT max_forces, scf_steps
        FROM structures
        WHERE status = 'completed' AND max_forces IS NOT NULL
        """
    )
    rows = cur.fetchall()

    if not rows:
        print("\nNo metrics data available yet.")
        return

    df = pd.DataFrame(rows, columns=["max_forces", "scf_steps"])

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

    print(f"\nTotal jobs with metrics: {len(df)}\n")


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
        [JobStatus.RUNNING, JobStatus.READY, JobStatus.FAILED]
    )

    if verbose:
        print(f"\nScanning {len(jobs)} jobs for status updates...")

    updated_counts = {
        JobStatus.COMPLETED: 0,
        JobStatus.FAILED: 0,
        JobStatus.RUNNING: 0,
    }

    for i, job in enumerate(jobs):
        # Format job directory name
        job_dir_name = job_dir_pattern.format(
            orig_index=job.orig_index,
            id=job.id,
            index_in_chunk=job.index_in_chunk,
        )
        job_dir = root_dir / job_dir_name

        if not job_dir.exists():
            # Job directory doesn't exist yet, keep as ready
            continue

        # Check status
        result = check_func(str(job_dir))

        if result == 1:
            new_status = JobStatus.COMPLETED
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
        f"{updated_counts[JobStatus.RUNNING]} running"
    )


def show_failed_jobs(workflow: ArchitectorWorkflow, limit: int = 20):
    """Display details of failed jobs."""
    failed = workflow.get_jobs_by_status(JobStatus.FAILED)

    if not failed:
        print("\nNo failed jobs found.")
        return

    print_header(f"Failed Jobs (showing up to {limit})")
    print(f"\n{'ID':<8} {'Orig Index':<12} {'Job Dir':<30} {'Error':<30}")
    print("-" * 90)

    for job in failed[:limit]:
        job_dir = (
            (job.job_dir[:27] + "...")
            if job.job_dir and len(job.job_dir) > 30
            else (job.job_dir or "N/A")
        )
        error = (
            (job.error_message[:27] + "...")
            if job.error_message and len(job.error_message) > 30
            else (job.error_message or "N/A")
        )
        print(f"{job.id:<8} {job.orig_index:<12} {job_dir:<30} {error:<30}")

    if len(failed) > limit:
        print(f"\n... and {len(failed) - limit} more failed jobs")


def show_ready_jobs(workflow: ArchitectorWorkflow, limit: int = 20):
    """Display jobs that are ready to run."""
    ready = workflow.get_jobs_by_status(JobStatus.READY)

    if not ready:
        print("\nNo jobs ready to run.")
        return

    print_header(f"Ready Jobs (showing up to {limit})")
    print(f"\n{'ID':<8} {'Orig Index':<12} {'Chunk File':<20} {'N Atoms':<10}")
    print("-" * 60)

    for job in ready[:limit]:
        print(f"{job.id:<8} {job.orig_index:<12} {job.chunk_file:<20} {job.natoms:<10}")

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
        failed_count = len(workflow.get_jobs_by_status(JobStatus.FAILED))
        workflow.reset_failed_jobs()
        print(f"\nReset {failed_count} failed jobs to ready status")

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

    # Show ready jobs if requested
    if args.show_ready:
        show_ready_jobs(workflow, limit=args.limit)

    # Show metrics if requested
    if args.show_metrics:
        print_metrics_summary(workflow)

    workflow.close()


if __name__ == "__main__":
    main()
