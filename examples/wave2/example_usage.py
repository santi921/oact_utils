"""Example usage of the wave2 workflow database.

This script demonstrates how to interact with the workflow database
created by create_wave2_workflow_db.py.
"""

from oact_utilities.workflows import ArchitectorWorkflow, JobStatus


def inspect_database(db_path: str):
    """Inspect the contents of a wave2 workflow database.

    Args:
        db_path: Path to the workflow database.
    """
    print("=" * 70)
    print("Wave2 Workflow Database Inspector")
    print("=" * 70)

    with ArchitectorWorkflow(db_path) as workflow:
        # Get summary statistics
        print("\nJob Status Summary:")
        print("-" * 70)
        summary = workflow.get_summary()
        print(summary.to_string(index=False))

        # Count by status
        print("\nDetailed Status Counts:")
        print("-" * 70)
        counts = workflow.count_by_status()
        for status, count in counts.items():
            print(f"  {status.value:15s}: {count:6d} jobs")

        # Get some example jobs
        print("\nExample Jobs (first 5 to_run):")
        print("-" * 70)
        jobs = workflow.get_jobs_by_status(JobStatus.TO_RUN)[:5]
        for job in jobs:
            print(
                f"  Job {job.id:5d}: {job.elements:30s} "
                f"(natoms={job.natoms:3d}, charge={job.charge}, spin={job.spin})"
            )
            if job.job_dir:
                print(f"             dir: {job.job_dir}")

        # Query by category using SQL (more efficient than filtering in Python)
        print("\nJobs by Category:")
        print("-" * 70)
        cur = workflow.conn.cursor()
        for category in ["Hard_Donors", "Organic", "Radical", "Soft_Donors"]:
            cur.execute(
                "SELECT COUNT(*) FROM structures WHERE category = ?", (category,)
            )
            count = cur.fetchone()[0]
            if count > 0:
                print(f"  {category:20s}: {count:6d} jobs")

        # Show unique ligand types per category
        print("\nUnique Ligand Types by Category:")
        print("-" * 70)
        for category in ["Hard_Donors", "Organic", "Radical", "Soft_Donors"]:
            cur.execute(
                "SELECT COUNT(DISTINCT ligand_type) FROM structures WHERE category = ?",
                (category,),
            )
            count = cur.fetchone()[0]
            if count > 0:
                print(f"  {category:20s}: {count:6d} unique types")

        # Show example ligand types for first category
        cur.execute(
            "SELECT DISTINCT ligand_type FROM structures WHERE category = ? LIMIT 5",
            ("Hard_Donors",),
        )
        ligand_types = [row[0] for row in cur.fetchall()]
        if ligand_types:
            print("\n  Example ligand types in Hard_Donors:")
            for lt in ligand_types:
                print(f"    - {lt}")

        # Show completed jobs with metrics (if any)
        completed = workflow.get_jobs_by_status(JobStatus.COMPLETED)
        if completed:
            print(f"\nCompleted Jobs with Metrics: {len(completed)}")
            print("-" * 70)
            for job in completed[:3]:
                print(f"  Job {job.id:5d}:")
                print(
                    f"    max_forces:   {job.max_forces:.6f} Eh/Bohr"
                    if job.max_forces
                    else "    max_forces:   N/A"
                )
                print(
                    f"    scf_steps:    {job.scf_steps}"
                    if job.scf_steps
                    else "    scf_steps:    N/A"
                )
                print(
                    f"    final_energy: {job.final_energy:.6f} Ha"
                    if job.final_energy
                    else "    final_energy: N/A"
                )

        # Show failed jobs (if any)
        failed = workflow.get_jobs_by_status(JobStatus.FAILED)
        if failed:
            print(f"\nFailed Jobs: {len(failed)}")
            print("-" * 70)
            for job in failed[:3]:
                print(f"  Job {job.id:5d}: {job.elements}")
                print(f"    fail_count: {job.fail_count}")
                if job.error_message:
                    print(f"    error: {job.error_message[:80]}...")

    print("\n" + "=" * 70)


def example_submission_workflow(db_path: str):
    """Example workflow for submitting and monitoring jobs.

    This demonstrates the typical workflow after creating the database.

    Args:
        db_path: Path to the workflow database.
    """
    print("\n" + "=" * 70)
    print("Example Submission Workflow")
    print("=" * 70)

    print(
        """
# Step 1: Create the database (already done via create_wave2_workflow_db.py)
python create_wave2_workflow_db.py \\
    --root-data-dir /path/to/wave2/data \\
    --db-path workflow.db \\
    --calc-root-dir /path/to/output

# Step 2: Submit first batch of jobs
python -m oact_utilities.workflows.submit_jobs \\
    workflow.db jobs/ \\
    --batch-size 200 \\
    --scheduler flux \\
    --n-cores 16 \\
    --n-hours 4 \\
    --queue pbatch \\
    --allocation dnn-sim \\
    --functional wB97M-V \\
    --simple-input omol \\
    --scf-maxiter 600

# Step 3: Monitor progress
python -m oact_utilities.workflows.dashboard workflow.db

# Step 4: Update job statuses by scanning output directories
python -m oact_utilities.workflows.dashboard workflow.db --update jobs/

# Step 5: View metrics for completed jobs
python -m oact_utilities.workflows.dashboard workflow.db --show-metrics

# Step 6: Reset failed jobs for retry (up to 3 attempts)
python -m oact_utilities.workflows.dashboard workflow.db --reset-failed --max-retries 3

# Step 7: Submit next batch, skipping chronic failures
python -m oact_utilities.workflows.submit_jobs \\
    workflow.db jobs/ \\
    --batch-size 200 \\
    --max-fail-count 3

# Alternative: Use Parsl mode for concurrent execution
# First, get an exclusive node allocation:
flux alloc -N 1 -n 64 -q pbatch -t 8h -B dnn-sim

# Then run Parsl-based submission inside the allocation:
python -m oact_utilities.workflows.submit_jobs \\
    workflow.db jobs/ \\
    --use-parsl \\
    --batch-size 200 \\
    --max-workers 4 \\
    --cores-per-worker 16 \\
    --n-cores 16 \\
    --job-timeout 7200 \\
    --functional wB97M-V \\
    --simple-input omol
    """
    )


def example_programmatic_usage(db_path: str):
    """Example programmatic usage of the workflow API.

    Args:
        db_path: Path to the workflow database.
    """
    print("\n" + "=" * 70)
    print("Example Programmatic API Usage")
    print("=" * 70)

    with ArchitectorWorkflow(db_path) as workflow:
        # Get jobs ready to run
        ready_jobs = workflow.get_jobs_by_status(JobStatus.TO_RUN)
        print(f"\nJobs ready to submit: {len(ready_jobs)}")

        # Get a specific job
        if ready_jobs:
            job = ready_jobs[0]
            print("\nExample job details:")
            print(f"  ID: {job.id}")
            print(f"  Elements: {job.elements}")
            print(f"  Atoms: {job.natoms}")
            print(f"  Charge: {job.charge}")
            print(f"  Spin: {job.spin}")
            print(f"  Status: {job.status}")
            if job.job_dir:
                print(f"  Output dir: {job.job_dir}")

        # Count jobs by status
        counts = workflow.count_by_status()
        print("\nStatus counts:")
        for status, count in counts.items():
            print(f"  {status.value}: {count}")

        # Example: Query jobs by category and ligand type
        print("\nQuerying by category and ligand type:")
        cur = workflow.conn.cursor()

        # Get all Hard_Donors jobs that are ready to run
        cur.execute(
            "SELECT COUNT(*) FROM structures WHERE category = ? AND status = ?",
            ("Hard_Donors", JobStatus.TO_RUN.value),
        )
        count = cur.fetchone()[0]
        print(f"  Hard_Donors ready to run: {count} jobs")

        # Get all jobs for a specific ligand type
        cur.execute("SELECT DISTINCT ligand_type FROM structures LIMIT 1")
        example_ligand = cur.fetchone()
        if example_ligand:
            ligand_type = example_ligand[0]
            cur.execute(
                "SELECT COUNT(*) FROM structures WHERE ligand_type = ?", (ligand_type,)
            )
            count = cur.fetchone()[0]
            print(f"  Jobs for ligand '{ligand_type}': {count}")

        # Example: Mark jobs as running (would happen during submission)
        # workflow.mark_jobs_as_running([job.id for job in ready_jobs[:10]])

        # Example: Update job metrics (would happen during status scanning)
        # workflow.update_job_metrics(
        #     job_id=42,
        #     max_forces=0.001234,
        #     scf_steps=15,
        #     final_energy=-1234.56,
        #     wall_time=120.5,
        #     n_cores=16
        # )

        # Example: Reset failed jobs for retry
        # workflow.reset_failed_jobs(max_fail_count=3)

        # Example: Query failed jobs in a specific category
        # cur.execute(
        #     "SELECT id, ligand_type, error_message FROM structures "
        #     "WHERE category = ? AND status = ? LIMIT 5",
        #     ("Hard_Donors", JobStatus.FAILED.value)
        # )
        # for row in cur.fetchall():
        #     print(f"  Failed job {row[0]}: ligand={row[1]}, error={row[2]}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python example_usage.py <db_path>")
        print("\nExample:")
        print("  python example_usage.py workflow.db")
        sys.exit(1)

    db_path = sys.argv[1]

    # Inspect database contents
    inspect_database(db_path)

    # Show example submission workflow
    example_submission_workflow(db_path)

    # Show programmatic API usage
    example_programmatic_usage(db_path)
