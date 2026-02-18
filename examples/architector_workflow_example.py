"""Example workflow for high-throughput architector calculations.

This script demonstrates how to:
1. Create a workflow from an architector CSV
2. Submit jobs to HPC
3. Monitor progress with the dashboard
4. Update job statuses
"""

from oact_utilities.workflows import ArchitectorWorkflow, JobStatus, create_workflow

# =============================================================================
# Step 1: Initialize workflow from architector CSV
# =============================================================================


def setup_workflow():
    """Create a new workflow from architector CSV file."""

    # /Users/santiagovargas/dev/oact_utils/data/homoleptics_oact/santi_results_compiled_2026-02-14_08-15-32_actinides_sample.csv
    # /Users/santiagovargas/dev/oact_utils/data/homoleptics_oact/santi_results_compiled_2026-02-14_08-15-32_actinides.csv
    # /Users/santiagovargas/dev/oact_utils/data/homoleptics_oact/santi_results_compiled_2026-02-14_08-15-32_non_actinides_sample.csv
    # /Users/santiagovargas/dev/oact_utils/data/homoleptics_oact/santi_results_compiled_2026-02-14_08-15-32_non_actinides.csv
    # /Users/santiagovargas/dev/oact_utils/data/homoleptics_oact/santi_results_compiled_2026-02-14_08-15-32.csv

    csv_path = "/Users/santiagovargas/dev/oact_utils/data/homoleptics_oact/santi_results_compiled_2026-02-14_08-15-32_actinides_sample.csv"  # Your architector CSV
    db_path = "./santi_results_compiled_2026-02-14_08-15-32_actinides_sample.db"  # SQLite database

    # Create workflow (chunks CSV and initializes database)
    db_path, workflow = create_workflow(
        csv_path=csv_path,
        db_path=db_path,
        geometry_column="structure",  # Column with XYZ geometries
        charge_column="charge",
        spin_column="spinmult",
        batch_size=10000,
        extra_columns={"metal": "TEXT"},  # Example extra columns to store
    )

    print("Workflow initialized!")
    print(f"  Database: {db_path}")

    # Show initial summary
    summary = workflow.get_summary()
    print("\nInitial status:")
    print(summary)
    # show row from db as an example
    example_job = workflow.get_jobs_by_status(JobStatus.READY)[0]
    print("\nExample job record:")
    print(example_job)

    workflow.close()
    return db_path


# =============================================================================
# Step 2: Submit jobs using the submit_jobs script
# =============================================================================


def submit_jobs_example():
    """
    Submit jobs using the command-line script.

    On HPC, run:

    python -m oact_utilities.workflows.submit_jobs \\
        architector_workflow.db \\
        /path/to/jobs \\
        --batch-size 100 \\
        --scheduler flux \\ # or slurm
        --n-cores 4 \\
        --n-hours 2 \\
        --queue pbatch \\
        --allocation dnn-sim

    This will:
    - Get 100 ready jobs from the database
    - Create job directories like job_0, job_1, etc.
    - Write input.xyz files
    - Generate flux_job.flux scripts
    - Submit to the scheduler
    - Mark jobs as "running" in the database
    """
    pass


# =============================================================================
# Step 3: Monitor progress with dashboard
# =============================================================================


def monitor_workflow():
    """
    Monitor workflow progress using the dashboard.

    On HPC, run:

    # Basic status summary
    python -m oact_utilities.workflows.dashboard architector_workflow.db

    # Update statuses by scanning job directories
    python -m oact_utilities.workflows.dashboard architector_workflow.db --update ./jobs --verbose

    # Show failed jobs
    python -m oact_utilities.workflows.dashboard \\
        architector_workflow.db \\
        --show-failed \\
        --limit 50

    # Show computational metrics (forces, SCF steps)
    python -m oact_utilities.workflows.dashboard architector_workflow.db --show-metrics

    # Reset failed jobs to retry
    python -m oact_utilities.workflows.dashboard \\
        architector_workflow.db \\
        --reset-failed
    """
    pass


# =============================================================================
# Step 4: Programmatic workflow management
# =============================================================================


def programmatic_workflow_management():
    """Use the workflow API directly in Python."""

    db_path = "architector_workflow.db"

    with ArchitectorWorkflow(db_path) as workflow:

        # Get jobs by status
        ready_jobs = workflow.get_jobs_by_status(JobStatus.READY)
        running_jobs = workflow.get_jobs_by_status(JobStatus.RUNNING)
        completed_jobs = workflow.get_jobs_by_status(JobStatus.COMPLETED)
        failed_jobs = workflow.get_jobs_by_status(JobStatus.FAILED)

        print(f"Ready: {len(ready_jobs)}")
        print(f"Running: {len(running_jobs)}")
        print(f"Completed: {len(completed_jobs)}")
        print(f"Failed: {len(failed_jobs)}")

        # Get summary statistics
        counts = workflow.count_by_status()
        print(f"\nStatus counts: {counts}")

        # Bulk status update
        job_ids = [j.id for j in ready_jobs[:10]]
        workflow.update_status_bulk(job_ids, JobStatus.RUNNING)

        # Reset failed jobs
        workflow.reset_failed_jobs()


# =============================================================================
# Step 5: Custom job setup
# =============================================================================


def submit_with_custom_setup():
    """Submit jobs with custom input file generation."""
    from oact_utilities.workflows.submit_jobs import submit_batch

    def setup_orca_input(job_dir, job_record):
        """Custom function to set up ORCA input files."""
        # Write ORCA input file
        orca_inp = job_dir / "calc.inp"
        with open(orca_inp, "w") as f:
            f.write("! B3LYP def2-SVP OPT\n")
            f.write(
                f"* xyzfile {job_record.charge or 0} {job_record.spin or 1} input.xyz\n"
            )

        # Write Python runner script
        orca_py = job_dir / "orca.py"
        with open(orca_py, "w") as f:
            f.write("from ase.io import read\n")
            f.write("from oact_utilities.core.orca import ase_relaxation\n")
            f.write("\n")
            f.write("atoms = read('input.xyz')\n")
            f.write("result = ase_relaxation(atoms, charge=0, mult=1)\n")
            f.write("print(f'Final energy: {result.get_potential_energy()}')\n")

    db_path = "architector_workflow.db"

    with ArchitectorWorkflow(db_path) as workflow:
        submitted = submit_batch(
            workflow=workflow,
            root_dir="/path/to/jobs",
            batch_size=50,
            scheduler="flux",
            setup_func=setup_orca_input,  # Custom setup
            n_cores=8,
            n_hours=4,
            dry_run=True,  # Set to False to actually submit
        )
        print(f"Submitted {len(submitted)} jobs")


# =============================================================================
# Typical workflow on HPC
# =============================================================================


def typical_hpc_workflow():
    """
    Typical workflow for running on HPC:

    1. On local machine: Prepare workflow database
       -----------------------------------------------
       python architector_workflow_example.py  # Creates DB
       scp architector_workflow.db user@hpc:/path/to/project/

    2. On HPC: Submit initial batch
       ------------------------------
       python -m oact_utilities.workflows.submit_jobs \\
           architector_workflow.db \\
           jobs/ \\
           --batch-size 500 \\
           --scheduler flux

    3. On HPC: Monitor progress (in a cron job or periodically)
       ---------------------------------------------------------
       # Update statuses and show dashboard
       python -m oact_utilities.workflows.dashboard architector_workflow.db --update jobs/ --show-metrics

    4. On HPC: Submit more batches as jobs complete
       ----------------------------------------------
       python -m oact_utilities.workflows.submit_jobs \\
           architector_workflow.db \\
           jobs/ \\
           --batch-size 500

    5. On HPC: Handle failures
       ------------------------
       # View failed jobs
       python -m oact_utilities.workflows.dashboard \\
           architector_workflow.db \\
           --show-failed \\
           --limit 100

       # After fixing issues, reset and resubmit
       python -m oact_utilities.workflows.dashboard \\
           architector_workflow.db \\
           --reset-failed

       python -m oact_utilities.workflows.submit_jobs \\
           architector_workflow.db \\
           jobs/ \\
           --batch-size 100

    6. Copy results back
       ------------------
       scp user@hpc:/path/to/project/architector_workflow.db ./
       # Analyze locally using workflow API
    """
    pass


if __name__ == "__main__":
    # Uncomment to run
    db_path = setup_workflow()
    # print(f"\nWorkflow database created: {db_path}")
    print(__doc__)
    print("\nSee function docstrings for usage examples.")
