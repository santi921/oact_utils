"""Job submission utilities for architector workflows.

This module provides utilities to submit jobs from the workflow database
to HPC systems (Flux or SLURM).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Callable

from .architector_workflow import ArchitectorWorkflow, JobStatus


def prepare_job_directory(
    job_record,
    root_dir: Path,
    job_dir_pattern: str = "job_{orig_index}",
    setup_func: Callable | None = None,
) -> Path:
    """Create a job directory and prepare input files.

    Args:
        job_record: JobRecord from the workflow database.
        root_dir: Root directory where job directories will be created.
        job_dir_pattern: Pattern for job directory names.
        setup_func: Optional function to set up input files. Called with
                   (job_dir, job_record) as arguments.

    Returns:
        Path to the created job directory.
    """
    job_dir_name = job_dir_pattern.format(
        orig_index=job_record.orig_index,
        id=job_record.id,
        index_in_chunk=job_record.index_in_chunk,
    )
    job_dir = root_dir / job_dir_name
    job_dir.mkdir(parents=True, exist_ok=True)

    # Write geometry to XYZ file
    if job_record.geometry:
        xyz_file = job_dir / "input.xyz"
        with open(xyz_file, "w") as f:
            f.write(job_record.geometry.strip())
            f.write("\n")

    # Call custom setup function if provided
    if setup_func:
        setup_func(job_dir, job_record)

    return job_dir


def write_flux_job_file(
    job_dir: Path,
    n_cores: int = 4,
    n_hours: int = 2,
    queue: str = "pbatch",
    allocation: str = "dnn-sim",
    command: str = "python orca.py",
) -> Path:
    """Write a Flux job submission script.

    Args:
        job_dir: Directory where the job file will be written.
        n_cores: Number of cores to request.
        n_hours: Number of hours for job runtime.
        queue: Queue/partition name.
        allocation: Allocation/account name.
        command: Command to execute.

    Returns:
        Path to the created flux job file.
    """
    flux_script = job_dir / "flux_job.flux"

    lines = [
        "#!/bin/sh\n",
        "#flux: -N 1\n",
        f"#flux: -n {n_cores}\n",
        f"#flux: -q {queue}\n",
        f"#flux: -B {allocation}\n",
        f"#flux: -t {n_hours*60}m\n",
        "\n",
        "source ~/.bashrc\n",
        "conda activate py10mpi\n",
        "export LD_LIBRARY_PATH=/usr/WS1/vargas58/miniconda3/envs/py10mpi/lib:$LD_LIBRARY_PATH\n",
        "export JAX_PLATFORMS=cpu\n",
        f"{command}\n",
    ]

    with open(flux_script, "w") as f:
        f.writelines(lines)

    # Make executable
    flux_script.chmod(0o755)

    return flux_script


def write_slurm_job_file(
    job_dir: Path,
    n_cores: int = 4,
    n_hours: int = 2,
    queue: str = "pbatch",
    allocation: str = "dnn-sim",
    command: str = "python orca.py",
) -> Path:
    """Write a SLURM job submission script.

    Args:
        job_dir: Directory where the job file will be written.
        n_cores: Number of cores to request.
        n_hours: Number of hours for job runtime.
        queue: QOS name.
        allocation: Account name.
        command: Command to execute.

    Returns:
        Path to the created SLURM job file.
    """
    slurm_script = job_dir / "slurm_job.sh"

    lines = [
        "#!/bin/sh\n",
        "#SBATCH -N 1\n",
        "#SBATCH --constraint standard\n",
        f"#SBATCH --qos {queue}\n",
        f"#SBATCH --account {allocation}\n",
        f"#SBATCH -t {n_hours}:00:00\n",
        "\n",
        "source ~/.bashrc\n",
        "conda activate py10mpi\n",
        "export LD_LIBRARY_PATH=/usr/WS1/vargas58/miniconda3/envs/py10mpi/lib:$LD_LIBRARY_PATH\n",
        f"{command}\n",
    ]

    with open(slurm_script, "w") as f:
        f.writelines(lines)

    # Make executable
    slurm_script.chmod(0o755)

    return slurm_script


def submit_batch(
    workflow: ArchitectorWorkflow,
    root_dir: str | Path,
    batch_size: int = 10,
    scheduler: str = "flux",
    job_dir_pattern: str = "job_{orig_index}",
    setup_func: Callable | None = None,
    n_cores: int = 4,
    n_hours: int = 2,
    queue: str = "pbatch",
    allocation: str = "dnn-sim",
    command: str = "python orca.py",
    dry_run: bool = False,
) -> list[int]:
    """Submit a batch of ready jobs to the HPC scheduler.

    Args:
        workflow: ArchitectorWorkflow instance.
        root_dir: Root directory for job directories.
        batch_size: Number of jobs to submit in this batch.
        scheduler: Either "flux" or "slurm".
        job_dir_pattern: Pattern for job directory names.
        setup_func: Optional function to set up job-specific files.
        n_cores: Number of cores per job.
        n_hours: Runtime in hours.
        queue: Queue/partition/QOS name.
        allocation: Allocation/account name.
        command: Command to run in each job.
        dry_run: If True, prepare directories but don't submit.

    Returns:
        List of job IDs that were submitted.
    """
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    # Get ready jobs
    ready_jobs = workflow.get_jobs_by_status(JobStatus.READY)

    if not ready_jobs:
        print("No ready jobs to submit")
        return []

    # Limit to batch size
    jobs_to_submit = ready_jobs[:batch_size]
    print(f"Preparing {len(jobs_to_submit)} jobs for submission...")

    submitted_ids = []

    for i, job in enumerate(jobs_to_submit):
        # Prepare job directory
        job_dir = prepare_job_directory(
            job, root_dir, job_dir_pattern=job_dir_pattern, setup_func=setup_func
        )

        # Write job submission script
        if scheduler.lower() == "flux":
            job_script = write_flux_job_file(
                job_dir,
                n_cores=n_cores,
                n_hours=n_hours,
                queue=queue,
                allocation=allocation,
                command=command,
            )
            submit_cmd = ["flux", "batch", str(job_script)]
        elif scheduler.lower() == "slurm":
            job_script = write_slurm_job_file(
                job_dir,
                n_cores=n_cores,
                n_hours=n_hours,
                queue=queue,
                allocation=allocation,
                command=command,
            )
            submit_cmd = ["sbatch", str(job_script)]
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")

        print(f"  [{i+1}/{len(jobs_to_submit)}] Prepared job {job.id} in {job_dir}")

        # Submit job
        if not dry_run:
            try:
                result = subprocess.run(
                    submit_cmd,
                    cwd=job_dir,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                print(f"    Submitted: {result.stdout.strip()}")
                submitted_ids.append(job.id)
            except subprocess.CalledProcessError as e:
                print(f"    Error submitting job: {e.stderr}")
                continue
        else:
            print(f"    [DRY RUN] Would submit: {' '.join(submit_cmd)}")
            submitted_ids.append(job.id)

    # Mark jobs as running
    if submitted_ids and not dry_run:
        workflow.mark_jobs_as_running(submitted_ids)
        print(f"\nMarked {len(submitted_ids)} jobs as running")

    return submitted_ids


def main():
    """Main entry point for job submission script."""
    parser = argparse.ArgumentParser(
        description="Submit architector workflow jobs to HPC scheduler"
    )
    parser.add_argument("db_path", help="Path to workflow SQLite database")
    parser.add_argument("root_dir", help="Root directory for job directories")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of jobs to submit (default: 10)",
    )
    parser.add_argument(
        "--scheduler",
        choices=["flux", "slurm"],
        default="flux",
        help="HPC scheduler (default: flux)",
    )
    parser.add_argument(
        "--job-dir-pattern",
        default="job_{orig_index}",
        help="Pattern for job directory names (default: job_{orig_index})",
    )
    parser.add_argument(
        "--n-cores",
        type=int,
        default=4,
        help="Number of cores per job (default: 4)",
    )
    parser.add_argument(
        "--n-hours",
        type=int,
        default=2,
        help="Runtime in hours (default: 2)",
    )
    parser.add_argument(
        "--queue",
        default="pbatch",
        help="Queue/partition name (default: pbatch)",
    )
    parser.add_argument(
        "--allocation",
        default="dnn-sim",
        help="Allocation/account name (default: dnn-sim)",
    )
    parser.add_argument(
        "--command",
        default="python orca.py",
        help="Command to run (default: python orca.py)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare jobs but don't submit",
    )

    args = parser.parse_args()

    # Open workflow
    try:
        workflow = ArchitectorWorkflow(args.db_path)
    except FileNotFoundError:
        print(f"Error: Database not found at {args.db_path}")
        sys.exit(1)

    # Submit batch
    submitted_ids = submit_batch(
        workflow=workflow,
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        scheduler=args.scheduler,
        job_dir_pattern=args.job_dir_pattern,
        n_cores=args.n_cores,
        n_hours=args.n_hours,
        queue=args.queue,
        allocation=args.allocation,
        command=args.command,
        dry_run=args.dry_run,
    )

    print(f"\nTotal jobs submitted: {len(submitted_ids)}")

    workflow.close()


if __name__ == "__main__":
    main()
