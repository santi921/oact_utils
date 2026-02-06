"""Job submission utilities for architector workflows.

This module provides utilities to submit jobs from the workflow database
to HPC systems (Flux or SLURM). Jobs generate ORCA input files directly.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Callable, TypedDict

from ..core.orca.calc import write_orca_inputs
from ..utils.architector import xyz_string_to_atoms
from .architector_workflow import ArchitectorWorkflow, JobStatus

try:
    from parsl import python_app

    PARSL_AVAILABLE = True
except ImportError:
    # Parsl not installed - Parsl features won't be available
    PARSL_AVAILABLE = False
    python_app = None


class OrcaConfig(TypedDict, total=False):
    """Configuration options for ORCA calculations.

    Attributes:
        functional: DFT functional (default: "wB97M-V").
        simple_input: Input template (default: "omol"). Options: "omol", "x2c", "dk3".
        actinide_basis: Basis set for actinides (default: "ma-def-TZVP").
        actinide_ecp: ECP for actinides (default: None).
        non_actinide_basis: Basis set for non-actinides (default: "def2-TZVPD").
        scf_MaxIter: Maximum SCF iterations (default: None, uses ORCA default).
        nbo: Enable NBO analysis (default: False).
        opt: Enable geometry optimization (default: False).
        orca_path: Path to ORCA executable (default: scheduler-specific).
    """

    functional: str
    simple_input: str
    actinide_basis: str
    actinide_ecp: str | None
    non_actinide_basis: str
    scf_MaxIter: int | None
    nbo: bool
    opt: bool
    orca_path: str


DEFAULT_ORCA_CONFIG: OrcaConfig = {
    "functional": "wB97M-V",
    "simple_input": "omol",
    "actinide_basis": "ma-def-TZVP",
    "actinide_ecp": "def-ECP",
    "non_actinide_basis": "def2-TZVPD",
    "scf_MaxIter": None,
    "nbo": False,
    "opt": False,
}

DEFAULT_ORCA_PATHS = {
    "flux": "/usr/workspace/vargas58/orca-6.1.0-f.0_linux_x86-64/bin/orca",
    "macos_arm64_openmpi411": "/Users/santiagovargas/Documents/orca_6_1_0_macosx_arm64_openmpi411/orca",
    "slurm": "orca",
}

DEFAULT_LD_LIBRARY_PATHS = {
    "flux": "/usr/WS1/vargas58/miniconda3/envs/py10mpi/lib",
    "slurm": "",
}


def prepare_job_directory(
    job_record,
    root_dir: Path,
    job_dir_pattern: str = "job_{orig_index}",
    orca_config: OrcaConfig | None = None,
    n_cores: int = 4,
    setup_func: Callable | None = None,
    return_full_path: bool = True,
) -> Path:
    """Create a job directory and prepare ORCA input files.

    Args:
        job_record: JobRecord from the workflow database.
        root_dir: Root directory where job directories will be created.
        job_dir_pattern: Pattern for job directory names.
        orca_config: ORCA calculation configuration.
        n_cores: Number of CPU cores for ORCA.
        setup_func: Optional function to set up additional files. Called with
                   (job_dir, job_record) as arguments.
        return_full_path: If True, return full path to job directory. If False, return relative path.

    Returns:
        Path to the created job directory.
    """
    # Use limited, explicit placeholder replacement to avoid format string issues.
    # Supported placeholders: {orig_index}, {id}. Any other braces are rejected.
    pattern = job_dir_pattern

    allowed_placeholders = ("{orig_index}", "{id}")
    temp_pattern = pattern
    for placeholder in allowed_placeholders:
        temp_pattern = temp_pattern.replace(placeholder, "")
    if "{" in temp_pattern or "}" in temp_pattern:
        raise ValueError(
            f"Unsupported placeholder or stray brace in job_dir_pattern: {job_dir_pattern!r}"
        )

    job_dir_name = pattern.replace("{orig_index}", str(job_record.orig_index)).replace(
        "{id}", str(job_record.id)
    )

    job_dir = root_dir / job_dir_name
    job_dir.mkdir(parents=True, exist_ok=True)

    # Merge with defaults
    config: OrcaConfig = {**DEFAULT_ORCA_CONFIG, **(orca_config or {})}

    # Write ORCA input file
    if job_record.geometry:
        atoms = xyz_string_to_atoms(job_record.geometry)
        charge = job_record.charge if job_record.charge is not None else 0
        # DB stores spin multiplicity directly
        mult = job_record.spin if job_record.spin is not None else 1

        write_orca_inputs(
            atoms=atoms,
            output_directory=str(job_dir),
            charge=charge,
            mult=mult,
            nbo=config.get("nbo", False),
            cores=n_cores,
            opt=config.get("opt", False),
            functional=config.get("functional", "wB97M-V"),
            simple_input=config.get("simple_input", "omol"),
            orca_path=config.get("orca_path"),
            scf_MaxIter=config.get("scf_MaxIter"),
            actinide_basis=config.get("actinide_basis", "ma-def-TZVP"),
            actinide_ecp=config.get("actinide_ecp"),
            non_actinide_basis=config.get("non_actinide_basis", "def2-TZVPD"),
        )

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
    orca_path: str = DEFAULT_ORCA_PATHS["flux"],
    conda_env: str = "py10mpi",
    input_file: str = "orca.inp",
    ld_library_path: str | None = None,
) -> Path:
    """Write a Flux job submission script.

    Args:
        job_dir: Directory where the job file will be written.
        n_cores: Number of cores to request.
        n_hours: Number of hours for job runtime.
        queue: Queue/partition name.
        allocation: Allocation/account name.
        orca_path: Path to ORCA executable.
        conda_env: Conda environment to activate.
        input_file: Name of the ORCA input file.

    Returns:
        Path to the created flux job file.
    """
    # Use absolute path so job runs in correct directory regardless of submission location
    job_dir_abs = job_dir.resolve()
    flux_script = job_dir_abs / "flux_job.flux"

    lines = [
        "#!/bin/sh\n",
        "#flux: -N 1\n",
        f"#flux: -n {n_cores}\n",
        f"#flux: -q {queue}\n",
        f"#flux: -B {allocation}\n",
        f"#flux: -t {n_hours*60}m\n",
        "\n",
        f"cd {job_dir_abs}\n",
        "\n",
        "source ~/.bashrc\n",
        f"conda activate {conda_env}\n",
        (
            f"export LD_LIBRARY_PATH={ld_library_path}:$LD_LIBRARY_PATH\n"
            if ld_library_path
            else f"export LD_LIBRARY_PATH={DEFAULT_LD_LIBRARY_PATHS['flux']}:$LD_LIBRARY_PATH\n"
        ),
        f"{orca_path} {input_file}\n",
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
    orca_path: str = "orca",
    conda_env: str = "py10mpi",
    input_file: str = "orca.inp",
    ld_library_path: str | None = None,
) -> Path:
    """Write a SLURM job submission script.

    Args:
        job_dir: Directory where the job file will be written.
        n_cores: Number of cores to request.
        n_hours: Number of hours for job runtime.
        queue: QOS name.
        allocation: Account name.
        orca_path: Path to ORCA executable.
        conda_env: Conda environment to activate.
        input_file: Name of the ORCA input file.

    Returns:
        Path to the created SLURM job file.
    """
    # Use absolute path so job runs in correct directory regardless of submission location
    job_dir_abs = job_dir.resolve()
    slurm_script = job_dir_abs / "slurm_job.sh"

    lines = [
        "#!/bin/sh\n",
        "#SBATCH -N 1\n",
        f"#SBATCH --ntasks-per-node {n_cores}\n",
        "#SBATCH --constraint standard\n",
        f"#SBATCH --qos {queue}\n",
        f"#SBATCH --account {allocation}\n",
        f"#SBATCH -t {n_hours}:00:00\n",
        f"#SBATCH -o {job_dir_abs}/slurm.out\n",
        f"#SBATCH -e {job_dir_abs}/slurm.err\n",
        "\n",
        f"cd {job_dir_abs}\n",
        "\n",
        "source ~/.bashrc\n",
        f"conda activate {conda_env}\n",
        (
            f"export LD_LIBRARY_PATH={ld_library_path}:$LD_LIBRARY_PATH\n"
            if ld_library_path
            else f"export LD_LIBRARY_PATH={DEFAULT_LD_LIBRARY_PATHS['slurm']}:$LD_LIBRARY_PATH\n"
        ),
        f"{orca_path} {input_file}\n",
    ]

    with open(slurm_script, "w") as f:
        f.writelines(lines)

    # Make executable
    slurm_script.chmod(0o755)

    return slurm_script


def filter_jobs_for_submission(
    workflow: ArchitectorWorkflow,
    num_jobs: int,
    max_fail_count: int | None = None,
) -> list:
    """Filter jobs that are ready to submit.

    Args:
        workflow: ArchitectorWorkflow instance
        num_jobs: Number of jobs to return
        max_fail_count: Skip jobs with fail_count >= this value

    Returns:
        List of JobRecords ready for submission
    """
    # Get ready jobs (DB is source of truth)
    ready_jobs = workflow.get_jobs_by_status([JobStatus.TO_RUN, JobStatus.READY])

    # Apply fail_count filter if specified
    if max_fail_count is not None:
        original_count = len(ready_jobs)
        ready_jobs = [j for j in ready_jobs if j.fail_count < max_fail_count]
        skipped = original_count - len(ready_jobs)
        if skipped > 0:
            print(f"Skipped {skipped} jobs with fail_count >= {max_fail_count}")

    # Limit to requested count
    jobs_to_submit = ready_jobs[:num_jobs]

    print(f"Found {len(ready_jobs)} ready jobs, submitting {len(jobs_to_submit)}")
    return jobs_to_submit


# Only define Parsl-related functions if Parsl is available
if PARSL_AVAILABLE:

    @python_app
    def orca_job_wrapper(
        job_id: int,
        job_dir: str,
        orca_config: dict,
        n_cores: int,
        timeout_seconds: int = 7200,
    ) -> dict:
        """Execute ORCA job within Parsl worker.

        This runs as a Parsl python_app, executing directly on the worker node.
        Parsl handles CPU affinity and worker management automatically.

        Args:
            job_id: Workflow database job ID
            job_dir: Absolute path to job directory
            orca_config: ORCA configuration dictionary
            n_cores: Number of cores for ORCA
            timeout_seconds: Job timeout in seconds (default: 7200 = 2 hours)

        Returns:
            Dict with job_id, status, metrics
        """
        import os
        import subprocess
        import time
        from pathlib import Path

        job_dir_path = Path(job_dir)
        input_file = job_dir_path / "orca.inp"

        # Verify input file exists
        if not input_file.exists():
            return {
                "job_id": job_id,
                "status": "failed",
                "error": f"Input file not found: {input_file}",
            }

        # Get ORCA path from config
        orca_cmd = orca_config.get("orca_path", "orca")

        # Let Parsl handle CPU affinity - no manual pinning needed
        os.environ["OMP_NUM_THREADS"] = "1"

        start_time = time.time()

        try:
            # Run ORCA
            result = subprocess.run(
                [orca_cmd, str(input_file)],
                cwd=job_dir,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )

            elapsed = time.time() - start_time

            if result.returncode == 0:
                return {
                    "job_id": job_id,
                    "status": "completed",
                    "wall_time": elapsed,
                }
            else:
                return {
                    "job_id": job_id,
                    "status": "failed",
                    "error": f"ORCA exited with code {result.returncode}",
                    "stderr": result.stderr[:500] if result.stderr else "",
                }

        except subprocess.TimeoutExpired:
            return {
                "job_id": job_id,
                "status": "timeout",
                "error": f"Job exceeded {timeout_seconds}s timeout",
            }
        except Exception as e:
            return {
                "job_id": job_id,
                "status": "failed",
                "error": str(e),
            }


def build_parsl_config_flux(
    max_workers: int = 4,
    cores_per_worker: int = 16,
    conda_env: str = "py10mpi",
    conda_base: str = "/usr/WS1/vargas58/miniconda3",
    ld_library_path: str | None = None,
):
    """Build Parsl Config for Flux single-node execution.

    Uses LocalProvider since Flux doesn't support scale-out in Parsl.
    This configuration runs all workers on the local node.

    Args:
        max_workers: Maximum number of concurrent workers
        cores_per_worker: CPU cores per worker
        conda_env: Conda environment name
        conda_base: Conda base path
        ld_library_path: Override LD_LIBRARY_PATH

    Returns:
        Parsl Config object
    """
    if not PARSL_AVAILABLE:
        raise ImportError(
            "Parsl is not installed. Please install with: pip install 'parsl>=2024.1'"
        )

    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    from parsl.providers import LocalProvider

    # Worker initialization commands
    ld_lib = ld_library_path or DEFAULT_LD_LIBRARY_PATHS.get("flux", "")
    worker_init = f"""
source ~/.bashrc
conda activate {conda_env}
export LD_LIBRARY_PATH={ld_lib}:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
export JAX_PLATFORMS=cpu
"""

    provider = LocalProvider(
        worker_init=worker_init,
    )

    executor = HighThroughputExecutor(
        label="flux_htex",
        cores_per_worker=cores_per_worker,
        max_workers=max_workers,
        provider=provider,
    )

    return Config(executors=[executor])


def submit_batch_parsl(
    workflow: ArchitectorWorkflow,
    root_dir: str | Path,
    num_jobs: int,
    max_workers: int = 4,
    cores_per_worker: int = 16,
    job_dir_pattern: str = "job_{orig_index}",
    orca_config: OrcaConfig | None = None,
    setup_func: Callable | None = None,
    n_cores: int = 16,
    conda_env: str = "py10mpi",
    conda_base: str = "/usr/WS1/vargas58/miniconda3",
    ld_library_path: str | None = None,
    dry_run: bool = False,
    max_fail_count: int | None = None,
    timeout_seconds: int = 72000,
) -> list[int]:
    """Submit batch of jobs using Parsl for concurrent execution.

    Args:
        workflow: ArchitectorWorkflow instance
        root_dir: Root directory for job directories
        num_jobs: Total number of jobs to submit
        max_workers: Maximum number of concurrent workers
        cores_per_worker: CPU cores per worker
        job_dir_pattern: Pattern for job directory names
        orca_config: ORCA configuration
        setup_func: Optional setup function per job
        n_cores: Cores per ORCA job
        conda_env: Conda environment name
        conda_base: Conda base path
        ld_library_path: Override LD_LIBRARY_PATH
        dry_run: Prepare but don't submit
        max_fail_count: Skip jobs with fail_count >= this value
        timeout_seconds: Job timeout in seconds (default: 7200 = 2 hours)

    Returns:
        List of submitted job IDs
    """
    if not PARSL_AVAILABLE:
        print(
            "Error: Parsl is not installed. Please install with: pip install 'parsl>=2024.1'"
        )
        return []

    from concurrent.futures import as_completed

    import parsl

    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    # Merge ORCA config
    config: OrcaConfig = {**DEFAULT_ORCA_CONFIG, **(orca_config or {})}
    if "orca_path" not in config or config.get("orca_path") is None:
        config["orca_path"] = DEFAULT_ORCA_PATHS.get("flux", DEFAULT_ORCA_PATHS["flux"])

    # Filter jobs for submission (DB status only)
    jobs_to_submit = filter_jobs_for_submission(
        workflow,
        num_jobs=num_jobs,
        max_fail_count=max_fail_count,
    )

    if not jobs_to_submit:
        print("No jobs available for submission after filtering")
        return []

    print(f"\nPreparing {len(jobs_to_submit)} jobs for Parsl submission...")

    # Prepare job directories
    print("Setting up job directories...")
    for i, job in enumerate(jobs_to_submit, 1):
        job_dir = prepare_job_directory(
            job,
            root_dir,
            job_dir_pattern=job_dir_pattern,
            orca_config=config,
            n_cores=n_cores,
            setup_func=setup_func,
        )
        print(f"  [{i}/{len(jobs_to_submit)}] Prepared {job_dir}")

    if dry_run:
        print("\n[DRY RUN] Would submit to Parsl executor")
        print(f"  Max workers: {max_workers}")
        print(f"  Cores per worker: {cores_per_worker}")
        return [j.id for j in jobs_to_submit]

    # Build Parsl configuration
    print("\nBuilding Parsl config (Flux single-node)...")
    parsl_config = build_parsl_config_flux(
        max_workers=max_workers,
        cores_per_worker=cores_per_worker,
        conda_env=conda_env,
        conda_base=conda_base,
        ld_library_path=ld_library_path,
    )

    # Initialize Parsl
    try:
        parsl.clear()
        parsl.load(parsl_config)
        print("Parsl executor loaded successfully")
    except Exception as e:
        print(f"Failed to initialize Parsl: {e}")
        print("Check your conda environment and ORCA installation")
        return []

    # Submit futures
    print(f"\nSubmitting {len(jobs_to_submit)} jobs to Parsl...")
    futures = []

    for job in jobs_to_submit:
        job_dir_name = job_dir_pattern.replace(
            "{orig_index}", str(job.orig_index)
        ).replace("{id}", str(job.id))
        job_dir_abs = (root_dir / job_dir_name).resolve()

        future = orca_job_wrapper(
            job_id=job.id,
            job_dir=str(job_dir_abs),
            orca_config=dict(config),
            n_cores=n_cores,
            timeout_seconds=timeout_seconds,
        )
        futures.append((job.id, future))

    # Mark jobs as running
    submitted_ids = [j.id for j in jobs_to_submit]
    workflow.mark_jobs_as_running(submitted_ids)
    print(f"Marked {len(submitted_ids)} jobs as RUNNING in database")

    # Monitor futures concurrently (CRITICAL: use as_completed, not sequential loop)
    print("\nMonitoring job execution...")
    print("(Press Ctrl+C for graceful shutdown)\n")

    completed_ids = []
    failed_ids = []

    # Create future->job_id mapping for concurrent completion
    futures_map = {future: job_id for job_id, future in futures}

    try:
        # as_completed() yields futures as they finish (concurrent, not sequential!)
        for future in as_completed(futures_map.keys()):
            job_id = futures_map[future]
            try:
                result = future.result()

                if result["status"] == "completed":
                    workflow.update_status(job_id, JobStatus.COMPLETED)
                    completed_ids.append(job_id)
                    print(
                        f"✓ Job {job_id} completed ({len(completed_ids)}/{len(futures)} done)"
                    )
                elif result["status"] == "timeout":
                    workflow.update_status(
                        job_id, JobStatus.TIMEOUT, error_message=result.get("error")
                    )
                    failed_ids.append(job_id)
                    print(f"⏱ Job {job_id} timeout")
                else:
                    workflow.update_status(
                        job_id, JobStatus.FAILED, error_message=result.get("error")
                    )
                    workflow._execute_with_retry(
                        "UPDATE structures SET fail_count = COALESCE(fail_count, 0) + 1 WHERE id = ?",
                        (job_id,),
                    )
                    failed_ids.append(job_id)
                    error_msg = result.get("error", "Unknown error")[:100]
                    print(f"✗ Job {job_id} failed: {error_msg}")

            except Exception as e:
                workflow.update_status(job_id, JobStatus.FAILED, error_message=str(e))
                workflow._execute_with_retry(
                    "UPDATE structures SET fail_count = COALESCE(fail_count, 0) + 1 WHERE id = ?",
                    (job_id,),
                )
                failed_ids.append(job_id)
                print(f"✗ Job {job_id} exception: {str(e)[:100]}")

    except KeyboardInterrupt:
        print("\n\nGraceful shutdown requested...")

    finally:
        # Cleanup Parsl
        print("\nCleaning up Parsl executor...")
        try:
            dfk = parsl.dfk()
            if dfk is not None:
                dfk.cleanup()
        except Exception as e:
            print(f"Warning: Parsl cleanup failed: {e}")

        try:
            parsl.clear()
        except Exception:
            pass

    print("\nSubmission complete:")
    print(f"  ✓ Completed: {len(completed_ids)}")
    print(f"  ✗ Failed: {len(failed_ids)}")
    print(f"  Total: {len(submitted_ids)}")

    return submitted_ids


def submit_batch(
    workflow: ArchitectorWorkflow,
    root_dir: str | Path,
    batch_size: int = 10,
    scheduler: str = "flux",
    job_dir_pattern: str = "job_{orig_index}",
    orca_config: OrcaConfig | None = None,
    setup_func: Callable | None = None,
    n_cores: int = 4,
    n_hours: int = 2,
    queue: str = "pbatch",
    allocation: str = "dnn-sim",
    conda_env: str = "py10mpi",
    dry_run: bool = False,
    max_fail_count: int | None = None,
) -> list[int]:
    """Submit a batch of ready jobs to the HPC scheduler.

    Args:
        workflow: ArchitectorWorkflow instance.
        root_dir: Root directory for job directories.
        batch_size: Number of jobs to submit in this batch.
        scheduler: Either "flux" or "slurm".
        job_dir_pattern: Pattern for job directory names.
        orca_config: ORCA calculation configuration.
        setup_func: Optional function to set up job-specific files.
        n_cores: Number of cores per job.
        n_hours: Runtime in hours.
        queue: Queue/partition/QOS name.
        allocation: Allocation/account name.
        conda_env: Conda environment to activate.
        dry_run: If True, prepare directories but don't submit.
        max_fail_count: If specified, skip jobs with fail_count >= this value.

    Returns:
        List of job IDs that were submitted.
    """
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    # Merge config with defaults and set scheduler-specific orca_path if not provided
    config: OrcaConfig = {**DEFAULT_ORCA_CONFIG, **(orca_config or {})}
    if "orca_path" not in config or config.get("orca_path") is None:
        config["orca_path"] = DEFAULT_ORCA_PATHS.get(
            scheduler.lower(), DEFAULT_ORCA_PATHS["flux"]
        )

    # Get ready jobs (includes both TO_RUN and READY for backward compatibility)
    ready_jobs = workflow.get_jobs_by_status([JobStatus.TO_RUN, JobStatus.READY])

    # Filter out jobs that have failed too many times
    if max_fail_count is not None:
        original_count = len(ready_jobs)
        ready_jobs = [j for j in ready_jobs if j.fail_count < max_fail_count]
        skipped = original_count - len(ready_jobs)
        if skipped > 0:
            print(f"Skipping {skipped} jobs that have failed {max_fail_count}+ times")

    if not ready_jobs:
        print("No ready jobs to submit")
        return []

    # Limit to batch size
    jobs_to_submit = ready_jobs[:batch_size]
    print(f"Preparing {len(jobs_to_submit)} jobs for submission...")

    submitted_ids = []

    for i, job in enumerate(jobs_to_submit):
        # Prepare job directory with ORCA input
        job_dir = prepare_job_directory(
            job,
            root_dir,
            job_dir_pattern=job_dir_pattern,
            orca_config=config,
            n_cores=n_cores,
            setup_func=setup_func,
        )

        # Write job submission script
        orca_path = config.get("orca_path", DEFAULT_ORCA_PATHS["flux"])
        if scheduler.lower() == "flux":
            job_script = write_flux_job_file(
                job_dir,
                n_cores=n_cores,
                n_hours=n_hours,
                queue=queue,
                allocation=allocation,
                orca_path=orca_path,
                conda_env=conda_env,
            )
            submit_cmd = ["flux", "batch", job_script.name]

        elif scheduler.lower() == "slurm":
            job_script = write_slurm_job_file(
                job_dir,
                n_cores=n_cores,
                n_hours=n_hours,
                queue=queue,
                allocation=allocation,
                orca_path=orca_path,
                conda_env=conda_env,
            )
            submit_cmd = ["sbatch", job_script.name]
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

    # Submission mode
    parser.add_argument(
        "--use-parsl",
        action="store_true",
        help="Use Parsl for concurrent execution on exclusive nodes (Flux single-node)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of jobs to submit (default: 10). For Parsl mode, this is the total job count.",
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
        "--conda-env",
        default="py10mpi",
        help="Conda environment to activate (default: py10mpi)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare jobs but don't submit",
    )

    # Parsl-specific options
    parsl_group = parser.add_argument_group("Parsl Options (--use-parsl)")
    parsl_group.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum concurrent workers for Parsl (default: 4)",
    )
    parsl_group.add_argument(
        "--cores-per-worker",
        type=int,
        default=16,
        help="CPU cores per Parsl worker (default: 16)",
    )
    parsl_group.add_argument(
        "--conda-base",
        default="/usr/WS1/vargas58/miniconda3",
        help="Conda base path for Parsl workers (default: /usr/WS1/vargas58/miniconda3)",
    )
    parsl_group.add_argument(
        "--job-timeout",
        type=int,
        default=7200,
        help="Job timeout in seconds for Parsl mode (default: 7200 = 2 hours)",
    )
    parser.add_argument(
        "--max-fail-count",
        type=int,
        default=None,
        help="Skip jobs that have failed this many times or more",
    )

    # ORCA configuration arguments
    orca_group = parser.add_argument_group("ORCA Configuration")
    orca_group.add_argument(
        "--functional",
        default="wB97M-V",
        help="DFT functional (default: wB97M-V)",
    )
    orca_group.add_argument(
        "--simple-input",
        choices=["omol", "x2c", "dk3"],
        default="omol",
        help="ORCA input template (default: omol)",
    )
    orca_group.add_argument(
        "--actinide-basis",
        default="ma-def-TZVP",
        help="Basis set for actinides (default: ma-def-TZVP)",
    )
    orca_group.add_argument(
        "--actinide-ecp",
        default="def-ECP",
        help="ECP for actinides (default: def-ECP)",
    )
    orca_group.add_argument(
        "--non-actinide-basis",
        default="def2-TZVPD",
        help="Basis set for non-actinides (default: def2-TZVPD)",
    )
    orca_group.add_argument(
        "--scf-maxiter",
        type=int,
        default=None,
        help="Maximum SCF iterations (default: ORCA default)",
    )
    orca_group.add_argument(
        "--nbo",
        action="store_true",
        help="Enable NBO analysis",
    )
    orca_group.add_argument(
        "--opt",
        action="store_true",
        help="Enable geometry optimization",
    )
    orca_group.add_argument(
        "--orca-path",
        default=None,
        help="Path to ORCA executable (default: scheduler-specific)",
    )

    args = parser.parse_args()

    # Build ORCA config from CLI arguments
    orca_config: OrcaConfig = {
        "functional": args.functional,
        "simple_input": args.simple_input,
        "actinide_basis": args.actinide_basis,
        "actinide_ecp": args.actinide_ecp,
        "non_actinide_basis": args.non_actinide_basis,
        "scf_MaxIter": args.scf_maxiter,
        "nbo": args.nbo,
        "opt": args.opt,
    }
    if args.orca_path:
        orca_config["orca_path"] = args.orca_path

    # Open workflow
    try:
        workflow = ArchitectorWorkflow(args.db_path)
    except FileNotFoundError:
        print(f"Error: Database not found at {args.db_path}")
        sys.exit(1)

    # Submit based on mode
    if args.use_parsl:
        # Parsl mode: concurrent execution on single node
        submitted_ids = submit_batch_parsl(
            workflow=workflow,
            root_dir=args.root_dir,
            num_jobs=args.batch_size,
            max_workers=args.max_workers,
            cores_per_worker=args.cores_per_worker,
            job_dir_pattern=args.job_dir_pattern,
            orca_config=orca_config,
            n_cores=args.n_cores,
            conda_env=args.conda_env,
            conda_base=args.conda_base,
            dry_run=args.dry_run,
            max_fail_count=args.max_fail_count,
            timeout_seconds=args.job_timeout,
        )
    else:
        # Traditional mode: one job script per job
        submitted_ids = submit_batch(
            workflow=workflow,
            root_dir=args.root_dir,
            batch_size=args.batch_size,
            scheduler=args.scheduler,
            job_dir_pattern=args.job_dir_pattern,
            orca_config=orca_config,
            n_cores=args.n_cores,
            n_hours=args.n_hours,
            queue=args.queue,
            allocation=args.allocation,
            conda_env=args.conda_env,
            dry_run=args.dry_run,
            max_fail_count=args.max_fail_count,
        )

    print(f"\nTotal jobs submitted: {len(submitted_ids)}")

    workflow.close()


if __name__ == "__main__":
    main()
