"""Job submission utilities for architector workflows.

This module provides utilities to submit jobs from the workflow database
to HPC systems (Flux or SLURM). Jobs generate ORCA input files directly.
"""

from __future__ import annotations

import argparse
import inspect
import os
import random
import shlex
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Literal, TypedDict

from ..core.orca.calc import write_orca_inputs
from ..core.orca.sella_runner import write_sella_runner_shim
from ..utils.analysis import (
    GENERATOR_AVAILABLE,
    find_timings_and_cores,
    parse_generator_data,
    parse_job_metrics,
)
from ..utils.architector import xyz_string_to_atoms
from ..utils.status import check_job_termination, parse_failure_reason, pull_log_file
from .architector_workflow import ArchitectorWorkflow, JobRecord, JobStatus
from .clean import (
    MARKER_ERROR_MESSAGE,
    _process_job,
    _purge_failed_job,
    _write_marker_file,
    is_marker_blocked,
)
from .job_dir_patterns import (
    DEFAULT_JOB_DIR_PATTERN,
    apply_job_dir_prefix,
    render_job_dir_pattern,
)
from .wandb_logger import (
    SNAPSHOT_INTERVAL_SEC,
    WANDB_AVAILABLE,
    add_wandb_args,
    backfill_terminal_cdfs,
    finish_wandb_run,
    init_wandb_run,
    log_job_result,
    log_progress_snapshot,
)

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
        simple_input: Input template (default: "omol"). Options: "omol", "omol_base", "x2c", "dk3", "pm3" (PM3 semiempirical, debug only -- no actinide support).
        actinide_basis: Basis set for actinides (default: "ma-def-TZVP").
        actinide_ecp: ECP for actinides (default: None).
        non_actinide_basis: Basis set for non-actinides (default: "def2-TZVPD").
        scf_MaxIter: Maximum SCF iterations (default: None, uses ORCA default).
        nbo: Enable NBO analysis (default: False).
        mbis: Enable MBIS population analysis (default: False).
        optimizer: Optimizer engine: "orca" (native), "sella" (external ASE), or None (single-point).
        opt_level: ORCA optimization convergence level (only with optimizer="orca").
        fmax: Sella force convergence threshold in Eh/Bohr (only with optimizer="sella").
        max_opt_steps: Maximum Sella optimization steps (only with optimizer="sella", default: 100).
        save_all_steps: Save all ORCA output files per Sella step (only with optimizer="sella").
        orca_path: Path to ORCA executable (default: scheduler-specific).
        ks_method: KS wavefunction type: "rks", "uks", or "roks" (default: None, ORCA auto-detects).
        mem_per_job_mb: Optional total-job ORCA memory budget in MB. When set,
            `%maxcore` is sized so total memory stays under 85% of this value.
            Recommended on memory-constrained nodes (Sandia CTS-1: ~60000,
            TLCC2: ~30000). Default: None (no clamp; per-process floor only).
    """

    functional: str
    simple_input: str
    actinide_basis: str
    actinide_ecp: str | None
    non_actinide_basis: str
    scf_MaxIter: int | None
    nbo: bool
    mbis: bool
    optimizer: Literal["orca", "sella"] | None
    opt_level: Literal["loose", "normal", "tight", "verytight"]
    fmax: float
    max_opt_steps: int | None
    save_all_steps: bool
    orca_path: str
    diis_option: str | None
    ks_method: str | None
    mem_per_job_mb: int | None


DEFAULT_ORCA_CONFIG: OrcaConfig = {
    "functional": "wB97M-V",
    "simple_input": "omol",
    "actinide_basis": "ma-def-TZVP",
    "actinide_ecp": "def-ECP",
    "non_actinide_basis": "def2-TZVPD",
    "scf_MaxIter": None,
    "nbo": False,
    "mbis": False,
    "optimizer": None,
    "opt_level": "normal",
    "fmax": 0.05,
    "max_opt_steps": None,
    "save_all_steps": False,
    "diis_option": None,
    "ks_method": None,
}

DEFAULT_ORCA_PATHS = {
    "flux": "/usr/workspace/vargas58/orca-6.1.0-f.0_linux_x86-64/bin/orca",
    "macos_arm64_openmpi411": "/Users/santiagovargas/Documents/orca_6_1_0_macosx_arm64_openmpi411/orca",
    "slurm": "orca",
    # Sandia intentionally has no default: ORCA is a user-installed shared
    # build under each user's $HOME, so a single hardcoded path is wrong for
    # every teammate but the one who installed it. The CLI requires
    # --orca-path when --hpc-site sandia is selected.
}

DEFAULT_LD_LIBRARY_PATHS = {
    "flux": "/usr/WS1/vargas58/miniconda3/envs/py10mpi/lib",
    "slurm": "",
}

# Sandia CTS1/TLCC2 defaults. The 'aue' OpenMPI module is the build that
# matches ORCA's shared-library build; the MPI MCA settings disable
# PSM2/Omni-Path because some compute partitions misroute traffic on it.
SANDIA_DEFAULT_OPENMPI_MODULE = "aue/openmpi/4.1.6-gcc-12.3.0"
SANDIA_DEFAULT_PARTITION = "attaway"
SANDIA_DEFAULT_QOS = "normal"
SANDIA_DEFAULT_ACCOUNT = "fy250086"
SANDIA_DEFAULT_NTASKS_PER_NODE = 36  # cts1=36; tlcc2=16
SANDIA_DEFAULT_OMPI_MCA = {
    "OMPI_MCA_pml": "ob1",
    "OMPI_MCA_mtl": "^psm2",
    "OMPI_MCA_btl": "tcp,self,vader",
}

# DRAC (Digital Research Alliance of Canada: Fir/Narval/Nibi/Rorqual/Trillium)
# defaults. Single-node SLURM; ORCA comes from the module system and is invoked
# via $EBROOTORCA/orca (never srun/mpirun -- ORCA spawns its own MPI from
# %pal nprocs). No --qos/--partition (auto-assigned by time+memory) and no OMPI
# MCA overrides (InfiniBand uses OpenMPI defaults; the Sandia TCP/PSM2 settings
# would cripple it). The venv is only needed for the Sella optimizer path;
# plain ORCA runs need no Python environment.
DRAC_DEFAULT_MODULE_LOAD = "StdEnv/2023 gcc/12.3 openmpi/4.1.5 orca/6.1.0"
DRAC_DEFAULT_ORCA_BIN = "$EBROOTORCA/orca"


def _write_job_update(
    workflow: ArchitectorWorkflow,
    update: dict,
) -> None:
    """Write a single job's status and metrics to the DB, committed immediately.

    Each call is its own transaction so a BUSY retry cannot roll back
    a different job's updates (the old batching bug).

    Args:
        update: Dict with keys job_id (int), status (JobStatus),
            error_message (str | None), increment_fail_count (bool),
            metrics (dict | None -- keys: job_dir, max_forces, scf_steps,
                     final_energy, wall_time, n_cores).
    """
    job_id = update["job_id"]

    set_clauses = [
        "status = ?",
        "worker_id = NULL",
        "updated_at = CURRENT_TIMESTAMP",
    ]
    params: list = [update["status"].value]

    if update.get("increment_fail_count"):
        set_clauses.append("fail_count = COALESCE(fail_count, 0) + 1")

    if update.get("error_message") is not None:
        set_clauses.append("error_message = ?")
        params.append(update["error_message"])

    # Merge metrics into the same UPDATE to use a single statement+commit.
    metrics = update.get("metrics")
    if metrics:
        for col in (
            "job_dir",
            "max_forces",
            "scf_steps",
            "final_energy",
            "wall_time",
            "n_cores",
        ):
            if metrics.get(col) is not None:
                set_clauses.append(f"{col} = ?")
                params.append(metrics[col])

    query = f"UPDATE structures SET {', '.join(set_clauses)} WHERE id = ?"
    params.append(job_id)
    workflow._execute_with_retry(query, tuple(params))
    workflow._commit_with_retry()


def _resolve_scheduler_job_id() -> str:
    """Return the scheduler-issued job ID for the current process.

    Reads SLURM_JOB_ID, FLUX_JOB_ID, and PBS_JOBID. For PBS Pro the leading
    numeric+host portion (e.g. ``123.server`` in ``123.server.fqdn``) is
    retained so it matches the set returned by ``qstat -u $USER``.

    Returns:
        Scheduler job ID string, or ``"pid_<PID>"`` sentinel when no
        scheduler env var is set. The sentinel is excluded from orphan
        recovery to avoid resetting jobs launched outside an allocation.
    """
    slurm = os.environ.get("SLURM_JOB_ID")
    if slurm:
        return slurm
    flux = os.environ.get("FLUX_JOB_ID")
    if flux:
        return flux
    pbs = os.environ.get("PBS_JOBID")
    if pbs:
        return pbs
    return f"pid_{os.getpid()}"


def _cleanup_completed_job_inline(
    job_dir: str,
    root_dir: Path,
    optimizer: str | None,
) -> None:
    """Run --clean-all (tmp + bas) on a job_dir that just completed.

    Reuses ``clean._process_job`` with ``skip_revalidation=True`` since the
    Parsl worker already verified ``ORCA TERMINATED NORMALLY``. Errors are
    printed but never propagate -- cleanup must not derail the submitter.
    """
    try:
        _matched, _freed, errs = _process_job(
            Path(job_dir).resolve(),
            root_dir.resolve(),
            categories={"tmp", "bas"},
            execute=True,
            hours_cutoff=24,
            optimizer=optimizer,
            skip_revalidation=True,
        )
        for e in errs[:3]:
            print(f"  cleanup warning ({Path(job_dir).name}): {e}")
    except Exception as e:
        print(f"  cleanup error ({Path(job_dir).name}): {e}")


def _write_prefailure_marker(
    job_dir: str,
    job: JobRecord,
    error_message: str | None,
) -> None:
    """Write the do-not-rerun marker before the DB is flipped to FAILED.

    Crash-safety: if SIGTERM lands between this call and the FAILED commit,
    the marker on disk still prevents resubmission via the existing submit
    guard. The full ``_purge_failed_job`` (called after the commit) overwrites
    this marker with richer failure metadata; here we only need enough to
    identify the job and block reruns.
    """
    metadata: dict[str, str | int | None] = {
        "orig_index": job.orig_index,
        "elements": job.elements,
        "charge": job.charge,
        "spin": job.spin,
        "fail_count": (job.fail_count or 0) + 1,
        "error_message": error_message,
    }
    try:
        _write_marker_file(Path(job_dir).resolve(), metadata)
    except Exception as e:
        # Best-effort: never block the FAILED DB commit on marker write.
        print(f"  pre-failure marker write failed ({Path(job_dir).name}): {e}")


def _purge_failed_job_inline(
    job_dir: str,
    root_dir: Path,
    db_path: Path,
    job: JobRecord,
    error_message: str | None,
) -> None:
    """Purge a just-failed job_dir: write marker, delete contents.

    Reuses ``clean._purge_failed_job``; its TOCTOU DB re-check still applies
    and is satisfied because we wrote ``FAILED`` to the DB immediately before
    calling this. ``job.fail_count`` reflects the pre-increment value from the
    in-memory record; ``+1`` matches the SQL-side increment.
    """
    metadata: dict[str, str | int | None] = {
        "orig_index": job.orig_index,
        "elements": job.elements,
        "charge": job.charge,
        "spin": job.spin,
        "fail_count": (job.fail_count or 0) + 1,
        "error_message": error_message,
    }
    try:
        _matched, _freed, errs = _purge_failed_job(
            Path(job_dir).resolve(),
            root_dir.resolve(),
            db_path,
            job.id,
            execute=True,
            job_metadata=metadata,
        )
        for e in errs[:3]:
            print(f"  purge warning ({Path(job_dir).name}): {e}")
    except Exception as e:
        print(f"  purge error ({Path(job_dir).name}): {e}")


def prepare_job_directory(
    job_record,
    root_dir: Path,
    job_dir_pattern: str = DEFAULT_JOB_DIR_PATTERN,
    orca_config: OrcaConfig | None = None,
    n_cores: int = 4,
    setup_func: Callable | None = None,
    return_full_path: bool = True,
    force_root_dir: bool = False,
) -> Path:
    """Create a job directory and prepare ORCA input files.

    Args:
        job_record: JobRecord from the workflow database.
        root_dir: Root directory where job directories will be created.
        job_dir_pattern: Pattern for job directory names. Supports
            {hostname}, {orig_index}, and {id}.
        orca_config: ORCA calculation configuration.
        n_cores: Number of CPU cores for ORCA.
        setup_func: Optional function to set up additional files. Called with
                   (job_dir, job_record) as arguments.
        return_full_path: If True, return full path to job directory. If False, return relative path.
        force_root_dir: If True, always construct job_dir from root_dir
            instead of using the path stored in the database. Useful when
            the database has been moved to a different location.

    Returns:
        Path to the created job directory.
    """
    if job_record.job_dir and not force_root_dir:
        job_dir = Path(job_record.job_dir)
    else:
        job_dir_name = render_job_dir_pattern(
            job_dir_pattern,
            orig_index=job_record.orig_index,
            job_id=job_record.id,
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

        optimizer = config.get("optimizer")
        # ORCA native opt: set opt=True so get_orca_blocks adds "Opt"/"TightOpt"/etc.
        # Sella opt: set opt=False so get_orca_blocks adds "EnGrad" (Sella drives optimization)
        use_orca_opt = optimizer == "orca"

        orcasimpleinput, orcablocks_list = write_orca_inputs(
            atoms=atoms,
            output_directory=str(job_dir),
            charge=charge,
            mult=mult,
            nbo=config.get("nbo", False),
            mbis=config.get("mbis", False),
            diis_option=config.get("diis_option"),
            cores=n_cores,
            opt=use_orca_opt,
            opt_level=config.get("opt_level", "normal"),
            functional=config.get("functional", "wB97M-V"),
            simple_input=config.get("simple_input", "omol"),
            orca_path=config.get("orca_path"),
            scf_MaxIter=config.get("scf_MaxIter"),
            actinide_basis=config.get("actinide_basis", "ma-def-TZVP"),
            actinide_ecp=config.get("actinide_ecp"),
            non_actinide_basis=config.get("non_actinide_basis", "def2-TZVPD"),
            ks_method=config.get("ks_method"),
            mem_per_job_mb=config.get("mem_per_job_mb"),
        )

        # For Sella jobs: generate the runner shim script
        if optimizer == "sella":
            orcablocks_str = "\n".join(orcablocks_list)

            max_steps = config.get("max_opt_steps")
            if max_steps is None:
                max_steps = 100
            write_sella_runner_shim(
                outputdir=job_dir,
                charge=charge,
                mult=mult,
                orcasimpleinput=orcasimpleinput,
                orcablocks=orcablocks_str,
                fmax=config.get("fmax", 0.05),
                max_steps=max_steps,
                orca_cmd=config.get("orca_path", "orca"),
                save_all_steps=config.get("save_all_steps", False),
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
    optimizer: str | None = None,
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
        optimizer: Optimizer engine ("orca", "sella", or None for single-point).

    Returns:
        Path to the created flux job file.
    """
    # Use absolute path so job runs in correct directory regardless of submission location
    job_dir_abs = job_dir.resolve()
    flux_script = job_dir_abs / "flux_job.flux"

    # Determine run command based on optimizer
    if optimizer == "sella":
        run_cmd = "python run_sella.py > sella_driver.log 2>&1\n"
    else:
        run_cmd = f"{orca_path} {input_file}\n"

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
        run_cmd,
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
    optimizer: str | None = None,
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
        ld_library_path: Override LD_LIBRARY_PATH.
        optimizer: Optimizer engine ("orca", "sella", or None for single-point).

    Returns:
        Path to the created SLURM job file.
    """
    # Use absolute path so job runs in correct directory regardless of submission location
    job_dir_abs = job_dir.resolve()
    slurm_script = job_dir_abs / "slurm_job.sh"

    # Determine run command based on optimizer
    if optimizer == "sella":
        run_cmd = "python run_sella.py > sella_driver.log 2>&1\n"
    else:
        run_cmd = f"{orca_path} {input_file}\n"

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
        *(
            [
                f"export LD_LIBRARY_PATH={ld_library_path or DEFAULT_LD_LIBRARY_PATHS['slurm']}:$LD_LIBRARY_PATH\n"
            ]
            if (ld_library_path or DEFAULT_LD_LIBRARY_PATHS["slurm"])
            else []
        ),
        run_cmd,
    ]

    with open(slurm_script, "w") as f:
        f.writelines(lines)

    # Make executable
    slurm_script.chmod(0o755)

    return slurm_script


def write_slurm_sandia_job_file(
    job_dir: Path,
    n_cores: int = SANDIA_DEFAULT_NTASKS_PER_NODE,
    n_hours: int = 48,
    qos: str = SANDIA_DEFAULT_QOS,
    partition: str = SANDIA_DEFAULT_PARTITION,
    account: str = SANDIA_DEFAULT_ACCOUNT,
    orca_path: str | None = None,
    openmpi_module: str = SANDIA_DEFAULT_OPENMPI_MODULE,
    input_file: str = "orca.inp",
    optimizer: str | None = None,
    job_name: str = "orca",
) -> Path:
    """Write a SLURM job submission script for Sandia CTS1/TLCC2 systems.

    Differs from ``write_slurm_job_file`` in three ways:

    - Uses ``module load`` for OpenMPI instead of ``conda activate``.
    - Sets ``OMPI_MCA_*`` env vars to force TCP/vader transport (PSM2/Omni-Path
      can misroute on some Sandia partitions; can be re-enabled later if
      benchmarks show TCP is the bottleneck).
    - Uses ``--partition`` instead of ``--constraint`` and derives
      ``LD_LIBRARY_PATH`` from ``mpirun`` at job runtime.

    Args:
        job_dir: Directory where the job file will be written.
        n_cores: Tasks per node. CTS1 nodes have 36 cores; TLCC2 have 16.
        n_hours: Wall-time in hours.
        qos: SLURM QOS (long, large, priority, normal).
        partition: SLURM partition.
        account: SLURM account.
        orca_path: Absolute path to the ORCA executable (the binary itself, not
            its parent directory).
        openmpi_module: ``module load`` argument matching ORCA's shared build.
        input_file: ORCA input file name.
        optimizer: Optimizer engine ("orca", "sella", or None for single-point).
        job_name: Slurm job name.

    Returns:
        Path to the created SLURM job file.

    Raises:
        ValueError: when ``orca_path`` is not provided. Sandia has no shared
            ORCA install; each user must point at their own shared-build
            binary.
    """
    if orca_path is None:
        raise ValueError(
            "write_slurm_sandia_job_file requires orca_path: "
            "Sandia has no shared ORCA install. Pass --orca-path on the CLI "
            "or set orca_path in your launch script."
        )

    job_dir_abs = job_dir.resolve()
    slurm_script = job_dir_abs / "slurm_job.sh"

    if optimizer == "sella":
        run_cmd = "python run_sella.py > sella_driver.log 2>&1\n"
    else:
        run_cmd = f"{orca_path} {input_file}\n"

    lines = [
        "#!/bin/bash\n",
        f"#SBATCH --job-name={job_name}\n",
        "#SBATCH --nodes=1\n",
        f"#SBATCH --ntasks-per-node={n_cores}\n",
        f"#SBATCH --time={n_hours}:00:00\n",
        f"#SBATCH --qos={qos}\n",
        f"#SBATCH --output={job_dir_abs}/orca_%j.out\n",
        f"#SBATCH --error={job_dir_abs}/orca_%j.err\n",
        f"#SBATCH --account={account}\n",
        f"#SBATCH --partition={partition}\n",
        "\n",
        "# Load the aue OpenMPI that matches ORCA's shared build\n",
        f"module load {openmpi_module}\n",
        "\n",
        "# Point to the correct libmpi at runtime\n",
        "export MPI_ROOT=$(dirname $(dirname $(which mpirun)))\n",
        "export LD_LIBRARY_PATH=${MPI_ROOT}/lib:$LD_LIBRARY_PATH\n",
        "\n",
        "# MPI transport: force TCP/vader; PSM2/Omni-Path can misroute on some partitions\n",
        "export OMPI_MCA_pml=ob1\n",
        "export OMPI_MCA_mtl=^psm2\n",
        "export OMPI_MCA_btl=tcp,self,vader\n",
        "\n",
        f"cd {job_dir_abs}\n",
        run_cmd,
    ]

    with open(slurm_script, "w") as f:
        f.writelines(lines)

    slurm_script.chmod(0o755)

    return slurm_script


def write_slurm_drac_job_file(
    job_dir: Path,
    n_cores: int = 4,
    n_hours: int = 2,
    account: str = "def-someuser",
    module_load: str = DRAC_DEFAULT_MODULE_LOAD,
    orca_bin: str = DRAC_DEFAULT_ORCA_BIN,
    venv_path: str | None = None,
    mem_per_cpu: str | None = None,
    input_file: str = "orca.inp",
    optimizer: str | None = None,
    job_name: str = "orca",
) -> Path:
    """Write a SLURM job script for Digital Research Alliance of Canada clusters.

    Differs from the default and Sandia writers:

    - ORCA comes from the module system: the script ``module load``s the full
      chain and invokes ``$EBROOTORCA/orca`` (never ``srun``/``mpirun`` -- ORCA
      spawns its own MPI from ``%pal nprocs``).
    - No ``--qos``, ``--partition``, or ``--constraint``. DRAC assigns the
      partition automatically from requested time and memory and exposes no
      user-selectable QOS, so emitting any of these breaks submission.
    - No OMPI MCA overrides. DRAC's InfiniBand uses OpenMPI defaults; the Sandia
      TCP/PSM2 settings would force slow TCP transport here.
    - Activates a virtualenv only when ``venv_path`` is given. The Sella
      optimizer path (``python run_sella.py``) requires it; plain ORCA runs need
      no Python environment.

    Args:
        job_dir: Directory where the job file will be written.
        n_cores: Tasks per node; must match ``%pal nprocs`` in the input file.
        n_hours: Wall-time in hours. Keep tight -- jobs under 3h ride backfill
            on the largest node pool.
        account: SLURM account / RAP, e.g. ``def-<pi>`` (a Default RAP keeps the
            fairshare hit off the sponsor's RAC priority).
        module_load: ``module load`` argument chain ending in the ORCA module.
            For the Sella path, include the python module the venv was built
            against.
        orca_bin: ORCA executable; defaults to the module-provided
            ``$EBROOTORCA/orca`` (shell-expanded at runtime after ``module load``).
        venv_path: virtualenv to activate (built by ``examples/drac/setup_venv.sh``).
            Required for the Sella optimizer; omit for plain ORCA.
        mem_per_cpu: Optional ``--mem-per-cpu`` value (e.g. ``"3900M"``). Sized
            at or below the node's MB-per-core ratio to avoid core-equivalent
            inflation.
        input_file: ORCA input file name.
        optimizer: Optimizer engine ("orca", "sella", or None for single-point).
        job_name: SLURM job name.

    Returns:
        Path to the created SLURM job file.

    Raises:
        ValueError: when ``optimizer == "sella"`` but no ``venv_path`` is given;
            the Sella driver needs the oact_utilities virtualenv.
    """
    if optimizer == "sella" and not venv_path:
        raise ValueError(
            "write_slurm_drac_job_file requires venv_path when optimizer='sella': "
            "the Sella driver runs `python run_sella.py` and needs the "
            "oact_utilities virtualenv. Pass --venv-path."
        )

    job_dir_abs = job_dir.resolve()
    slurm_script = job_dir_abs / "slurm_job.sh"

    if optimizer == "sella":
        run_cmd = "python run_sella.py > sella_driver.log 2>&1\n"
    else:
        run_cmd = f"{orca_bin} {input_file}\n"

    lines = [
        "#!/bin/bash\n",
        f"#SBATCH --job-name={job_name}\n",
        "#SBATCH --nodes=1\n",
        f"#SBATCH --ntasks-per-node={n_cores}\n",
        f"#SBATCH --time={n_hours}:00:00\n",
        f"#SBATCH --account={account}\n",
        *([f"#SBATCH --mem-per-cpu={mem_per_cpu}\n"] if mem_per_cpu else []),
        f"#SBATCH --output={job_dir_abs}/orca_%j.out\n",
        f"#SBATCH --error={job_dir_abs}/orca_%j.err\n",
        "\n",
        # Modules first (ORCA + its pinned MPI), then the venv -- never the
        # reverse (loading modules into an active venv breaks Lmod paths).
        f"module load {module_load}\n",
        *([f"source {venv_path}/bin/activate\n"] if venv_path else []),
        "\n",
        f"cd {job_dir_abs}\n",
        # ORCA spawns its own MPI from %pal nprocs; call by full path.
        run_cmd,
    ]

    with open(slurm_script, "w") as f:
        f.writelines(lines)

    slurm_script.chmod(0o755)

    return slurm_script


def _filter_marker_jobs(
    jobs: list,
    root_dir: Path,
    job_dir_pattern: str,
    workflow: ArchitectorWorkflow,
    force_root_dir: bool = False,
) -> list:
    """Filter out jobs that have a .do_not_rerun.json marker file.

    Checks the job directory (from DB or pattern-based path) for the marker.
    Jobs with markers are batch-updated to FAILED status.

    Args:
        jobs: List of JobRecord objects to filter.
        root_dir: Root directory for job directories.
        job_dir_pattern: Pattern for job directory names. Supports
            {hostname}, {orig_index}, and {id}.
        workflow: ArchitectorWorkflow instance for DB updates.
        force_root_dir: If True, skip DB job_dir and always use
            root_dir + pattern for lookups.

    Returns:
        Filtered list of jobs (without marker-blocked ones).
    """
    skip_ids: list[int] = []
    clean_jobs = []

    for job in jobs:
        # Try DB job_dir first, then pattern-based path
        marker_found = False
        if job.job_dir and not force_root_dir:
            marker_found = is_marker_blocked(Path(job.job_dir))
        if not marker_found:
            pattern = render_job_dir_pattern(
                job_dir_pattern,
                orig_index=job.orig_index,
                job_id=job.id,
            )
            marker_found = is_marker_blocked(root_dir / pattern)

        if marker_found:
            skip_ids.append(job.id)
        else:
            clean_jobs.append(job)

    if skip_ids:
        workflow.update_status_bulk(
            skip_ids,
            JobStatus.FAILED,
            increment_fail_count=True,
            error_message=MARKER_ERROR_MESSAGE,
        )
        print(f"Skipped {len(skip_ids)} jobs due to .do_not_rerun.json marker")

    return clean_jobs


def _skip_finished_on_disk(
    jobs: list,
    root_dir: Path,
    job_dir_pattern: str,
    workflow: ArchitectorWorkflow,
    hours_cutoff: float = 168,
    force_root_dir: bool = False,
) -> list:
    """Filter out jobs that already completed or failed on disk.

    Checks each candidate job's output directory before submission. Jobs
    found completed are auto-updated to COMPLETED in the DB. Jobs found
    failed are auto-updated to FAILED. This prevents re-submitting jobs
    whose results would be overwritten by prepare_job_directory().

    Uses the same dual-lookup pattern as _filter_marker_jobs: tries
    job.job_dir first, then falls back to constructing the path from
    job_dir_pattern.

    Args:
        jobs: List of JobRecord objects to filter.
        root_dir: Root directory for job directories.
        job_dir_pattern: Pattern for job directory names.
        workflow: ArchitectorWorkflow instance for DB updates.
        hours_cutoff: Hours threshold for timeout detection in
            check_job_termination. Defaults to 168 (1 week) to avoid
            false timeout classifications on idle directories.
        force_root_dir: If True, skip DB job_dir and always use
            root_dir + pattern for lookups.

    Returns:
        Filtered list of jobs (without completed/failed ones).
    """
    completed_ids: list[int] = []
    failed_ids: list[int] = []
    failed_errors: dict[int, str] = {}
    clean_jobs = []

    for job in jobs:
        # Resolve directory: try DB job_dir first, then pattern-based path
        job_dir = None
        if job.job_dir and not force_root_dir and Path(job.job_dir).is_dir():
            job_dir = job.job_dir
        else:
            pattern = render_job_dir_pattern(
                job_dir_pattern,
                orig_index=job.orig_index,
                job_id=job.id,
            )
            candidate = root_dir / pattern
            if candidate.is_dir():
                job_dir = str(candidate)

        if not job_dir:
            clean_jobs.append(job)
            continue

        status = check_job_termination(job_dir, hours_cutoff=hours_cutoff)

        if status == 1:
            completed_ids.append(job.id)
        elif status == -1:
            failed_ids.append(job.id)
            try:
                log_file = pull_log_file(job_dir)
                reason = parse_failure_reason(log_file)
                if reason:
                    failed_errors[job.id] = reason
            except (FileNotFoundError, OSError):
                pass
        else:
            # 0 (running/unknown) or -2 (timeout): submit normally
            clean_jobs.append(job)

    # Batch DB updates
    if completed_ids:
        workflow.update_status_bulk(completed_ids, JobStatus.COMPLETED)
        print(
            f"Skipped {len(completed_ids)} jobs already completed on disk "
            "(updated DB to COMPLETED)"
        )

    if failed_ids:
        # Failed jobs need per-job error messages for the ones we extracted
        for jid in failed_ids:
            error_msg = failed_errors.get(
                jid, "Failed on disk (detected at submission)"
            )
            workflow.update_status(
                jid,
                JobStatus.FAILED,
                error_message=error_msg,
                increment_fail_count=True,
            )
        print(
            f"Skipped {len(failed_ids)} jobs already failed on disk "
            "(updated DB to FAILED)"
        )

    return clean_jobs


def filter_jobs_for_submission(
    workflow: ArchitectorWorkflow,
    num_jobs: int,
    max_fail_count: int | None = None,
    randomize: bool = True,
    max_atoms: int | None = None,
    min_atoms: int | None = None,
) -> list:
    """Filter jobs that are ready to submit.

    Args:
        workflow: ArchitectorWorkflow instance
        num_jobs: Number of jobs to return
        max_fail_count: Skip jobs with fail_count >= this value
        max_atoms: If set, skip jobs with natoms > this value. Jobs above the
            cap are left in their current status for a later batch. Rows with a
            NULL natoms are excluded.
        min_atoms: If set, skip jobs with natoms < this value. Jobs below the
            floor are left for another lane. With max_atoms this selects the
            closed band [min_atoms, max_atoms].

    Returns:
        List of JobRecords ready for submission
    """
    # Get ready jobs (DB is source of truth)
    # include_geometry=True so prepare_job_directory can write the .inp file
    ready_jobs = workflow.get_jobs_by_status(JobStatus.TO_RUN, include_geometry=True)

    # Apply fail_count filter if specified
    if max_fail_count is not None:
        original_count = len(ready_jobs)
        ready_jobs = [j for j in ready_jobs if j.fail_count < max_fail_count]
        skipped = original_count - len(ready_jobs)
        if skipped > 0:
            print(f"Skipped {skipped} jobs with fail_count >= {max_fail_count}")

    # Apply atom-count band if specified. Jobs outside [min_atoms, max_atoms]
    # are left untouched for another lane. A NULL natoms is excluded rather
    # than raising, so a malformed row cannot break submission. Over-cap,
    # under-floor and missing-natoms skips are reported separately so the
    # counts are accurate.
    if max_atoms is not None or min_atoms is not None:
        kept = []
        over_cap = 0
        under_floor = 0
        missing = 0
        for j in ready_jobs:
            if j.natoms is None:
                missing += 1
            elif max_atoms is not None and j.natoms > max_atoms:
                over_cap += 1
            elif min_atoms is not None and j.natoms < min_atoms:
                under_floor += 1
            else:
                kept.append(j)
        ready_jobs = kept
        if over_cap > 0:
            print(f"Skipped {over_cap} jobs with natoms > {max_atoms}")
        if under_floor > 0:
            print(f"Skipped {under_floor} jobs with natoms < {min_atoms}")
        if missing > 0:
            print(f"Skipped {missing} jobs with missing (NULL) natoms")

    # Limit to requested count
    if randomize:
        jobs_to_submit = random.sample(ready_jobs, min(num_jobs, len(ready_jobs)))
    else:
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
        timeout_seconds: int = 7200,
    ) -> dict:
        """Execute ORCA (or Sella) job within Parsl worker.

        This runs as a Parsl python_app, executing directly on the worker node.
        Parsl handles CPU affinity and worker management automatically.

        When optimizer="sella", runs ``python run_sella.py`` instead of ORCA
        directly. The Sella shim script is generated by prepare_job_directory().

        ORCA output is written to files (orca.out / orca.err) instead of being
        captured via pipes.  Pipe-based capture can deadlock when ORCA's
        internal MPI processes fill the OS pipe buffer (~64 KB) during the
        atomic SCF initial-guess phase.

        Each worker also gets a private TMPDIR so that concurrent ORCA instances
        on the same node don't collide on temporary files or OpenMPI
        shared-memory segments.

        Core count is already baked into orca.inp by prepare_job_directory().

        Args:
            job_id: Workflow database job ID
            job_dir: Absolute path to job directory
            orca_config: ORCA configuration dictionary
            timeout_seconds: Job timeout in seconds (default: 7200 = 2 hours).

        Returns:
            Dict with job_id, status, metrics
        """
        import os
        import signal
        import subprocess
        import tempfile
        import time
        from pathlib import Path

        job_dir_path = Path(job_dir)
        optimizer = orca_config.get("optimizer")

        # Determine command based on optimizer
        if optimizer == "sella":
            sella_script = job_dir_path / "run_sella.py"
            if not sella_script.exists():
                return {
                    "job_id": job_id,
                    "status": "failed",
                    "error": f"Sella script not found: {sella_script}",
                }
            cmd = ["python", str(sella_script)]
            stdout_path = job_dir_path / "sella_driver.log"
            stderr_path = job_dir_path / "sella_driver.err"
        else:
            input_file = job_dir_path / "orca.inp"
            if not input_file.exists():
                return {
                    "job_id": job_id,
                    "status": "failed",
                    "error": f"Input file not found: {input_file}",
                }
            orca_cmd = orca_config.get("orca_path", "orca")
            cmd = [orca_cmd, str(input_file)]
            stdout_path = job_dir_path / "orca.out"
            stderr_path = job_dir_path / "orca.err"

        # --- Environment isolation for concurrent ORCA instances ---
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"

        # Give each worker a private TMPDIR inside the job directory so that
        # ORCA's temp files (orca_atom*.out/.gbw, MPI shared-memory segments)
        # don't collide between concurrent jobs on the same node.
        tmp_dir = tempfile.mkdtemp(prefix="orca_tmp_", dir=job_dir)
        env["TMPDIR"] = tmp_dir

        # Prevent OpenMPI from using shared-memory transport between unrelated
        # ORCA instances that happen to share the same node.  vader (or sm)
        # uses /dev/shm files whose names can collide.
        env["OMPI_MCA_btl"] = "self,tcp"

        # Restrict ORCA's mpirun to the local node only.  In multi-node
        # SLURM blocks, mpirun would otherwise read the SLURM hostfile and
        # place processes on neighboring nodes where different ORCA instances
        # are running -- silently corrupting results.  These env vars are
        # set unconditionally: harmless on single-node / Flux, protective
        # on multi-node SLURM.
        # DO NOT REMOVE -- see docs/plans/2026-03-16-feat-multi-node-slurm-parsl-support-plan.md
        import socket

        local_hostname = os.environ.get("SLURMD_NODENAME", socket.gethostname())
        env["OMPI_MCA_orte_default_hostfile"] = "/dev/null"
        env["SLURM_NODELIST"] = local_hostname
        env["SLURM_NNODES"] = "1"

        start_time = time.time()

        proc = None
        try:
            # Write output directly to files to avoid pipe buffer deadlocks.
            # start_new_session=True puts the process + children in a new
            # process group so we can kill the entire tree on timeout.
            f_out = open(stdout_path, "w")
            f_err = open(stderr_path, "w")

            proc = subprocess.Popen(
                cmd,
                cwd=job_dir,
                stdout=f_out,
                stderr=f_err,
                env=env,
                start_new_session=True,
            )

            try:
                proc.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                # Kill the entire process group (ORCA + mpirun + orca_main workers)
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait()
                return {
                    "job_id": job_id,
                    "status": "timeout",
                    "error": f"Job exceeded {timeout_seconds}s timeout",
                }
            finally:
                f_out.close()
                f_err.close()

            elapsed = time.time() - start_time

            if proc.returncode == 0:
                # For Sella jobs, check sella_status.txt — the process can
                # exit 0 even when the optimization did not converge.
                if optimizer == "sella":
                    status_path = job_dir_path / "sella_status.txt"
                    try:
                        status_text = status_path.read_text()
                        if "NOT_CONVERGED" in status_text or "ERROR" in status_text:
                            return {
                                "job_id": job_id,
                                "status": "failed",
                                "error": "Sella optimization did not converge",
                                "wall_time": elapsed,
                            }
                    except OSError:
                        # Status file missing/unreadable after clean exit —
                        # something unexpected happened; fail safe.
                        return {
                            "job_id": job_id,
                            "status": "failed",
                            "error": "Sella status file not found after process exit",
                            "wall_time": elapsed,
                        }
                else:
                    # ORCA can exit 0 even when an MPI child process
                    # (e.g. orca_leanscf_mpi) fails with "aborting the
                    # run".  Verify the output file actually shows normal
                    # termination before trusting the return code.
                    from collections import deque

                    out_path = job_dir_path / "orca.out"
                    if out_path.exists():
                        try:
                            with open(out_path, errors="replace") as fh:
                                tail = deque(fh, maxlen=10)
                            has_normal = any(
                                "ORCA TERMINATED NORMALLY" in ln for ln in tail
                            )
                            has_abort = any(
                                "aborting the run" in ln or "Error" in ln for ln in tail
                            )
                            if not has_normal and has_abort:
                                err_tail = "".join(tail).strip()[-300:]
                                return {
                                    "job_id": job_id,
                                    "status": "failed",
                                    "error": (
                                        "ORCA exited 0 but output shows "
                                        f"error: {err_tail}"
                                    ),
                                    "wall_time": elapsed,
                                }
                        except OSError:
                            pass  # Fall through to completed

                return {
                    "job_id": job_id,
                    "status": "completed",
                    "wall_time": elapsed,
                }
            else:
                # Read tail of stderr for error reporting
                err_tail = ""
                try:
                    err_tail = stderr_path.read_text()[-500:]
                except Exception:
                    pass
                return {
                    "job_id": job_id,
                    "status": "failed",
                    "error": f"ORCA exited with code {proc.returncode}",
                    "stderr": err_tail,
                }

        except Exception as e:
            # Kill process tree if it's still alive
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    proc.wait()
                except Exception:
                    pass
            return {
                "job_id": job_id,
                "status": "failed",
                "error": str(e),
            }
        finally:
            # NOTE: we intentionally keep tmp_dir (orca_tmp_*/) around — it
            # contains atomic SCF guess files that are useful for debugging
            # failed jobs and for ORCA restarts.
            pass


def _build_parsl_run_dir() -> str:
    """Create a unique Parsl run directory under ``runinfo/``."""
    return f"runinfo/run_{os.getpid()}_{int(time.time())}"


def _get_parsl_runtime_address() -> str:
    """Resolve a runtime-reachable address for Parsl workers."""
    from parsl.addresses import address_by_hostname

    try:
        return address_by_hostname()
    except OSError:
        # Fallback for environments where hostname is not resolvable (e.g. local dev)
        return "127.0.0.1"


def _get_monitoring_hub_class():
    """Import MonitoringHub across Parsl versions."""
    try:
        from parsl.monitoring.monitoring import MonitoringHub
    except ImportError:
        from parsl.monitoring import MonitoringHub

    return MonitoringHub


def _build_parsl_monitoring(run_dir: str, hub_address: str):
    """Create a MonitoringHub that writes to a per-run monitoring database."""
    MonitoringHub = _get_monitoring_hub_class()

    resolved_run_dir = Path(run_dir).resolve()
    resolved_run_dir.mkdir(parents=True, exist_ok=True)

    monitoring_db = resolved_run_dir / "monitoring.db"
    kwargs = {
        "hub_address": hub_address,
        "monitoring_debug": False,
        "resource_monitoring_interval": 10,
        "logging_endpoint": f"sqlite:///{monitoring_db}",
    }

    # Older Parsl releases accept logdir explicitly; newer ones infer it
    # via MonitoringHub.start(dfk_run_dir, config_run_dir) and reject the arg.
    if "logdir" in inspect.signature(MonitoringHub).parameters:
        kwargs["logdir"] = str(resolved_run_dir)

    return MonitoringHub(
        **kwargs,
    )


def _build_parsl_worker_init(
    scheduler: str,
    conda_env: str,
    conda_base: str,
    ld_library_path: str | None = None,
    orca_path: str | None = None,
    mpirun_path: str | None = None,
) -> str:
    """Build worker initialization commands for Parsl-launched workers."""
    ld_lib = ld_library_path
    if ld_lib is None:
        if scheduler == "flux":
            ld_lib = DEFAULT_LD_LIBRARY_PATHS.get("flux", "")
        elif scheduler in {"slurm", "pbspro"}:
            ld_lib = f"{conda_base}/envs/{conda_env}/lib"
        else:
            raise ValueError(f"Unknown scheduler: {scheduler!r}")

    orca_bin_dir = (
        os.path.dirname(orca_path) if orca_path and os.path.isabs(orca_path) else None
    )
    resolved_mpirun = mpirun_path or shutil.which("mpirun")
    mpirun_bin_dir = (
        os.path.dirname(resolved_mpirun)
        if resolved_mpirun and os.path.isabs(resolved_mpirun)
        else None
    )
    path_entries = [
        entry for entry in [mpirun_bin_dir, orca_bin_dir] if entry is not None
    ]
    path_entries = list(dict.fromkeys(path_entries))
    path_line = f"export PATH={':'.join(path_entries)}:$PATH\n" if path_entries else ""
    ld_line = f"export LD_LIBRARY_PATH={ld_lib}:$LD_LIBRARY_PATH\n" if ld_lib else ""

    return (
        "source ~/.bashrc\n"
        f"conda activate {conda_env}\n"
        f"{path_line}"
        f"{ld_line}"
        "export JAX_PLATFORMS=cpu\n"
        "export OMP_NUM_THREADS=1\n"
    )


def _build_parsl_sandia_worker_init(
    orca_path: str | None = None,
) -> str:
    """Build minimal worker_init for Parsl workers on Sandia CTS1/TLCC2.

    LocalProvider forks workers from the coordinator's shell, so they inherit
    the module-loaded OpenMPI, ``OMPI_MCA_*`` settings, and ``LD_LIBRARY_PATH``
    that the user prepared in their salloc'd shell before launching python.
    No ``module load`` here -- doing it in a non-login subprocess shell can
    fail (Lmod's ``module`` shell function isn't always defined) and risks
    polluting ``LD_LIBRARY_PATH`` with duplicates.

    User contract: launch python from inside an existing SLURM allocation,
    e.g. ``salloc -N1 -p attaway -A fy250086 -t 8:00:00`` followed by
    ``module load aue/openmpi/4.1.6-gcc-12.3.0`` and the OMPI MCA exports.
    """
    orca_bin_dir = (
        os.path.dirname(orca_path) if orca_path and os.path.isabs(orca_path) else None
    )
    path_line = (
        f"export PATH={orca_bin_dir}:$PATH\n" if orca_bin_dir is not None else ""
    )

    return f"{path_line}" "export JAX_PLATFORMS=cpu\n" "export OMP_NUM_THREADS=1\n"


def _build_parsl_sandia_worker_init_multi_block(
    orca_path: str | None = None,
    openmpi_module: str = SANDIA_DEFAULT_OPENMPI_MODULE,
    ld_library_path: str | None = None,
) -> str:
    """Build worker_init for Parsl SlurmProvider blocks on Sandia CTS1/TLCC2.

    Each Parsl block is its own ``sbatch`` allocation, so the manager process
    starts in a fresh non-login shell with no modules loaded. This worker_init
    must do the full bootstrap itself: ``module load`` OpenMPI, derive
    ``MPI_ROOT`` from ``which mpirun`` (only resolves after the module load),
    set ``LD_LIBRARY_PATH``, and export the Sandia-specific OMPI MCA settings
    that disable PSM2/Omni-Path on misrouting partitions.

    Order matters -- ``MPI_ROOT=$(dirname $(dirname $(which mpirun)))`` only
    resolves after ``module load``.

    Args:
        orca_path: Absolute path to the ORCA executable. Its parent directory
            is prepended to ``PATH``.
        openmpi_module: ``module load`` argument matching ORCA's shared build.
        ld_library_path: Override for ``LD_LIBRARY_PATH``. When given, prepended
            verbatim (no ``$MPI_ROOT`` interpolation). When omitted, the default
            ``${MPI_ROOT}/lib`` is used.

    Returns:
        Multi-line shell script for use as ``worker_init``.
    """
    if ld_library_path is not None:
        ld_line = f"export LD_LIBRARY_PATH={ld_library_path}:$LD_LIBRARY_PATH\n"
    else:
        ld_line = "export LD_LIBRARY_PATH=${MPI_ROOT}/lib:$LD_LIBRARY_PATH\n"

    mca_lines = "".join(
        f"export {key}={value}\n" for key, value in SANDIA_DEFAULT_OMPI_MCA.items()
    )

    orca_bin_dir = (
        os.path.dirname(orca_path) if orca_path and os.path.isabs(orca_path) else None
    )
    path_line = (
        f"export PATH={orca_bin_dir}:$PATH\n" if orca_bin_dir is not None else ""
    )

    return (
        f"module load {openmpi_module}\n"
        "export MPI_ROOT=$(dirname $(dirname $(which mpirun)))\n"
        f"{ld_line}"
        f"{mca_lines}"
        f"{path_line}"
        "export JAX_PLATFORMS=cpu\n"
        "export OMP_NUM_THREADS=1\n"
    )


def build_parsl_config_sandia_local(
    max_workers: int = 3,
    cores_per_worker: int = 12,
    orca_path: str | None = None,
    enable_monitoring: bool = True,
):
    """Build Parsl Config for Sandia CTS1/TLCC2 single-node execution.

    Uses LocalProvider with a minimal worker_init. The coordinator must be
    launched from inside an existing SLURM allocation that has already
    ``module load``-ed OpenMPI and exported the OMPI_MCA env. LocalProvider
    forks workers from the coordinator shell, so they inherit that env.

    Args:
        max_workers: Concurrent workers per node. Defaults to 3 (12*3=36 cores
            on a CTS1 attaway node).
        cores_per_worker: CPU cores per worker.
        orca_path: Absolute path to the ORCA executable. Its parent directory
            is prepended to PATH on workers so ORCA's bundled helpers resolve
            without the user having to add it to PATH manually.
        enable_monitoring: Attach Parsl MonitoringHub.

    Returns:
        Parsl Config object.
    """
    if not PARSL_AVAILABLE:
        raise ImportError(
            "Parsl is not installed. Please install with: pip install 'parsl>=2024.1'"
        )

    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    from parsl.providers import LocalProvider

    runtime_address = _get_parsl_runtime_address()
    worker_init = _build_parsl_sandia_worker_init(
        orca_path=orca_path,
    )

    provider = LocalProvider(
        worker_init=worker_init,
    )

    executor = HighThroughputExecutor(
        label="sandia_local_htex",
        address=runtime_address,
        cores_per_worker=cores_per_worker,
        max_workers_per_node=max_workers,
        cpu_affinity="block",
        provider=provider,
    )

    run_dir = _build_parsl_run_dir()
    monitoring = (
        _build_parsl_monitoring(run_dir=run_dir, hub_address=runtime_address)
        if enable_monitoring
        else None
    )

    return Config(executors=[executor], run_dir=run_dir, monitoring=monitoring)


def _build_parsl_drac_worker_init(
    orca_path: str | None = None,
) -> str:
    """Build minimal worker_init for Parsl workers on DRAC single-node runs.

    Like the Sandia LocalProvider path, workers are forked from the coordinator
    shell, so they inherit whatever the launch script set up before starting
    python: the ORCA module chain (``module load StdEnv/... orca/6.1.0``) and the
    activated virtualenv. No ``module load`` here -- Lmod's ``module`` function is
    not reliably defined in a forked non-login shell, and re-loading risks
    duplicate ``LD_LIBRARY_PATH`` entries.

    User contract: launch python from inside a SLURM allocation that has already
    run ``module load <chain>`` and ``source <venv>/bin/activate`` -- see
    ``launch/run_parsl_single_node_drac.sh``.

    Args:
        orca_path: Resolved absolute ORCA path (``$EBROOTORCA/orca`` expanded by
            the launch script). Its parent dir is prepended to ``PATH`` so ORCA's
            bundled helpers resolve.
    """
    orca_bin_dir = (
        os.path.dirname(orca_path) if orca_path and os.path.isabs(orca_path) else None
    )
    path_line = (
        f"export PATH={orca_bin_dir}:$PATH\n" if orca_bin_dir is not None else ""
    )
    return f"{path_line}" "export JAX_PLATFORMS=cpu\n" "export OMP_NUM_THREADS=1\n"


def build_parsl_config_drac_local(
    max_workers: int = 4,
    cores_per_worker: int = 16,
    orca_path: str | None = None,
    enable_monitoring: bool = True,
):
    """Build Parsl Config for DRAC single-node execution (LocalProvider).

    One SLURM allocation occupies a whole node (or part of one) and Parsl packs
    ``max_workers`` ORCA jobs onto it concurrently, each using
    ``cores_per_worker`` cores. The coordinator must be launched from inside that
    allocation after ``module load``-ing the ORCA chain and activating the venv;
    LocalProvider forks workers from that shell so they inherit the env.

    Defaults (4 x 16 = 64 cores) suit a partial Fir node; for a whole 192-core
    node use e.g. ``max_workers=12, cores_per_worker=16``.

    Args:
        max_workers: Concurrent ORCA jobs on the node.
        cores_per_worker: Cores per ORCA job (matches ``%pal nprocs``).
        orca_path: Resolved absolute ORCA path (``$EBROOTORCA/orca``).
        enable_monitoring: Attach Parsl MonitoringHub.

    Returns:
        Parsl Config object.
    """
    if not PARSL_AVAILABLE:
        raise ImportError(
            "Parsl is not installed. Please install with: pip install 'parsl>=2024.1'"
        )

    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    from parsl.providers import LocalProvider

    runtime_address = _get_parsl_runtime_address()
    worker_init = _build_parsl_drac_worker_init(orca_path=orca_path)

    provider = LocalProvider(worker_init=worker_init)

    executor = HighThroughputExecutor(
        label="drac_local_htex",
        address=runtime_address,
        cores_per_worker=cores_per_worker,
        max_workers_per_node=max_workers,
        cpu_affinity="block",
        provider=provider,
    )

    run_dir = _build_parsl_run_dir()
    monitoring = (
        _build_parsl_monitoring(run_dir=run_dir, hub_address=runtime_address)
        if enable_monitoring
        else None
    )

    return Config(executors=[executor], run_dir=run_dir, monitoring=monitoring)


def build_parsl_config_sandia(
    max_workers: int = 3,
    cores_per_worker: int = 12,
    cpus_per_node: int | None = None,
    nodes_per_block: int = 1,
    max_blocks: int = 10,
    init_blocks: int = 2,
    min_blocks: int = 1,
    walltime_hours: int = 8,
    qos: str = SANDIA_DEFAULT_QOS,
    account: str = SANDIA_DEFAULT_ACCOUNT,
    partition: str = SANDIA_DEFAULT_PARTITION,
    openmpi_module: str = SANDIA_DEFAULT_OPENMPI_MODULE,
    orca_path: str | None = None,
    ld_library_path: str | None = None,
    enable_monitoring: bool = True,
):
    """Build Parsl Config for Sandia CTS1/TLCC2 multi-node execution.

    Uses ``SlurmProvider`` to auto-provision worker nodes. Each block is a
    SLURM job with ``nodes_per_block`` nodes; each node runs ``max_workers``
    ORCA workers concurrently. The coordinator process owns the workflow
    SQLite, so scaling out via blocks does not multiply DB writers (the
    failure mode that motivates this builder over per-job ``sbatch``).

    Args:
        max_workers: Concurrent workers per node.
        cores_per_worker: CPU cores per worker (must match ORCA nprocs).
        cpus_per_node: Scheduler CPU cores requested per node. When omitted,
            defaults to ``max_workers * cores_per_worker``. For multi-node,
            ``exclusive=True`` already reserves whole nodes; an explicit
            override adds an ``--ntasks-per-node`` request.
        nodes_per_block: Nodes per SLURM block. >1 selects ``SrunLauncher``.
        max_blocks: Maximum SLURM blocks Parsl will provision.
        init_blocks: Blocks requested at startup.
        min_blocks: Minimum blocks to keep alive.
        walltime_hours: Walltime per block allocation in hours.
        qos: SLURM QOS (Sandia values: normal/long/large/priority).
        account: SLURM account.
        partition: SLURM partition. Sandia partitions are mandatory; emitted
            into ``scheduler_options`` (LLNL uses ``--constraint`` instead).
        openmpi_module: ``module load`` argument for the OpenMPI build that
            matches ORCA's shared libraries.
        orca_path: Absolute path to the ORCA executable. Its parent directory
            is prepended to ``PATH`` on workers so ORCA's bundled MPI helpers
            resolve before any system mpirun.
        ld_library_path: Override for ``LD_LIBRARY_PATH`` on workers. When
            omitted, defaults to ``${MPI_ROOT}/lib`` derived at runtime from
            the loaded OpenMPI module.
        enable_monitoring: Attach Parsl ``MonitoringHub``.

    Returns:
        Parsl ``Config`` object.
    """
    if not PARSL_AVAILABLE:
        raise ImportError(
            "Parsl is not installed. Please install with: pip install 'parsl>=2024.1'"
        )

    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    from parsl.providers import SlurmProvider

    runtime_address = _get_parsl_runtime_address()
    worker_init = _build_parsl_sandia_worker_init_multi_block(
        orca_path=orca_path,
        openmpi_module=openmpi_module,
        ld_library_path=ld_library_path,
    )

    active_cores_per_node = max_workers * cores_per_worker
    requested_cpus_per_node = cpus_per_node or active_cores_per_node
    if requested_cpus_per_node < active_cores_per_node:
        raise ValueError(
            "cpus_per_node must be >= max_workers * cores_per_worker "
            f"({active_cores_per_node})"
        )

    # Sandia partitions are mandatory; LLNL's build_parsl_config_slurm uses
    # --constraint instead, so this line is the only structural divergence.
    partition_line = f"#SBATCH --partition={partition}\n"
    ntasks_lines = (
        f"#SBATCH --ntasks-per-node={requested_cpus_per_node}\n"
        f"#SBATCH --cpus-per-task=1\n"
    )

    if nodes_per_block > 1:
        from parsl.launchers import SrunLauncher

        launcher = SrunLauncher()
        if cpus_per_node is not None:
            scheduler_options = partition_line + ntasks_lines
        else:
            scheduler_options = partition_line
    else:
        from parsl.launchers import SimpleLauncher

        launcher = SimpleLauncher()
        scheduler_options = partition_line + ntasks_lines

    provider = SlurmProvider(
        qos=qos,
        account=account,
        nodes_per_block=nodes_per_block,
        init_blocks=init_blocks,
        min_blocks=min_blocks,
        max_blocks=max_blocks,
        walltime=f"{walltime_hours:02d}:00:00",
        worker_init=worker_init,
        scheduler_options=scheduler_options,
        exclusive=True,
        launcher=launcher,
        parallelism=1.0,
    )

    executor = HighThroughputExecutor(
        label="sandia_htex",
        address=runtime_address,
        cores_per_worker=cores_per_worker,
        max_workers_per_node=max_workers,
        cpu_affinity="block",
        provider=provider,
    )

    run_dir = _build_parsl_run_dir()
    monitoring = (
        _build_parsl_monitoring(run_dir=run_dir, hub_address=runtime_address)
        if enable_monitoring
        else None
    )

    return Config(executors=[executor], run_dir=run_dir, monitoring=monitoring)


def _build_parsl_htex_launch_cmd(python_executable: str | None = None) -> str:
    """Build an HTEX manager launch command using an absolute Python path.

    Parsl's default HTEX launch command relies on the ``process_worker_pool.py``
    console script being present on ``PATH``. That is brittle for OpenPBS
    fanout via ``pbsdsh``, where child tasks may not inherit the full activated
    PATH from the parent shell. Launching via ``python -m ...`` against the
    coordinator's active interpreter is more robust.

    The installed Parsl version is treated as the source of truth for launch
    string placeholders. That avoids hard-coding parameter names such as
    ``task_port`` that vary across Parsl releases.
    """
    from parsl.executors.high_throughput.executor import DEFAULT_LAUNCH_CMD

    python_cmd = shlex.quote(python_executable or sys.executable)
    entrypoint = f"{python_cmd} -m parsl.executors.high_throughput.process_worker_pool"

    if "process_worker_pool.py" in DEFAULT_LAUNCH_CMD:
        return DEFAULT_LAUNCH_CMD.replace("process_worker_pool.py", entrypoint, 1)

    return f"{entrypoint} {DEFAULT_LAUNCH_CMD}"


def build_parsl_config_flux(
    max_workers: int = 4,
    cores_per_worker: int = 16,
    conda_env: str = "py10mpi",
    conda_base: str = "/usr/WS1/vargas58/miniconda3",
    ld_library_path: str | None = None,
    orca_path: str | None = None,
    mpirun_path: str | None = None,
    enable_monitoring: bool = True,
):
    """Build Parsl Config for Flux single-node execution.

    Uses LocalProvider since Flux doesn't support scale-out in Parsl.
    This configuration runs all workers on the local node.

    Args:
        max_workers: Maximum number of concurrent workers.
        cores_per_worker: CPU cores per worker.
        conda_env: Conda environment name.
        conda_base: Conda base path.
        ld_library_path: Override LD_LIBRARY_PATH.
        orca_path: Absolute path to ORCA executable. When provided and
            absolute, its parent directory is prepended to PATH on worker
            processes so that ORCA's bundled MPI helpers are found before
            any conda-provided mpirun.
        mpirun_path: Absolute path to the desired mpirun executable. When
            provided, its parent directory is prepended to PATH on worker
            processes ahead of the ORCA bin directory.
        enable_monitoring: If True, attach a MonitoringHub that writes
            resource usage to monitoring.db under the run directory.

    Returns:
        Parsl Config object.
    """
    if not PARSL_AVAILABLE:
        raise ImportError(
            "Parsl is not installed. Please install with: pip install 'parsl>=2024.1'"
        )

    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    from parsl.providers import LocalProvider

    runtime_address = _get_parsl_runtime_address()
    worker_init = _build_parsl_worker_init(
        scheduler="flux",
        conda_env=conda_env,
        conda_base=conda_base,
        ld_library_path=ld_library_path,
        orca_path=orca_path,
        mpirun_path=mpirun_path,
    )

    provider = LocalProvider(
        worker_init=worker_init,
    )

    executor = HighThroughputExecutor(
        label="flux_htex",
        address=runtime_address,
        cores_per_worker=cores_per_worker,
        max_workers_per_node=max_workers,
        cpu_affinity="block",
        provider=provider,
    )

    run_dir = _build_parsl_run_dir()
    monitoring = (
        _build_parsl_monitoring(run_dir=run_dir, hub_address=runtime_address)
        if enable_monitoring
        else None
    )

    return Config(executors=[executor], run_dir=run_dir, monitoring=monitoring)


def build_parsl_config_slurm(
    max_workers: int = 4,
    cores_per_worker: int = 16,
    cpus_per_node: int | None = None,
    nodes_per_block: int = 1,
    max_blocks: int = 10,
    init_blocks: int = 2,
    min_blocks: int = 1,
    walltime_hours: int = 2,
    qos: str = "frontier",
    account: str = "ODEFN5169CYFZ",
    conda_env: str = "py10mpi",
    conda_base: str = "/usr/WS1/vargas58/miniconda3",
    ld_library_path: str | None = None,
    orca_path: str | None = None,
    mpirun_path: str | None = None,
    enable_monitoring: bool = True,
):
    """Build Parsl Config for SLURM multi-node execution.

    Uses SlurmProvider to auto-provision worker nodes via SLURM.
    Each block is a SLURM job with ``nodes_per_block`` nodes, and each
    node runs ``max_workers`` ORCA workers concurrently.

    When ``nodes_per_block > 1``, SrunLauncher is used so that Parsl
    distributes worker managers across all nodes in the block.  When
    ``nodes_per_block == 1`` (default), SimpleLauncher is used for
    backwards compatibility.

    Args:
        max_workers: Maximum concurrent workers per node.
        cores_per_worker: CPU cores per worker (must match ORCA nprocs).
        cpus_per_node: Scheduler CPU cores requested per node. When omitted,
            defaults to ``max_workers * cores_per_worker`` for the
            single-node path. For multi-node Slurm, exclusive allocations
            already reserve whole nodes; when this is set explicitly, an
            ``--ntasks-per-node`` request is added to the batch allocation.
        nodes_per_block: Nodes per SLURM block. >1 enables multi-node
            blocks with SrunLauncher.  Total capacity is
            ``max_blocks * nodes_per_block * max_workers``.
        max_blocks: Maximum number of SLURM blocks to provision.
        init_blocks: Number of blocks to request at startup.
        min_blocks: Minimum number of blocks to keep alive.
        walltime_hours: Walltime per block allocation in hours.
        qos: SLURM QOS.
        account: SLURM account/allocation.
        conda_env: Conda environment name.
        conda_base: Conda base path.
        ld_library_path: Override LD_LIBRARY_PATH.
        orca_path: Absolute path to ORCA executable. When provided and
            absolute, its parent directory is prepended to PATH on worker
            nodes so that ORCA's bundled MPI helpers are found before any
            conda-provided mpirun (prevents intermittent MPI bootstrap
            failures on SLURM systems).
        mpirun_path: Absolute path to the desired mpirun executable. When
            provided, its parent directory is prepended to PATH on worker
            nodes ahead of the ORCA bin directory.
        enable_monitoring: If True, attach a MonitoringHub that writes
            resource usage to monitoring.db under the run directory.

    Returns:
        Parsl Config object.
    """
    if not PARSL_AVAILABLE:
        raise ImportError(
            "Parsl is not installed. Please install with: pip install 'parsl>=2024.1'"
        )

    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    from parsl.providers import SlurmProvider

    runtime_address = _get_parsl_runtime_address()
    worker_init = _build_parsl_worker_init(
        scheduler="slurm",
        conda_env=conda_env,
        conda_base=conda_base,
        ld_library_path=ld_library_path,
        orca_path=orca_path,
        mpirun_path=mpirun_path,
    )

    active_cores_per_node = max_workers * cores_per_worker
    requested_cpus_per_node = cpus_per_node or active_cores_per_node
    if requested_cpus_per_node < active_cores_per_node:
        raise ValueError(
            "cpus_per_node must be >= max_workers * cores_per_worker "
            f"({active_cores_per_node})"
        )

    # Launcher and scheduler_options differ for single-node vs multi-node.
    if nodes_per_block > 1:
        from parsl.launchers import SrunLauncher

        # SrunLauncher: srun starts 1 Parsl worker manager per node.
        # The worker manager forks max_workers processes internally.
        launcher = SrunLauncher()
        # exclusive=True reserves whole nodes by default. When an explicit
        # CPU-per-node override is requested, add a matching ntasks-per-node
        # request to the batch allocation while keeping the srun launcher.
        if cpus_per_node is not None:
            scheduler_options = (
                f"#SBATCH --ntasks-per-node={requested_cpus_per_node}\n"
                f"#SBATCH --cpus-per-task=1\n"
            )
        else:
            scheduler_options = ""
    else:
        from parsl.launchers import SimpleLauncher

        # SimpleLauncher: no srun, specify total tasks per node directly.
        launcher = SimpleLauncher()
        scheduler_options = (
            f"#SBATCH --ntasks-per-node={requested_cpus_per_node}\n"
            f"#SBATCH --cpus-per-task=1\n"
        )

    provider = SlurmProvider(
        qos=qos,
        account=account,
        nodes_per_block=nodes_per_block,
        init_blocks=init_blocks,
        min_blocks=min_blocks,
        max_blocks=max_blocks,
        walltime=f"{walltime_hours:02d}:00:00",
        worker_init=worker_init,
        scheduler_options=scheduler_options,
        exclusive=True,
        launcher=launcher,
        parallelism=1.0,
    )

    executor = HighThroughputExecutor(
        label="slurm_htex",
        address=runtime_address,
        cores_per_worker=cores_per_worker,
        max_workers_per_node=max_workers,
        cpu_affinity="block",
        provider=provider,
    )

    run_dir = _build_parsl_run_dir()
    monitoring = (
        _build_parsl_monitoring(run_dir=run_dir, hub_address=runtime_address)
        if enable_monitoring
        else None
    )

    return Config(executors=[executor], run_dir=run_dir, monitoring=monitoring)


def build_parsl_config_pbspro(
    max_workers: int = 4,
    cores_per_worker: int = 16,
    cpus_per_node: int | None = None,
    nodes_per_block: int = 1,
    max_blocks: int = 10,
    init_blocks: int = 2,
    min_blocks: int = 1,
    walltime_hours: int = 2,
    queue: str = "pbatch",
    account: str = "ODEFN5169CYFZ",
    conda_env: str = "py10mpi",
    conda_base: str = "/usr/WS1/vargas58/miniconda3",
    ld_library_path: str | None = None,
    orca_path: str | None = None,
    mpirun_path: str | None = None,
    enable_monitoring: bool = True,
):
    """Build Parsl Config for PBS Pro / OpenPBS multi-node execution.

    Uses PBSProProvider to auto-provision worker nodes via PBS Pro/OpenPBS.
    Each block is a PBS job with ``nodes_per_block`` nodes, and each
    node runs ``max_workers`` ORCA workers concurrently.

    By default, the per-node PBS resource request is derived from the worker
    layout: ``ncpus = max_workers * cores_per_worker`` and
    ``select_options = "mpiprocs=<ncpus>"``. ``cpus_per_node`` can be set
    larger than that derived value to reserve a full node while intentionally
    leaving some cores idle for memory headroom.

    Args:
        max_workers: Maximum concurrent workers per node.
        cores_per_worker: CPU cores per worker (must match ORCA nprocs).
        cpus_per_node: Scheduler CPU cores reserved per node. When omitted,
            defaults to ``max_workers * cores_per_worker``.
        nodes_per_block: Nodes per PBS block.
        max_blocks: Maximum number of PBS blocks to provision.
        init_blocks: Number of blocks to request at startup.
        min_blocks: Minimum blocks to keep alive.
        walltime_hours: Walltime per block allocation in hours.
        queue: PBS queue.
        account: PBS account/allocation.
        conda_env: Conda environment name.
        conda_base: Conda base path.
        ld_library_path: Override LD_LIBRARY_PATH.
        orca_path: Absolute path to ORCA executable. When provided and
            absolute, its parent directory is prepended to PATH on worker
            nodes so that ORCA's bundled MPI helpers are found before any
            conda-provided mpirun.
        mpirun_path: Absolute path to the desired mpirun executable. When
            provided, its parent directory is prepended to PATH on worker
            nodes ahead of the ORCA bin directory.
        enable_monitoring: If True, attach a MonitoringHub that writes
            resource usage to monitoring.db under the run directory.

    Returns:
        Parsl Config object.
    """
    if not PARSL_AVAILABLE:
        raise ImportError(
            "Parsl is not installed. Please install with: pip install 'parsl>=2024.1'"
        )

    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    from parsl.providers import PBSProProvider

    runtime_address = _get_parsl_runtime_address()
    worker_init = _build_parsl_worker_init(
        scheduler="pbspro",
        conda_env=conda_env,
        conda_base=conda_base,
        ld_library_path=ld_library_path,
        orca_path=orca_path,
        mpirun_path=mpirun_path,
    )

    # Validate before any side-effectful construction (matches SLURM builder order).
    active_cores_per_node = max_workers * cores_per_worker
    requested_cpus_per_node = cpus_per_node or active_cores_per_node
    if requested_cpus_per_node < active_cores_per_node:
        raise ValueError(
            "cpus_per_node must be >= max_workers * cores_per_worker "
            f"({active_cores_per_node})"
        )

    # run_dir is built before the launcher because PbsdshLauncher needs a
    # filesystem path for its per-node helper scripts.
    run_dir = _build_parsl_run_dir()

    if nodes_per_block > 1:
        from .parsl_launchers import PbsdshLauncher

        launcher = PbsdshLauncher(Path(run_dir) / "pbsdsh_helpers")
    else:
        from parsl.launchers import SimpleLauncher

        launcher = SimpleLauncher()

    # Note: PBSProProvider has no 'exclusive' keyword (unlike SlurmProvider).
    # Whole-node exclusivity is achieved via the select statement:
    # select=N:ncpus=<requested_cpus_per_node>:mpiprocs=<requested_cpus_per_node>.
    provider = PBSProProvider(
        queue=queue,
        account=account,
        nodes_per_block=nodes_per_block,
        cpus_per_node=requested_cpus_per_node,
        select_options=f"mpiprocs={requested_cpus_per_node}",
        init_blocks=init_blocks,
        min_blocks=min_blocks,
        max_blocks=max_blocks,
        walltime=f"{walltime_hours:02d}:00:00",
        worker_init=worker_init,
        launcher=launcher,
        parallelism=1.0,
    )

    executor = HighThroughputExecutor(
        label="pbspro_htex",
        address=runtime_address,
        launch_cmd=_build_parsl_htex_launch_cmd(),
        cores_per_worker=cores_per_worker,
        max_workers_per_node=max_workers,
        cpu_affinity="block",
        provider=provider,
    )

    monitoring = (
        _build_parsl_monitoring(run_dir=run_dir, hub_address=runtime_address)
        if enable_monitoring
        else None
    )

    return Config(executors=[executor], run_dir=run_dir, monitoring=monitoring)


def submit_batch_parsl(
    workflow: ArchitectorWorkflow,
    root_dir: str | Path,
    num_jobs: int,
    max_workers: int = 4,
    cores_per_worker: int = 16,
    cpus_per_node: int | None = None,
    scheduler: str = "flux",
    job_dir_pattern: str = DEFAULT_JOB_DIR_PATTERN,
    orca_config: OrcaConfig | None = None,
    setup_func: Callable | None = None,
    n_cores: int = 16,
    conda_env: str = "py10mpi",
    conda_base: str = "/usr/WS1/vargas58/miniconda3",
    ld_library_path: str | None = None,
    mpirun_path: str | None = None,
    dry_run: bool = False,
    max_fail_count: int | None = None,
    max_atoms: int | None = None,
    min_atoms: int | None = None,
    timeout_seconds: int = 72000,
    randomize: bool = True,
    nodes_per_block: int = 1,
    max_blocks: int = 10,
    init_blocks: int = 2,
    min_blocks: int = 1,
    walltime_hours: int = 2,
    queue: str = "pbatch",
    qos: str = "frontier",
    account: str = "ODEFN5169CYFZ",
    enable_monitoring: bool = True,
    wandb_run: Any | None = None,
    reroot: bool = False,
    site: str = "default",
    partition: str | None = None,
    openmpi_module: str = SANDIA_DEFAULT_OPENMPI_MODULE,
    clean_on_complete: bool = False,
    purge_on_fail: bool = False,
) -> list[int]:
    """Submit batch of jobs using Parsl for concurrent execution.

    Args:
        workflow: ArchitectorWorkflow instance
        root_dir: Root directory for job directories
        num_jobs: Total number of jobs to submit
        max_workers: Maximum number of concurrent workers
        cores_per_worker: CPU cores per worker
        cpus_per_node: Scheduler CPU cores reserved/requested per node. When
            omitted, defaults to ``max_workers * cores_per_worker`` where the
            scheduler-specific builder needs an explicit per-node CPU shape.
        scheduler: Parsl provider backend ("flux" for LocalProvider,
            "slurm" for SlurmProvider multi-node, "pbspro" for
            PBSProProvider multi-node).
        job_dir_pattern: Pattern for job directory names. Supports
            {hostname}, {orig_index}, and {id}.
        orca_config: ORCA configuration
        setup_func: Optional setup function per job
        n_cores: Cores per ORCA job
        conda_env: Conda environment name
        conda_base: Conda base path
        ld_library_path: Override LD_LIBRARY_PATH
        dry_run: Prepare but don't submit
        max_fail_count: Skip jobs with fail_count >= this value
        max_atoms: If set, only submit molecules with natoms <= this value
        min_atoms: If set, only submit molecules with natoms >= this value
        timeout_seconds: Job timeout in seconds (default: 72000 = 20 hours)
        randomize: Randomize job selection order (default: True)
        nodes_per_block: Nodes per scheduler block for scale-out Parsl.
        max_blocks: Maximum scheduler blocks to provision.
        init_blocks: Blocks to request at startup.
        min_blocks: Minimum blocks to keep alive.
        walltime_hours: Walltime per block allocation in hours.
        queue: Queue/partition name for supported schedulers.
        qos: (SLURM) SLURM QOS.
        account: (SLURM) SLURM account/allocation.
        enable_monitoring: If True, attach Parsl MonitoringHub (writes
            monitoring.db). Disable on systems where the monitoring hub
            port or SQLAlchemy dependency causes issues.
        wandb_run: Optional W&B run object from ``init_wandb_run()``. When
            provided, each job completion/failure/timeout is logged in
            real-time. Pass ``None`` (default) to disable W&B logging.
        reroot: If True, ignore job_dir paths stored in the database and
            always construct paths from root_dir + job_dir_pattern. Useful
            when the database has been relocated.
        clean_on_complete: If True, run --clean-all (remove .tmp/.core/
            orca_tmp_*/ and .bas/.basN) on each job_dir immediately after
            it completes successfully.
        purge_on_fail: If True, purge each job_dir immediately after the
            job fails: write a .do_not_rerun.json marker with failure
            metadata, then delete all other contents.

    Returns:
        List of submitted job IDs
    """
    if not PARSL_AVAILABLE:
        print(
            "Error: Parsl is not installed. Please install with: pip install 'parsl>=2024.1'"
        )
        return []

    import signal
    from concurrent.futures import as_completed

    import parsl

    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    # Auto-correct n_cores to match cores_per_worker for proper resource allocation
    if n_cores != cores_per_worker:
        print(
            f"Warning: n_cores ({n_cores}) doesn't match cores_per_worker ({cores_per_worker})"
        )
        print(
            f"   Auto-setting n_cores = {cores_per_worker} for proper resource allocation"
        )
        n_cores = cores_per_worker

    # Merge ORCA config
    config: OrcaConfig = {**DEFAULT_ORCA_CONFIG, **(orca_config or {})}
    if "orca_path" not in config or config.get("orca_path") is None:
        if site.lower() == "sandia":
            raise ValueError(
                "site='sandia' requires an explicit orca_path: "
                "Sandia has no shared ORCA install. Pass --orca-path on the "
                "CLI or set orca_path in orca_config."
            )
        if site.lower() == "drac":
            raise ValueError(
                "site='drac' in Parsl mode requires an explicit orca_path: the "
                "ASE/quacc calculator needs a concrete binary, and $EBROOTORCA "
                "is only defined inside the module-loaded shell. Resolve it in "
                'the launch script and pass --orca-path "$EBROOTORCA/orca".'
            )
        config["orca_path"] = DEFAULT_ORCA_PATHS.get(
            scheduler.lower(), DEFAULT_ORCA_PATHS["flux"]
        )

    # Filter jobs for submission (DB status only)
    jobs_to_submit = filter_jobs_for_submission(
        workflow,
        num_jobs=num_jobs,
        max_fail_count=max_fail_count,
        randomize=randomize,
        max_atoms=max_atoms,
        min_atoms=min_atoms,
    )

    if not jobs_to_submit:
        print("No jobs available for submission after filtering")
        return []

    if reroot:
        print("Reroot mode: ignoring database job_dir paths, using root_dir")

    # Filter out jobs with .do_not_rerun.json marker files
    jobs_to_submit = _filter_marker_jobs(
        jobs_to_submit,
        root_dir,
        job_dir_pattern,
        workflow,
        force_root_dir=reroot,
    )

    if not jobs_to_submit:
        print("No jobs available after marker filtering")
        return []

    # Filter out jobs that already completed or failed on disk
    jobs_to_submit = _skip_finished_on_disk(
        jobs_to_submit,
        root_dir,
        job_dir_pattern,
        workflow,
        force_root_dir=reroot,
    )

    if not jobs_to_submit:
        print("No jobs available after disk check")
        return []

    # Detect scheduler job ID early so it can be set atomically with the
    # RUNNING status claim (avoids a window where jobs are RUNNING but have
    # no worker_id, which would make them invisible to --recover-orphans).
    _scheduler_job_id = _resolve_scheduler_job_id()

    # Claim jobs atomically BEFORE slow directory preparation to prevent
    # concurrent submitters from grabbing the same jobs (TOCTOU fix).
    # Sets worker_id in the same UPDATE/commit for crash traceability.
    submitted_ids = [j.id for j in jobs_to_submit]
    if not dry_run:
        workflow.mark_jobs_as_running(submitted_ids, worker_id=_scheduler_job_id)
        print(
            f"Claimed {len(submitted_ids)} jobs as RUNNING "
            f"(worker_id={_scheduler_job_id})"
        )

    print(f"\nPreparing {len(jobs_to_submit)} jobs for Parsl submission...")

    # Prepare job directories, tracking any failures.
    # Collect job_dir updates to batch-commit once (avoids per-job lock contention).
    failed_prep_ids: list[int] = []
    job_dir_updates: list[dict] = []
    print("Setting up job directories...")
    for i, job in enumerate(jobs_to_submit, 1):
        try:
            job_dir = prepare_job_directory(
                job,
                root_dir,
                job_dir_pattern=job_dir_pattern,
                orca_config=config,
                n_cores=n_cores,
                setup_func=setup_func,
                force_root_dir=reroot,
            )
            # Collect for batch commit instead of per-job commit
            if not dry_run:
                job_dir_updates.append({"job_id": job.id, "job_dir": str(job_dir)})
            print(f"  [{i}/{len(jobs_to_submit)}] Prepared {job_dir}")
        except Exception as e:
            print(f"  [{i}/{len(jobs_to_submit)}] FAILED to prepare job {job.id}: {e}")
            failed_prep_ids.append(job.id)

    # Batch-commit all job_dir updates in a single transaction
    if job_dir_updates:
        workflow.update_job_metrics_bulk(job_dir_updates)
        print(f"Persisted {len(job_dir_updates)} job directories in one transaction")

    # Reset any jobs that failed during preparation back to TO_RUN
    if failed_prep_ids:
        for jid in failed_prep_ids:
            try:
                workflow.update_status(jid, JobStatus.TO_RUN)
            except Exception:
                pass
        jobs_to_submit = [j for j in jobs_to_submit if j.id not in set(failed_prep_ids)]
        submitted_ids = [j.id for j in jobs_to_submit]
        print(f"Reset {len(failed_prep_ids)} jobs back to TO_RUN due to prep failure")

    if dry_run:
        print("\n[DRY RUN] Would submit to Parsl executor")
        print(f"  Max workers: {max_workers}")
        print(f"  Cores per worker: {cores_per_worker}")
        return [j.id for j in jobs_to_submit]

    # Build Parsl configuration. Sandia dispatches to SlurmProvider when the
    # caller asks for multi-block scale-out, or to LocalProvider when running
    # inside an existing salloc allocation (single block, single node).
    use_multi_block = nodes_per_block > 1 or max_blocks > 1
    if site.lower() == "sandia":
        if use_multi_block:
            total_nodes = max_blocks * nodes_per_block
            total_workers = total_nodes * max_workers
            print(
                f"\nParsl Sandia SLURM config: {max_blocks} blocks x {nodes_per_block} "
                f"nodes/block x {max_workers} workers/node "
                f"= {total_workers} max concurrent jobs"
            )
            parsl_config = build_parsl_config_sandia(
                max_workers=max_workers,
                cores_per_worker=cores_per_worker,
                cpus_per_node=cpus_per_node,
                nodes_per_block=nodes_per_block,
                max_blocks=max_blocks,
                init_blocks=init_blocks,
                min_blocks=min_blocks,
                walltime_hours=walltime_hours,
                qos=qos,
                account=account,
                partition=partition or SANDIA_DEFAULT_PARTITION,
                openmpi_module=openmpi_module,
                orca_path=config.get("orca_path"),
                ld_library_path=ld_library_path,
                enable_monitoring=enable_monitoring,
            )
        else:
            print(
                f"\nParsl Sandia (single-node LocalProvider): "
                f"{max_workers} workers x {cores_per_worker} cores "
                f"= {max_workers * cores_per_worker} concurrent cores. "
                f"Coordinator must already be inside a SLURM allocation "
                f"(e.g. salloc -N1 -p attaway -A {SANDIA_DEFAULT_ACCOUNT}). "
                f"Job timeout: {timeout_seconds}s"
            )
            parsl_config = build_parsl_config_sandia_local(
                max_workers=max_workers,
                cores_per_worker=cores_per_worker,
                orca_path=config.get("orca_path"),
                enable_monitoring=enable_monitoring,
            )
    elif site.lower() == "drac":
        # DRAC Parsl is single-node LocalProvider only: one allocation packs
        # max_workers ORCA jobs onto the node it was launched in. Multi-block
        # SlurmProvider for DRAC is not implemented.
        print(
            f"\nParsl DRAC (single-node LocalProvider): "
            f"{max_workers} workers x {cores_per_worker} cores "
            f"= {max_workers * cores_per_worker} concurrent cores. "
            f"Coordinator must already be inside a SLURM allocation that "
            f"module-loaded the ORCA chain and activated the venv "
            f"(see launch/run_parsl_single_node_drac.sh). "
            f"Job timeout: {timeout_seconds}s"
        )
        parsl_config = build_parsl_config_drac_local(
            max_workers=max_workers,
            cores_per_worker=cores_per_worker,
            orca_path=config.get("orca_path"),
            enable_monitoring=enable_monitoring,
        )
    elif scheduler.lower() == "slurm":
        total_nodes = max_blocks * nodes_per_block
        total_workers = total_nodes * max_workers
        print(
            f"\nParsl SLURM config: {max_blocks} blocks x {nodes_per_block} "
            f"nodes/block x {max_workers} workers/node "
            f"= {total_workers} max concurrent jobs"
        )
        requested_cpus = cpus_per_node or (max_workers * cores_per_worker)
        if cpus_per_node is not None:
            print(
                f"Each SLURM node requests {requested_cpus} scheduler cores, "
                f"active worker cores per node: {max_workers * cores_per_worker}, "
                f"each worker: {cores_per_worker} cores, "
                f"job timeout: {timeout_seconds}s"
            )
        else:
            print(
                f"Each SLURM node uses exclusive allocation, "
                f"active worker cores per node: {max_workers * cores_per_worker}, "
                f"each worker: {cores_per_worker} cores, "
                f"job timeout: {timeout_seconds}s"
            )
        parsl_config = build_parsl_config_slurm(
            max_workers=max_workers,
            cores_per_worker=cores_per_worker,
            cpus_per_node=cpus_per_node,
            nodes_per_block=nodes_per_block,
            max_blocks=max_blocks,
            init_blocks=init_blocks,
            min_blocks=min_blocks,
            walltime_hours=walltime_hours,
            qos=qos,
            account=account,
            conda_env=conda_env,
            conda_base=conda_base,
            ld_library_path=ld_library_path,
            orca_path=config.get("orca_path"),
            mpirun_path=mpirun_path,
            enable_monitoring=enable_monitoring,
        )
    elif scheduler.lower() == "pbspro":
        total_nodes = max_blocks * nodes_per_block
        total_workers = total_nodes * max_workers
        print(
            f"\nParsl PBS Pro config: {max_blocks} blocks x {nodes_per_block} "
            f"nodes/block x {max_workers} workers/node "
            f"= {total_workers} max concurrent jobs"
        )
        print(
            f"Each PBS node reserves {cpus_per_node or (max_workers * cores_per_worker)} "
            f"ncpus/mpiprocs, active worker cores per node: {max_workers * cores_per_worker}, "
            f"each worker: {cores_per_worker} cores, "
            f"job timeout: {timeout_seconds}s"
        )
        parsl_config = build_parsl_config_pbspro(
            max_workers=max_workers,
            cores_per_worker=cores_per_worker,
            cpus_per_node=cpus_per_node,
            nodes_per_block=nodes_per_block,
            max_blocks=max_blocks,
            init_blocks=init_blocks,
            min_blocks=min_blocks,
            walltime_hours=walltime_hours,
            queue=queue,
            account=account,
            conda_env=conda_env,
            conda_base=conda_base,
            ld_library_path=ld_library_path,
            orca_path=config.get("orca_path"),
            mpirun_path=mpirun_path,
            enable_monitoring=enable_monitoring,
        )
    else:
        print("\nBuilding Parsl config (Flux single-node)...")
        parsl_config = build_parsl_config_flux(
            max_workers=max_workers,
            cores_per_worker=cores_per_worker,
            conda_env=conda_env,
            conda_base=conda_base,
            ld_library_path=ld_library_path,
            orca_path=config.get("orca_path"),
            mpirun_path=mpirun_path,
            enable_monitoring=enable_monitoring,
        )

    # Initialize Parsl
    try:
        parsl.clear()
        parsl.load(parsl_config)
        print("Parsl executor loaded successfully")
    except Exception as e:
        print(f"Failed to initialize Parsl: {e}")
        print("Check your conda environment and ORCA installation")
        # Reset claimed jobs back to TO_RUN since we can't run them
        workflow.update_status_bulk(submitted_ids, JobStatus.TO_RUN, worker_id=None)
        print(f"Reset {len(submitted_ids)} claimed jobs back to TO_RUN")
        return []

    # --- SIGTERM handler (register AFTER parsl.load to avoid Parsl overwriting) ---
    # When the scheduler cancels an allocation (scancel / flux cancel), it
    # sends SIGTERM to our process.  Workers die immediately, so no futures
    # will ever complete -- as_completed() blocks forever.  A flag-only
    # handler would never be checked and the finally block (which resets
    # orphaned jobs) would never run before SIGKILL arrives.
    #
    # Fix: after setting the flag, raise KeyboardInterrupt.  This interrupts
    # as_completed(), is caught by the existing `except KeyboardInterrupt`
    # handler, and falls through to the finally block which bulk-resets all
    # in-flight jobs to TO_RUN.  SQLite transactions are atomic, so an
    # interrupted commit simply rolls back -- at most one job's status update
    # is lost (and the finally block will correct it).
    _shutdown_requested = False
    _original_sigterm = signal.getsignal(signal.SIGTERM)

    def _sigterm_handler(signum, frame):
        nonlocal _shutdown_requested
        _shutdown_requested = True
        # Restore original handler so a second SIGTERM during cleanup
        # does not raise another KeyboardInterrupt mid-finally.
        signal.signal(signal.SIGTERM, _original_sigterm)
        try:
            sys.stderr.write(
                "\nSIGTERM received -- shutting down and resetting "
                "in-flight jobs...\n"
            )
        except Exception:
            pass  # Best-effort notification; flag is already set
        # Break out of as_completed() so the finally block can reset orphans
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _sigterm_handler)

    # Check if SIGTERM arrived during Parsl setup (before handler was installed,
    # the default handler may have set a pending signal that fires now).
    if _shutdown_requested:
        print("Shutdown requested during setup -- skipping submission")
        # Falls through to the finally block which resets orphans
        signal.signal(signal.SIGTERM, _original_sigterm)
        workflow.update_status_bulk(submitted_ids, JobStatus.TO_RUN, worker_id=None)
        print(f"Reset {len(submitted_ids)} jobs back to TO_RUN")
        return submitted_ids

    # Submit futures
    print(f"\nSubmitting {len(jobs_to_submit)} jobs to Parsl...")
    futures = []
    task_map_path = Path(parsl_config.run_dir).resolve() / "parsl_task_map.tsv"
    task_map_rows = ["task_id\tjob_id\tjob_dir\n"]
    print(f"Recording Parsl task mapping to {task_map_path}")

    for job in jobs_to_submit:
        job_dir_name = render_job_dir_pattern(
            job_dir_pattern,
            orig_index=job.orig_index,
            job_id=job.id,
        )
        job_dir_abs = (root_dir / job_dir_name).resolve()

        future = orca_job_wrapper(
            job_id=job.id,
            job_dir=str(job_dir_abs),
            orca_config=dict(config),
            timeout_seconds=timeout_seconds,
        )
        task_map_rows.append(f"{future.tid}\t{job.id}\t{job_dir_abs}\n")
        futures.append((job.id, str(job_dir_abs), future))

    with open(task_map_path, "w", buffering=1024 * 1024) as fh:
        fh.writelines(task_map_rows)

    # Monitor futures concurrently (CRITICAL: use as_completed, not sequential loop)
    print("\nMonitoring job execution...")
    print("(Press Ctrl+C for graceful shutdown)\n")

    completed_ids: list[int] = []
    failed_ids: list[int] = []
    pending_updates: list[dict] = []

    # Create future->(job_id, job_dir) mapping for concurrent completion
    futures_map = {future: (job_id, job_dir) for job_id, job_dir, future in futures}

    # JobRecord lookup for inline purge metadata (orig_index, elements, etc.)
    id_to_job = {j.id: j for j in jobs_to_submit}
    optimizer = config.get("optimizer")
    db_path = workflow.db_path

    # Seed t0 point.
    last_snap = 0.0
    if wandb_run is not None:
        log_progress_snapshot(wandb_run, workflow)
        last_snap = time.time()

    try:
        # as_completed() yields futures as they finish (concurrent, not sequential!)
        for future in as_completed(futures_map.keys()):
            job_id, job_dir = futures_map[future]
            try:
                result = future.result()

                if result["status"] == "completed":
                    completed_ids.append(job_id)

                    # Extract metrics on the fly for successful jobs
                    metrics_dict: dict | None = None
                    try:
                        metrics = parse_job_metrics(job_dir)
                        wall_time = None
                        n_cores_val = None
                        try:
                            log_file = pull_log_file(str(job_dir))
                            n_cores_parsed, time_dict = find_timings_and_cores(log_file)
                            if time_dict and "Total" in time_dict:
                                wall_time = time_dict["Total"]
                            n_cores_val = n_cores_parsed
                        except Exception as e:
                            print(
                                f"  Warning: timing extraction failed for job {job_id}: {e}"
                            )
                        if metrics["success"]:
                            metrics_dict = {
                                "job_dir": job_dir,
                                "max_forces": metrics.get("max_forces"),
                                "scf_steps": metrics.get("scf_steps"),
                                "final_energy": metrics.get("final_energy"),
                                "wall_time": wall_time,
                                "n_cores": n_cores_val,
                            }
                            if GENERATOR_AVAILABLE:
                                try:
                                    gen_data = parse_generator_data(job_dir)
                                    if gen_data is not None:
                                        metrics_dict["generator_data"] = gen_data
                                except Exception as e:
                                    print(
                                        f"  Warning: generator parsing failed for job {job_id}: {e}"
                                    )
                    except Exception as e:
                        print(
                            f"  Warning: metrics extraction failed for job {job_id}: {e}"
                        )

                    pending_updates.append(
                        {
                            "job_id": job_id,
                            "status": JobStatus.COMPLETED,
                            "metrics": metrics_dict,
                        }
                    )
                    print(
                        f" Job {job_id} completed ({len(completed_ids)}/{len(futures)} done)"
                    )
                    log_job_result(wandb_run, job_id, "completed", metrics_dict)
                elif result["status"] == "timeout":
                    pending_updates.append(
                        {
                            "job_id": job_id,
                            "status": JobStatus.TIMEOUT,
                            "error_message": result.get("error"),
                        }
                    )
                    failed_ids.append(job_id)
                    print(f"Job {job_id} timeout")
                    log_job_result(wandb_run, job_id, "timeout")
                else:
                    pending_updates.append(
                        {
                            "job_id": job_id,
                            "status": JobStatus.FAILED,
                            "error_message": result.get("error"),
                            "increment_fail_count": True,
                        }
                    )
                    failed_ids.append(job_id)
                    error_msg = result.get("error", "Unknown error")[:100]
                    print(f"Job {job_id} failed: {error_msg}")
                    log_job_result(wandb_run, job_id, "failed")

            except Exception as e:
                pending_updates.append(
                    {
                        "job_id": job_id,
                        "status": JobStatus.FAILED,
                        "error_message": str(e),
                        "increment_fail_count": True,
                    }
                )
                failed_ids.append(job_id)
                print(f"Job {job_id} exception: {str(e)[:100]}")
                log_job_result(wandb_run, job_id, "failed")

            last_update = pending_updates[-1]
            terminal_status = last_update["status"]

            # Crash-safety: write the do-not-rerun marker BEFORE the DB flips
            # to FAILED. If SIGTERM lands between the marker write and the
            # commit, the marker on disk still blocks resubmission via the
            # existing submit guard; without it, the next --reset-failed
            # cycle could re-queue a job we meant to purge.
            prefailure_job_record: JobRecord | None = None
            if purge_on_fail and terminal_status == JobStatus.FAILED:
                prefailure_job_record = id_to_job.get(job_id)
                if prefailure_job_record is not None:
                    _write_prefailure_marker(
                        job_dir,
                        prefailure_job_record,
                        last_update.get("error_message"),
                    )

            # Write this job's result to DB immediately (one commit per job).
            _write_job_update(workflow, last_update)
            pending_updates.pop()

            # Destructive cleanup runs after the DB write so _purge_failed_job's
            # TOCTOU re-check sees the FAILED status we just committed. It will
            # idempotently overwrite the marker file written above with richer
            # metadata (the in-loop _extract_failure_info pulls from disk).
            if clean_on_complete and terminal_status == JobStatus.COMPLETED:
                _cleanup_completed_job_inline(job_dir, root_dir, optimizer)
            elif (
                purge_on_fail
                and terminal_status == JobStatus.FAILED
                and prefailure_job_record is not None
            ):
                _purge_failed_job_inline(
                    job_dir,
                    root_dir,
                    db_path,
                    prefailure_job_record,
                    last_update.get("error_message"),
                )

            if (
                wandb_run is not None
                and time.time() - last_snap >= SNAPSHOT_INTERVAL_SEC
            ):
                last_snap = time.time()
                log_progress_snapshot(wandb_run, workflow)

            # Check shutdown flag AFTER DB writes for this future.
            if _shutdown_requested:
                print("Shutdown requested -- exiting monitoring loop...")
                break

    except KeyboardInterrupt:
        print("\n\nGraceful shutdown requested...")

    finally:
        # Restore original handler first so a second SIGTERM during cleanup
        # does not raise KeyboardInterrupt mid-finally and abandon dfk.cleanup().
        try:
            signal.signal(signal.SIGTERM, _original_sigterm)
        except Exception:
            pass

        # Write any remaining updates before resetting orphans.
        # Best-effort: if this fails, _skip_finished_on_disk catches it
        # on the next submission pass.
        for u in list(pending_updates):
            try:
                _write_job_update(workflow, u)
            except Exception:
                pass
        pending_updates.clear()

        # Reset orphaned jobs (still RUNNING) back to TO_RUN.
        # Use bulk update (single UPDATE + single commit) to stay well within
        # SLURM's ~30s SIGKILL grace window after SIGTERM.
        resolved_ids = set(completed_ids) | set(failed_ids)
        orphaned_ids = [jid for jid in submitted_ids if jid not in resolved_ids]
        if orphaned_ids:
            try:
                workflow.update_status_bulk(
                    orphaned_ids, JobStatus.TO_RUN, worker_id=None
                )
            except Exception:
                # Fallback: try individually if bulk fails
                for jid in orphaned_ids:
                    try:
                        workflow.update_status(jid, JobStatus.TO_RUN, worker_id=None)
                    except Exception:
                        pass
            print(f"Reset {len(orphaned_ids)} in-flight jobs back to TO_RUN")

        # Cleanup Parsl -- running workers are terminated by dfk.cleanup().
        # Their job IDs are already classified as orphans and reset above.
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

        log_progress_snapshot(wandb_run, workflow)
        finish_wandb_run(wandb_run)

    print("\nSubmission complete:")
    print(f"  Completed: {len(completed_ids)}")
    print(f"  Failed: {len(failed_ids)}")
    print(f"  Total: {len(submitted_ids)}")

    return submitted_ids


def submit_batch(
    workflow: ArchitectorWorkflow,
    root_dir: str | Path,
    batch_size: int = 10,
    scheduler: str = "flux",
    job_dir_pattern: str = DEFAULT_JOB_DIR_PATTERN,
    orca_config: OrcaConfig | None = None,
    setup_func: Callable | None = None,
    n_cores: int = 4,
    n_hours: int = 2,
    queue: str = "pbatch",
    allocation: str = "dnn-sim",
    conda_env: str = "py10mpi",
    ld_library_path: str | None = None,
    dry_run: bool = False,
    max_fail_count: int | None = None,
    max_atoms: int | None = None,
    min_atoms: int | None = None,
    randomize: bool = True,
    reroot: bool = False,
    site: str = "default",
    partition: str | None = None,
    openmpi_module: str = SANDIA_DEFAULT_OPENMPI_MODULE,
    module_load: str = DRAC_DEFAULT_MODULE_LOAD,
    venv_path: str | None = None,
    mem_per_cpu: str | None = None,
) -> list[int]:
    """Submit a batch of ready jobs to the HPC scheduler.

    Args:
        workflow: ArchitectorWorkflow instance.
        root_dir: Root directory for job directories.
        batch_size: Number of jobs to submit in this batch.
        scheduler: Either "flux" or "slurm".
        job_dir_pattern: Pattern for job directory names. Supports
            {hostname}, {orig_index}, and {id}.
        orca_config: ORCA calculation configuration.
        setup_func: Optional function to set up job-specific files.
        n_cores: Number of cores per job.
        n_hours: Runtime in hours.
        queue: Queue/partition/QOS name.
        allocation: Allocation/account name.
        conda_env: Conda environment to activate.
        ld_library_path: Override LD_LIBRARY_PATH in generated job scripts.
        dry_run: If True, prepare directories but don't submit.
        max_fail_count: If specified, skip jobs with fail_count >= this value.
        max_atoms: If set, only submit molecules with natoms <= this value.
        min_atoms: If set, only submit molecules with natoms >= this value.
        randomize: Randomize job selection order (default: True).
        reroot: If True, ignore job_dir paths stored in the database and
            always construct paths from root_dir + job_dir_pattern. Useful
            when the database has been relocated.
        site: HPC site flavor selecting the job-script writer. ``"default"``
            uses the standard SLURM/Flux writers (conda + constraint=standard).
            ``"sandia"`` selects ``write_slurm_sandia_job_file`` which uses
            ``module load`` + OMPI MCA env + ``--partition``. ``"drac"`` selects
            ``write_slurm_drac_job_file`` (module-system ORCA, no qos/partition/
            MCA). Both SLURM-only.
        partition: SLURM partition (only used when ``site="sandia"``). When
            ``None``, the Sandia default partition is used.
        openmpi_module: ``module load`` argument for the Sandia writer.
        module_load: ``module load`` chain for the DRAC writer (only used when
            ``site="drac"``); must end in the ORCA module.
        venv_path: virtualenv to activate in the DRAC writer (only used when
            ``site="drac"``); required for the Sella optimizer path.
        mem_per_cpu: SLURM ``--mem-per-cpu`` emitted in DRAC job scripts (only
            used when ``site="drac"``), e.g. ``"3900M"``. Without it DRAC grants
            a tiny default and ORCA is OOM-killed.

    Returns:
        List of job IDs that were submitted.
    """
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    # Merge config with defaults and set scheduler/site-specific orca_path
    config: OrcaConfig = {**DEFAULT_ORCA_CONFIG, **(orca_config or {})}
    if "orca_path" not in config or config.get("orca_path") is None:
        if site.lower() == "sandia":
            raise ValueError(
                "site='sandia' requires an explicit orca_path: "
                "Sandia has no shared ORCA install. Pass --orca-path on the "
                "CLI or set orca_path in orca_config."
            )
        config["orca_path"] = DEFAULT_ORCA_PATHS.get(
            scheduler.lower(), DEFAULT_ORCA_PATHS["flux"]
        )

    jobs_to_submit = filter_jobs_for_submission(
        workflow,
        num_jobs=batch_size,
        max_fail_count=max_fail_count,
        randomize=randomize,
        max_atoms=max_atoms,
        min_atoms=min_atoms,
    )

    if not jobs_to_submit:
        print("No ready jobs to submit")
        return []

    print(f"Preparing {len(jobs_to_submit)} jobs for submission...")

    if reroot:
        print("Reroot mode: ignoring database job_dir paths, using root_dir")

    # Filter out jobs with .do_not_rerun.json marker files
    jobs_to_submit = _filter_marker_jobs(
        jobs_to_submit,
        root_dir,
        job_dir_pattern,
        workflow,
        force_root_dir=reroot,
    )

    if not jobs_to_submit:
        print("No jobs available after marker filtering")
        return []

    # Filter out jobs that already completed or failed on disk
    jobs_to_submit = _skip_finished_on_disk(
        jobs_to_submit,
        root_dir,
        job_dir_pattern,
        workflow,
        force_root_dir=reroot,
    )

    if not jobs_to_submit:
        print("No jobs available after disk check")
        return []

    if site.lower() == "sandia" and scheduler.lower() == "slurm":
        warnings.warn(
            "Per-job sbatch on Sandia is deprecated. Many concurrent "
            "submit_jobs.py processes can corrupt the workflow SQLite on "
            "Lustre. Switch to --use-parsl with --nodes-per-block / "
            "--max-blocks. This path will be removed once the Parsl "
            "multi-block path is validated on real hardware.",
            FutureWarning,
            stacklevel=2,
        )

    # Claim jobs atomically BEFORE slow directory preparation to prevent
    # concurrent submitters from grabbing the same jobs (TOCTOU fix).
    all_claimed_ids = [j.id for j in jobs_to_submit]
    if not dry_run:
        workflow.mark_jobs_as_running(all_claimed_ids)
        print(f"Claimed {len(all_claimed_ids)} jobs as RUNNING in database")

    submitted_ids = []
    job_dir_map: dict[int, Path] = {}  # job_id -> job_dir for batch commit

    for i, job in enumerate(jobs_to_submit):
        try:
            # Prepare job directory with ORCA input
            job_dir = prepare_job_directory(
                job,
                root_dir,
                job_dir_pattern=job_dir_pattern,
                orca_config=config,
                n_cores=n_cores,
                setup_func=setup_func,
                force_root_dir=reroot,
            )

            # Collect for batch commit (avoids per-job lock contention)
            if not dry_run:
                job_dir_map[job.id] = job_dir

            # Write job submission script
            orca_path = config.get("orca_path", DEFAULT_ORCA_PATHS["flux"])
            optimizer = config.get("optimizer")
            if scheduler.lower() == "flux":
                job_script = write_flux_job_file(
                    job_dir,
                    n_cores=n_cores,
                    n_hours=n_hours,
                    queue=queue,
                    allocation=allocation,
                    orca_path=orca_path,
                    conda_env=conda_env,
                    optimizer=optimizer,
                    ld_library_path=ld_library_path,
                )
                submit_cmd = ["flux", "batch", job_script.name]

            elif scheduler.lower() == "slurm":
                if site.lower() == "sandia":
                    job_script = write_slurm_sandia_job_file(
                        job_dir,
                        n_cores=n_cores,
                        n_hours=n_hours,
                        qos=queue,
                        partition=partition or SANDIA_DEFAULT_PARTITION,
                        account=allocation,
                        orca_path=orca_path,
                        openmpi_module=openmpi_module,
                        optimizer=optimizer,
                    )
                elif site.lower() == "drac":
                    job_script = write_slurm_drac_job_file(
                        job_dir,
                        n_cores=n_cores,
                        n_hours=n_hours,
                        account=allocation,
                        module_load=module_load,
                        venv_path=venv_path,
                        mem_per_cpu=mem_per_cpu,
                        optimizer=optimizer,
                    )
                else:
                    job_script = write_slurm_job_file(
                        job_dir,
                        n_cores=n_cores,
                        n_hours=n_hours,
                        queue=queue,
                        allocation=allocation,
                        orca_path=orca_path,
                        conda_env=conda_env,
                        optimizer=optimizer,
                        ld_library_path=ld_library_path,
                    )
                submit_cmd = ["sbatch", job_script.name]
            else:
                raise ValueError(f"Unknown scheduler: {scheduler}")

            print(f"  [{i+1}/{len(jobs_to_submit)}] Prepared job {job.id} in {job_dir}")

            # Submit job
            if not dry_run:
                result = subprocess.run(
                    submit_cmd,
                    cwd=job_dir,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                print(f"    Submitted: {result.stdout.strip()}")
                submitted_ids.append(job.id)
            else:
                print(f"    [DRY RUN] Would submit: {' '.join(submit_cmd)}")
                submitted_ids.append(job.id)

        except Exception as e:
            print(f"  [{i+1}/{len(jobs_to_submit)}] Error for job {job.id}: {e}")
            if not dry_run:
                try:
                    workflow.update_status(job.id, JobStatus.TO_RUN)
                except Exception:
                    pass
            continue

    # Batch-commit all job_dir updates in one transaction
    if not dry_run and job_dir_map:
        bulk_updates = [
            {"job_id": jid, "job_dir": str(jdir)} for jid, jdir in job_dir_map.items()
        ]
        workflow.update_job_metrics_bulk(bulk_updates)

    # Reset any claimed-but-not-submitted jobs back to TO_RUN
    if not dry_run:
        not_submitted = set(all_claimed_ids) - set(submitted_ids)
        if not_submitted:
            for jid in not_submitted:
                try:
                    workflow.update_status(jid, JobStatus.TO_RUN)
                except Exception:
                    pass
            print(f"Reset {len(not_submitted)} unclaimed jobs back to TO_RUN")

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
        choices=["flux", "slurm", "pbspro"],
        default="flux",
        help="HPC scheduler (default: flux)",
    )
    parser.add_argument(
        "--job-dir-pattern",
        default=DEFAULT_JOB_DIR_PATTERN,
        help=(
            "Pattern for job directory names. Supports {hostname}, "
            "{orig_index}, and {id} (default: "
            f"{DEFAULT_JOB_DIR_PATTERN})"
        ),
    )
    parser.add_argument(
        "--job-prefix",
        default=None,
        help=(
            "Optional stable prefix to prepend to job directories, for example "
            "'campaignA' -> campaignA_job_{orig_index}. Useful across coordinator "
            "requeues when the same run should keep reusing the same job directories."
        ),
    )
    parser.add_argument(
        "--n-cores",
        type=int,
        default=4,
        help="Number of cores per job (default: 4)",
    )
    parser.add_argument(
        "--mem-per-job",
        type=int,
        default=None,
        metavar="MB",
        help=(
            "Optional per-ORCA-job memory budget in MB. When set, %%maxcore "
            "is sized so total memory stays under 85%% of this value. "
            "Recommended on memory-constrained nodes (Sandia CTS-1: ~60000, "
            "TLCC2: ~30000). Default: no clamp (per-process floor only)."
        ),
    )
    parser.add_argument(
        "--mem-per-cpu",
        default=None,
        metavar="MEM",
        help=(
            "SLURM --mem-per-cpu for generated job scripts (only used with "
            "--hpc-site drac, e.g. '3900M' = the ~4GB/core node ratio). Without "
            "it DRAC grants a tiny default and ORCA gets OOM-killed."
        ),
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
        "--ld-library-path",
        default=None,
        help="Override LD_LIBRARY_PATH in generated job scripts",
    )
    parser.add_argument(
        "--hpc-site",
        choices=["default", "sandia", "drac"],
        default="default",
        help=(
            "HPC site flavor. 'default' uses the standard SLURM/Flux script "
            "(conda + constraint=standard). 'sandia' (CTS1/TLCC2) uses "
            "module load + OMPI MCA env + --partition. 'drac' (Digital Research "
            "Alliance of Canada) uses module-system ORCA via $EBROOTORCA/orca "
            "with no qos/partition/MCA. 'sandia'/'drac' are SLURM-only."
        ),
    )
    parser.add_argument(
        "--partition",
        default=None,
        help=(
            "SLURM partition (only used with --hpc-site sandia; defaults to "
            f"{SANDIA_DEFAULT_PARTITION})."
        ),
    )
    parser.add_argument(
        "--openmpi-module",
        default=SANDIA_DEFAULT_OPENMPI_MODULE,
        help=(
            "module load argument matching ORCA's shared build "
            f"(only used with --hpc-site sandia; default: {SANDIA_DEFAULT_OPENMPI_MODULE})."
        ),
    )
    parser.add_argument(
        "--module-load",
        default=DRAC_DEFAULT_MODULE_LOAD,
        help=(
            "module load chain for --hpc-site drac; must end in the ORCA module "
            f"(default: '{DRAC_DEFAULT_MODULE_LOAD}')."
        ),
    )
    parser.add_argument(
        "--venv-path",
        default=None,
        help=(
            "virtualenv to activate in --hpc-site drac job scripts (built by "
            "examples/drac/setup_venv.sh). Required only for --optimizer sella."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare jobs but don't submit",
    )
    parser.add_argument(
        "--reroot",
        action="store_true",
        help=(
            "Ignore job_dir paths stored in the database and always "
            "construct paths from root_dir. Useful when the database "
            "has been moved to a different directory."
        ),
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
        default=72000,
        help="Job timeout in seconds for Parsl mode (default: 72000 = 20 hours)",
    )
    parsl_group.add_argument(
        "--no-parsl-monitoring",
        action="store_true",
        default=False,
        help="Disable Parsl MonitoringHub (skip monitoring.db). Useful on systems "
        "where the monitoring hub port or SQLAlchemy dependency causes issues.",
    )
    parsl_group.add_argument(
        "--clean-on-complete",
        action="store_true",
        default=False,
        help="After each job completes, remove scratch files (.tmp, .core, "
        "orca_tmp_*/) and basis-set files (.bas, .basN) from its job_dir. "
        "Equivalent to running `clean.py --clean-all --execute` inline. "
        "Parsl mode only.",
    )
    parsl_group.add_argument(
        "--purge-on-fail",
        action="store_true",
        default=False,
        help="After each job fails, write a .do_not_rerun.json marker with "
        "failure metadata into its job_dir and delete all other contents. "
        "Equivalent to running `clean.py --purge-failed --execute` inline. "
        "Parsl mode only.",
    )

    # W&B options (--use-parsl only)
    add_wandb_args(parser)

    # Scale-out Parsl options (--use-parsl --scheduler slurm|pbspro)
    scaleout_parsl_group = parser.add_argument_group(
        "Scale-Out Parsl Options (--use-parsl --scheduler slurm|pbspro)"
    )
    scaleout_parsl_group.add_argument(
        "--nodes-per-block",
        type=int,
        default=1,
        help="Nodes per scheduler block. >1 enables multi-node blocks. "
        "Total capacity = max_blocks * nodes_per_block * max_workers. (default: 1)",
    )
    scaleout_parsl_group.add_argument(
        "--max-blocks",
        type=int,
        default=10,
        help="Maximum scheduler blocks to provision (default: 10)",
    )
    scaleout_parsl_group.add_argument(
        "--init-blocks",
        type=int,
        default=2,
        help="Number of scheduler blocks to request at startup (default: 2)",
    )
    scaleout_parsl_group.add_argument(
        "--min-blocks",
        type=int,
        default=1,
        help="Minimum scheduler blocks to keep alive (default: 1)",
    )
    scaleout_parsl_group.add_argument(
        "--walltime-hours",
        type=int,
        default=2,
        help="Walltime per scheduler block allocation in hours (default: 2)",
    )
    scaleout_parsl_group.add_argument(
        "--cpus-per-node",
        type=int,
        default=None,
        help="Scheduler CPU cores reserved per node in Parsl scale-out mode. "
        "Defaults to max_workers * cores_per_worker. Useful on systems that "
        "require full-node requests while intentionally leaving some cores idle.",
    )
    scaleout_parsl_group.add_argument(
        "--qos",
        default="frontier",
        help="SLURM QOS (default: frontier)",
    )
    scaleout_parsl_group.add_argument(
        "--account",
        default="ODEFN5169CYFZ",
        help="Scheduler account/allocation for Parsl mode (default: ODEFN5169CYFZ)",
    )

    parser.add_argument(
        "--max-fail-count",
        type=int,
        default=None,
        help="Skip jobs that have failed this many times or more",
    )

    parser.add_argument(
        "--max-atoms",
        type=int,
        default=None,
        help="Only submit molecules with natoms <= N. Jobs above the cap are "
        "left in their current status for a later batch.",
    )

    parser.add_argument(
        "--min-atoms",
        type=int,
        default=None,
        help="Only submit molecules with natoms >= N. Jobs below the floor are "
        "left in their current status for another lane. Combine with --max-atoms "
        "to run a closed size band [min, max] as its own lane (concurrent lanes).",
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
        choices=["omol", "omol_base", "x2c", "dk3", "pm3"],
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
        "--mbis",
        action="store_true",
        help="Enable MBIS population analysis",
    )
    orca_group.add_argument(
        "--kdiis",
        action="store_true",
        help="Enable KDIIS SCF convergence acceleration",
    )
    orca_group.add_argument(
        "--optimizer",
        choices=["orca", "sella"],
        default=None,
        help="Geometry optimizer: 'orca' (native), 'sella' (external ASE). Default: None (single-point).",
    )
    orca_group.add_argument(
        "--opt-level",
        choices=["loose", "normal", "tight", "verytight"],
        default="normal",
        help="ORCA optimization convergence level (only with --optimizer orca). Default: normal.",
    )
    orca_group.add_argument(
        "--fmax",
        type=float,
        default=0.05,
        help="Sella force convergence threshold in Eh/Bohr (only with --optimizer sella). Default: 0.05.",
    )
    orca_group.add_argument(
        "--max-opt-steps",
        type=int,
        default=None,
        help="Maximum Sella optimization steps (only with --optimizer sella). Default: 100.",
    )
    orca_group.add_argument(
        "--save-all-steps",
        action="store_true",
        help="Save all ORCA output files per Sella step into step_NNN/ directories (only with --optimizer sella).",
    )
    orca_group.add_argument(
        "--ks-method",
        type=str,
        choices=["rks", "uks", "roks"],
        default=None,
        help="KS wavefunction type: rks (restricted), uks (unrestricted), roks (restricted open-shell). Default: ORCA auto-detects.",
    )
    orca_group.add_argument(
        "--orca-path",
        default=None,
        help="Path to ORCA executable (default: scheduler-specific)",
    )

    args = parser.parse_args()

    if args.max_atoms is not None and args.max_atoms < 1:
        parser.error("--max-atoms must be a positive integer (>= 1)")

    if args.min_atoms is not None and args.min_atoms < 1:
        parser.error("--min-atoms must be a positive integer (>= 1)")

    if (
        args.min_atoms is not None
        and args.max_atoms is not None
        and args.min_atoms > args.max_atoms
    ):
        parser.error(
            f"--min-atoms ({args.min_atoms}) must be <= "
            f"--max-atoms ({args.max_atoms})"
        )

    # Validate multi-node Parsl args
    if args.nodes_per_block < 1:
        parser.error("--nodes-per-block must be >= 1")
    if args.scheduler == "flux" and args.nodes_per_block > 1:
        parser.error(
            "--nodes-per-block > 1 is only supported with --scheduler slurm or pbspro. "
            "Flux uses LocalProvider which does not support multi-node blocks."
        )
    if args.scheduler == "pbspro" and not args.use_parsl:
        parser.error("--scheduler pbspro is currently only supported with --use-parsl")
    if (
        args.scheduler in {"slurm", "pbspro"}
        and args.cpus_per_node is not None
        and args.cpus_per_node < args.max_workers * args.cores_per_worker
    ):
        parser.error(
            "--cpus-per-node must be >= max_workers * cores_per_worker in Slurm/PBS Pro mode"
        )

    # Validate HPC site args
    if args.hpc_site == "drac":
        if args.scheduler != "slurm":
            parser.error("--hpc-site drac requires --scheduler slurm")
        if args.use_parsl and not args.orca_path:
            parser.error(
                "--hpc-site drac with --use-parsl requires --orca-path: the "
                "ASE/quacc calculator needs a concrete binary. Resolve it in the "
                "launch script after `module load` and pass --orca-path "
                '"$EBROOTORCA/orca".'
            )
        if args.allocation == "dnn-sim":
            parser.error(
                "--hpc-site drac requires --allocation: the default 'dnn-sim' is "
                "LLNL-only. Pass your DRAC account, e.g. --allocation def-<pi>."
            )

    if args.hpc_site == "sandia":
        if not args.use_parsl and args.scheduler != "slurm":
            parser.error(
                "--hpc-site sandia (traditional mode) requires --scheduler slurm"
            )
        if not args.orca_path:
            parser.error(
                "--hpc-site sandia requires --orca-path: Sandia has no shared "
                "ORCA install. Point at your user-installed shared build, e.g. "
                "--orca-path $HOME/orca_6_1_0_linux_x86-64_shared_openmpi418/orca"
            )
        # Swap LLNL-specific defaults for Sandia equivalents when not overridden.
        # The argparse default for --queue is the LLNL value "pbatch" and for
        # --allocation is "dnn-sim"; both fail outright at sbatch time on Sandia.
        if args.queue == "pbatch":
            args.queue = SANDIA_DEFAULT_QOS
        if args.allocation == "dnn-sim":
            args.allocation = SANDIA_DEFAULT_ACCOUNT

    # Validate optimizer-specific args
    if args.optimizer == "orca" and args.max_opt_steps is not None:
        parser.error("--max-opt-steps is only valid with --optimizer sella")
    if args.optimizer == "orca" and args.fmax != 0.05:
        parser.error("--fmax is only valid with --optimizer sella")
    if args.save_all_steps and args.optimizer != "sella":
        parser.error("--save-all-steps is only valid with --optimizer sella")
    if args.optimizer == "sella" and args.opt_level != "normal":
        parser.error("--opt-level is only valid with --optimizer orca")
    if args.optimizer is None and args.opt_level != "normal":
        parser.error("--opt-level requires --optimizer orca")

    try:
        effective_job_dir_pattern = apply_job_dir_prefix(
            args.job_dir_pattern, args.job_prefix
        )
    except ValueError as exc:
        parser.error(str(exc))

    # Build ORCA config from CLI arguments
    orca_config: OrcaConfig = {
        "functional": args.functional,
        "simple_input": args.simple_input,
        "actinide_basis": args.actinide_basis,
        "actinide_ecp": (
            args.actinide_ecp if args.actinide_ecp.lower() != "none" else None
        ),  # deal with None string, convert to NoneType
        "non_actinide_basis": args.non_actinide_basis,
        "scf_MaxIter": args.scf_maxiter,
        "nbo": args.nbo,
        "mbis": args.mbis,
        "optimizer": args.optimizer,
        "opt_level": args.opt_level,
        "fmax": args.fmax,
        "max_opt_steps": args.max_opt_steps,
        "save_all_steps": args.save_all_steps,
        "diis_option": "KDIIS" if args.kdiis else None,
        "ks_method": args.ks_method,
        "mem_per_job_mb": args.mem_per_job,
    }
    if args.orca_path:
        orca_config["orca_path"] = args.orca_path

    # Open workflow (higher retry budget for Parsl to survive Lustre contention)
    try:
        if args.use_parsl:
            workflow = ArchitectorWorkflow(
                args.db_path,
                timeout=15.0,
                max_retries=10,
                retry_delay_cap=10.0,
            )
        else:
            workflow = ArchitectorWorkflow(args.db_path)
    except FileNotFoundError:
        print(f"Error: Database not found at {args.db_path}")
        sys.exit(1)

    # Validate W&B options are only used with Parsl mode
    if args.wandb_project and not args.use_parsl:
        parser.error("--wandb-project requires --use-parsl")

    # Inline cleanup hooks only fire from the Parsl completion loop;
    # traditional mode submits and exits before any job finishes.
    if (args.clean_on_complete or args.purge_on_fail) and not args.use_parsl:
        parser.error("--clean-on-complete / --purge-on-fail require --use-parsl")

    # Submit based on mode
    if args.use_parsl:
        # Initialize W&B run if requested
        wandb_run = None
        if args.wandb_project:
            if not WANDB_AVAILABLE:
                print(
                    "Warning: wandb not installed; --wandb-project ignored. "
                    "Install with: pip install wandb"
                )
            else:
                wandb_run = init_wandb_run(
                    project=args.wandb_project,
                    run_name=args.wandb_run_name or Path(args.db_path).stem,
                    run_id=args.wandb_run_id,
                )
                # Seed terminal-state CDF curves with all prior history from
                # the DB. Cheap one-shot scan; emits one log per terminal row.
                backfill_terminal_cdfs(wandb_run, workflow)

        # Parsl mode: concurrent execution (single-node or multi-node)
        submitted_ids = submit_batch_parsl(
            workflow=workflow,
            root_dir=args.root_dir,
            num_jobs=args.batch_size,
            max_workers=args.max_workers,
            cores_per_worker=args.cores_per_worker,
            scheduler=args.scheduler,
            job_dir_pattern=effective_job_dir_pattern,
            orca_config=orca_config,
            n_cores=args.n_cores,
            conda_env=args.conda_env,
            conda_base=args.conda_base,
            ld_library_path=args.ld_library_path,
            dry_run=args.dry_run,
            max_fail_count=args.max_fail_count,
            max_atoms=args.max_atoms,
            min_atoms=args.min_atoms,
            timeout_seconds=args.job_timeout,
            randomize=True,
            cpus_per_node=args.cpus_per_node,
            nodes_per_block=args.nodes_per_block,
            max_blocks=args.max_blocks,
            init_blocks=args.init_blocks,
            min_blocks=args.min_blocks,
            walltime_hours=args.walltime_hours,
            qos=args.qos,
            account=args.account,
            enable_monitoring=not args.no_parsl_monitoring,
            wandb_run=wandb_run,
            reroot=args.reroot,
            site=args.hpc_site,
            partition=args.partition,
            openmpi_module=args.openmpi_module,
            clean_on_complete=args.clean_on_complete,
            purge_on_fail=args.purge_on_fail,
        )
    else:
        # Traditional mode: one job script per job
        submitted_ids = submit_batch(
            workflow=workflow,
            root_dir=args.root_dir,
            batch_size=args.batch_size,
            scheduler=args.scheduler,
            job_dir_pattern=effective_job_dir_pattern,
            orca_config=orca_config,
            n_cores=args.n_cores,
            n_hours=args.n_hours,
            queue=args.queue,
            allocation=args.allocation,
            conda_env=args.conda_env,
            ld_library_path=args.ld_library_path,
            dry_run=args.dry_run,
            max_fail_count=args.max_fail_count,
            max_atoms=args.max_atoms,
            min_atoms=args.min_atoms,
            randomize=True,
            reroot=args.reroot,
            site=args.hpc_site,
            partition=args.partition,
            openmpi_module=args.openmpi_module,
            module_load=args.module_load,
            venv_path=args.venv_path,
            mem_per_cpu=args.mem_per_cpu,
        )

    print(f"\nTotal jobs submitted: {len(submitted_ids)}")

    workflow.close()


if __name__ == "__main__":
    main()
