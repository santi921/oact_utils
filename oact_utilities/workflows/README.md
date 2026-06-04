# Architector Workflow Manager

High-throughput workflow management for architector calculations on HPC systems.

## Features

- **Job Tracking**: SQLite database tracks job status (to_run, ready, running, completed, failed, timeout)
- **Timeout Detection**: Automatically identifies jobs stuck for 6+ hours
- **Metrics Collection**: Automatically extracts max forces, SCF steps, final energy from ORCA outputs
- **Performance Tracking**: Records wall time and cores used for compute usage analysis (core-hours)
- **Concurrency Handling**: WAL mode + retry logic for concurrent database access
- **HPC Integration**: Works with existing Flux/SLURM job generation utilities
- **Parsl Mode**: Concurrent execution on exclusive nodes with real-time status updates
- **W&B Integration**: Optional Weights & Biases logging for team-visible campaign dashboards
- **Dashboard**: Command-line monitoring with metrics display
- **Batch Submission**: Submit jobs in batches with automatic status updates (traditional or Parsl mode)
- **Error Handling**: Robust parsers handle missing files and gzipped outputs (quacc)

## Quick Start

### 1. Initialize Workflow

```python
from oact_utilities.utils.architector import create_workflow_db

# Create SQLite database directly from architector CSV
db_path = create_workflow_db(
    csv_path="architector_output.csv",
    db_path="workflow.db",
    geometry_column="aligned_csd_core",
    charge_column="charge",
    spin_column="uhf",  # Unpaired electrons (converted to multiplicity)
)
```

This creates:
- `workflow.db` - SQLite database with job tracking and metrics columns

### 2. Submit Jobs to HPC

#### A. Traditional Mode (Individual Flux/SLURM Jobs)

```bash
# Basic submission with default ORCA settings
python -m oact_utilities.workflows.submit_jobs \\
    workflow.db \\
    jobs/ \\
    --batch-size 100 \\
    --scheduler flux \\
    --n-cores 4 \\
    --n-hours 2 \\
    --queue pbatch \\
    --allocation dnn-sim

# With custom ORCA configuration
python -m oact_utilities.workflows.submit_jobs \\
    workflow.db \\
    jobs/ \\
    --batch-size 100 \\
    --functional wB97M-V \\
    --simple-input x2c \\
    --opt  # Enable geometry optimization

# Skip jobs that have already failed 3+ times
python -m oact_utilities.workflows.submit_jobs \\
    workflow.db \\
    jobs/ \\
    --batch-size 100 \\
    --max-fail-count 3

# Stratify by molecule size: small molecules in a short allocation,
# large ones deferred to a longer one. Over-cap jobs stay TO_RUN for a later batch.
python -m oact_utilities.workflows.submit_jobs \\
    workflow.db \\
    jobs/ \\
    --max-atoms 40 \\
    --n-hours 2
```

Creates job directories `jobs/job_0/`, `jobs/job_1/`, etc. with:
- `orca.inp` - Complete ORCA input file (geometry + level of theory)
- `flux_job.flux` or `slurm_job.sh` - Submission script that runs ORCA directly
- Submits to scheduler and marks jobs as "running"

#### B. Parsl Mode (Concurrent Execution on Exclusive Node)

Use [Parsl](https://parsl-project.org/) to run multiple ORCA jobs concurrently on a single allocated node:

```bash
# 1. Request exclusive node via Flux
flux alloc -N 1 -n 64 -q pbatch -t 8h -B dnn-sim

# 2. Inside allocation: Run jobs concurrently with Parsl
python -m oact_utilities.workflows.submit_jobs \\
    workflow.db \\
    jobs/ \\
    --use-parsl \\
    --batch-size 100 \\
    --max-workers 4 \\
    --cores-per-worker 16 \\
    --n-cores 16 \\
    --job-timeout 7200

# With custom ORCA settings + longer timeout
python -m oact_utilities.workflows.submit_jobs \\
    workflow.db \\
    jobs/ \\
    --use-parsl \\
    --batch-size 200 \\
    --max-workers 4 \\
    --cores-per-worker 16 \\
    --functional wB97M-V \\
    --opt \\
    --job-timeout 14400  # 4 hours per job
```

**How Parsl Mode Works:**
1. Creates job directories with ORCA input files (same as traditional mode)
2. Initializes Parsl with `HighThroughputExecutor` on the local node
3. Submits all jobs as Parsl `python_app` futures
4. Parsl manages worker processes and executes ORCA jobs concurrently
5. Database updates happen in real-time as each job completes
6. Graceful shutdown on Ctrl+C or completion

**Parsl Mode Benefits:**
- ✅ **High throughput**: Run 4+ jobs simultaneously on one node
- ✅ **Real-time monitoring**: See jobs complete live with status updates
- ✅ **Efficient resource use**: Better node utilization vs. sequential execution
- ✅ **Automatic CPU pinning**: Parsl handles core affinity
- ✅ **Fault tolerance**: Individual job failures don't crash the workflow
- ✅ **Live database updates**: Status synced as jobs finish (not batch-updated later)

**When to use Parsl mode:**
- ✅ You have an exclusive node allocation (e.g., `flux alloc`)
- ✅ Running many short-medium jobs (< 4 hours each)
- ✅ Want to maximize node utilization and throughput
- ✅ Need real-time progress feedback
- ❌ Don't use for long jobs (> 8 hours) - use traditional mode instead
- ❌ Don't use if jobs have wildly different runtimes (load balancing issues)

**ORCA Configuration Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--functional` | wB97M-V | DFT functional |
| `--simple-input` | omol | Input template: `omol`, `omol_base`, `x2c`, `dk3`, or `pm3` |
| `--actinide-basis` | ma-def-TZVP | Basis set for actinides |
| `--actinide-ecp` | def-ECP | ECP for actinides. Pass `none` (case-insensitive) to disable. |
| `--non-actinide-basis` | def2-TZVPD | Basis set for other elements |
| `--scf-maxiter` | ORCA default | Maximum SCF iterations |
| `--ks-method` | None | KS wavefunction: `rks`, `uks`, `roks` (None = ORCA auto-detect) |
| `--nbo` | False | Enable NBO analysis |
| `--mbis` | False | Enable MBIS population analysis |
| `--kdiis` | False | Use KDIIS SCF convergence accelerator |
| `--optimizer` | None | Geometry optimizer: `orca` (native) or `sella` (external ASE) |
| `--mem-per-job` | None | Total-job memory budget (MB). Sizes `%maxcore` per MPI rank under 85% of this value. Recommended on memory-constrained nodes (Sandia CTS-1: ~60000, TLCC2: ~30000) |
| `--orca-path` | scheduler-specific | Path to ORCA executable |
| `--conda-env` | py10mpi | Conda environment to activate |

**Parsl Mode Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--use-parsl` | False | Enable Parsl concurrent execution mode |
| `--max-workers` | 4 | Maximum number of concurrent Parsl workers |
| `--cores-per-worker` | 16 | CPU cores allocated per worker |
| `--n-cores` | auto (matches `cores_per_worker`) | Cores per ORCA job (auto-set to `cores_per_worker` if mismatch) |
| `--job-timeout` | 72000 | Per-job timeout in seconds (20 hours) |
| `--conda-base` | /usr/WS1/vargas58/miniconda3 | Conda base path for worker init |

**Scale-Out Parsl Options (`--use-parsl --scheduler slurm|pbspro`):**

Parsl provisions worker allocations ("blocks") on the scheduler's behalf. Total capacity is `max_blocks * nodes_per_block * max_workers`.

| Option | Default | Description |
|--------|---------|-------------|
| `--nodes-per-block` | 1 | Nodes per scheduler block (>1 enables multi-node blocks with SrunLauncher) |
| `--max-blocks` | 10 | Maximum scheduler blocks Parsl will provision |
| `--init-blocks` | 2 | Blocks to request at startup |
| `--min-blocks` | 1 | Minimum blocks to keep alive |
| `--walltime-hours` | 2 | Walltime per block allocation (hours) |
| `--cpus-per-node` | `max_workers * cores_per_worker` | Scheduler CPU cores reserved per node (useful when a system requires whole-node requests but you intentionally idle some cores) |
| `--qos` | frontier | SLURM QOS |
| `--account` | ODEFN5169CYFZ | Scheduler account/allocation |
| `--mpirun-path` | autodetect via `which mpirun` | PBS Pro only: override mpirun discovery when the queue's default module set lacks it |

For single-node Parsl runs (`--max-blocks 1`), use the single-node launch scripts under `launch/` and run Parsl with `LocalProvider` inside an existing allocation.

### HPC Site Profiles

`--hpc-site` selects a per-site bundle of defaults (modules, partition, MPI env, ORCA path) so a single submit command works on every cluster. Only `slurm` schedulers consult this flag.

| `--hpc-site` | Job-script writer | When to use |
|--------------|-------------------|-------------|
| `default` | `write_slurm_job_file()` | Generic SLURM with `conda activate` + `--constraint=standard` (LLNL-style). ORCA expected on `PATH` unless `--orca-path` is set. |
| `sandia` | `write_slurm_sandia_job_file()` | Sandia CTS1/TLCC2 (attaway/ecl). Emits `module load <openmpi>`, sets `OMPI_MCA_pml/mtl/btl` to bypass PSM2/Omni-Path, requests `--partition`, and points at the user-installed shared ORCA build. |

**Sandia defaults** (overridable via CLI):

| Flag | Default | Notes |
|------|---------|-------|
| `--partition` | `attaway` | SLURM partition. CTS1 uses `attaway`; TLCC2 partitions differ. |
| `--openmpi-module` | `aue/openmpi/4.1.6-gcc-12.3.0` | Must match ORCA's shared-library build. |
| `--allocation` | `fy250086` | Sandia account. |
| `--queue` (QOS) | `normal` | |
| `--n-cores` | 36 | CTS1 attaway = 36 cores/node; TLCC2 = 16. |

**Sandia launch scripts** (under `oact_utilities/launch/`):

- `run_parsl_single_node_sandia.sh` -- LocalProvider inside an existing `salloc`. Best for development and short campaigns. Includes `--max-blocks 1` to pin to a single allocation.
- `run_parsl_multi_node_sandia.sh` -- Multi-block SlurmProvider. Parsl auto-provisions worker allocations; the coordinator script itself can run on a login node or as a small sbatch job, and only needs Python (no OpenMPI in the coordinator's env).

```bash
# Single-node, inside an interactive allocation
salloc -N1 -p attaway -A fy250086 -t 8:00:00
conda activate oact
bash oact_utilities/launch/run_parsl_single_node_sandia.sh

# Multi-block (Parsl provisions its own SLURM blocks)
sbatch oact_utilities/launch/run_parsl_multi_node_sandia.sh
```

**Adding a new HPC site:** mirror the Sandia pattern in `submit_jobs.py`:

1. Add `<SITE>_DEFAULT_*` constants near the existing `SANDIA_DEFAULT_*` block.
2. Add a `write_slurm_<site>_job_file()` if the site needs module loads / MPI env / partition not covered by the default writer.
3. Add `_build_parsl_<site>_worker_init*()` and a `build_parsl_config_<site>()` if multi-block Parsl behavior diverges from the generic SlurmProvider.
4. Extend the `--hpc-site` choices and the dispatch in `submit_batch_traditional()` / `submit_batch_parsl()`.
5. Add launch script templates under `launch/` (single-node and multi-block).

### PBS Pro Multi-Node Parsl

`--scheduler pbspro` selects `PBSProProvider` (via `build_parsl_config_pbspro()`). The per-block resource request is derived from `nodes_per_block` and the worker count; PBS Pro has no `exclusive` keyword, so request whole nodes via `--cpus-per-node`. `--mpirun-path` is recommended on systems where the default module set omits an MPI runtime from the coordinator's `PATH`.

Orphan recovery (`dashboard.py --recover-orphans --scheduler pbspro`) is supported: the dashboard queries the active PBS Pro job set and resets molecules whose `worker_id` no longer exists.

### 3. Monitor Progress

```bash
# Basic status summary
python -m oact_utilities.workflows.dashboard workflow.db

# Update statuses by scanning job directories
python -m oact_utilities.workflows.dashboard workflow.db --update jobs/

# Show computational metrics
python -m oact_utilities.workflows.dashboard workflow.db --show-metrics

# Show failed jobs (includes fail count)
python -m oact_utilities.workflows.dashboard workflow.db --show-failed

# Show timeout jobs (stuck for 6+ hours)
python -m oact_utilities.workflows.dashboard workflow.db --show-timeout

# Reset failed jobs to retry (increments fail_count)
python -m oact_utilities.workflows.dashboard workflow.db --reset-failed

# Reset timeout jobs to retry (increments fail_count)
python -m oact_utilities.workflows.dashboard workflow.db --reset-timeout

# Reset both failed and timeout jobs together
python -m oact_utilities.workflows.dashboard workflow.db --reset-failed --include-timeout-in-reset

# Reset failed jobs, but skip those that have failed 3+ times
python -m oact_utilities.workflows.dashboard workflow.db --reset-failed --max-retries 3

# Show jobs that have failed multiple times (chronic failures)
python -m oact_utilities.workflows.dashboard workflow.db --show-chronic-failures 3
```

### W&B Integration (Online Monitoring)

Stream job progress and campaign metrics to [Weights & Biases](https://wandb.ai) for a team-visible live dashboard:

```bash
# During Parsl submission -- logs each job outcome in real-time
python -m oact_utilities.workflows.submit_jobs \
    workflow.db \
    jobs/ \
    --use-parsl \
    --max-workers 4 \
    --wandb-project actinide-campaign \
    --wandb-run-name wave_two

# During dashboard scans -- logs aggregate campaign snapshot
python -m oact_utilities.workflows.dashboard \
    workflow.db \
    --update jobs/ \
    --extract-metrics \
    --wandb-project actinide-campaign \
    --wandb-run-id <run-id>  # Reuse the same W&B run across both CLIs
```

**W&B is optional.** If wandb is not installed or `--wandb-project` is not passed, nothing changes. All W&B calls are wrapped in try/except -- a failure never aborts a campaign.

**Installation:** `pip install wandb && wandb login`

| Key namespace | Logged by | Description |
|---------------|-----------|-------------|
| `progress/completed` | submit_jobs | +1 per completed job |
| `progress/failed` | submit_jobs | +1 per failed job |
| `metrics/max_forces` | submit_jobs | Per-job max gradient |
| `campaign/completed` | dashboard | Total completed count |
| `campaign/progress_pct` | dashboard | % complete |
| `metrics/wall_time_total_hours` | dashboard | Aggregate wall time |
| `metrics/core_hours_total` | dashboard | Total core-hours consumed |

See `docs/parsl_integration.md` for full W&B documentation.

## Database Schema

The SQLite database tracks each structure with:

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key (auto-increment) |
| `orig_index` | INTEGER | Original row index from CSV |
| `elements` | TEXT | Semicolon-separated element symbols |
| `natoms` | INTEGER | Number of atoms |
| `status` | TEXT | Job status: to_run, running, completed, failed, timeout (legacy "ready" auto-migrated) |
| `charge` | INTEGER | Molecular charge (optional) |
| `spin` | INTEGER | Spin multiplicity (2S+1) |
| `geometry` | TEXT | XYZ geometry string |
| `job_dir` | TEXT | Path to job directory |
| `max_forces` | REAL | Maximum force from optimization (Eh/Bohr) |
| `scf_steps` | INTEGER | Total SCF iterations |
| `final_energy` | REAL | Final energy (Hartree) |
| `error_message` | TEXT | Error message if failed |
| `fail_count` | INTEGER | Number of times job has failed (for retry tracking) |
| `wall_time` | REAL | Total wall time in seconds (extracted from ORCA output) |
| `n_cores` | INTEGER | Number of CPU cores used |
| `worker_id` | TEXT | Scheduler job ID (SLURM/Flux) for crash recovery |
| `created_at` | TIMESTAMP | Creation time |
| `updated_at` | TIMESTAMP | Last update time |

**Indexes:** `idx_status`, `idx_orig_index` for fast queries.

## Job Statuses

- **`to_run`**: Job is ready to be submitted (default for new jobs)
- **`ready`**: Legacy status, auto-migrated to `to_run` when the database is opened
- **`running`**: Job has been submitted to HPC scheduler and is executing. The `worker_id` column stores the scheduler job ID (SLURM/Flux) for crash recovery.
- **`completed`**: Job finished successfully
- **`failed`**: Job crashed or failed convergence (explicit error or abort)
- **`timeout`**: Job timed out (file not modified for `--hours-cutoff` hours, default 24, likely stuck or walltime exceeded)

## Parsl Mode Architecture

The Parsl integration provides a Python-native workflow executor for high-throughput ORCA calculations:

### How It Works

1. **Job Preparation**: Creates job directories with ORCA input files (same as traditional mode)
2. **Parsl Initialization**: Configures `HighThroughputExecutor` with `LocalProvider` (Flux) or `SlurmProvider` (SLURM multi-node)
   - **Flux**: Uses local node resources only; workers run on the allocated node
   - **SLURM**: Auto-provisions additional nodes via SLURM scheduler; scales out across multiple nodes
   - Each worker handles one ORCA job at a time
3. **Job Submission**: Wraps each ORCA run in a `python_app` decorated function
4. **Concurrent Execution**: Parsl schedules jobs across workers using `concurrent.futures`
5. **Real-Time Updates**: As jobs complete, database status is updated immediately
6. **Graceful Shutdown**: Supports Ctrl+C with Parsl cleanup

### Parsl Configuration Details

The `build_parsl_config_flux()` function creates a Parsl `Config` with:

```python
Config(
    executors=[
        HighThroughputExecutor(
            label="flux_htex",
            cores_per_worker=16,      # CPU cores per worker
            max_workers=4,            # Concurrent workers
            provider=LocalProvider(   # Single-node execution
                worker_init="""
                    source ~/.bashrc
                    conda activate py10mpi
                    export LD_LIBRARY_PATH=...
                    export OMP_NUM_THREADS=1
                    export JAX_PLATFORMS=cpu
                """
            )
        )
    ]
)
```

**Key Design Choices:**
- **LocalProvider (Flux)**: Uses the current allocated node only. Pass `--scheduler flux` (default).
- **SlurmProvider (SLURM)**: Auto-provisions additional nodes via SLURM. Pass `--use-parsl --scheduler slurm`. Each provisioned node runs `max_workers` concurrent ORCA jobs.
- **cores_per_worker**: CPU cores allocated per worker by Parsl
- **n_cores**: Automatically set to match `cores_per_worker` if not specified (ensures proper resource allocation)
- **max_workers**: Set based on node capacity (e.g., 64 cores → 4 workers × 16 cores each)
- **worker_init**: Ensures workers have conda environment and library paths

### Execution Flow

```python
# Simplified execution flow in submit_batch_parsl()

# 1. Prepare directories
for job in jobs_to_submit:
    prepare_job_directory(job, root_dir, orca_config, n_cores)

# 2. Submit Parsl futures
futures = []
for job in jobs_to_submit:
    future = orca_job_wrapper(job.id, job_dir, orca_config, n_cores, timeout)
    futures.append((job.id, future))

# 3. Monitor with as_completed() for concurrent handling
for future in as_completed(futures_map.keys()):
    result = future.result()
    if result["status"] == "completed":
        workflow.update_status(job_id, JobStatus.COMPLETED)
    # ... handle failures, timeouts
```

**Critical Implementation Detail:**
Uses `concurrent.futures.as_completed()` to process results as they finish (not sequentially). This is essential for true concurrent execution.

### Resource Calculation

**Node Capacity Check:**
```
Total cores = max_workers × cores_per_worker × n_cores

Example: 4 workers × 16 cores = 64 cores
```

Make sure this doesn't exceed your node allocation. For a 64-core node:
- ✅ 4 workers × 16 cores = 64 cores (full utilization)
- ✅ 2 workers × 32 cores = 64 cores (fewer, larger jobs)
- ❌ 8 workers × 16 cores = 128 cores (oversubscribed!)

### Error Handling

Parsl mode includes robust error handling:

1. **Import Check**: Gracefully disables Parsl features if not installed
2. **Job Timeout**: Per-job timeout prevents stuck jobs from blocking workers
3. **Worker Failures**: Individual job failures don't crash the executor
4. **Keyboard Interrupt**: Ctrl+C triggers graceful shutdown with cleanup
5. **Database Locking**: Retry logic handles concurrent DB access

### Performance Comparison

**Traditional Mode (Sequential):**
```
Job 1: ████████ (2 hours)
Job 2:         ████████ (2 hours)
Job 3:                 ████████ (2 hours)
Total: 6 hours for 3 jobs
```

**Parsl Mode (Concurrent, 3 workers):**
```
Job 1: ████████ (2 hours)
Job 2: ████████ (2 hours)
Job 3: ████████ (2 hours)
Total: 2 hours for 3 jobs
```

3× speedup with 3 workers (assuming jobs have similar runtime).

### Installation

Parsl is an optional dependency:

```bash
pip install 'parsl>=2024.1'

# Or install with oact_utilities
pip install -e ".[parsl]"  # if added to pyproject.toml
```

The package works without Parsl (traditional mode only) if not installed.

## Programmatic API

### Traditional Mode

```python
from oact_utilities.workflows import ArchitectorWorkflow, JobStatus

with ArchitectorWorkflow("workflow.db") as workflow:

    # Get jobs by status
    ready = workflow.get_jobs_by_status(JobStatus.TO_RUN)
    completed = workflow.get_jobs_by_status(JobStatus.COMPLETED)

    # Count jobs
    counts = workflow.count_by_status()
    # {'to_run': 1000, 'running': 200, 'completed': 50, 'failed': 8, 'timeout': 2}

    # Update job status
    workflow.update_status(job_id=42, new_status=JobStatus.COMPLETED)

    # Update job metrics
    workflow.update_job_metrics(
        job_id=42,
        job_dir="/path/to/job_42",
        max_forces=0.00123,
        scf_steps=15,
        final_energy=-1234.56
    )

    # Bulk update
    workflow.update_status_bulk([1, 2, 3], JobStatus.RUNNING)

    # Reset failed jobs (increments fail_count)
    workflow.reset_failed_jobs()

    # Reset timeout jobs (increments fail_count)
    workflow.reset_timeout_jobs()

    # Reset both failed and timeout jobs together
    workflow.reset_failed_jobs(include_timeout=True)

    # Reset failed jobs, but only those that haven't failed too many times
    workflow.reset_failed_jobs(max_fail_count=3)  # Skip jobs with fail_count >= 3

    # Find chronically failing jobs
    chronic_failures = workflow.get_jobs_by_fail_count(min_fail_count=3)
```

### Automatic Metrics Extraction

```python
from oact_utilities.workflows import update_job_status

# Automatically extract and store metrics
new_status = update_job_status(
    workflow=workflow,
    job_dir="jobs/job_42/",
    job_id=42,
    extract_metrics=True,  # Extract max_forces, scf_steps, final_energy
    unzip=False,  # Set True for quacc gzipped outputs
)
```

### Parsl Mode (Programmatic)

```python
from oact_utilities.workflows import ArchitectorWorkflow
from oact_utilities.workflows.submit_jobs import submit_batch_parsl, OrcaConfig

# Configure ORCA calculation
orca_config: OrcaConfig = {
    "functional": "wB97M-V",
    "simple_input": "omol",
    "actinide_basis": "ma-def-TZVP",
    "opt": True,
}

with ArchitectorWorkflow("workflow.db") as workflow:
    # Submit jobs using Parsl for concurrent execution
    submitted_ids = submit_batch_parsl(
        workflow=workflow,
        root_dir="jobs/",
        num_jobs=100,                  # Total jobs to run
        max_workers=4,                 # Concurrent workers
        cores_per_worker=16,           # Cores per worker
        orca_config=orca_config,
        n_cores=16,                    # Cores per ORCA job (must match cores_per_worker)
        conda_env="py10mpi",
        conda_base="/usr/WS1/vargas58/miniconda3",
        dry_run=False,
        max_fail_count=3,              # Skip chronic failures
        timeout_seconds=7200,          # 2 hours per job
    )

    print(f"Submitted {len(submitted_ids)} jobs via Parsl")

    # Jobs execute concurrently and update DB in real-time
    # When complete, check results:
    completed = workflow.get_jobs_by_status(JobStatus.COMPLETED)
    failed = workflow.get_jobs_by_status(JobStatus.FAILED)
    print(f"Completed: {len(completed)}, Failed: {len(failed)}")
```

**Key differences from traditional mode:**
- ✅ `submit_batch_parsl()` instead of `submit_batch()`
- ✅ Uses `max_workers` and `cores_per_worker` instead of scheduler queue parameters
- ✅ Jobs execute immediately on current node (no queue submission)
- ✅ Database updates in real-time as jobs complete
- ✅ Blocking call - returns when all jobs finish (or fail/timeout)

## ORCA Configuration

Jobs are configured using the `OrcaConfig` TypedDict:

```python
from oact_utilities.workflows.submit_jobs import submit_batch, OrcaConfig

# Configure ORCA calculation settings
orca_config: OrcaConfig = {
    "functional": "wB97M-V",       # DFT functional
    "simple_input": "x2c",         # Template: "omol", "omol_base", "x2c", or "dk3"
    "actinide_basis": "ma-def-TZVP",
    "actinide_ecp": None,          # Or "def-ECP" for ECP calculations
    "non_actinide_basis": "def2-TZVPD",
    "scf_MaxIter": None,           # Use ORCA default
    "nbo": False,                  # NBO analysis
    "mbis": False,                 # MBIS population analysis
    "opt": True,                   # Geometry optimization
    "diis_option": None,           # Or "KDIIS" for KDIIS convergence accelerator
    "orca_path": "/path/to/orca",  # Optional, defaults per scheduler
}

with ArchitectorWorkflow("workflow.db") as workflow:
    submit_batch(
        workflow=workflow,
        root_dir="jobs/",
        batch_size=100,
        scheduler="flux",
        orca_config=orca_config,
        n_cores=8,
        n_hours=4,
        conda_env="py10mpi",
    )
```

## Custom Job Setup

You can provide a custom setup function for additional files:

```python
from oact_utilities.workflows.submit_jobs import submit_batch

def add_restart_files(job_dir, job_record):
    """Copy restart files or add custom setup."""
    # The orca.inp is already created by submit_batch
    # Add any additional files here
    (job_dir / "notes.txt").write_text(f"Job for structure {job_record.orig_index}")

with ArchitectorWorkflow("workflow.db") as workflow:
    submit_batch(
        workflow=workflow,
        root_dir="jobs/",
        batch_size=100,
        scheduler="flux",
        setup_func=add_restart_files,
        n_cores=8,
        n_hours=4
    )
```

## Dashboard Output Example

```
================================================================================
                          Workflow Status Summary
================================================================================

Status          Count    Percent
----------------------------------------
to_run           8500      85.0%
running          1200      12.0%
completed         250       2.5%
failed             30       0.3%
timeout            20       0.2%
----------------------------------------
TOTAL           10000     100.0%

Completion: [████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 2.5% (250/10000)

================================================================================
              Computational Metrics (Completed Jobs)
================================================================================

Max Forces (Eh/Bohr):
  Mean:   0.001234
  Median: 0.000987
  Min:    0.000012
  Max:    0.004567

SCF Steps:
  Mean:   12.3
  Median: 11
  Min:    5
  Max:    45

Wall Time (seconds):
  Mean:   196.4
  Median: 180.2
  Min:    45.3
  Max:    892.1
  Total:  49100.0 (13.64 hours)

Cores Used:
  Mean:   4.0
  Min:    4
  Max:    4
  Total core-hours: 54.56

Total jobs with metrics: 250
Jobs with timing data: 250
```

## Typical HPC Workflow

1. **Prepare locally**:
   ```bash
   python setup_workflow.py  # Creates workflow.db
   scp workflow.db user@hpc:/project/
   ```

2. **Submit on HPC**:
   ```bash
   python -m oact_utilities.workflows.submit_jobs workflow.db jobs/ --batch-size 500
   ```

3. **Monitor periodically**:
   ```bash
   python -m oact_utilities.workflows.dashboard workflow.db --update jobs/
   ```

4. **Submit more as jobs finish**:
   ```bash
   python -m oact_utilities.workflows.submit_jobs workflow.db jobs/ --batch-size 500
   ```

5. **Handle failures and timeouts**:
   ```bash
   # View failed jobs with fail counts
   python -m oact_utilities.workflows.dashboard workflow.db --show-failed

   # View timeout jobs (stuck for 6+ hours)
   python -m oact_utilities.workflows.dashboard workflow.db --show-timeout

   # Reset failed jobs for retry (increments fail_count)
   python -m oact_utilities.workflows.dashboard workflow.db --reset-failed

   # Reset timeout jobs for retry with longer time limit
   python -m oact_utilities.workflows.dashboard workflow.db --reset-timeout

   # Reset both failed and timeout jobs together
   python -m oact_utilities.workflows.dashboard workflow.db --reset-failed --include-timeout-in-reset

   # Reset only jobs that haven't failed too many times
   python -m oact_utilities.workflows.dashboard workflow.db --reset-failed --max-retries 3

   # View chronically failing jobs (failed 3+ times)
   python -m oact_utilities.workflows.dashboard workflow.db --show-chronic-failures 3

   # Submit jobs but skip chronic failures
   python -m oact_utilities.workflows.submit_jobs workflow.db jobs/ --max-fail-count 3
   ```

## Failure Tracking & Retry Limits

The workflow tracks how many times each job has failed with the `fail_count` column:

- **Automatic increment**: When `reset_failed_jobs()` is called, `fail_count` is incremented
- **Retry limits**: Use `--max-retries N` to skip jobs that have failed N+ times
- **Submission filtering**: Use `--max-fail-count N` to avoid resubmitting chronic failures
- **Size stratification**: Use `--max-atoms N` to submit only molecules with `natoms <= N`; over-cap jobs stay `TO_RUN` for a later, longer-wall-time batch
- **Chronic failure detection**: Use `--show-chronic-failures N` or `get_jobs_by_fail_count(N)`

This prevents wasting HPC resources on jobs that consistently fail.

**Example workflow with retry limits:**
```bash
# First attempt - submit all ready jobs
python -m oact_utilities.workflows.submit_jobs workflow.db jobs/ --batch-size 500

# After jobs complete, update statuses
python -m oact_utilities.workflows.dashboard workflow.db --update jobs/

# Reset failures for retry, but only if they haven't failed 3 times yet
python -m oact_utilities.workflows.dashboard workflow.db --reset-failed --max-retries 3

# Submit again, skipping chronic failures
python -m oact_utilities.workflows.submit_jobs workflow.db jobs/ --batch-size 500 --max-fail-count 3

# Check which jobs are chronically failing
python -m oact_utilities.workflows.dashboard workflow.db --show-chronic-failures 3
```

## Crash Recovery

When a scheduler allocation is killed (walltime limit, scancel, node crash), jobs can get stuck in RUNNING status. The system provides three layers of recovery:

**Layer 1 -- SIGINT (Ctrl+C):** The existing `KeyboardInterrupt` handler resets orphaned jobs in the `finally` block.

**Layer 2 -- SIGTERM (scancel, walltime):** A flag-based SIGTERM handler is registered after `parsl.load()`. When SLURM sends SIGTERM (~30s before SIGKILL), the handler sets a shutdown flag. The monitoring loop checks this flag between jobs and exits cleanly, triggering the bulk orphan reset in `finally`.

**Layer 3 -- SIGKILL / node crash:** When no Python cleanup is possible, use the dashboard's `--recover-orphans` command. It queries the scheduler to find dead allocations and recovers their jobs:

```bash
# Check SLURM for dead allocations and recover orphaned jobs
python -m oact_utilities.workflows.dashboard workflow.db --recover-orphans --scheduler slurm

# Same for PBS Pro / OpenPBS
python -m oact_utilities.workflows.dashboard workflow.db --recover-orphans --scheduler pbspro

# Same for Flux
python -m oact_utilities.workflows.dashboard workflow.db --recover-orphans --scheduler flux
```

The recovery process:
1. Finds all RUNNING jobs with a `worker_id` (scheduler job ID set at submission time)
2. Queries the scheduler (`squeue`, `qstat`, or `flux jobs`) for active jobs -- single call for all
3. Jobs whose `worker_id` is no longer active are orphans
4. Each orphan is checked on disk: completed -> COMPLETED, failed -> FAILED, inconclusive -> TO_RUN
5. Content-based checks always take priority (a completed job is never incorrectly reset)

If the scheduler is unreachable, no jobs are modified (conservative default).

## Job Directory Cleanup

After large campaigns, job directories accumulate scratch files that waste disk quota and inodes. The cleanup utility removes these files safely.

### Basic Usage

```bash
# Preview what would be deleted (dry-run, default)
python -m oact_utilities.workflows.clean workflow.db jobs/ --clean-tmp

# Actually delete scratch files from completed jobs
python -m oact_utilities.workflows.clean workflow.db jobs/ --clean-tmp --execute

# Clean basis set files
python -m oact_utilities.workflows.clean workflow.db jobs/ --clean-bas --execute

# Clean everything (scratch + basis)
python -m oact_utilities.workflows.clean workflow.db jobs/ --clean-all --execute

# Purge failed job directories (extracts metadata, writes marker, deletes contents)
python -m oact_utilities.workflows.clean workflow.db jobs/ --purge-failed --execute

# Combine: clean completed jobs AND purge failed ones
python -m oact_utilities.workflows.clean workflow.db jobs/ --clean-all --purge-failed --execute
```

### What Gets Cleaned

**`--clean-tmp` (Scratch/Temp):**
- `*.tmp`, `*.tmp.[N]` -- ORCA intermediate scratch files
- `orca_tmp_*/` directories -- Parsl temp directories
- `core`, `core.[N]`, `*.core` -- crash dump files

**`--clean-bas` (Basis Set):**
- `*.bas`, `*.bas[N]` -- basis set scratch files

**`--clean-all`:** Fixed alias for `--clean-tmp --clean-bas`.

### Purging Failed Jobs (`--purge-failed`)

Removes all contents from failed job directories except a `.do_not_rerun.json` marker file containing job metadata (SCF steps, failure reason, charge, spin, etc.). Failure reasons are extracted using `parse_failure_reason()` from `oact_utilities/utils/status.py`, which reads the last lines of the ORCA output file. This marker file:
- Prevents the job from being resubmitted (submit guard in `submit_jobs.py`)
- Preserves diagnostic information for post-hoc analysis
- Reclaims disk space from jobs that will not be retried

### Purging Incomplete Jobs (`--purge-incomplete`)

> **FINAL cleanup only -- not for an ongoing campaign.** `--purge-incomplete`
> deletes directories for jobs the DB still calls `running` / `to_run` /
> `timeout`. During an active campaign those statuses mean "in flight", so
> running this would destroy live or pending work. Only use it once the campaign
> is finished and nothing is executing (e.g. after transferring the dataset and
> DB to their final home). For trimming scratch during a live run, use
> `--clean-all` (completed jobs) or the Parsl inline `--clean-on-complete` hook.

For reclaiming space on a "final home" (e.g. ALCF) after a dataset and its DB are
transferred, `--purge-incomplete` full-purges the leftover non-corpus jobs whose
DB status is `running` / `to_run` / `timeout`. The DB status only selects
candidates; the **on-disk content check decides the action** (content beats DB
status):

- content says completed (`1`) -> **protected**, never deleted (it belongs in the
  corpus, even if the DB row is linked to the wrong directory)
- content says failed (`-1`) -> **skipped**; run `--purge-failed` to preserve the
  failure reason in the marker
- content confirms incomplete (`0` / `-2`) -> **purged** (full delete +
  `.do_not_rerun.json` marker carrying `purge_type: "incomplete_archive"`,
  `db_status_at_purge`, and `disk_status_code`)

The content check runs *before* the marker write, so a job that finishes between
the candidate query and the worker is protected rather than marked.

`--purge-incomplete` runs `--validate-db` first (hard gate). Recommended sequence:

```bash
# 0. Reconcile completed status from content first (so completed jobs are excluded)
python -m oact_utilities.workflows.dashboard workflow.db --update jobs/ --recheck-completed --unzip
# 1. Sanity-check that the DB maps to these folders (read-only)
python -m oact_utilities.workflows.clean workflow.db jobs/ --validate-db
# 2. Dry-run, then execute
python -m oact_utilities.workflows.clean workflow.db jobs/ --clean-all --purge-failed --purge-incomplete
python -m oact_utilities.workflows.clean workflow.db jobs/ --clean-all --purge-failed --purge-incomplete --execute
```

### DB <-> Folder Validation (`--validate-db`)

A light sanity check that the DB handed to the tool maps to the directories on
disk, catching gross operator error (wrong `--root`, wrong `.db`) before any
destructive action. It samples up to 100 rows (stratified to include the
running/to_run/timeout purge population), parses each `orca.inp` with
`read_geom_from_inp_file`, and compares the element multiset + atom count against
the DB. Outcomes bucket into MATCH / MISMATCH / UNVERIFIABLE. The gate **hard-aborts**
(exit non-zero) on any MISMATCH, or fails closed when fewer than half the sampled
rows could be verified. `--skip-validation` (alias `--force`) bypasses the gate
with a loud warning. It is not the primary safety mechanism -- the per-job content
check is -- but it is a cheap early abort.

### Safety Features

- **Dry-run by default**: No `--execute` flag means preview only
- **Revalidation**: Each completed job is re-checked on disk before cleanup
- **Content check is the purge safety net**: `--purge-incomplete` deletes only
  when on-disk content confirms no successful termination; completed jobs are
  protected regardless of DB status
- **Validation gate**: `--purge-incomplete` runs `--validate-db` first and refuses
  to delete on a DB<->folder mismatch (override with `--skip-validation`)
- **Exclusion list**: Critical files (orca.out, orca.inp, orca.engrad, orca.gbw, etc.) are never deleted
- **Path safety**: Job directories must resolve within root_dir (prevents traversal attacks)
- **TOCTOU protection**: `--purge-failed` and `--purge-incomplete` re-check DB status before deletion
- **Marker-first delete**: nothing is deleted unless the `.do_not_rerun.json` marker is written successfully
- **Symlink-safe**: directory sizing does not traverse symlinks into the corpus
- **Read-only DB**: The cleanup utility never modifies the workflow database

### CLI Reference

```
python -m oact_utilities.workflows.clean <db_path> <root_dir> [options]

Action flags (at least one required):
  --clean-tmp             Remove scratch/temp files from completed jobs
  --clean-bas             Remove basis set files from completed jobs
  --clean-all             Remove all scratch categories (--clean-tmp + --clean-bas)
  --purge-failed          Purge failed job directories
  --purge-incomplete      Full-purge running/to_run/timeout dirs confirmed
                          incomplete by content (runs --validate-db first)
  --validate-db           DB<->folder sanity check (elements + atom count);
                          exits non-zero on mismatch

Execution control:
  --execute               Actually delete files (default: dry-run)
  --skip-validation       Bypass the --purge-incomplete validation gate (alias --force)

Performance:
  --workers N             Parallel workers (default: 4)
  --debug N               Limit to first N jobs for testing

Output:
  --verbose / -v          Show per-file listings

Revalidation:
  --hours-cutoff H        Hours before timeout detection (default: 24)
```

### Submit Guard

When `--purge-failed` creates a `.do_not_rerun.json` marker, the submit utility (`submit_jobs.py`) automatically detects it and skips the job. If the DB was reset (`--reset-failed`) but the marker file remains, the submit guard prevents resubmission and updates the job status to FAILED in the database.

### Inline Cleanup During Submission (Parsl Mode)

For long-running Parsl campaigns, you can have `submit_jobs.py` clean each job's directory the moment its future completes, instead of running `clean.py` afterwards. This prevents scratch files from accumulating during the run and keeps the workflow tree small on Lustre/VAST.

Two opt-in flags, both Parsl-mode only:

```bash
python -m oact_utilities.workflows.submit_jobs workflow.db jobs/ \
    --use-parsl --batch-size 200 \
    --clean-on-complete \
    --purge-on-fail
```

- **`--clean-on-complete`** -- After each job completes successfully, removes scratch files (`*.tmp`, `*.core`, `orca_tmp_*/`) and basis-set files (`*.bas`, `*.basN`) from that job's directory. Equivalent to `clean.py --clean-all --execute` applied to a single job. Critical files (`orca.out`, `orca.inp`, `orca.engrad`, `orca.gbw`, `orca_metrics.json`, etc.) are never touched -- the same exclusion list as the standalone cleaner applies.

- **`--purge-on-fail`** -- After each job fails, writes a `.do_not_rerun.json` marker file containing the job's failure metadata (orig_index, charge, spin, fail_count, error_message, parsed failure reason, SCF steps), then deletes all other contents of the job directory. Equivalent to `clean.py --purge-failed --execute` applied to a single job. The marker prevents resubmission via the existing submit guard.

Both hooks run from the Parsl completion loop, after the per-job DB update commits. Failures inside the cleanup hooks are logged but never abort the submitter -- the campaign continues regardless. Traditional mode (`sbatch`/`flux batch`) does not support these flags because the submitter exits before any job finishes; continue to use `clean.py` for that path.

## Concurrency & Database Handling

The workflow system uses SQLite with **WAL (Write-Ahead Logging)** mode for better concurrent access:

- **WAL mode**: Multiple readers and one writer can access DB simultaneously
- **Retry logic**: Automatic retry with exponential backoff on database locks
- **Timeout**: Configurable timeout (default 30s) for lock acquisition
- **Indexes**: Optimized queries with status and orig_index indexes

This allows a monitoring script to update job statuses while jobs are being submitted.

## Integration with Existing Code

The workflow manager builds on existing utilities:

- **`oact_utilities.utils.architector.create_workflow_db()`**: Creates SQLite DB from CSV
- **`oact_utilities.utils.analysis.parse_job_metrics()`**: Extracts metrics from ORCA outputs
  - Supports gzipped files (quacc format)
  - Falls back to `.engrad` file for max forces
  - Handles missing files gracefully
- **`oact_utilities.utils.status.check_job_termination()`**: Checks ORCA termination status
- **`oact_utilities.utils.hpc.write_flux_no_template()`**: Generates job scripts (to be integrated)

## Supported Output Formats

The parsers handle multiple ORCA output formats:

1. **Direct ORCA runs**: Standard text output files
2. **Quacc runs**: Gzipped output files (`.out.gz`, `.engrad.gz`)
3. **ORCA `.engrad` files**: Binary format with energy and gradients
4. **Sella optimization logs**: Force and energy trajectories

All parsers include robust error handling for missing or corrupted files.

See `tests/test_workflow_parsers.py` for examples using real ORCA output data.
