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
| `--simple-input` | omol | Input template: `omol`, `x2c`, or `dk3` |
| `--actinide-basis` | ma-def-TZVP | Basis set for actinides |
| `--actinide-ecp` | None | ECP for actinides |
| `--non-actinide-basis` | def2-TZVPD | Basis set for other elements |
| `--scf-maxiter` | ORCA default | Maximum SCF iterations |
| `--nbo` | False | Enable NBO analysis |
| `--opt` | False | Enable geometry optimization |
| `--orca-path` | scheduler-specific | Path to ORCA executable |
| `--conda-env` | py10mpi | Conda environment to activate |

**Parsl Mode Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--use-parsl` | False | Enable Parsl concurrent execution mode |
| `--max-workers` | 4 | Maximum number of concurrent Parsl workers |
| `--cores-per-worker` | 16 | CPU cores allocated per worker |
| `--n-cores` | auto (matches `cores_per_worker`) | Cores per ORCA job (auto-set to `cores_per_worker` if mismatch) |
| `--job-timeout` | 7200 | Per-job timeout in seconds (2 hours) |
| `--conda-base` | /usr/WS1/vargas58/miniconda3 | Conda base path for worker init |

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

## Database Schema

The SQLite database tracks each structure with:

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key (auto-increment) |
| `orig_index` | INTEGER | Original row index from CSV |
| `elements` | TEXT | Semicolon-separated element symbols |
| `natoms` | INTEGER | Number of atoms |
| `status` | TEXT | Job status: to_run, ready, running, completed, failed, timeout |
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
| `created_at` | TIMESTAMP | Creation time |
| `updated_at` | TIMESTAMP | Last update time |

**Indexes:** `idx_status`, `idx_orig_index` for fast queries.

## Job Statuses

- **`to_run`**: Job is ready to be submitted (default for new jobs)
- **`ready`**: Legacy status, use `to_run` instead (maintained for backward compatibility)
- **`running`**: Job has been submitted to HPC scheduler and is executing
- **`completed`**: Job finished successfully
- **`failed`**: Job crashed or failed convergence (explicit error or abort)
- **`timeout`**: Job timed out (file not modified in 6+ hours, likely stuck or walltime exceeded)

## Parsl Mode Architecture

The Parsl integration provides a Python-native workflow executor for high-throughput ORCA calculations:

### How It Works

1. **Job Preparation**: Creates job directories with ORCA input files (same as traditional mode)
2. **Parsl Initialization**: Configures `HighThroughputExecutor` with `LocalProvider`
   - Uses local node resources (no scale-out to other nodes)
   - Spawns worker processes on the allocated node
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
- **LocalProvider**: Uses the current node only (Flux doesn't support Parsl scale-out)
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
    ready = workflow.get_jobs_by_status(JobStatus.READY)
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
    "simple_input": "x2c",         # Template: "omol", "x2c", or "dk3"
    "actinide_basis": "ma-def-TZVP",
    "actinide_ecp": None,          # Or "def-ECP" for ECP calculations
    "non_actinide_basis": "def2-TZVPD",
    "scf_MaxIter": None,           # Use ORCA default
    "nbo": False,                  # NBO analysis
    "opt": True,                   # Geometry optimization
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
