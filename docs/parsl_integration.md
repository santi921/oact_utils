# Parsl Integration for Architector Workflows

**Date**: February 6, 2026
**Commit**: `045922c` - "parsl w/ flux"
**Author**: Santiago Vargas

## Overview

Added Parsl-based concurrent execution mode to the architector workflow manager. This enables running multiple ORCA calculations simultaneously on a single allocated node, dramatically improving throughput for high-throughput campaigns.

## Motivation

**Problem**: Traditional mode submits each ORCA job as a separate Flux/SLURM job, leading to:
- Queue overhead for each job submission
- Sequential execution on allocated nodes (one job at a time)
- Poor resource utilization on exclusive nodes
- Delayed status updates (must scan directories periodically)

**Solution**: Parsl mode runs multiple ORCA jobs concurrently within a single node allocation:
- 4× throughput on a 64-core node (4 concurrent 16-core jobs)
- Real-time database updates as jobs complete
- Better resource utilization (immediate scheduling)
- Simplified workflow management (one Parsl process manages all jobs)

## What Changed

### 1. Core Implementation (`submit_jobs.py`)

#### New Functions

**`orca_job_wrapper()`** - Parsl `@python_app` decorated function
- Executes a single ORCA job within a Parsl worker
- Handles subprocess management, timeout, error capture
- Returns structured result dict with status, metrics, errors
```python
@python_app
def orca_job_wrapper(job_id, job_dir, orca_config, n_cores, timeout_seconds=7200):
    # Runs ORCA via subprocess.run() with timeout
    # Returns: {"job_id": int, "status": str, "wall_time": float, "error": str}
```

**`build_parsl_config_flux()`** - Parsl configuration builder
- Creates `Config` with `HighThroughputExecutor` + `LocalProvider`
- Configures worker initialization (conda, LD_LIBRARY_PATH)
- Single-node execution (Flux doesn't support Parsl scale-out)
```python
def build_parsl_config_flux(max_workers=4, cores_per_worker=16, conda_env="py10mpi", ...)
    # Returns Parsl Config for local execution
```

**`submit_batch_parsl()`** - Batch submission with Parsl
- Prepares job directories (same as traditional mode)
- Initializes Parsl executor
- Submits all jobs as futures
- Monitors with `concurrent.futures.as_completed()` (critical for concurrency!)
- Updates database in real-time as jobs finish
- Handles graceful shutdown (Ctrl+C)
```python
def submit_batch_parsl(workflow, root_dir, num_jobs, max_workers=4, cores_per_worker=16, ...)
    # Returns list of submitted job IDs
```

**`filter_jobs_for_submission()`** - Job filtering helper
- Extracts jobs ready for submission (TO_RUN status)
- Applies `max_fail_count` filter to skip chronic failures
- Used by both traditional and Parsl modes

#### CLI Changes

Added `--use-parsl` flag and Parsl-specific options:
```bash
python -m oact_utilities.workflows.submit_jobs workflow.db jobs/ \
    --use-parsl \                      # Enable Parsl mode
    --batch-size 100 \                 # Total jobs to run
    --max-workers 4 \                  # Concurrent workers
    --cores-per-worker 16 \            # Cores per worker
    --job-timeout 7200 \               # Per-job timeout (seconds)
    --conda-base /path/to/miniconda3   # Conda path for workers
```

### 2. Documentation Updates

**`QUICKSTART.md`**:
- Added Section 2B: "Parsl Mode (Concurrent Execution on Exclusive Node)"
- Documented CLI usage with `flux alloc` + Parsl submission
- Added benefits, options, and when to use Parsl mode
- Updated Python API examples to show `submit_batch_parsl()`

**`README.md`**:
- Added Section 2B: Traditional vs. Parsl mode comparison
- New section: "Parsl Mode Architecture" with technical details
  - How it works (5-step execution flow)
  - Configuration details (`HighThroughputExecutor`, `LocalProvider`)
  - Resource calculation examples
  - Error handling strategy
  - Performance comparison (3× speedup example)
- Updated "Features" list to include Parsl mode
- Added "Parsl Mode Options" table
- Programmatic API examples for `submit_batch_parsl()`

### 3. Dependencies

**Optional Parsl dependency**:
```python
try:
    from parsl import python_app
    PARSL_AVAILABLE = True
except ImportError:
    PARSL_AVAILABLE = False
    python_app = None
```

- Graceful fallback if Parsl not installed
- Traditional mode works without Parsl
- Installation: `pip install 'parsl>=2024.1'`

## Technical Details

### Architecture

```
User (flux alloc) → Parsl Config → HighThroughputExecutor
                                    ↓
                                LocalProvider (single node)
                                    ↓
                            Worker Processes (max_workers)
                                    ↓
                        ORCA subprocess (one per worker)
                                    ↓
                        Database updates (as_completed)
```

### Key Design Decisions

1. **LocalProvider instead of FluxProvider**: Flux doesn't integrate well with Parsl's scale-out model. Use local execution on the allocated node instead.

2. **`as_completed()` for concurrency**: Uses `concurrent.futures.as_completed()` to process results as they finish, not sequentially. Essential for true concurrent execution.

3. **Real-time DB updates**: Status updated immediately when each job completes, not in a batch post-processing step.

4. **Per-job timeout**: Each ORCA job has its own timeout (default 2 hours) to prevent hung jobs from blocking workers.

5. **Graceful shutdown**: Ctrl+C triggers Parsl cleanup and database consistency check.

6. **cores_per_worker = n_cores**: Must match to ensure proper CPU allocation. If mismatch, jobs may compete for cores.

### Resource Calculation

For a 64-core node:
```python
# Good: 4 workers × 16 cores = 64 cores (full utilization)
--max-workers 4 --cores-per-worker 16 --n-cores 16

# Good: 2 workers × 32 cores = 64 cores (larger jobs)
--max-workers 2 --cores-per-worker 32 --n-cores 32

# BAD: 8 workers × 16 cores = 128 cores (oversubscribed!)
--max-workers 8 --cores-per-worker 16 --n-cores 16
```

### Error Handling

- **Worker failures**: Individual job failures don't crash the executor
- **Database locking**: Retry logic with exponential backoff
- **Timeout**: Per-job timeout tracked, status set to TIMEOUT
- **Import check**: Parsl features disabled if package not installed
- **Cleanup**: Parsl `dfk.cleanup()` called in finally block

### Crash Recovery and Graceful Shutdown

The Parsl submission flow has three layers of crash recovery:

**Worker ID tracking**: When jobs are marked RUNNING, the scheduler job ID (`$SLURM_JOB_ID` or `$FLUX_JOB_ID`) is stored in the `worker_id` column in a single atomic UPDATE. This enables post-crash identification of which allocation owned each job.

**SIGTERM handler**: A flag-based signal handler is registered after `parsl.load()` (to avoid Parsl overwriting it). When SLURM sends SIGTERM (walltime limit, scancel, preemption), the handler sets `_shutdown_requested = True`. The `as_completed()` monitoring loop checks this flag after each job's DB writes complete and exits cleanly. The `finally` block then bulk-resets all unresolved jobs to TO_RUN with `worker_id=None`.

**Dashboard orphan recovery**: For SIGKILL/OOM/node crash scenarios where no Python cleanup runs, the dashboard's `--recover-orphans --scheduler {slurm,flux}` command queries the scheduler for active jobs, identifies dead allocations by comparing against `worker_id` values of RUNNING jobs, and recovers orphans based on their disk output (completed/failed/reset). See `oact_utilities/utils/scheduler.py`.

## Performance Benefits

### Speedup Example

**Traditional Mode (Sequential on 64-core node)**:
```
Job 1: ████████████████ (2h, uses 16 cores, 48 idle)
Job 2:                 ████████████████ (2h, uses 16 cores, 48 idle)
Job 3:                                 ████████████████ (2h)
Job 4:                                                 ████████████████ (2h)
Total: 8 hours for 4 jobs (poor utilization)
```

**Parsl Mode (4 concurrent workers on 64-core node)**:
```
Job 1: ████████████████ (2h, 16 cores)
Job 2: ████████████████ (2h, 16 cores)
Job 3: ████████████████ (2h, 16 cores)
Job 4: ████████████████ (2h, 16 cores)
Total: 2 hours for 4 jobs (full utilization)
```

**4× speedup** for jobs with similar runtimes.

### Core-Hours Efficiency

- Traditional: 4 jobs × 2h × 16 cores = 128 core-hours (but 384 wasted)
- Parsl: 4 jobs × 2h × 16 cores = 128 core-hours (0 wasted, full node usage)

## Usage Examples

### CLI Usage

```bash
# 1. Allocate exclusive node
flux alloc -N 1 -n 64 -q pbatch -t 8h -B dnn-sim

# 2. Run Parsl inside allocation
python -m oact_utilities.workflows.submit_jobs \
    workflow.db \
    jobs/ \
    --use-parsl \
    --batch-size 200 \
    --max-workers 4 \
    --cores-per-worker 16 \
    --n-cores 16 \
    --functional wB97M-V \
    --opt \
    --job-timeout 7200

# Jobs execute concurrently, status updated live:
# ✓ Job 42 completed (1/200 done)
# ✓ Job 43 completed (2/200 done)
# ✗ Job 44 failed: SCF convergence error
# ...

# 3. Check results
python -m oact_utilities.workflows.dashboard workflow.db --show-metrics
```

### Programmatic Usage

```python
from oact_utilities.workflows import ArchitectorWorkflow
from oact_utilities.workflows.submit_jobs import submit_batch_parsl, OrcaConfig

orca_config: OrcaConfig = {
    "functional": "wB97M-V",
    "opt": True,
    "actinide_basis": "ma-def-TZVP",
}

with ArchitectorWorkflow("workflow.db") as workflow:
    submitted_ids = submit_batch_parsl(
        workflow=workflow,
        root_dir="jobs/",
        num_jobs=100,
        max_workers=4,
        cores_per_worker=16,
        orca_config=orca_config,
        n_cores=16,
        timeout_seconds=7200,
        max_fail_count=3,  # Skip chronic failures
    )

    print(f"Submitted {len(submitted_ids)} jobs")
    # Blocking call - returns when all jobs complete

    # Check results
    completed = workflow.get_jobs_by_status(JobStatus.COMPLETED)
    print(f"Completed: {len(completed)}/{len(submitted_ids)}")
```

## When to Use Parsl Mode

### ✅ Use Parsl Mode When:

- You have an exclusive node allocation (`flux alloc`, not `flux batch`)
- Running many jobs (50+) with similar runtimes (< 4 hours each)
- Want to maximize node utilization and throughput
- Need real-time progress monitoring
- Jobs are independent (no inter-job dependencies)

### ❌ Use Traditional Mode When:

- Jobs have wildly different runtimes (load balancing issues)
- Running very long jobs (> 8 hours) - queue scheduler better for this
- Don't have exclusive node access (Parsl needs dedicated resources)
- Need scale-out to multiple nodes (Flux provider not well-supported)

## Testing

**Manual Testing**:
```bash
# Test on Tuolumne with small batch
flux alloc -N 1 -n 16 -q pdebug -t 30m

python -m oact_utilities.workflows.submit_jobs \
    test_workflow.db \
    test_jobs/ \
    --use-parsl \
    --batch-size 4 \
    --max-workers 2 \
    --cores-per-worker 8 \
    --n-cores 8 \
    --dry-run  # Test without running ORCA
```

**Expected Behavior**:
- Job directories created with `orca.inp`
- Parsl executor initializes successfully
- All 4 jobs marked as RUNNING in DB
- Jobs execute concurrently (2 at a time)
- Real-time status updates printed to console
- Database shows COMPLETED or FAILED after execution
- Graceful shutdown on Ctrl+C

## Future Improvements

1. **Multi-node support**: Investigate using `FluxExecutor` or manual Flux integration for scale-out
2. **Dynamic worker count**: Adjust `max_workers` based on node detection
3. **Load balancing**: Use Parsl's built-in scheduling for heterogeneous runtimes
4. **Checkpointing**: Save Parsl state for resumable workflows
5. **Monitoring UI**: Web dashboard for live Parsl execution monitoring
6. **Resource profiles**: Pre-configured resource settings for common node types

## Related Files

- `oact_utilities/workflows/submit_jobs.py` - Main implementation
- `oact_utilities/workflows/QUICKSTART.md` - User-facing quick start guide
- `oact_utilities/workflows/README.md` - Comprehensive documentation
- `oact_utilities/workflows/architector_workflow.py` - Workflow database manager
- `pyproject.toml` - Optional Parsl dependency (if added)

## References

- [Parsl Documentation](https://parsl.readthedocs.io/)
- [HighThroughputExecutor](https://parsl.readthedocs.io/en/stable/stubs/parsl.executors.HighThroughputExecutor.html)
- [Flux Documentation](https://flux-framework.readthedocs.io/)
