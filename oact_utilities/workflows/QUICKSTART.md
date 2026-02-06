# Quick Start Guide: Architector Workflow Manager

## What is this?

A complete workflow management system for high-throughput architector calculations. It creates a SQLite database from your architector CSV to track job statuses, manage submissions, and monitor progress on HPC.

## 3-Minute Setup

### 1. Create workflow database from CSV

```python
from oact_utilities.utils.architector import create_workflow_db

# Creates a SQLite DB directly from your architector CSV
db_path = create_workflow_db(
    csv_path="architector_output.csv",
    db_path="workflow.db",
    geometry_column="aligned_csd_core",  # Column with XYZ coordinates
    charge_column="charge",              # Molecular charge
    spin_column="uhf",                   # Unpaired electrons (converted to 2S+1)
)
```

**Output:**

- `workflow.db` - SQLite database tracking ~31M structures with status, metrics columns

### 2. Submit jobs on HPC

```bash
# Basic submission (uses default ORCA settings)
python -m oact_utilities.workflows.submit_jobs \\
    workflow.db \\
    jobs/ \\
    --batch-size 100 \\
    --scheduler flux \\
    --n-cores 4 \\
    --n-hours 2

# With ORCA configuration options
python -m oact_utilities.workflows.submit_jobs \\
    workflow.db \\
    jobs/ \\
    --batch-size 100 \\
    --functional wB97M-V \\
    --simple-input omol \\
    --opt  # Enable geometry optimization
```

This creates `jobs/job_0/`, `job_1/`, etc. with:

- `orca.inp` - Complete ORCA input file (geometry + level of theory)
- `flux_job.flux` - Submission script (runs ORCA directly)
- Auto-submits and marks as "running"

**ORCA options:** `--functional`, `--simple-input {omol,x2c,dk3}`, `--actinide-basis`, `--nbo`, `--opt`, etc.

### 3. Monitor with dashboard

```bash
# Check status
python -m oact_utilities.workflows.dashboard workflow.db

# Update by scanning job directories
python -m oact_utilities.workflows.dashboard workflow.db --update jobs/

# Show metrics (forces, SCF steps)
python -m oact_utilities.workflows.dashboard workflow.db --show-metrics
```

**Example output:**

```
Workflow Status Summary
Status          Count    Percent
to_run           850      85.0%
running          120      12.0%
completed         25       2.5%
failed             3       0.3%
timeout            2       0.2%

Completion: [██░░░░░░...] 2.5% (25/1000)

Computational Metrics
Max Forces: mean=0.00123, median=0.00098
SCF Steps:  mean=12.3, median=11
```

## Database Columns

The SQLite DB automatically tracks:

- **Status tracking**: `status` (to_run/ready/running/completed/failed/timeout)
  - `to_run`: Job ready to be submitted (default for new jobs)
  - `ready`: Legacy status, use `to_run` instead
  - `running`: Job currently executing
  - `completed`: Job finished successfully
  - `failed`: Job failed (abort, error, etc.)
  - `timeout`: Job timed out (no file updates in 6+ hours)
- **Metrics**: `max_forces`, `scf_steps`, `final_energy` (auto-extracted from ORCA outputs)
- **Performance**: `wall_time` (seconds), `n_cores` (CPU cores used) - for tracking compute usage
- **Failure tracking**: `fail_count` (incremented each time a job is reset from failed/timeout to ready)
- **Metadata**: `job_dir`, `error_message`, `charge`, `spin`, `created_at`, `updated_at`
- **Structure info**: `elements`, `natoms`, `geometry` (XYZ string)
- **CSV reference**: `orig_index` (original CSV row number)

## Common Commands

```bash
# Submit 500 jobs
python -m oact_utilities.workflows.submit_jobs workflow.db jobs/ --batch-size 500

# Submit jobs, skipping those that have failed 3+ times
python -m oact_utilities.workflows.submit_jobs workflow.db jobs/ --batch-size 500 --max-fail-count 3

# Update statuses
python -m oact_utilities.workflows.dashboard workflow.db --update jobs/

# Show failed jobs (includes fail count)
python -m oact_utilities.workflows.dashboard workflow.db --show-failed

# Show timeout jobs
python -m oact_utilities.workflows.dashboard workflow.db --show-timeout

# Reset failed jobs to retry (increments fail_count)
python -m oact_utilities.workflows.dashboard workflow.db --reset-failed

# Reset timeout jobs to retry (increments fail_count)
python -m oact_utilities.workflows.dashboard workflow.db --reset-timeout

# Reset both failed and timeout jobs together
python -m oact_utilities.workflows.dashboard workflow.db --reset-failed --include-timeout-in-reset

# Reset only jobs that haven't failed 3+ times
python -m oact_utilities.workflows.dashboard workflow.db --reset-failed --max-retries 3

# Show jobs that have failed 3+ times (chronic failures)
python -m oact_utilities.workflows.dashboard workflow.db --show-chronic-failures 3

# Show computational metrics
python -m oact_utilities.workflows.dashboard workflow.db --show-metrics
```

## Python API

```python
from oact_utilities.workflows import ArchitectorWorkflow, JobStatus, update_job_status
from oact_utilities.workflows.submit_jobs import submit_batch, OrcaConfig

# Programmatic job submission with custom ORCA config
orca_config: OrcaConfig = {
    "functional": "wB97M-V",
    "simple_input": "x2c",  # X2C relativistic
    "opt": True,
}

with ArchitectorWorkflow("workflow.db") as wf:
    # Submit batch with ORCA config
    submitted = submit_batch(
        workflow=wf,
        root_dir="jobs/",
        batch_size=100,
        orca_config=orca_config,
        n_cores=8,
    )

    # Get jobs
    ready = wf.get_jobs_by_status(JobStatus.READY)

    # Update status manually
    wf.update_status(job_id=42, new_status=JobStatus.COMPLETED)

    # Update metrics manually
    wf.update_job_metrics(
        job_id=42,
        job_dir="/path/to/job_42",
        max_forces=0.00123,
        scf_steps=15,
        final_energy=-1234.56
    )

    # Or automatically extract metrics from ORCA output
    new_status = update_job_status(
        workflow=wf,
        job_dir="jobs/job_42/",
        job_id=42,
        extract_metrics=True,  # Auto-parse ORCA output
        unzip=False,  # Set True for quacc gzipped outputs
    )

    # Count jobs
    counts = wf.count_by_status()
    # {'to_run': 850, 'running': 120, 'completed': 25, 'failed': 3, 'timeout': 2}

    # Reset failed jobs (increments fail_count)
    wf.reset_failed_jobs()

    # Reset timeout jobs (increments fail_count)
    wf.reset_timeout_jobs()

    # Reset both failed and timeout jobs together
    wf.reset_failed_jobs(include_timeout=True)

    # Reset only jobs that haven't failed too many times
    wf.reset_failed_jobs(max_fail_count=3)

    # Find chronically failing jobs
    chronic = wf.get_jobs_by_fail_count(min_fail_count=3)
    for job in chronic:
        print(f"Job {job.id} has failed {job.fail_count} times: {job.error_message}")
```

## HPC Workflow Loop

```bash
# 1. Submit batch
python -m oact_utilities.workflows.submit_jobs workflow.db jobs/ --batch-size 500

# 2. Wait for jobs to run...

# 3. Update statuses
python -m oact_utilities.workflows.dashboard workflow.db --update jobs/

# 4. Repeat steps 1-3 until done
```

## Files Created

```
project/
├── workflow.db              # SQLite database (main tracking)
└── jobs/
    ├── job_0/              # Job directory (using orig_index by default)
    │   ├── orca.inp        # Complete ORCA input file
    │   ├── flux_job.flux   # Submission script (runs ORCA directly)
    │   ├── orca.out        # ORCA output (after running)
    │   ├── orca.engrad     # Energy and gradient
    │   └── ...
    ├── job_1/
    └── ...
```

**Note:** Job directories are named using `orig_index` (original CSV row) by default. Use `--job-dir-pattern "job_{id}"` to use the database ID instead.

## Full Documentation

- **Detailed guide**: `oact_utilities/workflows/README.md`
- **Usage examples**: `examples/architector_workflow_example.py`
- **Tests**: `tests/test_workflow.py`

## Key Features

✅ **Direct CSV to DB**: No chunking required, reads CSV in batches
✅ **Robust Parsing**: Handles regular and gzipped (quacc) ORCA outputs
✅ **Concurrent Access**: WAL mode + retry logic for database locks
✅ **Automatic Metrics**: Extracts max_forces, scf_steps, final_energy from outputs
✅ **Multiple Sources**: Tries `.engrad` file if text parsing fails
✅ **Error Handling**: Gracefully handles missing files and parse failures
✅ **Flux & SLURM**: Compatible with existing HPC job generation utilities
✅ **Real-time Dashboard**: Monitor progress with metrics display
✅ **Easy Retry**: Reset failed jobs with one command
✅ **Failure Tracking**: `fail_count` tracks retries; skip chronic failures automatically
✅ **Performance Tracking**: `wall_time` and `n_cores` for compute usage analysis (core-hours)
✅ **Timeout Detection**: Automatically detects jobs stuck for 6+ hours
✅ **Rich Status System**: Separate tracking for failed vs. timeout vs. running jobs

## Supported ORCA Output Formats

- **Direct ORCA**: Standard text output (`.out`, `logs`)
- **Quacc**: Gzipped files (`.out.gz`, `.engrad.gz`)
- **ORCA engrad**: Binary `.engrad` files for reliable force extraction
- **Sella logs**: Optimization trajectories

All formats tested with real examples in `tests/files/`.

That's it! You're ready to run high-throughput architector workflows.
