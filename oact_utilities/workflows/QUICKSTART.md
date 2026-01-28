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
python -m oact_utilities.workflows.submit_jobs \\
    workflow.db \\
    jobs/ \\
    --batch-size 100 \\
    --scheduler flux \\
    --n-cores 4 \\
    --n-hours 2
```

This creates `jobs/job_0/`, `job_1/`, etc. with:
- `input.xyz` - Structure
- `flux_job.flux` - Submission script
- Auto-submits and marks as "running"

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
ready            850      85.0%
running          120      12.0%
completed         25       2.5%
failed             5       0.5%

Completion: [██░░░░░░...] 2.5% (25/1000)

Computational Metrics
Max Forces: mean=0.00123, median=0.00098
SCF Steps:  mean=12.3, median=11
```

## Database Columns

The SQLite DB automatically tracks:

- **Status tracking**: `status` (ready/running/completed/failed)
- **Metrics**: `max_forces`, `scf_steps`, `final_energy` (auto-extracted from ORCA outputs)
- **Metadata**: `job_dir`, `error_message`, `charge`, `spin`, `created_at`, `updated_at`
- **Structure info**: `elements`, `natoms`, `geometry` (XYZ string)
- **CSV reference**: `orig_index` (original CSV row number)

## Common Commands

```bash
# Submit 500 jobs
python -m oact_utilities.workflows.submit_jobs workflow.db jobs/ --batch-size 500

# Update statuses
python -m oact_utilities.workflows.dashboard workflow.db --update jobs/

# Show failed jobs
python -m oact_utilities.workflows.dashboard workflow.db --show-failed

# Reset failed jobs to retry
python -m oact_utilities.workflows.dashboard workflow.db --reset-failed

# Show computational metrics
python -m oact_utilities.workflows.dashboard workflow.db --show-metrics
```

## Python API

```python
from oact_utilities.workflows import ArchitectorWorkflow, JobStatus, update_job_status

with ArchitectorWorkflow("workflow.db") as wf:
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
    # {'ready': 850, 'running': 120, 'completed': 25, 'failed': 5}
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
    ├── job_1/              # Job directory (using DB id)
    │   ├── input.xyz       # Structure from DB geometry
    │   ├── flux_job.flux   # Submission script
    │   ├── orca.out        # ORCA output (after running)
    │   ├── orca.engrad     # Energy and gradient
    │   └── ...
    ├── job_2/
    └── ...
```

**Note:** Job directories are named using the database `id` (auto-increment), not the original CSV row index.

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

## Supported ORCA Output Formats

- **Direct ORCA**: Standard text output (`.out`, `logs`)
- **Quacc**: Gzipped files (`.out.gz`, `.engrad.gz`)
- **ORCA engrad**: Binary `.engrad` files for reliable force extraction
- **Sella logs**: Optimization trajectories

All formats tested with real examples in `tests/files/`.

That's it! You're ready to run high-throughput architector workflows.
