# Architector Workflow Manager

High-throughput workflow management for architector calculations on HPC systems.

## Features

- **Job Tracking**: SQLite database tracks job status (ready, running, completed, failed)
- **Metrics Collection**: Automatically extracts max forces, SCF steps, final energy from ORCA outputs
- **Concurrency Handling**: WAL mode + retry logic for concurrent database access
- **HPC Integration**: Works with existing Flux/SLURM job generation utilities
- **Dashboard**: Command-line monitoring with metrics display
- **Batch Submission**: Submit jobs in batches with automatic status updates
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

```bash
python -m oact_utilities.workflows.submit_jobs \\
    workflow.db \\
    jobs/ \\
    --batch-size 100 \\
    --scheduler flux \\
    --n-cores 4 \\
    --n-hours 2 \\
    --queue pbatch \\
    --allocation dnn-sim
```

Creates job directories `jobs/job_0/`, `jobs/job_1/`, etc. with:
- `input.xyz` - Structure geometry
- `flux_job.flux` or `slurm_job.sh` - Submission script
- Submits to scheduler and marks jobs as "running"

### 3. Monitor Progress

```bash
# Basic status summary
python -m oact_utilities.workflows.dashboard workflow.db

# Update statuses by scanning job directories
python -m oact_utilities.workflows.dashboard workflow.db --update jobs/

# Show computational metrics
python -m oact_utilities.workflows.dashboard workflow.db --show-metrics

# Show failed jobs
python -m oact_utilities.workflows.dashboard workflow.db --show-failed

# Reset failed jobs to retry
python -m oact_utilities.workflows.dashboard workflow.db --reset-failed
```

## Database Schema

The SQLite database tracks each structure with:

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key (auto-increment) |
| `orig_index` | INTEGER | Original row index from CSV |
| `elements` | TEXT | Semicolon-separated element symbols |
| `natoms` | INTEGER | Number of atoms |
| `status` | TEXT | Job status: ready, running, completed, failed |
| `charge` | INTEGER | Molecular charge (optional) |
| `spin` | INTEGER | Spin multiplicity (2S+1) |
| `geometry` | TEXT | XYZ geometry string |
| `job_dir` | TEXT | Path to job directory |
| `max_forces` | REAL | Maximum force from optimization (Eh/Bohr) |
| `scf_steps` | INTEGER | Total SCF iterations |
| `final_energy` | REAL | Final energy (Hartree) |
| `error_message` | TEXT | Error message if failed |
| `created_at` | TIMESTAMP | Creation time |
| `updated_at` | TIMESTAMP | Last update time |

**Indexes:** `idx_status`, `idx_orig_index` for fast queries.

## Job Statuses

- **`ready`**: Job is queued and ready to submit
- **`running`**: Job has been submitted to HPC scheduler
- **`completed`**: Job finished successfully
- **`failed`**: Job crashed or failed convergence

## Programmatic API

```python
from oact_utilities.workflows import ArchitectorWorkflow, JobStatus

with ArchitectorWorkflow("workflow.db") as workflow:

    # Get jobs by status
    ready = workflow.get_jobs_by_status(JobStatus.READY)
    completed = workflow.get_jobs_by_status(JobStatus.COMPLETED)

    # Count jobs
    counts = workflow.count_by_status()
    # {'ready': 1000, 'running': 200, 'completed': 50, 'failed': 10}

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

    # Reset failed jobs
    workflow.reset_failed_jobs()
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

## Custom Job Setup

You can provide a custom setup function to `submit_batch()`:

```python
from oact_utilities.workflows.submit_jobs import submit_batch

def setup_orca_input(job_dir, job_record):
    """Write ORCA input file for each job."""
    inp_file = job_dir / "calc.inp"
    with open(inp_file, "w") as f:
        f.write("! B3LYP def2-SVP OPT\\n")
        f.write(f"* xyzfile 0 1 input.xyz\\n")

with ArchitectorWorkflow("workflow.db") as workflow:
    submit_batch(
        workflow=workflow,
        root_dir="jobs/",
        batch_size=100,
        scheduler="flux",
        setup_func=setup_orca_input,
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
ready            8500      85.0%
running          1200      12.0%
completed         250       2.5%
failed             50       0.5%
----------------------------------------
TOTAL           10000     100.0%

Completion: [████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 2.5% (250/10000)

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

Total jobs with metrics: 250
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

5. **Handle failures**:
   ```bash
   python -m oact_utilities.workflows.dashboard workflow.db --show-failed
   python -m oact_utilities.workflows.dashboard workflow.db --reset-failed
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
