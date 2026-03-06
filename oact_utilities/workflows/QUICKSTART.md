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

### 2. Submit jobs on HPC

#### A. Traditional Mode (Individual Flux/SLURM Jobs)

```bash
# Basic submission (uses default ORCA settings)
python -m oact_utilities.workflows.submit_jobs \
    workflow.db \
    jobs/ \
    --batch-size 100 \
    --scheduler flux \
    --n-cores 4 \
    --n-hours 2

# With ORCA configuration options
python -m oact_utilities.workflows.submit_jobs \
    workflow.db \
    jobs/ \
    --batch-size 100 \
    --functional wB97M-V \
    --simple-input omol \
    --opt  # Enable geometry optimization
```

This creates `jobs/job_0/`, `job_1/`, etc. with:

- `orca.inp` -- Complete ORCA input file (geometry + level of theory)
- `flux_job.flux` -- Submission script (runs ORCA directly)
- Auto-submits and marks as "running"

**ORCA options:** `--functional`, `--simple-input {omol,omol_base,x2c,dk3}`, `--actinide-basis`, `--nbo`, `--mbis`, `--kdiis`, `--opt`, etc.

#### B. Parsl Mode (Concurrent Execution on Exclusive Node)

Use Parsl to run multiple ORCA jobs concurrently on a single allocated node:

```bash
# Request exclusive node via Flux, then run Parsl inside
flux alloc -N 1 -n 64 -q pbatch -t 8h

# Inside allocation: Run jobs concurrently with Parsl
python -m oact_utilities.workflows.submit_jobs \
    workflow.db \
    jobs/ \
    --use-parsl \
    --batch-size 100 \
    --max-workers 4 \
    --cores-per-worker 16 \
    --n-cores 16 \
    --job-timeout 72000
```

**When to use Parsl mode:**

- You have an exclusive node allocation
- Running many short-medium jobs (< 2 hours each)
- Want to maximize node utilization
- Need real-time progress monitoring

See README.md for full Parsl architecture details and SLURM multi-node options.

### 3. Monitor with dashboard

```bash
# Check status
python -m oact_utilities.workflows.dashboard workflow.db

# Update statuses by scanning job directories (does NOT extract metrics)
python -m oact_utilities.workflows.dashboard workflow.db --update jobs/

# Update statuses AND extract metrics in one pass
python -m oact_utilities.workflows.dashboard workflow.db --update jobs/ --extract-metrics

# Show metrics summary (forces, SCF steps, energy, timing)
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

Completion: [##..........] 2.5% (25/1000)

Computational Metrics
Max Forces: mean=0.00123, median=0.00098
SCF Steps:  mean=12.3, median=11
```

## Extracting and Managing Metrics

The dashboard can extract computational metrics from ORCA output files and store them in the database. This is a separate step from status updates because parsing output files is slower than checking job completion.

### What gets extracted

- **max_forces** -- Maximum gradient (Eh/Bohr), from `.engrad` or text output
- **scf_steps** -- Total SCF iterations (pattern: `SCF CONVERGED AFTER X CYCLES`)
- **final_energy** -- Final energy in Hartree
- **wall_time** -- Wall time in seconds
- **n_cores** -- Number of CPU cores used

### Extract metrics for newly completed jobs

```bash
# During a status update: extract metrics for jobs that just completed
python -m oact_utilities.workflows.dashboard workflow.db --update jobs/ --extract-metrics
```

This does two things:
1. Scans job directories to update statuses (running -> completed/failed/timeout)
2. For newly completed jobs AND any previously completed jobs missing metrics, extracts and stores metrics

### Backfill or recompute metrics

```bash
# Recompute metrics for ALL completed jobs (even those that already have them)
# Useful after parser improvements or if you suspect stale data
python -m oact_utilities.workflows.dashboard workflow.db --update jobs/ --recompute-metrics
```

### Gzipped outputs (quacc)

If your jobs produce gzipped output files (`.out.gz`, `.engrad.gz`), add `--unzip`:

```bash
python -m oact_utilities.workflows.dashboard workflow.db --update jobs/ --extract-metrics --unzip
```

### Performance tuning

```bash
# Use more workers for parallel metric extraction (default: 4)
python -m oact_utilities.workflows.dashboard workflow.db --update jobs/ --extract-metrics --workers 8

# Test on a small subset before running on everything
python -m oact_utilities.workflows.dashboard workflow.db --update jobs/ --extract-metrics --debug 50

# Profile to identify bottlenecks (I/O, parsing, DB writes)
python -m oact_utilities.workflows.dashboard workflow.db --update jobs/ --extract-metrics --profile
```

The `--profile` flag prints a breakdown showing parse time per job, slowest jobs, and throughput (jobs/sec).

### Re-verify completed jobs

```bash
# Re-check that completed jobs actually terminated normally
# Catches tampered outputs or status checker changes
python -m oact_utilities.workflows.dashboard workflow.db --update jobs/ --recheck-completed
```

## Common Commands

```bash
# Submit 500 jobs
python -m oact_utilities.workflows.submit_jobs workflow.db jobs/ --batch-size 500

# Submit jobs, skipping those that have failed 3+ times
python -m oact_utilities.workflows.submit_jobs workflow.db jobs/ --batch-size 500 --max-fail-count 3

# Update statuses
python -m oact_utilities.workflows.dashboard workflow.db --update jobs/

# Update statuses + extract metrics
python -m oact_utilities.workflows.dashboard workflow.db --update jobs/ --extract-metrics

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

## HPC Workflow Loop

```bash
# 1. Submit batch
python -m oact_utilities.workflows.submit_jobs workflow.db jobs/ --batch-size 500

# 2. Wait for jobs to run...

# 3. Update statuses and extract metrics
python -m oact_utilities.workflows.dashboard workflow.db --update jobs/ --extract-metrics

# 4. Handle failures
python -m oact_utilities.workflows.dashboard workflow.db --reset-failed --max-retries 3

# 5. Repeat steps 1-4 until done
```

## Full Documentation

- **Detailed guide with Python API**: `oact_utilities/workflows/README.md`
- **Parsl architecture**: `docs/parsl_integration.md`
- **Usage examples**: `examples/architector_workflow_example.py`
- **Tests**: `tests/test_workflow.py`
