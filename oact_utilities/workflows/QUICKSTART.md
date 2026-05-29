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

#### C. Site-Specific Launch Scripts (recommended over hand-rolled commands)

Pre-built launch scripts under `oact_utilities/launch/` wire up every flag a site needs. Copy one, edit the configuration block at the top, and submit.

| Site / topology | Script | Notes |
|-----------------|--------|-------|
| LLNL Tuolumne (Flux, single-node) | `run_parsl_single_node.sh` | LocalProvider inside `flux alloc` |
| LLNL multi-node (SLURM) | `run_parsl_multi_node.sh` | SlurmProvider multi-block |
| Generic PBS Pro multi-node | `run_parsl_multi_node_pbs.sh` | PBSProProvider; supports `--mpirun-path` |
| Sandia CTS1/TLCC2 single-node | `run_parsl_single_node_sandia.sh` | Run inside `salloc -p attaway`; uses `--hpc-site sandia --max-blocks 1` |
| Sandia CTS1/TLCC2 multi-block | `run_parsl_multi_node_sandia.sh` | Parsl auto-provisions worker blocks; coordinator only needs Python |

Sandia example:

```bash
# Single-node, inside an interactive allocation
salloc -N1 -p attaway -A fy250086 -t 8:00:00
conda activate oact
bash oact_utilities/launch/run_parsl_single_node_sandia.sh

# Multi-block (Parsl provisions its own SLURM allocations)
sbatch oact_utilities/launch/run_parsl_multi_node_sandia.sh
```

The `--hpc-site sandia` profile switches the job-script writer to use `module load`, sets `OMPI_MCA_pml/mtl/btl` to bypass PSM2/Omni-Path, and requests `--partition` instead of `--constraint`. See README.md "HPC Site Profiles" for adding a new site.

#### Memory-constrained nodes

If your node has limited RAM per core (Sandia CTS1: ~64 GB / 36 cores; TLCC2: ~32 GB / 16 cores), pass `--mem-per-job MB` to clamp total ORCA memory:

```bash
python -m oact_utilities.workflows.submit_jobs workflow.db jobs/ \
    --use-parsl --hpc-site sandia --scheduler slurm \
    --n-cores 12 --mem-per-job 60000   # CTS-1: ~60 GB total ORCA budget
```

`%maxcore` is sized per MPI rank so total memory stays under 85% of `mem_per_job`. Without this flag a 1500 MB per-rank floor is applied.

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

### Online monitoring with W&B (optional)

Stream campaign progress to [Weights & Biases](https://wandb.ai) so the team can watch live without SSH access:

```bash
# Install once
pip install wandb && wandb login

# Add --wandb-project to submit_jobs (Parsl mode only)
python -m oact_utilities.workflows.submit_jobs \
    workflow.db jobs/ \
    --use-parsl --max-workers 4 \
    --wandb-project actinide-campaign

# Add --wandb-project to dashboard scans (any mode)
python -m oact_utilities.workflows.dashboard \
    workflow.db --update jobs/ --extract-metrics \
    --wandb-project actinide-campaign \
    --wandb-run-id <run-id>  # reuse the same W&B run
```

W&B is optional -- if not installed or not configured, everything else works normally.

## Extracting and Managing Metrics

The dashboard can extract computational metrics from ORCA output files and store them in the database. This is a separate step from status updates because parsing output files is slower than checking job completion.

### What gets extracted?

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

## Clean Up Job Directories

After campaigns complete, reclaim disk space by removing scratch files:

```bash
# Preview what would be deleted (dry-run)
python -m oact_utilities.workflows.clean workflow.db jobs/ --clean-all

# Actually delete scratch and basis set files from completed jobs
python -m oact_utilities.workflows.clean workflow.db jobs/ --clean-all --execute

# Purge failed jobs (writes .do_not_rerun.json marker, deletes contents)
python -m oact_utilities.workflows.clean workflow.db jobs/ --purge-failed --execute
```

The utility defaults to dry-run mode -- add `--execute` to actually delete files. See README.md for full CLI reference.

### Inline Cleanup During Parsl Runs

For long-running Parsl campaigns on scratch-tight systems (Sandia, Nibi), clean each job's directory the moment its future completes instead of running `clean.py` afterwards:

```bash
python -m oact_utilities.workflows.submit_jobs workflow.db jobs/ \
    --use-parsl --batch-size 500 \
    --clean-on-complete \
    --purge-on-fail
```

- `--clean-on-complete` removes `.tmp`, `.core`, `orca_tmp_*/`, `.bas`, `.basN` from each successful job. Critical outputs (`orca.out`, `orca.inp`, `orca.engrad`, `orca.gbw`, `orca_metrics.json`) are preserved.
- `--purge-on-fail` writes `.do_not_rerun.json` (with failure metadata) and deletes everything else in the failed job's directory. The marker blocks resubmission.

Both flags are Parsl-mode only. Failures inside the cleanup hooks are logged but never abort the campaign.

## Crash Recovery

When a node dies or a SLURM/PBS job is killed mid-run, molecules with `worker_id` set to the dead allocation become orphans. Recover them from any login node:

```bash
# SLURM (Tuolumne SLURM partitions, Sandia, Alliance Canada)
python -m oact_utilities.workflows.dashboard workflow.db --recover-orphans --scheduler slurm

# PBS Pro / OpenPBS
python -m oact_utilities.workflows.dashboard workflow.db --recover-orphans --scheduler pbspro

# Flux (Tuolumne default)
python -m oact_utilities.workflows.dashboard workflow.db --recover-orphans --scheduler flux
```

Orphans whose ORCA output shows `ORCA TERMINATED NORMALLY` are marked COMPLETED; clear failures become FAILED; everything else is reset to TO_RUN.

## HPC Workflow Loop

```bash
# 1. Submit batch
python -m oact_utilities.workflows.submit_jobs workflow.db jobs/ --batch-size 500

# 2. Wait for jobs to run...

# 3. Update statuses and extract metrics
python -m oact_utilities.workflows.dashboard workflow.db --update jobs/ --extract-metrics

# 4. Handle failures
python -m oact_utilities.workflows.dashboard workflow.db --reset-failed --max-retries 3

# 5. Clean up scratch files from completed jobs
#    (skip this step if you used --clean-on-complete during submission)
python -m oact_utilities.workflows.clean workflow.db jobs/ --clean-all --execute

# 6. If a node died: recover orphans before resubmitting
python -m oact_utilities.workflows.dashboard workflow.db --recover-orphans --scheduler slurm

# 7. Repeat steps 1-6 until done
```

## Full Documentation

- **Detailed guide with Python API**: `oact_utilities/workflows/README.md`
- **Parsl architecture**: `docs/parsl_integration.md`
- **Usage examples**: `examples/architector_workflow_example.py`
- **Tests**: `tests/test_workflow.py`
