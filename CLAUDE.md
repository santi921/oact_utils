# CLAUDE.md

## Project Overview

**oact_utilities** is a Python package for running ORCA quantum chemistry calculations with a focus on actinide chemistry workflows. It provides utilities for:

- Setting up and running ORCA calculations via ASE and quacc
- Managing HPC job submission (Flux and SLURM schedulers)
- Concurrent job execution via Parsl on exclusive nodes
- Geometry optimizations using Sella
- Job status monitoring, dashboards, and analysis
- Multi-spin state calculations
- Population analysis parsing (Mulliken, Loewdin, Hirshfeld, MBIS)

**Target Audience**: Internal team use. Code should be well-documented and robust for use by multiple group members.

## Development Workflow

### Installation

```bash
pip install -e .          # Basic install
pip install -e ".[dev]"   # With dev dependencies (pytest, black, ruff, mypy, pre-commit)
```

### Pre-commit Hooks

```bash
pre-commit install        # Set up hooks (run once after cloning)
pre-commit run --all-files  # Run all hooks manually
```

Run tests before committing: `pytest tests/` — ensure all tests pass locally before pushing or opening a PR.

**Hooks that run on every commit:**

- `black` - Code formatting
- `ruff` - Linting (pycodestyle, pyflakes, isort, bugbear, pyupgrade)
- `mypy` - Type checking
- `trailing-whitespace`, `end-of-file-fixer`, `check-yaml` - File hygiene

**Manual hooks (don't block commits):**

```bash
pre-commit run --hook-stage manual pytest  # Run pytest manually
```

### Running Tests

```bash
pytest tests/
```

### Linting/Formatting

```bash
black .                   # Format code
ruff check . --fix        # Lint and auto-fix
mypy oact_utilities/      # Type check
```

## Code Style Requirements

- **Type hints**: Always use type hints for function parameters and return values
- **Docstrings**: All public functions must have docstrings (Google or NumPy style)
- **No hardcoding**: Avoid hardcoded paths, HPC configs, or system-specific values. Use parameters with sensible defaults instead
- **Formatting**: Code must pass `black` (formatting) and `ruff` (linting)
- **Imports**: Use `ruff` for import sorting (isort-compatible, configured in pyproject.toml)

## Architecture

```
oact_utilities/
├── core/orca/                  # ORCA calculator setup and recipes
│   ├── _base.py                # Base job functions (run_and_summarize, prep_calculator)
│   ├── calc.py                 # Calculator setup, ORCA block generation, templates
│   └── recipes.py              # High-level recipes (single_point, ase_relaxation, pure_ase_relaxation)
├── utils/
│   ├── analysis.py             # ORCA output parsers (forces, SCF, energy, timings, populations)
│   ├── architector.py          # Architector CSV → SQLite DB creation
│   ├── create.py               # ORCA input file generation (write_orca_inputs)
│   ├── hpc.py                  # HPC job file writers (Flux, SLURM)
│   ├── jobs.py                 # Job launching utilities
│   ├── status.py               # Job termination/completion checking, failure reason parsing
│   ├── scheduler.py            # Scheduler liveness checks for crash recovery (SLURM/Flux)
│   ├── an66.py                 # Actinide-66 compound utilities
│   ├── baselines.py            # Baseline calculation helpers
│   └── table_summary.py        # Data table utilities
├── workflows/                  # High-throughput workflow management
│   ├── architector_workflow.py # Workflow manager with SQLite tracking (WAL mode)
│   ├── clean.py                # Job directory cleanup utility (scratch, basis, purge failed)
│   ├── dashboard.py            # CLI dashboard for monitoring + parallel status updates
│   ├── submit_jobs.py          # Batch job submission (Traditional + Parsl modes)
│   ├── parsl_launchers.py      # Parsl Config builders (LocalProvider, SlurmProvider, PBSProProvider, Sandia variants)
│   ├── job_dir_patterns.py     # Job directory name templates (hostname/orig_index/id)
│   ├── wandb_logger.py         # Optional W&B online monitoring hooks
│   ├── README.md               # Detailed workflow documentation
│   └── QUICKSTART.md           # Quick start guide
├── launch/                     # HPC launch scripts (one per site/topology)
│   ├── run_parsl_single_node.sh           # LLNL Tuolumne, Flux LocalProvider
│   ├── run_parsl_multi_node.sh            # LLNL multi-node SLURM (SlurmProvider)
│   ├── run_parsl_multi_node_pbs.sh        # Generic PBS Pro multi-node (PBSProProvider)
│   ├── run_parsl_multi_node_sandia.sh     # Sandia CTS1/TLCC2 multi-block SlurmProvider
│   ├── run_parsl_single_node_sandia.sh    # Sandia CTS1/TLCC2 single-node LocalProvider
│   └── run_parsl_coordinator.sh           # Lightweight coordinator wrapper
├── scripts/                    # Campaign-specific workflow scripts
│   ├── wave_one/               # First wave calculations
│   ├── wave_two/               # Second wave calculations
│   ├── multi_spin/             # Multi-spin state workflows
│   └── sella_test/             # Sella optimization testing
├── examples/                   # Example scripts
│   ├── architector_workflow_example.py
│   ├── wave2/                  # Wave 2 workflow examples + migration guide
│   └── an66_distance_spin_sweeps/  # AN66 dissociation curve analysis
└── docs/
    ├── parsl_integration.md    # Parsl mode technical documentation
    ├── plans/                  # Architecture decision records
    └── solutions/              # Documented bug fixes and patterns
```

## Local Development Environment

- **ORCA 6.1.0 (macOS ARM64)**: `/Users/santiagovargas/Documents/orca_6_1_0_macosx_arm64_openmpi411/orca`

## Key Dependencies

- **ORCA 6.0+**: Quantum chemistry engine (rigid requirement)
- **quacc**: Workflow management and calculator integration
- **ASE**: Atomic Simulation Environment for structure handling
- **Sella**: Geometry optimization
- **pandas**: Data analysis
- **Parsl >= 2024.1**: Concurrent job execution on HPC nodes (optional, graceful fallback)
- **wandb**: Online campaign monitoring via Weights & Biases (optional, graceful fallback). Install with `pip install wandb`. Enabled via `--wandb-project` on `submit_jobs` (Parsl mode only) and `dashboard`.
- **tqdm >= 4.0**: Progress bars for dashboard operations
- **periodictable**: Element data and actinide detection

## ORCA Templates

Five configurable templates defined in `core/orca/calc.py`:

| Template         | Use Case                  | Key Settings                                                                          |
| ---------------- | ------------------------- | ------------------------------------------------------------------------------------- |
| `omol` (default) | Standard DFT              | RIJCOSX, def2/J, DIIS, NormalConv, DEFGRID3, ALLPOP                                   |
| `omol_base`      | Difficult SCF convergence | Simplified SCF (MaxIter=600), MediumConv, fewer convergence tweaks                    |
| `x2c`            | Relativistic (actinides)  | DLU-X2C, RIJCOSX, AutoAux                                                             |
| `dk3`            | Heavy relativistic        | DKH (Douglas-Kroll-Hess), SARC/J                                                      |
| `pm3`            | Debug / fast CI runs      | PM3 semiempirical, no Gaussian basis, under 1s on small organics; no actinide support |

All templates now include `Print[ P_Hirshfeld ] 1` in the `%output` block for Hirshfeld population analysis by default.

**Additional ORCA config options:**

- `mbis=True` — append MBIS population analysis to simple input
- `nbo=True` — enable NBO analysis
- `diis_option="KDIIS"` — use KDIIS instead of DIIS for improved SCF convergence
- `opt=True` — enable geometry optimization

## HPC Systems

Three schedulers are wired into `submit_jobs.py` (`--scheduler {flux,slurm,pbspro}`) and the dashboard's `--recover-orphans` path (`--scheduler {slurm,pbspro,flux}`). Site-specific behavior (modules, partitions, MPI env, ORCA path) is selected with `--hpc-site` and a small set of overrides.

### Primary: Tuolumne (LLNL)

- Scheduler: **Flux**
- Job files: `flux_job.flux`
- Submit command: `flux batch <job_file>`
- Default conda env: `py10mpi`; default ORCA at `/usr/workspace/vargas58/orca-6.1.0-f.0_linux_x86-64/bin/orca`

### Sandia CTS1 / TLCC2 (attaway, ecl)

- Scheduler: **SLURM**, selected via `--hpc-site sandia`
- Job script writer: `write_slurm_sandia_job_file()` in `workflows/submit_jobs.py` (uses `module load`, OMPI MCA env, `--partition`)
- Defaults: account `fy250086`, qos `normal`, partition `attaway`, OpenMPI module `aue/openmpi/4.1.6-gcc-12.3.0`, ntasks-per-node 36 (CTS1) / 16 (TLCC2)
- OMPI MCA settings disable PSM2/Omni-Path: `pml=ob1`, `mtl=^psm2`, `btl=tcp,self,vader`
- Default ORCA at `/home/${USER}/orca_6_1_0_linux_x86-64_shared_openmpi418/orca` (user-installed shared build)
- Launch scripts: `launch/run_parsl_single_node_sandia.sh` (LocalProvider inside `salloc`), `launch/run_parsl_multi_node_sandia.sh` (multi-block SlurmProvider)

### Planned: Digital Research Alliance of Canada (Fir, Trillium, Narval, Nibi, Rorqual)

- Scheduler: **SLURM**, CVMFS-shared StdEnv stack, ORCA module gated by license registration
- Per-cluster storage is independent (no central shared filesystem); 3 of 5 clusters have no internet on compute nodes
- ORCA must be invoked by absolute path, never via `srun`/`mpirun` (ORCA spawns its own MPI from `%pal nprocs`)
- Trillium scheduling is whole-node only (192 cores/node) and home/project are read-only inside jobs

When writing HPC utilities, support Flux, SLURM, and PBS Pro with configurable parameters and avoid hardcoding paths/accounts/modules at the call site.

**Site-specific defaults to override on first use:** Several CLI defaults are seeded with the original developer's paths/accounts and will fail elsewhere. New users must override one or more of:

- `--orca-path` -- `DEFAULT_ORCA_PATHS` in `submit_jobs.py` has user-specific paths (Flux: `/usr/workspace/vargas58/...`, Sandia: `/home/svargas/...`). Always pass `--orca-path` until the binary is in `PATH`.
- `--conda-base` -- Defaults to `/usr/WS1/vargas58/miniconda3` (LLNL Tuolumne). Override on every other site.
- `--conda-env` -- Defaults to `py10mpi`. Sandia launch scripts use `oact`. Pick whatever matches `pip install -e .` on the target host.
- `--qos` / `--account` -- Scale-out defaults `frontier` / `ODEFN5169CYFZ` are LLNL-only. Pass site-correct values or use the per-site launch scripts under `launch/` which already encode them.

### Job Submission Modes

**Traditional Mode** — submits each job as a separate Flux/SLURM batch job:

```bash
python -m oact_utilities.workflows.submit_jobs workflow.db jobs/ \
    --scheduler flux --batch-size 50 --n-cores 4
```

**Parsl Mode** — concurrent execution on an exclusive allocated node (4x throughput):

```bash
flux alloc -N 1 -n 64 -q pbatch -t 8h -B dnn-sim
python -m oact_utilities.workflows.submit_jobs workflow.db jobs/ \
    --use-parsl --batch-size 200 --max-workers 4 --cores-per-worker 16
```

**Parsl Mode (Sandia CTS1/TLCC2, multi-block SLURM)** — Parsl auto-provisions worker blocks; the coordinator only needs Python:

```bash
sbatch oact_utilities/launch/run_parsl_multi_node_sandia.sh
# or, inside an interactive allocation:
salloc -N1 -p attaway -A fy250086 -t 8:00:00 && \
    bash oact_utilities/launch/run_parsl_single_node_sandia.sh
```

See `docs/parsl_integration.md` for full Parsl architecture details.

### ORCA Memory Sizing

ORCA interprets `%maxcore` as memory **per MPI rank**, not per job. `get_mem_estimate()` in `core/orca/calc.py` returns a per-process value with a 1500 MB floor. Pass `--mem-per-job MB` on `submit_jobs` to clamp total job memory to 85% of a budget; recommended on memory-constrained nodes:

- Sandia CTS1 (attaway, ~64 GB/node, 36 cores): `--mem-per-job 60000`
- Sandia TLCC2 (~32 GB/node, 16 cores): `--mem-per-job 30000`
- Tuolumne / large-memory nodes: leave unset

The PM3 debug path bypasses this and uses `%maxcore 512`.

## Workflow Database Schema

SQLite table `structures` with WAL mode for concurrent access:

| Column          | Type       | Notes                                                                                             |
| --------------- | ---------- | ------------------------------------------------------------------------------------------------- |
| `id`            | INTEGER PK | Auto-increment                                                                                    |
| `orig_index`    | INTEGER    | Original CSV row (indexed)                                                                        |
| `elements`      | TEXT       | Semicolon-separated element symbols                                                               |
| `natoms`        | INTEGER    | Atom count                                                                                        |
| `status`        | TEXT       | Job status (indexed):`to_run`, `running`, `completed`, `failed`, `timeout`                        |
| `charge`        | INTEGER    | Molecular charge                                                                                  |
| `spin`          | INTEGER    | Spin multiplicity (2S+1), read directly from CSV — no internal conversion from unpaired electrons |
| `geometry`      | TEXT       | XYZ string (**heavy** — exclude with `include_geometry=False`)                                    |
| `job_dir`       | TEXT       | Path to job directory                                                                             |
| `max_forces`    | REAL       | Max gradient (Eh/Bohr)                                                                            |
| `scf_steps`     | INTEGER    | Total SCF iterations                                                                              |
| `final_energy`  | REAL       | Final energy (Hartree)                                                                            |
| `wall_time`     | REAL       | Wall time in seconds                                                                              |
| `n_cores`       | INTEGER    | CPU cores used                                                                                    |
| `error_message` | TEXT       | Error message if failed                                                                           |
| `fail_count`    | INTEGER    | Retry counter (incremented on reset)                                                              |
| `worker_id`     | TEXT       | Scheduler job ID owning this molecule (SLURM/Flux ID), used for crash recovery                    |

**Performance notes:** Always use `include_geometry=False` or `_LIGHT_COLS` when you don't need XYZ coordinates. Push `LIMIT` into SQL, never slice in Python.

## Job Status Lifecycle

```
TO_RUN → RUNNING → COMPLETED
                  → FAILED → (reset) → TO_RUN
                  → TIMEOUT → (reset) → TO_RUN
```

- **TO_RUN**: Ready for submission
- **READY**: Legacy alias, auto-migrated to TO_RUN on database open
- **RUNNING**: Submitted and executing on HPC (has `worker_id` set to scheduler job ID)
- **COMPLETED**: Successfully finished (verified by content check)
- **FAILED**: Crashed or error detected
- **TIMEOUT**: No updates for 6+ hours (configurable via `hours_cutoff`)

Status detection in `check_file_termination()` uses **content-based checks first** (priority), then file-age heuristic:

1. Check last 10 lines for `ORCA TERMINATED NORMALLY` → completed
2. Check for `aborting the run` or `Error` → failed
3. If file older than `hours_cutoff` → timeout
4. Otherwise → still running

## Common Tasks for Claude

### 1. High-Throughput Workflows

Managing large-scale architector calculation campaigns. Key files:

- `oact_utilities/workflows/architector_workflow.py` - Workflow manager with SQLite tracking
- `oact_utilities/workflows/dashboard.py` - CLI dashboard for monitoring jobs
- `oact_utilities/workflows/submit_jobs.py` - Batch job submission (Traditional + Parsl)
- `oact_utilities/utils/architector.py` - Database creation from CSV files

**Quick example:**

```python
from oact_utilities.utils.architector import create_workflow_db
from oact_utilities.workflows import ArchitectorWorkflow, JobStatus

db_path = create_workflow_db(
    csv_path="architector_output.csv",
    db_path="workflow.db",
    geometry_column="aligned_csd_core",
)

with ArchitectorWorkflow(db_path) as wf:
    ready = wf.get_jobs_by_status(JobStatus.TO_RUN, include_geometry=False)
    print(f"{len(ready)} jobs ready")
```

See `oact_utilities/workflows/QUICKSTART.md` for details.

### 2. HPC Job Submission

Creating job submission scripts from custom datasets:

- `oact_utilities/utils/hpc.py` - Job file writers (Flux, SLURM)
- `oact_utilities/utils/jobs.py` - Job launchers
- `oact_utilities/workflows/submit_jobs.py` - Batch submission (Traditional + Parsl modes)
- `oact_utilities/launch/` - Shell scripts for launching Parsl on HPC

**Submit CLI reference:**

```bash
python -m oact_utilities.workflows.submit_jobs <db> <root_dir> [options]

# Scheduler options
--scheduler {flux,slurm,pbspro}   --batch-size N
--n-cores 4                       --n-hours 2
--queue pbatch                    --allocation dnn-sim
--max-fail-count 3                # Skip jobs that failed N+ times
--max-atoms N                     # Only submit molecules with natoms <= N (stratify by size)

# Common per-site overrides (often required on a new cluster)
--orca-path /path/to/orca         # Override scheduler-keyed default in DEFAULT_ORCA_PATHS
--conda-env py10mpi               # Conda env name in worker_init / job script
--conda-base /path/to/miniconda3  # Conda root (Parsl worker_init)
--ld-library-path PATH            # Override LD_LIBRARY_PATH in generated job scripts
--job-dir-pattern '{hostname}_job_{orig_index}'   # Template for job directory names
--job-prefix campaignA            # Stable prefix prepended to job_dir (survives requeues)
--reroot                          # Ignore stored job_dir, rebuild from <root_dir>
--dry-run                         # Prepare jobs but do not submit

# HPC site profile (SLURM only)
--hpc-site {default,sandia}       # Selects job-script writer + worker_init
--partition PART                  # Sandia: SLURM partition (default: attaway)
--openmpi-module MOD              # Sandia: OpenMPI module (default: aue/openmpi/4.1.6-gcc-12.3.0)

# Parsl mode (any scheduler)
--use-parsl                       --max-workers 4
--cores-per-worker 16             --job-timeout 7200
--clean-on-complete               # Inline clean.py --clean-all per completed job
--purge-on-fail                   # Inline clean.py --purge-failed per failed job
--no-parsl-monitoring             # Skip MonitoringHub / monitoring.db

# Parsl scale-out (--use-parsl --scheduler slurm|pbspro)
--nodes-per-block 1               --max-blocks 10
--init-blocks 2                   --min-blocks 1
--walltime-hours 2                --cpus-per-node N
--qos frontier                    --account ODEFN5169CYFZ
--mpirun-path /path/to/mpirun     # PBS Pro: override mpirun discovery

# W&B online monitoring (Parsl mode only, optional)
--wandb-project PROJECT           --wandb-run-name NAME
--wandb-run-id ID

# ORCA config
--functional wB97M-V              --simple-input {omol,omol_base,x2c,dk3,pm3}
--actinide-basis ma-def-TZVP      --non-actinide-basis def2-TZVPD
--actinide-ecp def-ECP            # Pass 'none' (case-insensitive) to disable the ECP
--scf-maxiter N                   --ks-method {rks,uks,roks}
--optimizer {orca,sella}          --opt-level {loose,normal,tight,verytight}
--fmax 0.05                       # Sella force convergence threshold (Eh/Bohr)
--max-opt-steps N                 # Sella max steps (default: 100)
--save-all-steps                  # Sella: keep per-step ORCA outputs for replay/debug
--nbo                             --mbis             --kdiis
--mem-per-job MB                  # Total-job memory clamp; sizes %maxcore per MPI rank
```

### 3. Job Monitoring & Dashboards

Status checking and visualization of running jobs:

- `oact_utilities/workflows/dashboard.py` - Workflow dashboard with parallel status updates
- `oact_utilities/utils/status.py` - Job termination/completion checks
  - `check_file_termination()` - Content-based status detection (1=completed, 0=running, -1=failed, -2=timeout)
  - `parse_failure_reason()` - Extract failure reason from last lines of ORCA output (shared by clean.py and dashboard)
  - `_read_last_lines()` - Efficient tail-read helper using `deque(f, maxlen=N)`
- `oact_utilities/utils/analysis.py` - Results parsing (forces, SCF, energies, timings, populations)

**Dashboard CLI reference:**

```bash
python -m oact_utilities.workflows.dashboard <db> [options]

# Display options
--show-metrics               # Force, SCF, energy, timing statistics
--show-failed                # Failed jobs with error messages
--show-timeout               # Timeout jobs
--show-ready                 # Jobs ready to run
--show-running               # Currently running jobs
--show-chronic-failures N    # Jobs failed N+ times

# Status updates
--update <job_dir>           # Scan directory for completions
--extract-metrics            # Extract metrics for completed jobs during --update
--recompute-metrics          # Re-extract metrics for ALL completed jobs
--recheck-completed          # Re-verify completed jobs
--unzip                      # Handle gzipped outputs (quacc)

# Status management
--reset-failed               # Reset failed -> TO_RUN (increments fail_count)
--reset-timeout              # Reset timeout -> TO_RUN (increments fail_count)
--reset-missing <job_dir>    # Reset jobs with missing directories -> TO_RUN
--fix-unlinked <job_dir>     # Repair NULL job_dir: auto-link directories or reset to TO_RUN
--include-timeout-in-reset   # Reset both failed and timeout
--max-retries N              # Only reset jobs with fail_count < N

# Crash recovery
--recover-orphans                 # Detect jobs orphaned by dead scheduler allocations
--scheduler {slurm,pbspro,flux}   # Scheduler type (required with --recover-orphans)

# Performance
--debug N                    # Limit to N jobs for testing
--workers N                  # Parallel workers for metrics extraction
--profile                    # Profile metrics extraction bottlenecks
--hours-cutoff H             # Hours before job is considered timed out (default: 24)

# W&B online monitoring (optional, requires pip install wandb)
--wandb-project PROJECT      # W&B project name (enables logging)
--wandb-run-name NAME        # W&B run display name (default: db filename stem)
--wandb-run-id ID            # Resume an existing W&B run (share run with submit_jobs)
```

### 3b. Job Directory Cleanup

Removing scratch files from completed job directories:

- `oact_utilities/workflows/clean.py` - Cleanup utility (standalone CLI)

**Cleanup CLI reference:**

```bash
python -m oact_utilities.workflows.clean <db> <root_dir> [options]

# Action flags (at least one required)
--clean-tmp             # Remove .tmp, .core, orca_tmp_*/ from completed jobs
--clean-bas             # Remove .bas, .bas[N] from completed jobs
--clean-all             # Both --clean-tmp and --clean-bas
--purge-failed          # Purge failed jobs (write .do_not_rerun.json marker, delete contents)
--purge-incomplete      # Full-purge running/to_run/timeout dirs confirmed incomplete by
                        # on-disk content (1=completed protected, -1=failed skipped,
                        # 0/-2 purged). Writes .do_not_rerun.json marker. Runs --validate-db first.
--validate-db           # DB<->folder sanity check (elements + atom count) on a stratified
                        # sample; hard-aborts on mismatch or too few verifiable rows. Exits
                        # non-zero on failure. Implied by --purge-incomplete.

# Execution
--execute               # Actually delete (default: dry-run preview)
--skip-validation       # (alias --force) Bypass the --purge-incomplete validation gate (loud warning)
--reroot                # Resolve each job to <root_dir>/<basename> instead of its stored
                        # job_dir. Use when the corpus was moved to a new root after
                        # submission (stored paths stale, leaf dir names preserved).
                        # Basename-based (pattern-independent); applies to all phases AND
                        # the --validate-db gate; writes nothing to the DB.

# Performance / output
--workers N             # Parallel workers (default: 4)
--debug N               # Limit to N jobs
--verbose / -v          # Per-file listings
--hours-cutoff H        # Timeout threshold for revalidation (default: 24)
```

Each phase prints a skip breakdown (`dir_missing` / `escapes_root` / `null_job_dir`)
when rows cannot be mapped to a directory, so a coverage gap is never silent. A large
`dir_missing`/`escapes_root` count means the stored `job_dir` paths no longer match
the directories on disk -- run `python -m oact_utilities.workflows.diagnose_coverage <db>
<root>` to confirm, then re-run clean with `--reroot`.

**Final-home reclamation (post-transfer to ALCF):** `--purge-incomplete` and
`--validate-db` are for FINAL cleanup of a completed/transferred corpus, **not
for an ongoing campaign**. They act on jobs the DB still calls
`running`/`to_run`/`timeout` -- during an active campaign those statuses mean "in
flight", so purging them would destroy live or pending work. Only run them when
the campaign is finished and nothing is executing. To reclaim space from leftover
non-corpus jobs after a dataset+DB are moved to their final home, first reconcile
completed status from content, then validate, dry-run, and execute:

```bash
python -m oact_utilities.workflows.dashboard final.db --update jobs/ --recheck-completed --unzip
python -m oact_utilities.workflows.clean final.db jobs/ --validate-db
python -m oact_utilities.workflows.clean final.db jobs/ --clean-all --purge-failed --purge-incomplete
python -m oact_utilities.workflows.clean final.db jobs/ --clean-all --purge-failed --purge-incomplete --execute
```

The per-job content check is the real safety net: a `running`/`timeout` row whose
output actually terminated normally is **protected** (kept in the corpus) even if
the DB<->folder mapping is imperfect. clean.py never writes the DB.

**Moved corpus (stored `job_dir` no longer matches disk):** if the corpus was
transferred to a new root after submission, the DB's stored `job_dir` paths are
stale and clean would skip those jobs (visible in the per-phase skip breakdown).
Add `--reroot` to every clean invocation above so each job resolves to
`<root>/<basename>` instead; it is basename-based (pattern-independent) and also
reroots the `--validate-db` gate. Writes nothing to the DB, so the stored column
stays as-is -- rerooting is per-run only.

### 4. Analysis & Parsing

Parsing ORCA outputs, extracting energies, gradients, timings, populations:

- `oact_utilities/utils/analysis.py` - Comprehensive ORCA output parsers
  - `parse_max_forces()` - Extract max gradient from output
  - `parse_scf_steps()` - Count SCF iterations (pattern: `SCF CONVERGED AFTER X CYCLES`)
  - `parse_final_energy()` - Extract final energy
  - `get_engrad()` - Parse `.engrad` file (includes max force computation)
  - `parse_job_metrics()` - All-in-one with gzip support for quacc
  - `find_timings_and_cores()` - Extract wall time and core count
  - `parse_mulliken_population()` - Mulliken and Loewdin charges/spin populations
  - `validate_charge_spin_conservation()` - Verify electronic property consistency
  - `_validate_file_path()` - Security: prevent path traversal attacks

**Supported formats:**

- Regular ORCA text output
- Gzipped quacc output (`.out.gz`, `.engrad.gz`)
- ORCA `.engrad` binary format
- Sella optimization logs

### 5. Debugging Job Failures

Check these locations for common issues:

- ORCA `.out` files for calculation errors
- `sella.log` for optimization failures
- HPC scheduler output for resource issues
- Workflow database `error_message` column for tracked failures
- `fail_count` column to identify chronic failures
- `parse_failure_reason()` in `status.py` extracts failure reasons from ORCA output last lines (used by `clean.py` for marker files)

**Workflow debugging:**

```bash
# Show failed jobs with error messages
python -m oact_utilities.workflows.dashboard workflow.db --show-failed

# Show jobs that keep failing
python -m oact_utilities.workflows.dashboard workflow.db --show-chronic-failures 3

# Reset failed jobs to retry (increments fail_count)
python -m oact_utilities.workflows.dashboard workflow.db --reset-failed

# Reset with retry limit (skip chronic failures)
python -m oact_utilities.workflows.dashboard workflow.db --reset-failed --max-retries 3
```

### 6. New ORCA Recipes

Adding calculation types in `oact_utilities/core/orca/recipes.py`. Follow existing patterns like `single_point_calculation`, `ase_relaxation`, and `pure_ase_relaxation`. All recipes support the `mbis`, `nbo`, and `diis_option` parameters.

## Testing Patterns

Tests live in `tests/` with test data in `tests/files/`. When adding new functionality:

1. Add corresponding test in `tests/test_<module>.py`
2. Use fixtures for common test data
3. Mock ORCA calls for unit tests (actual ORCA not available in CI)
4. Use `Path(__file__).parent / "files"` for test data paths (no hardcoded paths)

**Test data:**

- `tests/files/orca_direct_example/` - Direct ORCA run outputs (AmO molecule)
- `tests/files/quacc_example/` - Quacc gzipped outputs (NpF3 molecule)
- Both used to test parsers against real ORCA output formats

**Current test modules:**

- `test_analysis.py` - Parser functions (forces, SCF, energy, Mulliken, metrics)
- `test_calculator.py` - ORCA calculator setup
- `test_hpc.py` - Job file generation
- `test_io.py` - I/O utilities
- `test_status.py` - Status checking and timeout detection
- `test_clean.py` - Job directory cleanup (patterns, purge, submit guard)
- `test_fix_unlinked.py` - Fix unlinked jobs (auto-link, reset, status revalidation)
- `test_workflow.py` - Workflow DB operations
- `test_workflow_parsers.py` - Parsers with real ORCA data
- `test_submit_jobs.py` - Job submission
- `test_actinide_neighbors.py` - Actinide utilities
- `test_check_multi_spin.py` / `test_run_multi_spin.py` - Multi-spin workflows
- `test_chunk_architector.py` / `test_chunk_architector_lmdb.py` - CSV chunking
- `test_quacc.py` - Quacc integration

## Performance Patterns

When working on database-heavy code or dashboard operations:

- **Exclude geometry column**: Use `include_geometry=False` or `_LIGHT_COLS` — geometry strings can be 58KB+ per row
- **SQL-level LIMIT**: Push `LIMIT` into SQL queries, never `fetchall()` then slice in Python
- **Batch commits**: On Lustre/GPFS, each `conn.commit()` forces a filesystem sync (~100-500ms). Batch multiple updates before committing
- **Parallel I/O**: Use `ThreadPoolExecutor` for scanning directories and extracting metrics (see dashboard.py)
- **Streaming reads**: Use `deque(f, maxlen=10)` to read only the last N lines instead of loading entire files
- **WAL mode + retry**: Database uses WAL mode with exponential backoff + jitter for concurrent Parsl access

See `docs/solutions/performance-issues/` for detailed writeups.

## Documentation Index

- `oact_utilities/workflows/README.md` - Comprehensive workflow documentation
- `oact_utilities/workflows/QUICKSTART.md` - 3-minute quick start guide
- `docs/parsl_integration.md` - Parsl mode architecture and usage
- `docs/plans/` - Architecture decision records
- `docs/solutions/logic-errors/` - Documented bug fixes (SCF parsing, timeout priority)
- `docs/solutions/performance-issues/` - Performance optimization writeups
- `docs/solutions/architecture-patterns/` - Design patterns (omol_base, parallel metrics)
- `examples/wave2/MIGRATION_GUIDE.md` - Wave 2 workflow migration guide

## Important Notes

- The `data/` directory contains benchmark datasets and is excluded from the package
- Scripts in `oact_utilities/scripts/` often have hardcoded paths — these are working examples, not library code
- When modifying HPC utilities, test on both Flux (Tuolumne) and SLURM systems
- Parsl is an optional dependency — traditional mode works without it. Import is wrapped in try/except with `PARSL_AVAILABLE` flag
- SCF parsing: always use `SCF CONVERGED AFTER X CYCLES` pattern, NOT `SCF ITERATIONS` header
- Status checking: content-based checks MUST run before file-age timeout heuristic
- Spin format: CSV input already contains spin multiplicity (2S+1) — do NOT convert from unpaired electrons internally. `create_workflow_db` reads spin values as-is and validates via `validate_spin_multiplicity()`

## Agent Instructions

- Research the codebase before editing. Never change code you haven't read. Also don't make changes to code without asking first.
- No sycophantic openers or closing fluff.
- Do not re-read files already read unless file may have changed.
- Read the file before modifying it. Never edit blind.
- No em dashes, smart quotes, or decorative Unicode symbols.
- Plain hyphens and straight quotes only.
- User instructions always override this file.

###

    Coding

- Test your code before declaring done.
- Be concise in output but thorough in reasoning.
- No inline prose. Use comments sparingly - only where logic is unclear.
- No abstractions for single-use operations.
- Three similar lines is better than a premature abstraction.
- No error handling for scenarios that cannot happen.
- Code output must be copy-paste safe.
- No compliments on the code before or after the review.
- State the bug. Show the fix. Stop.
- Never speculate about a bug without reading the relevant code first.
- State what you found, where, and the fix. One pass.
- If bug cause is unclear: say so. Do not guess. We can iterate on finding the right cause.

###

    Analysis

- Reporting: Lead with the finding. Context and methodology after.
- Reporting: Summary first (3 bullets max).
- Reporting: Supporting data second.
- Reporting: Caveats and limitations last.
- Formatting: Safe for copy-paste into spreadsheets and documents.
- Formatting: Tables use plain pipe characters.
- Formatting: Numbers must include units. Never ambiguous values.
- Distinguish clearly between what the data shows and what is inferred.
- Label inferences explicitly: "Based on the trend..." not stated as fact.
- Never fabricate data points, statistics, or citations.
- If confidence is low: state it explicitly with a reason.
