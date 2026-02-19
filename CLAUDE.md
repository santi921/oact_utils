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
│   ├── status.py               # Job termination/completion checking
│   ├── an66.py                 # Actinide-66 compound utilities
│   ├── baselines.py            # Baseline calculation helpers
│   └── table_summary.py        # Data table utilities
├── workflows/                  # High-throughput workflow management
│   ├── architector_workflow.py # Workflow manager with SQLite tracking (WAL mode)
│   ├── dashboard.py            # CLI dashboard for monitoring + parallel status updates
│   ├── submit_jobs.py          # Batch job submission (Traditional + Parsl modes)
│   ├── README.md               # Detailed workflow documentation
│   └── QUICKSTART.md           # Quick start guide
├── launch/                     # HPC launch scripts
│   ├── run_parsl_single_node.sh
│   ├── run_parsl_multi_node.sh
│   └── run_parsl_coordinator.sh
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

## Key Dependencies

- **ORCA 6.0+**: Quantum chemistry engine (rigid requirement)
- **quacc**: Workflow management and calculator integration
- **ASE**: Atomic Simulation Environment for structure handling
- **Sella**: Geometry optimization
- **pandas**: Data analysis
- **Parsl >= 2024.1**: Concurrent job execution on HPC nodes (optional, graceful fallback)
- **tqdm >= 4.0**: Progress bars for dashboard operations
- **periodictable**: Element data and actinide detection

## ORCA Templates

Four configurable templates defined in `core/orca/calc.py`:

| Template | Use Case | Key Settings |
|----------|----------|-------------|
| `omol` (default) | Standard DFT | RIJCOSX, def2/J, DIIS, NormalConv, DEFGRID3, ALLPOP |
| `omol_base` | Difficult SCF convergence | Simplified SCF (MaxIter=600), MediumConv, fewer convergence tweaks |
| `x2c` | Relativistic (actinides) | DLU-X2C, RIJCOSX, AutoAux |
| `dk3` | Heavy relativistic | DKH (Douglas-Kroll-Hess), SARC/J |

All templates now include `Print[ P_Hirshfeld ] 1` in the `%output` block for Hirshfeld population analysis by default.

**Additional ORCA config options:**
- `mbis=True` — append MBIS population analysis to simple input
- `nbo=True` — enable NBO analysis
- `diis_option="KDIIS"` — use KDIIS instead of DIIS for improved SCF convergence
- `opt=True` — enable geometry optimization

## HPC Systems

### Primary: Tuolumne (LLNL)

- Scheduler: **Flux**
- Job files: `flux_job.flux`
- Submit command: `flux batch <job_file>`

### Future: DoD Systems

- Scheduler: **SLURM**
- Job files: `slurm_job.sh`
- Submit command: `sbatch <job_file>`

When writing HPC utilities, always support both Flux and SLURM with configurable parameters.

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

See `docs/parsl_integration.md` for full Parsl architecture details.

## Workflow Database Schema

SQLite table `structures` with WAL mode for concurrent access:

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PK | Auto-increment |
| `orig_index` | INTEGER | Original CSV row (indexed) |
| `elements` | TEXT | Semicolon-separated element symbols |
| `natoms` | INTEGER | Atom count |
| `status` | TEXT | Job status (indexed): `to_run`, `running`, `completed`, `failed`, `timeout` |
| `charge` | INTEGER | Molecular charge |
| `spin` | INTEGER | Spin multiplicity (2S+1) |
| `geometry` | TEXT | XYZ string (**heavy** — exclude with `include_geometry=False`) |
| `job_dir` | TEXT | Path to job directory |
| `max_forces` | REAL | Max gradient (Eh/Bohr) |
| `scf_steps` | INTEGER | Total SCF iterations |
| `final_energy` | REAL | Final energy (Hartree) |
| `wall_time` | REAL | Wall time in seconds |
| `n_cores` | INTEGER | CPU cores used |
| `error_message` | TEXT | Error message if failed |
| `fail_count` | INTEGER | Retry counter (incremented on reset) |

**Performance notes:** Always use `include_geometry=False` or `_LIGHT_COLS` when you don't need XYZ coordinates. Push `LIMIT` into SQL, never slice in Python.

## Job Status Lifecycle

```
TO_RUN → RUNNING → COMPLETED
                  → FAILED → (reset) → TO_RUN
                  → TIMEOUT → (reset) → TO_RUN
```

- **TO_RUN**: Ready for submission (new standard status)
- **READY**: Legacy alias for TO_RUN (backwards compatibility)
- **RUNNING**: Submitted and executing on HPC
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
--scheduler {flux,slurm}     --batch-size N
--n-cores 4                  --n-hours 2
--queue pbatch               --allocation dnn-sim
--max-fail-count 3           # Skip jobs that failed N+ times

# Parsl mode
--use-parsl                  --max-workers 4
--cores-per-worker 16        --job-timeout 7200

# ORCA config
--functional wB97M-V         --simple-input {omol,x2c,dk3,omol_base}
--actinide-basis ma-def-TZVP --non-actinide-basis def2-TZVPD
--opt                        --nbo
--mbis                       --diis-option KDIIS
```

### 3. Job Monitoring & Dashboards

Status checking and visualization of running jobs:

- `oact_utilities/workflows/dashboard.py` - Workflow dashboard with parallel status updates
- `oact_utilities/utils/status.py` - Job termination/completion checks
- `oact_utilities/utils/analysis.py` - Results parsing (forces, SCF, energies, timings, populations)

**Dashboard CLI reference:**
```bash
python -m oact_utilities.workflows.dashboard <db> [options]

# Display options
--show-metrics               # Force, SCF, energy, timing statistics
--show-failed                # Failed jobs with error messages
--show-timeout               # Timeout jobs
--show-chronic-failures N    # Jobs failed N+ times

# Status updates
--update <job_dir>           # Scan directory for completions
--recompute-metrics          # Re-extract metrics from all outputs
--recheck-completed          # Re-verify completed jobs
--parse-charges              # Parse Mulliken/Loewdin populations

# Status management
--reset-failed               # Reset failed → TO_RUN (increments fail_count)
--reset-timeout              # Reset timeout → TO_RUN
--include-timeout-in-reset   # Reset both failed and timeout
--max-retries N              # Only reset jobs with fail_count < N

# Performance
--debug N                    # Limit to N jobs for testing
--workers N                  # Parallel workers for status updates
```

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

## Security Notes

- `_validate_file_path()` in `analysis.py` prevents directory traversal attacks
- `validate_extra_columns()` in `architector.py` validates column names (alphanumeric + underscore only) to prevent SQL injection
- `validate_charge_spin_conservation()` and `validate_spin_multiplicity()` validate physical properties

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
