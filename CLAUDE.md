# CLAUDE.md

## Project Overview

**oact_utilities** is a Python package for running ORCA quantum chemistry calculations with a focus on actinide chemistry workflows. It provides utilities for:

- Setting up and running ORCA calculations via ASE and quacc
- Managing HPC job submission (Flux and SLURM schedulers)
- Geometry optimizations using Sella
- Job status monitoring and analysis
- Multi-spin state calculations

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
Run tests before committing: `pytest tests/` — ensure all tests pass locally before pushing or opening a PR. Consider adding a local hook or CI step that runs `pytest` as part of pre-commit.
```

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
├── core/orca/          # ORCA calculator setup and recipes
│   ├── _base.py        # Base job functions (run_and_summarize, prep_calculator)
│   ├── calc.py         # Calculator utilities, block generation
│   └── recipes.py      # High-level recipes (single_point_calculation, ase_relaxation)
├── utils/
│   ├── hpc.py          # HPC job file writers (Flux, SLURM)
│   ├── jobs.py         # Job launching utilities
│   ├── status.py       # Job status checking
│   ├── analysis.py     # Results analysis and ORCA output parsers
│   ├── architector.py  # Architector CSV processing and workflow DB creation
│   └── create.py       # Input file creation
├── workflows/          # High-throughput workflow management
│   ├── architector_workflow.py  # Workflow manager with SQLite tracking
│   ├── dashboard.py             # CLI dashboard for monitoring
│   ├── submit_jobs.py           # Batch job submission
│   ├── README.md                # Detailed documentation
│   └── QUICKSTART.md            # Quick start guide
└── scripts/            # Workflow scripts for various calculation campaigns
    ├── wave_one/       # First wave calculations
    ├── wave_two/       # Second wave calculations
    └── multi_spin/     # Multi-spin state workflows
```

## Key Dependencies

- **ORCA 6.0+**: Quantum chemistry engine (rigid requirement)
- **quacc**: Workflow management and calculator integration
- **ASE**: Atomic Simulation Environment for structure handling
- **Sella**: Geometry optimization
- **pandas**: Data analysis

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

## Common Tasks for Claude

### 1. High-Throughput Workflows

Managing large-scale architector calculation campaigns. Key files:

- `oact_utilities/workflows/architector_workflow.py` - Workflow manager with SQLite tracking
- `oact_utilities/workflows/dashboard.py` - CLI dashboard for monitoring jobs
- `oact_utilities/workflows/submit_jobs.py` - Batch job submission to Flux/SLURM
- `oact_utilities/utils/architector.py` - Database creation from CSV files

**Quick example:**
```python
from oact_utilities.utils.architector import create_workflow_db
from oact_utilities.workflows import ArchitectorWorkflow

# Create workflow from architector CSV
db_path = create_workflow_db(
    csv_path="architector_output.csv",
    db_path="workflow.db",
    geometry_column="aligned_csd_core",
)

# Manage workflow
with ArchitectorWorkflow(db_path) as wf:
    ready = wf.get_jobs_by_status(JobStatus.READY)
    print(f"{len(ready)} jobs ready")
```

See `oact_utilities/workflows/QUICKSTART.md` for details.

### 2. HPC Job Submission

Creating job submission scripts from custom datasets. Key files:

- `oact_utilities/utils/hpc.py` - Job file writers (Flux, SLURM)
- `oact_utilities/utils/jobs.py` - Job launchers
- `oact_utilities/workflows/submit_jobs.py` - Batch submission for workflows

### 3. Job Monitoring & Dashboards

Status checking and visualization of running jobs:

- `oact_utilities/workflows/dashboard.py` - Workflow dashboard with metrics
- `oact_utilities/utils/status.py` - Job termination/completion checks
- `oact_utilities/utils/analysis.py` - Results parsing (max forces, SCF steps, energies)

**Dashboard example:**
```bash
# Monitor workflow and update statuses
python -m oact_utilities.workflows.dashboard workflow.db --update jobs/ --show-metrics
```

### 4. Analysis & Parsing

Parsing ORCA outputs, extracting energies, gradients, timings:

- `oact_utilities/utils/analysis.py` - Comprehensive ORCA output parsers
  - `parse_max_forces()` - Extract max gradient from output
  - `parse_scf_steps()` - Count SCF iterations
  - `parse_final_energy()` - Extract final energy
  - `get_engrad()` - Parse `.engrad` file (includes max force computation)
  - `parse_job_metrics()` - All-in-one with gzip support for quacc
- `oact_utilities/scripts/wave_two/analysis_*.py` - Campaign-specific analysis

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

**Workflow debugging:**
```bash
# Show failed jobs with error messages
python -m oact_utilities.workflows.dashboard workflow.db --show-failed

# Reset failed jobs to retry
python -m oact_utilities.workflows.dashboard workflow.db --reset-failed
```

### 6. New ORCA Recipes

Adding calculation types in `oact_utilities/core/orca/recipes.py`. Follow existing patterns like `single_point_calculation` and `ase_relaxation`.

## Testing Patterns

Tests live in `tests/` with test data in `tests/files/`. When adding new functionality:

1. Add corresponding test in `tests/test_<module>.py`
2. Use fixtures for common test data
3. Mock ORCA calls for unit tests (actual ORCA not available in CI)

**Test data:**
- `tests/files/orca_direct_example/` - Direct ORCA run outputs
- `tests/files/quacc_example/` - Quacc gzipped outputs
- Both used to test parsers against real ORCA output formats

## Important Notes

- The `data/` directory contains benchmark datasets and is excluded from the package
- Scripts in `oact_utilities/scripts/` often have hardcoded paths - these are working examples, not library code
- When modifying HPC utilities, test on both Flux (Tuolumne) and SLURM systems
