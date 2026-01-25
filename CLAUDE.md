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

### Running Tests
```bash
pytest tests/
```

### Linting/Formatting
```bash
black .
```

### Installation
```bash
pip install -e .
```

## Code Style Requirements

- **Type hints**: Always use type hints for function parameters and return values
- **Docstrings**: All public functions must have docstrings (Google or NumPy style)
- **No hardcoding**: Avoid hardcoded paths, HPC configs, or system-specific values. Use parameters with sensible defaults instead
- **Black formatting**: Code must pass `black` formatting

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
│   ├── analysis.py     # Results analysis
│   └── create.py       # Input file creation
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

### 1. HPC Template Generation
Creating job submission scripts from custom datasets. Key files:
- `oact_utilities/utils/hpc.py` - Job file writers
- `oact_utilities/utils/jobs.py` - Job launchers

### 2. Job Monitoring & Dashboards
Status checking and visualization of running jobs:
- `oact_utilities/utils/status.py` - Job termination/completion checks
- `oact_utilities/utils/analysis.py` - Results parsing

### 3. Analysis Utilities
Parsing ORCA outputs, extracting energies, gradients, timings:
- `oact_utilities/utils/analysis.py`
- `oact_utilities/scripts/wave_two/analysis_*.py`

### 4. Debugging Job Failures
Check these locations for common issues:
- ORCA `.out` files for calculation errors
- `sella.log` for optimization failures
- HPC scheduler output for resource issues

### 5. New ORCA Recipes
Adding calculation types in `oact_utilities/core/orca/recipes.py`. Follow existing patterns like `single_point_calculation` and `ase_relaxation`.

## Testing Patterns

Tests live in `tests/` with test data in `tests/files/`. When adding new functionality:
1. Add corresponding test in `tests/test_<module>.py`
2. Use fixtures for common test data
3. Mock ORCA calls for unit tests (actual ORCA not available in CI)

## Important Notes

- The `data/` directory contains benchmark datasets and is excluded from the package
- Scripts in `oact_utilities/scripts/` often have hardcoded paths - these are working examples, not library code
- When modifying HPC utilities, test on both Flux (Tuolumne) and SLURM systems
