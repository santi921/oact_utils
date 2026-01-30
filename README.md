# oact_utilities

Python utilities for running ORCA quantum chemistry calculations with a focus on actinide chemistry workflows.

## Features

- **ORCA Integration**: ASE and quacc-based calculator setup and recipes
- **HPC Support**: Job file generation for Flux and SLURM schedulers
- **Geometry Optimization**: Sella-based optimization workflows
- **High-Throughput Workflows**: SQLite-based tracking for large-scale campaigns
- **Analysis Tools**: Parsers for ORCA outputs (energies, forces, timings)
- **Job Monitoring**: Dashboard and status checking utilities

## Installation

```bash
# Basic install
pip install -e .

# With development dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

## Quick Start

### Running a Single Calculation

```python
from ase.io import read
from oact_utilities.core.orca import ase_relaxation

atoms = read("molecule.xyz")
result = ase_relaxation(atoms, charge=0, mult=1)
print(f"Final energy: {result.get_potential_energy()} eV")
```

### High-Throughput Workflow

For large-scale architector campaigns:

```python
from oact_utilities.utils.architector import create_workflow_db
from oact_utilities.workflows import ArchitectorWorkflow, JobStatus

# 1. Create workflow database from CSV
db_path = create_workflow_db(
    csv_path="architector_output.csv",
    db_path="workflow.db",
    geometry_column="aligned_csd_core",
)

# 2. Submit batch of jobs
with ArchitectorWorkflow(db_path) as wf:
    ready = wf.get_jobs_by_status(JobStatus.READY)
    print(f"{len(ready)} jobs ready to submit")
```

**On HPC:**
```bash
# Submit 100 jobs to Flux
python -m oact_utilities.workflows.submit_jobs workflow.db jobs/ --batch-size 100 --scheduler flux

# Monitor progress
python -m oact_utilities.workflows.dashboard workflow.db --update jobs/ --show-metrics
```

See [`oact_utilities/workflows/QUICKSTART.md`](oact_utilities/workflows/QUICKSTART.md) for detailed workflow examples.

## Project Structure

```
oact_utilities/
├── core/orca/          # ORCA calculator and recipes
├── utils/              # HPC, analysis, and parsing utilities
├── workflows/          # High-throughput workflow management
└── scripts/            # Campaign-specific workflow scripts
```

## HPC Systems

Supports both **Flux** (LLNL Tuolumne) and **SLURM** (DoD systems):

```python
from oact_utilities.utils.hpc import write_flux_no_template

write_flux_no_template(
    root_dir="jobs/",
    n_cores=4,
    n_hours=2,
    queue="pbatch",
    allocation="dnn-sim",
)
```

## Analysis & Parsing

Extract metrics from ORCA outputs:

```python
from oact_utilities.utils.analysis import parse_job_metrics

# Works with regular and gzipped (quacc) outputs
metrics = parse_job_metrics("job_dir/", unzip=False)

print(f"Max forces: {metrics['max_forces']} Eh/Bohr")
print(f"SCF steps: {metrics['scf_steps']}")
print(f"Final energy: {metrics['final_energy']} Hartree")
```

Supported parsers:
- `parse_max_forces()` - Max gradient from output
- `parse_scf_steps()` - SCF iteration count
- `parse_final_energy()` - Final energy
- `get_engrad()` - Energy and gradient from `.engrad` file (includes max force)
- `parse_job_metrics()` - All-in-one with gzip support

## Development

```bash
# Run tests
pytest tests/

# Format code
black .
ruff check . --fix

# Type check
mypy oact_utilities/
```

See [CLAUDE.md](CLAUDE.md) for detailed development guidelines.

## Key Dependencies

- **ORCA 6.0+** - Quantum chemistry engine
- **ASE** - Atomic Simulation Environment
- **quacc** - Workflow management
- **Sella** - Geometry optimization
- **pandas** - Data analysis
- **SQLite** - Workflow tracking (built-in)

## Documentation

- **[CLAUDE.md](CLAUDE.md)** - Project instructions and architecture
- **[oact_utilities/workflows/README.md](oact_utilities/workflows/README.md)** - Workflow system documentation
- **[oact_utilities/workflows/QUICKSTART.md](oact_utilities/workflows/QUICKSTART.md)** - Workflow quick start guide

## License

Internal LLNL project - contact maintainers for access.
