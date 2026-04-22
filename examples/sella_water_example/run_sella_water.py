"""Sella geometry optimization of water using per-geometry job directories.

Mirrors the submit_jobs.py Sella workflow:
  1. For each geometry, create a separate job directory.
  2. write_orca_inputs()       -> orca.inp
  3. write_sella_runner_shim() -> sella_config.json + run_sella.py
  4. Execute run_sella.py      (same command as HPC scheduler uses)

The three water geometries are intentionally distorted so Sella has work to do.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ase.build import molecule

from oact_utilities.core.orca.calc import write_orca_inputs
from oact_utilities.core.orca.sella_runner import write_sella_runner_shim

# --- configuration -----------------------------------------------------------
ORCA_CMD = "/Users/santiagovargas/Documents/orca_6_1_0_macosx_arm64_openmpi411/orca"
JOBS_ROOT = Path(__file__).parent / "jobs"
CHARGE = 0
MULT = 1
NPROCS = 4
FUNCTIONAL = "wB97M-V"
SIMPLE_INPUT = "omol"
FMAX = 0.05
MAX_STEPS = 100
# -----------------------------------------------------------------------------

# Three slightly distorted water geometries
GEOMETRIES: list[tuple[str, list[list[float]]]] = [
    (
        "water_compressed",
        [[0.0, 0.0, 0.119], [0.0, 0.72, 0.52], [0.0, -0.72, 0.52]],  # short O-H
    ),
    (
        "water_stretched",
        [[0.0, 0.0, 0.119], [0.0, 1.05, 0.65], [0.0, -1.05, 0.65]],  # long O-H
    ),
    (
        "water_bent",
        [[0.0, 0.0, 0.119], [0.0, 0.80, 0.70], [0.0, -0.55, 0.80]],  # asymmetric
    ),
]


def prepare_job(name: str, positions: list[list[float]]) -> Path:
    """Create job directory, write orca.inp and run_sella.py."""

    job_dir = JOBS_ROOT / name
    job_dir.mkdir(parents=True, exist_ok=True)

    atoms = molecule("H2O")
    atoms.set_positions(positions)

    orcasimpleinput, orcablocks_list = write_orca_inputs(
        atoms=atoms,
        output_directory=str(job_dir),
        charge=CHARGE,
        mult=MULT,
        nbo=False,
        mbis=False,
        cores=NPROCS,
        functional=FUNCTIONAL,
        simple_input=SIMPLE_INPUT,
        orca_path=ORCA_CMD,
    )

    write_sella_runner_shim(
        outputdir=job_dir,
        charge=CHARGE,
        mult=MULT,
        orcasimpleinput=orcasimpleinput,
        orcablocks="\n".join(orcablocks_list),
        fmax=FMAX,
        max_steps=MAX_STEPS,
        orca_cmd=ORCA_CMD,
        save_all_steps=True,
    )

    return job_dir


def run_job(job_dir: Path) -> None:
    """Run run_sella.py in the job directory - same call as the HPC scheduler."""
    print(f"  Running: {job_dir.name}")
    result = subprocess.run(
        [sys.executable, "run_sella.py"],
        cwd=job_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  FAILED:\n{result.stderr[-500:]}")


def report(job_dir: Path) -> None:
    """Print sella_status.txt and final geometry."""
    status_file = job_dir / "sella_status.txt"
    xyz_file = job_dir / "orca.xyz"

    status = (
        status_file.read_text().strip() if status_file.exists() else "no status file"
    )
    print(f"\n  {job_dir.name}:")
    for line in status.splitlines():
        print(f"    {line}")

    if xyz_file.exists():
        lines = xyz_file.read_text().splitlines()
        # skip header (atom count + blank comment)
        for line in lines[2:]:
            print(f"    {line}")


if __name__ == "__main__":
    JOBS_ROOT.mkdir(exist_ok=True)

    print("=== Preparing job directories ===")
    job_dirs = []
    for name, positions in GEOMETRIES:
        job_dir = prepare_job(name, positions)
        print(f"  {job_dir.relative_to(Path(__file__).parent)}/")
        print("    orca.inp, sella_config.json, run_sella.py")
        job_dirs.append(job_dir)

    print("\n=== Running Sella optimizations ===")
    for job_dir in job_dirs:
        run_job(job_dir)

    print("\n=== Results ===")
    for job_dir in job_dirs:
        report(job_dir)
