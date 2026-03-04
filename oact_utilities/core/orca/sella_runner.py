"""Sella geometry optimization runner for ORCA/ASE workflows.

This module provides:
- run_sella_optimization(): Importable, testable Sella optimization logic.
- write_sella_runner_shim(): Generates a thin run_sella.py script for HPC batch execution.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

from ase.calculators.orca import ORCA, OrcaProfile
from ase.io import read, write
from sella import Sella


def run_sella_optimization(
    job_dir: str,
    charge: int,
    mult: int,
    orcasimpleinput: str,
    orcablocks: str,
    fmax: float = 0.05,
    max_steps: int = 100,
    order: int = 0,
    internal: bool = True,
    orca_cmd: str = "orca",
) -> None:
    """Run Sella geometry optimization using ASE ORCA calculator.

    Reads the initial geometry from ``orca.inp`` (written by write_orca_inputs),
    runs a Sella optimization, and writes machine-readable status output.

    Writes:
        - orca.xyz: Final optimized geometry.
        - sella_status.txt: Machine-readable status with metadata.
        - sella.log: Sella optimization log.
        - opt.traj: ASE trajectory file.

    Args:
        job_dir: Path to job directory containing orca.inp.
        charge: Molecular charge.
        mult: Spin multiplicity.
        orcasimpleinput: ORCA simple input line (e.g. "! wB97M-V def2-TZVPD ...").
        orcablocks: ORCA blocks as a single string (newline-separated).
        fmax: Force convergence threshold in Eh/Bohr.
        max_steps: Maximum optimization steps.
        order: Sella saddle order (0 = minimum, 1 = transition state).
        internal: Use internal coordinates for Sella.
        orca_cmd: Path to ORCA executable.
    """
    job_path = Path(job_dir).resolve()

    # ORCA needs cwd set to the job directory for scratch files.
    # Save and restore to avoid leaking a process-global side effect.
    saved_cwd = os.getcwd()
    os.chdir(job_path)

    try:
        # Read initial geometry from orca.inp
        atoms = read(str(job_path / "orca.inp"), format="orca-input")

        # Set up ORCA calculator for energy+gradient evaluations
        calc = ORCA(
            profile=OrcaProfile(command=orca_cmd),
            charge=charge,
            mult=mult,
            orcasimpleinput=orcasimpleinput,
            orcablocks=orcablocks,
            directory=str(job_path),
        )
        atoms.calc = calc

        traj_file = str(job_path / "opt.traj")
        log_file = str(job_path / "sella.log")
        status_file = job_path / "sella_status.txt"

        try:
            opt = Sella(
                atoms,
                trajectory=traj_file,
                logfile=log_file,
                append_trajectory=True,
                internal=internal,
                order=order,
            )

            converged = opt.run(fmax=fmax, steps=max_steps)
            n_steps = opt.nsteps

            # Get final max force
            forces = atoms.get_forces()
            final_fmax = float((forces**2).sum(axis=1).max() ** 0.5)

            # Write final geometry
            write(str(job_path / "orca.xyz"), atoms, format="xyz")

            if converged:
                status_file.write_text(
                    f"status: CONVERGED\nsteps: {n_steps}\nfinal_fmax: {final_fmax:.6f}\n"
                )
            else:
                status_file.write_text(
                    f"status: NOT_CONVERGED\nsteps: {n_steps}\nfinal_fmax: {final_fmax:.6f}\n"
                )

        except Exception as e:
            # Write error status so the status checker can detect the failure
            msg = str(e).replace("\n", " ")[:200]
            status_file.write_text(f"status: ERROR\nmessage: {msg}\n")
            raise

    finally:
        os.chdir(saved_cwd)


def write_sella_runner_shim(
    outputdir: str | Path,
    charge: int,
    mult: int,
    orcasimpleinput: str,
    orcablocks: str,
    fmax: float = 0.05,
    max_steps: int = 100,
    order: int = 0,
    internal: bool = True,
    orca_cmd: str = "orca",
) -> Path:
    """Generate a thin run_sella.py shim with all values embedded.

    The generated script simply imports and calls run_sella_optimization()
    with hardcoded parameter values. No re-parsing of orca.inp needed.

    Args:
        outputdir: Directory where run_sella.py will be written.
        charge: Molecular charge.
        mult: Spin multiplicity.
        orcasimpleinput: ORCA simple input line.
        orcablocks: ORCA blocks as a single string.
        fmax: Force convergence threshold.
        max_steps: Maximum optimization steps.
        order: Sella saddle order.
        internal: Use internal coordinates.
        orca_cmd: Path to ORCA executable.

    Returns:
        Path to the generated run_sella.py script.
    """
    outputdir = Path(outputdir)
    shim_path = outputdir / "run_sella.py"

    script = textwrap.dedent(
        f"""\
        #!/usr/bin/env python
        \"\"\"Generated Sella optimization runner. Do not edit — regenerate via submit_jobs.\"\"\"
        from oact_utilities.core.orca.sella_runner import run_sella_optimization

        run_sella_optimization(
            job_dir=".",
            charge={charge!r},
            mult={mult!r},
            orcasimpleinput={orcasimpleinput!r},
            orcablocks={orcablocks!r},
            fmax={fmax!r},
            max_steps={max_steps!r},
            order={order!r},
            internal={internal!r},
            orca_cmd={orca_cmd!r},
        )
    """
    )

    shim_path.write_text(script)
    shim_path.chmod(0o755)

    return shim_path
