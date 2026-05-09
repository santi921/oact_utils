"""Sella geometry optimization runner for ORCA/ASE workflows.

This module provides:
- run_sella_optimization(): Importable, testable Sella optimization logic.
- write_sella_runner_shim(): Generates a config + static shim for HPC batch execution.
"""

from __future__ import annotations

import json
import os
import shutil
import textwrap
import threading
from pathlib import Path

from oact_utilities.utils.create import read_geom_from_inp_file

# Guard os.chdir() calls -- process-global side effect that is unsafe
# if multiple threads call run_sella_optimization concurrently.
_chdir_lock = threading.Lock()


def _save_step_outputs(job_path: Path, step_counter: list[int]) -> None:
    """Copy all ORCA output files to a numbered step directory.

    Intended for use as a Sella ``attach()`` callback so that per-step
    ORCA outputs (charges, SCF info, wavefunctions) are preserved instead
    of being overwritten on the next gradient evaluation.

    Args:
        job_path: Job directory containing ORCA output files.
        step_counter: Single-element list used as a mutable counter.
    """
    step_dir = job_path / f"step_{step_counter[0]:03d}"
    step_dir.mkdir(exist_ok=True)

    for f in job_path.glob("orca.*"):
        if f.is_file():
            shutil.copy2(f, step_dir / f.name)

    step_counter[0] += 1


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
    save_all_steps: bool = False,
) -> None:
    """Run Sella geometry optimization using ASE ORCA calculator.

    Reads the initial geometry from ``orca.inp`` (written by write_orca_inputs),
    runs a Sella optimization, and writes machine-readable status output.

    Writes:
        - orca.xyz: Final optimized geometry.
        - sella_status.txt: Machine-readable status with metadata.
        - sella.log: Sella optimization log.
        - opt.traj: ASE trajectory file.
        - step_NNN/: (when save_all_steps=True) Per-step copies of all orca.* files.

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
        save_all_steps: Copy all ORCA output files to step_NNN/ directories
            after each gradient evaluation. Useful for preserving per-step
            Mulliken charges, SCF info, and wavefunctions. Defaults to False
            because .gbw files can be large.
    """
    # Lazy imports: these are optional dependencies that should not break
    # submit_jobs.py when only write_sella_runner_shim is needed.
    from ase.calculators.orca import ORCA, OrcaProfile
    from ase.io import write
    from sella import Sella

    job_path = Path(job_dir).resolve()

    # ORCA needs cwd set to the job directory for scratch files.
    # Lock spans the entire chdir-to-restore window so concurrent threads
    # cannot interleave their own os.chdir calls.
    with _chdir_lock:
        saved_cwd = os.getcwd()
        os.chdir(job_path)

        try:
            # Read initial geometry from orca.inp using our parser
            # (ASE orca-input format not available in all versions)
            atoms = read_geom_from_inp_file(
                str(job_path / "orca.inp"), ase_format_tf=True
            )

            # Set up ORCA calculator for energy+gradient evaluations
            calc = ORCA(
                profile=OrcaProfile(command=orca_cmd),
                charge=charge,
                mult=mult,
                orcasimpleinput=orcasimpleinput,
                orcablocks=orcablocks,
                directory=str(job_path),
            )

            # If orca.engrad + orca.out already exist (seeded from prior SP),
            # pre-populate the calculator's results cache so Sella skips the
            # redundant step-0 EnGrad call and goes straight to step 1.
            engrad_path = job_path / "orca.engrad"
            out_path = job_path / "orca.out"
            if engrad_path.exists() and out_path.exists():
                try:
                    calc.results = calc.template.read_results(job_path)
                    calc.atoms = atoms.copy()
                    print(
                        "[sella_runner] seeded from existing orca.engrad (step 0 skipped)"
                    )
                except Exception as e:
                    print(
                        f"[sella_runner] warning: could not pre-seed from orca.engrad: {e}"
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

                if save_all_steps:
                    step_counter: list[int] = [0]
                    opt.attach(
                        _save_step_outputs,
                        interval=1,
                        job_path=job_path,
                        step_counter=step_counter,
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
                        f"status: CONVERGED\nsteps: {n_steps}\n"
                        f"final_fmax: {final_fmax:.6f}\n"
                    )
                else:
                    status_file.write_text(
                        f"status: NOT_CONVERGED\nsteps: {n_steps}\n"
                        f"final_fmax: {final_fmax:.6f}\n"
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
    save_all_steps: bool = False,
) -> Path:
    """Generate a JSON config and static shim script for HPC batch execution.

    Writes two files:
    - sella_config.json: All optimization parameters as pure data.
    - run_sella.py: Static script that reads the JSON and calls
      run_sella_optimization(). No user values are interpolated into code.

    Args:
        outputdir: Directory where files will be written.
        charge: Molecular charge.
        mult: Spin multiplicity.
        orcasimpleinput: ORCA simple input line.
        orcablocks: ORCA blocks as a single string.
        fmax: Force convergence threshold.
        max_steps: Maximum optimization steps.
        order: Sella saddle order.
        internal: Use internal coordinates.
        orca_cmd: Path to ORCA executable.
        save_all_steps: Copy all ORCA output files per step.

    Returns:
        Path to the generated run_sella.py script.
    """
    outputdir = Path(outputdir)

    # Write parameters as JSON data -- no code generation with user values
    config = {
        "charge": charge,
        "mult": mult,
        "orcasimpleinput": orcasimpleinput,
        "orcablocks": orcablocks,
        "fmax": fmax,
        "max_steps": max_steps,
        "order": order,
        "internal": internal,
        "orca_cmd": orca_cmd,
        "save_all_steps": save_all_steps,
    }
    config_path = outputdir / "sella_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    # Write a static shim that reads the config -- no interpolation needed
    shim_path = outputdir / "run_sella.py"
    script = textwrap.dedent(
        """\
        #!/usr/bin/env python
        \"\"\"Generated Sella runner. Reads sella_config.json for parameters.\"\"\"
        import json
        from pathlib import Path
        from oact_utilities.core.orca.sella_runner import run_sella_optimization

        config = json.loads(Path("sella_config.json").read_text())
        run_sella_optimization(job_dir=".", **config)
    """
    )

    shim_path.write_text(script)
    shim_path.chmod(0o700)

    return shim_path
