"""MLIP (fairchem eSEN / UMA) pre-optimization with ASE LBFGS.

Mirrors ``examples/benchmarks/relax_tight.py``: ``FAIRChemCalculator`` with the ``omol`` task,
charge and spin multiplicity passed through ``atoms.info``. The model is loaded once per process
(cached) and never per step. Element coverage is checked before any model evaluation, because an
MLIP returns a number for an element it was never trained on.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from oact_utilities.core.preopt._common import (
    PreoptResult,
    format_xyz,
    kabsch_rmsd,
    max_atom_force,
    parse_xyz,
)
from oact_utilities.core.preopt.elements import (
    check_elements,
    get_supported_elements,
    supported_from_predict_unit,
)


def _is_file(checkpoint: str) -> bool:
    return Path(checkpoint).suffix == ".pt" or Path(checkpoint).is_file()


@lru_cache(maxsize=4)
def load_mlip(checkpoint: str, device: str = "cpu") -> tuple[Any, frozenset[str]]:
    """Load a predict unit once per process and return it with its supported elements.

    Args:
        checkpoint: Path to an OAct inference ``.pt`` file, or a fairchem registry name.
        device: ``"cpu"`` or ``"cuda"``.
    """
    if _is_file(checkpoint):
        from fairchem.core.units.mlip_unit import load_predict_unit

        unit = load_predict_unit(path=checkpoint, device=device)
        return unit, get_supported_elements(checkpoint)
    from fairchem.core.calculate.pretrained_mlip import get_predict_unit

    unit = get_predict_unit(checkpoint, device=device)
    return unit, supported_from_predict_unit(unit)


def run_mlip_preopt(
    xyz: str,
    charge: int,
    mult: int,
    checkpoint: str,
    device: str = "cpu",
    fmax: float = 0.05,
    max_steps: int = 200,
    task_name: str = "omol",
    job_id: str | int | None = None,
) -> tuple[str, PreoptResult]:
    """Relax ``xyz`` with an MLIP and ASE LBFGS.

    Args:
        xyz: Single-frame XYZ string.
        charge: Molecular charge.
        mult: Spin multiplicity (2S+1).
        checkpoint: OAct ``.pt`` path or fairchem registry name.
        device: ``"cpu"`` or ``"cuda"``.
        fmax: Largest per-atom force (eV/Angstrom) at which LBFGS stops.
        max_steps: Maximum LBFGS steps.
        task_name: fairchem task head (``"omol"``).
        job_id: Included in ``UnsupportedElementsError`` messages.

    Returns:
        The relaxed XYZ string (the input when the run failed) and the ``PreoptResult``.
        Raises ``UnsupportedElementsError`` for elements outside the checkpoint's coverage; any
        other failure is returned in ``result.error``.
    """
    from ase import Atoms
    from ase.optimize import LBFGS
    from fairchem.core import FAIRChemCalculator

    t0 = time.time()
    symbols, p0 = parse_xyz(xyz)
    name = Path(checkpoint).name
    result = PreoptResult(method="mlip", detail=f"checkpoint={name}; device={device}")
    unit, supported = load_mlip(checkpoint, device)
    check_elements(symbols, supported, f"MLIP {name}", job_id)

    try:
        atoms = Atoms(symbols, positions=p0)
        atoms.info = {"charge": int(charge), "spin": int(mult)}
        atoms.calc = FAIRChemCalculator(unit, task_name=task_name)
        opt = LBFGS(atoms, logfile=os.devnull)
        opt.run(fmax=fmax, steps=max_steps)
        forces = atoms.get_forces()
        result.energy_ev = float(atoms.get_potential_energy())
    except (
        Exception
    ) as exc:  # noqa: BLE001 -- recorded in the status, caller applies preopt_on_fail
        result.error = f"{type(exc).__name__}: {exc}"[:500]
        result.seconds = time.time() - t0
        return xyz, result

    result.fmax_final = max_atom_force(np.asarray(forces))
    result.n_steps = int(opt.nsteps)
    result.converged = result.fmax_final <= fmax
    result.rmsd_from_input = kabsch_rmsd(p0, atoms.positions)
    result.seconds = time.time() - t0
    return format_xyz(symbols, atoms.positions), result
