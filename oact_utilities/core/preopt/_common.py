"""Shared result record and geometry helpers for the pre-optimizers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

EH_BOHR_TO_EV_ANG = 51.42206313
HARTREE_EV = 27.211386245988
STATUS_FILE = "preopt_status.json"
PREOPT_XYZ = "preopt.xyz"


@dataclass
class PreoptResult:
    """Outcome of one pre-optimization, serialized to ``preopt_status.json``.

    Attributes:
        method: ``"gxtb"`` or ``"mlip"``.
        converged: True when ``fmax_final`` is at or below the requested fmax.
        fmax_final: Largest per-atom force norm of the returned geometry (eV/Angstrom).
        n_steps: Optimizer cycles taken.
        energy_ev: Energy of the returned geometry (eV).
        rmsd_from_input: Heavy-atom-agnostic Kabsch RMSD between input and output (Angstrom).
        seconds: Wall time (s).
        error: Empty on success, else ``"<ExceptionType>: <message>"``.
        detail: Method-specific provenance (xtb opt level, checkpoint name, ...).
    """

    method: str
    converged: bool = False
    fmax_final: float | None = None
    n_steps: int | None = None
    energy_ev: float | None = None
    rmsd_from_input: float | None = None
    seconds: float | None = None
    error: str = ""
    detail: str = ""

    @property
    def ok(self) -> bool:
        """True when the run produced a geometry without an error."""
        return not self.error

    def write(self, job_dir: str | Path) -> Path:
        """Write ``preopt_status.json`` into ``job_dir`` and return its path."""
        path = Path(job_dir) / STATUS_FILE
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return path

    @classmethod
    def read(cls, job_dir: str | Path) -> PreoptResult | None:
        """Read ``preopt_status.json`` from ``job_dir``; None when absent or unreadable."""
        path = Path(job_dir) / STATUS_FILE
        try:
            return cls(**json.loads(path.read_text()))
        except (OSError, ValueError, TypeError):
            return None


def preopt_db_fields(job_dir: str | Path) -> dict[str, Any]:
    """Workflow-DB ``preopt_*`` columns for a job directory; ``{}`` when it had no pre-opt.

    A failed pre-opt that ORCA continued past records the method and ``preopt_converged = 0``.
    """
    res = PreoptResult.read(job_dir)
    if res is None:
        return {}
    return {
        "preopt_method": res.method,
        "preopt_converged": int(res.converged and res.ok),
        "preopt_fmax": res.fmax_final,
        "preopt_energy_ev": res.energy_ev,
        "preopt_rmsd": res.rmsd_from_input,
        "preopt_seconds": res.seconds,
    }


def parse_xyz(text: str) -> tuple[list[str], np.ndarray]:
    """Symbols and coordinates (Angstrom) of a single-frame XYZ string."""
    lines = text.strip("\n").splitlines()
    n = int(lines[0].split()[0])
    rows = [ln.split() for ln in lines[2 : 2 + n]]
    return [r[0] for r in rows], np.array([[float(v) for v in r[1:4]] for r in rows])


def format_xyz(symbols: list[str], coords: np.ndarray, comment: str = "") -> str:
    """XYZ string from symbols and coordinates (Angstrom)."""
    body = "\n".join(
        f"{s:<3} {x:16.10f} {y:16.10f} {z:16.10f}"
        for s, (x, y, z) in zip(symbols, coords)
    )
    return f"{len(symbols)}\n{comment}\n{body}\n"


def kabsch_rmsd(p: np.ndarray, q: np.ndarray) -> float:
    """RMSD (Angstrom) after optimal superposition; rows are the same atoms."""
    p, q = p - p.mean(0), q - q.mean(0)
    u, _, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return float(np.sqrt(((p @ rot.T - q) ** 2).sum(1).mean()))


def max_atom_force(grad_or_forces: np.ndarray) -> float:
    """Largest per-atom norm of an (n, 3) array."""
    return float(np.sqrt((np.asarray(grad_or_forces) ** 2).sum(1)).max())
