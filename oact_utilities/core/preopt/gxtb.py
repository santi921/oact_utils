"""g-xTB pre-optimization through the g-xTB build of xtb (``xtb --gxtb``).

Logic carried over from ``examples/benchmarks/run_gxtb_bench.py``:

- Singlets run restricted (no ``--uhf``; any ``--uhf``, even 0, forces UHF); 2S+1 > 1 uses
  ``--uhf 2S``.
- The SCF cap is raised to 1000 iterations (xtb default 250 stops some actinide optimizations).
- ANCopt converges on the Euclidean norm of the full gradient, which bounds the largest per-atom
  force, so the optimizer level is the loosest one whose threshold is at or below the requested
  fmax. The result is then verified with a gradient single point restarted from the optimizer's
  ``xtbrestart``: a fresh SCF on an open-shell actinide can land on a different solution.
- If the first SCF fails, the guess is seeded from an SCF at a higher electronic temperature and
  the optimization rerun at the default temperature from that density.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from oact_utilities.core.preopt._common import (
    EH_BOHR_TO_EV_ANG,
    HARTREE_EV,
    PreoptResult,
    format_xyz,
    kabsch_rmsd,
    max_atom_force,
    parse_xyz,
)
from oact_utilities.core.preopt.elements import check_elements, gxtb_supported_elements

# xtb ANCopt gradient-norm criterion per level (Eh/bohr), read from xtb's own output.
OPT_LEVELS: dict[str, float] = {
    "crude": 1e-2,
    "sloppy": 6e-3,
    "loose": 4e-3,
    "lax": 2.5e-3,
    "normal": 1e-3,
    "tight": 8e-4,
    "vtight": 2e-4,
    "extreme": 5e-5,
}


class SCFNotConverged(RuntimeError):
    """xtb stopped because an SCF hit the iteration cap."""


# vtight's bound is 0.0103 eV/A; the margin keeps it the choice for fmax = 0.01 eV/A. The
# verification gradient decides ``converged``, so the margin cannot hide an unconverged result.
LEVEL_MARGIN = 1.05


def select_opt_level(fmax_ev_ang: float) -> str:
    """Loosest xtb level whose gradient-norm threshold is within ``LEVEL_MARGIN`` of ``fmax``."""
    ok = [
        (thr, name)
        for name, thr in OPT_LEVELS.items()
        if thr * EH_BOHR_TO_EV_ANG <= fmax_ev_ang * LEVEL_MARGIN
    ]
    return max(ok)[1] if ok else "extreme"


def read_engrad(path: Path) -> tuple[float, np.ndarray]:
    """Energy (Eh) and gradient (n, 3) in Eh/bohr from an ORCA-style ``.engrad`` from xtb."""
    vals = [ln for ln in path.read_text().splitlines() if ln.strip() and ln[0] != "#"]
    n = int(vals[0])
    return float(vals[1]), np.array([float(v) for v in vals[2 : 2 + 3 * n]]).reshape(
        n, 3
    )


def log_energies(path: Path) -> list[float]:
    """Per-cycle energies (Eh) from the comment lines of ``xtbopt.log``."""
    return [
        float(ln.split()[1])
        for ln in path.read_text().splitlines()
        if ln.strip().startswith("energy:")
    ]


def xtb_version(xtb_path: str) -> str:
    """First line of ``xtb --version`` that names the version, or ``""``."""
    try:
        out = subprocess.run(
            [xtb_path, "--version"], capture_output=True, text=True, timeout=30
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""
    for ln in out.splitlines():
        if "version" in ln.lower():
            return ln.strip(" |")
    return ""


def _unlimit_stack() -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY,) * 2)
    except (ImportError, ValueError, OSError):
        pass


def _run_xtb(
    xtb_path: str,
    xyz: str,
    workdir: Path,
    flags: list[str],
    env: dict[str, str],
    timeout: float | None,
    restart: Path | None = None,
) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "in.xyz").write_text(xyz)
    if restart is not None:
        shutil.copy(restart, workdir / "xtbrestart")
    with open(workdir / "xtb.out", "w") as fo, open(workdir / "xtb.err", "w") as fe:
        proc = subprocess.run(
            [xtb_path, "in.xyz", *flags],
            cwd=workdir,
            stdout=fo,
            stderr=fe,
            env=env,
            timeout=timeout,
            preexec_fn=_unlimit_stack if os.name == "posix" else None,
        )
    # xtb prints its termination banner on stderr.
    out = (workdir / "xtb.out").read_text(errors="replace")
    banner = out + (workdir / "xtb.err").read_text(errors="replace")
    if proc.returncode != 0 or "normal termination of xtb" not in banner:
        tail = (
            (workdir / "xtb.err").read_text(errors="replace").strip().splitlines()[-3:]
        )
        msg = f"xtb {' '.join(flags)} exit {proc.returncode}: {tail}"
        if "SCF not converged" in out:
            raise SCFNotConverged(msg + " (SCF not converged)")
        raise RuntimeError(msg)


def run_gxtb_preopt(
    xyz: str,
    charge: int,
    mult: int,
    workdir: str | Path,
    xtb_path: str = "xtb",
    fmax: float = 0.05,
    max_cycles: int = 500,
    scf_iterations: int = 1000,
    threads: int = 1,
    timeout: float | None = None,
    seed_etemp: float = 3000.0,
) -> tuple[str, PreoptResult]:
    """Relax ``xyz`` with g-xTB and verify the final per-atom force.

    Args:
        xyz: Single-frame XYZ string.
        charge: Molecular charge.
        mult: Spin multiplicity (2S+1).
        workdir: Directory for xtb scratch; any existing contents are deleted first.
        xtb_path: g-xTB build of the xtb executable.
        fmax: Target largest per-atom force (eV/Angstrom). Sets the xtb optimizer level.
        max_cycles: Maximum ANCopt cycles.
        scf_iterations: SCF iteration cap passed to xtb.
        threads: OMP threads for xtb.
        timeout: Seconds allowed per xtb call; None for no limit.
        seed_etemp: Electronic temperature (K) of the guess-seeding SCF used after a failed
            first SCF.

    Returns:
        The relaxed XYZ string (the input when the run failed) and the ``PreoptResult``.
        Raises ``UnsupportedElementsError`` for elements beyond Z = 103; any other failure is
        returned in ``result.error``.
    """
    t0 = time.time()
    workdir = Path(workdir)
    symbols, p0 = parse_xyz(xyz)
    check_elements(symbols, gxtb_supported_elements(), "g-xTB")

    level = select_opt_level(fmax)
    result = PreoptResult(method="gxtb", detail=f"opt={level}; {xtb_version(xtb_path)}")
    env = os.environ | {
        "OMP_NUM_THREADS": f"{threads},1",
        "OMP_MAX_ACTIVE_LEVELS": "1",
        "OMP_STACKSIZE": "4G",
    }
    base = ["--gxtb", "--chrg", str(charge), "--iterations", str(scf_iterations)]
    if mult > 1:
        base += ["--uhf", str(mult - 1)]
    opt_flags = base + ["--opt", level, "--cycles", str(max_cycles)]
    opt_dir = workdir / "opt"

    try:
        # xtb restarts from any xtbrestart in its cwd, so a requeued job would start
        # from the previous attempt's density.
        if workdir.exists():
            shutil.rmtree(workdir)
        try:
            _run_xtb(xtb_path, xyz, opt_dir, opt_flags, env, timeout)
        except SCFNotConverged:
            seed_dir = workdir / "scf_seed"
            _run_xtb(
                xtb_path,
                xyz,
                seed_dir,
                base + ["--etemp", f"{seed_etemp:g}"],
                env,
                timeout,
            )
            shutil.rmtree(opt_dir)
            _run_xtb(
                xtb_path, xyz, opt_dir, opt_flags, env, timeout, seed_dir / "xtbrestart"
            )
            result.detail += f"; scf_seed etemp {seed_etemp:g} K"
        final = (opt_dir / "xtbopt.xyz").read_text()
        sp_dir = workdir / "sp"
        _run_xtb(
            xtb_path,
            final,
            sp_dir,
            base + ["--grad"],
            env,
            timeout,
            opt_dir / "xtbrestart",
        )
        energy_eh, grad = read_engrad(sp_dir / "in.engrad")
    except (
        Exception
    ) as exc:  # noqa: BLE001 -- recorded in the status, caller applies preopt_on_fail
        result.error = f"{type(exc).__name__}: {exc}"[:500]
        result.seconds = time.time() - t0
        return xyz, result

    _, p1 = parse_xyz(final)
    result.fmax_final = max_atom_force(grad) * EH_BOHR_TO_EV_ANG
    result.energy_ev = energy_eh * HARTREE_EV
    result.n_steps = len(log_energies(opt_dir / "xtbopt.log"))
    # xtb exits 0 with "normal termination" even when ANCopt runs out of cycles, so the
    # measured force is the only convergence signal.
    result.converged = result.fmax_final <= fmax
    result.rmsd_from_input = kabsch_rmsd(p0, p1)
    result.seconds = time.time() - t0
    return format_xyz(symbols, p1), result
