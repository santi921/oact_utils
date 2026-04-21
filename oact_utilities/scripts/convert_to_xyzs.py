"""Convert bare ORCA job directories to extended XYZ files for MLIP training.

Walks a root directory of job folders (each containing orca.out and orca.engrad),
parses energy, forces, positions, atomic numbers, charge, spin, and actinide mask,
and writes one extxyz file per job into an output directory.

The output directory can be fed directly to fairchem's create_finetune_dataset.py:

    python create_finetune_dataset.py \\
        --train-dir ./output/train \\
        --val-dir   ./output/val \\
        --output-dir ./lmdb

Units in output:
    energy  : eV  (converted from Hartree)
    forces  : eV/Angstrom  (converted from Eh/Bohr, sign-flipped from gradient)
    positions: Angstrom  (converted from Bohr)

Usage:
    python -m oact_utilities.scripts.convert_to_xyzs \\
        /eagle/BLASTNet/nonact_226_michael \\
        --output-dir /eagle/BLASTNet/fairchem_data_ishan/nonact_226_michael \\
        --workers 16

    python -m oact_utilities.scripts.convert_to_xyzs \\
        /eagle/BLASTNet/nonact_226_michael \\
        --output-dir /eagle/BLASTNet/fairchem_data_ishan/nonact_226_michael \\
        --workers 16 --job-glob "job_*" --debug
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write as ase_write

from oact_utilities.utils.analysis import get_engrad, parse_mulliken_population, parse_scf_steps
from oact_utilities.utils.create import fetch_actinides
from oact_utilities.utils.status import check_file_termination

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HARTREE_TO_EV: float = 27.211386245988
_BOHR_TO_ANG: float = 0.529177210903
_FORCE_CONVERSION: float = _HARTREE_TO_EV / _BOHR_TO_ANG  # Eh/Bohr -> eV/Ang

# Build atomic-number -> symbol lookup from periodictable
try:
    from periodictable import elements as _pt_elements

    _Z_TO_SYMBOL: dict[int, str] = {el.number: el.symbol for el in _pt_elements}
except ImportError:
    _Z_TO_SYMBOL = {}

_ACTINIDE_SET: set[str] = set(fetch_actinides())


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a named logger with a timestamped StreamHandler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Per-job helpers
# ---------------------------------------------------------------------------


def _z_to_symbol(z: int) -> str:
    return _Z_TO_SYMBOL.get(z, f"Z{z}")


def _parse_charge_spin(inp_path: Path) -> tuple[int, int]:
    """Extract charge and spin multiplicity from an ORCA input file.

    Reads the ``* xyz charge spin`` (or ``*xyz charge spin``) coordinate block
    header. Raises ValueError if not found.

    Args:
        inp_path: Path to orca.inp.

    Returns:
        Tuple of (charge, spin_multiplicity).
    """
    with open(inp_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("* xyz"):
                parts = stripped.split()
                return int(parts[2]), int(parts[3])
            if stripped.startswith("*xyz"):
                parts = stripped.split()
                return int(parts[1]), int(parts[2])
    raise ValueError(f"No '* xyz charge spin' block found in {inp_path}")


def _build_atoms(
    engrad_data: dict,
    charge: int,
    spin: int,
    job_id: str,
    mulliken_data: dict | None = None,
    scf_steps: int | None = None,
) -> Atoms:
    """Construct an ASE Atoms object with a SinglePointCalculator attached.

    Args:
        engrad_data: Dict returned by ``get_engrad()``.
        charge: Molecular charge.
        spin: Spin multiplicity (2S+1).
        job_id: Directory name used as identifier (e.g. "job_13").
        mulliken_data: Dict from ``parse_mulliken_population()`` or None.
        scf_steps: Total SCF iterations or None.

    Returns:
        ASE Atoms with energy, forces, actinide_mask, charge, spin, job_id set.
        Per-atom mulliken/loewdin arrays and scf_steps added when available.
    """
    atomic_numbers: list[int] = engrad_data["elements"]
    coords_bohr: list[list[float]] = engrad_data["coords_bohr"]
    gradient_flat: list[float] = engrad_data["gradient_Eh_per_bohr"]
    energy_eh: float = engrad_data["total_energy_Eh"]

    natoms = len(atomic_numbers)
    positions_ang = np.array(coords_bohr) * _BOHR_TO_ANG  # (N, 3) Angstrom
    gradient_arr = np.array(gradient_flat).reshape(natoms, 3)  # Eh/Bohr
    forces_ev_ang = -gradient_arr * _FORCE_CONVERSION  # eV/Angstrom

    symbols = [_z_to_symbol(z) for z in atomic_numbers]
    actinide_mask = np.array(
        [1 if sym in _ACTINIDE_SET else 0 for sym in symbols], dtype=np.int8
    )

    atoms = Atoms(
        symbols=symbols,
        positions=positions_ang,
        pbc=False,
    )

    calc = SinglePointCalculator(
        atoms,
        energy=energy_eh * _HARTREE_TO_EV,
        forces=forces_ev_ang,
    )
    atoms.calc = calc

    # Frame-level metadata stored in atoms.info (written to extxyz comment line)
    atoms.info["charge"] = charge
    atoms.info["spin"] = spin
    atoms.info["job_id"] = job_id

    # Per-atom array stored in atoms.arrays (written as extxyz per-atom column)
    atoms.arrays["actinide_mask"] = actinide_mask

    if mulliken_data is not None:
        mc = mulliken_data.get("mulliken_charges")
        ms = mulliken_data.get("mulliken_spins")
        lc = mulliken_data.get("loewdin_charges")
        ls = mulliken_data.get("loewdin_spins")
        if mc and len(mc) == natoms:
            atoms.arrays["mulliken_charges"] = np.array(mc, dtype=np.float64)
        if ms and len(ms) == natoms:
            atoms.arrays["mulliken_spins"] = np.array(ms, dtype=np.float64)
        if lc and len(lc) == natoms:
            atoms.arrays["loewdin_charges"] = np.array(lc, dtype=np.float64)
        if ls and len(ls) == natoms:
            atoms.arrays["loewdin_spins"] = np.array(ls, dtype=np.float64)

    if scf_steps is not None:
        atoms.info["scf_steps"] = scf_steps

    return atoms


def _find_out_file(job_dir: Path) -> Path | None:
    """Return the primary ORCA output file in job_dir, or None."""
    candidate = job_dir / "orca.out"
    if candidate.exists():
        return candidate
    matches = list(job_dir.glob("*.out"))
    # Prefer orca.out; skip atom-specific outputs like orca_atom84.out
    primary = [m for m in matches if "atom" not in m.name]
    return primary[0] if primary else None


def _find_engrad(job_dir: Path) -> Path | None:
    """Return the engrad file in job_dir, or None."""
    candidate = job_dir / "orca.engrad"
    if candidate.exists():
        return candidate
    matches = list(job_dir.glob("*.engrad"))
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Main per-job entry point
# ---------------------------------------------------------------------------


def parse_job_dir(
    job_dir: Path,
    logger: logging.Logger | None = None,
) -> tuple[Atoms | None, str | None]:
    """Parse one ORCA job directory and return an ASE Atoms object.

    Checks that ORCA terminated normally, then parses the engrad file for
    energy/forces/positions and the inp file for charge/spin.

    Args:
        job_dir: Path to a single job directory.
        logger: Optional logger for debug messages.

    Returns:
        Tuple of (atoms, failure_reason). atoms is None on failure, with
        failure_reason explaining why (e.g. "no_out", "not_terminated",
        "no_engrad", "parse_failed", "no_inp", "inp_parse_failed").
    """

    def _dbg(msg: str, *args: object) -> None:
        if logger:
            logger.debug(f"{job_dir.name}: {msg}", *args)

    # 1. Termination check: trust content, not just file existence
    out_file = _find_out_file(job_dir)
    if out_file is None:
        _dbg("no .out file")
        return None, "no_out"

    status = check_file_termination(str(out_file))
    if status != 1:
        _dbg("not terminated normally (status=%d)", status)
        return None, "not_terminated"

    # 2. Parse engrad
    engrad_path = _find_engrad(job_dir)
    if engrad_path is None:
        _dbg("no .engrad file")
        return None, "no_engrad"

    try:
        engrad_data = get_engrad(str(engrad_path))
    except Exception as exc:
        _dbg("engrad parse error: %s", exc)
        return None, "parse_failed"

    required = {"total_energy_Eh", "gradient_Eh_per_bohr", "elements", "coords_bohr"}
    missing = required - engrad_data.keys()
    if missing or not engrad_data.get("elements"):
        _dbg("incomplete engrad data (missing: %s)", missing)
        return None, "incomplete_engrad"

    natoms = len(engrad_data["elements"])
    if len(engrad_data["gradient_Eh_per_bohr"]) != 3 * natoms:
        _dbg("gradient length mismatch")
        return None, "bad_gradient"

    # 3. Parse charge and spin from inp
    inp_path = job_dir / "orca.inp"
    if not inp_path.exists():
        _dbg("no orca.inp")
        return None, "no_inp"

    try:
        charge, spin = _parse_charge_spin(inp_path)
    except Exception as exc:
        _dbg("inp parse error: %s", exc)
        return None, "inp_parse_failed"

    # 4. Parse optional per-atom and frame-level metadata from .out
    mulliken_data = parse_mulliken_population(out_file)
    scf_steps_val = parse_scf_steps(out_file)

    # 5. Build ASE Atoms
    atoms = _build_atoms(
        engrad_data, charge, spin, job_dir.name,
        mulliken_data=mulliken_data,
        scf_steps=scf_steps_val,
    )
    return atoms, None


# ---------------------------------------------------------------------------
# Parallel dataset extraction
# ---------------------------------------------------------------------------


def extract_dataset(
    root_dir: str | Path,
    output_dir: str | Path,
    workers: int = 4,
    job_glob: str = "job_*",
    logger: logging.Logger | None = None,
) -> dict:
    """Extract all jobs from root_dir and write extxyz files to output_dir.

    Each successfully parsed job is written to ``output_dir/{job_id}.xyz``
    in extended XYZ format (ASE-compatible, readable by fairchem).
    Failed jobs are recorded in ``output_dir/{job_id}.failed``.
    A summary is written to ``output_dir/extraction_summary.json``.

    Args:
        root_dir: Directory containing job subdirectories.
        output_dir: Directory to write extxyz files into.
        workers: Number of parallel I/O workers.
        job_glob: Glob pattern for job subdirectory names (default: ``job_*``).
        logger: Logger instance.

    Returns:
        Summary dict with keys: total, succeeded, failed, failure_counts.
    """
    if logger is None:
        logger = _setup_logger("convert_to_xyzs")

    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    job_dirs = sorted(
        [p for p in root_dir.glob(job_glob) if p.is_dir()],
        key=lambda p: p.name,
    )
    if not job_dirs:
        logger.warning("No directories matching '%s' found in %s", job_glob, root_dir)
        return {"total": 0, "succeeded": 0, "failed": 0, "failure_counts": {}}

    logger.info(
        "Found %d job directories. Extracting with %d workers...",
        len(job_dirs),
        workers,
    )

    succeeded = 0
    failed = 0
    failure_counts: dict[str, int] = {}

    def _process(job_dir: Path) -> tuple[str, Atoms | None, str | None]:
        atoms, reason = parse_job_dir(job_dir, logger=logger)
        return job_dir.name, atoms, reason

    def _handle_result(job_id: str, atoms: Atoms | None, reason: str | None) -> None:
        nonlocal succeeded, failed
        if atoms is not None:
            out_path = output_dir / f"{job_id}.xyz"
            ase_write(str(out_path), atoms, format="extxyz")
            succeeded += 1
        else:
            (output_dir / f"{job_id}.failed").write_text(reason or "unknown")
            failure_counts[reason or "unknown"] = (
                failure_counts.get(reason or "unknown", 0) + 1
            )
            failed += 1

    with tqdm(total=len(job_dirs), unit="job") as pbar:
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_process, jd): jd for jd in job_dirs}
                for future in as_completed(futures):
                    job_id, atoms, reason = future.result()
                    _handle_result(job_id, atoms, reason)
                    pbar.set_postfix(ok=succeeded, fail=failed)
                    pbar.update(1)
        else:
            for jd in job_dirs:
                job_id, atoms, reason = _process(jd)
                _handle_result(job_id, atoms, reason)
                pbar.set_postfix(ok=succeeded, fail=failed)
                pbar.update(1)

    summary = {
        "total": len(job_dirs),
        "succeeded": succeeded,
        "failed": failed,
        "failure_counts": failure_counts,
    }

    summary_path = output_dir / "extraction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    logger.info(
        "Done. %d succeeded, %d failed. Summary: %s",
        succeeded,
        failed,
        summary_path,
    )
    if failure_counts:
        logger.info(
            "Failures: %s",
            ", ".join(f"{k}={v}" for k, v in sorted(failure_counts.items())),
        )

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ORCA job directories to extended XYZ for MLIP training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output per job:
  output_dir/{job_id}.xyz    -- extxyz frame (energy eV, forces eV/Ang)
  output_dir/{job_id}.failed -- reason string if parsing failed
  output_dir/extraction_summary.json

Feed to fairchem create_finetune_dataset.py:
  python create_finetune_dataset.py \\
      --train-dir output_dir/train \\
      --val-dir   output_dir/val \\
      --output-dir lmdb/
  (split train/val manually by moving files before running fairchem)
        """,
    )
    parser.add_argument("root_dir", help="Root directory containing job_* subdirectories.")
    parser.add_argument(
        "--output-dir",
        "-o",
        default="orca_extxyz",
        help="Output directory for extxyz files (default: orca_extxyz).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel I/O workers (default: 4).",
    )
    parser.add_argument(
        "--job-glob",
        default="job_*",
        help="Glob pattern for job directories (default: job_*).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    level = logging.DEBUG if args.debug else logging.INFO
    logger = _setup_logger("convert_to_xyzs", level=level)

    summary = extract_dataset(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        workers=args.workers,
        job_glob=args.job_glob,
        logger=logger,
    )

    if summary["succeeded"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
