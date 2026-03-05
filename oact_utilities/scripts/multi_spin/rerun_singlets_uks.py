#!/usr/bin/env python3
"""
Re-run singlet (spin_1) calculations with UKS for actinide multi-spin campaign.

The original spin_1 calculations were run with RKS (ORCA default for singlets),
which is incorrect for actinide systems that may have open-shell singlet character.
This script:

1. Reads spin_1 geometries from the multi-spin SQLite database
2. Renames existing spin_1 folders to spin_1_rks (preserving old RKS data)
3. Regenerates ORCA inputs in fresh spin_1 folders using the current codebase,
   which auto-adds UKS + symmetry breaking for actinide singlets

Usage:
    python rerun_singlets_uks.py <db_path> --orca-exe /path/to/orca

    # Preview without writing:
    python rerun_singlets_uks.py <db_path> --dry-run

    # Also regenerate failed jobs:
    python rerun_singlets_uks.py <db_path> --orca-exe /path/to/orca --include-failed
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import chemical_symbols

from oact_utilities.scripts.multi_spin.multi_spin_from_converged_calcs import (
    wrapper_write_job_folder,
)

# LOT-specific configuration matching the original setup
# from multi_spin_from_converged_calcs.py:main()
LOT_CONFIG: dict[str, dict[str, Any]] = {
    "omol": {
        "functional": "wB97M-V",
        "simple_input": "omol",
        "actinide_basis": "ma-def-TZVP",
        "actinide_ecp": "def-ECP",
        "non_actinide_basis": "def2-TZVPD",
    },
    "x2c": {
        "functional": "PBE0",
        "simple_input": "x2c",
        "actinide_basis": "/usr/workspace/vargas58/orca_test/basis_sets/cc_pvtz_x2c.bas",
        "actinide_ecp": None,
        "non_actinide_basis": "X2C-TZVPPall",
    },
}


def extract_charge_from_orca_inp(job_dir: str) -> int | None:
    """Parse charge from ORCA input file's ``* xyz <charge> <mult>`` line.

    Args:
        job_dir: Path to job directory containing ``orca.inp``.

    Returns:
        Parsed charge as int, or None if not found.
    """
    inp_file = os.path.join(job_dir, "orca.inp")
    if not os.path.exists(inp_file):
        return None
    with open(inp_file, errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("* xyz") or stripped.startswith("*xyz"):
                parts = stripped.split()
                # Format: * xyz <charge> <mult>
                if len(parts) >= 4:
                    try:
                        return int(parts[2])
                    except ValueError:
                        return None
    return None


def atoms_from_db_row(
    final_coords: str | None,
    final_elements: str | None,
) -> Atoms | None:
    """Reconstruct ASE Atoms from DB final_coords and final_elements JSON.

    Args:
        final_coords: JSON string of coordinates (list of [x, y, z]).
        final_elements: JSON string of elements (symbols or atomic numbers).

    Returns:
        ASE Atoms object, or None if data is missing/invalid.
    """
    if final_coords is None or final_elements is None:
        return None

    try:
        coords = json.loads(final_coords)
        elements = json.loads(final_elements)
    except (json.JSONDecodeError, TypeError):
        return None

    if not elements or not coords:
        return None

    # Handle atomic numbers vs symbols
    if isinstance(elements[0], str) and elements[0].isdigit():
        elements = [chemical_symbols[int(e)] for e in elements]
    elif isinstance(elements[0], int):
        elements = [chemical_symbols[e] for e in elements]

    return Atoms(symbols=elements, positions=np.array(coords))


def atoms_from_orca_xyz(job_dir: str) -> Atoms | None:
    """Read atoms from orca.xyz in a job directory (fallback for failed jobs).

    Args:
        job_dir: Path to job directory.

    Returns:
        ASE Atoms object, or None if file not found.
    """
    xyz_file = os.path.join(job_dir, "orca.xyz")
    if not os.path.exists(xyz_file):
        return None

    try:
        from oact_utilities.utils.create import read_xyz_from_orca

        atoms, _ = read_xyz_from_orca(xyz_file)
        return atoms
    except Exception:
        return None


def is_sella_run(job_dir: str) -> bool:
    """Check if a job directory contains a Sella optimization run.

    Args:
        job_dir: Path to job directory.

    Returns:
        True if sella.log or opt.traj exists.
    """
    return os.path.exists(os.path.join(job_dir, "sella.log")) or os.path.exists(
        os.path.join(job_dir, "opt.traj")
    )


def rerun_singlets(
    db_path: str,
    orca_exe: str = "orca",
    cores: int = 24,
    n_hours: int = 24,
    queue: str = "pbatch",
    allocation: str = "dnn-sim",
    max_scf_iterations: int = 600,
    dry_run: bool = False,
    include_failed: bool = False,
    verbose: bool = False,
) -> dict[str, int]:
    """Re-generate spin_1 ORCA inputs with UKS for actinide systems.

    Args:
        db_path: Path to multi-spin SQLite database.
        orca_exe: Path to ORCA executable.
        cores: Number of cores per job.
        n_hours: Walltime hours per job.
        queue: HPC queue name.
        allocation: HPC allocation name.
        max_scf_iterations: Maximum SCF iterations.
        dry_run: If True, only print what would be done.
        include_failed: If True, also regenerate failed spin_1 jobs.
        verbose: Print detailed output.

    Returns:
        Summary dict with counts of regenerated, skipped, and failed jobs.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Query spin_1 jobs
    if include_failed:
        query = "SELECT * FROM jobs WHERE spin = 'spin_1' AND status IN (1, -1)"
    else:
        query = "SELECT * FROM jobs WHERE spin = 'spin_1' AND status = 1"

    rows = conn.execute(query).fetchall()
    conn.close()

    summary = {
        "regenerated": 0,
        "skipped_no_geometry": 0,
        "skipped_no_charge": 0,
        "skipped_unknown_lot": 0,
        "skipped_already_renamed": 0,
        "failed": 0,
        "total": len(rows),
    }

    print(f"Found {len(rows)} spin_1 jobs to process")
    if dry_run:
        print("DRY RUN — no files will be modified\n")

    for row in rows:
        path = row["path"]
        lot = row["lot"]
        name = row["name"]
        category = row["category"]
        status = row["status"]

        status_str = "completed" if status == 1 else "failed"
        print(f"\n--- {name} ({category}, {lot}, {status_str}) ---")
        if verbose:
            print(f"  Path: {path}")

        # Validate LOT
        if lot not in LOT_CONFIG:
            print(f"  SKIP: Unknown LOT '{lot}'")
            summary["skipped_unknown_lot"] += 1
            continue

        config = LOT_CONFIG[lot]

        # Determine the source directory for charge extraction
        rks_dir = path.rstrip("/") + "_rks"  # e.g., .../spin_1_rks

        # Figure out where to read charge from
        if os.path.exists(rks_dir):
            # Already renamed from a previous run
            charge_source_dir = rks_dir
            print("  spin_1_rks already exists, reading charge from there")
        elif os.path.exists(path):
            charge_source_dir = path
        else:
            print(f"  SKIP: Neither {path} nor {rks_dir} exists on filesystem")
            summary["skipped_no_geometry"] += 1
            continue

        # Extract charge
        charge = extract_charge_from_orca_inp(charge_source_dir)
        if charge is None:
            print("  SKIP: Could not extract charge from orca.inp")
            summary["skipped_no_charge"] += 1
            continue

        print(f"  Charge: {charge}")

        # Get geometry: prefer DB, fall back to orca.xyz
        atoms = atoms_from_db_row(row["final_coords"], row["final_elements"])
        if atoms is None:
            # Try reading from the filesystem (useful for failed jobs)
            atoms = atoms_from_orca_xyz(charge_source_dir)
        if atoms is None:
            print("  SKIP: No geometry available (DB or filesystem)")
            summary["skipped_no_geometry"] += 1
            continue

        print(f"  Atoms: {len(atoms)} atoms, formula: {atoms.get_chemical_formula()}")

        # Detect if original was a Sella run
        tf_sella = is_sella_run(charge_source_dir)
        print(f"  Run type: {'Sella' if tf_sella else 'ORCA direct'}")

        if dry_run:
            print(f"  DRY-RUN: Would rename {path} -> {rks_dir}")
            print(f"  DRY-RUN: Would write UKS inputs to {path}")
            summary["regenerated"] += 1
            continue

        # Rename spin_1 -> spin_1_rks (if not already done)
        if os.path.exists(path) and not os.path.exists(rks_dir):
            try:
                shutil.move(path, rks_dir)
                print("  Renamed: spin_1 -> spin_1_rks")
            except OSError as e:
                print(f"  FAILED: Could not rename: {e}")
                summary["failed"] += 1
                continue

        # Create fresh spin_1 directory and write UKS inputs
        try:
            wrapper_write_job_folder(
                output_folder=path,
                atoms=atoms,
                tf_sella=tf_sella,
                n_cores=cores,
                n_hours=n_hours,
                queue=queue,
                allocation=allocation,
                charge=charge,
                mult=1,
                functional=config["functional"],
                max_scf_iterations=max_scf_iterations,
                lot=config["simple_input"],
                orca_exe=orca_exe,
                actinide_basis=config["actinide_basis"],
                actinide_ecp=config["actinide_ecp"],
                non_actinide_basis=config["non_actinide_basis"],
                error_code=0,
                skip_done=False,
                skip_running=False,
            )
            print(f"  OK: UKS inputs written to {path}")
            summary["regenerated"] += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            summary["failed"] += 1

    return summary


def print_summary(summary: dict[str, int], dry_run: bool = False) -> None:
    """Print a summary of the re-run operation.

    Args:
        summary: Dict with counts of regenerated, skipped, and failed jobs.
        dry_run: Whether this was a dry run.
    """
    action = "Would regenerate" if dry_run else "Regenerated"
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total spin_1 jobs:           {summary['total']}")
    print(f"{action}:              {summary['regenerated']}")
    print(f"Skipped (no geometry):       {summary['skipped_no_geometry']}")
    print(f"Skipped (no charge):         {summary['skipped_no_charge']}")
    print(f"Skipped (unknown LOT):       {summary['skipped_unknown_lot']}")
    print(f"Failed:                      {summary['failed']}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-run singlet calculations with UKS for actinide multi-spin"
    )
    parser.add_argument(
        "db_path",
        help="Path to multi-spin SQLite database",
    )
    parser.add_argument(
        "--orca-exe",
        default="orca",
        help="Path to ORCA executable (default: 'orca' from PATH)",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=24,
        help="Number of cores per job (default: 24)",
    )
    parser.add_argument(
        "--n-hours",
        type=int,
        default=24,
        help="Walltime hours per job (default: 24)",
    )
    parser.add_argument(
        "--queue",
        default="pbatch",
        help="HPC queue name (default: pbatch)",
    )
    parser.add_argument(
        "--allocation",
        default="dnn-sim",
        help="HPC allocation name (default: dnn-sim)",
    )
    parser.add_argument(
        "--max-scf-iterations",
        type=int,
        default=600,
        help="Maximum SCF iterations (default: 600)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be done without modifying files",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Also regenerate failed spin_1 jobs",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.db_path):
        parser.error(f"Database not found: {args.db_path}")

    summary = rerun_singlets(
        db_path=args.db_path,
        orca_exe=args.orca_exe,
        cores=args.cores,
        n_hours=args.n_hours,
        queue=args.queue,
        allocation=args.allocation,
        max_scf_iterations=args.max_scf_iterations,
        dry_run=args.dry_run,
        include_failed=args.include_failed,
        verbose=args.verbose,
    )

    print_summary(summary, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
