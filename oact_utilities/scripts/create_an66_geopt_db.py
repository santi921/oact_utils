#!/usr/bin/env python3
"""Create a workflow database for AN66 geometry optimization jobs.

Reads AN66 molecule geometries from orca.inp files and creates a SQLite DB
compatible with ArchitectorWorkflow for running geometry optimizations
with either ORCA native opt (--opt) or Sella external optimizer.

Usage:
    python -m oact_utilities.scripts.create_an66_geopt_db \
        --an66-dir data/an66/an66_new \
        --output-db an66_geopt.db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from oact_utilities.utils.architector import _init_db, _insert_row

if TYPE_CHECKING:
    from ase import Atoms
from oact_utilities.utils.create import read_geom_from_inp_file


def discover_an66_molecules(an66_dir: Path) -> list[Path]:
    """Find all molecule directories containing orca.inp files.

    Args:
        an66_dir: Root directory containing AN66 molecule subdirectories.

    Returns:
        Sorted list of orca.inp file paths.
    """
    inp_files = sorted(an66_dir.glob("*/orca.inp"))
    if not inp_files:
        raise FileNotFoundError(
            f"No orca.inp files found in subdirectories of {an66_dir}"
        )
    return inp_files


def atoms_to_xyz_string(atoms: Atoms, comment: str = "an66") -> str:
    """Convert ASE Atoms to standard XYZ format string.

    Args:
        atoms: ASE Atoms object.
        comment: Comment line for XYZ format.

    Returns:
        XYZ format string.
    """
    n = len(atoms)
    lines = [str(n), comment]
    for symbol, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
        lines.append(f"{symbol}  {pos[0]:.10f}  {pos[1]:.10f}  {pos[2]:.10f}")
    return "\n".join(lines)


def create_an66_geopt_db(
    an66_dir: Path,
    output_db: Path,
) -> Path:
    """Create workflow DB from AN66 orca.inp files.

    Args:
        an66_dir: Directory containing AN66 molecule subdirectories.
        output_db: Path to output SQLite database.

    Returns:
        Path to created database.
    """
    inp_files = discover_an66_molecules(an66_dir)
    print(f"Found {len(inp_files)} AN66 molecules")

    extra_columns = {"molecule_name": "TEXT"}
    conn = _init_db(output_db, extra_columns=extra_columns)

    inserted = 0
    for idx, inp_path in enumerate(inp_files):
        mol_name = inp_path.parent.name
        atoms = read_geom_from_inp_file(str(inp_path), ase_format_tf=True)

        charge = int(atoms.charge)
        spin = int(atoms.spin)
        symbols = atoms.get_chemical_symbols()
        xyz_str = atoms_to_xyz_string(atoms, comment=mol_name)

        _insert_row(
            conn,
            orig_index=idx,
            elements=";".join(symbols),
            natoms=len(symbols),
            geometry=xyz_str,
            status="to_run",
            charge=charge,
            spin=spin,
            extra_values={"molecule_name": mol_name},
        )
        inserted += 1
        print(
            f"  [{idx + 1}/{len(inp_files)}] {mol_name}: "
            f"charge={charge}, spin={spin}, natoms={len(symbols)}"
        )

    conn.commit()
    conn.close()
    print(f"\nCreated workflow database with {inserted} structures at: {output_db}")
    return output_db


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Create AN66 geometry optimization workflow database"
    )
    parser.add_argument(
        "--an66-dir",
        type=Path,
        default=Path("data/an66/an66_new"),
        help="Directory containing AN66 molecule subdirectories (default: data/an66/an66_new)",
    )
    parser.add_argument(
        "--output-db",
        type=Path,
        default=Path("an66_geopt.db"),
        help="Output SQLite database path (default: an66_geopt.db)",
    )
    args = parser.parse_args()

    if not args.an66_dir.exists():
        print(f"Error: AN66 directory not found: {args.an66_dir}", file=sys.stderr)
        sys.exit(1)

    create_an66_geopt_db(args.an66_dir, args.output_db)


if __name__ == "__main__":
    main()
