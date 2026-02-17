#!/usr/bin/env python3
"""Create a workflow database for AN66 distance-spin sweep DFT jobs.

Reads AN66 molecule geometries from orca.inp files, generates single-ligand
dissociation sweep geometries using Architector, computes spin states
(GS, GS+2, maximal single-ligand dissociation), and creates a SQLite DB
compatible with ArchitectorWorkflow.

Usage:
    python create_an66_dissociation_db.py --an66-dir ../../data/an66_new --output-db an66_dissociation.db
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms

from oact_utilities.utils.architector import _init_db, _insert_row, parse_xyz_elements
from oact_utilities.utils.create import fetch_actinides, read_geom_from_inp_file

# ---------------------------------------------------------------------------
# Lookup tables for unpaired electrons (atomic ground states)
# ---------------------------------------------------------------------------

ACTINIDE_UNPAIRED: dict[str, int] = {
    "Ac": 1,
    "Th": 2,
    "Pa": 3,
    "U": 4,
    "Np": 5,
    "Pu": 6,
    "Am": 7,
    "Cm": 8,
    "Bk": 9,
    "Cf": 10,
    "Es": 11,
    "Fm": 12,
    "Md": 13,
    "No": 14,
    "Lr": 1,
}

LIGAND_UNPAIRED: dict[str, int] = {
    "H": 1,
    "F": 1,
    "Cl": 1,
    "Br": 1,
    "I": 1,
    "O": 2,
    "S": 2,
    "N": 3,
    "C": 2,
    "Si": 2,
    "P": 3,
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def atoms_to_xyz_string(atoms: Atoms, comment: str = "generated") -> str:
    """Convert ASE Atoms to standard XYZ format string."""
    n = len(atoms)
    lines = [str(n), comment]
    for symbol, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
        lines.append(f"{symbol}  {pos[0]:.10f}  {pos[1]:.10f}  {pos[2]:.10f}")
    return "\n".join(lines)


def compute_rmsd(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Compute RMSD between two position arrays (same ordering assumed)."""
    diff = pos1 - pos2
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))


def parse_xyz_positions(xyz_str: str) -> np.ndarray:
    """Parse positions from a standard XYZ string (natoms, comment, coords)."""
    lines = [ln for ln in xyz_str.strip().splitlines() if ln.strip()]
    positions: list[list[float]] = []
    # Skip first two lines (atom count + comment)
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 4:
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(positions)


def deduplicate_xyz_strings(
    xyz_strings: list[str],
    rmsd_threshold: float = 0.02,
) -> list[str]:
    """Remove duplicate geometries based on RMSD threshold.

    Keeps the first occurrence of each unique structure.
    Expects standard XYZ format strings.
    """
    if not xyz_strings:
        return []

    # Parse positions from each XYZ string
    positions_list: list[np.ndarray] = []
    for xyz_str in xyz_strings:
        positions_list.append(parse_xyz_positions(xyz_str))

    unique_indices: list[int] = [0]
    for i in range(1, len(xyz_strings)):
        is_duplicate = False
        for j in unique_indices:
            if positions_list[i].shape == positions_list[j].shape:
                rmsd = compute_rmsd(positions_list[i], positions_list[j])
                if rmsd < rmsd_threshold:
                    is_duplicate = True
                    break
        if not is_duplicate:
            unique_indices.append(i)

    return [xyz_strings[i] for i in unique_indices]


def _pick_first_ligand_sweep(result: list) -> list[str]:
    """Pick only the first ligand's sweep from lig_dissociation_sample(do_all=False).

    With do_all=False, the return is a list of lists — one inner list per
    ligand bond, each containing mol2 strings for each distance step.
    We take only the first ligand bond (one representative dissociation path).
    """
    if not result:
        return []
    first = result[0]
    if isinstance(first, list):
        return first
    # If result is already flat (single ligand molecule), return as-is
    return result


def mol2_to_xyz(mol2_str: str) -> str:
    """Convert a MOL2 format string to XYZ format via Architector."""
    from architector import convert_io_molecule

    mol = convert_io_molecule(mol2_str)
    atoms = mol.ase_atoms
    return atoms_to_xyz_string(atoms)


def generate_dissociation_geometries(
    mol: object,
    rmsd_threshold: float = 0.02,
) -> list[str]:
    """Generate deduplicated single-ligand dissociation sweep geometries.

    Uses do_all=False so each ligand is dissociated independently.
    Non-uniform spacing:
    - Fine:   -0.2 to 2.8 Angstrom, 0.3 Angstrom spacing (11 steps)
    - Coarse:  3.3 to 4.8 Angstrom, 0.5 Angstrom spacing (4 steps)

    Args:
        mol: Architector Molecule object.
        rmsd_threshold: RMSD threshold for deduplication.

    Returns:
        List of unique XYZ strings. Empty list if generation fails.
    """
    all_mol2: list[str] = []

    # Fine range: -0.2 to 2.8, 0.3 spacing → 11 points
    try:
        fine_result = mol.lig_dissociation_sample(
            min_shift=-0.2,
            max_dist=2.8,
            steps=11,
            do_all=False,
        )
        all_mol2.extend(_pick_first_ligand_sweep(fine_result))
    except Exception as e:
        print(f"    Warning: fine range dissociation failed: {e}")

    # Coarse range: 3.3 to 4.8, 0.5 spacing → 4 points
    try:
        coarse_result = mol.lig_dissociation_sample(
            min_shift=3.3,
            max_dist=4.8,
            steps=4,
            do_all=False,
        )
        all_mol2.extend(_pick_first_ligand_sweep(coarse_result))
    except Exception as e:
        print(f"    Warning: coarse range dissociation failed: {e}")

    if not all_mol2:
        return []

    # Convert all MOL2 strings to XYZ format
    all_xyz: list[str] = []
    for mol2_str in all_mol2:
        try:
            xyz = mol2_to_xyz(mol2_str)
            all_xyz.append(xyz)
        except Exception as e:
            print(f"    Warning: failed to convert mol2 to xyz: {e}")

    if not all_xyz:
        return []

    # Deduplicate
    unique = deduplicate_xyz_strings(all_xyz, rmsd_threshold=rmsd_threshold)
    return unique


def identify_actinide(elements: list[str], actinides_list: list[str]) -> str | None:
    """Find the actinide element in a list of element symbols."""
    for elem in elements:
        if elem in actinides_list:
            return elem
    return None


# Priority order for choosing which ligand to dissociate.
# Halogens first (F preferred), then chalcogens, then others.
LIGAND_PRIORITY: list[str] = ["F", "Cl", "Br", "I", "O", "S", "N", "C", "Si", "P", "H"]


def choose_dissociating_ligand(
    elements: list[str],
    actinides_list: list[str],
) -> str | None:
    """Choose the single ligand to dissociate from a molecule.

    Uses priority order: F > Cl > Br > I > O > S > N > ...
    For mixed-ligand molecules, always picks the highest-priority ligand.

    Args:
        elements: List of element symbols in the molecule.
        actinides_list: List of actinide element symbols.

    Returns:
        Element symbol of the chosen ligand, or None if no ligands found.
    """
    ligand_set = {elem for elem in elements if elem not in actinides_list}
    for preferred in LIGAND_PRIORITY:
        if preferred in ligand_set:
            return preferred
    # Fallback: return first non-actinide element found
    for elem in elements:
        if elem not in actinides_list:
            return elem
    return None


def compute_spin_states(
    gs_mult: int,
    actinide: str,
    dissociating_ligand: str,
) -> list[int]:
    """Compute unique spin states for a single-ligand dissociation.

    Returns sorted list of unique spin multiplicities:
    1. GS multiplicity (from orca.inp)
    2. GS + 2 (one spin state above, if <= max)
    3. Maximal single-ligand dissociation: metal_unpaired + ligand_unpaired + 1

    Args:
        gs_mult: Ground state spin multiplicity from orca.inp.
        actinide: Actinide element symbol.
        dissociating_ligand: Element symbol of the dissociating ligand.

    Returns:
        Sorted list of unique spin multiplicities.
    """
    metal_unpaired = ACTINIDE_UNPAIRED.get(actinide, 0)
    ligand_unpaired = LIGAND_UNPAIRED.get(dissociating_ligand, 0)
    max_dissoc = metal_unpaired + ligand_unpaired + 1

    # Sanity check
    if max_dissoc < gs_mult:
        print(
            f"    WARNING: max_dissoc ({max_dissoc}) < gs_mult ({gs_mult}) "
            f"for {actinide} + {dissociating_ligand}. "
            f"Using gs_mult as max instead."
        )
        max_dissoc = gs_mult

    # Collect spin states
    spins: set[int] = {gs_mult, max_dissoc}

    # GS + 2 (one state above), only if it doesn't exceed max
    gs_plus_2 = gs_mult + 2
    if gs_plus_2 <= max_dissoc:
        spins.add(gs_plus_2)

    return sorted(spins)


# ---------------------------------------------------------------------------
# Molecule discovery and processing
# ---------------------------------------------------------------------------


def discover_an66_molecules(an66_dir: Path) -> list[tuple[str, Path]]:
    """Find all AN66 molecule directories containing orca.inp files.

    Args:
        an66_dir: Path to the AN66 data directory.

    Returns:
        Sorted list of (molecule_name, orca_inp_path) tuples.
    """
    molecules: list[tuple[str, Path]] = []
    for subdir in sorted(an66_dir.iterdir()):
        if not subdir.is_dir():
            continue
        inp_file = subdir / "orca.inp"
        if inp_file.exists():
            molecules.append((subdir.name, inp_file))
    return molecules


def process_molecule(
    name: str,
    inp_path: Path,
    actinides_list: list[str],
    rmsd_threshold: float = 0.02,
) -> list[dict]:
    """Process a single AN66 molecule.

    Reads geometry, generates single-ligand dissociation sweep, computes
    spin states based on the chosen ligand, and returns DB row dicts.

    One ligand is chosen per molecule using priority order (F > Cl > Br > ...).

    Args:
        name: Molecule name (folder name).
        inp_path: Path to orca.inp file.
        actinides_list: List of actinide element symbols.
        rmsd_threshold: RMSD threshold for geometry deduplication.

    Returns:
        List of dicts, each representing one DB row with keys:
        elements, natoms, charge, spin, geometry, molecule_name, ligand_type.
    """
    from architector import convert_io_molecule

    # Read geometry and GS spin from orca.inp
    atoms = read_geom_from_inp_file(str(inp_path), ase_format_tf=True)
    charge = int(atoms.charge)
    gs_mult = int(atoms.spin)
    elements = atoms.get_chemical_symbols()

    # Identify actinide
    actinide = identify_actinide(elements, actinides_list)
    if actinide is None:
        print(f"  WARNING: No actinide found in {name}, skipping")
        return []

    # Choose one ligand to dissociate (F > Cl > Br > I > O > ...)
    ligand_type = choose_dissociating_ligand(elements, actinides_list)
    if ligand_type is None:
        print(f"  WARNING: No ligand found in {name}, skipping")
        return []

    # Convert to XYZ and then to Architector molecule
    xyz_str = atoms_to_xyz_string(atoms)
    mol = convert_io_molecule(xyz_str)

    # Generate dissociation sweep geometries (single ligand at a time)
    geom_list = generate_dissociation_geometries(
        mol,
        rmsd_threshold=rmsd_threshold,
    )

    if not geom_list:
        print(f"  WARNING: No dissociation geometries generated for {name}")
        return []

    print(f"  Generated {len(geom_list)} unique geometries")

    # Compute spin states for the chosen dissociating ligand
    spins = compute_spin_states(gs_mult, actinide, ligand_type)
    spin_str = ", ".join(str(s) for s in spins)
    print(
        f"  Dissociating {ligand_type}: spins = [{spin_str}] "
        f"(metal={actinide}, gs={gs_mult})"
    )

    # Cross-product: geometry x spin
    rows: list[dict] = []
    for geom_xyz in geom_list:
        geom_elements = parse_xyz_elements(geom_xyz)
        for spin_mult in spins:
            rows.append(
                {
                    "elements": ";".join(geom_elements),
                    "natoms": len(geom_elements),
                    "charge": charge,
                    "spin": spin_mult,
                    "geometry": geom_xyz,
                    "molecule_name": name,
                    "ligand_type": ligand_type,
                }
            )

    return rows


# ---------------------------------------------------------------------------
# Database creation
# ---------------------------------------------------------------------------


def create_dissociation_db(
    an66_dir: Path,
    output_db: Path,
    rmsd_threshold: float = 0.02,
) -> Path:
    """Create the full workflow database for AN66 dissociation sweeps.

    Args:
        an66_dir: Path to AN66 molecule directories.
        output_db: Path for the output SQLite database.
        rmsd_threshold: RMSD threshold for geometry deduplication.

    Returns:
        Path to the created database.
    """
    actinides_list = fetch_actinides()

    # Discover molecules
    molecules = discover_an66_molecules(an66_dir)
    print(f"Found {len(molecules)} AN66 molecules in {an66_dir}")

    if not molecules:
        print("ERROR: No molecules found. Check --an66-dir path.")
        sys.exit(1)

    # Initialize DB
    conn = _init_db(output_db)

    # Track mapping and stats
    mapping: dict[str, dict] = {}
    total_rows = 0
    failed_molecules: list[str] = []
    row_counter = 0

    for i, (name, inp_path) in enumerate(molecules, 1):
        print(f"\n[{i}/{len(molecules)}] Processing {name}...")

        try:
            rows = process_molecule(
                name, inp_path, actinides_list, rmsd_threshold=rmsd_threshold
            )
        except Exception as e:
            print(f"  ERROR: Failed to process {name}: {e}")
            failed_molecules.append(name)
            continue

        if not rows:
            failed_molecules.append(name)
            continue

        # Insert rows into DB
        mol_row_ids: list[int] = []
        for row in rows:
            _insert_row(
                conn,
                orig_index=row_counter,
                elements=row["elements"],
                natoms=row["natoms"],
                geometry=row["geometry"],
                status="ready",
                charge=row["charge"],
                spin=row["spin"],
            )
            mol_row_ids.append(row_counter)
            row_counter += 1

        # Save mapping info
        ligand_types = sorted(set(r["ligand_type"] for r in rows))
        spin_states = sorted(set(r["spin"] for r in rows))
        n_geoms = len(set(r["geometry"] for r in rows))
        mapping[name] = {
            "row_ids": mol_row_ids,
            "n_rows": len(rows),
            "n_geometries": n_geoms,
            "ligand_types": ligand_types,
            "spin_states": spin_states,
            "charge": rows[0]["charge"],
            "natoms": rows[0]["natoms"],
        }

        total_rows += len(rows)
        print(
            f"  Inserted {len(rows)} rows "
            f"({n_geoms} geoms x {len(spin_states)} spins, ligand={ligand_types[0]})"
        )

    # Commit
    conn.commit()
    conn.close()

    # Save mapping JSON
    mapping_path = output_db.with_suffix(".mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Database:           {output_db}")
    print(f"Mapping:            {mapping_path}")
    print(f"Total molecules:    {len(molecules)}")
    print(f"Successful:         {len(molecules) - len(failed_molecules)}")
    print(f"Failed:             {len(failed_molecules)}")
    if failed_molecules:
        print(f"  Failed list:      {', '.join(failed_molecules)}")
    print(f"Total DB rows:      {total_rows}")
    print(f"RMSD threshold:     {rmsd_threshold} Angstrom")

    # Per-actinide breakdown
    print("\nPer-actinide breakdown:")
    actinide_stats: dict[str, dict] = {}
    for mol_name, info in mapping.items():
        for act in sorted(actinides_list, key=len, reverse=True):
            if mol_name.startswith(act):
                if act not in actinide_stats:
                    actinide_stats[act] = {"molecules": 0, "rows": 0}
                actinide_stats[act]["molecules"] += 1
                actinide_stats[act]["rows"] += info["n_rows"]
                break

    for act in sorted(actinide_stats.keys()):
        stats = actinide_stats[act]
        print(f"  {act}: {stats['molecules']} molecules, {stats['rows']} jobs")

    return output_db


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Create AN66 dissociation sweep workflow database",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--an66-dir",
        type=Path,
        default=Path("../../data/an66_new"),
        help="Path to AN66 molecule directories",
    )
    parser.add_argument(
        "--output-db",
        type=Path,
        default=Path("an66_dissociation.db"),
        help="Output SQLite database path",
    )
    parser.add_argument(
        "--rmsd-threshold",
        type=float,
        default=0.02,
        help="RMSD threshold in Angstroms for geometry deduplication",
    )
    args = parser.parse_args()

    # Resolve paths
    an66_dir = args.an66_dir.resolve()
    output_db = args.output_db.resolve()

    if not an66_dir.exists():
        print(f"ERROR: AN66 directory not found: {an66_dir}")
        sys.exit(1)

    print(f"AN66 directory: {an66_dir}")
    print(f"Output DB:      {output_db}")
    print(f"RMSD threshold: {args.rmsd_threshold} Angstrom")
    print()

    create_dissociation_db(
        an66_dir=an66_dir,
        output_db=output_db,
        rmsd_threshold=args.rmsd_threshold,
    )


if __name__ == "__main__":
    main()
