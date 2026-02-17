"""Create workflow database from wave2 directory structure.

This script migrates the wave2 directory structure (used in run_jobs_quacc_wave2.py)
to the new database-based workflow system. It reads geometry and multiplicity files
from structured directories and creates a SQLite database compatible with
ArchitectorWorkflow.

Directory structure:
    root_data_dir/
        Hard_Donors/
            subfolder1/
                data_geom.txt
                data_charge_mult.txt
            subfolder2/
                ...
        Organic/
            ...
        Radical/
            ...
        Soft_Donors/
            ...

Usage:
    python create_wave2_workflow_db.py \
        --root-data-dir /path/to/wave2/data \
        --db-path workflow.db \
        --calc-root-dir /path/to/output

After creating the database, submit jobs using:
    python -m oact_utilities.workflows.submit_jobs workflow.db jobs/ \
        --batch-size 100 \
        --scheduler flux \
        --functional wB97M-V \
        --simple-input omol
"""

import argparse
import sqlite3
from pathlib import Path

from ase import Atoms

from oact_utilities.utils.architector import _init_db, _insert_row
from oact_utilities.utils.baselines import (
    process_geometry_file,
    process_multiplicity_file,
)


def atoms_to_xyz_string(atoms: Atoms) -> str:
    """Convert ASE Atoms object to XYZ string format.

    Args:
        atoms: ASE Atoms object.

    Returns:
        XYZ-format string compatible with ORCA input.
    """
    lines = []
    for symbol, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
        lines.append(f"{symbol:2s} {pos[0]:16.10f} {pos[1]:16.10f} {pos[2]:16.10f}")
    return "\n".join(lines)


def create_wave2_workflow_db(
    root_data_dir: str | Path,
    db_path: str | Path,
    calc_root_dir: str | Path | None = None,
    status: str = "to_run",
    debug: bool = False,
) -> Path:
    """Create workflow database from wave2 directory structure.

    This function reads the wave2 directory structure and creates a SQLite
    database compatible with ArchitectorWorkflow. Jobs are organized using
    the 'category' and 'ligand_type' columns, eliminating the need to
    replicate the input directory structure in the output.

    Args:
        root_data_dir: Root directory containing wave2 data subfolders.
        db_path: Path for the SQLite database to create.
        calc_root_dir: Optional root directory for calculation outputs.
            If not provided, job_dir will be set based on the workflow
            submission location. The category and ligand_type columns provide
            all necessary organization.
        status: Initial status for all jobs (default: "to_run").
        debug: If True, print detailed progress information.

    Returns:
        Path to the created database.
    """
    root_data_dir = Path(root_data_dir)
    db_path = Path(db_path)
    if calc_root_dir:
        calc_root_dir = Path(calc_root_dir)

    if not root_data_dir.exists():
        raise FileNotFoundError(f"Root data directory not found: {root_data_dir}")

    # Initialize database with standard schema
    conn = _init_db(db_path)

    # Add wave2-specific columns for category and ligand type
    cur = conn.cursor()
    try:
        cur.execute("ALTER TABLE structures ADD COLUMN category TEXT")
        cur.execute("ALTER TABLE structures ADD COLUMN ligand_type TEXT")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_category ON structures(category)")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_ligand_type ON structures(ligand_type)"
        )
        conn.commit()
    except sqlite3.OperationalError:
        # Columns may already exist if database is being updated
        pass

    # Wave2 subdirectories
    base_dirs = ["Hard_Donors", "Organic", "Radical", "Soft_Donors"]

    total_inserted = 0
    job_index = 0  # Global job index for orig_index

    if debug:
        print(f"Creating workflow database at: {db_path}")
        print(f"Reading wave2 data from: {root_data_dir}")
        print(f"Initial status: {status}")
        if calc_root_dir:
            print(f"Calculation root directory: {calc_root_dir}")
        else:
            print(
                "Note: calc_root_dir not set - job_dir will be assigned during submission"
            )
            print("      Use 'category' and 'ligand_type' columns for job organization")

    try:
        for base_dir in base_dirs:
            base_path = root_data_dir / base_dir

            if not base_path.exists():
                if debug:
                    print(f"  Skipping missing directory: {base_dir}")
                continue

            if debug:
                print(f"\nProcessing {base_dir}/")

            # Get all subfolders
            subfolders = [f for f in base_path.iterdir() if f.is_dir()]

            for folder in sorted(subfolders):
                folder_name = folder.name
                geom_file = folder / "data_geom.txt"
                mult_file = folder / "data_charge_mult.txt"

                # Skip if required files don't exist
                if not geom_file.exists() or not mult_file.exists():
                    if debug:
                        print(
                            f"  Skipping {base_dir}/{folder_name} (missing data files)"
                        )
                    continue

                if debug:
                    print(f"  Processing {base_dir}/{folder_name}/")

                # Parse geometry and multiplicity files
                try:
                    dict_geoms = process_geometry_file(
                        str(geom_file), ase_format_tf=True
                    )
                    dict_multiplicity = process_multiplicity_file(str(mult_file))
                except Exception as e:
                    print(f"  ERROR parsing files in {base_dir}/{folder_name}: {e}")
                    continue

                # Create unified dictionary combining geometries with each spin state
                for molecule_key, atoms in dict_geoms.items():
                    if molecule_key not in dict_multiplicity:
                        if debug:
                            print(f"    Skipping {molecule_key} (no multiplicity data)")
                        continue

                    # Process each spin state for this molecule
                    for spin_entry in dict_multiplicity[molecule_key]:
                        mult = spin_entry["multiplicity"]
                        charge = spin_entry["charge"]

                        # Create job identifier
                        job_key = f"{molecule_key}_{mult}"

                        # Convert ASE Atoms to XYZ string
                        xyz_str = atoms_to_xyz_string(atoms)

                        # Parse elements
                        elements = ";".join(atoms.get_chemical_symbols())
                        natoms = len(atoms)

                        # Construct job directory path if calc_root_dir provided
                        job_dir = None
                        if calc_root_dir:
                            job_dir = str(
                                calc_root_dir / base_dir / folder_name / job_key
                            )

                        # Insert into database using standard _insert_row
                        _insert_row(
                            conn,
                            orig_index=job_index,
                            elements=elements,
                            natoms=natoms,
                            geometry=xyz_str,
                            status=status,
                            charge=charge,
                            spin=mult,
                            job_dir=job_dir,
                        )

                        # Update with wave2-specific metadata (category and ligand_type)
                        cur = conn.cursor()
                        cur.execute(
                            "UPDATE structures SET category = ?, ligand_type = ? WHERE orig_index = ?",
                            (base_dir, folder_name, job_index),
                        )

                        total_inserted += 1
                        job_index += 1

                        if debug and total_inserted % 100 == 0:
                            print(f"    Inserted {total_inserted} jobs...")

                # Commit after each folder
                conn.commit()

        # Final commit
        conn.commit()

        if debug:
            print("\n" + "=" * 70)
            # Print summary statistics
            cur = conn.cursor()

            # Count by category
            print("\nJobs by category:")
            for base_dir in base_dirs:
                cur.execute(
                    "SELECT COUNT(*) FROM structures WHERE category = ?", (base_dir,)
                )
                count = cur.fetchone()[0]
                if count > 0:
                    print(f"  {base_dir:20s}: {count:6d} jobs")

            # Count unique ligand types per category
            print("\nUnique ligand types by category:")
            for base_dir in base_dirs:
                cur.execute(
                    "SELECT COUNT(DISTINCT ligand_type) FROM structures WHERE category = ?",
                    (base_dir,),
                )
                count = cur.fetchone()[0]
                if count > 0:
                    print(f"  {base_dir:20s}: {count:6d} unique ligand types")

            # Print first job as example
            cur.execute(
                "SELECT id, orig_index, elements, natoms, charge, spin, category, ligand_type, geometry, job_dir "
                "FROM structures ORDER BY id LIMIT 1"
            )
            row = cur.fetchone()
            if row:
                (
                    idx,
                    orig_index,
                    elems,
                    natoms,
                    charge,
                    spin,
                    category,
                    ligand_type,
                    xyz_str,
                    job_dir,
                ) = row
                print("\nExample job (first entry):")
                print(f"  ID: {idx}")
                print(f"  orig_index: {orig_index}")
                print(f"  natoms: {natoms}")
                print(f"  charge: {charge}")
                print(f"  spin: {spin}")
                print(f"  category: {category}")
                print(f"  ligand_type: {ligand_type}")
                print(f"  elements: {elems}")
                print(f"  job_dir: {job_dir}")
                print("  geometry (first 3 lines):")
                for line in xyz_str.split("\n")[:3]:
                    print(f"    {line}")

    finally:
        conn.close()

    print(f"\n{'='*70}")
    print(f"Created workflow database with {total_inserted} jobs at: {db_path}")
    print(f"{'='*70}")

    return db_path


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Create workflow database from wave2 directory structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python create_wave2_workflow_db.py \\
      --root-data-dir /path/to/wave2/data \\
      --db-path workflow.db

  # With output directory structure
  python create_wave2_workflow_db.py \\
      --root-data-dir /path/to/wave2/data \\
      --db-path workflow.db \\
      --calc-root-dir /path/to/output

  # Debug mode
  python create_wave2_workflow_db.py \\
      --root-data-dir /path/to/wave2/data \\
      --db-path workflow.db \\
      --debug

After creating the database, submit jobs using:
  python -m oact_utilities.workflows.submit_jobs workflow.db jobs/ \\
      --batch-size 100 \\
      --scheduler flux \\
      --functional wB97M-V \\
      --simple-input omol
        """,
    )

    parser.add_argument(
        "--root-data-dir",
        type=str,
        required=True,
        help="Root directory containing wave2 data (Hard_Donors, Organic, etc.)",
    )

    parser.add_argument(
        "--db-path",
        type=str,
        required=True,
        help="Path for the SQLite database to create",
    )

    parser.add_argument(
        "--calc-root-dir",
        type=str,
        default=None,
        help="Root directory for calculation outputs (optional). "
        "If provided, job_dir will be set for each job.",
    )

    parser.add_argument(
        "--status",
        type=str,
        default="to_run",
        choices=["to_run", "ready"],
        help="Initial status for all jobs (default: to_run)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed progress information",
    )

    args = parser.parse_args()

    create_wave2_workflow_db(
        root_data_dir=args.root_data_dir,
        db_path=args.db_path,
        calc_root_dir=args.calc_root_dir,
        status=args.status,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
