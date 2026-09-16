"""Convert the genesis lanthanide pickle into a workflow SQLite DB.

The ``down_solv_sampled_ln_structs`` pickle does not match the architector CSV
schema that :func:`create_workflow` expects, so a few adaptations are applied
before building the database:

1. Input is a pickled ``pandas.DataFrame`` (not a CSV). It is loaded and passed
   directly to :func:`create_workflow_db`, which accepts a DataFrame.
2. Geometry is stored as MOL2 (``mol2string``). It is converted to XYZ into a
   new ``structure`` column (the DB stores/consumes XYZ).
3. There is no spin column. Spin multiplicity (2S+1) is derived from the MOL2
   header field ``Unpaired_Electrons: N`` (multiplicity = N + 1) into a new
   ``spinmult`` column.
4. Charge is the existing ``total_charge`` column.

The database is written under ``data/genesis/``.

No pre-run screening here, deliberately. The ``filter_risk`` classifier was trained on 19
metal centres -- the actinides Ac-Lr plus Po, At, Fr and Ra -- and has seen no lanthanide
centre at all, so this entire corpus is out of its domain. Its applicability gate would mark
every row ``screen_in_domain = 0`` and drop nothing, which is correct but useless. See
``oact_utilities/workflows/screening.py``.

Usage:
    python -m examples.convert_genesis_ln_to_workflow          # use defaults
    python examples/convert_genesis_ln_to_workflow.py --help   # override paths
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from oact_utilities.utils.architector import create_workflow_db
from oact_utilities.workflows import ArchitectorWorkflow, JobStatus

# Default paths for this campaign. Override on the CLI if needed.
DEFAULT_PICKLE = Path(
    "/Users/santiagovargas/dev/oact_utils/data/genesis/"
    "down_solv_sampled_ln_structs_7_10_26.pkl"
)
DEFAULT_DB = Path(
    "/Users/santiagovargas/dev/oact_utils/data/genesis/"
    "down_solv_sampled_ln_structs_7_10_26.db"
)

# Source columns in the pickle.
MOL2_COLUMN = "mol2string"
CHARGE_COLUMN = "total_charge"

# New columns this script adds and then points the DB builder at.
GEOMETRY_COLUMN = "structure"
SPIN_COLUMN = "spinmult"

# Metadata to carry into the DB. Keys must exist in the pickle. Numeric columns
# are cast to native Python ints in prepare_dataframe() so sqlite can bind them.
EXTRA_COLUMNS: dict[str, str] = {
    "metal": "TEXT",
    "metal_ox": "INTEGER",
    "solvent": "TEXT",
    "nsolv": "INTEGER",
    "label": "TEXT",
}

# Header field that is NOT the one we want (avoid matching XTB_Unpaired_Electrons).
_UNPAIRED_RE = re.compile(r"(?<!XTB_)Unpaired_Electrons:\s*(\d+)")


def mol2_to_xyz(mol2: str) -> str | None:
    """Convert a MOL2 string to a headerless XYZ block (``El x y z`` per line).

    Elements are taken from the SYBYL atom-type column (e.g. ``N.am`` -> ``N``),
    which is more reliable than the atom-name column.

    The output is headerless (no atom-count / comment lines) to match the
    architector convention used elsewhere in this codebase. Both
    ``parse_xyz_elements`` and ``xyz_string_to_atoms`` parse headerless XYZ
    correctly, whereas they disagree on how many leading lines to skip in a
    standard count+comment header.

    Args:
        mol2: MOL2-format molecule block.

    Returns:
        Headerless XYZ string, or ``None`` if no ``@<TRIPOS>ATOM`` section is
        present.
    """
    lines = mol2.splitlines()
    try:
        atom_start = next(
            i for i, ln in enumerate(lines) if ln.strip() == "@<TRIPOS>ATOM"
        )
    except StopIteration:
        return None

    atoms: list[str] = []
    for ln in lines[atom_start + 1 :]:
        stripped = ln.strip()
        if stripped.startswith("@<TRIPOS>"):
            break
        if not stripped:
            continue
        parts = stripped.split()
        # MOL2 ATOM columns: id name x y z sybyl_type [subst_id subst_name charge]
        x, y, z = parts[2], parts[3], parts[4]
        element = parts[5].split(".")[0]
        atoms.append(f"{element} {x} {y} {z}")

    if not atoms:
        return None

    return "\n".join(atoms)


def unpaired_from_mol2(mol2: str) -> int | None:
    """Extract the ``Unpaired_Electrons`` count from a MOL2 header.

    Args:
        mol2: MOL2-format molecule block.

    Returns:
        Number of unpaired electrons, or ``None`` if not found.
    """
    match = _UNPAIRED_RE.search(mol2)
    return int(match.group(1)) if match else None


def charge_from_mol2(mol2: str) -> int | None:
    """Extract the ``Charge`` value from a MOL2 header (for a consistency check)."""
    match = re.search(r"(?<!XTB_)Charge:\s*(-?\d+)", mol2)
    return int(match.group(1)) if match else None


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Build the geometry/spin columns and cast metadata for DB insertion.

    Adds a ``structure`` (XYZ) column and a ``spinmult`` (2S+1) column, casts the
    numeric extra columns to native Python ints, and resets the index so
    ``orig_index`` reflects row position within this pickle. Prints data-quality
    warnings for charge mismatches and geometry rows that fail to parse.

    Args:
        df: Raw DataFrame loaded from the pickle.

    Returns:
        A copy ready to pass to :func:`create_workflow_db`.
    """
    df = df.reset_index(drop=True).copy()

    df[GEOMETRY_COLUMN] = df[MOL2_COLUMN].apply(mol2_to_xyz)
    unpaired = df[MOL2_COLUMN].apply(unpaired_from_mol2)
    df[SPIN_COLUMN] = unpaired.apply(lambda n: int(n) + 1 if pd.notna(n) else None)

    # Data-quality checks (non-fatal, reported only).
    n_geom_fail = int(df[GEOMETRY_COLUMN].isna().sum())
    n_spin_fail = int(df[SPIN_COLUMN].isna().sum())

    header_charge = df[MOL2_COLUMN].apply(charge_from_mol2)
    charge_mismatch = (header_charge.notna()) & (header_charge != df[CHARGE_COLUMN])
    n_charge_mismatch = int(charge_mismatch.sum())

    parsed_natoms = df[GEOMETRY_COLUMN].apply(
        lambda s: len(s.splitlines()) if isinstance(s, str) else None
    )
    natoms_mismatch = (parsed_natoms.notna()) & (parsed_natoms != df["n_atoms_total"])
    n_natoms_mismatch = int(natoms_mismatch.sum())

    print(f"Rows: {len(df)}")
    if n_geom_fail:
        print(f"  WARNING: {n_geom_fail} rows failed MOL2->XYZ (will be skipped)")
    if n_spin_fail:
        print(f"  WARNING: {n_spin_fail} rows missing Unpaired_Electrons header")
    if n_charge_mismatch:
        print(
            f"  WARNING: {n_charge_mismatch} rows where MOL2 header Charge "
            f"!= {CHARGE_COLUMN} (using {CHARGE_COLUMN})"
        )
    if n_natoms_mismatch:
        print(
            f"  NOTE: {n_natoms_mismatch} rows where parsed atom count "
            f"!= n_atoms_total (geometry is used as ground truth)"
        )

    # Cast numeric extra columns to native Python ints so sqlite can bind them.
    for col, sql_type in EXTRA_COLUMNS.items():
        if sql_type == "INTEGER":
            df[col] = df[col].apply(lambda v: int(v) if pd.notna(v) else None)

    return df


def build_workflow(pickle_path: Path, db_path: Path) -> Path:
    """Load the pickle, adapt its schema, and create the workflow DB.

    Args:
        pickle_path: Path to the genesis lanthanide pickle.
        db_path: Output SQLite database path.

    Returns:
        Path to the created database.
    """
    if not pickle_path.exists():
        raise FileNotFoundError(f"Pickle not found: {pickle_path}")

    df = pd.read_pickle(pickle_path)
    df = prepare_dataframe(df)

    db_path.parent.mkdir(parents=True, exist_ok=True)

    create_workflow_db(
        csv_path=df,
        db_path=db_path,
        geometry_column=GEOMETRY_COLUMN,
        charge_column=CHARGE_COLUMN,
        spin_column=SPIN_COLUMN,
        extra_columns=EXTRA_COLUMNS,
    )

    workflow = ArchitectorWorkflow(db_path)
    try:
        print("\nWorkflow initialized!")
        print(f"  Database: {db_path}")
        print("\nStatus summary:")
        print(workflow.get_summary())

        ready = workflow.get_jobs_by_status(JobStatus.TO_RUN, include_geometry=False)
        if ready:
            print("\nExample job record (geometry excluded):")
            print(ready[0])
    finally:
        workflow.close()

    return db_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pickle", type=Path, default=DEFAULT_PICKLE, help="Input pickle DataFrame."
    )
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_DB, help="Output SQLite database path."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_workflow(args.pickle, args.db)
