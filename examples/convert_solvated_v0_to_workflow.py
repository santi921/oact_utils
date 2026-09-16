"""Convert the solvated_v0 20K actinide pickle into workflow SQLite DBs.

Two databases are written, split on the geometry-derived metal class:
``<stem>_actinides.db`` (Ac-Lr centres) and ``<stem>_non_actinides.db``
(everything else, including the ~9% of rows that hold no actinide atom at all).
No combined database is produced.

The ``sample_20K_Santi_init_9_10_26`` pickle is a ``pandas.DataFrame`` with
columns ``uid, natoms, metal, ind, unpaired_electrons, spin_multiplicity,
total_charge, xyz``. Three adaptations are applied before building the DB:

1. ``xyz`` carries a standard two-line XYZ header (count + blank comment). The
   header is stripped so ``geometry`` holds a bare coordinate block, matching
   the architector convention used elsewhere in this codebase.
2. ``metal`` is derived from the geometry via :func:`census.pick_metal` rather
   than taken from the pickle's ``metal`` column -- see the data-quality note
   below.
3. Charge is ``total_charge`` and spin multiplicity (2S+1) is
   ``spin_multiplicity``; both are used as-is.

Data quality (verified 2026-09-10 on the 20,000-row pickle)
-----------------------------------------------------------
``uid``, ``ind``, ``natoms`` and ``metal`` are BROADCAST PER METAL GROUP: each
of the 8 groups carries a single uid/ind/natoms value repeated across all ~2200
of its rows, so those four columns describe one representative structure, not
the row they sit on. Concretely, the pickle's ``metal`` matches the actinide
actually present in ``xyz`` for only 1393/20000 rows, and its ``natoms`` matches
the atom count in ``xyz`` for only 964/20000. All four are therefore DROPPED and
``metal``/``natoms``/``elements`` are recomputed from the geometry.

``total_charge`` and ``spin_multiplicity`` are NOT broadcast (they vary within
every group) and are consistent with the geometries: electron-count parity
(sum(Z) - charge vs multiplicity) holds for 20000/20000 rows. They are trusted.
:func:`check_alignment` re-runs both checks on every invocation.

Row order is preserved within each shard and the DataFrame index is not reset
after splitting, so ``orig_index`` still points at the row's position in the
source pickle and is unique across the two databases.

Usage:
    python -m examples.convert_solvated_v0_to_workflow           # use defaults
    python examples/convert_solvated_v0_to_workflow.py --help    # override paths
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from ase.data import atomic_numbers

from oact_utilities.utils.architector import create_workflow_db
from oact_utilities.workflows import ArchitectorWorkflow, JobStatus
from oact_utilities.workflows.census import metal_class, pick_metal
from oact_utilities.workflows.screening import (
    add_screening_args,
    apply_screening,
    screening_extra_columns,
)

DEFAULT_PICKLE = Path(
    "/Users/santiagovargas/dev/oact_utils/data/solvated_v0/"
    "sample_20K_Santi_init_9_10_26.pkl"
)
# Base path: the two split DBs are written as <stem>_actinides.db and
# <stem>_non_actinides.db next to it. No combined DB is written.
DEFAULT_DB = Path(
    "/Users/santiagovargas/dev/oact_utils/data/solvated_v0/"
    "sample_20K_Santi_init_9_10_26.db"
)

# Output split: metal_class value -> DB filename suffix. Rows whose metal_class
# is NULL (no metal centre found) go to the non-actinide shard.
SPLITS: dict[str, str] = {
    "actinide": "actinides",
    "non_actinide": "non_actinides",
}

# Source columns in the pickle.
XYZ_COLUMN = "xyz"
CHARGE_COLUMN = "total_charge"
SPIN_COLUMN = "spin_multiplicity"

# New column this script adds and then points the DB builder at.
GEOMETRY_COLUMN = "structure"

# Metadata carried into the DB. Both are derived from the geometry, never from
# the pickle's own (broadcast) metal column.
EXTRA_COLUMNS: dict[str, str] = {
    "metal": "TEXT",
    "metal_class": "TEXT",
}

# Dropped on purpose: broadcast per metal group, so meaningless per row.
UNRELIABLE_COLUMNS = ("uid", "ind", "natoms", "metal")


def strip_xyz_header(xyz: str) -> str | None:
    """Strip the count+comment header from a standard XYZ string.

    Returns a headerless ``El x y z`` block. Lines are kept only when they hold
    at least four whitespace-separated fields, so a blank comment line or a
    trailing blank line cannot leak into the coordinates.

    Args:
        xyz: XYZ-format geometry, with or without the two-line header.

    Returns:
        Headerless XYZ string, or ``None`` if no coordinate lines were found.
    """
    lines = [ln.strip() for ln in xyz.splitlines()]
    coords = [ln for ln in lines if len(ln.split()) >= 4]
    return "\n".join(coords) if coords else None


def elements_of(xyz_body: str) -> list[str]:
    """Return element symbols from a headerless XYZ block."""
    return [ln.split()[0] for ln in xyz_body.splitlines() if ln.strip()]


def check_alignment(df: pd.DataFrame, elements: pd.Series) -> None:
    """Report how far the pickle's own metadata is from its geometries.

    Runs two independent checks and prints the result of each:

    * electron-count parity -- ``sum(Z) - charge`` parity against multiplicity
      parity. A failure means charge and/or spin do not belong to the geometry.
    * metal / atom-count agreement between the pickle's columns and the
      geometry, quantifying the broadcast-metadata problem documented in the
      module docstring.

    Args:
        df: The raw DataFrame (pre-rename), holding charge/spin/metal/natoms.
        elements: Per-row list of element symbols parsed from the geometry.
    """
    n = len(df)

    z_sum = elements.apply(lambda els: sum(atomic_numbers[e] for e in els))
    n_elec = z_sum - df[CHARGE_COLUMN]
    parity_ok = (n_elec % 2) != (df[SPIN_COLUMN] % 2)
    n_parity_bad = int((~parity_ok).sum())

    print(f"Rows: {n}")
    if n_parity_bad:
        print(
            f"  WARNING: {n_parity_bad}/{n} rows fail electron-count parity "
            f"(charge/spin inconsistent with the geometry)"
        )
    else:
        print(f"  OK: charge/spin parity consistent with geometry for all {n} rows")

    if "metal" in df.columns:
        derived = elements.apply(pick_metal)
        n_metal_ok = int((derived == df["metal"]).sum())
        print(
            f"  NOTE: pickle 'metal' matches the geometry for {n_metal_ok}/{n} "
            f"rows (column is broadcast per group; using the derived metal)"
        )
    if "natoms" in df.columns:
        n_atoms_ok = int((elements.apply(len) == df["natoms"]).sum())
        print(
            f"  NOTE: pickle 'natoms' matches the geometry for {n_atoms_ok}/{n} "
            f"rows (column is broadcast per group; recomputing from geometry)"
        )


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Build the geometry/metal columns and drop the unreliable metadata.

    Adds a headerless ``structure`` column plus geometry-derived ``metal`` and
    ``metal_class``, drops the broadcast columns, and resets the index so
    ``orig_index`` reflects row position within this pickle.

    Args:
        df: Raw DataFrame loaded from the pickle.

    Returns:
        A copy ready to pass to :func:`create_workflow_db`.
    """
    df = df.reset_index(drop=True).copy()

    body = df[XYZ_COLUMN].apply(strip_xyz_header)
    n_geom_fail = int(body.isna().sum())
    if n_geom_fail:
        print(f"  WARNING: {n_geom_fail} rows have no parsable geometry (skipped)")

    elements = body.apply(lambda s: elements_of(s) if isinstance(s, str) else [])
    check_alignment(df, elements)

    df = df.drop(columns=[c for c in UNRELIABLE_COLUMNS if c in df.columns])
    df[GEOMETRY_COLUMN] = body
    df["metal"] = elements.apply(pick_metal)
    df["metal_class"] = df["metal"].apply(metal_class)

    n_no_metal = int(df["metal"].isna().sum())
    if n_no_metal:
        print(f"  NOTE: {n_no_metal} rows hold no metal centre (metal left NULL)")

    # Cast to native Python ints so sqlite can bind them.
    for col in (CHARGE_COLUMN, SPIN_COLUMN):
        df[col] = df[col].apply(lambda v: int(v) if pd.notna(v) else None)

    return df


def split_paths(db_path: Path) -> dict[str, Path]:
    """Map each metal_class to its output DB path, derived from ``db_path``.

    Args:
        db_path: Base output path (its stem and parent are reused).

    Returns:
        ``{metal_class: path}`` for every entry in :data:`SPLITS`.
    """
    return {
        cls: db_path.with_name(f"{db_path.stem}_{suffix}{db_path.suffix}")
        for cls, suffix in SPLITS.items()
    }


def build_workflow(
    pickle_path: Path, db_path: Path, screen_args: argparse.Namespace | None = None
) -> list[Path]:
    """Load the pickle, adapt its schema, and create one workflow DB per class.

    The corpus is split on the geometry-derived ``metal_class`` into an actinide
    and a non-actinide database. Row order is preserved within each shard and
    the DataFrame index is not reset after splitting, so ``orig_index`` still
    points at the row's position in the source pickle.

    Args:
        pickle_path: Path to the solvated_v0 pickle.
        db_path: Base output path; the split names are derived from its stem.
        screen_args: Parsed CLI namespace carrying the ``--screen-*`` options. ``None``
            or a namespace without ``--screen-with`` means no screening.

    Returns:
        Paths to the created databases, in :data:`SPLITS` order.

    Raises:
        FileNotFoundError: If the pickle does not exist.
    """
    if not pickle_path.exists():
        raise FileNotFoundError(f"Pickle not found: {pickle_path}")

    df = pd.read_pickle(pickle_path)
    df = prepare_dataframe(df)
    # Screen before the metal_class split so both shards carry the same annotation.
    df, _ = apply_screening(
        df,
        screen_args,
        geometry_column=GEOMETRY_COLUMN,
        charge_column=CHARGE_COLUMN,
        spin_column=SPIN_COLUMN,
        metal_column="metal",
    )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    outputs = split_paths(db_path)

    # NULL metal_class (no metal centre) falls in neither group; keep it.
    assigned = df["metal_class"].fillna("non_actinide")
    unknown = set(assigned.unique()) - set(SPLITS)
    if unknown:
        raise ValueError(f"Unhandled metal_class values: {sorted(unknown)}")

    created: list[Path] = []
    for cls, out_path in outputs.items():
        shard = df[assigned == cls]
        print(f"\n=== {cls}: {len(shard)} structures -> {out_path.name}")
        if shard.empty:
            print("  (empty, no database written)")
            continue

        create_workflow_db(
            csv_path=shard,
            db_path=out_path,
            geometry_column=GEOMETRY_COLUMN,
            charge_column=CHARGE_COLUMN,
            spin_column=SPIN_COLUMN,
            extra_columns={**EXTRA_COLUMNS, **screening_extra_columns(screen_args)},
        )

        workflow = ArchitectorWorkflow(out_path)
        try:
            print(workflow.get_summary())
            ready = workflow.get_jobs_by_status(
                JobStatus.TO_RUN, include_geometry=False
            )
            if ready:
                print("Example job record (geometry excluded):")
                print(ready[0])
        finally:
            workflow.close()

        created.append(out_path)

    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pickle", type=Path, default=DEFAULT_PICKLE, help="Input pickle DataFrame."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help="Base output path; _actinides / _non_actinides are derived from it.",
    )
    add_screening_args(parser)
    args = parser.parse_args()
    if args.screen_with and args.screen_bundle is None:
        parser.error("--screen-with requires --screen-bundle PATH")
    return args


if __name__ == "__main__":
    args = parse_args()
    build_workflow(args.pickle, args.db, screen_args=args)
