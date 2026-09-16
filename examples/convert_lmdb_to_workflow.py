"""Convert an LMDB of ASE Atoms into a workflow SQLite DB.

This is the LMDB analogue of ``convert_to_workflows.py`` (CSV input) and
``convert_genesis_ln_to_workflow.py`` (pickled DataFrame input). The LMDB is
read into a DataFrame with the same column names the CSV path uses, and that
DataFrame is handed to :func:`create_workflow_db` unchanged, so the resulting
database is identical in schema and semantics to a CSV-derived one.

LMDB layout expected (as produced by the optimized-structures pull):

* Keys are stringified integers ``b"0" .. b"N-1"`` plus a ``b"length"`` key
  holding the pickled record count.
* Each value is a pickled ``ase.Atoms``.
* Per-structure metadata lives in ``Atoms.info``:
  ``charge``, ``spin``, ``orig_idx``, ``name``, ``gen_unique_name``,
  ``is_distorted``, ``distortion_variant_index``, ``metal``, ``met_chrg_spn``,
  ``selection_rank``.

Adaptations applied before building the DB:

1. Keys are visited in numeric order (not the cursor's lexicographic order) so
   DataFrame row position equals LMDB key, which becomes ``orig_index``.
2. ``Atoms`` positions are serialized to a headerless XYZ block into a
   ``structure`` column (the DB stores/consumes XYZ). Headerless matches the
   architector convention: ``parse_xyz_elements`` and ``xyz_string_to_atoms``
   disagree on how many leading lines a count+comment header occupies.
3. ``info["spin"]`` is already spin multiplicity (2S+1) -- verified against the
   ``met_chrg_spn`` field, whose trailing value is the unpaired-electron count
   (``nunpaired + 1 == spin`` for every record). It is copied to a ``spinmult``
   column with no conversion.
4. ``info["charge"]`` is copied to a ``charge`` column.
5. The remaining ``info`` keys are carried through as extra DB columns.

Optional pre-run screening: ``--screen-with filter_risk`` scores each structure with the
v4 filter classifier before it is written, recording the probability that the filter would
discard the result. ``--screen-mode filter`` also drops the rejects. This corpus is squarely
inside that model's training domain (architector-style actinide complexes), which is why
screening is wired here and not into every converter.

Usage:
    python examples/convert_lmdb_to_workflow.py                      # single DB
    python examples/convert_lmdb_to_workflow.py --split-names a b c  # sharded DBs
    python examples/convert_lmdb_to_workflow.py --max-atoms 120      # size cut
    python examples/convert_lmdb_to_workflow.py --help               # override paths

    # score every structure, drop nothing
    python examples/convert_lmdb_to_workflow.py \\
        --screen-with filter_risk --screen-bundle data/v4_model_dev

    # also drop the structures the model rejects
    python examples/convert_lmdb_to_workflow.py \\
        --screen-with filter_risk --screen-bundle data/v4_model_dev --screen-mode filter
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms

from oact_utilities.utils.architector import create_workflow_db, parse_xyz_elements
from oact_utilities.workflows import ArchitectorWorkflow, JobStatus
from oact_utilities.workflows.screening import (
    add_screening_args,
    apply_screening,
    screening_extra_columns,
)

# Default paths for this campaign. Override on the CLI if needed.
DEFAULT_LMDB = Path(
    "/Users/santiagovargas/dev/oact_utils/data/oact_dbs/"
    "optimized_structures_entropy_0804_ishan.lmdb"
)
DEFAULT_DB = Path(
    "/Users/santiagovargas/dev/oact_utils/data/oact_dbs/"
    "optimized_structures_entropy_0804_ishan.db"
)

# Columns this script builds and then points the DB builder at.
GEOMETRY_COLUMN = "structure"
CHARGE_COLUMN = "charge"
SPIN_COLUMN = "spinmult"

# Atoms.info keys carried into the DB as extra columns. Numeric columns are cast
# to native Python ints in prepare_dataframe() so sqlite can bind them.
EXTRA_COLUMNS: dict[str, str] = {
    "metal": "TEXT",
    "name": "TEXT",
    "gen_unique_name": "TEXT",
    "met_chrg_spn": "TEXT",
    "orig_idx": "INTEGER",
    "is_distorted": "INTEGER",
    "distortion_variant_index": "INTEGER",
    "selection_rank": "INTEGER",
}

# Atoms.info key holding spin multiplicity (2S+1) as stored by the generator.
SPIN_INFO_KEY = "spin"


def atoms_to_headerless_xyz(atoms: Atoms) -> str:
    """Serialize ASE Atoms to a headerless XYZ block (``El x y z`` per line).

    No atom-count or comment line is emitted, matching the architector geometry
    convention used by the CSV and pickle converters.

    Args:
        atoms: Structure to serialize.

    Returns:
        Newline-joined ``El x y z`` lines.
    """
    return "\n".join(
        f"{symbol}  {pos[0]:.10f}  {pos[1]:.10f}  {pos[2]:.10f}"
        for symbol, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions())
    )


def unpaired_from_met_chrg_spn(met_chrg_spn: str) -> int | None:
    """Extract the unpaired-electron count from a ``met_chrg_spn`` tag.

    The tag is ``<metal>_<oxidation_state>_<n_unpaired>`` (e.g. ``Po_2_0``).

    Args:
        met_chrg_spn: Tag string from ``Atoms.info``.

    Returns:
        Unpaired-electron count, or ``None`` if the tag does not parse.
    """
    parts = str(met_chrg_spn).split("_")
    if len(parts) != 3:
        return None
    try:
        return int(parts[-1])
    except ValueError:
        return None


def lmdb_to_dataframe(lmdb_path: Path) -> pd.DataFrame:
    """Read an LMDB of pickled ASE Atoms into a DataFrame.

    Records are read in numeric key order so DataFrame row position equals the
    LMDB key. Every ``Atoms.info`` key is kept as a column alongside the
    serialized geometry.

    Args:
        lmdb_path: Path to the LMDB file or directory.

    Returns:
        DataFrame with a ``structure`` column plus one column per ``info`` key.

    Raises:
        FileNotFoundError: If ``lmdb_path`` does not exist.
        ImportError: If the ``lmdb`` package is not installed.
        ValueError: If no numeric records are present.
    """
    if not lmdb_path.exists():
        raise FileNotFoundError(f"LMDB not found: {lmdb_path}")

    try:
        import lmdb
    except ImportError as exc:
        raise ImportError("lmdb package is required to read lmdb files") from exc

    # The pull ships a single-file LMDB; a directory env is also accepted.
    env = lmdb.open(
        str(lmdb_path),
        subdir=lmdb_path.is_dir(),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    try:
        with env.begin() as txn:
            raw_length = txn.get(b"length")
            if raw_length is not None:
                n_records = int(pickle.loads(raw_length))
            else:
                n_records = sum(1 for key, _ in txn.cursor() if key.decode().isdigit())
            if n_records == 0:
                raise ValueError(f"No records found in {lmdb_path}")

            records: list[dict[str, object]] = []
            n_missing = 0
            for i in range(n_records):
                raw = txn.get(str(i).encode())
                if raw is None:
                    n_missing += 1
                    continue
                atoms = pickle.loads(raw)
                records.append(
                    {GEOMETRY_COLUMN: atoms_to_headerless_xyz(atoms), **atoms.info}
                )
    finally:
        env.close()

    print(f"Read {len(records)} records from {lmdb_path.name}")
    if n_missing:
        print(f"  WARNING: {n_missing} of {n_records} keys were absent")

    return pd.DataFrame(records)


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the spin column, cast metadata, and report data quality.

    Copies ``spin`` to ``spinmult`` (no conversion -- it is already 2S+1), casts
    the numeric extra columns to native Python ints, and resets the index so
    ``orig_index`` reflects row position within this LMDB. Prints non-fatal
    warnings for missing geometry/charge/spin and for spin values that disagree
    with the unpaired-electron count in ``met_chrg_spn``.

    Args:
        df: Raw DataFrame from :func:`lmdb_to_dataframe`.

    Returns:
        A copy ready to pass to :func:`create_workflow_db`.
    """
    df = df.reset_index(drop=True).copy()

    missing = [
        col
        for col in (GEOMETRY_COLUMN, CHARGE_COLUMN, SPIN_INFO_KEY, *EXTRA_COLUMNS)
        if col not in df.columns
    ]
    if missing:
        raise ValueError(f"Columns absent from LMDB records: {missing}")

    df[SPIN_COLUMN] = df[SPIN_INFO_KEY]

    # Data-quality checks (non-fatal, reported only).
    empty_geom = df[GEOMETRY_COLUMN].isna() | (df[GEOMETRY_COLUMN] == "")
    n_geom_fail = int(empty_geom.sum())
    n_charge_missing = int(df[CHARGE_COLUMN].isna().sum())
    n_spin_missing = int(df[SPIN_COLUMN].isna().sum())

    unpaired = df["met_chrg_spn"].apply(unpaired_from_met_chrg_spn)
    spin_mismatch = (unpaired.notna()) & (unpaired + 1 != df[SPIN_COLUMN])
    n_spin_mismatch = int(spin_mismatch.sum())

    print(f"Rows: {len(df)}")
    if n_geom_fail:
        print(f"  WARNING: {n_geom_fail} rows have empty geometry (will be skipped)")
    if n_charge_missing:
        print(f"  WARNING: {n_charge_missing} rows missing charge")
    if n_spin_missing:
        print(f"  WARNING: {n_spin_missing} rows missing spin")
    if n_spin_mismatch:
        print(
            f"  WARNING: {n_spin_mismatch} rows where met_chrg_spn unpaired count "
            f"+ 1 != {SPIN_INFO_KEY} (using {SPIN_INFO_KEY} as multiplicity)"
        )

    # Cast numeric extra columns to native Python ints so sqlite can bind them.
    for col, sql_type in EXTRA_COLUMNS.items():
        if sql_type == "INTEGER":
            df[col] = (
                df[col].astype(object).apply(lambda v: int(v) if pd.notna(v) else None)
            )

    return df


def filter_by_natoms(df: pd.DataFrame, max_atoms: int | None) -> pd.DataFrame:
    """Drop rows whose geometry exceeds ``max_atoms`` atoms.

    Atom counts come from :func:`parse_xyz_elements`, the same parser
    :func:`create_workflow_db` uses to populate the ``natoms`` column, so the
    cut matches what lands in the DB. The index is not reset, so surviving rows
    keep the LMDB key they carry as ``orig_index``.

    Args:
        df: Prepared DataFrame from :func:`prepare_dataframe`.
        max_atoms: Keep rows with ``natoms <= max_atoms``. ``None`` keeps all.

    Returns:
        The filtered DataFrame (the input itself when ``max_atoms`` is ``None``).
    """
    if max_atoms is None:
        return df

    natoms = df[GEOMETRY_COLUMN].apply(
        lambda s: len(parse_xyz_elements(str(s))) if pd.notna(s) else 0
    )
    keep = natoms <= max_atoms
    n_dropped = int((~keep).sum())
    print(
        f"  natoms <= {max_atoms}: keeping {int(keep.sum())} of {len(df)} rows "
        f"({n_dropped} dropped)"
    )
    return df[keep]


def build_workflow(
    lmdb_path: Path,
    db_path: Path,
    max_atoms: int | None = None,
    screen_args: argparse.Namespace | None = None,
) -> Path:
    """Read the LMDB, adapt its schema, and create one workflow DB.

    Args:
        lmdb_path: Path to the LMDB of pickled ASE Atoms.
        db_path: Output SQLite database path.
        max_atoms: Keep only structures with ``natoms <= max_atoms``.
        screen_args: Parsed CLI namespace carrying the ``--screen-*`` options. ``None``
            or a namespace without ``--screen-with`` means no screening.

    Returns:
        Path to the created database.
    """
    df = filter_by_natoms(prepare_dataframe(lmdb_to_dataframe(lmdb_path)), max_atoms)
    df, _ = apply_screening(
        df,
        screen_args,
        geometry_column=GEOMETRY_COLUMN,
        charge_column=CHARGE_COLUMN,
        spin_column=SPIN_COLUMN,
        metal_column="metal",
    )

    db_path.parent.mkdir(parents=True, exist_ok=True)

    create_workflow_db(
        csv_path=df,
        db_path=db_path,
        geometry_column=GEOMETRY_COLUMN,
        charge_column=CHARGE_COLUMN,
        spin_column=SPIN_COLUMN,
        extra_columns={**EXTRA_COLUMNS, **screening_extra_columns(screen_args)},
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


def build_split_workflows(
    lmdb_path: Path,
    db_dir: Path,
    db_name: str,
    split_names: list[str],
    fractions: list[float] | None = None,
    seed: int = 42,
    max_atoms: int | None = None,
    screen_args: argparse.Namespace | None = None,
) -> list[Path]:
    """Split the LMDB across several independent workflow DBs.

    Mirrors the random-split branch of
    :func:`oact_utilities.workflows.create_split_workflows`, which cannot be
    reused here because it requires a CSV path on disk. ``orig_index`` is
    preserved across shards (the index is reset once, before splitting), so a
    row keeps its LMDB key in whichever shard it lands.

    Args:
        lmdb_path: Path to the LMDB of pickled ASE Atoms.
        db_dir: Directory where shard DBs are written.
        db_name: Base name; outputs are ``{db_dir}/{db_name}_{split_name}.db``.
        split_names: Shard names; determines the number of shards.
        fractions: Per-shard fractions (must sum to 1.0 and match
            ``split_names`` in length). Defaults to an equal split.
        seed: Seed for the reproducible shuffle.
        max_atoms: Keep only structures with ``natoms <= max_atoms``. Applied
            before sharding, so fractions apply to the surviving rows.

    Returns:
        Paths to the created databases, in ``split_names`` order.

    Raises:
        ValueError: On invalid shard counts or fractions.
    """
    n_shards = len(split_names)
    if n_shards < 2:
        raise ValueError("split_names must contain at least 2 entries.")

    if fractions is None:
        fractions = [1.0 / n_shards] * n_shards
    if len(fractions) != n_shards:
        raise ValueError(
            f"fractions must have the same length as split_names ({n_shards}), "
            f"got {len(fractions)}."
        )
    total_frac = sum(fractions)
    if abs(total_frac - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0, got {total_frac:.6f}.")

    df = filter_by_natoms(prepare_dataframe(lmdb_to_dataframe(lmdb_path)), max_atoms)
    # Screen before sharding so every shard carries the same annotation, and a filtered
    # run drops the same structures it would have dropped in single-DB mode.
    df, _ = apply_screening(
        df,
        screen_args,
        geometry_column=GEOMETRY_COLUMN,
        charge_column=CHARGE_COLUMN,
        spin_column=SPIN_COLUMN,
        metal_column="metal",
    )
    n = len(df)

    counts = [int(round(f * n)) for f in fractions]
    counts[-1] += n - sum(counts)  # absorb rounding error into last shard

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n)
    shards = np.empty(n, dtype=int)
    offset = 0
    for shard_i, count in enumerate(counts):
        shards[shuffled[offset : offset + count]] = shard_i
        offset += count
    df["_shard"] = shards

    db_dir.mkdir(parents=True, exist_ok=True)

    db_paths: list[Path] = []
    for shard_i, name in enumerate(split_names):
        shard_df = df[df["_shard"] == shard_i].drop(columns=["_shard"])
        db_path = db_dir / f"{db_name}_{name}.db"
        create_workflow_db(
            csv_path=shard_df,
            db_path=db_path,
            geometry_column=GEOMETRY_COLUMN,
            charge_column=CHARGE_COLUMN,
            spin_column=SPIN_COLUMN,
            extra_columns=EXTRA_COLUMNS,
        )
        workflow = ArchitectorWorkflow(db_path)
        try:
            n_ready = workflow.count_by_status().get(JobStatus.TO_RUN, 0)
            print(f"  {db_path.name}: {n_ready} jobs ready")
        finally:
            workflow.close()
        db_paths.append(db_path)

    return db_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lmdb", type=Path, default=DEFAULT_LMDB, help="Input LMDB of ASE Atoms."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help="Output SQLite database path (single-DB mode).",
    )
    parser.add_argument(
        "--split-names",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Shard names. When given, writes one DB per shard next to --db, "
        "named <db-stem>_<name>.db, instead of a single DB.",
    )
    parser.add_argument(
        "--split-fractions",
        nargs="+",
        type=float,
        default=None,
        metavar="FRAC",
        help="Per-shard fractions, must sum to 1.0. Default: equal split.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for the split shuffle."
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=None,
        metavar="N",
        help="Keep only structures with natoms <= N. Applied before sharding.",
    )
    add_screening_args(parser)
    args = parser.parse_args()
    if args.screen_with and args.screen_bundle is None:
        parser.error("--screen-with requires --screen-bundle PATH")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.split_names:
        build_split_workflows(
            lmdb_path=args.lmdb,
            db_dir=args.db.parent,
            db_name=args.db.stem,
            split_names=args.split_names,
            fractions=args.split_fractions,
            seed=args.seed,
            max_atoms=args.max_atoms,
            screen_args=args,
        )
    else:
        build_workflow(args.lmdb, args.db, max_atoms=args.max_atoms, screen_args=args)
