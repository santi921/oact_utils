"""Pull structures into one workflow-style SQLite DB, labelled filtered or kept.

The v4 model-dev pipeline (``data/v4_model_dev/build_dataset.py``) loads one
combined ``*_structures.extxyz``, drops structures that fail a quality or
energy check, and records every dropped one in ``filtered_structures.csv``
(``job_path, job_id, folder, stage, reason``). This utility joins those two
files on ``job_path`` and writes a single DB that can be copied off the cluster.

Two uses, one code path:

1. **Visualise a slice of what was thrown away.**  ``--only-filtered`` keeps
   just the dropped structures.  ``--per-reason N`` draws N of each rejection
   reason and ``--per-folder N`` draws N of each campaign, so the sample spans
   every reason (or every campaign) rather than only the common ones -- both
   rejection reasons and campaign sizes span three orders of magnitude, so a
   plain ``--limit`` sample shows almost nothing of the small ones.

2. **Train a classifier on filtered-vs-kept.**  The default keeps every frame
   in the extxyz with ``filtered`` set to 1 or 0, which is the label universe
   ``build_dataset.py`` itself saw: a frame is in the extxyz, so it either
   survived both filters or appears in the CSV.  Job directories that never
   produced a frame are absent from both and are correctly not labelled.

The DB uses the repo's ``structures`` schema, so ``pandas.read_sql_query`` on
it feeds ``notebooks/classifier_homoleptics.ipynb`` directly: ``elements``,
``natoms``, ``charge``, ``spin``, and ``metal`` are the columns that notebook's
``build_scalar_features`` consumes, and ``geometry`` carries the coordinates
for its Phase 1 SOAP descriptors.

LEAKY COLUMNS.  ``fmax_ev_ang``, ``s_squared``, ``homo_lumo_gap_min``,
``n_electrons_scf``, and ``energy_ev`` are the *inputs to the filter itself*.
A classifier trained on them scores ~1.0 and has learned nothing.  They are
stored for inspection and for checking a label, never as features.

UNITS.  ``energy_ev`` is eV and ``fmax_ev_ang`` is eV/Angstrom, both as the
extxyz stores them.  The schema's own ``final_energy`` (Hartree) and
``max_forces`` (Eh/Bohr) are left NULL rather than silently changing units.

Usage:
    # 300 dropped structures, evenly spread over the rejection reasons
    python -m oact_utilities.workflows.pull_filtered \\
        all_structures.extxyz filtered_structures.csv -o look.db \\
        --only-filtered --per-reason 50

    # 20 dropped structures from each of the 25 campaign folders
    python -m oact_utilities.workflows.pull_filtered \\
        all_structures.extxyz filtered_structures.csv -o by_folder.db \\
        --only-filtered --per-folder 20

    # every frame, labelled, no coordinates (smallest classifier DB)
    python -m oact_utilities.workflows.pull_filtered \\
        all_structures.extxyz filtered_structures.csv -o labels.db --no-geometry

    # two campaigns only
    python -m oact_utilities.workflows.pull_filtered \\
        all_structures.extxyz filtered_structures.csv -o act531.db \\
        --folder act_531 --folder nonact_531
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import iread

from ..utils.architector import _init_db, _insert_row
from .census import hill_formula, metal_class, pick_metal

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]

# Bucket label for structures that survived both filters, so --per-reason can
# draw them alongside each rejection reason.
KEPT = "(kept)"

# How :class:`Reservoir` groups rows before applying its cap.
BUCKET_ALL = "all"
BUCKET_REASON = "reason"
BUCKET_FOLDER = "folder"

# Intermediate directories build_dataset.py strips when deriving a campaign
# name from a job path.
_JOB_PARENTS = ("jobs_parsl", "jobs_parsl_backup")
_CHUNK_RE = re.compile(r"_chunk_?\d+$")

# Columns beyond the standard structures schema.
EXTRA_COLUMNS: dict[str, str] = {
    "filtered": "INTEGER",
    "filter_stage": "TEXT",
    "filter_reason": "TEXT",
    "folder": "TEXT",
    "job_path": "TEXT",
    "job_id": "TEXT",
    "metal": "TEXT",
    "metal_class": "TEXT",
    "formula": "TEXT",
    "frame_index": "INTEGER",
    # Filter inputs. Diagnostics only -- see LEAKY COLUMNS above.
    "fmax_ev_ang": "REAL",
    "s_squared": "REAL",
    "homo_lumo_gap_min": "REAL",
    "n_electrons_scf": "REAL",
    "energy_ev": "REAL",
}


@dataclass
class Label:
    """One row of ``filtered_structures.csv``."""

    job_id: str
    folder: str
    stage: str
    reason: str


@dataclass
class Row:
    """One extxyz frame, ready to insert."""

    frame_index: int
    job_path: str
    job_id: str
    folder: str
    filtered: int
    filter_stage: str | None
    filter_reason: str | None
    symbols: list[str]
    geometry: str
    charge: int | None
    spin: int | None
    metal: str | None
    metal_class: str | None
    formula: str
    fmax_ev_ang: float | None
    s_squared: float | None
    homo_lumo_gap_min: float | None
    n_electrons_scf: float | None
    energy_ev: float | None

    @property
    def reason_bucket(self) -> str:
        """The rejection reason, or ``KEPT`` for a structure that survived."""
        return self.filter_reason or KEPT


def bucket_key(row: Row, mode: str) -> str:
    """The sampling bucket a row falls in under ``mode``.

    Args:
        row: The row.
        mode: ``BUCKET_REASON`` buckets by rejection reason (kept structures
            form their own bucket), ``BUCKET_FOLDER`` by campaign folder, and
            ``BUCKET_ALL`` puts everything in one bucket so a cap is global.

    Returns:
        The bucket label.
    """
    if mode == BUCKET_REASON:
        return row.reason_bucket
    if mode == BUCKET_FOLDER:
        return row.folder
    return "(all)"


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def normalise_job_path(job_path: str) -> str:
    """Strip trailing slashes so CSV and extxyz paths compare equal.

    POSIX semantics are used regardless of the host OS: the paths are cluster
    paths that may well be read on a laptop.
    """
    return str(PurePosixPath(job_path.strip()))


def folder_of(job_path: str | None) -> str:
    """Derive the campaign name from a job path.

    Mirrors ``build_dataset.py:_folder_of`` exactly, so kept structures (which
    the CSV never mentions) get the same folder label as filtered ones: drop
    the trailing ``job_XXXX`` and any ``jobs_parsl`` directory, collapse a
    sub-campaign onto its parent (``entropy_grad_0`` -> ``entropy_grad``), and
    strip a chunk suffix (``act_531_chunk12`` -> ``act_531``).

    Args:
        job_path: Absolute job directory path, or None.

    Returns:
        The campaign name, or ``"unknown"`` when no path was recorded.
    """
    if not job_path:
        return "unknown"
    parts = list(PurePosixPath(job_path).parts[:-1])
    while parts and parts[-1] in _JOB_PARENTS:
        parts = parts[:-1]
    if not parts:
        return "unknown"
    name = parts[-1]
    if len(parts) >= 2 and name.startswith(parts[-2]):
        name = parts[-2]
    return _CHUNK_RE.sub("", name)


def load_labels(csv_path: Path) -> dict[str, Label]:
    """Read ``filtered_structures.csv`` into a ``job_path -> Label`` map.

    Args:
        csv_path: The CSV written alongside the filter report.

    Returns:
        One entry per dropped structure, keyed by normalised job path.

    Raises:
        ValueError: If the CSV lacks the ``job_path`` column.
    """
    labels: dict[str, Label] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "job_path" not in reader.fieldnames:
            raise ValueError(
                f"{csv_path} has no 'job_path' column (found: {reader.fieldnames})"
            )
        for record in reader:
            job_path = (record.get("job_path") or "").strip()
            if not job_path:
                continue
            labels[normalise_job_path(job_path)] = Label(
                job_id=(record.get("job_id") or "").strip(),
                folder=(record.get("folder") or "").strip(),
                stage=(record.get("stage") or "").strip(),
                reason=(record.get("reason") or "").strip(),
            )
    return labels


# ---------------------------------------------------------------------------
# Frame -> Row
# ---------------------------------------------------------------------------


def _as_float(value: Any) -> float | None:
    """Coerce an extxyz info value to a single float, or None if it is not one."""
    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        arr = np.asarray(value, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    return float(arr[0]) if arr.size == 1 else None


def _min_float(value: Any) -> float | None:
    """Smallest element of an extxyz info value, or None if it holds no numbers."""
    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        arr = np.asarray(value, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    return float(arr.min()) if arr.size else None


def _as_int(value: Any) -> int | None:
    """Coerce an extxyz info value to an int, or None."""
    number = _as_float(value)
    return None if number is None else int(round(number))


def geometry_block(atoms: Atoms) -> str:
    """Headerless XYZ block, the storage form the workflow DB parses most safely.

    A comment line in the ``geometry`` column has historically broken the
    repo's XYZ readers, so none is written.
    """
    return "\n".join(
        f"{symbol}  {x:.8f}  {y:.8f}  {z:.8f}"
        for symbol, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.positions)
    )


def build_row(
    atoms: Atoms,
    frame_index: int,
    labels: dict[str, Label],
    include_geometry: bool = True,
) -> Row:
    """Turn one extxyz frame into a labelled row.

    Args:
        atoms: The frame, with ``job_path`` in ``atoms.info``.
        frame_index: Position of the frame in the extxyz.
        labels: Output of :func:`load_labels`.
        include_geometry: False leaves the geometry column empty.

    Returns:
        A :class:`Row` whose ``filtered`` is 1 when the frame's job path is in
        ``labels`` and 0 otherwise.
    """
    info = atoms.info
    raw_path = info.get("job_path")
    job_path = normalise_job_path(str(raw_path)) if raw_path else ""
    label = labels.get(job_path)

    symbols = atoms.get_chemical_symbols()
    metal = pick_metal(symbols)

    try:
        energy_ev: float | None = float(atoms.get_potential_energy())
    except (RuntimeError, AttributeError):
        energy_ev = None

    # job_id: the CSV's when labelled, else the job directory's own name.
    job_id = (
        label.job_id if label else (PurePosixPath(job_path).name if job_path else "")
    )

    return Row(
        frame_index=frame_index,
        job_path=job_path,
        job_id=job_id,
        folder=folder_of(job_path),
        filtered=1 if label else 0,
        filter_stage=label.stage if label else None,
        filter_reason=label.reason if label else None,
        symbols=symbols,
        geometry=geometry_block(atoms) if include_geometry else "",
        charge=_as_int(info.get("charge")),
        spin=_as_int(info.get("spin")),
        metal=metal,
        metal_class=metal_class(metal),
        formula=hill_formula(symbols),
        fmax_ev_ang=_as_float(info.get("fmax")),
        s_squared=_as_float(info.get("s_squared_expectation")),
        homo_lumo_gap_min=_min_float(info.get("homo_lumo_gap")),
        n_electrons_scf=_as_float(info.get("num_electrons_scf")),
        energy_ev=energy_ev,
    )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


class Reservoir:
    """Per-bucket reservoir sampler (algorithm R), so one streaming pass suffices.

    With ``cap`` None every row is kept and nothing is buffered by the caller;
    with a cap, each bucket holds at most ``cap`` rows chosen uniformly at
    random from every row that bucket saw. ``mode`` selects the bucketing --
    see :func:`bucket_key`; ``BUCKET_ALL`` makes the cap a global limit.
    """

    def __init__(
        self, cap: int | None, mode: str = BUCKET_ALL, seed: int | None = 0
    ) -> None:
        self.cap = cap
        self.mode = mode
        self.rng = random.Random(seed)
        self.kept: dict[str, list[Row]] = {}
        self.seen: Counter = Counter()

    def add(self, row: Row) -> None:
        """Offer one row to its bucket."""
        key = bucket_key(row, self.mode)
        self.seen[key] += 1
        pool = self.kept.setdefault(key, [])
        if self.cap is None or len(pool) < self.cap:
            pool.append(row)
            return
        # Replace a random member with probability cap/seen, which leaves every
        # row seen so far equally likely to be held.
        index = self.rng.randrange(self.seen[key])
        if index < self.cap:
            pool[index] = row

    def rows(self) -> list[Row]:
        """Every selected row, in extxyz order."""
        out = [row for pool in self.kept.values() for row in pool]
        out.sort(key=lambda r: r.frame_index)
        return out


# ---------------------------------------------------------------------------
# Scan and write
# ---------------------------------------------------------------------------


def collect_rows(
    extxyz_path: Path,
    labels: dict[str, Label],
    only: str = "all",
    folders: frozenset[str] | None = None,
    per_reason: int | None = None,
    per_folder: int | None = None,
    limit: int | None = None,
    include_geometry: bool = True,
    seed: int | None = 0,
    max_frames: int | None = None,
) -> tuple[list[Row], Counter, Counter]:
    """Stream the extxyz once and return the selected rows.

    Args:
        extxyz_path: The combined ``*_structures.extxyz``.
        labels: Output of :func:`load_labels`.
        only: ``"all"``, ``"filtered"``, or ``"kept"``.
        folders: Keep only these campaign names; None keeps all.
        per_reason: Cap on rows per rejection reason (kept rows form their own
            bucket).
        per_folder: Cap on rows per campaign folder.
        limit: Cap on total rows. ``per_reason``, ``per_folder`` and ``limit``
            are mutually exclusive; the first one set wins in that order.
        include_geometry: False leaves the geometry column empty.
        seed: Seed for the reservoir; None gives a fresh random draw.
        max_frames: Stop after this many frames, for testing.

    Returns:
        ``(rows, by_reason, by_folder)``. Both counters cover every frame that
        passed the ``only`` and ``folders`` filters, so they are the true
        population counts even when sampling reduced ``rows``.
    """
    cap: int | None
    if per_reason is not None:
        cap, mode = per_reason, BUCKET_REASON
    elif per_folder is not None:
        cap, mode = per_folder, BUCKET_FOLDER
    else:
        cap, mode = limit, BUCKET_ALL
    reservoir = Reservoir(cap=cap, mode=mode, seed=seed)
    stream = iread(str(extxyz_path), index=":", format="extxyz")
    progress = tqdm(desc="Reading frames", unit="frame") if tqdm is not None else None
    by_reason: Counter = Counter()
    by_folder: Counter = Counter()
    try:
        for frame_index, atoms in enumerate(stream):
            if max_frames is not None and frame_index >= max_frames:
                break
            if progress is not None:
                progress.update(1)
            row = build_row(atoms, frame_index, labels, include_geometry)
            if only == "filtered" and not row.filtered:
                continue
            if only == "kept" and row.filtered:
                continue
            if folders is not None and row.folder not in folders:
                continue
            by_reason[row.reason_bucket] += 1
            by_folder[row.folder] += 1
            reservoir.add(row)
    finally:
        if progress is not None:
            progress.close()
    return reservoir.rows(), by_reason, by_folder


def write_db(rows: list[Row], out_path: Path) -> None:
    """Write the rows to a fresh workflow-style SQLite DB.

    ``status`` is ``completed`` for every row: each frame came from a job whose
    ORCA run finished, and the DB is for analysis, not resubmission.
    ``orig_index`` is the row's position in the output; ``frame_index`` keeps
    its position in the source extxyz.

    Args:
        rows: Selected rows.
        out_path: DB to create. An existing file is replaced.
    """
    out_path.unlink(missing_ok=True)
    conn = _init_db(out_path, extra_columns=EXTRA_COLUMNS)
    try:
        for index, row in enumerate(rows):
            _insert_row(
                conn,
                orig_index=index,
                elements=";".join(row.symbols),
                natoms=len(row.symbols),
                geometry=row.geometry,
                status="completed",
                charge=row.charge,
                spin=row.spin,
                job_dir=row.job_path or None,
                extra_values={
                    "filtered": row.filtered,
                    "filter_stage": row.filter_stage,
                    "filter_reason": row.filter_reason,
                    "folder": row.folder,
                    "job_path": row.job_path,
                    "job_id": row.job_id,
                    "metal": row.metal,
                    "metal_class": row.metal_class,
                    "formula": row.formula,
                    "frame_index": row.frame_index,
                    "fmax_ev_ang": row.fmax_ev_ang,
                    "s_squared": row.s_squared,
                    "homo_lumo_gap_min": row.homo_lumo_gap_min,
                    "n_electrons_scf": row.n_electrons_scf,
                    "energy_ev": row.energy_ev,
                },
            )
        conn.commit()
    finally:
        conn.close()


def _print_report(
    rows: list[Row], by_reason: Counter, by_folder: Counter, out_path: Path
) -> None:
    """Print the eligible population against the written sample, by reason and folder."""
    written_by_reason = Counter(row.reason_bucket for row in rows)
    total_seen = sum(by_reason.values())
    print(f"\nEligible frames: {total_seen:,}   written: {len(rows):,} -> {out_path}")

    if by_reason:
        print(f"\n{'reason':<52}{'eligible':>12}{'written':>10}")
        print("-" * 74)
        for reason, count in by_reason.most_common():
            print(f"{reason:<52}{count:>12,}{written_by_reason[reason]:>10,}")

    if not rows:
        return

    n_filtered = sum(row.filtered for row in rows)
    rate = 100.0 * n_filtered / len(rows)
    print(
        f"\nLabels written: filtered={n_filtered:,}  kept={len(rows) - n_filtered:,}  ({rate:.1f}% filtered)"
    )

    written_by_folder = Counter(row.folder for row in rows)
    filtered_by_folder = Counter(row.folder for row in rows if row.filtered)
    print(f"\n{'folder':<30}{'eligible':>12}{'written':>10}{'filtered':>10}{'%':>8}")
    print("-" * 70)
    for folder, eligible in by_folder.most_common():
        count = written_by_folder[folder]
        share = 100.0 * filtered_by_folder[folder] / count if count else 0.0
        print(
            f"{folder:<30}{eligible:>12,}{count:>10,}"
            f"{filtered_by_folder[folder]:>10,}{share:>7.1f}%"
        )

    no_path = sum(1 for row in rows if not row.job_path)
    if no_path:
        print(
            f"\nWARNING: {no_path:,} frame(s) carry no job_path in atoms.info and "
            "were labelled kept by default. Their labels are not trustworthy."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m oact_utilities.workflows.pull_filtered",
        description=(
            "Join a combined *_structures.extxyz with filtered_structures.csv "
            "and write one workflow-style SQLite DB labelled filtered/kept."
        ),
    )
    parser.add_argument("extxyz", type=Path, help="Combined *_structures.extxyz")
    parser.add_argument("csv", type=Path, help="filtered_structures.csv")
    parser.add_argument(
        "-o", "--output", type=Path, required=True, metavar="PATH", help="DB to create"
    )
    parser.add_argument(
        "--only-filtered",
        action="store_true",
        help="Keep only structures the pipeline dropped (visualisation)",
    )
    parser.add_argument(
        "--only-kept",
        action="store_true",
        help="Keep only structures that survived both filters",
    )
    parser.add_argument(
        "--folder",
        action="append",
        default=None,
        metavar="NAME",
        help="Keep only this campaign folder; repeat for several",
    )
    parser.add_argument(
        "--per-reason",
        type=int,
        default=None,
        metavar="N",
        help="Draw at most N structures per rejection reason (kept rows are "
        "their own bucket), so a sample spans every reason",
    )
    parser.add_argument(
        "--per-folder",
        type=int,
        default=None,
        metavar="N",
        help="Draw at most N structures per campaign folder, so a sample spans "
        "every campaign instead of being dominated by the largest",
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Draw at most N structures in total",
    )
    parser.add_argument(
        "--no-geometry",
        action="store_true",
        help="Leave the geometry column empty (much smaller DB; enough for a "
        "scalar-feature classifier, not for SOAP descriptors)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        metavar="N",
        help="Seed for the random draw (default: 0)",
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=None,
        metavar="N",
        help="Stop after reading N frames, for testing",
    )
    args = parser.parse_args(argv)

    if args.only_filtered and args.only_kept:
        parser.error("--only-filtered and --only-kept are mutually exclusive")
    caps = {
        "--per-reason": args.per_reason,
        "--per-folder": args.per_folder,
        "--limit": args.limit,
    }
    given = [name for name, value in caps.items() if value is not None]
    if len(given) > 1:
        parser.error(f"{' and '.join(given)} are mutually exclusive")
    if not args.extxyz.exists():
        parser.error(f"extxyz not found: {args.extxyz}")
    if not args.csv.exists():
        parser.error(f"csv not found: {args.csv}")

    labels = load_labels(args.csv)
    print(f"Loaded {len(labels):,} filtered job paths from {args.csv}")

    only = "filtered" if args.only_filtered else "kept" if args.only_kept else "all"
    rows, by_reason, by_folder = collect_rows(
        extxyz_path=args.extxyz,
        labels=labels,
        only=only,
        folders=frozenset(args.folder) if args.folder else None,
        per_reason=args.per_reason,
        per_folder=args.per_folder,
        limit=args.limit,
        include_geometry=not args.no_geometry,
        seed=args.seed,
        max_frames=args.debug,
    )

    if not rows:
        print("No frames matched the selection. Nothing written.")
        return 1

    write_db(rows, args.output)
    _print_report(rows, by_reason, by_folder, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
