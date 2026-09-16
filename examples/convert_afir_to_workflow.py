"""Convert the AFIR filtered extended-XYZ trajectories into a workflow SQLite DB.

The ``afir_local_filtered`` corpus is one extended-XYZ file per AFIR trajectory,
each holding N already-subsampled frames. Every frame becomes one row in the
workflow database, so a trajectory of 26 frames yields 26 independent ORCA jobs.

Adaptations applied on the way in:

1. Input is a directory of multi-frame extXYZ files, not a CSV. Frames are
   flattened into a single row stream and inserted in batches of trajectory
   files so peak memory stays bounded.
2. Each frame's comment line carries ``charge=`` and ``spin=`` (already a
   multiplicity, 2S+1 -- it matches the ``UHF<n>`` filename field plus one), so
   both are read per frame rather than derived.
3. The stored ``geometry`` is the bare ``El x y z`` coordinate block. The source
   lines also carry three force columns and sit under a
   ``Properties=species:S:1:pos:R:3:forces:R:3 ...`` comment line; both are
   stripped. ``xyz_string_to_atoms`` treats the comment line of a headered XYZ
   as a coordinate line and would raise on it, so the block must be headerless.
4. Reference energy and reference max force from the source frame are kept in
   ``ref_energy`` / ``ref_max_force`` so they are not confused with the ORCA
   results that ``final_energy`` / ``max_forces`` will hold.

No pre-run screening here, deliberately. The ``filter_risk`` classifier keys heavily on
short interatomic contacts (``min_cov_ratio`` is its single most important feature), but an
AFIR trajectory is a reaction path: compressed bonds are the phenomenon being sampled, not a
defect. Screening these frames would preferentially discard the transition-state-like
geometries this corpus exists to collect. See ``oact_utilities/workflows/screening.py``.

Usage:
    python -m examples.convert_afir_to_workflow            # use defaults
    python examples/convert_afir_to_workflow.py --help     # override paths
"""

from __future__ import annotations

import argparse
import math
import re
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from oact_utilities.utils.architector import create_workflow_db
from oact_utilities.workflows import ArchitectorWorkflow, JobStatus
from oact_utilities.workflows.overlap import canonicalize

DEFAULT_SRC = Path(
    "/Users/santiagovargas/dev/oact_utils/data/afir_v1/afir_local_filtered"
)
DEFAULT_DB = Path("/Users/santiagovargas/dev/oact_utils/data/afir_v1/afir_v1.db")

# Decimal places used when canonicalizing coordinates for the uid hash. Matches
# the overlap.py CLI default, so uids computed here compare directly against a
# `python -m oact_utilities.workflows.overlap` scan of job directories.
UID_DECIMALS = 3

GEOMETRY_COLUMN = "structure"
CHARGE_COLUMN = "charge"
SPIN_COLUMN = "spinmult"

# AFIR provenance carried into the DB. template + frame_index uniquely identify
# a frame in the source corpus; filter_rank orders frames within a trajectory.
EXTRA_COLUMNS: dict[str, str] = {
    "uid": "TEXT",
    "formula": "TEXT",
    "formula_key": "TEXT",
    "template": "TEXT",
    "family": "TEXT",
    "is_rev": "INTEGER",
    "frame_index": "INTEGER",
    "filter_rank": "INTEGER",
    "n_frames": "INTEGER",
    "afir_force": "REAL",
    "ref_energy": "REAL",
    "ref_max_force": "REAL",
}

_FAMILY_RE = re.compile(r"^([A-Za-z]+)")


def parse_comment(comment: str) -> dict[str, str]:
    """Parse the ``key=value`` fields of an extXYZ comment line.

    Quoted values (e.g. ``pbc="F F F"``) are handled; unquoted values run to the
    next whitespace.

    Args:
        comment: The extXYZ comment (second) line of a frame.

    Returns:
        Mapping of field name to raw string value.
    """
    return {
        m.group(1): (m.group(2) if m.group(2) is not None else m.group(3))
        for m in re.finditer(r'(\w+)=(?:"([^"]*)"|(\S+))', comment)
    }


def iter_frames(path: Path) -> Iterator[tuple[str, list[str]]]:
    """Yield ``(comment, atom_lines)`` for each frame in a multi-frame XYZ file.

    Args:
        path: Path to a multi-frame extended-XYZ file.

    Yields:
        Tuples of the frame's comment line and its raw atom lines.

    Raises:
        ValueError: If the file is truncated or an atom-count line is malformed.
    """
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        try:
            natoms = int(lines[i].strip())
        except ValueError as exc:
            raise ValueError(f"{path.name}: bad atom count at line {i + 1}") from exc
        body = lines[i + 2 : i + 2 + natoms]
        if len(body) < natoms:
            raise ValueError(f"{path.name}: truncated frame at line {i + 1}")
        yield lines[i + 1], body
        i += 2 + natoms


def frame_rows(path: Path) -> list[dict[str, object]]:
    """Build one row dict per frame of a single AFIR trajectory file.

    Args:
        path: Path to a multi-frame extended-XYZ trajectory.

    Returns:
        List of row dicts ready to be assembled into a DataFrame.
    """
    frames = list(iter_frames(path))
    template = path.stem
    family_match = _FAMILY_RE.match(template)
    family = family_match.group(1) if family_match else None

    rows: list[dict[str, object]] = []
    for comment, body in frames:
        meta = parse_comment(comment)

        coords: list[str] = []
        atoms: list[tuple[str, float, float, float]] = []
        max_force = 0.0
        for line in body:
            parts = line.split()
            coords.append(f"{parts[0]} {parts[1]} {parts[2]} {parts[3]}")
            atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
            if len(parts) >= 7:
                fx, fy, fz = float(parts[4]), float(parts[5]), float(parts[6])
                max_force = max(max_force, math.sqrt(fx * fx + fy * fy + fz * fz))

        charge = int(meta["charge"])
        spin = int(meta["spin"])
        uid, formula_key, formula = canonicalize(charge, spin, atoms, UID_DECIMALS)

        rows.append(
            {
                GEOMETRY_COLUMN: "\n".join(coords),
                CHARGE_COLUMN: charge,
                SPIN_COLUMN: spin,
                "uid": uid,
                "formula": formula,
                "formula_key": formula_key,
                "template": template,
                "family": family,
                "is_rev": int(template.endswith("_rev")),
                "frame_index": int(meta["frame_index"]),
                "filter_rank": int(meta["filter_rank"]),
                "n_frames": len(frames),
                "afir_force": float(meta["afir_force"]),
                "ref_energy": float(meta["energy"]),
                "ref_max_force": max_force,
            }
        )
    return rows


def build_workflow(src_dir: Path, db_path: Path, files_per_batch: int = 250) -> Path:
    """Flatten every AFIR frame under ``src_dir`` into a workflow DB.

    ``create_workflow_db`` opens the table with ``CREATE TABLE IF NOT EXISTS``,
    so it is called once per batch of trajectory files and appends. A global
    running offset is applied to each batch's DataFrame index so ``orig_index``
    is a stable 0..N-1 position in the flattened frame stream.

    Args:
        src_dir: Directory of ``*.xyz`` AFIR trajectories.
        db_path: Output SQLite database path.
        files_per_batch: Trajectory files to hold in memory at once.

    Returns:
        Path to the created database.

    Raises:
        FileNotFoundError: If ``src_dir`` does not exist or holds no XYZ files.
        FileExistsError: If ``db_path`` already exists (appending would duplicate).
    """
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")
    paths = sorted(src_dir.glob("*.xyz"))
    if not paths:
        raise FileNotFoundError(f"No .xyz files under: {src_dir}")
    if db_path.exists():
        raise FileExistsError(
            f"Database already exists: {db_path}. Remove it first -- this script "
            "appends, so re-running over an existing DB would duplicate rows."
        )

    print(f"Trajectories: {len(paths)}")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    offset = 0
    for start in range(0, len(paths), files_per_batch):
        batch = paths[start : start + files_per_batch]
        rows: list[dict[str, object]] = []
        for path in batch:
            rows.extend(frame_rows(path))

        df = pd.DataFrame(rows)
        df.index = pd.RangeIndex(offset, offset + len(df))
        offset += len(df)

        create_workflow_db(
            csv_path=df,
            db_path=db_path,
            geometry_column=GEOMETRY_COLUMN,
            charge_column=CHARGE_COLUMN,
            spin_column=SPIN_COLUMN,
            extra_columns=EXTRA_COLUMNS,
        )
        print(f"  files {start + len(batch)}/{len(paths)}  frames so far: {offset}")

    # uid is the primary lookup key for this corpus, so index it. The chunker
    # copies index DDL from sqlite_master, so children inherit these.
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_uid ON structures(uid)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_formula_key ON structures(formula_key)"
        )
        conn.commit()
        n_rows, n_uid = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT uid) FROM structures"
        ).fetchone()
    finally:
        conn.close()

    print(f"\nuid: {n_uid} distinct across {n_rows} rows", end="")
    print(" (unique)" if n_uid == n_rows else f" -- {n_rows - n_uid} COLLISIONS")

    workflow = ArchitectorWorkflow(db_path)
    try:
        print(f"\nDatabase: {db_path}")
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
        "--src", type=Path, default=DEFAULT_SRC, help="Directory of AFIR .xyz files."
    )
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_DB, help="Output SQLite database path."
    )
    parser.add_argument(
        "--files-per-batch",
        type=int,
        default=250,
        help="Trajectory files buffered in memory per insert batch.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_workflow(args.src, args.db, args.files_per_batch)
