"""Extract per-atom Mulliken/Loewdin populations into a standalone database.

Reads completed jobs from a workflow database, pulls the per-atom charges and
spin populations that ``parse_job_metrics`` already caches in each job's
``orca_metrics.json``, and writes a new SQLite database with two tables:

``structures``
    One row per sampled job: ids, composition, charge/spin, energy, metal
    centre, spin contamination, the XYZ text, and a ``parse_note`` when
    extraction failed.

``atoms``
    One row per atom: coordinates in Angstrom plus ``mulliken_charge``,
    ``mulliken_spin``, ``loewdin_charge``, ``loewdin_spin``, and the
    energy gradient ``grad_x``, ``grad_y``, ``grad_z`` in Eh/Bohr straight
    from ``orca.engrad`` (the force is its negative).

The source workflow database is opened read-only and never written to, not
even to migrate its schema. Coordinates come from
``orca.engrad`` (the geometry the populations were computed at); a job with no
engrad still gets its populations, with NULL coordinates and gradients and
``geometry_source = 'none'``.

Selection is a seeded random sample by default. ``--select`` and the
``--min/--max-contamination`` flags instead pick by spin contamination
``|<S^2> - S(S+1)|``, read from the workflow DB's ``s_squared`` column and
falling back to each job's ``generator_metrics.json``.

Usage:
    python -m oact_utilities.scripts.extract_populations workflow.db \\
        -o populations.db --root-dir /path/to/jobs --n-samples 100 --summary

    # the 100 worst spin-contaminated completed jobs
    python -m oact_utilities.scripts.extract_populations workflow.db \\
        -o contaminated.db --root-dir /path/to/jobs --n-samples 100 \\
        --select worst-contamination --summary

    # the 100 worst spin-contaminated jobs that fail the contamination filter
    python -m oact_utilities.scripts.extract_populations workflow.db \\
        -o bad_spin.db --root-dir /path/to/jobs -n 100 \\
        --failing-quality spin --select worst-contamination

    # every completed job that fails the force or spin-contamination filter,
    # with its whole job directory copied alongside
    python -m oact_utilities.scripts.extract_populations workflow.db \\
        -o bad_quality.db --root-dir /path/to/jobs --all \\
        --failing-quality --copy-jobs /path/to/bad_jobs

    # also copy the sampled job directories somewhere portable
    python -m oact_utilities.scripts.extract_populations workflow.db \\
        -o populations.db --root-dir /path/to/jobs --n-samples 100 \\
        --copy-jobs /path/to/sample_jobs --copy-skip-scratch
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

from oact_utilities.utils.analysis import parse_job_metrics
from oact_utilities.workflows.census import (
    BOHR_TO_ANG,
    DEFAULT_FORCE_THRESH_EV_ANG,
    EH_BOHR_TO_EV_ANG,
    GENERATOR_CACHE_FILENAME,
    extract_quality_fields,
    metal_class,
    parse_engrad,
    pick_metal,
    read_generator_metrics,
    spin_contamination,
)

# Private, but reusing it keeps --copy-skip-scratch identical to what
# clean.py --clean-all would delete, file/dir distinction and exclusions included.
from oact_utilities.workflows.clean import _match_cleanup_patterns

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is a soft dependency
    tqdm = None  # type: ignore[assignment]

_CREATE_STRUCTURES = """
CREATE TABLE IF NOT EXISTS structures (
    job_id              INTEGER PRIMARY KEY,
    orig_index          INTEGER,
    job_name            TEXT,
    job_dir             TEXT,
    elements            TEXT,
    natoms              INTEGER,
    charge              INTEGER,
    spin                INTEGER,
    metal               TEXT,
    metal_class         TEXT,
    final_energy        REAL,
    max_forces          REAL,
    s_squared           REAL,
    force_max           REAL,
    spin_contamination  REAL,
    contamination_cutoff REAL,
    n_population_atoms  INTEGER,
    geometry_source     TEXT,
    xyz                 TEXT,
    parse_note          TEXT,
    copied_to           TEXT
)
"""

_CREATE_ATOMS = """
CREATE TABLE IF NOT EXISTS atoms (
    job_id          INTEGER,
    atom_index      INTEGER,
    element         TEXT,
    x               REAL,
    y               REAL,
    z               REAL,
    mulliken_charge REAL,
    mulliken_spin   REAL,
    loewdin_charge  REAL,
    loewdin_spin    REAL,
    grad_x          REAL,
    grad_y          REAL,
    grad_z          REAL,
    PRIMARY KEY (job_id, atom_index)
)
"""

_STRUCTURE_FIELDS = (
    "job_id",
    "orig_index",
    "job_name",
    "job_dir",
    "elements",
    "natoms",
    "charge",
    "spin",
    "metal",
    "metal_class",
    "final_energy",
    "max_forces",
    "s_squared",
    "force_max",
    "spin_contamination",
    "contamination_cutoff",
    "n_population_atoms",
    "geometry_source",
    "xyz",
    "parse_note",
)

# An upsert rather than INSERT OR REPLACE: REPLACE deletes the old row, which
# would wipe copied_to from an earlier --copy-jobs run on every --append.
_INSERT_STRUCTURE = (
    f"INSERT INTO structures ({', '.join(_STRUCTURE_FIELDS)}) "
    f"VALUES ({', '.join('?' for _ in _STRUCTURE_FIELDS)}) "
    "ON CONFLICT(job_id) DO UPDATE SET "
    + ", ".join(f"{col} = excluded.{col}" for col in _STRUCTURE_FIELDS[1:])
)

_INSERT_ATOM = """
INSERT OR REPLACE INTO atoms (
    job_id, atom_index, element, x, y, z,
    mulliken_charge, mulliken_spin, loewdin_charge, loewdin_spin,
    grad_x, grad_y, grad_z
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create and configure a logger.

    Args:
        name: Logger name.
        level: Logging level (default: INFO).

    Returns:
        Configured logger instance.
    """
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


def resolve_job_dir(job_dir: str | None, root_dir: str | Path | None) -> Path | None:
    """Resolve a stored ``job_dir`` against an optional new root.

    Mirrors ``clean.py --reroot``: the leaf directory name is preserved when a
    corpus is moved, so ``<root_dir>/<basename>`` finds it again after the
    stored absolute path has gone stale.

    Args:
        job_dir: The ``job_dir`` value stored in the workflow DB.
        root_dir: Root to resolve against, or None to use the stored path.

    Returns:
        The resolved path, or None when the DB holds no ``job_dir``.
    """
    if not job_dir:
        return None
    path = Path(job_dir)
    if root_dir is not None:
        return Path(root_dir) / path.name
    return path


class JobRow(NamedTuple):
    """The workflow DB columns this script needs, read without migrating it.

    ``ArchitectorWorkflow`` would be the natural reader, but opening a database
    through it runs ``_ensure_schema``, which ALTERs in missing columns and
    rewrites legacy ``ready`` statuses. This script promises not to touch the
    source, and a read-only mount must not fail it, so it reads the base
    columns directly instead. They all predate every migration.
    """

    id: int
    orig_index: int | None
    elements: str | None
    natoms: int | None
    charge: int | None
    spin: int | None
    job_dir: str | None
    final_energy: float | None
    max_forces: float | None


_JOB_COLUMNS = (
    "id",
    "orig_index",
    "elements",
    "natoms",
    "charge",
    "spin",
    "job_dir",
    "final_energy",
    "max_forces",
)

_STATUS_COMPLETED = "completed"


def read_completed_jobs(db_path: str | Path) -> list[JobRow]:
    """Read the completed jobs from a workflow DB without writing to it.

    Args:
        db_path: Path to the workflow SQLite database.

    Returns:
        One ``JobRow`` per completed job.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        # ORDER BY so a seeded sample does not depend on the query plan.
        rows = conn.execute(
            f"SELECT {', '.join(_JOB_COLUMNS)} FROM structures WHERE status = ? "
            "ORDER BY id",
            (_STATUS_COMPLETED,),
        ).fetchall()
    finally:
        conn.close()
    return [JobRow(*row) for row in rows]


# The quality scalars this script screens on, in the workflow DB and in the
# per-job caches. JobRow carries neither, so both are raw reads.
_SCALAR_COLUMNS = ("s_squared", "force_max")


def read_scalar_column(db_path: str | Path, column: str) -> dict[int, float]:
    """Read one quality scalar from the workflow DB, keyed by job id.

    Args:
        db_path: Path to the workflow SQLite database.
        column: One of ``_SCALAR_COLUMNS``.

    Returns:
        Mapping of job id to value, for rows where it is not NULL. Empty on a
        database predating that column.
    """
    if column not in _SCALAR_COLUMNS:
        raise ValueError(f"unknown scalar column: {column}")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            f"SELECT id, {column} FROM structures WHERE {column} IS NOT NULL"
        ).fetchall()
    except sqlite3.OperationalError:
        return {}  # pre-quality-scalar database
    finally:
        conn.close()
    return {int(i): float(v) for i, v in rows}


def job_s_squared(job_dir: Path | None, fallback: float | None) -> float | None:
    """Get ``<S^2>`` for one job, preferring the workflow DB value.

    Falls back to the job's ``generator_metrics.json``. Never triggers a fresh
    qtaim parse, so a corpus without that cache simply has no value.

    Args:
        job_dir: Resolved job directory, or None.
        fallback: The workflow DB's ``s_squared`` for this job, if any.

    Returns:
        ``<S^2>`` or None.
    """
    if fallback is not None:
        return fallback
    if job_dir is None or not (job_dir / GENERATOR_CACHE_FILENAME).exists():
        return None
    gen = read_generator_metrics(job_dir, [GENERATOR_CACHE_FILENAME])
    return extract_quality_fields(gen).get("s_squared")


def job_force_max(job_dir: Path | None, fallback: float | None) -> float | None:
    """Get the per-atom max gradient norm (Eh/Bohr), DB value first.

    Falls back to the job's ``orca_metrics.json``, read plainly rather than
    through ``read_orca_cache``: this is a selection screen, and re-statting
    every output to validate a cache would cost more than it saves. The
    extraction pass that follows uses the real cache logic.

    Args:
        job_dir: Resolved job directory, or None.
        fallback: The workflow DB's ``force_max`` for this job, if any.

    Returns:
        ``force_max`` in Eh/Bohr, or None.
    """
    if fallback is not None:
        return fallback
    if job_dir is None:
        return None
    cache = job_dir / "orca_metrics.json"
    try:
        value = json.loads(cache.read_text()).get("force_max")
    except (OSError, ValueError, AttributeError):
        return None
    return float(value) if isinstance(value, (int, float)) else None


def _find_engrad(job_dir: Path) -> Path | None:
    """Locate the .engrad file in a job directory, plain or gzipped."""
    for name in ("orca.engrad", "orca.engrad.gz"):
        candidate = job_dir / name
        if candidate.exists():
            return candidate
    for pattern in ("*.engrad", "*.engrad.gz"):
        matches = sorted(job_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def read_geometry(
    job_dir: Path,
) -> tuple[
    list[str], list[tuple[float, float, float]], list[tuple[float, float, float]]
]:
    """Read the computed geometry and gradient from a job's engrad file.

    Args:
        job_dir: Resolved job directory.

    Returns:
        ``(symbols, coords, gradient)``: coordinates in Angstrom, gradient in
        Eh/Bohr, one triple per atom. All empty when there is no readable
        engrad; ``gradient`` alone is empty when the file holds coordinates
        but no gradient block of matching length.
    """
    engrad = _find_engrad(job_dir)
    if engrad is None:
        return [], [], []
    data = parse_engrad(engrad)
    symbols = data.get("symbols") or []
    flat = data.get("coords_bohr") or []
    if not symbols or len(flat) != 3 * len(symbols):
        return [], [], []
    coords = [
        (
            flat[3 * i] * BOHR_TO_ANG,
            flat[3 * i + 1] * BOHR_TO_ANG,
            flat[3 * i + 2] * BOHR_TO_ANG,
        )
        for i in range(len(symbols))
    ]
    grad_flat = data.get("gradient") or []
    gradient = (
        [
            (grad_flat[3 * i], grad_flat[3 * i + 1], grad_flat[3 * i + 2])
            for i in range(len(symbols))
        ]
        if len(grad_flat) == 3 * len(symbols)
        else []
    )
    return symbols, coords, gradient


def xyz_text(
    symbols: list[str],
    coords: list[tuple[float, float, float]],
    comment: str = "",
) -> str | None:
    """Format symbols and Angstrom coordinates as standard XYZ text."""
    if not symbols or len(coords) != len(symbols):
        return None
    lines = [str(len(symbols)), comment]
    lines += [f"{s} {x:.8f} {y:.8f} {z:.8f}" for s, (x, y, z) in zip(symbols, coords)]
    return "\n".join(lines)


def extract_job(
    job: JobRow,
    job_dir: Path | None,
    s_squared: float | None,
    force_max: float | None,
    unzip: bool,
    hours_cutoff: float,
    recompute: bool = False,
) -> tuple[tuple, list[tuple]]:
    """Build the structures row and atoms rows for one job.

    Args:
        job: ``JobRow`` from the workflow DB.
        job_dir: Resolved job directory, or None when the DB has no path.
        s_squared: ``<S^2>`` for this job, or None.
        force_max: Per-atom max gradient norm (Eh/Bohr) for this job, or None.
        unzip: Force the gzipped-quacc read path. A directory holding a
            ``*.out.gz`` takes it regardless, so a mixed corpus needs no flag.
        hours_cutoff: Timeout threshold handed to ``parse_job_metrics``.

    Returns:
        ``(structure_row, atom_rows)``. ``atom_rows`` is empty when no
        population analysis was found; the structure row then carries a
        ``parse_note``.
    """
    deviation, cutoff = spin_contamination(s_squared, job.spin, job.elements)
    symbols_db = (job.elements or "").split(";") if job.elements else []
    metal = pick_metal(symbols_db) if symbols_db else None

    def _row(
        population_symbols: list[str],
        geometry_source: str,
        xyz: str | None,
        note: str | None,
    ) -> tuple:
        return (
            job.id,
            job.orig_index,
            job_dir.name if job_dir is not None else None,
            str(job_dir) if job_dir is not None else job.job_dir,
            job.elements,
            job.natoms,
            job.charge,
            job.spin,
            metal,
            metal_class(metal),
            job.final_energy,
            job.max_forces,
            s_squared,
            force_max,
            deviation,
            cutoff,
            len(population_symbols),
            geometry_source,
            xyz,
            note,
        )

    if job_dir is None:
        return _row([], "none", None, "no_job_dir"), []
    if not job_dir.is_dir():
        return _row([], "none", None, "dir_not_found"), []

    metrics = parse_job_metrics(
        job_dir,
        unzip=unzip or any(job_dir.glob("*.out.gz")),
        hours_cutoff=hours_cutoff,
        recompute=recompute,
        # with_engrad=True so the .engrad supplies force_max: passing False
        # makes parse_job_metrics cache force_max as null, degrading the very
        # cache the force screen reads on the next run.
        with_engrad=True,
    )
    if force_max is None:
        parsed_force = metrics.get("force_max")
        force_max = parsed_force if isinstance(parsed_force, float) else None

    # parse_job_metrics is annotated as returning scalars, but
    # mulliken_population is the parse_mulliken_population dict.
    population = metrics.get("mulliken_population")
    if not isinstance(population, dict):
        return _row([], "none", None, "no_populations"), []

    elements = population.get("elements") or []
    if not elements:
        return _row([], "none", None, "no_populations"), []

    mulliken_charges = population.get("mulliken_charges") or []
    mulliken_spins = population.get("mulliken_spins") or []
    loewdin_charges = population.get("loewdin_charges") or []
    loewdin_spins = population.get("loewdin_spins") or []

    geom_symbols, coords, gradient = read_geometry(job_dir)
    note = None
    if not geom_symbols:
        geometry_source = "none"
        note = "no_engrad"
    elif geom_symbols != elements:
        geometry_source = "none"
        coords = []
        gradient = []
        note = "geometry_mismatch"
    else:
        geometry_source = "engrad"
        if not gradient:
            note = "no_gradient"

    def _at(values: list, i: int):
        return values[i] if i < len(values) else None

    atom_rows = [
        (
            job.id,
            i,
            element,
            coords[i][0] if i < len(coords) else None,
            coords[i][1] if i < len(coords) else None,
            coords[i][2] if i < len(coords) else None,
            _at(mulliken_charges, i),
            _at(mulliken_spins, i),
            _at(loewdin_charges, i),
            _at(loewdin_spins, i),
            gradient[i][0] if i < len(gradient) else None,
            gradient[i][1] if i < len(gradient) else None,
            gradient[i][2] if i < len(gradient) else None,
        )
        for i, element in enumerate(elements)
    ]

    xyz = xyz_text(geom_symbols, coords, comment=f"job_id={job.id}")
    return _row(elements, geometry_source, xyz, note), atom_rows


def _scratch_ignore(src: str, names: list[str]) -> list[str]:
    """``copytree`` ignore callback dropping what ``clean.py --clean-all`` deletes."""
    return [
        name
        for name in names
        if _match_cleanup_patterns(
            name, os.path.isdir(os.path.join(src, name)), {"tmp", "bas"}
        )
    ]


# Wavefunction files legacy visualizers (Multiwfn, VMD, Chimera) open. Copies
# are checked for one of these so a missing .gbw is reported, not discovered
# later in front of the viewer.
_WAVEFUNCTION_SUFFIXES = (".gbw", ".wfn", ".wfx", ".molden.input", ".nbo")

# Written into every copy, holding the source path, so a later run can tell
# its own earlier copy from a different job that happens to share the name.
_SOURCE_MARKER = ".extract_populations_source"


def _tree_stats(path: Path) -> tuple[int, bool]:
    """Total bytes and wavefunction presence of a copied tree, in one walk.

    Looks through a ``.gz`` suffix, since a quacc corpus stores
    ``orca.gbw.gz`` rather than ``orca.gbw``. Unreadable entries are skipped.
    """
    total, has_wavefunction = 0, False
    for root, _, files in os.walk(path):
        for name in files:
            stem = name[:-3] if name.endswith(".gz") else name
            has_wavefunction = has_wavefunction or stem.endswith(_WAVEFUNCTION_SUFFIXES)
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total, has_wavefunction


def _copied_from_elsewhere(dest: Path, src: Path) -> bool:
    """True when ``dest`` is a marked copy of some directory other than ``src``.

    A destination without a marker (written before markers existed) cannot be
    attributed, so it is taken to be this job's copy, as before.
    """
    try:
        return (dest / _SOURCE_MARKER).read_text().strip() != str(src)
    except OSError:
        return False


def copy_job_dirs(
    pairs: list[tuple[int, Path]],
    dest_root: str | Path,
    skip_scratch: bool = False,
    overwrite: bool = False,
    workers: int = 8,
    logger: logging.Logger | None = None,
) -> dict[int, str]:
    """Copy whole job directories to ``dest_root/<job_name>``.

    Two sources sharing a directory name (``chunk00/job_12`` and
    ``chunk01/job_12``) would otherwise be copied into one destination and
    mixed. A name that occurs more than once in ``pairs``, or whose existing
    destination is marked as another source's copy, is written to
    ``<job_name>__id<job_id>`` instead. Every copy carries a
    ``.extract_populations_source`` file naming its source.

    Args:
        pairs: ``(job_id, source_dir)`` for the jobs to copy.
        dest_root: Directory to copy into; created if missing.
        skip_scratch: Drop ORCA scratch using clean.py's own matcher, i.e.
            exactly what ``clean.py --clean-all`` would delete.
        overwrite: Delete an existing destination and copy afresh, instead of
            leaving it alone. Never applied to another source's copy.
        workers: Parallel copy workers.
        logger: Logger for the per-copy warnings and the total.

    Returns:
        Mapping of job id to destination path, for every job whose data is at
        that destination (freshly copied or already there).
    """
    if logger is None:
        logger = _setup_logger("populations")

    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    ignore = _scratch_ignore if skip_scratch else None

    name_counts: dict[str, int] = {}
    for _, src in pairs:
        name_counts[src.name] = name_counts.get(src.name, 0) + 1
    shared = sum(1 for _, src in pairs if name_counts[src.name] > 1)
    if shared:
        logger.warning(
            "%d jobs share a directory name with another selected job; their "
            "copies are suffixed __id<job_id> to keep them apart",
            shared,
        )

    def _copy(pair: tuple[int, Path]) -> tuple[int, str | None, str, int, bool]:
        job_id, src = pair
        dest = dest_root / src.name
        if name_counts[src.name] > 1 or (
            dest.exists() and _copied_from_elsewhere(dest, src)
        ):
            dest = dest_root / f"{src.name}__id{job_id}"
        if dest.exists():
            if not overwrite:
                return job_id, str(dest), "existing", 0, _tree_stats(dest)[1]
            # copytree into an existing tree merges, which would keep files
            # the fresh copy skips or no longer has; start from nothing.
            shutil.rmtree(dest)
        outcome = "copied"
        try:
            # symlinks are dereferenced: a .gbw symlinked into node-local
            # scratch must arrive as data, not as a dangling link.
            shutil.copytree(src, dest, ignore=ignore)
        except shutil.Error as exc:
            # copytree collects per-file failures (broken symlinks, unreadable
            # scratch) and raises at the end; the rest of the tree is there.
            logger.warning("partial copy for %s: %s", src, exc)
            outcome = "partial"
        except OSError as exc:
            logger.warning("copy failed for %s: %s", src, exc)
            return job_id, None, "failed", 0, False
        (dest / _SOURCE_MARKER).write_text(f"{src}\n")
        size, has_wavefunction = _tree_stats(dest)
        return job_id, str(dest), outcome, size, has_wavefunction

    logger.info("Copying %d job directories to %s ...", len(pairs), dest_root)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        iterator = pool.map(_copy, pairs)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(pairs), unit="dir")
        results = list(iterator)

    counts: dict[str, int] = {}
    total_bytes = 0
    destinations: dict[int, str] = {}
    no_wavefunction = 0
    for job_id, dest, outcome, size, has_wavefunction in results:
        counts[outcome] = counts.get(outcome, 0) + 1
        total_bytes += size
        if dest is not None:
            destinations[job_id] = dest
            no_wavefunction += not has_wavefunction

    logger.info(
        "Copied %d dirs (%.2f GB), %d partial, %d already present, %d failed",
        counts.get("copied", 0),
        total_bytes / 1e9,
        counts.get("partial", 0),
        counts.get("existing", 0),
        counts.get("failed", 0),
    )
    if no_wavefunction:
        logger.warning(
            "%d of %d copies hold no %s file (legacy visualization needs one)",
            no_wavefunction,
            len(destinations),
            "/".join(_WAVEFUNCTION_SUFFIXES),
        )
    return destinations


def _grade_job(
    job: JobRow,
    scalars: dict[int, dict[str, float | None]],
    force_thresh_ev_ang: float,
) -> tuple[float | None, float | None, bool | None, bool | None]:
    """Score one job against the force and spin-contamination filters.

    Args:
        job: ``JobRow`` from the workflow DB.
        scalars: ``{job_id: {"s_squared": ..., "force_max": ...}}``.
        force_thresh_ev_ang: fmax cutoff in eV/Angstrom.

    Returns:
        ``(deviation, fmax_ev_ang, spin_fails, force_fails)``. Either verdict
        is None when the scalar behind it is missing. The comparisons are
        ``>=``, matching ``census.quality_filter``.
    """
    values = scalars.get(job.id, {})
    deviation, cutoff = spin_contamination(
        values.get("s_squared"), job.spin, job.elements
    )
    spin_fails = None if (deviation is None or cutoff is None) else deviation >= cutoff

    force_max = values.get("force_max")
    fmax_ev = None if force_max is None else force_max * EH_BOHR_TO_EV_ANG
    force_fails = None if fmax_ev is None else fmax_ev >= force_thresh_ev_ang

    return deviation, fmax_ev, spin_fails, force_fails


def select_jobs(
    jobs: list[JobRow],
    scalars: dict[int, dict[str, float | None]],
    n_samples: int | None,
    select: str,
    seed: int,
    min_contamination: float | None,
    max_contamination: float | None,
    over_cutoff: bool,
    failing_quality: str | None,
    min_force_ev_ang: float | None,
    force_thresh_ev_ang: float,
    logger: logging.Logger,
) -> list[JobRow]:
    """Filter and sample the candidate jobs.

    ``failing_quality`` picks which filter a job has to fail: ``"any"`` keeps a
    job failing either one, ``"force"`` only the fmax check, ``"spin"`` only
    the contamination check. A job failing the named check is kept whatever the
    other check says. Every other filter is an AND that narrows further.

    Args:
        jobs: Candidate ``JobRow`` list (completed jobs).
        scalars: ``{job_id: {"s_squared": ..., "force_max": ...}}``.
        n_samples: Number to keep, or None for all.
        select: ``random``, ``worst-contamination``, ``best-contamination``,
            ``worst-force``, or ``best-force``.
        seed: RNG seed for random selection.
        min_contamination: Drop jobs with deviation below this.
        max_contamination: Drop jobs with deviation above this.
        over_cutoff: Keep only jobs at or above the contamination cutoff.
        failing_quality: ``"any"``, ``"force"``, ``"spin"``, or None to skip
            this filter.
        min_force_ev_ang: Drop jobs with fmax below this (eV/Angstrom).
        force_thresh_ev_ang: fmax cutoff used by ``failing_quality``.
        logger: Logger for the filter breakdown.

    Returns:
        The selected jobs.
    """
    sort_key = {
        "worst-contamination": "contamination",
        "best-contamination": "contamination",
        "worst-force": "force",
        "best-force": "force",
    }.get(select)

    needs_grading = (
        sort_key is not None
        or failing_quality
        or over_cutoff
        or min_contamination is not None
        or max_contamination is not None
        or min_force_ev_ang is not None
    )

    if not needs_grading:
        if n_samples is None or n_samples >= len(jobs):
            return jobs
        return random.Random(seed).sample(jobs, n_samples)

    kept: list[tuple[float, JobRow]] = []
    n_ungraded = 0
    reasons = {"force only": 0, "spin only": 0, "both": 0}

    for job in jobs:
        deviation, fmax_ev, spin_fails, force_fails = _grade_job(
            job, scalars, force_thresh_ev_ang
        )

        if failing_quality == "spin":
            if spin_fails is None:
                n_ungraded += 1
                continue
            if not spin_fails:
                continue
        elif failing_quality == "force":
            if force_fails is None:
                n_ungraded += 1
                continue
            if not force_fails:
                continue
        elif failing_quality == "any":
            if spin_fails is None and force_fails is None:
                n_ungraded += 1
                continue
            if not (spin_fails or force_fails):
                continue

        # A filter that needs a missing scalar cannot pass or fail the job, so
        # it is counted as ungraded rather than read as "not contaminated".
        if min_contamination is not None or max_contamination is not None:
            if deviation is None:
                n_ungraded += 1
                continue
            if min_contamination is not None and deviation < min_contamination:
                continue
            if max_contamination is not None and deviation > max_contamination:
                continue
        if over_cutoff:
            if spin_fails is None:
                n_ungraded += 1
                continue
            if not spin_fails:
                continue
        if min_force_ev_ang is not None:
            if fmax_ev is None:
                n_ungraded += 1
                continue
            if fmax_ev < min_force_ev_ang:
                continue

        if sort_key == "contamination":
            if deviation is None:
                n_ungraded += 1
                continue
            rank = deviation
        elif sort_key == "force":
            if fmax_ev is None:
                n_ungraded += 1
                continue
            rank = fmax_ev
        else:
            rank = 0.0

        if failing_quality:
            if spin_fails and force_fails:
                reasons["both"] += 1
            elif force_fails:
                reasons["force only"] += 1
            else:
                reasons["spin only"] += 1

        kept.append((rank, job))

    if n_ungraded:
        logger.info(
            "%d jobs skipped for want of a scalar to grade them on (needs "
            "s_squared / force_max in the workflow DB, or the per-job caches)",
            n_ungraded,
        )
    logger.info("%d jobs pass the selection filters", len(kept))
    if failing_quality:
        logger.info(
            "Failure mode: %s (fmax cutoff %g eV/A)",
            ", ".join(f"{k}={v}" for k, v in reasons.items()),
            force_thresh_ev_ang,
        )

    if sort_key is not None:
        kept.sort(key=lambda pair: pair[0], reverse=select.startswith("worst"))
        ordered = [job for _, job in kept]
        return ordered if n_samples is None else ordered[:n_samples]

    pool = [job for _, job in kept]
    if n_samples is None or n_samples >= len(pool):
        return pool
    return random.Random(seed).sample(pool, n_samples)


def _fill_scalar_gaps(
    jobs: list[JobRow],
    scalars: dict[int, dict[str, float | None]],
    job_dirs: dict[int, Path | None],
    workers: int,
    logger: logging.Logger,
) -> None:
    """Fill missing ``s_squared`` / ``force_max`` from the per-job caches.

    Mutates ``scalars`` in place. Jobs already carrying both values are
    skipped, so calling this twice costs nothing the second time.

    Args:
        jobs: The jobs to fill.
        scalars: ``{job_id: {"s_squared": ..., "force_max": ...}}``.
        job_dirs: Resolved directory per job id.
        workers: Parallel workers for the cache reads.
        logger: Logger for the progress line.
    """
    gaps = [
        job
        for job in jobs
        if scalars[job.id]["s_squared"] is None or scalars[job.id]["force_max"] is None
    ]
    if not gaps:
        return

    logger.info(
        "Reading per-job caches for %d jobs whose s_squared or force_max is "
        "missing from the workflow DB...",
        len(gaps),
    )

    def _fill(job: JobRow) -> tuple[float | None, float | None]:
        job_dir = job_dirs[job.id]
        return (
            job_s_squared(job_dir, scalars[job.id]["s_squared"]),
            job_force_max(job_dir, scalars[job.id]["force_max"]),
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        filled = list(pool.map(_fill, gaps))
    for job, (s2, fmax) in zip(gaps, filled):
        scalars[job.id]["s_squared"] = s2
        scalars[job.id]["force_max"] = fmax


def _prepare_output(
    conn: sqlite3.Connection,
    output_path: str | Path,
    append: bool,
    logger: logging.Logger,
) -> None:
    """Create the output tables, clearing anything already there.

    A rerun with different filters must not leave the previous run's rows
    behind, so an existing table is emptied unless ``append`` is set. A table
    written by an older version of this script has a different column set and
    is rebuilt rather than inserted into.

    Args:
        conn: Open connection to the output database.
        output_path: Path, for the log line only.
        append: Keep existing rows and merge into them.
        logger: Logger for the replace/merge notice.

    Raises:
        ValueError: On ``append`` into a table with a stale column set.
    """
    existing_tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND "
            "name IN ('structures', 'atoms')"
        )
    }

    def _declared(create_sql: str) -> set[str]:
        # One column per line; the table-level PRIMARY KEY line is not a column.
        body = create_sql.split("(", 1)[1].rsplit(")", 1)[0]
        return {
            line.split()[0]
            for line in body.splitlines()
            if line.strip() and not line.strip().startswith("PRIMARY KEY")
        }

    def _stale(table: str, create_sql: str) -> bool:
        if table not in existing_tables:
            return False
        columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        return columns != _declared(create_sql)

    if existing_tables:
        if _stale("structures", _CREATE_STRUCTURES) or _stale("atoms", _CREATE_ATOMS):
            if append:
                raise ValueError(
                    f"{output_path} was written by an older version of this "
                    "script (different columns); --append cannot merge into it"
                )
            logger.info("Rebuilding %s: its schema predates this version", output_path)
            conn.execute("DROP TABLE IF EXISTS structures")
            conn.execute("DROP TABLE IF EXISTS atoms")
            existing_tables = set()

    conn.execute(_CREATE_STRUCTURES)
    conn.execute(_CREATE_ATOMS)

    if not existing_tables:
        return

    n_existing = conn.execute("SELECT COUNT(*) FROM structures").fetchone()[0]
    if not n_existing:
        return
    if append:
        logger.info("Merging into %d rows already in %s", n_existing, output_path)
    else:
        logger.info(
            "Replacing %d rows already in %s (pass --append to keep them)",
            n_existing,
            output_path,
        )
        conn.execute("DELETE FROM structures")
        conn.execute("DELETE FROM atoms")


def extract_populations(
    db_path: str | Path,
    output_path: str | Path,
    root_dir: str | Path | None = None,
    n_samples: int | None = 100,
    select: str = "random",
    seed: int = 0,
    min_contamination: float | None = None,
    max_contamination: float | None = None,
    over_cutoff: bool = False,
    failing_quality: str | None = None,
    min_force_ev_ang: float | None = None,
    force_thresh_ev_ang: float = DEFAULT_FORCE_THRESH_EV_ANG,
    workers: int = 8,
    unzip: bool = False,
    hours_cutoff: float = 24.0,
    recompute: bool = False,
    append: bool = False,
    copy_jobs: str | Path | None = None,
    copy_skip_scratch: bool = False,
    copy_overwrite: bool = False,
    logger: logging.Logger | None = None,
) -> int:
    """Sample completed jobs and write their populations to a new database.

    Args:
        db_path: Path to the source workflow SQLite database. Opened
            read-only; no schema migration is applied to it.
        output_path: Path for the output SQLite database.
        root_dir: Resolve each job to ``<root_dir>/<basename>`` instead of its
            stored ``job_dir``. Use when the corpus has been moved.
        n_samples: Number of jobs to sample, or None for all completed jobs.
        select: ``random``, ``worst-contamination``, or ``best-contamination``.
        seed: RNG seed for random selection.
        min_contamination: Keep only jobs with deviation at or above this.
        max_contamination: Keep only jobs with deviation at or below this.
        over_cutoff: Keep only jobs at or above the contamination cutoff.
        failing_quality: Keep only completed jobs failing a quality filter:
            ``"any"`` (either), ``"force"``, or ``"spin"``.
        min_force_ev_ang: Keep only jobs with fmax at or above this
            (eV/Angstrom).
        force_thresh_ev_ang: fmax cutoff for ``failing_quality``.
        workers: Parallel workers for the per-job reads.
        unzip: Force the gzipped-quacc read path. Auto-detected per job dir,
            so this is only needed for an output named unusually.
        hours_cutoff: Timeout threshold handed to ``parse_job_metrics``.
        recompute: Bypass each job's ``orca_metrics.json`` cache.
        append: Merge into an existing output database instead of replacing
            its rows.
        copy_jobs: Copy each sampled job directory into this directory and
            record the destination in the ``copied_to`` column.
        copy_skip_scratch: Drop ORCA scratch from the copies.
        copy_overwrite: Overwrite an existing destination directory.
        logger: Logger instance.

    Returns:
        Number of structure rows written.
    """
    if logger is None:
        logger = _setup_logger("populations")

    completed = read_completed_jobs(db_path)

    if not completed:
        logger.warning("No completed jobs found in database.")
        return 0

    logger.info("Found %d completed jobs.", len(completed))

    job_dirs = {job.id: resolve_job_dir(job.job_dir, root_dir) for job in completed}

    scalars: dict[int, dict[str, float | None]] = {
        job.id: {"s_squared": None, "force_max": None} for job in completed
    }
    for column in _SCALAR_COLUMNS:
        for job_id, value in read_scalar_column(db_path, column).items():
            if job_id in scalars:
                scalars[job_id][column] = value

    needs_grading = (
        select != "random"
        or failing_quality is not None
        or over_cutoff
        or min_contamination is not None
        or max_contamination is not None
        or min_force_ev_ang is not None
    )
    if needs_grading:
        # Every candidate has to be graded before the filters can run.
        _fill_scalar_gaps(completed, scalars, job_dirs, workers, logger)

    selected = select_jobs(
        completed,
        scalars,
        n_samples,
        select,
        seed,
        min_contamination,
        max_contamination,
        over_cutoff,
        failing_quality,
        min_force_ev_ang,
        force_thresh_ev_ang,
        logger,
    )
    if not selected:
        logger.warning("No jobs left after selection.")
        return 0

    # The pre-selection pass only runs when a filter needs grading, so an
    # unfiltered sample would otherwise write NULL scalars for jobs whose
    # values live in the per-job caches rather than the DB columns. Filling
    # here costs one cache read per *selected* job, not per candidate.
    _fill_scalar_gaps(selected, scalars, job_dirs, workers, logger)

    logger.info(
        "Extracting populations for %d jobs (%d workers)...", len(selected), workers
    )

    def _process(job: JobRow) -> tuple[tuple, list[tuple]]:
        return extract_job(
            job,
            job_dirs[job.id],
            scalars[job.id]["s_squared"],
            scalars[job.id]["force_max"],
            unzip,
            hours_cutoff,
            recompute,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        iterator = pool.map(_process, selected)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(selected), unit="job")
        results = list(iterator)

    structure_rows = [row for row, _ in results]
    atom_rows = [atom for _, atoms in results for atom in atoms]

    out_conn = sqlite3.connect(str(output_path))
    _prepare_output(out_conn, output_path, append, logger)
    # On --append a re-extracted job may now yield fewer (or no) atoms than
    # before; clear its old atom rows so none outlive the new structures row.
    out_conn.executemany(
        "DELETE FROM atoms WHERE job_id = ?", [(row[0],) for row in structure_rows]
    )
    out_conn.executemany(_INSERT_STRUCTURE, structure_rows)
    out_conn.executemany(_INSERT_ATOM, atom_rows)
    out_conn.commit()

    if copy_jobs is not None:
        pairs: list[tuple[int, Path]] = []
        for job in selected:
            source = job_dirs[job.id]
            if source is not None and source.is_dir():
                pairs.append((job.id, source))
        destinations = copy_job_dirs(
            pairs,
            copy_jobs,
            skip_scratch=copy_skip_scratch,
            overwrite=copy_overwrite,
            workers=workers,
            logger=logger,
        )
        out_conn.executemany(
            "UPDATE structures SET copied_to = ? WHERE job_id = ?",
            [(dest, job_id) for job_id, dest in destinations.items()],
        )
        out_conn.commit()

    out_conn.close()

    note_counts: dict[str | None, int] = {}
    for row in structure_rows:
        note = row[-1]
        note_counts[note] = note_counts.get(note, 0) + 1
    ok = note_counts.pop(None, 0)

    logger.info(
        "Wrote %d structures (%d clean) and %d atoms to %s",
        len(structure_rows),
        ok,
        len(atom_rows),
        output_path,
    )
    if note_counts:
        logger.info(
            "Notes: %s",
            ", ".join(f"{k}={v}" for k, v in sorted(note_counts.items())),
        )
    return len(structure_rows)


def print_summary(db_path: str | Path) -> None:
    """Print summary statistics from a populations database.

    Args:
        db_path: Path to the populations SQLite database.
    """
    conn = sqlite3.connect(str(db_path))

    n_struct = conn.execute("SELECT COUNT(*) FROM structures").fetchone()[0]
    n_atoms = conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0]
    n_coords = conn.execute(
        "SELECT COUNT(*) FROM atoms WHERE x IS NOT NULL"
    ).fetchone()[0]
    n_grad = conn.execute(
        "SELECT COUNT(*) FROM atoms WHERE grad_x IS NOT NULL"
    ).fetchone()[0]
    print("\n--- Populations Summary ---")
    print(
        f"Structures: {n_struct}  |  atoms: {n_atoms}  |  with coordinates: "
        f"{n_coords}  |  with gradient: {n_grad}"
    )

    notes = conn.execute(
        "SELECT parse_note, COUNT(*) FROM structures WHERE parse_note IS NOT NULL "
        "GROUP BY parse_note ORDER BY COUNT(*) DESC"
    ).fetchall()
    if notes:
        print("Notes: " + ", ".join(f"{n}={c}" for n, c in notes))

    row = conn.execute(
        "SELECT COUNT(*), MIN(spin_contamination), AVG(spin_contamination), "
        "MAX(spin_contamination) FROM structures WHERE spin_contamination IS NOT NULL"
    ).fetchone()
    if row[0]:
        print(
            f"\nSpin contamination |<S^2> - S(S+1)| -- {row[0]} jobs\n"
            f"  min: {row[1]:.4f}  avg: {row[2]:.4f}  max: {row[3]:.4f}"
        )
        over = conn.execute(
            "SELECT COUNT(*) FROM structures WHERE spin_contamination IS NOT NULL "
            "AND contamination_cutoff IS NOT NULL "
            "AND spin_contamination >= contamination_cutoff"
        ).fetchone()[0]
        # >= to match _grade_job and census.quality_filter.
        print(f"  at or over element-dependent cutoff: {over}")

    force_row = conn.execute(
        "SELECT COUNT(*), MIN(force_max), AVG(force_max), MAX(force_max) "
        "FROM structures WHERE force_max IS NOT NULL"
    ).fetchone()
    if force_row[0]:
        scale = EH_BOHR_TO_EV_ANG
        print(
            f"\nMax force per atom (eV/A) -- {force_row[0]} jobs\n"
            f"  min: {force_row[1] * scale:.4f}  avg: {force_row[2] * scale:.4f}  "
            f"max: {force_row[3] * scale:.4f}"
        )

    classes = conn.execute(
        "SELECT metal_class, COUNT(*) FROM structures GROUP BY metal_class "
        "ORDER BY COUNT(*) DESC"
    ).fetchall()
    if classes:
        print("\nMetal class: " + ", ".join(f"{c or 'none'}={n}" for c, n in classes))

    spin_row = conn.execute(
        "SELECT COUNT(*), MIN(mulliken_spin), MAX(mulliken_spin) FROM atoms "
        "WHERE mulliken_spin IS NOT NULL"
    ).fetchone()
    if spin_row[0]:
        print(
            f"\nMulliken spin per atom -- {spin_row[0]} atoms  "
            f"min: {spin_row[1]:.4f}  max: {spin_row[2]:.4f}"
        )
    loewdin_atoms = conn.execute(
        "SELECT COUNT(*) FROM atoms WHERE loewdin_charge IS NOT NULL"
    ).fetchone()[0]
    print(f"Atoms with Loewdin charges: {loewdin_atoms}")

    copied = conn.execute(
        "SELECT COUNT(*) FROM structures WHERE copied_to IS NOT NULL"
    ).fetchone()[0]
    if copied:
        print(f"Job directories copied: {copied}")

    conn.close()


def main() -> None:
    """Command-line entry point: parse arguments, extract, optionally summarise.

    Exits with status 1 when no structures were written.
    """
    parser = argparse.ArgumentParser(
        description="Extract per-atom Mulliken/Loewdin populations from "
        "completed ORCA jobs into a standalone database."
    )
    parser.add_argument("db_path", help="Path to the workflow SQLite database.")
    parser.add_argument(
        "--output",
        "-o",
        default="populations.db",
        help="Output SQLite database path (default: populations.db).",
    )
    parser.add_argument(
        "--root-dir",
        default=None,
        help="Resolve each job to <root-dir>/<basename> instead of its stored "
        "job_dir. Use when the corpus was moved after submission.",
    )
    parser.add_argument(
        "--n-samples",
        "-n",
        type=int,
        default=100,
        help="Number of completed jobs to sample (default: 100). "
        "Use 0 or --all for every completed job.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Extract every completed job (overrides --n-samples).",
    )
    parser.add_argument(
        "--select",
        choices=(
            "random",
            "worst-contamination",
            "best-contamination",
            "worst-force",
            "best-force",
        ),
        default="random",
        help="How to pick the sample (default: random).",
    )
    parser.add_argument(
        "--failing-quality",
        nargs="?",
        const="any",
        choices=("any", "force", "spin"),
        default=None,
        metavar="MODE",
        help="Keep only completed jobs that FAIL a quality filter. "
        "'any' (the default when the flag is bare) keeps a job failing either "
        "check; 'force' keeps fmax >= --force-thresh; 'spin' keeps "
        "|<S^2> - S(S+1)| >= the element-dependent cutoff.",
    )
    parser.add_argument(
        "--force-thresh",
        type=float,
        default=DEFAULT_FORCE_THRESH_EV_ANG,
        help=f"fmax cutoff in eV/Angstrom for --failing-quality "
        f"(default: {DEFAULT_FORCE_THRESH_EV_ANG:g}, matching the dataset "
        "build and dashboard --show-quality).",
    )
    parser.add_argument(
        "--min-force",
        type=float,
        default=None,
        metavar="EV_ANG",
        help="Keep only jobs with fmax at or above this, in eV/Angstrom.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for random selection (default: 0).",
    )
    parser.add_argument(
        "--min-contamination",
        type=float,
        default=None,
        help="Keep only jobs with |<S^2> - S(S+1)| at or above this value.",
    )
    parser.add_argument(
        "--max-contamination",
        type=float,
        default=None,
        help="Keep only jobs with |<S^2> - S(S+1)| at or below this value.",
    )
    parser.add_argument(
        "--over-cutoff",
        action="store_true",
        help="Keep only jobs at or above the element-dependent contamination "
        "cutoff (0.5 for open d/f metals, 1.1 otherwise). Same filter as "
        "--failing-quality spin.",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Parallel workers (default: 8)."
    )
    parser.add_argument(
        "--unzip",
        action="store_true",
        help="Force the gzipped-quacc read path. Auto-detected per job "
        "directory, so this is rarely needed.",
    )
    parser.add_argument(
        "--hours-cutoff",
        type=float,
        default=24.0,
        help="Timeout threshold in hours for status checks (default: 24).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Merge into an existing output database. Without it, a rerun "
        "replaces the rows already in the output file.",
    )
    parser.add_argument(
        "--copy-jobs",
        default=None,
        metavar="DIR",
        help="Also copy each sampled job directory to DIR/<job_name>. The "
        "destination is recorded in the copied_to column.",
    )
    parser.add_argument(
        "--copy-skip-scratch",
        action="store_true",
        help="With --copy-jobs, skip ORCA scratch (.tmp, core, .bas*, "
        "orca_tmp_*/) so the copies hold only the result files.",
    )
    parser.add_argument(
        "--copy-overwrite",
        action="store_true",
        help="With --copy-jobs, copy over an existing destination directory "
        "(default: leave it alone).",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Bypass each job's orca_metrics.json cache and re-read the ORCA "
        "output (slow; use when a cache predates the population columns).",
    )
    parser.add_argument(
        "--summary", action="store_true", help="Print summary statistics afterwards."
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logger = _setup_logger("populations", logging.DEBUG if args.debug else logging.INFO)

    n_samples = None if (args.all or args.n_samples <= 0) else args.n_samples

    count = extract_populations(
        db_path=args.db_path,
        output_path=args.output,
        root_dir=args.root_dir,
        n_samples=n_samples,
        select=args.select,
        seed=args.seed,
        min_contamination=args.min_contamination,
        max_contamination=args.max_contamination,
        over_cutoff=args.over_cutoff,
        failing_quality=args.failing_quality,
        min_force_ev_ang=args.min_force,
        force_thresh_ev_ang=args.force_thresh,
        workers=args.workers,
        unzip=args.unzip,
        hours_cutoff=args.hours_cutoff,
        recompute=args.recompute,
        append=args.append,
        copy_jobs=args.copy_jobs,
        copy_skip_scratch=args.copy_skip_scratch,
        copy_overwrite=args.copy_overwrite,
        logger=logger,
    )

    if count == 0:
        sys.exit(1)

    if args.summary:
        print_summary(args.output)


if __name__ == "__main__":
    main()
