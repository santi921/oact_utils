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
    ``mulliken_spin``, ``loewdin_charge``, ``loewdin_spin``.

The source workflow database is never written to. Coordinates come from
``orca.engrad`` (the geometry the populations were computed at); a job with no
engrad still gets its populations, with NULL coordinates and
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
"""

from __future__ import annotations

import argparse
import logging
import random
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from oact_utilities.utils.analysis import parse_job_metrics
from oact_utilities.workflows.architector_workflow import (
    ArchitectorWorkflow,
    JobStatus,
)
from oact_utilities.workflows.census import (
    BOHR_TO_ANG,
    GENERATOR_CACHE_FILENAME,
    extract_quality_fields,
    metal_class,
    parse_engrad,
    pick_metal,
    read_generator_metrics,
    spin_contamination,
)

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
    spin_contamination  REAL,
    contamination_cutoff REAL,
    n_population_atoms  INTEGER,
    geometry_source     TEXT,
    xyz                 TEXT,
    parse_note          TEXT
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
    PRIMARY KEY (job_id, atom_index)
)
"""

_INSERT_STRUCTURE = """
INSERT OR REPLACE INTO structures (
    job_id, orig_index, job_name, job_dir, elements, natoms, charge, spin,
    metal, metal_class, final_energy, max_forces, s_squared,
    spin_contamination, contamination_cutoff, n_population_atoms,
    geometry_source, xyz, parse_note
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_INSERT_ATOM = """
INSERT OR REPLACE INTO atoms (
    job_id, atom_index, element, x, y, z,
    mulliken_charge, mulliken_spin, loewdin_charge, loewdin_spin
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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


def read_s_squared_column(db_path: str | Path) -> dict[int, float]:
    """Read the workflow DB's ``s_squared`` column, keyed by job id.

    ``JobRecord`` does not carry the quality scalars, so this is a raw read.

    Args:
        db_path: Path to the workflow SQLite database.

    Returns:
        Mapping of job id to ``<S^2>`` for rows where it is not NULL.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT id, s_squared FROM structures WHERE s_squared IS NOT NULL"
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


def read_geometry(job_dir: Path) -> tuple[list[str], list[tuple[float, float, float]]]:
    """Read the computed geometry from a job's engrad file, in Angstrom.

    Args:
        job_dir: Resolved job directory.

    Returns:
        ``(symbols, coords)``. Both empty when there is no readable engrad.
    """
    engrad = _find_engrad(job_dir)
    if engrad is None:
        return [], []
    data = parse_engrad(engrad)
    symbols = data.get("symbols") or []
    flat = data.get("coords_bohr") or []
    if not symbols or len(flat) != 3 * len(symbols):
        return [], []
    coords = [
        (
            flat[3 * i] * BOHR_TO_ANG,
            flat[3 * i + 1] * BOHR_TO_ANG,
            flat[3 * i + 2] * BOHR_TO_ANG,
        )
        for i in range(len(symbols))
    ]
    return symbols, coords


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
    job,
    job_dir: Path | None,
    s_squared: float | None,
    unzip: bool,
    hours_cutoff: float,
    recompute: bool = False,
) -> tuple[tuple, list[tuple]]:
    """Build the structures row and atoms rows for one job.

    Args:
        job: ``JobRecord`` from the workflow DB.
        job_dir: Resolved job directory, or None when the DB has no path.
        s_squared: ``<S^2>`` for this job, or None.
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
        with_engrad=False,
    )
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

    geom_symbols, coords = read_geometry(job_dir)
    note = None
    if not geom_symbols:
        geometry_source = "none"
        note = "no_engrad"
    elif geom_symbols != elements:
        geometry_source = "none"
        coords = []
        note = "geometry_mismatch"
    else:
        geometry_source = "engrad"

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
        )
        for i, element in enumerate(elements)
    ]

    xyz = xyz_text(geom_symbols, coords, comment=f"job_id={job.id}")
    return _row(elements, geometry_source, xyz, note), atom_rows


def select_jobs(
    jobs: list,
    s_squared_by_id: dict[int, float],
    n_samples: int | None,
    select: str,
    seed: int,
    min_contamination: float | None,
    max_contamination: float | None,
    over_cutoff: bool,
    logger: logging.Logger,
) -> list:
    """Filter and sample the candidate jobs.

    Args:
        jobs: Candidate ``JobRecord`` list (completed jobs).
        s_squared_by_id: ``<S^2>`` per job id, already resolved.
        n_samples: Number to keep, or None for all.
        select: ``random``, ``worst-contamination``, or ``best-contamination``.
        seed: RNG seed for ``random`` selection.
        min_contamination: Drop jobs with deviation below this.
        max_contamination: Drop jobs with deviation above this.
        over_cutoff: Keep only jobs above the element-dependent cutoff.
        logger: Logger for the filter breakdown.

    Returns:
        The selected jobs.
    """
    by_contamination = select != "random"
    needs_contamination = (
        by_contamination
        or over_cutoff
        or min_contamination is not None
        or max_contamination is not None
    )

    if not needs_contamination:
        if n_samples is None or n_samples >= len(jobs):
            return jobs
        return random.Random(seed).sample(jobs, n_samples)

    scored: list[tuple[float, object]] = []
    n_missing = 0
    for job in jobs:
        deviation, cutoff = spin_contamination(
            s_squared_by_id.get(job.id), job.spin, job.elements
        )
        if deviation is None:
            n_missing += 1
            continue
        if min_contamination is not None and deviation < min_contamination:
            continue
        if max_contamination is not None and deviation > max_contamination:
            continue
        if over_cutoff and (cutoff is None or deviation <= cutoff):
            continue
        scored.append((deviation, job))

    if n_missing:
        logger.info(
            "%d completed jobs have no <S^2> and were skipped by the "
            "contamination selection (needs the s_squared column or "
            "generator_metrics.json)",
            n_missing,
        )
    logger.info("%d jobs pass the contamination filter", len(scored))

    if by_contamination:
        scored.sort(key=lambda pair: pair[0], reverse=select == "worst-contamination")
        selected = [job for _, job in scored]
        return selected if n_samples is None else selected[:n_samples]

    pool = [job for _, job in scored]
    if n_samples is None or n_samples >= len(pool):
        return pool
    return random.Random(seed).sample(pool, n_samples)


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
    workers: int = 8,
    unzip: bool = False,
    hours_cutoff: float = 24.0,
    recompute: bool = False,
    logger: logging.Logger | None = None,
) -> int:
    """Sample completed jobs and write their populations to a new database.

    Args:
        db_path: Path to the source workflow SQLite database (read only).
        output_path: Path for the output SQLite database.
        root_dir: Resolve each job to ``<root_dir>/<basename>`` instead of its
            stored ``job_dir``. Use when the corpus has been moved.
        n_samples: Number of jobs to sample, or None for all completed jobs.
        select: ``random``, ``worst-contamination``, or ``best-contamination``.
        seed: RNG seed for random selection.
        min_contamination: Keep only jobs with deviation at or above this.
        max_contamination: Keep only jobs with deviation at or below this.
        over_cutoff: Keep only jobs above the element-dependent cutoff.
        workers: Parallel workers for the per-job reads.
        unzip: Force the gzipped-quacc read path. Auto-detected per job dir,
            so this is only needed for an output named unusually.
        hours_cutoff: Timeout threshold handed to ``parse_job_metrics``.
        recompute: Bypass each job's ``orca_metrics.json`` cache.
        logger: Logger instance.

    Returns:
        Number of structure rows written.
    """
    if logger is None:
        logger = _setup_logger("populations")

    with ArchitectorWorkflow(db_path) as wf:
        completed = wf.get_jobs_by_status(JobStatus.COMPLETED, include_geometry=False)

    if not completed:
        logger.warning("No completed jobs found in database.")
        return 0

    logger.info("Found %d completed jobs.", len(completed))

    job_dirs = {job.id: resolve_job_dir(job.job_dir, root_dir) for job in completed}

    s_squared_by_id = read_s_squared_column(db_path)
    needs_contamination = (
        select != "random"
        or over_cutoff
        or min_contamination is not None
        or max_contamination is not None
    )
    if needs_contamination:
        missing = [job for job in completed if job.id not in s_squared_by_id]
        if missing:
            logger.info(
                "Reading generator_metrics.json for %d jobs with no s_squared "
                "in the workflow DB...",
                len(missing),
            )
            with ThreadPoolExecutor(max_workers=workers) as pool:
                values = list(
                    pool.map(lambda job: job_s_squared(job_dirs[job.id], None), missing)
                )
            for job, value in zip(missing, values):
                if value is not None:
                    s_squared_by_id[job.id] = value

    selected = select_jobs(
        completed,
        s_squared_by_id,
        n_samples,
        select,
        seed,
        min_contamination,
        max_contamination,
        over_cutoff,
        logger,
    )
    if not selected:
        logger.warning("No jobs left after selection.")
        return 0

    logger.info(
        "Extracting populations for %d jobs (%d workers)...", len(selected), workers
    )

    def _process(job):
        return extract_job(
            job,
            job_dirs[job.id],
            s_squared_by_id.get(job.id),
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
    out_conn.execute(_CREATE_STRUCTURES)
    out_conn.execute(_CREATE_ATOMS)
    out_conn.executemany(_INSERT_STRUCTURE, structure_rows)
    out_conn.executemany(_INSERT_ATOM, atom_rows)
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
    print("\n--- Populations Summary ---")
    print(
        f"Structures: {n_struct}  |  atoms: {n_atoms}  |  with coordinates: {n_coords}"
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
            "AND spin_contamination > contamination_cutoff"
        ).fetchone()[0]
        print(f"  over element-dependent cutoff: {over}")

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

    conn.close()


def main() -> None:
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
        choices=("random", "worst-contamination", "best-contamination"),
        default="random",
        help="How to pick the sample (default: random).",
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
        help="Keep only jobs above the element-dependent contamination cutoff "
        "(0.5 for open d/f metals, 1.1 otherwise).",
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
        workers=args.workers,
        unzip=args.unzip,
        hours_cutoff=args.hours_cutoff,
        recompute=args.recompute,
        logger=logger,
    )

    if count == 0:
        sys.exit(1)

    if args.summary:
        print_summary(args.output)


if __name__ == "__main__":
    main()
