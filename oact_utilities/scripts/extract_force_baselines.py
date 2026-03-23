"""Extract per-atom force statistics and energies from completed ORCA jobs.

Reads completed jobs from a workflow database, parses .engrad files,
and produces a new SQLite database with job metadata (no geometry) plus
force baselines: max force, mean force, forces on actinide atoms, and
forces on actinide-neighbor atoms (non-actinide atoms within a distance
cutoff of any actinide center).

Energy data is stored from two sources: the workflow DB ``final_energy``
column and the engrad file ``total_energy_engrad``. The summary report
includes statistics for both and flags any mismatches.

All completed jobs are written to the output DB. Jobs where force
extraction fails get a short ``parse_note`` explaining why
(e.g. "no_job_dir", "no_engrad", "parse_failed", "bad_gradient").

Usage:
    python -m oact_utilities.scripts.extract_force_baselines \\
        workflow.db --output force_baselines.db

    python -m oact_utilities.scripts.extract_force_baselines \\
        workflow.db --root-dir /path/to/jobs --output force_baselines.db \\
        --neighbor-cutoff 4.0 --log-name my_campaign
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
from periodictable import elements as pt_elements

from oact_utilities.utils.analysis import get_engrad
from oact_utilities.utils.create import fetch_actinides
from oact_utilities.workflows.architector_workflow import (
    ArchitectorWorkflow,
    JobStatus,
)

# Build a lookup from atomic number to element symbol
_Z_TO_SYMBOL: dict[int, str] = {el.number: el.symbol for el in pt_elements}

# Set of actinide symbols for fast membership checks
_ACTINIDE_SET: set[str] = set(fetch_actinides())

# Bohr to Angstrom conversion factor
_BOHR_TO_ANGSTROM: float = 0.529177210903

# Output table schema (job metadata + force baselines, no geometry)
_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS force_baselines (
    job_id          INTEGER PRIMARY KEY,
    orig_index      INTEGER,
    elements        TEXT,
    natoms          INTEGER,
    charge          INTEGER,
    spin            INTEGER,
    status          TEXT,
    job_dir         TEXT,
    final_energy    REAL,
    max_forces_db   REAL,
    scf_steps       INTEGER,
    wall_time       REAL,
    n_cores         INTEGER,
    fail_count      INTEGER,
    max_force       REAL,
    mean_force      REAL,
    total_energy_engrad     REAL,
    actinide_elements       TEXT,
    actinide_max_force      REAL,
    actinide_mean_force     REAL,
    actinide_forces_detail  TEXT,
    non_actinide_max_force  REAL,
    non_actinide_mean_force REAL,
    non_actinide_forces_detail TEXT,
    has_actinide            INTEGER,
    neighbor_indices        TEXT,
    neighbor_elements       TEXT,
    neighbor_max_force      REAL,
    neighbor_mean_force     REAL,
    neighbor_forces_detail  TEXT,
    neighbor_cutoff         REAL,
    n_neighbors             INTEGER,
    parse_note              TEXT
)
"""

_INSERT_ROW = """
INSERT OR REPLACE INTO force_baselines (
    job_id, orig_index, elements, natoms, charge, spin, status,
    job_dir, final_energy, max_forces_db, scf_steps, wall_time,
    n_cores, fail_count,
    max_force, mean_force, total_energy_engrad,
    actinide_elements, actinide_max_force, actinide_mean_force,
    actinide_forces_detail,
    non_actinide_max_force, non_actinide_mean_force,
    non_actinide_forces_detail, has_actinide,
    neighbor_indices, neighbor_elements, neighbor_max_force,
    neighbor_mean_force, neighbor_forces_detail, neighbor_cutoff,
    n_neighbors, parse_note
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
          ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
          ?, ?, ?, ?, ?, ?, ?, ?)
"""

# Null placeholder for the 18 force columns when parsing fails
# (11 original + 7 neighbor columns)
_NULL_FORCE_FIELDS = (None,) * 18


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
        fmt = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def _atomic_number_to_symbol(z: int) -> str:
    """Convert atomic number to element symbol."""
    return _Z_TO_SYMBOL.get(z, f"Z{z}")


def extract_force_info(
    engrad_path: str | Path,
    neighbor_cutoff: float = 4.0,
    logger: logging.Logger | None = None,
) -> tuple[dict | None, str | None]:
    """Extract per-atom force statistics from an engrad file.

    Computes global, actinide, non-actinide, and actinide-neighbor force
    statistics. Actinide neighbors are non-actinide atoms within
    ``neighbor_cutoff`` Angstroms of any actinide center.

    Args:
        engrad_path: Path to the orca.engrad file.
        neighbor_cutoff: Distance cutoff in Angstroms for identifying
            actinide neighbors (default: 4.0).
        logger: Optional logger for debug messages.

    Returns:
        Tuple of (force_dict, parse_note). force_dict is None on failure,
        with parse_note explaining why.
    """
    engrad_path = Path(engrad_path)
    if not engrad_path.exists():
        return None, "no_engrad"

    try:
        data = get_engrad(str(engrad_path))
    except Exception as exc:
        if logger:
            logger.debug("engrad exception for %s: %s", engrad_path, exc)
        return None, "parse_failed"

    gradient = data.get("gradient_Eh_per_bohr")
    atomic_numbers = data.get("elements")
    if not gradient or not atomic_numbers:
        return None, "empty_engrad"

    gradient_arr = np.array(gradient)
    natoms = len(gradient_arr) // 3
    if len(gradient_arr) % 3 != 0 or natoms == 0:
        return None, "bad_gradient"
    if natoms != len(atomic_numbers):
        return None, "atom_count_mismatch"

    gradient_3d = gradient_arr.reshape((natoms, 3))
    force_magnitudes = np.linalg.norm(gradient_3d, axis=1)

    symbols = [_atomic_number_to_symbol(z) for z in atomic_numbers]

    actinide_indices = [i for i, sym in enumerate(symbols) if sym in _ACTINIDE_SET]
    non_actinide_indices = [
        i for i, sym in enumerate(symbols) if sym not in _ACTINIDE_SET
    ]

    result: dict = {
        "max_force": float(np.max(force_magnitudes)),
        "mean_force": float(np.mean(force_magnitudes)),
        "total_energy_engrad": data.get("total_energy_Eh"),
        "has_actinide": 1 if actinide_indices else 0,
    }

    # Actinide forces
    if actinide_indices:
        act_forces = force_magnitudes[actinide_indices]
        act_elements = [symbols[i] for i in actinide_indices]
        result["actinide_elements"] = ";".join(act_elements)
        result["actinide_max_force"] = float(np.max(act_forces))
        result["actinide_mean_force"] = float(np.mean(act_forces))
        result["actinide_forces_detail"] = ";".join(
            f"{sym}:{force_magnitudes[i]:.8f}"
            for i, sym in zip(actinide_indices, act_elements)
        )
    else:
        result["actinide_elements"] = None
        result["actinide_max_force"] = None
        result["actinide_mean_force"] = None
        result["actinide_forces_detail"] = None

    # Non-actinide forces
    if non_actinide_indices:
        non_act_forces = force_magnitudes[non_actinide_indices]
        non_act_elements = [symbols[i] for i in non_actinide_indices]
        result["non_actinide_max_force"] = float(np.max(non_act_forces))
        result["non_actinide_mean_force"] = float(np.mean(non_act_forces))
        result["non_actinide_forces_detail"] = ";".join(
            f"{sym}:{force_magnitudes[i]:.8f}"
            for i, sym in zip(non_actinide_indices, non_act_elements)
        )
    else:
        result["non_actinide_max_force"] = None
        result["non_actinide_mean_force"] = None
        result["non_actinide_forces_detail"] = None

    # Actinide-neighbor forces: non-actinide atoms within cutoff of any actinide
    result["neighbor_cutoff"] = neighbor_cutoff
    if actinide_indices and non_actinide_indices:
        coords_bohr = data.get("coords_bohr")
        if coords_bohr is not None and len(coords_bohr) == natoms:
            coords_ang = np.array(coords_bohr) * _BOHR_TO_ANGSTROM
            non_act_set = set(non_actinide_indices)

            # Find union of all non-actinide atoms within cutoff of any actinide
            neighbor_set: set[int] = set()
            for ai in actinide_indices:
                dists = np.linalg.norm(coords_ang - coords_ang[ai], axis=1)
                for ni in non_act_set:
                    if dists[ni] <= neighbor_cutoff:
                        neighbor_set.add(ni)

            neighbor_indices_sorted = sorted(neighbor_set)
            if neighbor_indices_sorted:
                nbr_forces = force_magnitudes[neighbor_indices_sorted]
                nbr_symbols = [symbols[i] for i in neighbor_indices_sorted]
                result["neighbor_indices"] = ";".join(
                    str(i) for i in neighbor_indices_sorted
                )
                result["neighbor_elements"] = ";".join(nbr_symbols)
                result["neighbor_max_force"] = float(np.max(nbr_forces))
                result["neighbor_mean_force"] = float(np.mean(nbr_forces))
                result["neighbor_forces_detail"] = ";".join(
                    f"{symbols[i]}:{force_magnitudes[i]:.8f}"
                    for i in neighbor_indices_sorted
                )
                result["n_neighbors"] = len(neighbor_indices_sorted)
            else:
                result["neighbor_indices"] = None
                result["neighbor_elements"] = None
                result["neighbor_max_force"] = None
                result["neighbor_mean_force"] = None
                result["neighbor_forces_detail"] = None
                result["n_neighbors"] = 0
        else:
            # No coordinate data in engrad -- cannot compute neighbors
            result["neighbor_indices"] = None
            result["neighbor_elements"] = None
            result["neighbor_max_force"] = None
            result["neighbor_mean_force"] = None
            result["neighbor_forces_detail"] = None
            result["n_neighbors"] = None
    else:
        # No actinides or no non-actinides -- no neighbors to find
        result["neighbor_indices"] = None
        result["neighbor_elements"] = None
        result["neighbor_max_force"] = None
        result["neighbor_mean_force"] = None
        result["neighbor_forces_detail"] = None
        result["n_neighbors"] = None

    return result, None


def _find_engrad(job_dir: Path) -> Path | None:
    """Locate the .engrad file in a job directory."""
    engrad = job_dir / "orca.engrad"
    if engrad.exists():
        return engrad

    matches = list(job_dir.glob("*.engrad"))
    if matches:
        return matches[0]

    return None


def _job_base_tuple(job) -> tuple:
    """Build the job-metadata portion of an insert row."""
    return (
        job.id,
        job.orig_index,
        job.elements,
        job.natoms,
        job.charge,
        job.spin,
        job.status.value,
        job.job_dir,
        job.final_energy,
        job.max_forces,
        job.scf_steps,
        job.wall_time,
        job.n_cores,
        job.fail_count,
    )


def extract_forces_from_db(
    db_path: str | Path,
    output_path: str | Path,
    root_dir: str | Path | None = None,
    workers: int = 1,
    neighbor_cutoff: float = 4.0,
    logger: logging.Logger | None = None,
) -> int:
    """Extract force baselines for all completed jobs and write to a new DB.

    Every completed job is written. Jobs where force extraction fails
    have NULL force columns and a ``parse_note`` explaining the reason.

    Args:
        db_path: Path to the source workflow SQLite database.
        output_path: Path for the output SQLite database.
        root_dir: Optional override root directory for job folders.
            If not set, uses the job_dir paths stored in the DB.
        workers: Number of parallel workers for I/O (default: 1).
        neighbor_cutoff: Distance cutoff in Angstroms for identifying
            actinide neighbors (default: 4.0).
        logger: Logger instance for status and debug messages.

    Returns:
        Number of rows written to the output database.
    """
    if logger is None:
        logger = _setup_logger("force_baselines")

    with ArchitectorWorkflow(db_path) as wf:
        completed = wf.get_jobs_by_status(JobStatus.COMPLETED, include_geometry=False)

    if not completed:
        logger.warning("No completed jobs found in database.")
        return 0

    logger.info("Found %d completed jobs. Extracting forces...", len(completed))

    def _process_job(job) -> tuple:
        """Return (job, info_dict_or_None, parse_note_or_None)."""
        base = _job_base_tuple(job)

        job_dir_str = job.job_dir
        if not job_dir_str:
            logger.debug("job %d: no job_dir in DB", job.id)
            return base + _NULL_FORCE_FIELDS + ("no_job_dir",)

        job_dir = Path(job_dir_str)
        if root_dir is not None:
            job_dir = Path(root_dir) / job_dir.name

        if not job_dir.exists():
            logger.debug("job %d: dir not found: %s", job.id, job_dir)
            return base + _NULL_FORCE_FIELDS + ("dir_not_found",)

        engrad_path = _find_engrad(job_dir)
        if engrad_path is None:
            logger.debug("job %d: no engrad in %s", job.id, job_dir)
            return base + _NULL_FORCE_FIELDS + ("no_engrad",)

        info, note = extract_force_info(
            engrad_path, neighbor_cutoff=neighbor_cutoff, logger=logger
        )
        if info is None:
            logger.debug("job %d: parse note: %s", job.id, note)
            return base + _NULL_FORCE_FIELDS + (note,)

        return base + (
            info["max_force"],
            info["mean_force"],
            info.get("total_energy_engrad"),
            info.get("actinide_elements"),
            info.get("actinide_max_force"),
            info.get("actinide_mean_force"),
            info.get("actinide_forces_detail"),
            info.get("non_actinide_max_force"),
            info.get("non_actinide_mean_force"),
            info.get("non_actinide_forces_detail"),
            info.get("has_actinide"),
            info.get("neighbor_indices"),
            info.get("neighbor_elements"),
            info.get("neighbor_max_force"),
            info.get("neighbor_mean_force"),
            info.get("neighbor_forces_detail"),
            info.get("neighbor_cutoff"),
            info.get("n_neighbors"),
            None,  # parse_note = None means success
        )

    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_process_job, completed))
    else:
        rows = [_process_job(job) for job in completed]

    # Write output database
    out_conn = sqlite3.connect(str(output_path))
    out_conn.execute(_CREATE_TABLE)
    out_conn.executemany(_INSERT_ROW, rows)
    out_conn.commit()

    # Log parse note breakdown
    note_counts: dict[str | None, int] = {}
    # parse_note is the last element in each row
    for row in rows:
        note = row[-1]
        note_counts[note] = note_counts.get(note, 0) + 1

    ok_count = note_counts.pop(None, 0)
    logger.info(
        "Wrote %d rows (%d with forces, %d without) to %s",
        len(rows),
        ok_count,
        len(rows) - ok_count,
        output_path,
    )
    if note_counts:
        logger.info(
            "Parse failures: %s",
            ", ".join(f"{k}={v}" for k, v in sorted(note_counts.items())),
        )

    out_conn.close()
    return len(rows)


def print_summary(db_path: str | Path) -> None:
    """Print summary statistics from a force baselines database.

    Args:
        db_path: Path to the force baselines SQLite database.
    """
    conn = sqlite3.connect(str(db_path))

    total = conn.execute("SELECT COUNT(*) FROM force_baselines").fetchone()[0]
    with_forces = conn.execute(
        "SELECT COUNT(*) FROM force_baselines WHERE max_force IS NOT NULL"
    ).fetchone()[0]
    without_forces = total - with_forces

    print("\n--- Force Baselines Summary ---")
    print(
        f"Total jobs: {total}  |  with forces: {with_forces}  |  without: {without_forces}"
    )

    if without_forces > 0:
        note_rows = conn.execute(
            "SELECT parse_note, COUNT(*) FROM force_baselines "
            "WHERE parse_note IS NOT NULL GROUP BY parse_note ORDER BY COUNT(*) DESC"
        ).fetchall()
        if note_rows:
            print("Parse notes: " + ", ".join(f"{n}={c}" for n, c in note_rows))

    # Energy statistics (available even when force extraction fails)
    energy_row = conn.execute(
        "SELECT COUNT(*), MIN(final_energy), MAX(final_energy), AVG(final_energy) "
        "FROM force_baselines WHERE final_energy IS NOT NULL"
    ).fetchone()
    engrad_energy_row = conn.execute(
        "SELECT COUNT(*), MIN(total_energy_engrad), MAX(total_energy_engrad), "
        "AVG(total_energy_engrad) "
        "FROM force_baselines WHERE total_energy_engrad IS NOT NULL"
    ).fetchone()

    if energy_row[0] > 0:
        print(f"\nFinal energy (DB) -- {energy_row[0]} jobs")
        print(
            f"  min: {energy_row[1]:.6f}  max: {energy_row[2]:.6f}  "
            f"avg: {energy_row[3]:.6f} Eh"
        )
    if engrad_energy_row[0] > 0:
        print(f"Engrad energy     -- {engrad_energy_row[0]} jobs")
        print(
            f"  min: {engrad_energy_row[1]:.6f}  max: {engrad_energy_row[2]:.6f}  "
            f"avg: {engrad_energy_row[3]:.6f} Eh"
        )

    # Check for mismatches between DB and engrad energies
    mismatch_row = conn.execute(
        "SELECT COUNT(*) FROM force_baselines "
        "WHERE final_energy IS NOT NULL AND total_energy_engrad IS NOT NULL "
        "AND ABS(final_energy - total_energy_engrad) > 1e-6"
    ).fetchone()
    if mismatch_row[0] > 0:
        print(
            f"  WARNING: {mismatch_row[0]} jobs have DB/engrad energy mismatch (>1e-6 Eh)"
        )

    if with_forces == 0:
        conn.close()
        return

    row = conn.execute(
        "SELECT MIN(max_force), MAX(max_force), AVG(max_force), "
        "MIN(mean_force), MAX(mean_force), AVG(mean_force) "
        "FROM force_baselines WHERE max_force IS NOT NULL"
    ).fetchone()
    print(f"\nMax force  -- min: {row[0]:.6f}  max: {row[1]:.6f}  avg: {row[2]:.6f}")
    print(f"Mean force -- min: {row[3]:.6f}  max: {row[4]:.6f}  avg: {row[5]:.6f}")

    # Actinide stats
    act_count = conn.execute(
        "SELECT COUNT(*) FROM force_baselines WHERE has_actinide = 1"
    ).fetchone()[0]
    no_act_count = conn.execute(
        "SELECT COUNT(*) FROM force_baselines "
        "WHERE has_actinide = 0 AND max_force IS NOT NULL"
    ).fetchone()[0]
    print(f"\nJobs with actinides: {act_count}  |  without: {no_act_count}")

    if act_count > 0:
        act_row = conn.execute(
            "SELECT MIN(actinide_max_force), MAX(actinide_max_force), "
            "AVG(actinide_max_force), AVG(actinide_mean_force) "
            "FROM force_baselines WHERE has_actinide = 1"
        ).fetchone()
        print(
            f"Actinide max force  -- min: {act_row[0]:.6f}  "
            f"max: {act_row[1]:.6f}  avg: {act_row[2]:.6f}"
        )
        print(f"Actinide mean force -- avg: {act_row[3]:.6f}")

    # Non-actinide stats
    non_act_row = conn.execute(
        "SELECT MIN(non_actinide_max_force), MAX(non_actinide_max_force), "
        "AVG(non_actinide_max_force), AVG(non_actinide_mean_force) "
        "FROM force_baselines WHERE non_actinide_max_force IS NOT NULL"
    ).fetchone()
    if non_act_row[0] is not None:
        print(
            f"\nNon-actinide max force  -- min: {non_act_row[0]:.6f}  "
            f"max: {non_act_row[1]:.6f}  avg: {non_act_row[2]:.6f}"
        )
        print(f"Non-actinide mean force -- avg: {non_act_row[3]:.6f}")

    # Actinide-neighbor stats
    nbr_count = conn.execute(
        "SELECT COUNT(*) FROM force_baselines "
        "WHERE n_neighbors IS NOT NULL AND n_neighbors > 0"
    ).fetchone()[0]
    if nbr_count > 0:
        cutoff_row = conn.execute(
            "SELECT DISTINCT neighbor_cutoff FROM force_baselines "
            "WHERE neighbor_cutoff IS NOT NULL"
        ).fetchall()
        cutoff_str = ", ".join(f"{r[0]:.1f}" for r in cutoff_row)
        print(f"\nActinide neighbors (cutoff: {cutoff_str} A): {nbr_count} jobs")
        nbr_row = conn.execute(
            "SELECT MIN(neighbor_max_force), MAX(neighbor_max_force), "
            "AVG(neighbor_max_force), AVG(neighbor_mean_force), "
            "MIN(n_neighbors), MAX(n_neighbors), AVG(n_neighbors) "
            "FROM force_baselines WHERE n_neighbors > 0"
        ).fetchone()
        print(
            f"Neighbor max force  -- min: {nbr_row[0]:.6f}  "
            f"max: {nbr_row[1]:.6f}  avg: {nbr_row[2]:.6f}"
        )
        print(f"Neighbor mean force -- avg: {nbr_row[3]:.6f}")
        print(
            f"Neighbors per job   -- min: {nbr_row[4]}  "
            f"max: {nbr_row[5]}  avg: {nbr_row[6]:.1f}"
        )

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-atom force baselines from completed ORCA jobs."
    )
    parser.add_argument("db_path", help="Path to workflow SQLite database.")
    parser.add_argument(
        "--root-dir",
        default=None,
        help="Override root directory for job folders.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="force_baselines.db",
        help="Output SQLite database path (default: force_baselines.db).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel workers for I/O (default: 4).",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary statistics after extraction.",
    )
    parser.add_argument(
        "--log-name",
        default="force_baselines",
        help="Logger name (default: force_baselines).",
    )
    parser.add_argument(
        "--neighbor-cutoff",
        type=float,
        default=4.0,
        help="Distance cutoff in Angstroms for actinide neighbors (default: 4.0).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    level = logging.DEBUG if args.debug else logging.INFO
    logger = _setup_logger(args.log_name, level=level)

    count = extract_forces_from_db(
        db_path=args.db_path,
        output_path=args.output,
        root_dir=args.root_dir,
        workers=args.workers,
        neighbor_cutoff=args.neighbor_cutoff,
        logger=logger,
    )

    if count == 0:
        sys.exit(1)

    if args.summary:
        print_summary(args.output)


if __name__ == "__main__":
    main()
