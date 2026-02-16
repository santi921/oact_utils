#!/usr/bin/env python3
"""
Simple helper to traverse a folder tree up to a given depth and check status
of `flux_job.flux` jobs it finds.

Usage:
    python check_multi_spin.py /path/to/root

The script accepts a single positional argument (root folder). Optional flags:
    --max-depth N    : maximum directory depth to traverse (default: 5)
    --verbose        : print extra diagnostic info
    --verbose-results: show which result keys are non-None for completed jobs
    --print-table    : print the full jobs table
    --truncate N     : truncate table values to N characters (default: 30)
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections.abc import Iterator
from time import time
from typing import Any

from oact_utilities.utils.analysis import (
    get_engrad,
    get_geo_forces,
    get_rmsd_between_traj_frames,
    get_rmsd_start_final,
    parse_mulliken_population,
    parse_sella_log,
)
from oact_utilities.utils.status import (
    check_job_termination,
    check_sella_complete,
    pull_log_file,
)


def iter_dirs_limited(root: str, max_depth: int) -> Iterator[str]:
    """Yield directory paths under `root` up to `max_depth` levels deep.

    Depth 0 yields `root` itself. Depth 1 yields immediate subdirectories, etc.
    """
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        return

    for current_dir, dirs, _ in os.walk(root):
        # compute relative depth
        rel = os.path.relpath(current_dir, root)
        if rel == ".":
            depth = 0
        else:
            depth = len(rel.split(os.sep))
        if depth > max_depth:
            # tell os.walk not to recurse deeper
            dirs[:] = []
            continue
        yield current_dir


def parse_info_from_path(path: str) -> dict[str, Any]:
    """Parse information from the path string in a robust way.

    Attempts to locate a part beginning with 'spin' and extracts the spin
    identifier plus the molecule name (the directory immediately before the
    spin directory when possible), category, and lot when present.
    Returns keys: 'lot', 'cat', 'name', 'spin'. Missing values are ''.
    """

    def sanitize_name(name_raw: str) -> str:
        # if the name ends with _TPS, _PBE, PBE0 remove
        for suffix in ["_TPS", "_PBE", "_PBE0", "_B3LYP", "_M06L"]:
            if name_raw.endswith(suffix):
                return name_raw[: -len(suffix)]
        return name_raw

    parts = [p for p in path.split(os.sep) if p]
    info: dict[str, Any] = {"lot": "", "cat": "", "name": "", "spin": ""}

    # find index of element that starts with 'spin'
    spin_idx = None
    for i, p in enumerate(parts[::-1]):
        if p.startswith("spin"):
            # convert reversed index to normal index
            spin_idx = len(parts) - 1 - i
            break

    if spin_idx is not None:
        info["spin"] = parts[spin_idx]
        if spin_idx - 1 >= 0:
            name_raw = parts[spin_idx - 1]
            info["name"] = sanitize_name(name_raw)
        if spin_idx - 2 >= 0:
            info["cat"] = parts[spin_idx - 2]
        if spin_idx - 3 >= 0:
            info["lot"] = parts[spin_idx - 3]
    else:
        # fallback: use last 4 parts if available
        if len(parts) >= 1:
            name_raw = parts[-1]
            info["name"] = sanitize_name(name_raw)
        if len(parts) >= 2:
            info["spin"] = parts[-2]
        if len(parts) >= 3:
            info["cat"] = parts[-3]
        if len(parts) >= 4:
            info["lot"] = parts[-4]

    return info


def truncate_str(value: Any, max_len: int = 30) -> str:
    """Truncate a string value for table display.

    Args:
        value: Value to truncate (will be converted to str).
        max_len: Maximum length before truncation.

    Returns:
        Truncated string with '...' if needed.
    """
    if value is None:
        return "N/A"
    s = str(value)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def _ensure_db(conn: sqlite3.Connection) -> None:
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE,
            lot TEXT,
            category TEXT,
            name TEXT,
            spin TEXT,
            status INTEGER,
            note TEXT,
            checked_at REAL,
            run_type TEXT,
            final_energy REAL,
            max_force REAL,
            rmsd_start_final REAL,
            has_final_geometry INTEGER DEFAULT 0,
            n_opt_steps INTEGER,
            final_rms_force REAL,
            final_coords TEXT,
            final_elements TEXT,
            mulliken_charges TEXT,
            mulliken_spins TEXT,
            loewdin_charges TEXT,
            loewdin_spins TEXT
        )
        """
    )
    # Add columns if they don't exist (for older databases)
    for col, col_type in [
        ("run_type", "TEXT"),
        ("final_energy", "REAL"),
        ("max_force", "REAL"),
        ("rmsd_start_final", "REAL"),
        ("has_final_geometry", "INTEGER DEFAULT 0"),
        ("n_opt_steps", "INTEGER"),
        ("final_rms_force", "REAL"),
        ("final_coords", "TEXT"),
        ("final_elements", "TEXT"),
        ("mulliken_charges", "TEXT"),
        ("mulliken_spins", "TEXT"),
        ("loewdin_charges", "TEXT"),
        ("loewdin_spins", "TEXT"),
    ]:
        try:
            c.execute(f"ALTER TABLE jobs ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass  # Column already exists
    conn.commit()


def extract_orca_analysis(
    job_dir: str, verbose_results: bool = False
) -> dict[str, Any]:
    """Extract analysis data for an ORCA direct run (no Sella).

    Uses get_rmsd_start_final and get_geo_forces for ORCA geom opt runs.

    Args:
        job_dir: Path to the job directory.
        verbose_results: If True, print which keys have non-None values.

    Returns:
        Dictionary with analysis data.
    """
    import numpy as np

    result: dict[str, Any] = {
        "run_type": "orca",
        "final_energy": None,
        "max_force": None,
        "rmsd_start_final": None,
        "has_final_geometry": 0,
        "n_opt_steps": None,
        "final_rms_force": None,
        "final_coords": None,
        "final_elements": None,
        "mulliken_charges": None,
        "mulliken_spins": None,
        "loewdin_charges": None,
        "loewdin_spins": None,
    }

    try:
        # Get RMSD and geometry info using the wave2 analysis approach
        geom_info = get_rmsd_start_final(job_dir)
        result["rmsd_start_final"] = geom_info.get("rmsd")

        # Extract final coordinates and elements
        coords_final = geom_info.get("coords_final")
        elements_final = geom_info.get("elements_final")

        if coords_final is not None:
            result["has_final_geometry"] = 1
            # Convert numpy array to list for JSON serialization
            if isinstance(coords_final, np.ndarray):
                coords_final = coords_final.tolist()
            result["final_coords"] = json.dumps(coords_final)

        if elements_final is not None:
            # Convert to list if needed
            if isinstance(elements_final, np.ndarray):
                elements_final = elements_final.tolist()
            result["final_elements"] = json.dumps(elements_final)

        energies = geom_info.get("energies_frames", [])
        if energies:
            result["final_energy"] = energies[-1]
            result["n_opt_steps"] = len(energies)
    except Exception:
        pass

    try:
        # Get forces from log file
        log_file = pull_log_file(job_dir)
        if log_file:
            geo_forces = get_geo_forces(log_file)
            if geo_forces:
                # Get final max and RMS forces
                last_forces = geo_forces[-1]
                result["max_force"] = last_forces.get("Max_Gradient")
                result["final_rms_force"] = last_forces.get("RMS_Gradient")

            # Parse Mulliken population analysis
            mulliken_data = parse_mulliken_population(log_file)
            if mulliken_data:
                # Store as JSON strings for database
                result["mulliken_charges"] = json.dumps(
                    mulliken_data.get("mulliken_charges", [])
                )
                result["mulliken_spins"] = json.dumps(
                    mulliken_data.get("mulliken_spins", [])
                )
                if mulliken_data.get("loewdin_charges"):
                    result["loewdin_charges"] = json.dumps(
                        mulliken_data.get("loewdin_charges", [])
                    )
                if mulliken_data.get("loewdin_spins"):
                    result["loewdin_spins"] = json.dumps(
                        mulliken_data.get("loewdin_spins", [])
                    )
    except Exception:
        pass

    if verbose_results:
        non_none_keys = [
            k for k, v in result.items() if v is not None and v != 0 and k != "run_type"
        ]
        if non_none_keys:
            print(f"  [ORCA] Analysis keys with data: {', '.join(non_none_keys)}")

    return result


def extract_sella_analysis(
    job_dir: str, verbose_results: bool = False
) -> dict[str, Any]:
    """Extract analysis data for a Sella optimizer run.

    Uses get_rmsd_between_traj_frames, parse_sella_log, and get_engrad.

    Args:
        job_dir: Path to the job directory.
        verbose_results: If True, print which keys have non-None values.

    Returns:
        Dictionary with analysis data. Energies in Hartree, forces in Eh/Bohr.
    """
    import numpy as np

    # Conversion factors: ASE/Sella use eV and Angstrom, ORCA uses Hartree and Bohr
    EV_TO_HARTREE = 1.0 / 27.211386
    # 1 eV/Å = 0.0194469 Eh/Bohr (1 Bohr = 0.529177 Å, 1 Eh = 27.211386 eV)
    EV_PER_ANG_TO_EH_PER_BOHR = 0.529177 / 27.211386  # ≈ 0.0194469

    result: dict[str, Any] = {
        "run_type": "sella",
        "final_energy": None,  # Always stored in Hartree
        "max_force": None,
        "rmsd_start_final": None,
        "has_final_geometry": 0,
        "n_opt_steps": None,
        "final_rms_force": None,
        "final_coords": None,
        "final_elements": None,
        "mulliken_charges": None,
        "mulliken_spins": None,
        "loewdin_charges": None,
        "loewdin_spins": None,
    }

    # Try to get trajectory info (preferred for Sella)
    traj_file = os.path.join(job_dir, "opt.traj")
    if os.path.exists(traj_file):
        try:
            traj_info = get_rmsd_between_traj_frames(traj_file)
            result["rmsd_start_final"] = traj_info.get("rmsd_value")

            # Extract final coordinates and elements
            coords_final = traj_info.get("coords_final")
            elements_final = traj_info.get("elements_final")

            if coords_final is not None:
                result["has_final_geometry"] = 1
                # Convert numpy array to list for JSON serialization
                if isinstance(coords_final, np.ndarray):
                    coords_final = coords_final.tolist()
                result["final_coords"] = json.dumps(coords_final)

            if elements_final is not None:
                # Convert to list if needed
                if isinstance(elements_final, np.ndarray):
                    elements_final = elements_final.tolist()
                result["final_elements"] = json.dumps(elements_final)

            # ASE trajectory energies are in eV - convert to Hartree
            energies = traj_info.get("energies_frames", [])
            if energies:
                result["final_energy"] = energies[-1] * EV_TO_HARTREE
            # ASE trajectory forces are in eV/Å - convert to Eh/Bohr
            rms_forces = traj_info.get("rms_forces_frames", [])
            if rms_forces:
                result["final_rms_force"] = rms_forces[-1] * EV_PER_ANG_TO_EH_PER_BOHR
        except Exception:
            pass

    # Try to get sella log data
    sella_log = os.path.join(job_dir, "sella.log")
    if os.path.exists(sella_log):
        try:
            sella_data = parse_sella_log(sella_log)
            if sella_data:
                steps = sella_data.get("steps", [])
                forces = sella_data.get("forces", [])
                if steps:
                    result["n_opt_steps"] = len(steps)
                if forces:
                    # Convert from eV/Å (Sella/ASE units) to Eh/Bohr (ORCA units)
                    result["max_force"] = forces[-1] * EV_PER_ANG_TO_EH_PER_BOHR
        except Exception:
            pass

    # Fallback: try engrad file for energy/forces and geometry if not from traj
    engrad_file = os.path.join(job_dir, "orca.engrad")
    if os.path.exists(engrad_file):
        try:
            engrad_data = get_engrad(engrad_file)

            if result["final_energy"] is None:
                result["final_energy"] = engrad_data.get("total_energy_Eh")

            if result["max_force"] is None:
                result["max_force"] = engrad_data.get("max_force_Eh_per_bohr")

            # Get geometry from engrad if not already from traj
            if result["final_coords"] is None:
                coords_bohr = engrad_data.get("coords_bohr")
                # convert to ang
                # coords_bohr is list of lists with coords in Bohr
                coords_ang = None
                if coords_bohr:
                    coords_ang = [
                        [x * 0.529177 for x in atom_coords]
                        for atom_coords in coords_bohr
                    ]
                elements = engrad_data.get("elements")

                if coords_ang:
                    result["has_final_geometry"] = 1
                    result["final_coords"] = json.dumps(coords_ang)

                if elements:
                    result["final_elements"] = json.dumps(elements)

            # Compute gradient norm as RMS force proxy
            gradient = engrad_data.get("gradient_Eh_per_bohr")
            if gradient and result["final_rms_force"] is None:
                grad_arr = np.array(gradient)
                natoms = len(grad_arr) // 3
                if natoms > 0:
                    grad_3d = grad_arr.reshape((natoms, 3))
                    result["final_rms_force"] = float(
                        np.sqrt(np.mean(np.linalg.norm(grad_3d, axis=1) ** 2))
                    )
        except Exception:
            pass

    # Parse Mulliken population analysis from ORCA output
    try:
        log_file = pull_log_file(job_dir)
        if log_file:
            mulliken_data = parse_mulliken_population(log_file)
            if mulliken_data:
                # Store as JSON strings for database
                result["mulliken_charges"] = json.dumps(
                    mulliken_data.get("mulliken_charges", [])
                )
                result["mulliken_spins"] = json.dumps(
                    mulliken_data.get("mulliken_spins", [])
                )
                if mulliken_data.get("loewdin_charges"):
                    result["loewdin_charges"] = json.dumps(
                        mulliken_data.get("loewdin_charges", [])
                    )
                if mulliken_data.get("loewdin_spins"):
                    result["loewdin_spins"] = json.dumps(
                        mulliken_data.get("loewdin_spins", [])
                    )
    except Exception:
        pass

    if verbose_results:
        non_none_keys = [
            k for k, v in result.items() if v is not None and v != 0 and k != "run_type"
        ]
        if non_none_keys:
            print(f"  [Sella] Analysis keys with data: {', '.join(non_none_keys)}")

    return result


def find_and_get_status(
    root: str,
    max_depth: int = 5,
    verbose: bool = False,
    *,
    print_table: bool = False,
    running_age_seconds: int = 3600,
    verbose_results: bool = False,
    truncate_len: int = 30,
) -> int:
    """Traverse `root` up to `max_depth` and check status of flux jobs.

    Returns the number of jobs processed.
    """
    db_path = os.path.join(root, "multi_spin_jobs.sqlite3")
    conn = sqlite3.connect(db_path)
    _ensure_db(conn)

    processed = 0
    for d in iter_dirs_limited(root, max_depth=max_depth):
        flux_file = os.path.join(d, "flux_job.flux")
        status = 0
        note = ""
        analysis_data: dict[str, Any] = {}
        is_sella_run = False

        if os.path.exists(flux_file):
            processed += 1
            # Check if this is a Sella run (has sella.log or opt.traj)
            sella_log = os.path.join(d, "sella.log")
            opt_traj = os.path.join(d, "opt.traj")
            is_sella_run = os.path.exists(sella_log) or os.path.exists(
                opt_traj
            )  # is this true??

            # default is remaining (0). Determine status in order:
            # 1) failed (-1) 2) completed (1) 3) running (2 if recent flux-*.out) 4) remaining (0)

            term = check_job_termination(d)

            if term == -1:
                if verbose:
                    print(f"Found failed job at {d}")
                status = -1
                note = "job_failed"

            elif term:
                if verbose:
                    print(f"Found completed job at {d}")
                status = 1
                note = "job_completed"
                # Extract analysis data for completed jobs
                if is_sella_run:
                    analysis_data = extract_sella_analysis(d, verbose_results)
                else:
                    analysis_data = extract_orca_analysis(d, verbose_results)

            elif check_sella_complete(d):
                if verbose:
                    print(f"Found completed job (sella) at {d}")
                status = 1
                note = "sella_complete"
                # Extract analysis data for sella-completed jobs
                analysis_data = extract_sella_analysis(d, verbose_results)

            else:
                # check for running indicator: recent flux-*.out read/write
                import glob

                out_files = glob.glob(os.path.join(d, "flux-*.out"))
                if out_files:
                    latest_out = max(out_files, key=os.path.getmtime)
                    latest_mtime = os.path.getmtime(latest_out)
                    if (time() - latest_mtime) < running_age_seconds:
                        if verbose:
                            print(
                                f"Found recent flux output {latest_out}; marking as running"
                            )
                        status = 2
                        note = f"running_recent:{os.path.basename(latest_out)}"

            path_data = parse_info_from_path(d)

            # This script is a status checker only; always record status

            # insert/update into DB with analysis data
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO jobs (path, lot, category, name, spin, status, note, checked_at,
                                  run_type, final_energy, max_force, rmsd_start_final,
                                  has_final_geometry, n_opt_steps, final_rms_force,
                                  final_coords, final_elements, mulliken_charges, mulliken_spins,
                                  loewdin_charges, loewdin_spins)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    lot=excluded.lot,
                    category=excluded.category,
                    name=excluded.name,
                    spin=excluded.spin,
                    status=excluded.status,
                    note=excluded.note,
                    checked_at=excluded.checked_at,
                    run_type=excluded.run_type,
                    final_energy=excluded.final_energy,
                    max_force=excluded.max_force,
                    rmsd_start_final=excluded.rmsd_start_final,
                    has_final_geometry=excluded.has_final_geometry,
                    n_opt_steps=excluded.n_opt_steps,
                    final_rms_force=excluded.final_rms_force,
                    final_coords=excluded.final_coords,
                    final_elements=excluded.final_elements,
                    mulliken_charges=excluded.mulliken_charges,
                    mulliken_spins=excluded.mulliken_spins,
                    loewdin_charges=excluded.loewdin_charges,
                    loewdin_spins=excluded.loewdin_spins
                """,
                (
                    d,
                    path_data.get("lot", ""),
                    path_data.get("cat", ""),
                    path_data.get("name", ""),
                    path_data.get("spin", ""),
                    status,
                    note,
                    time(),
                    analysis_data.get("run_type"),
                    analysis_data.get("final_energy"),
                    analysis_data.get("max_force"),
                    analysis_data.get("rmsd_start_final"),
                    analysis_data.get("has_final_geometry", 0),
                    analysis_data.get("n_opt_steps"),
                    analysis_data.get("final_rms_force"),
                    analysis_data.get("final_coords"),
                    analysis_data.get("final_elements"),
                    analysis_data.get("mulliken_charges"),
                    analysis_data.get("mulliken_spins"),
                    analysis_data.get("loewdin_charges"),
                    analysis_data.get("loewdin_spins"),
                ),
            )
            conn.commit()

    # Print a clean summary per molecule/name and spin state
    c = conn.cursor()
    c.execute(
        """
        SELECT name, category, spin,
               SUM(CASE WHEN status = 2 THEN 1 ELSE 0 END) AS running,
               SUM(CASE WHEN status = 0 THEN 1 ELSE 0 END) AS remaining,
               SUM(CASE WHEN status = 1 THEN 1 ELSE 0 END) AS done,
               SUM(CASE WHEN status = -1 THEN 1 ELSE 0 END) AS failed
        FROM jobs
        GROUP BY name, category, spin
        ORDER BY name, category, spin
        """
    )
    rows = c.fetchall()
    if rows:
        print("\nSummary of multi-spin jobs:\n")
        current_name = None
        current_cat = None
        for name, category, spin, running, remaining, done, failed in rows:
            if name != current_name or category != current_cat:
                if current_name is not None:
                    print("")
                print(f"Molecule: {name} (category: {category})")
                current_name = name
                current_cat = category
            print(
                f"  {spin:<10} -> running: {running:3d}  remaining: {remaining:3d}  done: {done:3d}  failed: {failed:3d}"
            )
    else:
        print("No job records found in DB.")

    # Print legend for status codes
    print(
        "\nStatus legend:\n  running: 2 (recent flux-*.out)\n  remaining: 0 (no run detected)\n  done: 1 (completed)\n  failed: -1 (failed)\n"
    )

    # Optionally print the entire jobs table for debugging (pretty formatted)
    if print_table:
        print("\nFull jobs table:\n")
        # print legend above the table for clarity
        print("Status legend: running=2, remaining=0, done=1, failed=-1\n")
        c2 = conn.cursor()
        c2.execute(
            """SELECT id, name, spin, status, run_type, final_energy, max_force,
                      rmsd_start_final, has_final_geometry, n_opt_steps, final_rms_force,
                      note, path
               FROM jobs ORDER BY name, spin"""
        )
        all_rows = c2.fetchall()
        headers = []
        if c2.description:
            headers = [d[0] for d in c2.description]

        # Apply truncation to all values
        truncated_rows = [
            tuple(truncate_str(val, truncate_len) for val in row) for row in all_rows
        ]

        # prefer tabulate if available, otherwise do basic aligned columns
        try:
            from tabulate import tabulate

            print(tabulate(truncated_rows, headers=headers, tablefmt="github"))
        except Exception:
            # fallback simple table with truncation applied
            if headers:
                # Compute widths respecting truncate_len
                widths = [
                    min(
                        truncate_len,
                        max(
                            len(str(h)),
                            max((len(str(r[i])) for r in truncated_rows), default=0),
                        ),
                    )
                    for i, h in enumerate(headers)
                ]
                header_line = " | ".join(
                    truncate_str(h, widths[i]).ljust(widths[i])
                    for i, h in enumerate(headers)
                )
                sep = "-+-".join("-" * widths[i] for i in range(len(widths)))
                print(header_line)
                print(sep)
                for r in truncated_rows:
                    print(" | ".join(str(r[i]).ljust(widths[i]) for i in range(len(r))))
            else:
                for r in truncated_rows:
                    print("\t".join([str(x) for x in r]))

    conn.close()
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check status of flux_job.flux jobs under a folder (depth-limited)"
    )
    parser.add_argument("root", help="Root folder to traverse")
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Max subdirectory depth to traverse (default: 5)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--verbose-results",
        "-vr",
        action="store_true",
        help="Show which analysis keys have non-None values for completed jobs",
    )
    parser.add_argument(
        "--print-table",
        action="store_true",
        help="Print the full jobs table with analysis results",
    )
    parser.add_argument(
        "--truncate",
        type=int,
        default=30,
        help="Truncate table values to N characters (default: 30)",
    )
    parser.add_argument(
        "--running-age-seconds",
        type=int,
        default=3600,
        help="Seconds to treat a flux-*.out as 'running' (default: 3600)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        parser.error(f"Root path does not exist or is not a directory: {args.root}")

    n = find_and_get_status(
        args.root,
        max_depth=args.max_depth,
        verbose=args.verbose,
        print_table=args.print_table,
        running_age_seconds=args.running_age_seconds,
        verbose_results=args.verbose_results,
        truncate_len=args.truncate,
    )
    print(f"Total flux jobs found: {n}")


if __name__ == "__main__":
    main()
