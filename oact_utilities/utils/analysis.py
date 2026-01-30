import os
import re
from pathlib import Path

import numpy as np
from ase.io.trajectory import TrajectoryReader
from spyrmsd.rmsd import rmsd

from oact_utilities.utils.an66 import dict_to_numpy
from oact_utilities.utils.create import (
    elements_to_atomic_numbers,
    read_geom_from_inp_file,
    read_xyz_single_file,
)
from oact_utilities.utils.status import (
    check_file_termination,
    check_job_termination,
    check_sella_complete,
    pull_log_file,
)


def get_rmsd_start_final(root_dir: str) -> tuple:
    """
    Calculate RMSD between initial and final geometries from a trajectory file.
    Also extract energies from the trajectory or log file.
    Args:
        root_dir (str): Directory containing the trajectory and input files.
    Returns:
        dict: {
            "rmsd": float,
            "energies_frames": List[float],
            "elements": List[str],
            "coords": np.ndarray
        }
    """
    # find xyz folder, traj file, and .inp file
    folder_results = root_dir
    files_in = os.listdir(folder_results)
    # find file with ".inp" in name
    file_density = [f for f in files_in if f.endswith(".inp")]
    # get the shortest name
    root_name = min(file_density, key=len).split(".inp")[0]

    xyz_output = os.path.join(folder_results, f"{root_name}.xyz")
    inp_file = os.path.join(folder_results, f"{root_name}.inp")
    traj_output = os.path.join(folder_results, f"{root_name}_trj.xyz")

    initial_geom = read_geom_from_inp_file(inp_file)
    # read lines from traj_output that starts with Coordinates
    # ccheck if traj_output exists

    if os.path.exists(traj_output):

        print(f"Reading trajectory from {traj_output} for RMSD calculation.")

        with open(traj_output) as f:
            lines = f.readlines()
        lines_coords = [
            i for i, line in enumerate(lines) if line.startswith("Coordinates")
        ]
        energies = [float(lines[i].strip().split()[-1]) for i in lines_coords]

    else:
        print(
            f"Trajectory file {traj_output} does not exist. Cannot compute RMSD @ ea. step."
        )

        log_file = pull_log_file(root_dir)
        energies = get_energy_from_log_file(log_file)

    if os.path.exists(xyz_output) is False:
        raise FileNotFoundError(
            f"XYZ output file {xyz_output} not found...reading positions from .inp file instead."
        )

    atoms, _ = read_xyz_single_file(xyz_output)
    elements, coords = dict_to_numpy(atoms)
    elements_ref, coords_ref = dict_to_numpy(initial_geom)

    atomic_numbers = elements_to_atomic_numbers(elements)
    atomic_numbers_ref = elements_to_atomic_numbers(elements_ref)

    dict_return = {
        "rmsd": rmsd(coords, coords_ref, atomic_numbers, atomic_numbers_ref),
        "energies_frames": energies,
        "elements_final": elements,
        "coords_final": coords,
        "elements_init": elements_ref,
        "coords_init": coords_ref,
    }
    return dict_return


def get_rmsd_between_traj_frames(traj_file: str) -> dict:
    # traj_file = root + "opt.traj"
    traj = TrajectoryReader(traj_file)
    atoms_init_traj = traj[0]
    atoms_final_traj = traj[-1]
    # elements_init_traj = [atom.symbol for atom in atoms_init_traj]
    coords_init_traj = atoms_init_traj.get_positions()
    elements_final_traj = [atom.symbol for atom in atoms_final_traj]
    coords_final_traj = atoms_final_traj.get_positions()

    # get rmsd between first and last frame
    from spyrmsd.rmsd import rmsd

    atomic_numbers = elements_to_atomic_numbers(elements_final_traj)
    rmsd_value = rmsd(
        coords_final_traj, coords_init_traj, atomic_numbers, atomic_numbers
    )
    # print energy at each frame

    energies_frames = []
    rms_forces_frames = []
    for _, frame in enumerate(traj):
        energy = frame.get_potential_energy()
        energies_frames.append(energy)
        force = frame.get_calculator().get_forces(frame)
        # compute rms force from numpy
        mean_squared_force = np.mean(force**2)
        rms_force = float(np.sqrt(mean_squared_force))
        rms_forces_frames.append(rms_force)

    ret_dict = {
        "rmsd_value": rmsd_value,
        "elements_final": elements_final_traj,
        "coords_final": coords_final_traj,
        "energies_frames": energies_frames,
        "rms_forces_frames": rms_forces_frames,
    }
    return ret_dict


def get_geo_forces(log_file: str) -> list:
    """
    Extract geometry optimization forces from log file.
    Args:
        log_file (str): Path to the log file.
    Returns:
        List[Dict[str, float]]: List of dictionaries with RMS and Max gradients.
    """

    list_info = []

    # read output_file, find lines between
    with open(log_file) as f:
        lines = f.readlines()

    trigger = "Geometry convergence"
    # trigger_end_a = (
    #    "-------------------------------------------------------------------------"
    # )
    # trigger_end_b = "........................................................"

    info_block_tf = False
    for _, line in enumerate(lines):

        if trigger in line.strip():
            info_dict = {}
            info_block_tf = True

        if info_block_tf:
            if len(line.split()) > 1:
                if (
                    line.split()[0].lower() == "rms"
                    and line.split()[1].lower() == "gradient"
                ):
                    info_dict["RMS_Gradient"] = float(line.split()[2])
                if (
                    line.split()[0].lower() == "max"
                    and line.split()[1].lower() == "gradient"
                ):
                    info_dict["Max_Gradient"] = float(line.split()[2])
                    info_block_tf = False
                    list_info.append(info_dict)
            else:
                continue

    return list_info


def parse_max_forces(output_file: str) -> float | None:
    """Extract the maximum force from an ORCA output file.

    Args:
        output_file: Path to ORCA .out file.

    Returns:
        Maximum force value (in Eh/Bohr), or None if not found.
    """
    try:
        with open(output_file) as f:
            lines = f.readlines()

        # Look for "MAX gradient" in geometry optimization output
        # Example line: "MAX gradient             0.00123456"
        max_force = None
        for line in reversed(lines):
            if "MAX gradient" in line:
                parts = line.split()
                try:
                    max_force = float(parts[-1])
                    break
                except (ValueError, IndexError):
                    continue

        return max_force

    except Exception:
        return None


def parse_scf_steps(output_file: str | Path) -> int | None:
    """Extract the number of SCF cycles from an ORCA output file.

    Args:
        output_file: Path to ORCA .out file.

    Returns:
        Total number of SCF iterations, or None if not found.
    """
    output_file = Path(output_file)
    if not output_file.exists():
        return None

    try:
        with open(output_file) as f:
            content = f.read()

        # Count SCF iterations
        # Look for lines like "SCF ITERATIONS"
        # Or count the number of times "SCF converged" appears
        scf_pattern = r"SCF ITERATIONS\s+(\d+)"
        matches = re.findall(scf_pattern, content)

        if matches:
            # Sum all SCF iterations if multiple geometry steps
            return sum(int(m) for m in matches)

        # Alternative: look for cycle numbers in SCF output
        # Example: "ITER       Energy         Delta-E        Max-DP      RMS-DP      [F,P]     Damp"
        #          "  0   -123.456789012   0.000000e+00  0.12345678  0.01234567  0.1234567  0.7000"
        cycle_pattern = r"^\s*(\d+)\s+-?\d+\.\d+\s+"
        cycles = re.findall(cycle_pattern, content, re.MULTILINE)
        if cycles:
            # Return the highest cycle number + 1 (since cycles start at 0)
            return max(int(c) for c in cycles) + 1

        return None

    except Exception:
        return None


""" Format  of orca.engrad file:
# Number of atoms
#
 148
#
# The current total energy in Eh
#
  -6038.704591758878
#
# The current gradient in Eh/bohr
#
      -0.001219869801
      -0.000908215297
       0.004618780904
       0.004660568634
#
# The atomic numbers and current coordinates in Bohr
#
  95     0.0000000    0.0000000    0.0000000
   7    -7.5442081    0.6697378   -0.0184305
"""


def get_engrad(engrad_file: str) -> dict:
    """
    Extract energy and gradient information from orca.engrad file.

    Also computes the maximum force magnitude from the gradient.

    Args:
        engrad_file (str): Path to the orca.engrad file.
    Returns:
        Dict with keys: total_energy_Eh, gradient_Eh_per_bohr, elements,
        coords_bohr, max_force_Eh_per_bohr
    """

    dict_info = {}
    with open(engrad_file) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "The current total energy in Eh" in line:
            energy = float(lines[i + 2].strip())
            dict_info["total_energy_Eh"] = energy

        if "The current gradient in Eh/bohr" in line:
            gradient = []
            j = i + 2
            while lines[j].strip() != "#":
                gradient.append(float(lines[j].strip()))
                j += 1
            dict_info["gradient_Eh_per_bohr"] = gradient

        if "The atomic numbers and current coordinates in Bohr" in line:
            coords = []
            elements = []
            j = i + 2
            while j < len(lines) and lines[j].strip() != "":
                parts = lines[j].strip().split()
                elements.append(int(parts[0]))
                coords.append([float(x) for x in parts[1:4]])
                j += 1
            dict_info["elements"] = elements
            dict_info["coords_bohr"] = coords

    # Compute max force from gradient
    gradient = dict_info.get("gradient_Eh_per_bohr")
    if gradient and len(gradient) > 0:
        try:
            gradient_arr = np.array(gradient)
            natoms = len(gradient_arr) // 3

            if len(gradient_arr) % 3 == 0 and natoms > 0:
                gradient_3d = gradient_arr.reshape((natoms, 3))
                force_magnitudes = np.linalg.norm(gradient_3d, axis=1)
                dict_info["max_force_Eh_per_bohr"] = float(np.max(force_magnitudes))
        except Exception:
            pass

    return dict_info


def find_timings_and_cores(log_file: str) -> list:
    """
    Extract number of processors and timing information from log file.
    Args:
        log_file (str): Path to the log file.
    Returns:
        Tuple[int, float]: Number of processors and total time in seconds.
    """
    # get dir of log_file

    termination_status = check_file_termination(log_file)

    if termination_status != 1:
        print(
            f"Job in {log_file} did not complete successfully. Cannot extract timings and cores."
        )
        return None, None

    # iterate through files_out until you hit line with "nprocs" line
    with open(log_file) as f:
        # don't read whole file into memory
        for line in f:
            if "nprocs" in line:
                nprocs = int(line.strip().split()[-1])
                # break after finding first occurrence
                break
        # get last line
        last_lines = f.readlines()[-10:-3]

        # format is TOTAL RUN TIME: 0 days 0 hours 3 minutes 16 seconds 840 msec
        time_dict = {}
        for line in last_lines:
            if "Sum of individual times" in line:
                time_dict["Total"] = float(line.split()[5])
            if "Startup" in line:
                time_dict["Startup"] = float(line.split()[3])
            if "SCF iterations " in line:
                time_dict["SCF_iterations"] = float(line.split()[3])
            if "Property" in line:
                time_dict["Property"] = float(line.split()[3])
            if "Gradient" in line:
                time_dict["Gradient"] = float(line.split()[4])
            if "Geometry" in line:
                time_dict["Geometry"] = float(line.split()[3])

    return nprocs, time_dict


def get_full_info_all_jobs(
    root_dir: str, flux_tf: bool, check_many: bool = False, verbose: bool = False
) -> list:
    """
    Get full performance and geometry info for all jobs in a root directory.
    Args:
        root_dir (str): Root directory containing job subdirectories.
        flux_tf (bool): Whether to look for flux- log files.
    Returns:
        Dict[str, Any]: Dictionary with performance and geometry info for each job.
    """
    perf_info = {}
    # iterate through every subfolder in root_dir
    for folder in os.listdir(root_dir):

        name = folder.split("_")[0]

        folder_to_use = os.path.join(root_dir, folder)

        if os.path.isdir(folder_to_use):
            status = check_job_termination(
                folder_to_use, check_many=check_many, flux_tf=flux_tf
            )
            if verbose:
                print(f"Status for job in {folder_to_use}: {status}")

            files = os.listdir(folder_to_use)
            # print("files: ", files)
            if flux_tf:
                files_out = [
                    f for f in files if f.startswith("flux") and f.endswith("out")
                ]
            else:
                files_out = [f for f in files if f.endswith("out")]
                if not files_out:
                    files_out = [f for f in files if f.endswith("logs")]

            if len(files_out) > 1 and isinstance(files_out, list):
                files_out.sort(
                    key=lambda x: os.path.getmtime(os.path.join(folder_to_use, x)),
                    reverse=True,
                )

            log_file = os.path.join(folder_to_use, files_out[0])

            perf_info[name] = {"status": status}

            """
            if status != 1:
                # print(f"Job in {folder_to_use} did not complete successfully. Skipping.")
                perf_info[name] = {
                    "nprocs": None,
                    "total_time_seconds": None,
                    "geo_forces": None,
                    "rmsd_start_final": None,
                    "energies_opt": None,
                    "elements_final": None,
                    "coords_final": None,
                }
                continue
            """
            try:
                nprocs, total_time_seconds = find_timings_and_cores(log_file)
                perf_info[name]["nprocs"] = nprocs
                perf_info[name]["total_time_seconds"] = total_time_seconds

            except (FileNotFoundError, ValueError, IndexError, KeyError) as e:
                perf_info[name]["nprocs"] = None
                perf_info[name]["total_time_seconds"] = None
                print(
                    f"Could not extract timings and cores for job in {folder_to_use}: {e}"
                )

            try:
                geo_forces = get_geo_forces(log_file=log_file)
                perf_info[name]["geo_forces"] = geo_forces

            except (FileNotFoundError, ValueError, IndexError, KeyError) as e:
                perf_info[name]["geo_forces"] = None
                print(
                    f"Could not extract geometry forces for job in {folder_to_use}: {e}"
                )

            geom_info = get_rmsd_start_final(folder_to_use)
            perf_info[name]["rmsd_start_final"] = geom_info.get("rmsd", None)
            perf_info[name]["energies_opt"] = geom_info.get("energies_frames", None)
            perf_info[name]["elements_final"] = geom_info.get("elements_final", None)
            perf_info[name]["coords_final"] = geom_info.get("coords_final", None)
            perf_info[name]["elements_init"] = geom_info.get("elements_init", None)
            perf_info[name]["coords_init"] = geom_info.get("coords_init", None)

    return perf_info


def actinide_first_neighbor_distances(
    elements: list[str],
    coords: np.ndarray,
    center_symbols: list[str] | None = None,
    neighbor_symbols: list[str] = ("O",),
    max_distance: float = 5.0,
) -> list[dict]:
    """
    Compute the distance to the first (nearest) neighbor for actinide center atoms.

    For each center element (by default the actinide series), find the nearest atom
    among `neighbor_symbols` within `max_distance` and return its distance.

    Args:
        elements: List of element symbols for the structure (e.g., ["U", "O", ...]).
        coords: Nx3 array of coordinates in Angstroms.
        center_symbols: Sequence of center element symbols to treat as centers. If
            None, defaults to the common actinide series.
        neighbor_symbols: Sequence of neighbor element symbols to consider (default: ("O",)).
        max_distance: Maximum distance (Angstrom) to search for a first neighbor.

    Returns:
        A list of dicts with keys: 'center_index', 'center_symbol', 'first_distance',
        'n_neighbors_within_cutoff', 'neighbor_index', 'neighbor_symbol'. If no
        neighbor is found within `max_distance`, 'first_distance' is np.nan and
        'n_neighbors_within_cutoff' is 0.
    """

    if center_symbols is None:
        center_symbols = [
            "Ac",
            "Th",
            "Pa",
            "U",
            "Np",
            "Pu",
            "Am",
            "Cm",
            "Bk",
            "Cf",
            "Es",
            "Fm",
            "Md",
            "No",
            "Lr",
        ]

    elements = [str(e) for e in elements]
    coords = np.asarray(coords, dtype=float)
    results: list[dict] = []

    for i, el in enumerate(elements):
        if el not in center_symbols:
            continue

        # distances to all other atoms
        dists = np.linalg.norm(coords - coords[i], axis=1)
        # exclude self
        mask = (dists > 1e-8) & (dists <= max_distance)
        neighbor_idxs = [
            j
            for j in range(len(elements))
            if mask[j] and elements[j] in neighbor_symbols
        ]

        if len(neighbor_idxs) == 0:
            results.append(
                {
                    "center_index": i,
                    "center_symbol": el,
                    "first_distance": float("nan"),
                    "n_neighbors_within_cutoff": 0,
                    "neighbor_index": None,
                    "neighbor_symbol": None,
                }
            )
            continue

        # find nearest neighbor
        nearest_idx = min(neighbor_idxs, key=lambda j: dists[j])
        results.append(
            {
                "center_index": i,
                "center_symbol": el,
                "first_distance": float(dists[nearest_idx]),
                "n_neighbors_within_cutoff": len(neighbor_idxs),
                "neighbor_index": int(nearest_idx),
                "neighbor_symbol": elements[nearest_idx],
            }
        )

    return results


def actinide_neighbor_mean_distances(*args, **kwargs):
    """Deprecated wrapper kept for backward compatibility.

    Use `actinide_first_neighbor_distances` which now provides first-neighbor
    distances. This wrapper will emit a DeprecationWarning.
    """
    import warnings

    warnings.warn(
        "actinide_neighbor_mean_distances is deprecated and now returns first-neighbor distances; use actinide_first_neighbor_distances instead",
        DeprecationWarning,
        stacklevel=1,
    )
    return actinide_first_neighbor_distances(*args, **kwargs)


def get_sp_info_all_jobs(root_dir: str, flux_tf: bool) -> list:
    """
    Get full performance and geometry info for all jobs in a root directory.
    Args:
        root_dir (str): Root directory containing job subdirectories.
        flux_tf (bool): Whether to look for flux- log files.
    Returns:
        Dict[str, Any]: Dictionary with performance and geometry info for each job.
    """
    perf_info = {}
    # iterate through every subfolder in root_dir
    for folder in os.listdir(root_dir):

        name = folder.split("_")[0]

        folder_to_use = os.path.join(root_dir, folder)

        if os.path.isdir(folder_to_use):
            status = check_job_termination(folder_to_use, flux_tf)

            if status != 1:
                # print(f"Job in {folder_to_use} did not complete successfully. Skipping.")
                perf_info[name] = {
                    "nprocs": None,
                    "total_time_seconds": None,
                    "energy": None,
                    "elements": None,
                    "coords_bohr": None,
                    "gradient": None,
                }
                continue

            files = os.listdir(folder_to_use)
            # print("files: ", files)
            if flux_tf:
                files_out = [
                    f for f in files if f.startswith("flux") and f.endswith("out")
                ]
            else:
                files_out = [f for f in files if f.endswith("out")]
                if not files_out:
                    files_out = [f for f in files if f.endswith("logs")]

            if len(files_out) > 1 and isinstance(files_out, list):
                files_out.sort(
                    key=lambda x: os.path.getmtime(os.path.join(folder_to_use, x)),
                    reverse=True,
                )

            log_file = os.path.join(folder_to_use, files_out[0])
            # just called orca.engrad
            engrad_file = os.path.join(folder_to_use, "orca.engrad")
            # print(f"Using log file: {log_file}")
            # info block
            nprocs, total_time_seconds = find_timings_and_cores(log_file)
            engrad = get_engrad(engrad_file=engrad_file)

            if nprocs is not None and total_time_seconds is not None:
                perf_info[name] = {
                    "nprocs": nprocs,
                    "total_time_seconds": total_time_seconds,
                    "energy": engrad["total_energy_Eh"],
                    "gradient": engrad["gradient_Eh_per_bohr"],
                    "elements": engrad["elements"],
                    "coords_bohr": engrad["coords_bohr"],
                }
            else:
                perf_info[name] = {
                    "nprocs": None,
                    "total_time_seconds": None,
                    "energy": engrad["total_energy_Eh"],
                    "gradient": engrad["gradient_Eh_per_bohr"],
                    "elements": engrad["elements"],
                    "coords_bohr": engrad["coords_bohr"],
                }
    return perf_info


def get_energy_from_log_file(log_file):
    """
    Extract energies from log file.
    Args:
        log_file (str): Path to the log file.
    Returns:
        List[float]: List of energies extracted from the log file.
    """
    energy_arr = []

    with open(log_file) as f:
        # don't load all into memory
        for line in f:
            if "Total Energy       :" in line:
                energy = float(line.strip().split()[3])
                energy_arr.append(energy)
    return energy_arr


def parse_final_energy(output_file: str) -> float | None:
    """Extract the final energy from an ORCA output file.

    Args:
        output_file: Path to ORCA .out file.

    Returns:
        Final energy in Hartree, or None if not found.
    """
    try:
        with open(output_file) as f:
            lines = f.readlines()

        # Look for "FINAL SINGLE POINT ENERGY"
        for line in reversed(lines):
            if "FINAL SINGLE POINT ENERGY" in line:
                parts = line.split()
                try:
                    return float(parts[-1])
                except (ValueError, IndexError):
                    continue

        return None

    except Exception:
        return None


def parse_job_metrics(
    job_dir: str | Path, unzip: bool = False
) -> dict[str, float | int | None]:
    """Extract multiple metrics from ORCA output files in a job directory.

    This function handles both regular and gzipped ORCA output files.
    It tries to extract max forces from .engrad file first (more reliable),
    then falls back to parsing text output.

    Args:
        job_dir: Path to job directory containing ORCA output.
        unzip: If True, look for gzipped files (e.g., quacc output).

    Returns:
        Dictionary with keys: max_forces, scf_steps, final_energy, success.
    """
    job_dir = Path(job_dir)

    try:
        # Find output file
        if unzip:
            import gzip
            import tempfile

            # Look for gzipped output
            gz_files = list(job_dir.glob("*.out.gz"))
            if not gz_files:
                return {
                    "max_forces": None,
                    "scf_steps": None,
                    "final_energy": None,
                    "success": False,
                }

            # Unzip to temp file
            with gzip.open(gz_files[0], "rt") as f_in:
                content = f_in.read()
                with tempfile.NamedTemporaryFile(
                    mode="w", delete=False, suffix=".out"
                ) as f_out:
                    f_out.write(content)
                    temp_path = f_out.name

            try:
                output_file = temp_path
                max_forces = parse_max_forces(output_file)
                scf_steps = parse_scf_steps(output_file)
                final_energy = parse_final_energy(output_file)

                # Try to get max forces from engrad file if available
                engrad_gz = list(job_dir.glob("*.engrad.gz"))
                if engrad_gz and max_forces is None:
                    with gzip.open(engrad_gz[0], "rt") as f_in:
                        with tempfile.NamedTemporaryFile(
                            mode="w", delete=False, suffix=".engrad"
                        ) as f_out:
                            f_out.write(f_in.read())
                            engrad_temp = f_out.name
                    try:
                        engrad_data = get_engrad(engrad_temp)
                        max_forces = engrad_data.get("max_force_Eh_per_bohr")
                    finally:
                        os.unlink(engrad_temp)

                # Check termination from content (for gzipped files)
                success = "ORCA TERMINATED NORMALLY" in content

            finally:
                os.unlink(temp_path)
        else:
            # Regular output file
            output_file = pull_log_file(str(job_dir))
            max_forces = parse_max_forces(output_file)
            scf_steps = parse_scf_steps(output_file)
            final_energy = parse_final_energy(output_file)

            # Try to get max forces from engrad file if available and not yet found
            engrad_file = job_dir / "orca.engrad"
            if engrad_file.exists() and max_forces is None:
                try:
                    engrad_data = get_engrad(str(engrad_file))
                    max_forces = engrad_data.get("max_force_Eh_per_bohr")
                except Exception:
                    pass

            # Check if job completed successfully
            termination = check_job_termination(str(job_dir))
            success = termination == 1

        return {
            "max_forces": max_forces,
            "scf_steps": scf_steps,
            "final_energy": final_energy,
            "success": success,
        }

    except Exception as e:
        return {
            "max_forces": None,
            "scf_steps": None,
            "final_energy": None,
            "success": False,
            "error": str(e),
        }


def parse_sella_log(sella_log_file, filter: bool = False) -> dict:
    """
    Check if a Sella optimization has completed successfully by examining the log file.

    Args:
        sella_log_file (str): Path to the Sella log file.
    Returns:
        bool: True if the optimization completed successfully, False otherwise.

    """

    # check if sella.log exists in root_dir

    if not os.path.exists(sella_log_file):
        return False
    # read sella.log and check for final forces
    with open(sella_log_file) as f:
        lines = f.readlines()
    forces = []
    steps = []
    energy = []

    for line in lines:
        if "Sella" in line:
            steps.append(int(line.split()[1]))
            forces.append(float(line.split()[4]))
            energy.append(float(line.split()[3]))

    if len(forces) == 0:
        return {}

    dict_ret = {"steps": steps, "forces": forces, "energy_frames": energy}

    return dict_ret


def get_full_info_all_jobs_sella(
    root_dir: str, verbose: bool = False, fmax: float = 0.05
) -> list:
    """
    Get full performance and geometry info for all jobs in a root directory.
    Args:
        root_dir (str): Root directory containing job subdirectories.
        flux_tf (bool): Whether to look for flux- log files.
    Returns:
        Dict[str, Any]: Dictionary with performance and geometry info for each job.
    """

    perf_info = {}
    # iterate through every subfolder in root_dir
    for folder in os.listdir(root_dir):

        name = folder.split("_")[0]

        folder_to_use = os.path.join(root_dir, folder)

        if os.path.isdir(folder_to_use):

            sella_log_tf = check_sella_complete(folder_to_use, fmax=fmax)
            dft_log_tf = check_job_termination(
                folder_to_use, check_many=False, flux_tf=False
            )

            if verbose:
                print(
                    f"Status for job in {folder_to_use}: DFT: {dft_log_tf}, Sella: {sella_log_tf}"
                )

            status = 1 if (sella_log_tf) else 0
            if dft_log_tf == -1:
                status = -1

            # check if the files "opt.traj", "sella.log", "orca.engrad" exist
            # files = os.listdir(folder_to_use)
            traj_file = os.path.join(folder_to_use, "opt.traj")
            sella_log = os.path.join(folder_to_use, "sella.log")
            engrad_file = os.path.join(folder_to_use, "orca.engrad")

            perf_info[name] = {"status": status}
            print(f"Processing job {name} in {folder_to_use} with status {status}.")

            try:
                rmsd_traj = get_rmsd_between_traj_frames(traj_file)
                perf_info[name] = {"status": status}
                perf_info[name]["rmsd_start_final"] = rmsd_traj["rmsd_value"]
                perf_info[name]["elements_final"] = rmsd_traj["elements_final"]
                perf_info[name]["coords_final"] = rmsd_traj["coords_final"]
                perf_info[name]["energies_opt"] = rmsd_traj["energies_frames"]
                perf_info[name]["rms_forces_frames"] = rmsd_traj["rms_forces_frames"]
            except (FileNotFoundError, ValueError, IndexError, KeyError, OSError) as e:
                perf_info[name]["rmsd_start_final"] = None
                perf_info[name]["elements_final"] = None
                perf_info[name]["coords_final"] = None
                perf_info[name]["energies_opt"] = None
                perf_info[name]["rms_forces_frames"] = None
                if verbose:
                    print(
                        f"Could not extract RMSD from traj for job in {folder_to_use}: {e}"
                    )

            try:
                dict_sella = parse_sella_log(sella_log)
                perf_info[name]["sella_steps"] = dict_sella.get("steps", [])
                perf_info[name]["sella_forces"] = dict_sella.get("forces", [])
                perf_info[name]["sella_energy_frames"] = dict_sella.get(
                    "energy_frames", []
                )
            except (FileNotFoundError, ValueError, IndexError, KeyError) as e:
                perf_info[name]["sella_steps"] = None
                perf_info[name]["sella_forces"] = None
                perf_info[name]["sella_energy_frames"] = None
                if verbose:
                    print(
                        f"Could not extract Sella log info for job in {folder_to_use}: {e}"
                    )

            try:
                dict_engrad = get_engrad(engrad_file)
                perf_info[name]["energy_final_Eh"] = dict_engrad.get(
                    "total_energy_Eh", None
                )
                perf_info[name]["gradient_final_Eh_per_bohr"] = dict_engrad.get(
                    "gradient_Eh_per_bohr", None
                )
                perf_info[name]["elements_engrad"] = dict_engrad.get("elements", None)
                perf_info[name]["coords_final_bohr"] = dict_engrad.get(
                    "coords_bohr", None
                )
            except (FileNotFoundError, ValueError, IndexError, KeyError) as e:
                perf_info[name]["energy_final_Eh"] = None
                perf_info[name]["gradient_final_Eh_per_bohr"] = None
                perf_info[name]["elements_engrad"] = None
                perf_info[name]["coords_final_bohr"] = None
                if verbose:
                    print(
                        f"Could not extract orca.engrad info for job in {folder_to_use}: {e}"
                    )
    return perf_info
