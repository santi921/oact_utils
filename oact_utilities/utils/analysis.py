from spyrmsd.rmsd import rmsd

import os
from typing import Any, List, Dict, Tuple
from oact_utilities.utils.an66 import dict_to_numpy
from oact_utilities.utils.create import (
    read_xyz_single_file,
    elements_to_atomic_numbers,
    read_geom_from_inp_file,
)

from oact_utilities.utils.status import (
    check_file_termination,
    check_job_termination,
    pull_log_file,
)


def get_rmsd_start_final(root_dir: str) -> Tuple[float, List[float]]:
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
    # find file with ".density" in name
    file_density = [f for f in files_in if f.endswith(".densities")]
    # get the shortest name
    root_name = min(file_density, key=len).split(".densities")[0]

    xyz_output = os.path.join(folder_results, f"{root_name}.xyz")
    inp_file = os.path.join(folder_results, f"{root_name}.inp")
    traj_output = os.path.join(folder_results, f"{root_name}_trj.xyz")

    initial_geom = read_geom_from_inp_file(inp_file)
    # read lines from traj_output that starts with Coordinates
    # ccheck if traj_output exists

    if os.path.exists(traj_output):
        print(
            f"Trajectory file {traj_output} does not exist. Cannot compute RMSD @ ea. step."
        )
        # find log file in folder_to_use

        print(f"Reading trajectory from {traj_output} for RMSD calculation.")
        with open(traj_output, "r") as f:
            lines = f.readlines()
        lines_coords = [
            i for i, line in enumerate(lines) if line.startswith("Coordinates")
        ]
        energies = [float(lines[i].strip().split()[-1]) for i in lines_coords]

    else:
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
    }
    return dict_return


def get_geo_forces(log_file: str) -> List[Dict[str, float]]:
    """
    Extract geometry optimization forces from log file.
    Args:
        log_file (str): Path to the log file.
    Returns:
        List[Dict[str, float]]: List of dictionaries with RMS and Max gradients.
    """

    list_info = []

    # read output_file, find lines between
    with open(log_file, "r") as f:
        lines = f.readlines()

    trigger = "Geometry convergence"
    trigger_end = "........................................................"

    info_block_tf = False
    for i, line in enumerate(lines):

        if trigger in line.strip():
            info_dict = {}
            info_block_tf = True

        if info_block_tf:
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

        if trigger_end in line.strip():
            info_block_tf = False
            list_info.append(info_dict)

    return list_info


def find_timings_and_cores(log_file: str) -> Tuple[int, float]:
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
    with open(log_file, "r") as f:
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
    root_dir: str, flux_tf: bool
) -> List[Tuple[str, int, float]]:
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
                continue

            # find log file in folder_to_use
            if flux_tf:
                # check for "flux-"
                # get all files that contains  flux-
                files_flux = [f for f in os.listdir(folder_to_use) if "flux-" in f]
                files_flux.sort(
                    key=lambda x: os.path.getmtime(os.path.join(folder_to_use, x)),
                    reverse=True,
                )
                log_file = os.path.join(folder_to_use, files_flux[0])

            else:
                log_file = [f for f in os.listdir(folder_to_use) if f.endswith("logs")]
                if len(log_file) == 0:
                    log_file = [
                        f for f in os.listdir(folder_to_use) if f.endswith(".out")
                    ]

            if len(log_file) == 0:
                # print(f"No log file found in {folder_to_use}. Skipping.")
                continue
            if len(log_file) > 1 and type(log_file) is list:
                # print(f"Multiple log files found in {folder_to_use}. Using the most recent one.")
                log_file.sort(
                    key=lambda x: os.path.getmtime(os.path.join(folder_to_use, x)),
                    reverse=True,
                )

            log_file = os.path.join(folder_to_use, log_file[0])
            print(f"Using log file: {log_file}")
            # info block
            nprocs, total_time_seconds = find_timings_and_cores(log_file)
            geo_forces = get_geo_forces(log_file=log_file)
            geom_info = get_rmsd_start_final(folder_to_use)

            if nprocs is not None and total_time_seconds is not None:
                perf_info[name] = {
                    "nprocs": nprocs,
                    "total_time_seconds": total_time_seconds,
                    "geo_forces": geo_forces,
                    "rmsd_start_final": geom_info["rmsd"],
                    "energies_opt": geom_info["energies_frames"],
                    "elements": geom_info["elements"],
                    "coords_final": geom_info["coords"],
                }
            else:
                perf_info[name] = {
                    "nprocs": None,
                    "total_time_seconds": None,
                    "geo_forces": None,
                    "rmsd_start_final": None,
                    "energies_opt": None,
                    "elements": None,
                    "coords_final": None,
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

    with open(log_file, "r") as f:
        # don't load all into memory
        for line in f:
            if "Total Energy       :" in line:
                energy = float(line.strip().split()[3])
                energy_arr.append(energy)
    return energy_arr
