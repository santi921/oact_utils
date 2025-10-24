from spyrmsd.rmsd import rmsd
import matplotlib.pyplot as plt
import os
from typing import Any, List, Dict, Tuple
from .an66 import dict_to_numpy, elements_to_atomic_numbers
from .create import read_xyz_single_file


# TODO: get number of cores and job run time from orca output files


def get_rmsd_start_final(name: str, dict_geoms: Dict[str, Any], root_dir: str) -> Tuple[float, List[float]]:
    initial_geom = dict_geoms[name]

    folder_results = f"{root_dir}/{name}_done"
    xyz_output = f"{folder_results}/{name}_orca.xyz"
    traj_output = f"{folder_results}/{name}_orca_trj.xyz"

    #read lines from traj_output that starts with Coordinates
    with open(traj_output, "r") as f:
        lines = f.readlines()
    lines_coords = [i for i, line in enumerate(lines) if line.startswith("Coordinates")]
    energies = [float(lines[i].strip().split()[-1]) for i in lines_coords]


    atoms, _ = read_xyz_single_file(xyz_output)
    elements, coords = dict_to_numpy(atoms)
    elements_ref, coords_ref = dict_to_numpy(initial_geom)

    atomic_numbers = elements_to_atomic_numbers(elements)
    atomic_numbers_ref = elements_to_atomic_numbers(elements_ref)


    return rmsd(
        coords, 
        coords_ref,
        atomic_numbers,
        atomic_numbers_ref
    ), energies

def get_geo_forces(name: str, root_dir: str) -> List[Dict[str, float]]:
    list_info = []

    folder_results = f"{root_dir}/{name}_done"

    # iterate through files in folder_results, find one ending in .out or log 
    for file in os.listdir(folder_results):
        if file.endswith("logs"):
            output_file = os.path.join(folder_results, file)
            break

    # read output_file, find lines between
    
    with open(output_file, "r") as f:
        lines = f.readlines()
    
    trigger = "Geometry convergence"    
    trigger_end = "........................................................"

    info_block_tf = False
    for i, line in enumerate(lines):
        
        if trigger in line.strip():
            info_dict = {}
            info_block_tf = True
        
        if info_block_tf: 
            if line.split()[0].lower() == "rms" and line.split()[1].lower() == "gradient":
                info_dict["RMS_Gradient"] = float(line.split()[2])
            if line.split()[0].lower() == "max" and line.split()[1].lower() == "gradient":
                info_dict["Max_Gradient"] = float(line.split()[2])
        
        if trigger_end in line.strip():
            info_block_tf = False
            list_info.append(info_dict)
    
    return list_info
