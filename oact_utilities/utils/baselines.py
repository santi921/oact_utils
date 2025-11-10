from typing import Any, List, Dict, Tuple
import pandas as pd
from ase import Atoms
import numpy as np


def process_geometry_file(
    file: str, ase_format_tf: bool = False
) -> Dict[str, List[Dict[str, Any]]]:
    if ase_format_tf:
        syms_list = []
        coords_list = []

    with open(file, "r") as f:
        lines = f.readlines()

    dict_geoms = {}
    for ind, line in enumerate(lines):

        # check if line.split has length < 2
        if len(line.split()) < 2:
            # if the line is just a number skip
            if line.strip().isdigit():
                continue
            if line.strip() == "":
                continue

            if ase_format_tf and ind != 0:
                dict_geoms[molecule] = Atoms(symbols=syms_list, positions=coords_list)
                syms_list = []
                coords_list = []

            molecule = line.split()[-1]

            if not ase_format_tf:
                dict_geoms[molecule] = []

        elif line[0].isalpha():
            parts = line.split()
            atom = parts[0]
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])

            if ase_format_tf:
                syms_list.append(atom)
                coords_list.append([x, y, z])

            else:
                dict_geoms[molecule].append({"element": atom, "x": x, "y": y, "z": z})

        if ase_format_tf:
            dict_geoms[molecule] = Atoms(symbols=syms_list, positions=coords_list)
    return dict_geoms


def process_multiplicity_file(file: str) -> pd.DataFrame:
    with open(file, "r") as f:
        lines = f.readlines()

    # iterate through lines_cleaned, first word is molecule, second is zpve, third is multiplicity

    data = {}
    for line in lines:
        # format is name, charge=0, mult=1
        if line.strip() == "":
            continue
        parts = line.split(",")
        molecule = parts[0].strip()
        charge_part = parts[1].strip()
        mult_part = parts[2].strip()
        charge = int(charge_part.split("=")[1])
        multiplicity = int(mult_part.split("=")[1])
        data[molecule] = {"charge": charge, "multiplicity": multiplicity}
    return data


def dict_to_numpy(atoms: List[Dict[str, Any]]) -> Tuple[List[str], Any]:

    elements = [atom["element"] for atom in atoms]
    coords = np.array([[atom["x"], atom["y"], atom["z"]] for atom in atoms])
    return elements, coords
