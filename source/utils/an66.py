import pandas as pd 
import os

from periodictable import elements as ptelements


def process_geometry_file(file):
    with open(file, "r") as f:
        lines = f.readlines()

    dict_geoms = {}
    for line in lines:
        if line.startswith("Geometry for"):
            molecule = line.split()[-1]
            dict_geoms[molecule] = []
        elif line[0].isalpha():
            parts = line.split()
            atom = parts[0]
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            dict_geoms[molecule].append({
                "element": atom,
                "x": x,
                "y": y,
                "z": z
            })
    return dict_geoms


def process_multiplicity_file(file):
    with open(file, "r") as f:
        lines = f.readlines()

    # if a line starts with a letter, if it starts with an integer it it is part of the previous line and should be merged
    lines_cleaned = []
    for line in lines:
        if line[0].isalpha():
            lines_cleaned.append(line.strip())
        else:
            lines_cleaned[-1] += "" + line.strip()
    # iterate through lines_cleaned, first word is molecule, second is zpve, third is multiplicity

    data = []
    for line in lines_cleaned:
        parts = line.split()
        molecule = parts[0]
        zpve = float(parts[1])
        multiplicity = int(parts[2])
        data.append({
            "molecule": molecule,
            "zpve": zpve,
            "multiplicity": multiplicity
        })

    df_multiplicity = pd.DataFrame(data)
    return df_multiplicity


def dict_to_numpy(atoms):
    import numpy as np
    elements = [atom["element"] for atom in atoms]
    coords = np.array([[atom["x"], atom["y"], atom["z"]] for atom in atoms])
    return elements, coords