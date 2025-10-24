import os 
from periodictable import elements as ptelements

from .create import write_orca_input
from .an66 import process_geometry_file, process_multiplicity_file, dict_to_numpy

def fetch_actinides(): 
    return ["Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"]

def elements_to_atomic_numbers(elements: list[str]) -> list[int]:
    atomic_numbers = [ptelements.symbol(e).number for e in elements]
    return atomic_numbers

def read_xyz_single_file(xyz_file: str) -> tuple[list[dict[str, float | str]], str]:
    with open(xyz_file, "r") as f:
        lines = f.readlines()
    
    num_atoms = int(lines[0].strip())
    comment = lines[1].strip()
    atoms = []
    for line in lines[2:2+num_atoms]:
        parts = line.split()
        element = parts[0]
        x = float(parts[1])
        y = float(parts[2])
        z = float(parts[3])
        atoms.append({
            "element": element,
            "x": x,
            "y": y,
            "z": z
        })
    return atoms, comment


def read_template(template_file: str) -> list[str]:
    with open(template_file, "r") as f:
        lines = f.readlines()

    # remove lines that start with #* and * and lines in between
    lines_cleaned_template = []
    in_block = False
    for line in lines:
        if not line.startswith("#") and not line.startswith("\n"):
            lines_cleaned_template.append(line)
    return lines_cleaned_template

def write_orca_input(
    name: str,
    root_folder: str,
    dict_geoms: dict[str, list[dict[str, float | str]]],
    df_multiplicity,
    lines_cleaned_template: list[str],
    charge: int = 0,
    actinide_basis: str = "ma-def-TZVP",
    actinide_ecp: str | None = None,
    non_actinide_basis: str = "def2-TZVPD",
    two_step: str | None = None
) -> None:

    _, _, spin = df_multiplicity[df_multiplicity["molecule"]==name].iloc[0].tolist()
    element_list = [element["element"] for element in dict_geoms[name]]
    element_set = set(element_list)
    print(f"Writing ORCA input for {name} with charge {charge} and spin {spin}")
    actinide_list = fetch_actinides()
    # write lines to list first 
    lines_to_write = []
    lines_to_write.append(f"%basis\n")
    for element in element_set:        
        if element in actinide_list:
            if os.path.isfile(actinide_basis):
                lines_to_write.append(f"  GTOName      = \"{actinide_basis}\"      # read orbital basis\n")
            else:
                lines_to_write.append(f"  NewGTO {element} \"{actinide_basis}\" end\n")
            
            if actinide_ecp is not None:
                lines_to_write.append(f"  NewECP {element} \"{actinide_ecp}\" end\n")
        else:
            if os.path.isfile(non_actinide_basis):
                lines_to_write.append(f"  GTOName      = \"{non_actinide_basis}\"      # read orbital basis\n")
            else:
                lines_to_write.append(f"  NewGTO {element} \"{non_actinide_basis}\" end\n")

    lines_to_write.append(f"end\n\n")

    lines_to_write.append(f"* xyz {charge} {spin}\n")
    for atom in dict_geoms[name]:
        element = atom["element"]
        x = atom["x"]
        y = atom["y"]
        z = atom["z"]
        lines_to_write.append(f"{element}\t{x:.6f}\t{y:.6f}\t{z:.6f}\n")

    lines_to_write.append("*\n")

    # create folder if it does not exist
    if not os.path.exists(f"{root_folder}/{name}_done") and not os.path.exists(f"{root_folder}/{name}_failed_done"):   
        if not os.path.exists(f"{root_folder}/{name}"):         
            os.makedirs(f"{root_folder}/{name}")
        folder_to_use = f"{root_folder}/{name}"        
    
    # write to file
    if two_step is not None:
        if two_step == "loose": 
            file_name = f"{folder_to_use}/omol_loose.inp"
        elif two_step == "tight":
            file_name = f"{folder_to_use}/omol_tight.inp"
    else:
        file_name = f"{folder_to_use}/{name}_orca.inp"

    with open(file_name, "w") as f:
        for line in lines_cleaned_template:
                f.write(line)
        f.write("\n")
        for line in lines_to_write:
            f.write(line)



def write_flux(
    template_file: str,
    root_dir: str,
    two_step: bool = False
) -> None:
    
    with open(template_file, "r") as f:
        lines = f.readlines()
    

    # remove lines that start with #* and * and lines in between
    lines_cleaned_template = []
    for line in lines:
        if not line.startswith("#") and not line.startswith("\n"):
            lines_cleaned_template.append(line)

    # create folder if it does not exist
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # go through each subfolder in root_directory and write a flux job for each, scan for .inp files and add them to last line of template
    for folder in os.listdir(root_dir):
        folder_to_use = os.path.join(root_dir, folder)
        if os.path.isdir(folder_to_use):
            if two_step: 
                inp_files_loose = [f for f in os.listdir(folder_to_use) if f.endswith("omol_loose.inp")]
                inp_files_tight = [f for f in os.listdir(folder_to_use) if f.endswith("omol_tight.inp")]
                inp_files_loose_full_path = [os.path.join(folder_to_use, f) for f in inp_files_loose]
                inp_files_tight_full_path = [os.path.join(folder_to_use, f) for f in inp_files_tight]
                inp_files_loose_line = " ".join(inp_files_loose_full_path)
                inp_files_tight_line = " ".join(inp_files_tight_full_path)
                lines_cleaned_template_modified = lines_cleaned_template.copy()
                # rm \n from last line if present
                if lines_cleaned_template_modified[-1].endswith("\n"):
                    lines_cleaned_template_modified[-1] = lines_cleaned_template_modified[-1][:-1]
                lines_cleaned_template_modified_loose = lines_cleaned_template_modified.copy()
                lines_cleaned_template_modified_tight = lines_cleaned_template_modified.copy()

                lines_cleaned_template_modified_loose[-1] = lines_cleaned_template_modified_loose[-1] + f" {inp_files_loose_line}\n"
                lines_cleaned_template_modified_tight[-1] = lines_cleaned_template_modified_tight[-1] + f" {inp_files_tight_line}\n"
                file_name_loose = f"{folder_to_use}/flux_job_loose.inp"
                file_name_tight = f"{folder_to_use}/flux_job_tight.inp"
                with open(file_name_loose, "w") as f:
                    for line in lines_cleaned_template_modified_loose:
                            f.write(line)
                with open(file_name_tight, "w") as f:
                    for line in lines_cleaned_template_modified_tight:
                            f.write(line)
            else:

                inp_files = [f for f in os.listdir(folder_to_use) if f.endswith(".inp")]
                inp_files_full_path = [os.path.join(folder_to_use, f) for f in inp_files]
                inp_files_line = " ".join(inp_files_full_path)
                lines_cleaned_template_modified = lines_cleaned_template.copy()
                # rm \n from last line if present
                if lines_cleaned_template_modified[-1].endswith("\n"):
                    lines_cleaned_template_modified[-1] = lines_cleaned_template_modified[-1][:-1]
                lines_cleaned_template_modified[-1] = lines_cleaned_template_modified[-1] + f" {inp_files_line}\n"
            
                file_name = f"{folder_to_use}/flux_job.inp"
                with open(file_name, "w") as f:
                    for line in lines_cleaned_template_modified:
                            f.write(line)


def write_jobs(
    actinide_basis: str = "ma-def-TZVP",
    non_actinide_basis: str = "def2-TZVPD",
    actinide_ecp: str = "def-ECP",
    template_file: str = "template_orca.inp",
    root_dir: str = "./orca_jobs/",
    ref_geom_file: str = "/Users/santiagovargas/dev/data/ref_geoms.txt",
    ref_multiplicity_file: str = "/Users/santiagovargas/dev/data/ref_multiplicity.txt",
    two_step: str | None = None
) -> None:

    df_multiplicity = process_multiplicity_file(ref_multiplicity_file)
    dict_geoms = process_geometry_file(ref_geom_file)
    lines_cleaned_template = read_template(template_file)


    job_list = df_multiplicity['molecule'].tolist()
    for job in job_list:
        write_orca_input(
            name=job, 
            root_folder=root_dir, 
            dict_geoms=dict_geoms, 
            df_multiplicity=df_multiplicity, 
            lines_cleaned_template=lines_cleaned_template,
            charge=0,
            actinide_basis=actinide_basis,
            actinide_ecp=actinide_ecp,
            non_actinide_basis=non_actinide_basis,
            two_step=two_step
        )

