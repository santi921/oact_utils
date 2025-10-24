import os 
from periodictable import elements as ptelements

from .create import write_orca_input
from .an66 import process_geometry_file, process_multiplicity_file, dict_to_numpy

def fetch_actinides(): 
     return["Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"]

def elements_to_atomic_numbers(elements):
    atomic_numbers = [ptelements.symbol(e).number for e in elements]
    return atomic_numbers

def read_xyz_single_file(xyz_file):
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


def read_template(template_file):
    with open(template_file, "r") as f:
        lines = f.readlines()

    # remove lines that start with #* and * and lines in between
    lines_cleaned_template = []
    in_block = False
    for line in lines:
        if not line.startswith("#") and not line.startswith("\n"):
            lines_cleaned_template.append(line)
    return lines_cleaned_template


def write_flux(
    template_file,
    root_dir,
    two_step=False
): 
    
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
            if two_step=True: 
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
        actinide_basis="ma-def-TZVP",
        non_actinide_basis="def2-TZVPD",
        actinide_ecp="def-ECP",
        template_file="template_orca.inp",
        root_dir="./orca_inputs/",
        ref_geom_file="ref_geoms.txt",
        ref_multiplicity_file="ref_multiplicity.txt", 
        two_step=None
):

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

