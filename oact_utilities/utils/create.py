import os
from periodictable import elements as ptelements
from typing import Dict, Any
from oact_utilities.utils.an66 import process_geometry_file, process_multiplicity_file
from ase import Atoms

def fetch_actinides():
    return [
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


def elements_to_atomic_numbers(elements: list[str]) -> list[int]:
    atomic_numbers = [ptelements.symbol(e).number for e in elements]
    return atomic_numbers


def atomic_numbers_to_elements(atomic_numbers: list[int]) -> list[str]:
    elements = [ptelements[n].symbol for n in atomic_numbers]
    return elements


def read_geom_from_inp_file(
    inp_file: str, ase_format_tf: bool = False
) -> list[dict[str, float | str]]:
    with open(inp_file, "r") as f:
        lines = f.readlines()

    geom_start = False
    atoms = []

    if ase_format_tf:
        syms_list = []
        coords_list = []

    for line in lines:
        if line.strip().startswith("* xyz"):
            charge = line.strip().split()[2]
            spin = line.strip().split()[3]
            geom_start = True
            continue

        if line.strip().startswith("*xyz"):
            charge = line.strip().split()[1]
            spin = line.strip().split()[2]
            geom_start = True
            continue

        if geom_start:
            if line.strip().startswith("*"):
                break
            parts = line.split()
            element = parts[0]
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])

            if ase_format_tf:
                syms_list.append(element)
                coords_list.append([x, y, z])
            else:
                atoms.append({"element": element, "x": x, "y": y, "z": z})
    if ase_format_tf:
        # print(syms_list)
        # print(coords_list)
        atoms = Atoms(symbols=syms_list, positions=coords_list)
        atoms.charge = int(charge)
        atoms.spin = int(spin)

    return atoms


def read_xyz_single_file(xyz_file: str) -> tuple[list[dict[str, float | str]], str]:
    with open(xyz_file, "r") as f:
        lines = f.readlines()

    num_atoms = int(lines[0].strip())
    comment = lines[1].strip()
    atoms = []
    for line in lines[2 : 2 + num_atoms]:
        parts = line.split()
        element = parts[0]
        x = float(parts[1])
        y = float(parts[2])
        z = float(parts[3])
        atoms.append({"element": element, "x": x, "y": y, "z": z})
    return atoms, comment


def read_xyz_from_orca(xyz_file: str) -> Atoms:
    with open(xyz_file, "r") as f:
        lines = f.readlines()
    # find first non-empty line
    index_first_non_empty = 0
    for i, line in enumerate(lines):
        if line.strip() != "":
            index_first_non_empty = i
            break

    num_atoms = int(lines[index_first_non_empty].strip())
    comment = lines[index_first_non_empty + 1].strip()
    elem_list = []
    coord_list = []
    for line in lines[
        index_first_non_empty + 2 : index_first_non_empty + 2 + num_atoms
    ]:
        parts = line.split()
        element = parts[0]
        x = float(parts[1])
        y = float(parts[2])
        z = float(parts[3])
        elem_list.append(element)
        coord_list.append([float(x), float(y), float(z)])

    atoms_ase = Atoms(symbols=elem_list, positions=coord_list)
    return atoms_ase, comment


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
    cores: int = 8,
    actinide_basis: str = "ma-def-TZVP",
    actinide_ecp: str | None = None,
    non_actinide_basis: str = "def2-TZVPD",
    two_step: str | None = None,
) -> None:

    _, _, spin = df_multiplicity[df_multiplicity["molecule"] == name].iloc[0].tolist()
    element_list = [element["element"] for element in dict_geoms[name]]
    element_set = set(element_list)
    print(f"Writing ORCA input for {name} with charge {charge} and spin {spin}")
    actinide_list = fetch_actinides()
    # write lines to list first
    lines_to_write = []

    lines_to_write.append(f"%pal\n nprocs {cores} \nend\n\n")

    lines_to_write.append(f"%basis\n")
    for element in element_set:
        if element in actinide_list:
            if os.path.isfile(actinide_basis):
                lines_to_write.append(
                    f'  GTOName      = "{actinide_basis}"      # read orbital basis\n'
                )
            else:
                lines_to_write.append(f'  NewGTO {element} "{actinide_basis}" end\n')

            if actinide_ecp is not None:
                lines_to_write.append(f'  NewECP {element} "{actinide_ecp}" end\n')
        else:
            if os.path.isfile(non_actinide_basis):
                lines_to_write.append(
                    f'  GTOName      = "{non_actinide_basis}"      # read orbital basis\n'
                )
            else:
                lines_to_write.append(
                    f'  NewGTO {element} "{non_actinide_basis}" end\n'
                )

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
    if not os.path.exists(f"{root_folder}/{name}_done") and not os.path.exists(
        f"{root_folder}/{name}_failed_done"
    ):
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


def write_inputs_ase(
    output_directory: str,
    charge: int,
    mult: int,
    nbo: bool,
    cores: int,
    functional: str,
    scf_MaxIter: int,
    simple_input: str,
    orca_path: str,
    actinide_basis: str,
    actinide_ecp: str | None,
    traj_file: str,
    non_actinide_basis: str,
    opt: bool = False,
    error_handle: bool = False,
    error_code: int = 0,
    tight_two_e_int: bool = False,
    restart=True,
):
    """
    Write ORCA input files for a given set of parameters.

    Takes:
        - output_directory(str): Directory to write input files to
        - charge(int): Charge of the system
        - mult(int): Multiplicity of the system
        - nbo(bool): Whether to include NBO analysis
        - cores(int): Number of cores to use
        - functional(str): Exchange-correlation functional
        - scf_MaxIter(int): Maximum number of SCF iterations
        - simple_input(str): Simple input type ("omol" or "x2c")
        - orca_path(str): Path to ORCA executable
        - actinide_basis(str): Basis set for actinides
        - actinide_ecp(str|None): ECP for actinides
        - non_actinide_basis(str): Basis set for non-actinides
        - opt(bool): Whether to include optimization keywords
        - error_handle(bool): Whether to include error handling keywords
        - error_code(int): Error code from previous job (0 if no error)
        - tight_two_e_int(bool): Whether to use tight two-electron integrals
    """
    # write the above template to output_directory

    with open(os.path.join(output_directory, "orca.py"), "w") as f:
        f.write(f"import time \n")
        f.write(f"import os\n")
        f.write("from oact_utilities.core.orca.recipes import pure_ase_relaxation\n")
        f.write("from oact_utilities.utils.create import read_geom_from_inp_file\n\n")
        f.write("def main():\n")
        f.write("    os.environ['JAX_PLATFORMS'] = 'cpu'\n")
        f.write(f"    inp_test = '{os.path.join(output_directory, 'orca.inp')}'\n")
        f.write(
            "    atoms_orca = read_geom_from_inp_file(inp_test, ase_format_tf=True)\n"
        )
        f.write(f"    charge = {charge}\n")
        f.write(f"    mult = {mult}\n")
        f.write(f"    output_directory = '{output_directory}'\n")
        f.write(f"    orca_path = '{orca_path}'\n")
        f.write(f"    nbo_tf = {nbo}\n")
        f.write(f"    cores={cores}\n")
        f.write(f"    actinide_basis = '{actinide_basis}'\n")
        if actinide_ecp is None:
            f.write(f"    actinide_ecp = None\n")
        else:
            f.write(f"    actinide_ecp = '{actinide_ecp}'\n")
        f.write(f"    non_actinide_basis = '{non_actinide_basis}'\n")
        f.write("    time_start = time.time()\n")
        f.write("    res_dict = pure_ase_relaxation(\n")
        f.write("        atoms=atoms_orca,\n")
        f.write("        charge=charge,\n")
        f.write("        spin_multiplicity=mult,\n")
        f.write(f"        functional='{functional}',\n")
        f.write(f"        simple_input='{simple_input}',\n")
        if restart:
            f.write("        restart=True,\n")
        f.write(f"        scf_MaxIter={scf_MaxIter},\n")
        f.write("        outputdir=output_directory,\n")
        f.write("        orca_cmd=orca_path,\n")
        f.write("        nbo=nbo_tf,\n")
        f.write(f"        traj_file='{traj_file}',\n")
        f.write("        nprocs=cores,\n")
        f.write("        actinide_basis=actinide_basis,\n")
        f.write("        actinide_ecp=actinide_ecp,\n")
        f.write("        non_actinide_basis=non_actinide_basis,\n")
        f.write(f"        opt={opt},\n")
        f.write(f"        error_handle={error_handle},\n")
        f.write(f"        error_code={error_code},\n")
        f.write(f"        tight_two_e_int={tight_two_e_int}\n")
        f.write("    )\n")
        f.write("    time_end = time.time()\n")
        f.write("    print('Total time (s): ', time_end - time_start)\n\n")
        f.write("if __name__ == '__main__':\n")
        f.write("    main()\n")


def write_jobs(
    actinide_basis: str = "ma-def-TZVP",
    non_actinide_basis: str = "def2-TZVPD",
    actinide_ecp: str = "def-ECP",
    template_file: str = "template_orca.inp",
    root_dir: str = "./orca_jobs/",
    ref_geom_file: str = "/Users/santiagovargas/dev/data/ref_geoms.txt",
    ref_multiplicity_file: str = "/Users/santiagovargas/dev/data/ref_multiplicity.txt",
    cores: int = 8,
    two_step: str | None = None,
) -> None:

    df_multiplicity = process_multiplicity_file(ref_multiplicity_file)
    dict_geoms = process_geometry_file(ref_geom_file)
    lines_cleaned_template = read_template(template_file)

    job_list = df_multiplicity["molecule"].tolist()
    for job in job_list:
        write_orca_input(
            name=job,
            root_folder=root_dir,
            dict_geoms=dict_geoms,
            df_multiplicity=df_multiplicity,
            lines_cleaned_template=lines_cleaned_template,
            charge=0,
            cores=cores,
            actinide_basis=actinide_basis,
            actinide_ecp=actinide_ecp,
            non_actinide_basis=non_actinide_basis,
            two_step=two_step,
        )
