
import os 
import pickle as pkl 
from oact_utilities.core.orca.recipes import ase_relaxation
from oact_utilities.utils.create import read_geom_from_inp_file, read_template, fetch_actinides
from oact_utilities.utils.analysis import (
    get_geo_forces,
    get_rmsd_start_final,
)
from spyrmsd.rmsd import rmsd
import numpy as np 

hartree_to_ev = 27.2114

def test_H2(
    orca_cmd: str = "/Users/santiagovargas/Documents/orca_6_1_0_macosx_arm64_openmpi411/orca",
):
    
    
    """ QUACC baseline jobs for AN66 dataset """
    inp_test = "./files/H2O.inp"
    atoms=read_geom_from_inp_file(inp_test, ase_format_tf=True)
    charge = atoms.charge
    mult = atoms.spin
    cores=12
    nbo_tf = False
    actinide_basis = "def2-TZVP"
    actinide_ecp = None
    non_actinide_basis = "def2-TZVP"
    output_directory = "/Users/santiagovargas/dev/oact_utils/tests/files/quacc/"
    
    res_dict = ase_relaxation(
        atoms=atoms,
        charge=charge,
        spin_multiplicity=mult,
        functional="wB97M-V",
        simple_input="omol",
        scf_MaxIter=200,
        outputdir=output_directory,
        orca_cmd=orca_cmd,
        nbo=nbo_tf,
        nprocs=cores,
        actinide_basis=actinide_basis,
        actinide_ecp=actinide_ecp,
        non_actinide_basis=non_actinide_basis
    )
    
    res_dict = dict(res_dict)
    # save dict as pickle for later analysis
    with open(f"{output_directory}/results.pkl", "wb") as f:
        pkl.dump(res_dict, f)  

    run_folder = res_dict['dir_name']

    # assert job is completed 
    assert res_dict["converged"], "quacc baseline job did not converge"
    

    """ Normal baseline jobs for AN66 dataset """
    
    template_file = "./files/template.inp"
    root_dir = "./files/test_orca/h2o/"
    # clean root_dir
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    else:
        # remove all files in root_dir
        files_in_root = os.listdir(root_dir)
        for f in files_in_root:
            os.remove(os.path.join(root_dir, f))

    lines_cleaned_template = read_template(template_file)
    spin = 1
    element_set = set(["H", "O"])
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
    for atom in read_geom_from_inp_file(inp_test):
        element = atom["element"]
        x = atom["x"]
        y = atom["y"]
        z = atom["z"]
        lines_to_write.append(f"{element}\t{x:.6f}\t{y:.6f}\t{z:.6f}\n")

    lines_to_write.append("*\n")

    file_name = f"{root_dir}/test_orca.inp"
    # if file exists, remove it
    if os.path.exists(file_name):
        os.remove(file_name)


    with open(file_name, "w") as f:
        for line in lines_cleaned_template:
            f.write(line)
        f.write("\n")
        for line in lines_to_write:
            f.write(line)

    # get current directory
    current_directory = os.getcwd()
    
    command_loose = (
        f"cd {root_dir} && {orca_cmd} test_orca.inp > h2o.log && cd {current_directory}"
    )

    os.system(command_loose)

    # find log file
    log_file_name = [f for f in os.listdir(root_dir) if f.endswith(".log")][0]
    log_file_path = os.path.join(root_dir, log_file_name)


    #nprocs, total_time_seconds = find_timings_and_cores(log_file_path)
    geo_forces = get_geo_forces(log_file=log_file_path)
    geom_info = get_rmsd_start_final(root_dir)
    
    orca_final_energies = geom_info["energies_frames"][-1] * hartree_to_ev
    coords_final = geom_info["coords_final"]
    

    quacc_res_dict = res_dict['results']
    energy_quacc = quacc_res_dict["energy"]
    forces_quacc = quacc_res_dict["forces"]
    coords_quacc = res_dict['trajectory'][-1].get_positions()
    #syms_quacc = res_dict['trajectory'][-1].get_chemical_symbols()
    atomic_numbers = [atom.number for atom in res_dict['trajectory'][-1]]
    rmsd_orca_quacc = rmsd(coords_quacc, coords_final, atomic_numbers, atomic_numbers)
    
    # rmsd
    assert rmsd_orca_quacc < 1e-3, f"RMSD between ORCA and Quacc final geometries is too large: {rmsd_orca_quacc}"
    # energies
    assert abs(energy_quacc - orca_final_energies) < 1e-2, f"Final energies between ORCA and Quacc differ too much: {abs(energy_quacc - orca_final_energies)} eV"
    # forces
    rms_forces_quacc = np.sqrt(np.mean(forces_quacc**2))
    rms_forces_orca = geo_forces[-1]["RMS_Gradient"]
    # check both of these are small 
    assert abs(rms_forces_quacc) < 1e-3, f"RMS forces between ORCA and Quacc differ too much: {abs(rms_forces_quacc - rms_forces_orca)} eV/A"
    assert abs(rms_forces_orca) < 1e-3, f"RMS forces between ORCA and Quacc differ too much: {abs(rms_forces_quacc - rms_forces_orca)} eV/A"

