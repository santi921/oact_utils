import os
import time

from oact_utilities.utils.baselines import (
    process_multiplicity_file,
    process_geometry_file,
)
from oact_utilities.core.orca.calc import write_orca_inputs
from oact_utilities.utils.create import write_flux_no_template, write_slurm_no_template
from oact_utilities.utils.status import check_job_termination


def write_flux_orca_wave_two(
    actinide_basis: str,
    actinide_ecp: str,
    non_actinide_basis: str,
    tight_two_e_int: bool,
    root_data_dir: str,
    calc_root_dir: str,
    orca_exe: str,
    cores: int,
    safety: bool = True,
    n_hours: int = 24,
    max_scf_iterations: int = 1000,
    allocation: str = "dnn-sim",
    two_step: bool = False,
    queue: str = "pbatch",
    skip_done: bool = True,
    replicates: int = 1,
    lot="omol",
    functional="wB97M-V",
    opt=False,
    job_handler="flux",
    source_bashrc: str = "source ~/.bashrc",
    conda_env: str = "py10mpi",
    LD_LIBRARY_PATH: str = "/usr/WS1/vargas58/miniconda3/envs/py10mpi/lib",
):
    """
    Write ORCA input files and job submission scripts for Wave 2 calculations. Can run sp or opt

    Takes:
        - actinide_basis(str): Basis set for actinides
        - actinide_ecp(str): ECP for actinides
        - non_actinide_basis(str): Basis set for non-actinides
        - tight_two_e_int(bool): Whether to use tight two-electron integrals
        - root_data_dir(str): Root directory where data folders are located
        - calc_root_dir(str): Root directory where calculation folders will be created
        - orca_exe(str): Path to ORCA executable
        - cores(int): Number of cores to use
        - safety(bool): Whether to add safety cores to job submission script
        - n_hours(int): Number of hours to request for job submission script
        - max_scf_iterations(int): Maximum number of SCF iterations
        - allocation(str): Allocation name for job submission script
        - two_step(bool): Whether to use two-step submission script
        - queue(str): Queue name for job submission script
        - skip_done(bool): Whether to skip jobs that are already done
        - replicates(int): Number of replicates to create for each molecule
        - lot(str): Level of theory to use ("omol" or "x2c")
        - functional(str): Functional to use
        - opt(bool): Whether to run optimization or single point
        - job_handler(str): "flux" or "slurm" to determine job submission script
        - source_bashrc(str): Command to source bashrc for jobs
        - conda_env(str): Conda environment name for jobs
        - LD_LIBRARY_PATH(str): LD_LIBRARY_PATH for jobs
    """

    hard_donors_dir = "Hard_Donors/"
    organic_dir = "Organic/"
    radical_dir = "Radical/"
    soft_donors_dir = "Soft_Donors/"

    # find subfolders in each directory, reconstruct dur structure in calc_root_dir
    count = 0

    list_jobs = [hard_donors_dir, organic_dir, radical_dir, soft_donors_dir]

    for base_dir in list_jobs:
        # if base_dir does not exist in calc_root_dir, create it
        if not os.path.exists(os.path.join(calc_root_dir, base_dir)):
            os.makedirs(os.path.join(calc_root_dir, base_dir))

        subfolders = [
            f.path for f in os.scandir(root_data_dir + base_dir) if f.is_dir()
        ]

        for folder_to_use in subfolders:
            count_subfolders = 0
            print(f"Processing folder: {folder_to_use}")
            folder_name = folder_to_use.split("/")[-1]
            folder_to_use = os.path.join(calc_root_dir, base_dir, folder_name)

            if not os.path.exists(folder_to_use):
                os.mkdir(
                    os.path.join(calc_root_dir, base_dir, folder_to_use.split("/")[-1])
                )

            # open the data_geom.txt file in the original folder, also open data_charge_muilt.txt
            orig_folder = os.path.join(root_data_dir, base_dir, folder_name)

            geom_file = os.path.join(orig_folder, "data_geom.txt")
            mult_file = os.path.join(orig_folder, "data_charge_mult.txt")
            ase_format_tf = True
            df_multiplicity = process_multiplicity_file(mult_file)
            dict_geoms = process_geometry_file(geom_file, ase_format_tf=ase_format_tf)
            # zip dict_geoms and df_multiplicity
            dict_unified = {
                k: {
                    "geometry": dict_geoms[k],
                    "multiplicity": df_multiplicity[k]["multiplicity"],
                    "charge": df_multiplicity[k]["charge"],
                }
                for k in dict_geoms.keys()
                if k in df_multiplicity.keys()
            }

            for mol_name, vals in dict_geoms.items():
                # print(f"Processing molecule: {mol_name}, geometry with {len(vals)} atoms")
                # if orca.inp already exists in folder_to_use/mol_name, delete
                if replicates > 1:
                    for rep in range(replicates):
                        print(f"  Writing replicate {rep+1} for molecule {mol_name}")
                        folder_rep = os.path.join(
                            folder_to_use, f"{mol_name}_rep{rep+1}"
                        )
                        if not os.path.exists(folder_rep):
                            os.mkdir(folder_rep)
                        error_code = check_job_termination(folder_rep)

                        # print(f"Using calculation folder: {folder_to_use}")
                        if skip_done:
                            # check if folder has successful flux job
                            if error_code:
                                print(
                                    f"Skipping {folder_rep} as it has a completed job."
                                )
                                continue

                        write_orca_inputs(
                            atoms=dict_unified[mol_name]["geometry"],
                            output_directory=os.path.join(folder_rep),
                            charge=dict_unified[mol_name]["charge"],
                            mult=dict_unified[mol_name]["multiplicity"],
                            nbo=False,
                            cores=cores,
                            functional=functional,
                            scf_MaxIter=max_scf_iterations,
                            simple_input=lot,
                            orca_path=orca_exe,
                            actinide_basis=actinide_basis,
                            actinide_ecp=actinide_ecp,
                            non_actinide_basis=non_actinide_basis,
                            opt=opt,
                            error_handle=True,
                            error_code=error_code,
                            tight_two_e_int=tight_two_e_int
                        )
                        count += 1
                        count_subfolders += 1
                else:

                    folder_mol = os.path.join(folder_to_use, mol_name)
                    if not os.path.exists(folder_mol):
                        os.mkdir(folder_mol)
                    error_code = check_job_termination(folder_mol)

                    # print(f"Using calculation folder: {folder_to_use}")
                    if skip_done:
                        # check if folder has successful flux job
                        if error_code == 1:
                            print(f"Skipping {folder_mol} as it has a completed job.")
                            continue

                    write_orca_inputs(
                        atoms=dict_unified[mol_name]["geometry"],
                        output_directory=folder_mol,
                        charge=dict_unified[mol_name]["charge"],
                        mult=dict_unified[mol_name]["multiplicity"],
                        nbo=False,
                        cores=cores,
                        functional=functional,
                        scf_MaxIter=max_scf_iterations,
                        simple_input=lot,
                        orca_path=orca_exe,
                        actinide_basis=actinide_basis,
                        actinide_ecp=actinide_ecp,
                        non_actinide_basis=non_actinide_basis,
                        opt=opt,
                        error_handle=True,
                        error_code=error_code,
                        tight_two_e_int=tight_two_e_int
                    )
                    count += 1
                    count_subfolders += 1

            if safety:
                cores += 2

            if job_handler == "flux":
                write_flux_no_template(
                    root_dir=folder_to_use,
                    two_step=two_step,
                    n_cores=cores,
                    n_hours=n_hours,
                    queue=queue,
                    allocation=allocation,
                )
            elif job_handler == "slurm":

                write_slurm_no_template(
                    root_dir=folder_to_use,
                    two_step=two_step,
                    n_cores=cores,
                    n_hours=n_hours,
                    queue=queue,
                    allocation=allocation,
                    conda_env=conda_env,
                    source_bashrc=source_bashrc,
                    LD_LIBRARY_PATH=LD_LIBRARY_PATH,
                    orca_command=orca_exe,
                )
            else:
                raise ValueError(f"Unknown job handler: {job_handler}")

    print(f"Total number of jobs prepared: {count}")


if __name__ == "__main__":

    two_step = None
    ################################## OMOL BLOCK ##################################

    # 1) baseline omol
    actinide_basis = "ma-def-TZVP"
    actinide_ecp = "def-ECP"
    non_actinide_basis = "def2-TZVPD"

    calc_root_dir = "/usr/workspace/vargas58/orca_test/an66_benchmarks/wave_2_omol_sp/"
    root_data_dir = "UPDATE ON TUO"
    orca_exe = (
        "/usr/workspace/vargas58/orca_test/orca_6_2_1_linux_x86-64_openmpi411/orca"
    )

    root_data_dir = "/Users/santiagovargas/dev/oact_utils/data/big_benchmark/"
    calc_root_dir = "/Users/santiagovargas/dev/oact_utils/data/big_benchmark_out/"
    orca_exe = "/Users/santiagovargas/Documents/orca_6_1_0_macosx_arm64_openmpi411/orca"

    ##############################################################################
    # Ritwik - Things to modify for your system
    job_handler = "flux"
    queue = "pbatch"
    allocation = "dnn-sim"
    source_bashrc = "source ~/.bashrc"
    conda_env = "py10mpi"
    LD_LIBRARY_PATH = "/usr/WS1/vargas58/miniconda3/envs/py10mpi/lib"
    n_hours = 10
    replicates = 1
    cores = 20
    tight_two_e_int=True
    ##############################################################################

    write_flux_orca_wave_two(
        actinide_basis=actinide_basis,
        actinide_ecp=actinide_ecp,
        non_actinide_basis=non_actinide_basis,
        two_step=two_step,
        cores=cores,
        orca_exe=orca_exe,
        safety=False,
        max_scf_iterations=600,
        n_hours=n_hours,
        allocation=allocation,
        queue=queue,
        root_data_dir=root_data_dir,
        calc_root_dir=calc_root_dir,
        skip_done=True,
        replicates=replicates,
        lot="omol",
        functional="wB97M-V",
        job_handler=job_handler,
        opt=False,
        source_bashrc=source_bashrc,
        conda_env=conda_env,
        LD_LIBRARY_PATH=LD_LIBRARY_PATH,
        tight_two_e_int=tight_two_e_int,
    )
