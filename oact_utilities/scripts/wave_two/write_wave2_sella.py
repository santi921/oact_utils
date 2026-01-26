import argparse
import os

from oact_utilities.core.orca.calc import write_orca_inputs
from oact_utilities.utils.baselines import (
    process_geometry_file,
    process_multiplicity_file,
)
from oact_utilities.utils.create import write_inputs_ase
from oact_utilities.utils.hpc import write_flux_no_template_sella_ase
from oact_utilities.utils.status import check_job_termination

os.environ["JAX_PLATFORMS"] = "cpu"


def write_sella_python_ase_job(
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
    lot="omol",
    functional="wB97M-V",
    opt=False,
    dry_run: bool = False,
    overwrite=False,
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

            for mol_name, _ in dict_geoms.items():
                # print(f"Processing molecule: {mol_name}, geometry with {len(vals)} atoms")
                # if orca.inp already exists in folder_to_use/mol_name, delete

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

                if not dry_run:
                    # check if traj_file exists in folder_mol, also if so, traj_file = folder_mol/opt.traj
                    traj_file = os.path.join(folder_mol, "opt.traj")
                    if not os.path.exists(traj_file):
                        traj_file = None

                    restart = False
                    # check if there is no *inp file already there
                    if "orca.inp" not in os.listdir(folder_mol) and not overwrite:
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
                            tight_two_e_int=tight_two_e_int,
                        )
                        restart = True
                    else:
                        print(
                            f"Input file already exists in {folder_mol}, skipping input writing."
                        )

                    write_inputs_ase(
                        output_directory=os.path.join(folder_mol),
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
                        restart=restart,
                        error_handle=True,
                        error_code=error_code,
                        tight_two_e_int=tight_two_e_int,
                        traj_file=traj_file,
                    )
                    count += 1
                    count_subfolders += 1

            if safety:
                cores += 2

            if not dry_run:
                write_flux_no_template_sella_ase(
                    root_dir=folder_to_use,
                    two_step=two_step,
                    n_cores=cores,
                    n_hours=n_hours,
                    queue=queue,
                    allocation=allocation,
                )

    print(f"Total number of jobs prepared: {count}")


if __name__ == "__main__":
    # Example defaults (kept as comments for reference):
    # job_handler = "flux"
    # queue = "pbatch"
    # allocation = "dnn-sim"
    # source_bashrc = "source ~/.bashrc"
    # conda_env = "py10mpi"
    # LD_LIBRARY_PATH = "/usr/WS1/vargas58/miniconda3/envs/py10mpi/lib"
    # n_hours = 10
    # cores = 20
    # tight_two_e_int = True
    # root_data_dir = "/Users/santiagovargas/dev/oact_utils/data/big_benchmark/"
    # calc_root_dir = "/Users/santiagovargas/dev/oact_utils/data/big_benchmark_out_sella/"
    # orca_exe = "/Users/santiagovargas/Documents/orca_6_1_0_macosx_arm64_openmpi411/orca"
    # actinide_basis = "ma-def-TZVP"
    # actinide_ecp = "def-ECP"
    # non_actinide_basis = "def2-TZVPD"

    parser = argparse.ArgumentParser(
        description="Prepare Sella/ORCA ASE jobs for Wave 2."
    )
    parser.add_argument(
        "--root-data-dir",
        default="/Users/santiagovargas/dev/oact_utils/data/big_benchmark/",
        help="Root data directory",
    )
    parser.add_argument(
        "--calc-root-dir",
        default="/Users/santiagovargas/dev/oact_utils/data/big_benchmark_out_sella/",
        help="Calculation root directory",
    )
    parser.add_argument(
        "--orca-exe",
        default="/Users/santiagovargas/Documents/orca_6_1_0_macosx_arm64_openmpi411/orca",
        help="Path to ORCA executable",
    )
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--n-hours", type=int, default=10)
    parser.add_argument("--queue", default="pbatch")
    parser.add_argument("--allocation", default="dnn-sim")
    parser.add_argument("--lot", default="omol", choices=["omol", "x2c"])
    parser.add_argument("--functional", default="wB97M-V")
    parser.add_argument("--actinide-basis", default="ma-def-TZVP")
    parser.add_argument("--actinide-ecp", default="def-ECP")
    parser.add_argument("--non-actinide-basis", default="def2-TZVPD")
    parser.add_argument("--tight-two-e-int", action="store_true")
    parser.add_argument(
        "--skip-done", action="store_true", help="Skip folders with completed jobs"
    )
    parser.add_argument("--opt", action="store_true", help="Run geometry optimizations")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    write_sella_python_ase_job(
        actinide_basis=args.actinide_basis,
        actinide_ecp=args.actinide_ecp,
        non_actinide_basis=args.non_actinide_basis,
        cores=args.cores,
        orca_exe=args.orca_exe,
        safety=False,
        max_scf_iterations=600,
        n_hours=args.n_hours,
        allocation=args.allocation,
        queue=args.queue,
        root_data_dir=args.root_data_dir,
        calc_root_dir=args.calc_root_dir,
        skip_done=args.skip_done,
        lot=args.lot,
        functional=args.functional,
        job_handler="flux",
        opt=args.opt,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        tight_two_e_int=args.tight_two_e_int,
    )
