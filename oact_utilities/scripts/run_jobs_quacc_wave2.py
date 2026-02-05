import argparse
import os
import pickle as pkl
import socket
import sys

import parsl
from parsl import python_app
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import SimpleLauncher
from parsl.providers import SlurmProvider

from oact_utilities.utils.baselines import (
    process_geometry_file,
    process_multiplicity_file,
)
from oact_utilities.utils.status import check_job_termination

from concurrent.futures import as_completed

should_stop = False


def handle_signal(signum, frame):
    global should_stop
    print(f"Received signal {signum}, initiating graceful shutdown...")
    should_stop = True


def base_config(
    max_blocks: int = 10,
    walltime_hours: int = 2,
    qos: str = "frontier",
    account: str = "ODEFN5169CYFZ",
    source_cmd: str = "source ~/.bashrc",
    conda_env: str = "py10mpi",
    conda_base: str = "/usr/WS1/vargas58/miniconda3",
) -> Config:
    """Returns a Parsl config using SlurmProvider with HighThroughputExecutor."""

    worker_init_commands = f"""{source_cmd}
conda activate {conda_env}
export LD_LIBRARY_PATH={conda_base}/envs/{conda_env}/lib:$LD_LIBRARY_PATH
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
"""

    provider = SlurmProvider(
        qos=qos,
        account=account,
        nodes_per_block=1,
        init_blocks=2,
        min_blocks=1,
        max_blocks=max_blocks,
        walltime=f"{walltime_hours:02d}:00:00",
        worker_init=worker_init_commands,
        scheduler_options=(
            "#SBATCH --ntasks-per-node=64\n" "#SBATCH --cpus-per-task=1\n"
        ),
        exclusive=True,
        launcher=SimpleLauncher(),  # Do not use SrunLauncher because of ORCA's internal mpirun
        parallelism=1.0,
    )

    executor = HighThroughputExecutor(
        label="slurm_htex",
        cores_per_worker=16,
        max_workers_per_node=4,
        cpu_affinity="block",
        provider=provider,
    )

    return Config(executors=[executor])


@python_app
def jobs_wrapper_an66(
    actinide_basis: str,
    non_actinide_basis: str,
    actinide_ecp: str,
    functional: str,
    simple_input: str,
    scf_MaxIter: int,
    nprocs: int,
    orca_cmd: str,
    root_directory: str,
    job: str,
    mult: int,
    atoms,
    charge: int = 0,
):
    """
    Runs one ORCA single-point job.
    """
    import os
    import time
    import pickle as pkl

    from oact_utilities.core.orca.recipes import single_point_calculation

    # Each ORCA MPI rank gets 1 CPU (16 ranks total per job)
    os.environ["OMP_NUM_THREADS"] = "1"

    nbo_tf = False
    root_directory_job = root_directory
    os.makedirs(root_directory_job, exist_ok=True)

    results_pkl = os.path.join(root_directory_job, "results.pkl")
    if os.path.exists(results_pkl):
        print(f"Job {job} already completed. Skipping.")
        return

    time_start = time.time()

    try:
        res_dict = single_point_calculation(
            atoms=atoms,
            charge=charge,
            spin_multiplicity=mult,
            functional=functional,
            simple_input=simple_input,
            scf_MaxIter=scf_MaxIter,
            actinide_basis=actinide_basis,
            actinide_ecp=actinide_ecp,
            non_actinide_basis=non_actinide_basis,
            nprocs=nprocs,
            outputdir=root_directory_job,
            nbo=nbo_tf,
            orca_cmd=orca_cmd,
        )

        time_end = time.time()
        print(f"Job {job} completed in {time_end - time_start:.1f} seconds.")

        res_dict = dict(res_dict)
        res_dict["time_seconds"] = time_end - time_start

        save_loc = res_dict.get("dir_name", root_directory_job)
        os.makedirs(save_loc, exist_ok=True)
        with open(os.path.join(save_loc, "results.pkl"), "wb") as f:
            pkl.dump(res_dict, f)

    except Exception as e:
        print(f"Job {job} failed: {e}")
        pass


def parsl_wave2(
    root_data_dir: str,
    calc_root_dir: str,
    actinide_basis: str = "ma-def-TZVP",
    non_actinide_basis: str = "def2-TZVPD",
    actinide_ecp: str = "def-ECP",
    functional: str = "wB97M-V",
    simple_input: str = "omol",
    scf_MaxIter: int = 100,
    nprocs: int = 16,
    skip_done: bool = True,
    dry_run: bool = False,
    overwrite: bool = False,
    max_blocks: int = 10,
    walltime_hours: int = 2,
    qos: str = "frontier",
    account: str = "ODEFN5169CYFZ",
    source_cmd: str = "source ~/.bashrc",
    conda_env: str = "py10mpi",
    conda_base: str = "/usr/WS1/vargas58/miniconda3",
    orca_cmd: str = "",
):
    parsl_config = base_config(
        max_blocks=max_blocks,
        walltime_hours=walltime_hours,
        qos=qos,
        account=account,
        source_cmd=source_cmd,
        conda_env=conda_env,
        conda_base=conda_base,
    )
    parsl.clear()
    parsl.load(parsl_config)
    print("Parsl config loaded. Submitting jobs...")
    print(parsl_config)

    hard_donors_dir = "Hard_Donors/"
    organic_dir = "Organic/"
    radical_dir = "Radical/"
    soft_donors_dir = "Soft_Donors/"
    list_jobs = [hard_donors_dir, organic_dir, radical_dir, soft_donors_dir]
    dict_full_set = {}

    for base_dir in list_jobs:
        os.makedirs(os.path.join(calc_root_dir, base_dir), exist_ok=True)

        subfolders = [
            f.path for f in os.scandir(root_data_dir + base_dir) if f.is_dir()
        ]

        dict_unified = {}
        dict_geoms = {}
        for folder_to_use in subfolders:
            print(f"Processing folder: {folder_to_use}")
            folder_name = folder_to_use.split("/")[-1]
            folder_out_base = os.path.join(calc_root_dir, base_dir, folder_name)
            os.makedirs(folder_out_base, exist_ok=True)

            orig_folder = os.path.join(root_data_dir, base_dir, folder_name)
            geom_file = os.path.join(orig_folder, "data_geom.txt")
            mult_file = os.path.join(orig_folder, "data_charge_mult.txt")

            ase_format_tf = True
            df_multiplicity = process_multiplicity_file(mult_file)
            folder_dict_geoms = process_geometry_file(
                geom_file, ase_format_tf=ase_format_tf
            )
            dict_geoms.update(folder_dict_geoms)

            dict_unified.update(
                {
                    k: {
                        "geometry": folder_dict_geoms[k],
                        "multiplicity": df_multiplicity[k]["multiplicity"],
                        "charge": df_multiplicity[k]["charge"],
                    }
                    for k in folder_dict_geoms.keys()
                    if k in df_multiplicity.keys()
                }
            )

        print(f"Number of unified geometries in folder {base_dir}: {len(dict_unified)}")

        for mol_name, _ in dict_geoms.items():
            folder_mol = os.path.join(folder_out_base, mol_name)
            dict_unified[mol_name]["dir_name"] = folder_mol
            os.makedirs(folder_mol, exist_ok=True)
            # this is running with orca that creates another folder inside, find the latest sub sub dir to check status
            folder_check = folder_mol
            # get list of subdirs
            sub_dirs = [f.path for f in os.scandir(folder_mol) if f.is_dir()]
            if len(sub_dirs) > 0:
                latest_subdir = max(sub_dirs, key=os.path.getmtime)
                folder_check = latest_subdir

                error_code = check_job_termination(folder_check)
                if skip_done and error_code == 1:
                    # Completed job detected
                    continue

                if dry_run:
                    continue

            if ("orca.inp" not in os.listdir(folder_mol)) or overwrite:
                dict_full_set[mol_name] = dict_unified[mol_name]

    print(f"Total jobs to run: {len(dict_full_set)}")
    exit()

    futures = []
    for job, vals in dict_full_set.items():
        atoms = vals["geometry"]
        mult = vals["multiplicity"]
        charge = vals["charge"]
        root_directory_job = vals["dir_name"]

        futures.append(
            jobs_wrapper_an66(
                actinide_basis=actinide_basis,
                non_actinide_basis=non_actinide_basis,
                actinide_ecp=actinide_ecp,
                functional=functional,
                simple_input=simple_input,
                scf_MaxIter=scf_MaxIter,
                nprocs=nprocs,
                orca_cmd=orca_cmd,
                root_directory=root_directory_job,
                job=job,
                mult=mult,
                charge=charge,
                atoms=atoms,
            )
        )

    future_to_job = {f: job for f, job in zip(futures, dict_full_set.keys())}

    failures = []

    try:
        for f in as_completed(futures):
            if should_stop:
                print("Graceful shutdown requested. Exiting before all jobs complete.")
                break

            job = future_to_job.get(f, "<unknown>")
            try:
                f.result()
            except Exception as e:
                print(f"[FAILED] {job}: {e}")
                failures.append((job, repr(e)))
                # continue running remaining jobs
                continue
    finally:
        try:
            dfk = parsl.dfk()
            if dfk is not None:
                dfk.cleanup()
        except Exception as e:
            print("Warning: cleanup failed:", e)
        try:
            parsl.clear()
        except Exception:
            pass

    print(f"Total failures: {len(failures)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_data_dir",
        type=str,
        default="/Users/santiagovargas/dev/oact_utils/data/big_benchmark/",
    )
    parser.add_argument(
        "--calc_root_dir",
        type=str,
        default="/Users/santiagovargas/dev/oact_utils/data/big_benchmark_out/",
    )
    parser.add_argument(
        "--orca_cmd",
        type=str,
        required=False,
        help="Path to Orca executable on cluster (required unless --clean is used)",
    )
    parser.add_argument("--scf_MaxIter", type=int, default=600)
    parser.add_argument(
        "--nprocs",
        type=int,
        default=16,
        help="Number of cores per Orca process (should match cores_per_worker=16)",
    )
    parser.add_argument(
        "--max_blocks",
        type=int,
        default=10,
        dest="max_blocks",
        help="Maximum number of Slurm nodes to provision",
    )
    parser.add_argument(
        "--walltime_hours",
        type=int,
        default=2,
        dest="walltime_hours",
        help="Walltime in hours (converted to HH:00:00 format)",
    )
    parser.add_argument(
        "--source_cmd",
        type=str,
        default="source ~/.bashrc",
        dest="source_cmd",
        help="Command to source bashrc/profile",
    )
    parser.add_argument(
        "--conda_env",
        type=str,
        default="py10mpi",
        dest="conda_env",
        help="Conda environment name to activate",
    )
    parser.add_argument(
        "--conda_base",
        type=str,
        default="/usr/WS1/vargas58/miniconda3",
        dest="conda_base",
        help="Base path for conda installation",
    )
    parser.add_argument(
        "--qos",
        type=str,
        default="frontier",
        dest="qos",
        help="QOS for Slurm allocation",
    )
    parser.add_argument(
        "--account",
        type=str,
        default="ODEFN5169CYFZ",
        dest="account",
        help="Account for Slurm allocation",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Load Parsl config and run cleanup, then exit",
    )
    args = parser.parse_args()

    if args.clean:
        parsl_config = base_config(
            max_blocks=args.max_blocks,
            walltime_hours=args.walltime_hours,
            source_cmd=args.source_cmd,
            conda_env=args.conda_env,
            conda_base=args.conda_base,
        )
        parsl.clear()
        parsl.load(parsl_config)
        print("Parsl config loaded. Running cleanup...")
        try:
            dfk = parsl.dfk()
            if dfk is not None:
                dfk.cleanup()
                print("Cleanup completed.")
            else:
                print("No Parsl DFK found.")
        except Exception as e:
            print(f"Warning: cleanup failed: {e}")
        finally:
            try:
                parsl.clear()
            except Exception:
                pass
        sys.exit(0)

    if args.orca_cmd is None:
        parser.error("--orca_cmd is required unless --clean is used")

    parsl_wave2(
        scf_MaxIter=args.scf_MaxIter,
        nprocs=args.nprocs,
        max_blocks=args.max_blocks,
        walltime_hours=args.walltime_hours,
        source_cmd=args.source_cmd,
        conda_env=args.conda_env,
        conda_base=args.conda_base,
        orca_cmd=args.orca_cmd,
        qos=args.qos,
        account=args.account,
        root_data_dir=args.root_data_dir,
        calc_root_dir=args.calc_root_dir,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
