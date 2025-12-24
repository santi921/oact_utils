import os
import time
import parsl
import pickle as pkl

from parsl.executors.threads import ThreadPoolExecutor
from parsl.config import Config
from parsl import python_app

from oact_utilities.core.orca.recipes import single_point_calculation
from oact_utilities.utils.baselines import (
    process_multiplicity_file,
    process_geometry_file,
)
from oact_utilities.utils.status import check_job_termination

import argparse

should_stop = False


def handle_signal(signum, frame):
    global should_stop
    print(f"Received signal {signum}, initiating graceful shutdown...")
    should_stop = True


def base_config(n_workers: int = 128) -> Config:
    """Returns a basic Parsl config using local threads executor.

    Returns:
        Config: A Parsl configuration object.
    """
    local_threads = Config(
        executors=[ThreadPoolExecutor(max_threads=n_workers, label="local_threads")]
    )

    return local_threads


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

    nbo_tf = False
    root_directory_job = os.path.join(root_directory, job)

    if not os.path.exists(root_directory_job):
        os.makedirs(root_directory_job)

    try:

        if not os.path.exists(root_directory_job):
            os.makedirs(root_directory_job)
        else:
            # check if the folder has a results.pkl file, if so skip
            if os.path.exists(os.path.join(root_directory_job, "results.pkl")):
                print(f"Job for {job} already completed. Skipping.")
                return
            # check if the folder has recently modified files, if so skip
            files_in_job = os.listdir(root_directory_job)
            recent_modification = False
            for f in files_in_job:
                file_path = os.path.join(root_directory_job, f)
                if os.path.getmtime(file_path) > time.time() - 900:  # 15 mins
                    recent_modification = True
                    print(f"Job for {job} is currently running. Skipping.")
                    break

        time_start = time.time()

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
        print(f"Job for {job} completed in {time_end - time_start} seconds.")

        # convert res_dict to normal dict
        res_dict = dict(res_dict)
        res_dict["time_seconds"] = time_end - time_start
        # save res_dict as json
        save_loc = dict(res_dict)["dir_name"]
        with open(os.path.join(save_loc, "results.pkl"), "wb") as f:
            pkl.dump(res_dict, f)

    except Exception as e:
        print(f"Job for {job} failed with error: {e}")

    return


def parsl_wave2(
    root_data_dir: str,
    calc_root_dir: str,
    actinide_basis: str = "ma-def-TZVP",
    non_actinide_basis: str = "def2-TZVPD",
    actinide_ecp: str = "def-ECP",
    functional: str = "wB97M-V",
    simple_input: str = "omol",
    scf_MaxIter: int = 100,
    nprocs: int = 4,
    skip_done: bool = True,
    dry_run: bool = False,
    overwrite: bool = False,
    concurrency: int = 2,
    orca_cmd: str = "/Users/santiagovargas/Documents/orca_6_1_0_macosx_arm64_openmpi411/orca",
):

    ##################### Gather Configs for Parsl

    parsl_config = base_config(n_workers=concurrency)
    parsl.clear()
    parsl.load(parsl_config)
    print("Parsl config loaded. Submitting jobs...")
    print(parsl_config)

    ####################

    os.environ["OMP_NUM_THREADS"] = "{}".format(nprocs)
    # signal.signal(signal.SIGINT, handle_signal)
    # signal.signal(signal.SIGTERM, handle_signal)

    hard_donors_dir = "Hard_Donors/"
    organic_dir = "Organic/"
    radical_dir = "Radical/"
    soft_donors_dir = "Soft_Donors/"
    count = 0

    list_jobs = [hard_donors_dir, organic_dir, radical_dir, soft_donors_dir]
    dict_full_set = {}

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
                folder_mol = os.path.join(folder_to_use, mol_name)
                # add this to unified dict
                dict_unified[mol_name]["dir_name"] = folder_mol

                if not os.path.exists(folder_mol):
                    os.mkdir(folder_mol)
                error_code = check_job_termination(folder_mol)

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
                    if "orca.inp" not in os.listdir(folder_mol) and overwrite == False:
                        # add to dict_full_set which will launch jobs
                        dict_full_set[mol_name] = dict_unified[mol_name]
                        count_subfolders += 1

    print(f"Found {count_subfolders} jobs to run in folder {folder_name}.")
    print(f"Total jobs to run: {len(dict_full_set)}")

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

    try:
        # block until all done
        for f in futures:
            if should_stop:
                print("Graceful shutdown requested. Exiting before all jobs complete.")
                break
            f.result()
    finally:
        # ensure we cleanup even on exceptions
        try:
            dfk = parsl.dfk()
            if dfk is not None:
                dfk.cleanup()  # shutdown workers/executors
        except Exception as e:
            # log warning, don't crash on cleanup failure
            print("Warning: cleanup failed:", e)
        # remove Parsl DFK from module so subsequent imports/config changes are possible
        try:
            parsl.clear()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_data_dir", type=str, default="/Users/santiagovargas/dev/oact_utils/data/big_benchmark/")
    parser.add_argument("--calc_root_dir", type=str, default="/Users/santiagovargas/dev/oact_utils/data/big_benchmark_out/")
    parser.add_argument("--orca_cmd", type=str, default="/Users/santiagovargas/Documents/orca_6_1_0_macosx_arm64_openmpi411/orca")
    parser.add_argument("--scf_MaxIter", type=int, default=600)
    parser.add_argument("--nprocs", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    parsl_wave2(
        scf_MaxIter=args.scf_MaxIter,
        nprocs=args.nprocs,
        concurrency=args.concurrency,
        orca_cmd=args.orca_cmd,
        root_data_dir=args.root_data_dir,
        calc_root_dir=args.calc_root_dir,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
