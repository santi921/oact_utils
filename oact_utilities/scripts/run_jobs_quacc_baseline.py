import pickle as pkl
from oact_utilities.core.orca.recipes import single_point_calculation
from oact_utilities.utils.create import *
import time
import random
import parsl

from parsl.executors.threads import ThreadPoolExecutor
from parsl.config import Config
from parsl import python_app

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

    """
    local_threads = Config(
        executors=[
            HighThroughputExecutor(
                label="htex_Local",
                worker_debug=True,
                cpu_affinity='alternating',
                provider=LocalProvider(
                    init_blocks=1,
                    max_blocks=1,
                ),
                cores_per_worker=4, 
                max_workers_per_node=n_workers,
            )
            ],
            strategy='none',
    )"""
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

    # ind = random.randint(0, len(job_list) - 1)
    # mult = spin_list[ind]
    # job = job_list[ind]
    # atoms = dict_geoms_ase[job]

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


def parsl_an66(
    actinide_basis: str = "ma-def-TZVP",
    non_actinide_basis: str = "def2-TZVPD",
    actinide_ecp: str = "def-ECP",
    functional: str = "wB97M-V",
    simple_input: str = "omol",
    scf_MaxIter: int = 100,
    nprocs: int = 4,
    concurrency: int = 2,
    orca_cmd: str = "/Users/santiagovargas/Documents/orca_6_1_0_macosx_arm64_openmpi411/orca",
    ref_geom_file: str = "/Users/santiagovargas/dev/oact_utils/data/data/ref_geoms.txt",
    ref_multiplicity_file: str = "/Users/santiagovargas/dev/oact_utils/data/data/ref_multiplicity.txt",
    root_directory: str = "./test_quacc_baseline/",
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

    df_multiplicity_ase = process_multiplicity_file(ref_multiplicity_file)
    dict_geoms_ase = process_geometry_file(ref_geom_file, ase_format_tf=True)

    job_list = df_multiplicity_ase["molecule"].tolist()
    spin_list = df_multiplicity_ase["multiplicity"].tolist()

    n_draws = len(job_list)

    # create folder if it does not exist
    if not os.path.exists(root_directory):
        os.makedirs(root_directory)

    futures = []
    for draw in range(n_draws):
        # randomly select a job
        ind = random.randint(0, len(job_list) - 1)
        job = job_list[ind]
        atoms = dict_geoms_ase[job]
        nbo_tf = False
        charge = 0
        mult = spin_list[ind]

        futures.append(
            [
                jobs_wrapper_an66(
                    actinide_basis=actinide_basis,
                    non_actinide_basis=non_actinide_basis,
                    actinide_ecp=actinide_ecp,
                    functional=functional,
                    simple_input=simple_input,
                    scf_MaxIter=scf_MaxIter,
                    nprocs=nprocs,
                    orca_cmd=orca_cmd,
                    root_directory=root_directory,
                    job=job,
                    mult=mult,
                    charge=0,
                    atoms=atoms,
                )
            ]
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
    parsl_an66(
        scf_MaxIter=600,
        nprocs=4,
        concurrency=4,
        orca_cmd="/Users/santiagovargas/Documents/orca_6_1_0_macosx_arm64_openmpi411/orca",
        root_directory="/Users/santiagovargas/dev/oact_utils/data/an66_quacc",
        ref_geom_file="/Users/santiagovargas/dev/oact_utils/data/data/ref_geoms.txt",
        ref_multiplicity_file="/Users/santiagovargas/dev/oact_utils/data/data/ref_multiplicity.txt",
    )
