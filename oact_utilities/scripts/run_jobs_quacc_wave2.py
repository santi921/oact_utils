import os
import sys
import time
import parsl
import pickle as pkl

from parsl.executors import HighThroughputExecutor
from parsl.providers import SlurmProvider
from parsl.launchers import SimpleLauncher
from parsl.config import Config
from parsl import python_app

from oact_utilities.core.orca.recipes import single_point_calculation
from oact_utilities.utils.baselines import (
    process_multiplicity_file,
    process_geometry_file,
)
from oact_utilities.utils.status import check_job_termination

import argparse
import socket

should_stop = False


def handle_signal(signum, frame):
    global should_stop
    print(f"Received signal {signum}, initiating graceful shutdown...")
    should_stop = True


def base_config(
    max_blocks: int = 10,
    walltime_hours: int = 2,
    source_cmd: str = "source ~/.bashrc",
    conda_env: str = "py10mpi",
    conda_base: str = "/usr/WS1/vargas58/miniconda3",
) -> Config:
    """Returns a Parsl config using SlurmProvider with HighThroughputExecutor.

    Goals:
      - 1 node per block with 64 cores total
      - 4 concurrent ORCA runs per node (max_workers_per_node=4)
      - Each ORCA run gets pinned to a disjoint 16-CPU NUMA domain (0-15, 16-31, 32-47, 48-63)
      - Uses SimpleLauncher to avoid srun job-step issues with ORCA's internal mpirun
    """

    worker_init_commands = f"""{source_cmd}
conda activate {conda_env}
export LD_LIBRARY_PATH={conda_base}/envs/{conda_env}/lib:$LD_LIBRARY_PATH
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
"""

    provider = SlurmProvider(
        qos="frontier",
        account="ODEFN5169CYFZ",
        nodes_per_block=1,
        init_blocks=2,
        min_blocks=1,
        max_blocks=max_blocks,
        walltime=f"{walltime_hours:02d}:00:00",
        worker_init=worker_init_commands,
        # We want full-node resources available so ORCA's internal mpirun has "slots"
        # to launch 16 ranks per job, up to 4 jobs concurrently per node.
        scheduler_options=(
            "#SBATCH --ntasks-per-node=64\n"
            "#SBATCH --cpus-per-task=1\n"
        ),
        exclusive=True,
        # CRITICAL: Use SimpleLauncher, NOT SrunLauncher.
        # SrunLauncher creates job steps with limited task slots (SLURM_NTASKS=1),
        # causing ORCA's internal mpirun to fail when trying to launch 16 MPI ranks.
        # SimpleLauncher runs workers directly in the full 64-slot allocation,
        # allowing ORCA's mpirun to see all available slots and launch successfully.
        launcher=SimpleLauncher(),
        parallelism=1.0,
    )

    executor = HighThroughputExecutor(
        label="slurm_htex",
        cores_per_worker=16,
        max_workers_per_node=4,
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
    Runs one ORCA single-point job with proper CPU pinning for concurrent execution.

    CPU Pinning Strategy:
      - HTEX runs up to 4 tasks per node (max_workers_per_node=4)
      - Each task gets pinned to a disjoint 16-CPU NUMA domain using numactl
      - ORCA runs within the pinned region and uses all available CPUs for its 16 MPI ranks
      - This ensures no CPU contention between concurrent ORCA jobs on the same node

    Slot Assignment:
      - Uses a simple per-node counter with file-based locking for thread safety
      - Slots 0-3 map to CPU ranges: 0-15, 16-31, 32-47, 48-63
    """
    import os
    import time
    import pickle as pkl
    import tempfile
    import fcntl

    from oact_utilities.core.orca.recipes import single_point_calculation

    # Each ORCA MPI rank gets 1 CPU (16 ranks total per job)
    os.environ["OMP_NUM_THREADS"] = "1"

    def acquire_slot():
        """Acquire a per-node slot (0-3) using atomic file operations."""
        hostname = os.environ.get("SLURMD_NODENAME") or socket.gethostname()
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
        counter_file = os.path.join(tempfile.gettempdir(), f"parsl_slot_counter_{slurm_job_id}_{hostname}")

        # Use atomic file operations to get next slot
        for attempt in range(10):  # Retry a few times if contention
            try:
                with open(counter_file, 'a+') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock
                    f.seek(0)
                    content = f.read().strip()
                    current_slot = int(content) if content else 0
                    slot_id = current_slot % 4  # Cycle through 0-3
                    f.seek(0)
                    f.truncate()
                    f.write(str(current_slot + 1))
                    f.flush()
                    os.fsync(f.fileno())
                    return slot_id
            except Exception:
                if attempt == 9:
                    raise
                time.sleep(0.01)  # Brief pause before retry

    slot_id = acquire_slot()

    # Map slots to NUMA-aligned CPU ranges (16 CPUs each)
    cpu_ranges = ["0-15", "16-31", "32-47", "48-63"]
    cpu_set = cpu_ranges[slot_id]

    # Debug logging
    hostname = os.environ.get("SLURMD_NODENAME") or socket.gethostname()
    print(f"[ORCA CPU Pinning] host={hostname} job={job} slot={slot_id} cpus={cpu_set}")

    # Create a simple wrapper script that uses numactl to pin ORCA to specific CPUs
    wrapper_path = os.path.join(root_directory, f"orca_numactl_slot{slot_id}.sh")
    with open(wrapper_path, "w") as f:
        f.write(f"#!/bin/bash\nexec numactl --physcpubind={cpu_set} {orca_cmd} \"$@\"\n")
    os.chmod(wrapper_path, 0o755)

    nbo_tf = False
    root_directory_job = root_directory
    os.makedirs(root_directory_job, exist_ok=True)

    try:
        # Skip if already completed
        results_pkl = os.path.join(root_directory_job, "results.pkl")
        if os.path.exists(results_pkl):
            print(f"Job {job} already completed. Skipping.")
            return

        # Skip if recently modified (basic duplicate prevention)
        # Only check for actual ORCA output files, not our wrapper scripts
        try:
            file_list = os.listdir(root_directory_job)
            orca_output_files = [
                fn for fn in file_list
                if os.path.isfile(os.path.join(root_directory_job, fn)) and
                not fn.startswith('orca_numactl_slot') and  # Ignore our wrapper scripts
                not fn.startswith('orca_wrapper_slot') and  # Ignore old wrapper scripts
                fn not in ['results.pkl']  # Ignore completed results (checked separately above)
            ]
            recent_activity = any(
                os.path.getmtime(os.path.join(root_directory_job, fn)) > time.time() - 900
                for fn in orca_output_files
            )
        except Exception:
            recent_activity = False

        if recent_activity:
            print(f"Job {job} appears to be running (recent ORCA file activity). Skipping.")
            return

        time_start = time.time()

        # Run ORCA with the numactl wrapper for CPU pinning
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
            orca_cmd=wrapper_path,  # Use the numactl wrapper
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
        raise

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
    nprocs: int = 16,
    skip_done: bool = True,
    dry_run: bool = False,
    overwrite: bool = False,
    max_blocks: int = 10,
    walltime_hours: int = 2,
    source_cmd: str = "source ~/.bashrc",
    conda_env: str = "py10mpi",
    conda_base: str = "/usr/WS1/vargas58/miniconda3",
    orca_cmd: str = None,
):
    parsl_config = base_config(
        max_blocks=max_blocks,
        walltime_hours=walltime_hours,
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

        subfolders = [f.path for f in os.scandir(root_data_dir + base_dir) if f.is_dir()]

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
            dict_geoms = process_geometry_file(geom_file, ase_format_tf=ase_format_tf)

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
                folder_mol = os.path.join(folder_out_base, mol_name)
                dict_unified[mol_name]["dir_name"] = folder_mol
                os.makedirs(folder_mol, exist_ok=True)

                error_code = check_job_termination(folder_mol)
                if skip_done and error_code == 1:
                    # Completed job detected
                    continue

                if dry_run:
                    continue

                if ("orca.inp" not in os.listdir(folder_mol)) and (overwrite is False):
                    dict_full_set[mol_name] = dict_unified[mol_name]

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
        for f in futures:
            if should_stop:
                print("Graceful shutdown requested. Exiting before all jobs complete.")
                break
            f.result()
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_data_dir", type=str, default="/Users/santiagovargas/dev/oact_utils/data/big_benchmark/")
    parser.add_argument("--calc_root_dir", type=str, default="/Users/santiagovargas/dev/oact_utils/data/big_benchmark_out/")
    parser.add_argument("--orca_cmd", type=str, required=False, help="Path to Orca executable on cluster (required unless --clean is used)")
    parser.add_argument("--scf_MaxIter", type=int, default=600)
    parser.add_argument("--nprocs", type=int, default=16, help="Number of cores per Orca process (should match cores_per_worker=16)")
    parser.add_argument("--max_blocks", type=int, default=10, dest="max_blocks", help="Maximum number of Slurm nodes to provision")
    parser.add_argument("--walltime_hours", type=int, default=2, dest="walltime_hours", help="Walltime in hours (converted to HH:00:00 format)")
    parser.add_argument("--source_cmd", type=str, default="source ~/.bashrc", dest="source_cmd", help="Command to source bashrc/profile")
    parser.add_argument("--conda_env", type=str, default="py10mpi", dest="conda_env", help="Conda environment name to activate")
    parser.add_argument("--conda_base", type=str, default="/usr/WS1/vargas58/miniconda3", dest="conda_base", help="Base path for conda installation")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--clean", action="store_true", help="Load Parsl config and run cleanup, then exit")
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
        root_data_dir=args.root_data_dir,
        calc_root_dir=args.calc_root_dir,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
