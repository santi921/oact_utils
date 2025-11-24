import os
from typing import Any, List, Dict, Optional

from oact_utilities.utils.status import check_job_termination


def launch_flux_jobs(
    root_dir: str,
    second_step: bool = False,
    skip_done: bool = True,
    dry: bool = False,
    verbose: bool = False,
) -> None:
    # iterate through every subfolder in root_dir
    for folder in os.listdir(root_dir):
        folder_to_use = os.path.join(root_dir, folder)
        # safely deal with (, ?, and ) in folder names
        folder_to_use = folder_to_use.replace("(", "\\(").replace(")", "\\)").replace("?", "\\?")

        if os.path.isdir(folder_to_use):
            if skip_done:
                # check if folder has successful flux job
                if check_job_termination(folder_to_use):
                    if verbose:
                        print(f"Skipping {folder_to_use} as it has a completed job.")
                    continue

            # check for flux_job.inp or flux_job_loose.inp and flux_job_tight.inp
            if os.path.exists(os.path.join(folder_to_use, "flux_job.flux")):
                print(f"Launching job in {folder_to_use}")
                command = f"cd {folder_to_use} && flux batch flux_job.flux"

                if not dry:
                    os.system(command)

            elif os.path.exists(
                os.path.join(folder_to_use, "flux_job_loose.inp")
            ) and os.path.exists(os.path.join(folder_to_use, "flux_job_tight.flux")):
                if second_step:
                    print(f"Launching tight job in {folder_to_use}")
                    command_tight = (
                        f"cd {folder_to_use} && flux batch flux_job_tight.flux"
                    )

                    if not dry:
                        os.system(command_tight)
                else:
                    print(f"Launching loose job in {folder_to_use}")
                    command_loose = (
                        f"cd {folder_to_use} && flux batch flux_job_loose.flux"
                    )

                    if not dry:
                        os.system(command_loose)





#!/bin/bash -l
#PBS -l select=1:system=crux
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -q workq-route
#PBS -A Generator

# Change the directory to work directory, which is the directory you submit the job.
cd ${PBS_O_WORKDIR}

module use /soft/modulefiles
module load conda

conda activate generator

# local
full-runner-parsl-alcf --num_folders 18 --orca_2mkl_cmd $HOME/orca_6_0_0/orca_2mkl \
     --multiwfn_cmd $HOME/Multiwfn_3_8/Multiwfn_noGUI --clean --job_file \
    ./test.txt --full_set 1 --type_runner local      \
    --n_threads 10 --safety_factor 5 --move_results --preprocess_compressed \
      --overwrite --job_file ./jobs_by_topdir/noble_gas_compounds.txt \
    --preprocess_compressed --root_omol_results /lus/eagle/projects/generator/OMol25_postprocessing/ --root_omol_inputs /lus/eagle/projects/OMol25/

# as a pilot job
full-runner-parsl-alcf --num_folders 18 --orca_2mkl_cmd $HOME/orca_6_0_0/orca_2mkl \
     --multiwfn_cmd $HOME/Multiwfn_3_8/Multiwfn_noGUI --clean --job_file \
    ./test.txt --full_set 1 --type_runner hpc      \
     --n_threads 8 --safety_factor 3 --move_results --preprocess_compressed --timeout_hr 1 \
    --queue workq-route  --overwrite             --n_nodes 1 --type_runner hpc --job_file ./job_lists/noble_gas_compounds_zst_folders.txt \
    --preprocess_compressed --root_omol_results /lus/eagle/projects/generator/OMol25_postprocessing/ --root_omol_inputs /lus/eagle/projects/OMol25/

