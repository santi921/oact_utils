import os
from typing import Any, List, Dict, Optional

from oact_utilities.utils.status import check_job_termination


def launch_flux_jobs(
    root_dir: str,
    second_step: bool = False,
    skip_done: bool = True,
    skip_failed: bool = False,
    dry: bool = False,
    verbose: bool = False,
) -> None:
    """
    Utility to launch flux jobs in all subdirectories of a given root directory.
    It checks for the presence of job input files and whether jobs have already been completed.
    Args:
        root_dir (str): Path to the root directory containing job subdirectories.
        second_step (bool, optional): If True, launches the second step (tight) jobs.
                                       Defaults to False.
        skip_done (bool, optional): If True, skips directories with completed jobs. Defaults to True.
        skip_failed (bool, optional): If True, skips directories with failed jobs. Defaults to False.
        dry (bool, optional): If True, performs a dry run without executing commands. Defaults to False.
        verbose (bool, optional): If True, prints detailed information during execution. Defaults to False.
    Returns:
        None
    """

    # iterate through every subfolder in root_dir
    for folder in os.listdir(root_dir):
        folder_to_use = os.path.join(root_dir, folder)
        # safely deal with (, ?, and ) in folder names
        folder_to_use = (
            folder_to_use.replace("(", "\\(").replace(")", "\\)").replace("?", "\\?")
        )

        if os.path.isdir(folder_to_use):
            if skip_done:
                # check if folder has successful flux job
                if check_job_termination(folder_to_use):
                    if verbose:
                        print(f"Skipping {folder_to_use} - succcessful job found.")
                    if skip_failed:
                        # check if folder has failed flux job
                        if check_job_termination(folder_to_use) == -1:
                            if verbose:
                                print(f"Skipping {folder_to_use} - failed job found.")
                            continue
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


def launch_slurm_jobs(
    root_dir: str,
    second_step: bool = False,
    skip_done: bool = True,
    dry: bool = False,
    verbose: bool = False,
) -> None:
    """
    Utility to launch SLURM jobs in all subdirectories of a given root directory.
    It checks for the presence of job scripts and whether jobs have already been completed.
    Args:
        root_dir (str): Path to the root directory containing job subdirectories.
        second_step (bool, optional): If True, launches the second step (tight) jobs.
                                       Defaults to False.
        skip_done (bool, optional): If True, skips directories with completed jobs. Defaults to True.
        dry (bool, optional): If True, performs a dry run without executing commands. Defaults to False.
        verbose (bool, optional): If True, prints detailed information during execution. Defaults to False.
    Returns:
        None
    """
    # iterate through every subfolder in root_dir
    for folder in os.listdir(root_dir):
        folder_to_use = os.path.join(root_dir, folder)
        # safely deal with (, ?, and ) in folder names
        folder_to_use = (
            folder_to_use.replace("(", "\\(").replace(")", "\\)").replace("?", "\\?")
        )

        if os.path.isdir(folder_to_use):
            if skip_done:
                # check if folder has successful flux job
                if check_job_termination(folder_to_use):
                    if verbose:
                        print(f"Skipping {folder_to_use} as it has a completed job.")
                    continue

            # check for flux_job.inp or flux_job_loose.inp and flux_job_tight.inp
            if os.path.exists(os.path.join(folder_to_use, "slurm_job.sh")):
                print(f"Launching job in {folder_to_use}")
                command = f"cd {folder_to_use} && sbatch slurm_job.sh"

                if not dry:
                    os.system(command)

            elif os.path.exists(
                os.path.join(folder_to_use, "slurm_job_loose.sh")
            ) and os.path.exists(os.path.join(folder_to_use, "slurm_job_tight.sh")):
                if second_step:
                    print(f"Launching tight job in {folder_to_use}")
                    command_tight = f"cd {folder_to_use} && sbatch slurm_job_tight.sh"

                    if not dry:
                        os.system(command_tight)
                else:
                    print(f"Launching loose job in {folder_to_use}")
                    command_loose = f"cd {folder_to_use} && sbatch slurm_job_loose.sh"

                    if not dry:
                        os.system(command_loose)
