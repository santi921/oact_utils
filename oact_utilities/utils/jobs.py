import os
from typing import Any, List, Dict, Optional

from oact_utilities.utils.status import check_job_termination


def launch_flux_jobs(
    root_dir: str, second_step: bool = False, skip_done: bool = True, dry: bool = False
) -> None:
    # iterate through every subfolder in root_dir
    for folder in os.listdir(root_dir):
        folder_to_use = os.path.join(root_dir, folder)
        
        if os.path.isdir(folder_to_use):
            if skip_done:
                # check if folder has successful flux job
                if check_job_termination(folder_to_use):
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

