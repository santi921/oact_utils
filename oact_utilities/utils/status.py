import os
from typing import List, Optional


def check_file_termination(file_path: str) -> int:
    # read last line of file_path
    with open(file_path, "r") as f:
        lines = f.readlines()
    # if last line contains "ORCA TERMINATED NORMALLY", then return 1
    for line in lines[-5:]:
        if "ORCA TERMINATED NORMALLY" in line:
            return 1
        if "aborting the run" in line:
            return -1
    return 0


def check_job_termination(
    dir: str, check_many: bool = False, flux_tf: bool = False
) -> int | bool:
    # sweep folder file for flux*out files
    files = os.listdir(dir)
    # print("files: ", files)
    if flux_tf:
        files_out = [f for f in files if f.startswith("flux") and f.endswith("out")]
    else:
        files_out = [f for f in files if f.endswith("out")]
        if not files_out:
            files_out = [f for f in files if f.endswith("logs")]

    if len(files_out) > 1:
        files_out.sort(
            key=lambda x: os.path.getmtime(os.path.join(dir, x)), reverse=True
        )
        if not check_many:
            files_out = files_out[:1]

    if len(files_out) == 0:
        return False

    if check_many and len(files_out) > 1:
        status_list = []
        for file_out in files_out:
            output_file = dir + "/" + file_out
            # read last line of output_file
            file_status = check_file_termination(output_file)

            status_list.append(file_status)
        if all(status == 1 for status in status_list):
            return 1
        elif any(status == -1 for status in status_list):
            return -1
        else:
            return 0

    else:
        output_file = dir + "/" + files_out[0]

        # read last line of output_file
        with open(output_file, "r") as f:
            lines = f.readlines()
        # if last line contains "ORCA TERMINATED NORMALLY", then get geometry forces
        # ORCA TERMINATED NORMALLY
        for line in lines[-5:]:
            if "ORCA TERMINATED NORMALLY" in line:
                return 1
            if "aborting the run" in line:
                return -1
        return 0


def check_sucessful_jobs(
    root_dir: str, 
    check_many: bool = False, 
    flux_tf: bool = False, 
    verbose: bool = False, 
    check_traj: bool = False,
) -> None:
    
    count_folder = 0
    count_success = 0
    count_still_running = 0
    if check_traj: 
        print("Checking trajectory files as well.")
        traj_count = 0 
    # iterate through every subfolder in root_dir
    for folder in os.listdir(root_dir):
        folder_to_use = os.path.join(root_dir, folder)
        if os.path.isdir(folder_to_use):
            count_folder += 1
            # check if folder has successful flux job
            if (
                check_job_termination(
                    folder_to_use, check_many=check_many, flux_tf=flux_tf
                )
                == 1
            ):
                if verbose:
                    print(f"Job in {folder_to_use} completed successfully.")
                count_success += 1
                if check_traj:
                    # traj file should end with "_trj.xyz"
                    files_in_folder = [f for f in os.listdir(folder_to_use) if f.endswith("_trj.xyz")]
                    if files_in_folder:
                        traj_count += 1
                    else:
                        if verbose:
                            print(f"Trajectory file missing in {folder_to_use}.")
            elif (
                check_job_termination(
                    folder_to_use, check_many=check_many, flux_tf=flux_tf
                )
                == 0
            ):
                count_still_running += 1
                if verbose:
                    print(f"Job in {folder_to_use} is still running or incomplete.")
            else:
                if verbose:
                    print(f"Job in {folder_to_use} did not complete successfully.")
    root_final = root_dir.split("/")[-2]
    # add tabs depending on length of root_final
    root_len = len(root_final)
    tab_count = "\t" * int(4 - int(root_len // 8))
    print(f"Results in {root_final} (S / R / F): {tab_count} {count_success} / {count_still_running} / {count_folder - count_success - count_still_running}")
    if check_traj:
        print(f"Traj Results in {root_final} {tab_count}: {traj_count} / {count_success - traj_count} (With Traj / Without Traj)")


def check_job_termination_whole(root_dir: str, df_multiplicity) -> None:

    job_list = df_multiplicity["molecule"].tolist()

    for job in job_list:
        folder_results = f"{root_dir}/{job}_done"
        # check if folder_results exists
        if os.path.exists(folder_results):
            # iterate and see if there is a file ending in .out or .log
            for file in os.listdir(folder_results):
                if file.endswith("logs"):
                    output_file = os.path.join(folder_results, file)
                    break

            # read last line of output_file
            with open(output_file, "r") as f:
                lines = f.readlines()
            # if last line contains "ORCA TERMINATED NORMALLY", then get geometry forces
            if "ORCA TERMINATED NORMALLY" not in lines[-2]:
                print(f"Job {job} did not terminate normally.")


def pull_log_file(root_dir: str) -> str:
    """
    Pulls the most recent log file from a given directory.

    Args:
        root_dir (str): The directory to search for log files.
    Returns:
        str: The path to the most recent log file.
    """
    try:
        log_file = [f for f in os.listdir(root_dir) if f.endswith("logs")]

        if len(log_file) == 0:
            log_file = [f for f in os.listdir(root_dir) if f.endswith(".out")]

        if len(log_file) > 1:
            log_file.sort(
                key=lambda x: os.path.getmtime(os.path.join(root_dir, x)),
                reverse=True,
            )
        log_file = os.path.join(root_dir, log_file[0])

    except:
        # check for "flux-"
        # get all files that contains  flux-
        files_flux = [f for f in os.listdir(root_dir) if "flux-" in f]
        files_flux.sort(
            key=lambda x: os.path.getmtime(os.path.join(root_dir, x)),
            reverse=True,
        )
        log_file = os.path.join(root_dir, files_flux[0])
    # if type(log_file) is list:
    #    log_file = log_file[0]

    return log_file
