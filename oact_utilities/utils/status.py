from __future__ import annotations

import gzip
import os
import re
import time
from collections import deque

# Pattern matching ORCA's atomic SCF guess files (e.g. orca_atom0.out, orca_atom12.out)
_ORCA_ATOM_RE = re.compile(r"^orca_atom\d+\.out$")


def _is_orca_atom_scf(filename: str) -> bool:
    """Return True if filename is an ORCA atomic SCF initial-guess output."""
    return _ORCA_ATOM_RE.match(os.path.basename(filename)) is not None


def _read_last_lines(file_path: str, maxlen: int = 10) -> list[str]:
    """Read the last N lines of a file, handling gzip and encoding errors.

    Args:
        file_path: Path to the file (.out or .out.gz).
        maxlen: Number of trailing lines to return.

    Returns:
        List of the last N lines. Empty list on read failure.
    """
    try:
        if file_path.endswith(".gz"):
            with gzip.open(file_path, "rt", errors="replace") as f:
                return list(deque(f, maxlen=maxlen))
        else:
            with open(file_path, errors="replace") as f:
                return list(deque(f, maxlen=maxlen))
    except (OSError, UnicodeDecodeError, EOFError):
        return []


def parse_failure_reason(file_path: str) -> str | None:
    """Extract the failure reason from an ORCA output file.

    Scans the last 10 lines for known failure patterns
    ("aborting the run", "Error"). Returns the first matching line
    (stripped), or None if no failure pattern is found.

    Args:
        file_path: Path to the ORCA output file (.out or .out.gz).

    Returns:
        The failure line text, or None if no failure detected.
    """
    last_lines = _read_last_lines(file_path)
    if not last_lines:
        return None

    # If the job succeeded, there is no failure reason
    if any("ORCA TERMINATED NORMALLY" in line for line in last_lines):
        return None

    for line in last_lines:
        if "aborting the run" in line:
            return line.strip()
        if "Error" in line:
            return line.strip()
    return None


def check_file_termination(
    file_path: str, is_gzipped: bool = False, hours_cutoff=6
) -> int:
    """Check if an ORCA calculation terminated successfully.

    Args:
        file_path: Path to the ORCA output file (.out or .out.gz).
        is_gzipped: If True, decompress the file before reading. Auto-detected if None.

    Returns:
        1 if terminated normally, -1 if aborted, -2 if timeout, 0 if still running/incomplete.
    """
    # Auto-detect gzipped files if not specified
    if is_gzipped is None:
        is_gzipped = file_path.endswith(".gz")

    last_lines = _read_last_lines(file_path)
    if not last_lines:
        # Read failure (corrupted, inaccessible) -- treat as still running
        return 0

    # Check last 10 lines for definitive termination status.
    # Content-based checks take priority over file age so that completed
    # or failed jobs keep their correct status regardless of how old they are.
    #
    # IMPORTANT: scan ALL lines for success first, then for failure.
    # ORCA can print benign lines containing "Error" (e.g.,
    # "Some of the Onep Onebody Integrals have an Error larger than 1.0e-05")
    # before the final "ORCA TERMINATED NORMALLY" line. A single-pass loop
    # would short-circuit on the "Error" line and miss the success signal.
    if any("ORCA TERMINATED NORMALLY" in line for line in last_lines):
        return 1
    for line in last_lines:
        if "aborting the run" in line:
            return -1
        if "Error" in line:
            return -1

    # No definitive signal found -- check file age to detect stale/timed-out jobs
    if os.path.getmtime(file_path) < (time.time() - hours_cutoff * 3600):
        return -2

    return 0


def done_geo_opt_ase(opt_log_file, fmax_cutoff=0.01):
    """
    Check if geometry optimization is done based on final forces using ASE.
    Args:
        opt_log_file (str): Path to the optimization log file.
        fmax_cutoff (float, optional): Force maximum cutoff to consider optimization done. Defaults to 0.01.        Returns:
    Returns:
        bool: True if optimization is done, False otherwise.
    """

    try:
        # iterate thgouth lines to find final forces
        with open(opt_log_file, errors="replace") as f:
            lines = f.readlines()

        forces = []
        for line in lines:
            # check if it's a float
            if line.split()[-1].replace(".", "", 1).isdigit():
                forces.append(float(line.split()[3]))
        if len(forces) == 0:
            return False
        force_check = forces[-1]
        if force_check < fmax_cutoff:
            return True
        else:
            return False
    except (OSError, ValueError, IndexError):
        return False


def _check_sella_termination(dir: str, hours_cutoff: int = 6) -> int:
    """Check Sella optimization termination via sella_status.txt.

    The sella_status.txt file is the authoritative completion signal written
    by run_sella_optimization() in sella_runner.py.

    Args:
        dir: Path to the job directory.
        hours_cutoff: Hours before considering a job timed out.

    Returns:
        1 if converged, -1 if error or not converged, -2 if timeout,
        0 if still running.
    """
    status_file = os.path.join(dir, "sella_status.txt")
    if os.path.exists(status_file):
        try:
            with open(status_file, errors="replace") as f:
                content = f.read()
            if "CONVERGED" in content and "NOT_CONVERGED" not in content:
                return 1
            if "NOT_CONVERGED" in content or "ERROR" in content:
                return -1
        except OSError:
            pass

    # No status file yet — check if sella.log exists and is stale
    sella_log = os.path.join(dir, "sella.log")
    if os.path.exists(sella_log):
        if os.path.getmtime(sella_log) < (time.time() - hours_cutoff * 3600):
            return -2
        return 0

    # No Sella artifacts at all — fall back to directory age
    if os.path.getmtime(dir) < (time.time() - hours_cutoff * 3600):
        return -2
    return 0


def check_job_termination(
    dir: str,
    check_many: bool = False,
    flux_tf: bool = False,
    hours_cutoff=6,
    optimizer: str | None = None,
) -> int:
    """Check if a job in a given directory has terminated successfully.

    Supports both regular (.out, .logs) and gzipped (.out.gz) ORCA output files.
    When optimizer="sella", checks sella_status.txt instead of ORCA output.

    Args:
        dir: Path to the directory containing the job.
        check_many: Whether to check multiple output files. Defaults to False.
        flux_tf: Whether to check for flux output files. Defaults to False.
        hours_cutoff: Hours before considering a job timed out.
        optimizer: Optimizer engine ("orca", "sella", or None). When "sella",
            uses sella_status.txt as the authoritative termination signal.

    Returns:
        1 if the job terminated successfully, -1 if it failed,
        -2 if timeout, 0 if still running or incomplete.
    """
    # Sella jobs: check sella_status.txt instead of ORCA output.
    # Auto-detect when optimizer is None by looking for run_sella.py marker.
    if optimizer == "sella" or (
        optimizer is None and os.path.exists(os.path.join(dir, "run_sella.py"))
    ):
        return _check_sella_termination(dir, hours_cutoff=hours_cutoff)

    # sweep folder file for flux*out files
    files = os.listdir(dir)
    # print("files: ", files)
    if flux_tf:
        files_out = [f for f in files if f.startswith("flux") and f.endswith("out")]
    else:
        # Check for regular .out files (skip ORCA atomic SCF guess files)
        files_out = [f for f in files if f.endswith("out") and not _is_orca_atom_scf(f)]

        # Check for gzipped .out.gz files (e.g., from quacc)
        if not files_out:
            files_out = [f for f in files if f.endswith(".out.gz")]

        # Check for .logs files
        if not files_out:
            files_out = [f for f in files if f.endswith("logs")]

    if len(files_out) > 1:
        files_out.sort(
            key=lambda x: os.path.getmtime(os.path.join(dir, x)), reverse=True
        )
        if not check_many:
            files_out = files_out[:1]

    if len(files_out) == 0:
        # No output files found — check if the directory itself is stale
        if os.path.getmtime(dir) < (time.time() - hours_cutoff * 3600):
            return -2
        return 0

    if check_many and len(files_out) > 1:
        status_list = []
        for file_out in files_out:
            output_file = os.path.join(str(dir), file_out)
            # Use check_file_termination which handles both regular and gzipped files
            file_status = check_file_termination(output_file, hours_cutoff=hours_cutoff)
            status_list.append(file_status)
        # check all these files, if most recent file edit is more than 24 hours ago, then consider it failed

        if all(status == 1 for status in status_list):
            return 1
        elif any(status == -2 for status in status_list):
            return -2
        elif any(status == -1 for status in status_list):
            return -1
        else:
            return 0

    else:
        output_file = os.path.join(str(dir), files_out[0])
        # Use check_file_termination which auto-detects gzipped files
        return check_file_termination(output_file, hours_cutoff=hours_cutoff)


def check_geometry_steps(
    dir: str, check_many: bool = False, flux_tf: bool = False, hours_cutoff=6
) -> int:
    """
    Utility function to check if a job in a given directory has gone beyond 1 geometry optimization step.
    Args:
        dir (str): Path to the directory containing the job.
        check_many (bool, optional): Whether to check multiple output files. Defaults to False.
        flux_tf (bool, optional): Whether to check for flux output files. Defaults to False
    Returns:
        int: True if the job went beyond 1 geometry optimization step, False otherwise.
    """
    # sweep folder file for flux*out files
    files = os.listdir(dir)
    # print("files: ", files)
    if flux_tf:
        files_out = [f for f in files if f.startswith("flux") and f.endswith("out")]
    else:
        files_out = [f for f in files if f.endswith("out") and not _is_orca_atom_scf(f)]
        # also check slurm-*.log files
        files_out += [f for f in files if f.startswith("slurm-") and f.endswith(".log")]
        if not files_out:
            files_out = [f for f in files if f.endswith("logs")]

    if len(files_out) > 1:
        files_out.sort(
            key=lambda x: os.path.getmtime(os.path.join(dir, x)), reverse=True
        )
        if not check_many:
            files_out = files_out[:1]

    if len(files_out) == 0:
        return 0

    if check_many and len(files_out) > 1:
        status_list = []
        for file_out in files_out:
            output_file = dir + "/" + file_out
            # read last line of output_file
            file_status = check_file_termination(output_file, hours_cutoff=hours_cutoff)

            status_list.append(file_status)
        if all(status == 1 for status in status_list):
            return 1
        elif any(status == -1 for status in status_list):
            return -1
        else:
            return 0

    else:
        output_file = dir + "/" + files_out[0]

        try:
            # Efficiently find last occurrence of "GEOMETRY OPTIMIZATION CYCLE"
            # without loading entire file into memory
            # example line: *                GEOMETRY OPTIMIZATION CYCLE   1            *
            last_cycle_line = None
            with open(output_file, errors="replace") as f:
                for line in f:
                    if "GEOMETRY OPTIMIZATION CYCLE" in line:
                        last_cycle_line = line

            if last_cycle_line:
                try:
                    cycle_number = int(last_cycle_line.split()[-2])
                    return cycle_number > 1
                except (ValueError, IndexError):
                    return False
            return False
        except OSError:
            return False


def pull_log_file(root_dir: str) -> str:
    """
    Pulls the most recent log file from a given directory.

    Supports both regular (.out, .logs) and gzipped (.out.gz) output files.

    Args:
        root_dir (str): The directory to search for log files.
    Returns:
        str: The path to the most recent log file.
    Raises:
        FileNotFoundError: If no log files are found in the directory.
    """
    try:
        files = os.listdir(root_dir)
    except (FileNotFoundError, PermissionError) as e:
        # raise .. from err
        raise FileNotFoundError(f"Could not access directory {root_dir}: {e}") from e
    # Try to find .logs files first
    log_file = [f for f in files if f.endswith("logs")]

    # If none, try .out files (skip ORCA atomic SCF guess files)
    if len(log_file) == 0:
        # Prefer orca.out explicitly -- Sella jobs produce both orca.out and
        # sella_run.out, and we want the ORCA output for metric extraction.
        if "orca.out" in files:
            log_file = ["orca.out"]
        else:
            log_file = [
                f for f in files if f.endswith(".out") and not _is_orca_atom_scf(f)
            ]

    # If none, try .out.gz files (e.g., from quacc)
    if len(log_file) == 0:
        log_file = [f for f in files if f.endswith(".out.gz")]

    # If still none, try flux files
    if len(log_file) == 0:
        log_file = [f for f in files if "flux-" in f]

    # If we still have no log files, raise an error
    if len(log_file) == 0:
        raise FileNotFoundError(
            f"No log files (.logs, .out, .out.gz, or flux-*) found in {root_dir}"
        )

    # If multiple files, sort by modification time (most recent first)
    if len(log_file) > 1:
        log_file.sort(
            key=lambda x: os.path.getmtime(os.path.join(root_dir, x)),
            reverse=True,
        )

    return os.path.join(root_dir, log_file[0])


def check_sella_complete(root_dir: str, fmax: float = 0.05) -> int:
    """Check if a Sella optimization completed successfully.

    Prefers sella_status.txt (written by sella_runner.py) when available,
    falling back to parsing sella.log for legacy jobs.

    Args:
        root_dir: The directory containing Sella output files.
        fmax: Force convergence threshold (used only for sella.log fallback).

    Returns:
        1 if converged, 0 otherwise.
    """
    # Prefer sella_status.txt (new jobs from sella_runner.py)
    status_file = os.path.join(root_dir, "sella_status.txt")
    if os.path.exists(status_file):
        try:
            with open(status_file, errors="replace") as f:
                content = f.read()
            if "CONVERGED" in content and "NOT_CONVERGED" not in content:
                return 1
            return 0
        except OSError:
            pass

    # Fallback: parse sella.log for legacy jobs without sella_status.txt
    sella_log = os.path.join(root_dir, "sella.log")
    if not os.path.exists(sella_log):
        return 0

    try:
        with open(sella_log, errors="replace") as f:
            lines = f.readlines()
        forces = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                val = parts[4].replace(".", "", 1).replace("-", "", 1)
                if val.isdigit():
                    forces.append(float(parts[4]))

        if len(forces) == 0:
            return 0
        return 1 if forces[-1] < fmax else 0
    except (OSError, ValueError, IndexError):
        return 0


################################################################
# Aggregate functions to check all jobs in a root directory and report the status of each job
################################################################


def check_sucessful_jobs_sella(
    root_dir: str,
    verbose: bool = False,
    fmax: float = 0.05,
) -> None:
    """
    Utility function to check and report the status of jobs in a given root directory.
    Args:
        root_dir (str): Path to the root directory containing job subfolders.
        check_many (bool, optional): Whether to check multiple output files per job. Defaults to False.
        flux_tf (bool, optional): Whether to check for flux output files. Defaults to False.
        verbose (bool, optional): Whether to print detailed status messages. Defaults to False.
        check_traj (bool, optional): Whether to check for trajectory files. Defaults to False.
    Returns:
        None
    """

    count_folder = 0
    count_success = 0
    count_still_running = 0
    # count_geom_beyond_1_then_fail = 0
    # count_geom_beyond_1 = 0
    # iterate through every subfolder in root_dir

    for folder in os.listdir(root_dir):
        folder_to_use = os.path.join(root_dir, folder)
        if os.path.isdir(folder_to_use):
            count_folder += 1
            sella_log_tf = check_sella_complete(folder_to_use, fmax=fmax)
            dft_log_tf = check_job_termination(
                folder_to_use, check_many=False, flux_tf=False
            )

            # check if folder has successful flux job
            if sella_log_tf:
                if verbose:
                    print(f"Job in {folder_to_use} completed successfully.")
                count_success += 1

            elif dft_log_tf == 0:
                count_still_running += 1
                if verbose:
                    print(f"Job in {folder_to_use} is still running or incomplete.")

            elif dft_log_tf == -1:

                if verbose:
                    print(f"Job in {folder_to_use} did not complete successfully.")

    root_final = root_dir.split("/")[-2]
    # add tabs depending on length of root_final
    root_len = len(root_final)
    tab_count = "\t" * int(4 - int(root_len // 8))
    print(
        f"Results in {root_final} (S / R / F): {tab_count} {count_success} / {count_still_running} / {count_folder - count_success - count_still_running}"
    )


def check_job_termination_whole(root_dir: str, df_multiplicity) -> None:
    """
    Utility function to check the termination status of jobs listed in a dataframe.
    Args:
        root_dir (str): Path to the root directory containing job subfolders.
        df_multiplicity (DataFrame): DataFrame containing job information with a 'molecule
        ' column.
    Returns:
        None
    """
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

            try:
                # read last line of output_file
                with open(output_file, errors="replace") as f:
                    lines = f.readlines()
                # if last line contains "ORCA TERMINATED NORMALLY", then get geometry forces
                if "ORCA TERMINATED NORMALLY" not in lines[-2]:
                    print(f"Job {job} did not terminate normally.")
            except (OSError, IndexError):
                print(f"Could not read output file for job {job}.")


def check_sucessful_jobs(
    root_dir: str,
    check_many: bool = False,
    flux_tf: bool = False,
    verbose: bool = False,
    check_traj: bool = False,
) -> None:
    """
    Utility function to check and report the status of jobs in a given root directory.
    Args:
        root_dir (str): Path to the root directory containing job subfolders.
        check_many (bool, optional): Whether to check multiple output files per job. Defaults to False.
        flux_tf (bool, optional): Whether to check for flux output files. Defaults to False.
        verbose (bool, optional): Whether to print detailed status messages. Defaults to False.
        check_traj (bool, optional): Whether to check for trajectory files. Defaults to False.
    Returns:
        None
    """

    count_folder = 0
    count_success = 0
    count_still_running = 0
    count_geom_beyond_1_then_fail = 0
    count_geom_beyond_1 = 0

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
                    files_in_folder = [
                        f for f in os.listdir(folder_to_use) if f.endswith("_trj.xyz")
                    ]
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
                if check_geometry_steps(
                    folder_to_use, check_many=check_many, flux_tf=flux_tf
                ):
                    count_geom_beyond_1 += 1

                if verbose:
                    print(f"Job in {folder_to_use} is still running or incomplete.")
            else:
                if check_geometry_steps(
                    folder_to_use, check_many=check_many, flux_tf=flux_tf
                ):
                    count_geom_beyond_1_then_fail += 1

                if verbose:
                    print(f"Job in {folder_to_use} did not complete successfully.")

    root_final = root_dir.split("/")[-2]
    # add tabs depending on length of root_final
    root_len = len(root_final)
    tab_count = "\t" * int(4 - int(root_len // 8))
    print(
        f"Results in {root_final} (S / R / F): {tab_count} {count_success} / {count_still_running} / {count_folder - count_success - count_still_running}"
    )
    if check_traj:
        print(
            f"Traj Results in {root_final} {tab_count}: {traj_count} / {count_success - traj_count} (With Traj / Without Traj)"
        )
        print(
            f"Geometry optimization >1 step (running / fail): {count_geom_beyond_1} / {count_geom_beyond_1_then_fail}"
        )
