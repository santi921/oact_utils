"""Scheduler liveness checks for crash recovery.

Query SLURM or Flux to determine which scheduler jobs are still active.
Used by the dashboard's --recover-orphans command to detect jobs that were
orphaned when a scheduler allocation was killed (SIGKILL, OOM, node crash).
"""

from __future__ import annotations

import getpass
import os
import subprocess

# Scheduler commands that list active job IDs (one per line, no header).
# SLURM: squeue -u <user> -h -o "%A" outputs numeric job IDs.
# Flux: flux jobs outputs compact alphanumeric IDs (e.g., "f2xgUVYLJs27")
#       that match the $FLUX_JOB_ID env var format.
_SCHEDULER_COMMANDS: dict[str, list[str]] = {
    "slurm": ["squeue", "-u", "{user}", "-h", "-o", "%A"],
    "pbspro": ["qstat", "-u", "{user}"],
    "flux": ["flux", "jobs", "--filter=pending,running,completing", "-no", "{{id}}"],
}


def get_active_scheduler_jobs(scheduler: str) -> set[str] | None:
    """Get the set of all currently active scheduler job IDs for the current user.

    Makes a single scheduler query to get all active jobs, then returns them
    as a set for O(1) membership testing.

    Args:
        scheduler: Scheduler type ("slurm", "pbspro", or "flux").

    Returns:
        set[str]: Job IDs of active jobs. Empty set means "nothing is running,
            all orphans are confirmed dead."
        None: Could not reach the scheduler (timeout, command not found, etc.).
            Caller MUST skip recovery when None is returned (conservative default).
    """
    cmd_template = _SCHEDULER_COMMANDS.get(scheduler)
    if cmd_template is None:
        print(f"Unknown scheduler: {scheduler}")
        return None

    # Determine current user for SLURM's -u flag. Falls back to
    # getpass.getuser() which uses pwd.getpwuid(os.getuid()) when $USER
    # is unset (containers, cron). Without a user identity, squeue -u ""
    # would return all users' jobs, causing false-negative orphan detection.
    user = os.environ.get("USER") or getpass.getuser()
    if not user:
        return None  # Cannot determine user -- skip recovery

    # Substitute {user} placeholder in the command template
    cmd = [arg.format(user=user) for arg in cmd_template]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None
        output = result.stdout.strip()
        if not output:
            return set()

        if scheduler == "pbspro":
            active_jobs: set[str] = set()
            for line in output.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                first_token = stripped.split()[0]
                if first_token in {"Job", "---"}:
                    continue
                if first_token[0].isdigit():
                    active_jobs.add(first_token)
            return active_jobs

        return set(output.split("\n"))
    except (subprocess.TimeoutExpired, OSError):
        # OSError covers FileNotFoundError (command not found) and other
        # OS-level errors (permissions, etc.)
        return None
