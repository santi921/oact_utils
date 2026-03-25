"""Scheduler liveness checks for crash recovery.

Query SLURM or Flux to determine which scheduler jobs are still active.
Used by the dashboard's --recover-orphans command to detect jobs that were
orphaned when a scheduler allocation was killed (SIGKILL, OOM, node crash).
"""

from __future__ import annotations

import os
import subprocess


def get_active_scheduler_jobs(scheduler: str) -> set[str] | None:
    """Get the set of all currently active scheduler job IDs for the current user.

    Makes a single scheduler query to get all active jobs, then returns them
    as a set for O(1) membership testing.

    Args:
        scheduler: Scheduler type ("slurm" or "flux").

    Returns:
        set[str]: Job IDs of active jobs. Empty set means "nothing is running,
            all orphans are confirmed dead."
        None: Could not reach the scheduler (timeout, command not found, etc.).
            Caller MUST skip recovery when None is returned (conservative default).
    """
    try:
        if scheduler == "slurm":
            # squeue -u $USER -h -o "%A" outputs one numeric job ID per line,
            # no header. Only shows active jobs (PENDING, RUNNING, COMPLETING, etc.)
            result = subprocess.run(
                [
                    "squeue",
                    "-u",
                    os.environ.get("USER", ""),
                    "-h",
                    "-o",
                    "%A",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return None
            output = result.stdout.strip()
            return set(output.split("\n")) if output else set()

        elif scheduler == "flux":
            # flux jobs outputs compact alphanumeric IDs (e.g., "f2xgUVYLJs27").
            # --filter=pending,running,completing limits to active jobs.
            # -no "{id}" outputs one job ID per line with no header.
            # These IDs match the $FLUX_JOB_ID env var format.
            result = subprocess.run(
                [
                    "flux",
                    "jobs",
                    "--filter=pending,running,completing",
                    "-no",
                    "{id}",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return None
            output = result.stdout.strip()
            return set(output.split("\n")) if output else set()

        else:
            print(f"Unknown scheduler: {scheduler}")
            return None

    except subprocess.TimeoutExpired:
        return None
    except FileNotFoundError:
        # squeue or flux command not found
        return None
    except OSError:
        return None
