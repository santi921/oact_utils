"""Workflow manager for high-throughput architector calculations.

This module provides utilities to manage large-scale architector job campaigns:
- Track job status (ready, running, completed, failed)
- Generate HPC job submission scripts
- Update status based on output file checks
- Dashboard reporting
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd

from ..utils.architector import create_workflow_db


class JobStatus(str, Enum):
    """Status values for jobs in the workflow."""

    TO_RUN = "to_run"  # Job is ready to be submitted
    READY = "ready"  # Job is queued and ready to run (legacy, use TO_RUN)
    RUNNING = "running"  # Job has been submitted and is running
    COMPLETED = "completed"  # Job finished successfully
    FAILED = "failed"  # Job failed or crashed
    TIMEOUT = "timeout"  # Job timed out without completing


@dataclass
class JobRecord:
    """Represents a single job in the workflow."""

    id: int
    orig_index: int
    elements: str
    natoms: int
    status: JobStatus
    charge: int | None = None
    spin: int | None = None
    geometry: str | None = None
    job_dir: str | None = None
    max_forces: float | None = None
    scf_steps: int | None = None
    final_energy: float | None = None
    error_message: str | None = None
    fail_count: int = 0
    wall_time: float | None = None
    n_cores: int | None = None


class ArchitectorWorkflow:
    """Manage high-throughput architector calculation workflows.

    This class provides methods to:
    - Initialize workflows from architector CSV files
    - Track job statuses in a SQLite database
    - Generate HPC submission scripts
    - Update statuses based on output files
    - Generate dashboard reports
    """

    def __init__(self, db_path: str | Path, timeout: float = 30.0):
        """Initialize workflow manager with existing database.

        Args:
            db_path: Path to the SQLite database file.
            timeout: Timeout in seconds for database locks.
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found at {db_path}")
        self.timeout = timeout
        self.conn = self._get_connection()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with WAL mode enabled."""
        conn = sqlite3.connect(str(self.db_path), timeout=self.timeout)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _execute_with_retry(self, query: str, params: tuple = (), max_retries: int = 3):
        """Execute a query with retry logic for handling database locks.

        Args:
            query: SQL query string.
            params: Query parameters.
            max_retries: Maximum number of retries.

        Returns:
            Cursor after execution.
        """
        for attempt in range(max_retries):
            try:
                cur = self.conn.cursor()
                cur.execute(query, params)
                return cur
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    time.sleep(0.1 * (2**attempt))  # Exponential backoff
                    continue
                raise

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def get_jobs_by_status(
        self, status: JobStatus | list[JobStatus] | None = None
    ) -> list[JobRecord]:
        """Retrieve jobs filtered by status.

        Args:
            status: Single status, list of statuses, or None for all jobs.

        Returns:
            List of JobRecord objects matching the filter.
        """
        if status is None:
            query = "SELECT * FROM structures"
            cur = self._execute_with_retry(query)
        elif isinstance(status, list):
            placeholders = ",".join("?" * len(status))
            query = f"SELECT * FROM structures WHERE status IN ({placeholders})"
            cur = self._execute_with_retry(query, tuple(s.value for s in status))
        else:
            query = "SELECT * FROM structures WHERE status = ?"
            cur = self._execute_with_retry(query, (status.value,))

        rows = cur.fetchall()
        return [
            JobRecord(
                id=r[0],
                orig_index=r[1],
                elements=r[2],
                natoms=r[3],
                status=JobStatus(r[4]),
                charge=r[5],
                spin=r[6],
                geometry=r[7],
                job_dir=r[8],
                max_forces=r[9],
                scf_steps=r[10],
                final_energy=r[11],
                error_message=r[12],
                fail_count=r[13] if len(r) > 13 and r[13] is not None else 0,
                wall_time=r[14] if len(r) > 14 else None,
                n_cores=r[15] if len(r) > 15 else None,
            )
            for r in rows
        ]

    def update_status(
        self,
        job_id: int,
        new_status: JobStatus,
        error_message: str | None = None,
    ):
        """Update the status of a single job.

        Args:
            job_id: Database ID of the job.
            new_status: New status value.
            error_message: Optional error message.
        """
        if error_message is not None:
            query = "UPDATE structures SET status = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?"
            self._execute_with_retry(query, (new_status.value, error_message, job_id))
        else:
            query = "UPDATE structures SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?"
            self._execute_with_retry(query, (new_status.value, job_id))
        self.conn.commit()

    def update_job_metrics(
        self,
        job_id: int,
        job_dir: str | None = None,
        max_forces: float | None = None,
        scf_steps: int | None = None,
        final_energy: float | None = None,
        error_message: str | None = None,
        wall_time: float | None = None,
        n_cores: int | None = None,
    ):
        """Update job metrics (forces, SCF steps, etc).

        Args:
            job_id: Database ID of the job.
            job_dir: Path to job directory.
            max_forces: Maximum force value from optimization.
            scf_steps: Number of SCF steps taken.
            final_energy: Final energy in Hartree.
            error_message: Error message if job failed.
            wall_time: Total wall time in seconds.
            n_cores: Number of CPU cores used.
        """
        updates = []
        values = []

        if job_dir is not None:
            updates.append("job_dir = ?")
            values.append(job_dir)
        if max_forces is not None:
            updates.append("max_forces = ?")
            values.append(max_forces)
        if scf_steps is not None:
            updates.append("scf_steps = ?")
            values.append(scf_steps)
        if final_energy is not None:
            updates.append("final_energy = ?")
            values.append(final_energy)
        if error_message is not None:
            updates.append("error_message = ?")
            values.append(error_message)
        if wall_time is not None:
            updates.append("wall_time = ?")
            values.append(wall_time)
        if n_cores is not None:
            updates.append("n_cores = ?")
            values.append(n_cores)

        if updates:
            updates.append("updated_at = CURRENT_TIMESTAMP")
            query = f"UPDATE structures SET {', '.join(updates)} WHERE id = ?"
            values.append(job_id)
            self._execute_with_retry(query, tuple(values))
            self.conn.commit()

    def update_status_bulk(
        self,
        job_ids: list[int],
        new_status: JobStatus,
    ):
        """Update status for multiple jobs at once.

        Args:
            job_ids: List of database IDs.
            new_status: New status to set for all jobs.
        """
        if not job_ids:
            return

        placeholders = ",".join("?" * len(job_ids))
        query = f"UPDATE structures SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id IN ({placeholders})"
        self._execute_with_retry(query, tuple([new_status.value] + job_ids))
        self.conn.commit()

    def count_by_status(self) -> dict[JobStatus, int]:
        """Count jobs grouped by status.

        Returns:
            Dictionary mapping status to count.
        """
        query = "SELECT status, COUNT(*) FROM structures GROUP BY status"
        cur = self._execute_with_retry(query)
        rows = cur.fetchall()
        return {JobStatus(status): count for status, count in rows}

    def get_summary(self) -> pd.DataFrame:
        """Get summary statistics as a DataFrame.

        Returns:
            DataFrame with status counts and percentages.
        """
        counts = self.count_by_status()
        total = sum(counts.values())
        data = []
        for status in JobStatus:
            count = counts.get(status, 0)
            pct = 100 * count / total if total > 0 else 0
            data.append({"status": status.value, "count": count, "percent": pct})
        return pd.DataFrame(data)

    def mark_jobs_as_running(self, job_ids: list[int]):
        """Mark jobs as running (e.g., after submission to HPC queue).

        Args:
            job_ids: List of job database IDs.
        """
        self.update_status_bulk(job_ids, JobStatus.RUNNING)

    def reset_failed_jobs(
        self, max_fail_count: int | None = None, include_timeout: bool = False
    ):
        """Reset all failed jobs back to ready status for retry.

        Increments the fail_count for each job being reset.

        Args:
            max_fail_count: If specified, only reset jobs with fail_count < max_fail_count.
                Jobs that have already failed this many times will remain in FAILED status.
            include_timeout: If True, also reset TIMEOUT jobs along with FAILED jobs.
        """
        statuses_to_reset = [JobStatus.FAILED.value]
        if include_timeout:
            statuses_to_reset.append(JobStatus.TIMEOUT.value)

        placeholders = ",".join("?" * len(statuses_to_reset))

        if max_fail_count is not None:
            query = f"""
                UPDATE structures
                SET status = ?,
                    error_message = NULL,
                    fail_count = COALESCE(fail_count, 0) + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE status IN ({placeholders}) AND COALESCE(fail_count, 0) < ?
            """
            self._execute_with_retry(
                query,
                tuple([JobStatus.TO_RUN.value] + statuses_to_reset + [max_fail_count]),
            )
        else:
            query = f"""
                UPDATE structures
                SET status = ?,
                    error_message = NULL,
                    fail_count = COALESCE(fail_count, 0) + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE status IN ({placeholders})
            """
            self._execute_with_retry(
                query, tuple([JobStatus.TO_RUN.value] + statuses_to_reset)
            )
        self.conn.commit()

    def reset_timeout_jobs(self, max_fail_count: int | None = None):
        """Reset all timeout jobs back to ready status for retry.

        Increments the fail_count for each job being reset.

        Args:
            max_fail_count: If specified, only reset jobs with fail_count < max_fail_count.
                Jobs that have already timed out this many times will remain in TIMEOUT status.
        """
        if max_fail_count is not None:
            query = """
                UPDATE structures
                SET status = ?,
                    error_message = NULL,
                    fail_count = COALESCE(fail_count, 0) + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE status = ? AND COALESCE(fail_count, 0) < ?
            """
            self._execute_with_retry(
                query, (JobStatus.TO_RUN.value, JobStatus.TIMEOUT.value, max_fail_count)
            )
        else:
            query = """
                UPDATE structures
                SET status = ?,
                    error_message = NULL,
                    fail_count = COALESCE(fail_count, 0) + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE status = ?
            """
            self._execute_with_retry(
                query, (JobStatus.TO_RUN.value, JobStatus.TIMEOUT.value)
            )
        self.conn.commit()

    def make_jobs_failed_with_max_fail_count(self, max_fail_count: int):
        """Mark jobs as failed if they have reached the maximum fail count.
        Args:
            max_fail_count: Maximum allowed fail count before marking as failed.
        """
        query = """
            UPDATE structures
            SET status = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE COALESCE(fail_count, 0) >= ?
        """
        self._execute_with_retry(query, (JobStatus.FAILED.value, max_fail_count))
        self.conn.commit()

    def get_jobs_by_fail_count(self, min_fail_count: int = 1) -> list[JobRecord]:
        """Get jobs that have failed at least min_fail_count times.

        Args:
            min_fail_count: Minimum number of failures (default: 1).

        Returns:
            List of JobRecord objects with fail_count >= min_fail_count.
        """
        query = "SELECT * FROM structures WHERE COALESCE(fail_count, 0) >= ?"
        cur = self._execute_with_retry(query, (min_fail_count,))
        rows = cur.fetchall()
        return [
            JobRecord(
                id=r[0],
                orig_index=r[1],
                elements=r[2],
                natoms=r[3],
                status=JobStatus(r[4]),
                charge=r[5],
                spin=r[6],
                geometry=r[7],
                job_dir=r[8],
                max_forces=r[9],
                scf_steps=r[10],
                final_energy=r[11],
                error_message=r[12],
                fail_count=r[13] if len(r) > 13 and r[13] is not None else 0,
                wall_time=r[14] if len(r) > 14 else None,
                n_cores=r[15] if len(r) > 15 else None,
            )
            for r in rows
        ]


def create_workflow(
    csv_path: str | Path,
    db_path: str | Path,
    geometry_column: str = "aligned_csd_core",
    charge_column: str | None = "charge",
    spin_column: str | None = "uhf",
    batch_size: int = 10000,
) -> tuple[Path, ArchitectorWorkflow]:
    """Initialize a new workflow from an architector CSV file.

    This function:
    1. Creates a SQLite database from the CSV
    2. Returns a workflow manager instance

    Args:
        csv_path: Path to architector CSV file.
        db_path: Path for the SQLite database.
        geometry_column: CSV column containing XYZ geometry strings.
        charge_column: CSV column containing molecular charges.
        spin_column: CSV column containing unpaired electrons.
        batch_size: Number of rows to process at a time.

    Returns:
        Tuple of (db_path, ArchitectorWorkflow instance).
    """
    db_path_ret = create_workflow_db(
        csv_path=csv_path,
        db_path=db_path,
        geometry_column=geometry_column,
        charge_column=charge_column,
        spin_column=spin_column,
        batch_size=batch_size,
    )

    workflow = ArchitectorWorkflow(db_path_ret)
    return db_path_ret, workflow


def update_job_status(
    workflow: ArchitectorWorkflow,
    job_dir: str | Path,
    job_id: int,
    extract_metrics: bool = True,
    unzip: bool = False,
) -> JobStatus:
    """Update a job's status by checking its output files.

    Args:
        workflow: ArchitectorWorkflow instance.
        job_dir: Directory containing job output files.
        job_id: Database ID of the job.
        extract_metrics: If True, extract and store max_forces and scf_steps.
        unzip: If True, handle gzipped output files (quacc).

    Returns:
        The new status of the job.
    """
    from ..utils.analysis import find_timings_and_cores, parse_job_metrics

    job_dir = Path(job_dir)
    if not job_dir.exists():
        workflow.update_status(
            job_id, JobStatus.FAILED, error_message="Job directory not found"
        )
        return JobStatus.FAILED

    # Extract metrics
    if extract_metrics:
        try:
            metrics = parse_job_metrics(job_dir, unzip=unzip)

            new_status = JobStatus.COMPLETED if metrics["success"] else JobStatus.FAILED

            # Extract timing info for successful jobs
            wall_time = None
            n_cores = None
            if new_status == JobStatus.COMPLETED:
                # Find the log file for timing extraction
                log_file = None
                for pattern in ["*.out", "orca.out", "*.log"]:
                    matches = list(job_dir.glob(pattern))
                    if matches:
                        log_file = str(matches[0])
                        break

                if log_file:
                    try:
                        n_cores, time_dict = find_timings_and_cores(log_file)
                        if time_dict and "Total" in time_dict:
                            wall_time = time_dict["Total"]
                    except Exception:
                        # Timing extraction is best-effort, don't fail on errors
                        pass

            workflow.update_job_metrics(
                job_id,
                job_dir=str(job_dir),
                max_forces=metrics.get("max_forces"),
                scf_steps=metrics.get("scf_steps"),
                final_energy=metrics.get("final_energy"),
                error_message=metrics.get("error") if not metrics["success"] else None,
                wall_time=wall_time,
                n_cores=n_cores,
            )

            workflow.update_status(job_id, new_status)
            return new_status

        except Exception as e:
            workflow.update_status(job_id, JobStatus.FAILED, error_message=str(e))
            return JobStatus.FAILED
    else:
        # Just check termination
        from ..utils.status import check_job_termination

        result = check_job_termination(str(job_dir))

        if result == 1:
            new_status = JobStatus.COMPLETED
        elif result == -2:
            new_status = JobStatus.TIMEOUT
        elif result == -1:
            new_status = JobStatus.FAILED
        else:
            new_status = JobStatus.RUNNING

        workflow.update_status(job_id, new_status)
        return new_status
