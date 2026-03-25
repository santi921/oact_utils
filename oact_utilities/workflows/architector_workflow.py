"""Workflow manager for high-throughput architector calculations.

This module provides utilities to manage large-scale architector job campaigns:
- Track job status (ready, running, completed, failed)
- Generate HPC job submission scripts
- Update status based on output file checks
- Dashboard reporting
"""

from __future__ import annotations

import random
import re
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
    optimizer: str | None = None
    worker_id: str | None = None


class ArchitectorWorkflow:
    """Manage high-throughput architector calculation workflows.

    This class provides methods to:
    - Initialize workflows from architector CSV files
    - Track job statuses in a SQLite database
    - Generate HPC submission scripts
    - Update statuses based on output files
    - Generate dashboard reports
    """

    def __init__(self, db_path: str | Path, timeout: float = 5.0):
        """Initialize workflow manager with existing database.

        Args:
            db_path: Path to the SQLite database file.
            timeout: SQLite busy-wait timeout in seconds per lock attempt.
                Kept short so the retry loop can fail fast for interactive
                use. Parsl workers can pass a higher value if needed.
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found at {db_path}")
        self.timeout = timeout
        self.conn = self._get_connection()
        self._ensure_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with WAL mode enabled if possible.

        Falls back to DELETE journal mode on network filesystems (NFS, Lustre)
        that don't support WAL locking protocols. If both journal mode pragmas
        fail (common on Lustre where POSIX locking is broken), skips the
        pragma entirely -- SQLite defaults to DELETE mode anyway.

        Uses sqlite3.Row as the row factory so columns can be accessed by
        name instead of positional index. This eliminates fragile positional
        indexing that breaks when columns are added via ALTER TABLE.
        """
        conn = sqlite3.connect(str(self.db_path), timeout=self.timeout)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            # Network filesystem (NFS/Lustre) can fail with various
            # errors: "locking protocol", "database is locked", etc.
            try:
                conn.execute("PRAGMA journal_mode=DELETE")
            except sqlite3.OperationalError:
                # Lustre POSIX locking completely broken for this path.
                # Skip the pragma -- SQLite defaults to DELETE mode.
                pass
        return conn

    def _ensure_schema(self) -> None:
        """Auto-migrate schema by adding missing columns.

        Adds columns that may not exist in older databases:
        - ``optimizer`` TEXT (added in v1)
        - ``worker_id`` TEXT (added in v2 -- scheduler job ID for crash recovery)

        Also migrates legacy ``ready`` status values to ``to_run``.

        Safe under concurrent access: catches the duplicate column error
        that arises when another process adds the column between our
        PRAGMA check and the ALTER TABLE statement.
        """
        cur = self._execute_with_retry("PRAGMA table_info(structures)")
        existing_cols = {row[1] for row in cur.fetchall()}
        if not existing_cols:
            raise RuntimeError(
                f"Database at {self.db_path} has no 'structures' table. "
                "It may be empty or corrupted. Recreate it with "
                "create_workflow_db()."
            )

        # Add missing columns (each wrapped individually for concurrent safety)
        migrations = {
            "optimizer": "ALTER TABLE structures ADD COLUMN optimizer TEXT DEFAULT NULL",
            "worker_id": "ALTER TABLE structures ADD COLUMN worker_id TEXT DEFAULT NULL",
        }
        for col_name, alter_sql in migrations.items():
            if col_name not in existing_cols:
                try:
                    self._execute_with_retry(alter_sql)
                    self._commit_with_retry()
                except sqlite3.OperationalError as e:
                    if "duplicate column name" in str(e).lower():
                        pass  # Another process already added the column
                    else:
                        raise

        # Migrate legacy "ready" status to "to_run" (idempotent)
        self._execute_with_retry(
            "UPDATE structures SET status = ? WHERE status = ?",
            (JobStatus.TO_RUN.value, JobStatus.READY.value),
        )
        self._commit_with_retry()

    @staticmethod
    def _is_retryable(e: sqlite3.OperationalError) -> bool:
        """Check if an OperationalError is a transient lock/busy error."""
        msg = str(e).lower()
        return any(
            token in msg
            for token in ("lock", "busy", "locking protocol", "database is locked")
        )

    def _execute_with_retry(self, query: str, params: tuple = (), max_retries: int = 5):
        """Execute a query with retry logic for handling database locks.

        Uses exponential backoff with jitter to handle concurrent access
        during long-running Parsl workflows. For write statements (INSERT,
        UPDATE, DELETE), uses BEGIN IMMEDIATE to acquire the write lock
        early and avoid late SQLITE_BUSY errors at commit time.

        Args:
            query: SQL query string.
            params: Query parameters.
            max_retries: Maximum number of retries.

        Returns:
            Cursor after execution.
        """
        is_write = query.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE"))
        for attempt in range(max_retries):
            try:
                cur = self.conn.cursor()
                if is_write and not self.conn.in_transaction:
                    cur.execute("BEGIN IMMEDIATE")
                cur.execute(query, params)
                return cur
            except sqlite3.OperationalError as e:
                if self._is_retryable(e) and attempt < max_retries - 1:
                    # Roll back any half-started transaction before retrying
                    try:
                        self.conn.rollback()
                    except Exception:
                        pass
                    delay = min(0.1 * (2**attempt), 5.0)
                    jitter = random.uniform(0, delay * 0.2)
                    time.sleep(delay + jitter)
                    continue
                raise

    def _commit_with_retry(self, max_retries: int = 5):
        """Commit with retry logic for handling database locks.

        The conn.commit() call can also raise 'database is locked' when
        another process holds the write lock (e.g. Parsl workers updating
        job statuses concurrently).

        Args:
            max_retries: Maximum number of retries.
        """
        for attempt in range(max_retries):
            try:
                self.conn.commit()
                return
            except sqlite3.OperationalError as e:
                if self._is_retryable(e) and attempt < max_retries - 1:
                    delay = min(0.1 * (2**attempt), 5.0)
                    jitter = random.uniform(0, delay * 0.2)
                    time.sleep(delay + jitter)
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

    # Columns to select by default (excludes the heavy 'geometry' column).
    _LIGHT_COLS = (
        "id, orig_index, elements, natoms, status, charge, spin, "
        "job_dir, max_forces, scf_steps, final_energy, error_message, "
        "fail_count, wall_time, n_cores, optimizer, worker_id"
    )

    @staticmethod
    def _row_to_record(r: sqlite3.Row) -> JobRecord:
        """Convert a sqlite3.Row to a JobRecord.

        Uses column-name access so the code is independent of column
        ordering. Columns added via ALTER TABLE (which appends to the
        end) work without any positional adjustments.

        Args:
            r: sqlite3.Row from cursor with row_factory=sqlite3.Row.
        """
        keys = r.keys()
        return JobRecord(
            id=r["id"],
            orig_index=r["orig_index"],
            elements=r["elements"],
            natoms=r["natoms"],
            status=JobStatus(r["status"]),
            charge=r["charge"],
            spin=r["spin"],
            geometry=r["geometry"] if "geometry" in keys else None,
            job_dir=r["job_dir"],
            max_forces=r["max_forces"],
            scf_steps=r["scf_steps"],
            final_energy=r["final_energy"],
            error_message=r["error_message"],
            fail_count=r["fail_count"] if r["fail_count"] is not None else 0,
            wall_time=r["wall_time"] if "wall_time" in keys else None,
            n_cores=r["n_cores"] if "n_cores" in keys else None,
            optimizer=r["optimizer"] if "optimizer" in keys else None,
            worker_id=r["worker_id"] if "worker_id" in keys else None,
        )

    def get_jobs_by_status(
        self,
        status: JobStatus | list[JobStatus] | None = None,
        limit: int | None = None,
        include_geometry: bool = False,
    ) -> list[JobRecord]:
        """Retrieve jobs filtered by status.

        Args:
            status: Single status, list of statuses, or None for all jobs.
            limit: If set, return at most this many rows (SQL LIMIT).
            include_geometry: If True, include the geometry column (large).
                Defaults to False for performance.

        Returns:
            List of JobRecord objects matching the filter.
        """
        cols = "*" if include_geometry else self._LIGHT_COLS
        suffix = f" LIMIT {int(limit)}" if limit is not None else ""

        if status is None:
            query = f"SELECT {cols} FROM structures{suffix}"
            cur = self._execute_with_retry(query)
        elif isinstance(status, list):
            placeholders = ",".join("?" * len(status))
            query = f"SELECT {cols} FROM structures WHERE status IN ({placeholders}){suffix}"
            cur = self._execute_with_retry(query, tuple(s.value for s in status))
        else:
            query = f"SELECT {cols} FROM structures WHERE status = ?{suffix}"
            cur = self._execute_with_retry(query, (status.value,))

        rows = cur.fetchall()
        return [self._row_to_record(r) for r in rows]

    _SENTINEL = object()  # distinguishes "not passed" from None

    def update_status(
        self,
        job_id: int,
        new_status: JobStatus,
        error_message: str | None = None,
        increment_fail_count: bool = False,
        worker_id: object = _SENTINEL,
    ):
        """Update the status of a single job.

        Args:
            job_id: Database ID of the job.
            new_status: New status value.
            error_message: Optional error message.
            increment_fail_count: If True, atomically increment fail_count by 1.
            worker_id: If provided, set worker_id to this value. Pass None to
                clear it. Omit (default sentinel) to leave it unchanged.
        """
        set_clauses = ["status = ?", "updated_at = CURRENT_TIMESTAMP"]
        values: list = [new_status.value]

        if error_message is not None:
            set_clauses.append("error_message = ?")
            values.append(error_message)

        if increment_fail_count:
            set_clauses.append("fail_count = COALESCE(fail_count, 0) + 1")

        if worker_id is not self._SENTINEL:
            set_clauses.append("worker_id = ?")
            values.append(worker_id)

        query = f"UPDATE structures SET {', '.join(set_clauses)} WHERE id = ?"
        values.append(job_id)
        self._execute_with_retry(query, tuple(values))
        self._commit_with_retry()

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
            self._commit_with_retry()

    def update_job_metrics_bulk(self, metrics_list: list[dict]):
        """Update metrics for multiple jobs in a single transaction.

        Each dict in metrics_list must have a 'job_id' key and may have:
        job_dir, max_forces, scf_steps, final_energy, error_message,
        wall_time, n_cores.

        Args:
            metrics_list: List of dicts with job_id and metric values.
        """
        if not metrics_list:
            return

        for metrics in metrics_list:
            updates = []
            values = []
            job_id = metrics["job_id"]

            for col in (
                "job_dir",
                "max_forces",
                "scf_steps",
                "final_energy",
                "error_message",
                "wall_time",
                "n_cores",
            ):
                if metrics.get(col) is not None:
                    updates.append(f"{col} = ?")
                    values.append(metrics[col])

            if updates:
                updates.append("updated_at = CURRENT_TIMESTAMP")
                query = f"UPDATE structures SET {', '.join(updates)} WHERE id = ?"
                values.append(job_id)
                self._execute_with_retry(query, tuple(values))

        self._commit_with_retry()

    def update_status_bulk(
        self,
        job_ids: list[int],
        new_status: JobStatus,
        increment_fail_count: bool = False,
        error_message: str | None = None,
        worker_id: object = _SENTINEL,
    ):
        """Update status for multiple jobs at once.

        Args:
            job_ids: List of database IDs.
            new_status: New status to set for all jobs.
            increment_fail_count: If True, atomically increment fail_count by 1.
            error_message: If provided, set error_message for all jobs.
            worker_id: If provided, set worker_id for all jobs. Pass None to
                clear it. Omit (default sentinel) to leave unchanged.
        """
        if not job_ids:
            return

        set_clauses = ["status = ?", "updated_at = CURRENT_TIMESTAMP"]
        params: list = [new_status.value]

        if increment_fail_count:
            set_clauses.append("fail_count = COALESCE(fail_count, 0) + 1")

        if error_message is not None:
            set_clauses.append("error_message = ?")
            params.append(error_message)

        if worker_id is not self._SENTINEL:
            set_clauses.append("worker_id = ?")
            params.append(worker_id)

        placeholders = ",".join("?" * len(job_ids))
        query = f"UPDATE structures SET {', '.join(set_clauses)} WHERE id IN ({placeholders})"
        self._execute_with_retry(query, tuple(params + job_ids))
        self._commit_with_retry()

    def update_status_bulk_multi(
        self,
        status_groups: dict[JobStatus, list[int]],
    ):
        """Update multiple status groups in a single transaction.

        This minimises commit() calls (one instead of per-group), which is
        critical when another process (e.g. Parsl) is concurrently writing
        to the same database on a parallel filesystem.

        Args:
            status_groups: Mapping of new_status -> list of job IDs.
        """
        if not status_groups:
            return

        for new_status, job_ids in status_groups.items():
            if not job_ids:
                continue
            placeholders = ",".join("?" * len(job_ids))
            query = f"UPDATE structures SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id IN ({placeholders})"
            self._execute_with_retry(query, tuple([new_status.value] + job_ids))
        self._commit_with_retry()

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

    def mark_jobs_as_running(self, job_ids: list[int], worker_id: str | None = None):
        """Mark jobs as running (e.g., after submission to HPC queue).

        Args:
            job_ids: List of job database IDs.
            worker_id: Optional scheduler job ID to associate with these jobs
                (e.g., SLURM_JOB_ID or FLUX_JOB_ID). Used for crash recovery.
        """
        self.update_status_bulk(job_ids, JobStatus.RUNNING, worker_id=worker_id)

    def get_running_jobs_by_worker(self, worker_id: str) -> list[JobRecord]:
        """Get all RUNNING jobs owned by a specific scheduler job.

        Args:
            worker_id: Scheduler job ID to filter by.

        Returns:
            List of JobRecord objects that are RUNNING with this worker_id.
        """
        query = (
            f"SELECT {self._LIGHT_COLS} FROM structures "
            "WHERE status = ? AND worker_id = ?"
        )
        cur = self._execute_with_retry(query, (JobStatus.RUNNING.value, worker_id))
        return [self._row_to_record(r) for r in cur.fetchall()]

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
        self._commit_with_retry()

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
        self._commit_with_retry()

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
        self._commit_with_retry()

    def get_jobs_by_fail_count(self, min_fail_count: int = 1) -> list[JobRecord]:
        """Get jobs that have failed at least min_fail_count times.

        Args:
            min_fail_count: Minimum number of failures (default: 1).

        Returns:
            List of JobRecord objects with fail_count >= min_fail_count.
        """
        query = f"SELECT {self._LIGHT_COLS} FROM structures WHERE COALESCE(fail_count, 0) >= ?"
        cur = self._execute_with_retry(query, (min_fail_count,))
        rows = cur.fetchall()
        return [self._row_to_record(r) for r in rows]


def create_workflow(
    csv_path: str | Path,
    db_path: str | Path,
    geometry_column: str = "aligned_csd_core",
    charge_column: str | None = "charge",
    spin_column: str | None = "uhf",
    batch_size: int = 10000,
    extra_columns: dict[str, str] | None = None,
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
        extra_columns: Dictionary mapping CSV column names to SQL types
            (e.g., {"metal": "TEXT", "ligand_count": "INTEGER"}).
            These columns will be added to the database and populated from the CSV.

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
        extra_columns=extra_columns,
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

    Uses parse_job_metrics() which calls check_file_termination() to robustly
    detect job completion status, including normal termination, errors, aborted
    runs, and timeouts.

    Args:
        workflow: ArchitectorWorkflow instance.
        job_dir: Directory containing job output files.
        job_id: Database ID of the job.
        extract_metrics: If True, extract and store max_forces and scf_steps.
        unzip: If True, handle gzipped output files (quacc).

    Returns:
        The new status of the job (COMPLETED, FAILED, or TIMEOUT).
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

            # Determine job status based on termination check
            # Check for timeout first, then success, then failed
            if metrics.get("is_timeout", False):
                new_status = JobStatus.TIMEOUT
            elif metrics["success"]:
                new_status = JobStatus.COMPLETED
            else:
                new_status = JobStatus.FAILED

            # Extract timing info for successful jobs
            wall_time = None
            n_cores = None
            if new_status == JobStatus.COMPLETED:
                # Find the log file for timing extraction
                log_file = None
                for pattern in ["*.out", "orca.out", "*.log"]:
                    matches = [
                        m
                        for m in job_dir.glob(pattern)
                        if not re.match(r"^orca_atom\d+\.out$", m.name)
                    ]
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
