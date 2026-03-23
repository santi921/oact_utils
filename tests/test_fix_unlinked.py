"""Tests for --fix-unlinked dashboard feature."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

from oact_utilities.workflows.dashboard import fix_unlinked_jobs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_test_db(db_path: Path, jobs: list[dict]) -> Path:
    """Create a minimal workflow SQLite database for testing."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE structures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            orig_index INTEGER,
            elements TEXT,
            natoms INTEGER,
            status TEXT,
            charge INTEGER,
            spin INTEGER,
            geometry TEXT,
            job_dir TEXT,
            max_forces REAL,
            scf_steps INTEGER,
            final_energy REAL,
            error_message TEXT,
            fail_count INTEGER DEFAULT 0,
            wall_time REAL,
            n_cores INTEGER,
            optimizer TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    for job in jobs:
        conn.execute(
            """
            INSERT INTO structures
                (orig_index, elements, natoms, status, charge, spin,
                 job_dir, error_message, fail_count, optimizer)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job.get("orig_index", 1),
                job.get("elements", "U;O"),
                job.get("natoms", 3),
                job.get("status", "to_run"),
                job.get("charge", 0),
                job.get("spin", 1),
                job.get("job_dir"),
                job.get("error_message"),
                job.get("fail_count", 0),
                job.get("optimizer"),
            ),
        )
    conn.commit()
    conn.close()
    return db_path


def _create_job_dir(root: Path, name: str, files: list[str] | None = None) -> Path:
    """Create a job directory with optional files."""
    job_dir = root / name
    job_dir.mkdir(parents=True, exist_ok=True)
    for f in files or []:
        (job_dir / f).write_text("test content")
    return job_dir


def _get_job(db_path: Path, job_id: int) -> dict:
    """Fetch a single job row as a dict."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM structures WHERE id = ?", (job_id,)).fetchone()
    conn.close()
    return dict(row)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFixUnlinkedJobs:
    """Tests for fix_unlinked_jobs."""

    def test_links_existing_directory_completed(self, tmp_path: Path):
        """Job with NULL job_dir is linked when matching directory exists with completed output."""
        db_path = _create_test_db(
            tmp_path / "test.db",
            [{"orig_index": 42, "status": "to_run", "job_dir": None}],
        )
        jobs_root = tmp_path / "jobs"
        job_dir = _create_job_dir(jobs_root, "job_42", ["calc.out"])
        # Write a completed output marker
        (job_dir / "calc.out").write_text("stuff\nORCA TERMINATED NORMALLY\n")

        from oact_utilities.workflows.architector_workflow import ArchitectorWorkflow

        with ArchitectorWorkflow(str(db_path)) as wf:
            with patch(
                "oact_utilities.workflows.dashboard.check_job_termination",
                return_value=1,
            ):
                result = fix_unlinked_jobs(wf, str(jobs_root))

        assert result["linked"] == 1
        assert result["reset"] == 0
        job = _get_job(db_path, 1)
        assert job["job_dir"] == str(job_dir)
        assert job["status"] == "completed"

    def test_links_existing_directory_failed(self, tmp_path: Path):
        """Failed disk status updates DB to failed with error message."""
        db_path = _create_test_db(
            tmp_path / "test.db",
            [{"orig_index": 10, "status": "to_run", "job_dir": None}],
        )
        jobs_root = tmp_path / "jobs"
        job_dir = _create_job_dir(jobs_root, "job_10", ["calc.out"])
        (job_dir / "calc.out").write_text("stuff\nError: SCF not converged\n")

        from oact_utilities.workflows.architector_workflow import ArchitectorWorkflow

        with ArchitectorWorkflow(str(db_path)) as wf:
            with (
                patch(
                    "oact_utilities.workflows.dashboard.check_job_termination",
                    return_value=-1,
                ),
                patch(
                    "oact_utilities.workflows.dashboard.parse_failure_reason",
                    return_value="Error: SCF not converged",
                ),
            ):
                result = fix_unlinked_jobs(wf, str(jobs_root))

        assert result["linked"] == 1
        job = _get_job(db_path, 1)
        assert job["status"] == "failed"
        assert job["error_message"] == "Error: SCF not converged"

    def test_resets_when_no_directory(self, tmp_path: Path):
        """Job with NULL job_dir and no directory is reset to to_run."""
        db_path = _create_test_db(
            tmp_path / "test.db",
            [{"orig_index": 99, "status": "failed", "job_dir": None, "fail_count": 1}],
        )
        jobs_root = tmp_path / "jobs"
        jobs_root.mkdir()

        from oact_utilities.workflows.architector_workflow import ArchitectorWorkflow

        with ArchitectorWorkflow(str(db_path)) as wf:
            result = fix_unlinked_jobs(wf, str(jobs_root))

        assert result["reset"] == 1
        job = _get_job(db_path, 1)
        assert job["status"] == "to_run"
        assert job["fail_count"] == 2  # incremented

    def test_excludes_running_jobs(self, tmp_path: Path):
        """Running jobs with NULL job_dir are skipped entirely."""
        db_path = _create_test_db(
            tmp_path / "test.db",
            [{"orig_index": 5, "status": "running", "job_dir": None}],
        )
        jobs_root = tmp_path / "jobs"
        jobs_root.mkdir()

        from oact_utilities.workflows.architector_workflow import ArchitectorWorkflow

        with ArchitectorWorkflow(str(db_path)) as wf:
            result = fix_unlinked_jobs(wf, str(jobs_root))

        # Running is excluded from the query, so nothing processed
        assert result["linked"] == 0
        assert result["reset"] == 0
        job = _get_job(db_path, 1)
        assert job["status"] == "running"  # unchanged

    def test_skips_jobs_already_linked(self, tmp_path: Path):
        """Jobs with non-NULL job_dir are not affected."""
        db_path = _create_test_db(
            tmp_path / "test.db",
            [
                {"orig_index": 1, "status": "completed", "job_dir": "/some/path"},
                {"orig_index": 2, "status": "to_run", "job_dir": None},
            ],
        )
        jobs_root = tmp_path / "jobs"
        jobs_root.mkdir()

        from oact_utilities.workflows.architector_workflow import ArchitectorWorkflow

        with ArchitectorWorkflow(str(db_path)) as wf:
            result = fix_unlinked_jobs(wf, str(jobs_root))

        # Only job 2 is unlinked, job 1 already has job_dir
        assert result["reset"] == 1  # job 2 has no directory
        job1 = _get_job(db_path, 1)
        assert job1["job_dir"] == "/some/path"  # unchanged

    def test_directory_exists_no_output_file(self, tmp_path: Path):
        """Directory exists but no .out file -- link dir, set to_run."""
        db_path = _create_test_db(
            tmp_path / "test.db",
            [{"orig_index": 7, "status": "failed", "job_dir": None}],
        )
        jobs_root = tmp_path / "jobs"
        # Directory exists but only has input files
        _create_job_dir(jobs_root, "job_7", ["calc.inp", "calc.xyz"])

        from oact_utilities.workflows.architector_workflow import ArchitectorWorkflow

        with ArchitectorWorkflow(str(db_path)) as wf:
            # check_job_termination returns 0 for no output => maps to to_run
            with patch(
                "oact_utilities.workflows.dashboard.check_job_termination",
                return_value=0,
            ):
                result = fix_unlinked_jobs(wf, str(jobs_root))

        assert result["linked"] == 1
        job = _get_job(db_path, 1)
        assert job["job_dir"] is not None
        assert job["status"] == "to_run"

    def test_max_retries_skips_chronic_failures(self, tmp_path: Path):
        """Jobs at max retries are skipped during reset."""
        db_path = _create_test_db(
            tmp_path / "test.db",
            [
                {"orig_index": 1, "status": "failed", "job_dir": None, "fail_count": 5},
                {"orig_index": 2, "status": "failed", "job_dir": None, "fail_count": 1},
            ],
        )
        jobs_root = tmp_path / "jobs"
        jobs_root.mkdir()

        from oact_utilities.workflows.architector_workflow import ArchitectorWorkflow

        with ArchitectorWorkflow(str(db_path)) as wf:
            result = fix_unlinked_jobs(wf, str(jobs_root), max_retries=3)

        assert result["skipped"] == 1  # job 1 skipped (fail_count=5 >= 3)
        assert result["reset"] == 1  # job 2 reset (fail_count=1 < 3)
        job1 = _get_job(db_path, 1)
        assert job1["status"] == "failed"  # unchanged
        assert job1["fail_count"] == 5  # unchanged

    def test_max_jobs_limits_processing(self, tmp_path: Path):
        """--debug N limits the number of jobs processed."""
        db_path = _create_test_db(
            tmp_path / "test.db",
            [{"orig_index": i, "status": "to_run", "job_dir": None} for i in range(10)],
        )
        jobs_root = tmp_path / "jobs"
        jobs_root.mkdir()

        from oact_utilities.workflows.architector_workflow import ArchitectorWorkflow

        with ArchitectorWorkflow(str(db_path)) as wf:
            result = fix_unlinked_jobs(wf, str(jobs_root), max_jobs=3)

        total = result["linked"] + result["reset"] + result["skipped"]
        assert total == 3

    def test_disk_timeout_updates_status(self, tmp_path: Path):
        """Disk timeout (-2) updates DB status to timeout."""
        db_path = _create_test_db(
            tmp_path / "test.db",
            [{"orig_index": 15, "status": "to_run", "job_dir": None}],
        )
        jobs_root = tmp_path / "jobs"
        _create_job_dir(jobs_root, "job_15", ["calc.out"])

        from oact_utilities.workflows.architector_workflow import ArchitectorWorkflow

        with ArchitectorWorkflow(str(db_path)) as wf:
            with patch(
                "oact_utilities.workflows.dashboard.check_job_termination",
                return_value=-2,
            ):
                result = fix_unlinked_jobs(wf, str(jobs_root))

        assert result["linked"] == 1
        job = _get_job(db_path, 1)
        assert job["status"] == "timeout"

    def test_custom_job_dir_pattern(self, tmp_path: Path):
        """Custom job_dir_pattern is respected."""
        db_path = _create_test_db(
            tmp_path / "test.db",
            [{"orig_index": 3, "status": "to_run", "job_dir": None}],
        )
        jobs_root = tmp_path / "jobs"
        _create_job_dir(jobs_root, "calc_3", ["output.out"])

        from oact_utilities.workflows.architector_workflow import ArchitectorWorkflow

        with ArchitectorWorkflow(str(db_path)) as wf:
            with patch(
                "oact_utilities.workflows.dashboard.check_job_termination",
                return_value=1,
            ):
                result = fix_unlinked_jobs(
                    wf, str(jobs_root), job_dir_pattern="calc_{orig_index}"
                )

        assert result["linked"] == 1
        job = _get_job(db_path, 1)
        assert "calc_3" in job["job_dir"]

    def test_no_unlinked_jobs(self, tmp_path: Path):
        """When all jobs have job_dir set, nothing happens."""
        db_path = _create_test_db(
            tmp_path / "test.db",
            [{"orig_index": 1, "status": "completed", "job_dir": "/some/dir"}],
        )
        jobs_root = tmp_path / "jobs"
        jobs_root.mkdir()

        from oact_utilities.workflows.architector_workflow import ArchitectorWorkflow

        with ArchitectorWorkflow(str(db_path)) as wf:
            result = fix_unlinked_jobs(wf, str(jobs_root))

        assert result["linked"] == 0
        assert result["reset"] == 0
        assert result["skipped"] == 0

    def test_completed_missing_dir_resets(self, tmp_path: Path):
        """Completed job with NULL job_dir and no directory resets to to_run."""
        db_path = _create_test_db(
            tmp_path / "test.db",
            [{"orig_index": 20, "status": "completed", "job_dir": None}],
        )
        jobs_root = tmp_path / "jobs"
        jobs_root.mkdir()

        from oact_utilities.workflows.architector_workflow import ArchitectorWorkflow

        with ArchitectorWorkflow(str(db_path)) as wf:
            result = fix_unlinked_jobs(wf, str(jobs_root))

        assert result["reset"] == 1
        job = _get_job(db_path, 1)
        assert job["status"] == "to_run"
        assert job["fail_count"] == 1
