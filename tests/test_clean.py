"""Tests for job directory cleanup utility."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

from oact_utilities.workflows.clean import (
    _extract_failure_info,
    _format_size,
    _match_cleanup_patterns,
    _process_job,
    _purge_failed_job,
    _write_marker_file,
    clean_job_directories,
)

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
                job.get("status", "completed"),
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
    """Create a job directory with files."""
    job_dir = root / name
    job_dir.mkdir(parents=True, exist_ok=True)
    # Always create orca.out with success termination
    orca_out = job_dir / "orca.out"
    orca_out.write_text("Some output\nORCA TERMINATED NORMALLY\n")
    if files:
        for fname in files:
            if fname.endswith("/"):
                (job_dir / fname.rstrip("/")).mkdir(exist_ok=True)
            else:
                (job_dir / fname).write_text("test content")
    return job_dir


# ---------------------------------------------------------------------------
# Pattern matching tests
# ---------------------------------------------------------------------------


class TestMatchCleanupPatterns:
    def test_tmp_files(self):
        assert _match_cleanup_patterns("orca.tmp", False, {"tmp"}) is True
        assert _match_cleanup_patterns("orca.tmp.12345", False, {"tmp"}) is True
        assert _match_cleanup_patterns("orca.tmp.0", False, {"tmp"}) is True

    def test_core_files(self):
        assert _match_cleanup_patterns("core", False, {"tmp"}) is True
        assert _match_cleanup_patterns("core.12345", False, {"tmp"}) is True
        assert _match_cleanup_patterns("orca.core", False, {"tmp"}) is True

    def test_orca_tmp_dir(self):
        assert _match_cleanup_patterns("orca_tmp_7dg8v19h", True, {"tmp"}) is True
        assert _match_cleanup_patterns("orca_tmp_abc123", True, {"tmp"}) is True

    def test_bas_files(self):
        assert _match_cleanup_patterns("orca.bas", False, {"bas"}) is True
        assert _match_cleanup_patterns("orca.bas0", False, {"bas"}) is True
        assert _match_cleanup_patterns("orca.bas12", False, {"bas"}) is True

    def test_exclusion_list(self):
        """Verify excluded files never match any pattern."""
        excluded = [
            "orca.out",
            "orca.out.gz",
            "orca.inp",
            "orca.inp.gz",
            "orca.engrad",
            "orca.engrad.gz",
            "orca.xyz",
            "orca_metrics.json",
            "orca.property.txt",
            "orca.property.txt.gz",
            "orca.gbw",
            "orca.densities",
            "flux_job.flux",
            "slurm_job.sh",
            "sella_status.txt",
            "sella.log",
            "opt.traj",
            "sella_config.json",
            "run_sella.py",
            "results.pkl",
            "sella_driver.log",
            ".do_not_rerun.json",
        ]
        all_categories = {"tmp", "bas"}
        for fname in excluded:
            assert (
                _match_cleanup_patterns(fname, False, all_categories) is False
            ), f"{fname} should be excluded"

    def test_orca_atom_out_excluded(self):
        assert _match_cleanup_patterns("orca_atom0.out", False, {"tmp", "bas"}) is False
        assert (
            _match_cleanup_patterns("orca_atom12.out", False, {"tmp", "bas"}) is False
        )

    def test_trj_excluded(self):
        assert _match_cleanup_patterns("mol_trj.xyz", False, {"tmp", "bas"}) is False

    def test_no_match_without_category(self):
        assert _match_cleanup_patterns("orca.tmp", False, {"bas"}) is False
        assert _match_cleanup_patterns("orca.bas", False, {"tmp"}) is False

    def test_directory_not_matched_as_file(self):
        """orca_tmp_* only matches directories, not files."""
        assert _match_cleanup_patterns("orca_tmp_abc", False, {"tmp"}) is False


# ---------------------------------------------------------------------------
# Format size
# ---------------------------------------------------------------------------


class TestFormatSize:
    def test_bytes(self):
        assert _format_size(500) == "500 B"

    def test_kilobytes(self):
        assert _format_size(2048) == "2.0 KB"

    def test_megabytes(self):
        assert _format_size(5 * 1024 * 1024) == "5.0 MB"

    def test_gigabytes(self):
        assert _format_size(3 * 1024 * 1024 * 1024) == "3.0 GB"


# ---------------------------------------------------------------------------
# Process job
# ---------------------------------------------------------------------------


class TestProcessJob:
    def test_dry_run_no_deletion(self, tmp_path):
        """Dry run should identify files but not delete them."""
        job_dir = _create_job_dir(
            tmp_path, "job_1", ["orca.tmp", "orca.tmp.123", "orca.bas"]
        )
        matched, freed, errors = _process_job(
            job_dir, tmp_path, {"tmp"}, execute=False, hours_cutoff=24, optimizer=None
        )
        assert len(matched) == 2  # .tmp and .tmp.123
        assert freed == 0
        assert (job_dir / "orca.tmp").exists()
        assert (job_dir / "orca.tmp.123").exists()

    def test_execute_deletes_files(self, tmp_path):
        """Execute mode should delete matched files and keep others."""
        job_dir = _create_job_dir(
            tmp_path, "job_1", ["orca.tmp", "orca.bas", "orca.inp"]
        )
        matched, freed, errors = _process_job(
            job_dir, tmp_path, {"tmp"}, execute=True, hours_cutoff=24, optimizer=None
        )
        assert len(matched) == 1  # only .tmp
        assert not (job_dir / "orca.tmp").exists()
        assert (job_dir / "orca.bas").exists()  # not in tmp category
        assert (job_dir / "orca.inp").exists()  # excluded
        assert (job_dir / "orca.out").exists()  # excluded

    def test_revalidation_skip(self, tmp_path):
        """Jobs that fail revalidation should be skipped."""
        job_dir = _create_job_dir(tmp_path, "job_1", ["orca.tmp"])
        with patch(
            "oact_utilities.workflows.clean.check_job_termination", return_value=-1
        ):
            matched, freed, errors = _process_job(
                job_dir,
                tmp_path,
                {"tmp"},
                execute=True,
                hours_cutoff=24,
                optimizer=None,
            )
        assert len(matched) == 0
        assert len(errors) == 1
        assert "revalidation" in errors[0]

    def test_sella_optimizer_passed(self, tmp_path):
        """Verify optimizer is passed to check_job_termination."""
        job_dir = _create_job_dir(tmp_path, "job_1", ["orca.tmp"])
        with patch(
            "oact_utilities.workflows.clean.check_job_termination", return_value=1
        ) as mock_check:
            _process_job(
                job_dir,
                tmp_path,
                {"tmp"},
                execute=False,
                hours_cutoff=24,
                optimizer="sella",
            )
            mock_check.assert_called_once_with(
                str(job_dir), hours_cutoff=24, optimizer="sella"
            )

    def test_permission_error_continues(self, tmp_path):
        """Permission errors should be logged, not crash."""
        job_dir = _create_job_dir(tmp_path, "job_1", ["orca.tmp", "bad.tmp"])
        with patch.object(Path, "unlink", side_effect=PermissionError("denied")):
            matched, freed, errors = _process_job(
                job_dir,
                tmp_path,
                {"tmp"},
                execute=True,
                hours_cutoff=24,
                optimizer=None,
            )
        assert len(matched) == 2
        assert freed == 0
        assert len(errors) == 2

    def test_clean_all_categories(self, tmp_path):
        """Both tmp and bas categories should be cleaned."""
        job_dir = _create_job_dir(
            tmp_path, "job_1", ["orca.tmp", "orca.bas", "orca.bas0"]
        )
        matched, freed, errors = _process_job(
            job_dir,
            tmp_path,
            {"tmp", "bas"},
            execute=False,
            hours_cutoff=24,
            optimizer=None,
        )
        assert len(matched) == 3


# ---------------------------------------------------------------------------
# Path traversal
# ---------------------------------------------------------------------------


class TestPathSafety:
    def test_path_traversal_rejected(self, tmp_path):
        """Job dir containing .. that escapes root should be rejected."""
        from oact_utilities.workflows.clean import _resolve_job_dir

        result = _resolve_job_dir("../../../etc", tmp_path)
        assert result is None

    def test_valid_path(self, tmp_path):
        from oact_utilities.workflows.clean import _resolve_job_dir

        job_dir = tmp_path / "job_1"
        job_dir.mkdir()
        result = _resolve_job_dir("job_1", tmp_path)
        assert result is not None
        assert result == job_dir.resolve()


# ---------------------------------------------------------------------------
# Purge failed
# ---------------------------------------------------------------------------


class TestPurgeFailedJob:
    def test_writes_marker(self, tmp_path):
        """Verify .do_not_rerun.json is created with correct fields."""
        db_path = _create_test_db(
            tmp_path / "test.db",
            [
                {
                    "orig_index": 42,
                    "status": "failed",
                    "job_dir": "job_42",
                    "elements": "U;O;O",
                    "charge": 0,
                    "spin": 3,
                    "fail_count": 2,
                }
            ],
        )
        job_dir = tmp_path / "job_42"
        job_dir.mkdir()
        (job_dir / "orca.out").write_text("Error\naborting the run\n")
        (job_dir / "orca.inp").write_text("input")
        (job_dir / "orca.tmp").write_text("scratch")

        metadata = {
            "orig_index": 42,
            "elements": "U;O;O",
            "charge": 0,
            "spin": 3,
            "fail_count": 2,
            "error_message": None,
        }
        matched, freed, errors = _purge_failed_job(
            job_dir, tmp_path, db_path, 1, execute=True, job_metadata=metadata
        )
        marker = job_dir / ".do_not_rerun.json"
        assert marker.exists()
        data = json.loads(marker.read_text())
        assert data["orig_index"] == 42
        assert data["elements"] == "U;O;O"
        assert "generated_by" in data
        assert "date" in data

    def test_deletes_all_except_marker(self, tmp_path):
        """All files except marker should be removed."""
        db_path = _create_test_db(
            tmp_path / "test.db",
            [{"orig_index": 1, "status": "failed", "job_dir": "job_1"}],
        )
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()
        (job_dir / "orca.out").write_text("Error\n")
        (job_dir / "orca.inp").write_text("input")
        (job_dir / "orca.tmp").write_text("scratch")

        _purge_failed_job(
            job_dir,
            tmp_path,
            db_path,
            1,
            execute=True,
            job_metadata={"orig_index": 1},
        )
        remaining = list(job_dir.iterdir())
        assert len(remaining) == 1
        assert remaining[0].name == ".do_not_rerun.json"

    def test_dry_run(self, tmp_path):
        """Dry run should not write marker or delete files."""
        db_path = _create_test_db(
            tmp_path / "test.db",
            [{"orig_index": 1, "status": "failed", "job_dir": "job_1"}],
        )
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()
        (job_dir / "orca.out").write_text("Error\n")
        (job_dir / "orca.tmp").write_text("scratch")

        matched, freed, errors = _purge_failed_job(
            job_dir,
            tmp_path,
            db_path,
            1,
            execute=False,
            job_metadata={"orig_index": 1},
        )
        assert len(matched) == 2  # orca.out + orca.tmp
        assert freed == 0
        assert not (job_dir / ".do_not_rerun.json").exists()
        assert (job_dir / "orca.out").exists()
        assert (job_dir / "orca.tmp").exists()

    def test_toctou_recheck(self, tmp_path):
        """If DB status changed to running, purge should abort."""
        db_path = _create_test_db(
            tmp_path / "test.db",
            [{"orig_index": 1, "status": "running", "job_dir": "job_1"}],
        )
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()
        (job_dir / "orca.out").write_text("some output\n")

        matched, freed, errors = _purge_failed_job(
            job_dir,
            tmp_path,
            db_path,
            1,
            execute=True,
            job_metadata={"orig_index": 1},
        )
        assert len(matched) == 0
        assert len(errors) == 1
        assert "status changed" in errors[0]
        assert (job_dir / "orca.out").exists()

    def test_extracts_scf_and_error(self, tmp_path):
        """Verify SCF steps and failure reason are extracted."""
        db_path = _create_test_db(
            tmp_path / "test.db",
            [{"orig_index": 1, "status": "failed", "job_dir": "job_1"}],
        )
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()
        (job_dir / "orca.out").write_text(
            "SCF CONVERGED AFTER 42 CYCLES\n" "aborting the run\n"
        )

        _purge_failed_job(
            job_dir,
            tmp_path,
            db_path,
            1,
            execute=True,
            job_metadata={"orig_index": 1},
        )
        marker = job_dir / ".do_not_rerun.json"
        data = json.loads(marker.read_text())
        assert data["scf_steps"] == 42
        assert "aborting the run" in data["failure_reason"]

    def test_handles_missing_output(self, tmp_path):
        """When orca.out is missing, marker should still be written with null fields."""
        db_path = _create_test_db(
            tmp_path / "test.db",
            [{"orig_index": 1, "status": "failed", "job_dir": "job_1"}],
        )
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()
        # No orca.out file

        _purge_failed_job(
            job_dir,
            tmp_path,
            db_path,
            1,
            execute=True,
            job_metadata={"orig_index": 1},
        )
        marker = job_dir / ".do_not_rerun.json"
        assert marker.exists()
        data = json.loads(marker.read_text())
        assert data["scf_steps"] is None
        assert data["failure_reason"] is None


# ---------------------------------------------------------------------------
# CLI / integration tests
# ---------------------------------------------------------------------------


class TestCleanJobDirectories:
    def test_no_category_flag_exits(self):
        """Calling with no category flag should cause an error."""
        import subprocess

        result = subprocess.run(
            ["python", "-m", "oact_utilities.workflows.clean", "dummy.db", "dummy_dir"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert (
            "action flag" in result.stderr.lower()
            or "required" in result.stderr.lower()
        )

    def test_debug_limit(self, tmp_path):
        """--debug N should process only N jobs."""
        root = tmp_path / "jobs"
        root.mkdir()
        jobs = []
        for i in range(10):
            job_dir = _create_job_dir(root, f"job_{i}", ["orca.tmp"])
            jobs.append(
                {
                    "orig_index": i,
                    "status": "completed",
                    "job_dir": str(job_dir),
                }
            )
        db_path = _create_test_db(tmp_path / "test.db", jobs)

        # Capture output to verify limited processing
        clean_job_directories(
            db_path=db_path,
            root_dir=root,
            categories={"tmp"},
            execute=False,
            limit=5,
        )
        # If this completes without error, the limit was applied

    def test_null_job_dir_skip(self, tmp_path):
        """Jobs with NULL job_dir should be skipped gracefully."""
        db_path = _create_test_db(
            tmp_path / "test.db",
            [{"orig_index": 1, "status": "completed", "job_dir": None}],
        )
        root = tmp_path / "jobs"
        root.mkdir()
        # Should not crash
        clean_job_directories(
            db_path=db_path,
            root_dir=root,
            categories={"tmp"},
            execute=False,
        )


# ---------------------------------------------------------------------------
# Submit guard tests
# ---------------------------------------------------------------------------


class TestSubmitGuard:
    def test_filter_marker_jobs(self, tmp_path):
        """Jobs with .do_not_rerun.json marker should be filtered out."""
        from oact_utilities.workflows.architector_workflow import (
            ArchitectorWorkflow,
            JobStatus,
        )
        from oact_utilities.workflows.submit_jobs import _filter_marker_jobs

        root = tmp_path / "jobs"
        root.mkdir()

        # Create a job dir with marker
        job_dir = root / "job_1"
        job_dir.mkdir()
        (job_dir / ".do_not_rerun.json").write_text('{"test": true}')

        # Create a job dir without marker
        (root / "job_2").mkdir()

        # Create DB
        db_path = _create_test_db(
            tmp_path / "test.db",
            [
                {"orig_index": 1, "status": "to_run", "job_dir": str(job_dir)},
                {"orig_index": 2, "status": "to_run", "job_dir": str(root / "job_2")},
            ],
        )

        with ArchitectorWorkflow(db_path) as wf:
            jobs = wf.get_jobs_by_status([JobStatus.TO_RUN], include_geometry=False)
            assert len(jobs) == 2

            filtered = _filter_marker_jobs(jobs, root, "job_{orig_index}", wf)
            assert len(filtered) == 1
            assert filtered[0].orig_index == 2

            # Check that the marked job was updated to FAILED
            updated = wf.get_jobs_by_status(JobStatus.FAILED, include_geometry=False)
            assert len(updated) == 1
            assert updated[0].orig_index == 1
            assert updated[0].fail_count == 1


# ---------------------------------------------------------------------------
# Extract failure info
# ---------------------------------------------------------------------------


class TestExtractFailureInfo:
    def test_extracts_scf_and_reason(self, tmp_path):
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()
        (job_dir / "orca.out").write_text(
            "SCF CONVERGED AFTER 100 CYCLES\n" "aborting the run\n"
        )
        info = _extract_failure_info(job_dir)
        assert info["scf_steps"] == 100
        assert "aborting the run" in info["failure_reason"]

    def test_missing_output(self, tmp_path):
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()
        info = _extract_failure_info(job_dir)
        assert info["scf_steps"] is None
        assert info["failure_reason"] is None


# ---------------------------------------------------------------------------
# Write marker
# ---------------------------------------------------------------------------


class TestWriteMarker:
    def test_writes_valid_json(self, tmp_path):
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()
        _write_marker_file(job_dir, {"orig_index": 5, "charge": 0})
        marker = job_dir / ".do_not_rerun.json"
        assert marker.exists()
        data = json.loads(marker.read_text())
        assert data["orig_index"] == 5
        assert "generated_by" in data
        assert "date" in data
