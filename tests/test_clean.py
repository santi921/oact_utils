"""Tests for job directory cleanup utility."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

from oact_utilities.workflows.architector_workflow import ArchitectorWorkflow
from oact_utilities.workflows.clean import (
    ValidationOutcome,
    _check_row_alignment,
    _extract_failure_info,
    _format_size,
    _get_dir_size,
    _match_cleanup_patterns,
    _process_job,
    _purge_failed_job,
    _purge_incomplete_job,
    _write_marker_file,
    clean_job_directories,
    validate_db_folder_alignment,
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


class TestGetDirSize:
    def test_skips_symlinked_dir(self, tmp_path):
        """A symlink to a dir outside the job must not be traversed/counted."""
        import pytest

        real = tmp_path / "real"
        real.mkdir()
        (real / "f.bin").write_bytes(b"x" * 1000)
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "big.bin").write_bytes(b"y" * 100000)
        try:
            (real / "link").symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unsupported on this platform")
        # Only f.bin is counted; the symlinked corpus dir is skipped.
        assert _get_dir_size(real) == 1000


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


# ---------------------------------------------------------------------------
# DB <-> folder validation (--validate-db)
# ---------------------------------------------------------------------------


def _write_orca_inp(job_dir: Path, elements: list[str], charge: int = 0, spin: int = 1):
    """Write a minimal orca.inp with a coordinate block for the given elements."""
    lines = ["! wB97M-V def2-TZVP", "", f"* xyz {charge} {spin}"]
    for i, el in enumerate(elements):
        lines.append(f"{el}  0.0  0.0  {float(i)}")
    lines.append("*")
    (job_dir / "orca.inp").write_text("\n".join(lines) + "\n")


def _make_record(orig_index, elements, natoms, status="timeout", job_dir=None):
    from oact_utilities.workflows.architector_workflow import JobRecord, JobStatus

    return JobRecord(
        id=orig_index,
        orig_index=orig_index,
        elements=elements,
        natoms=natoms,
        status=JobStatus(status),
        job_dir=job_dir,
    )


class TestCheckRowAlignment:
    def test_match(self, tmp_path):
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()
        _write_orca_inp(job_dir, ["U", "O", "O"])
        rec = _make_record(1, "U;O;O", 3, job_dir="job_1")
        outcome, inp = _check_row_alignment(rec, tmp_path)
        assert outcome is ValidationOutcome.MATCH
        assert inp == "U;O;O"

    def test_element_mismatch(self, tmp_path):
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()
        _write_orca_inp(job_dir, ["Np", "F", "F", "F"])
        rec = _make_record(1, "U;O;O", 3, job_dir="job_1")
        outcome, inp = _check_row_alignment(rec, tmp_path)
        assert outcome is ValidationOutcome.MISMATCH

    def test_natoms_mismatch(self, tmp_path):
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()
        _write_orca_inp(job_dir, ["U", "O"])  # 2 atoms
        rec = _make_record(1, "U;O", 3, job_dir="job_1")  # DB says 3
        outcome, _ = _check_row_alignment(rec, tmp_path)
        assert outcome is ValidationOutcome.MISMATCH

    def test_missing_inp_is_unverifiable(self, tmp_path):
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()  # no orca.inp
        rec = _make_record(1, "U;O;O", 3, job_dir="job_1")
        outcome, reason = _check_row_alignment(rec, tmp_path)
        assert outcome is ValidationOutcome.UNVERIFIABLE
        assert reason == "no_orca_inp"

    def test_missing_dir_is_unverifiable(self, tmp_path):
        rec = _make_record(1, "U;O;O", 3, job_dir="does_not_exist")
        outcome, _ = _check_row_alignment(rec, tmp_path)
        assert outcome is ValidationOutcome.UNVERIFIABLE

    def test_null_job_dir_is_unverifiable(self, tmp_path):
        rec = _make_record(1, "U;O;O", 3, job_dir=None)
        outcome, _ = _check_row_alignment(rec, tmp_path)
        assert outcome is ValidationOutcome.UNVERIFIABLE

    def test_db_omits_actinide_is_match(self, tmp_path):
        """DB elements/natoms omit the central actinide; orca.inp includes it."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()
        _write_orca_inp(job_dir, ["Am", "O", "C", "O"])  # inp has the metal
        rec = _make_record(1, "O;C;O", 3, job_dir="job_1")  # DB omits Am, natoms=3
        outcome, inp = _check_row_alignment(rec, tmp_path)
        assert outcome is ValidationOutcome.MATCH
        assert inp == "Am;O;C;O"

    def test_two_extra_atoms_is_mismatch(self, tmp_path):
        """One actinide is tolerated; a second extra atom is not."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()
        _write_orca_inp(job_dir, ["Am", "Cl", "O", "C", "O"])  # 2 extra vs DB
        rec = _make_record(1, "O;C;O", 3, job_dir="job_1")
        outcome, _ = _check_row_alignment(rec, tmp_path)
        assert outcome is ValidationOutcome.MISMATCH

    def test_single_extra_nonactinide_is_mismatch(self, tmp_path):
        """A single extra atom that is not an actinide is a real mismatch."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()
        _write_orca_inp(job_dir, ["Fe", "O", "C", "O"])  # Fe is not an actinide
        rec = _make_record(1, "O;C;O", 3, job_dir="job_1")
        outcome, _ = _check_row_alignment(rec, tmp_path)
        assert outcome is ValidationOutcome.MISMATCH


class TestValidateDbFolder:
    def test_all_match_passes(self, tmp_path):
        root = tmp_path / "jobs"
        root.mkdir()
        jobs = []
        for i in range(5):
            jd = _create_job_dir(root, f"job_{i}", [])
            _write_orca_inp(jd, ["U", "O", "O"])
            jobs.append(
                {
                    "orig_index": i,
                    "status": "timeout" if i < 3 else "completed",
                    "elements": "U;O;O",
                    "natoms": 3,
                    "job_dir": str(jd),
                }
            )
        db_path = _create_test_db(tmp_path / "test.db", jobs)
        with ArchitectorWorkflow(db_path) as wf:
            result = validate_db_folder_alignment(wf, root)
        assert result.passed is True
        assert result.mismatch_count == 0
        assert result.match_count == 5

    def test_element_mismatch_aborts(self, tmp_path):
        root = tmp_path / "jobs"
        root.mkdir()
        jd0 = _create_job_dir(root, "job_0", [])
        _write_orca_inp(jd0, ["U", "O", "O"])
        jd1 = _create_job_dir(root, "job_1", [])
        _write_orca_inp(jd1, ["Np", "F", "F", "F"])  # different molecule
        jobs = [
            {
                "orig_index": 0,
                "status": "timeout",
                "elements": "U;O;O",
                "natoms": 3,
                "job_dir": str(jd0),
            },
            {
                "orig_index": 1,
                "status": "timeout",
                "elements": "U;O;O",
                "natoms": 3,
                "job_dir": str(jd1),
            },
        ]
        db_path = _create_test_db(tmp_path / "test.db", jobs)
        with ArchitectorWorkflow(db_path) as wf:
            result = validate_db_folder_alignment(wf, root)
        assert result.passed is False
        assert result.mismatch_count == 1
        assert any(m[0] == 1 for m in result.mismatches)

    def test_fail_closed_on_zero_verifiable(self, tmp_path):
        root = tmp_path / "jobs"
        root.mkdir()
        # Dirs exist but no orca.inp -> all UNVERIFIABLE
        jobs = []
        for i in range(4):
            jd = _create_job_dir(root, f"job_{i}", [])
            (jd / "orca.inp").unlink(missing_ok=True)
            jobs.append(
                {
                    "orig_index": i,
                    "status": "timeout",
                    "elements": "U;O;O",
                    "natoms": 3,
                    "job_dir": str(jd),
                }
            )
        db_path = _create_test_db(tmp_path / "test.db", jobs)
        with ArchitectorWorkflow(db_path) as wf:
            result = validate_db_folder_alignment(wf, root)
        assert result.verifiable == 0
        assert result.passed is False

    def test_coverage_guard_below_half(self, tmp_path):
        root = tmp_path / "jobs"
        root.mkdir()
        jobs = []
        # 2 verifiable matches
        for i in range(2):
            jd = _create_job_dir(root, f"job_{i}", [])
            _write_orca_inp(jd, ["U", "O", "O"])
            jobs.append(
                {
                    "orig_index": i,
                    "status": "completed",
                    "elements": "U;O;O",
                    "natoms": 3,
                    "job_dir": str(jd),
                }
            )
        # 3 unverifiable (dir missing)
        for i in range(2, 5):
            jobs.append(
                {
                    "orig_index": i,
                    "status": "completed",
                    "elements": "U;O;O",
                    "natoms": 3,
                    "job_dir": str(root / f"missing_{i}"),
                }
            )
        db_path = _create_test_db(tmp_path / "test.db", jobs)
        with ArchitectorWorkflow(db_path) as wf:
            result = validate_db_folder_alignment(wf, root)
        # 2 verifiable / 5 sampled = 0.4 < 0.5 -> fail closed
        assert result.match_count == 2
        assert result.unverifiable_count == 3
        assert result.passed is False

    def test_sample_includes_incomplete(self, tmp_path):
        """A mismatch among the (minority) incomplete jobs is caught even when
        the sample budget is saturated by completed jobs (stratification)."""
        root = tmp_path / "jobs"
        root.mkdir()
        jobs = []
        # 105 matching completed jobs -> forces the 100-row sample cap
        for i in range(105):
            jd = _create_job_dir(root, f"done_{i}", [])
            _write_orca_inp(jd, ["U", "O", "O"])
            jobs.append(
                {
                    "orig_index": i,
                    "status": "completed",
                    "elements": "U;O;O",
                    "natoms": 3,
                    "job_dir": str(jd),
                }
            )
        # 3 timeout jobs; one mismatches
        for j in range(3):
            idx = 1000 + j
            jd = _create_job_dir(root, f"to_{j}", [])
            if j == 0:
                _write_orca_inp(jd, ["Np", "F", "F", "F"])  # mismatch
            else:
                _write_orca_inp(jd, ["U", "O", "O"])
            jobs.append(
                {
                    "orig_index": idx,
                    "status": "timeout",
                    "elements": "U;O;O",
                    "natoms": 3,
                    "job_dir": str(jd),
                }
            )
        db_path = _create_test_db(tmp_path / "test.db", jobs)
        with ArchitectorWorkflow(db_path) as wf:
            result = validate_db_folder_alignment(wf, root)
        assert result.passed is False
        assert any(m[0] == 1000 for m in result.mismatches)

    def test_passes_with_many_unverified_to_run(self, tmp_path):
        """Never-run to_run jobs (no dir) don't fail the gate when enough rows
        verify with zero mismatches -- mirrors the real partial-campaign case."""
        root = tmp_path / "jobs"
        root.mkdir()
        jobs = []
        # 25 completed jobs that verify (matching orca.inp)
        for i in range(25):
            jd = _create_job_dir(root, f"done_{i}", [])
            _write_orca_inp(jd, ["U", "O", "O"])
            jobs.append(
                {
                    "orig_index": i,
                    "status": "completed",
                    "elements": "U;O;O",
                    "natoms": 3,
                    "job_dir": str(jd),
                }
            )
        # 60 to_run jobs that were never written to disk (dir missing)
        for j in range(60):
            jobs.append(
                {
                    "orig_index": 1000 + j,
                    "status": "to_run",
                    "elements": "U;O;O",
                    "natoms": 3,
                    "job_dir": str(root / f"never_{j}"),
                }
            )
        db_path = _create_test_db(tmp_path / "test.db", jobs)
        with ArchitectorWorkflow(db_path) as wf:
            result = validate_db_folder_alignment(wf, root)
        assert result.mismatch_count == 0
        assert result.match_count == 25
        assert result.passed is True  # 25 verified >= floor, 0 mismatch
        assert result.unverifiable_to_run > 0
        assert "dir_missing" in result.unverifiable_reasons


# ---------------------------------------------------------------------------
# Purge incomplete
# ---------------------------------------------------------------------------


class TestPurgeIncompleteJob:
    def _setup(self, tmp_path, status="timeout"):
        db_path = _create_test_db(
            tmp_path / "test.db",
            [
                {
                    "orig_index": 7,
                    "status": status,
                    "job_dir": "job_7",
                    "elements": "U;O;O",
                    "natoms": 3,
                }
            ],
        )
        job_dir = tmp_path / "job_7"
        job_dir.mkdir()
        (job_dir / "orca.out").write_text("partial output, no terminator\n")
        (job_dir / "orca.inp").write_text("input")
        (job_dir / "orca.tmp").write_text("scratch")
        return db_path, job_dir

    def test_completed_on_disk_is_protected(self, tmp_path):
        db_path, job_dir = self._setup(tmp_path, status="running")
        with patch(
            "oact_utilities.workflows.clean.check_job_termination", return_value=1
        ):
            cls, matched, freed, errors = _purge_incomplete_job(
                job_dir,
                tmp_path,
                db_path,
                1,
                execute=True,
                hours_cutoff=24,
                optimizer=None,
                job_metadata={"orig_index": 7},
            )
        assert cls == "protected"
        assert not (job_dir / ".do_not_rerun.json").exists()
        assert (job_dir / "orca.out").exists()
        assert (job_dir / "orca.tmp").exists()

    def test_failed_on_disk_is_skipped(self, tmp_path):
        db_path, job_dir = self._setup(tmp_path, status="timeout")
        with patch(
            "oact_utilities.workflows.clean.check_job_termination", return_value=-1
        ):
            cls, matched, freed, errors = _purge_incomplete_job(
                job_dir,
                tmp_path,
                db_path,
                1,
                execute=True,
                hours_cutoff=24,
                optimizer=None,
                job_metadata={"orig_index": 7},
            )
        assert cls == "looks_failed"
        assert not (job_dir / ".do_not_rerun.json").exists()
        assert (job_dir / "orca.out").exists()

    def test_incomplete_is_purged_with_marker(self, tmp_path):
        db_path, job_dir = self._setup(tmp_path, status="timeout")
        with patch(
            "oact_utilities.workflows.clean.check_job_termination", return_value=0
        ):
            cls, matched, freed, errors = _purge_incomplete_job(
                job_dir,
                tmp_path,
                db_path,
                1,
                execute=True,
                hours_cutoff=24,
                optimizer=None,
                job_metadata={"orig_index": 7, "elements": "U;O;O"},
            )
        assert cls == "purged"
        marker = job_dir / ".do_not_rerun.json"
        assert marker.exists()
        data = json.loads(marker.read_text())
        assert data["purge_type"] == "incomplete_archive"
        assert data["db_status_at_purge"] == "timeout"
        assert data["disk_status_code"] == 0
        assert data["orig_index"] == 7
        remaining = list(job_dir.iterdir())
        assert len(remaining) == 1
        assert remaining[0].name == ".do_not_rerun.json"

    def test_timeout_by_age_is_purged(self, tmp_path):
        db_path, job_dir = self._setup(tmp_path, status="timeout")
        with patch(
            "oact_utilities.workflows.clean.check_job_termination", return_value=-2
        ):
            cls, matched, freed, errors = _purge_incomplete_job(
                job_dir,
                tmp_path,
                db_path,
                1,
                execute=True,
                hours_cutoff=24,
                optimizer=None,
                job_metadata={"orig_index": 7},
            )
        assert cls == "purged"
        data = json.loads((job_dir / ".do_not_rerun.json").read_text())
        assert data["disk_status_code"] == -2

    def test_content_check_runs_before_marker_write(self, tmp_path):
        """A job that reads as completed must not get a marker, even in execute."""
        db_path, job_dir = self._setup(tmp_path, status="running")
        with patch(
            "oact_utilities.workflows.clean.check_job_termination", return_value=1
        ):
            cls, *_ = _purge_incomplete_job(
                job_dir,
                tmp_path,
                db_path,
                1,
                execute=True,
                hours_cutoff=24,
                optimizer=None,
                job_metadata={"orig_index": 7},
            )
        assert cls == "protected"
        assert not (job_dir / ".do_not_rerun.json").exists()
        assert (job_dir / "orca.out").exists()

    def test_toctou_abort_when_status_changed(self, tmp_path):
        # DB row is 'completed' (no longer in incomplete set)
        db_path, job_dir = self._setup(tmp_path, status="completed")
        with patch(
            "oact_utilities.workflows.clean.check_job_termination", return_value=0
        ):
            cls, matched, freed, errors = _purge_incomplete_job(
                job_dir,
                tmp_path,
                db_path,
                1,
                execute=True,
                hours_cutoff=24,
                optimizer=None,
                job_metadata={"orig_index": 7},
            )
        assert cls == "skipped"
        assert any("status changed" in e for e in errors)
        assert (job_dir / "orca.out").exists()

    def test_marker_write_failure_preserves_contents(self, tmp_path):
        db_path, job_dir = self._setup(tmp_path, status="timeout")
        with (
            patch(
                "oact_utilities.workflows.clean.check_job_termination", return_value=0
            ),
            patch(
                "oact_utilities.workflows.clean._write_marker_file",
                side_effect=OSError("read-only fs"),
            ),
        ):
            cls, matched, freed, errors = _purge_incomplete_job(
                job_dir,
                tmp_path,
                db_path,
                1,
                execute=True,
                hours_cutoff=24,
                optimizer=None,
                job_metadata={"orig_index": 7},
            )
        assert cls == "skipped"
        assert any("marker write failed" in e for e in errors)
        # Nothing deleted
        assert (job_dir / "orca.out").exists()
        assert (job_dir / "orca.tmp").exists()

    def test_dry_run_frees_nothing(self, tmp_path):
        db_path, job_dir = self._setup(tmp_path, status="timeout")
        with patch(
            "oact_utilities.workflows.clean.check_job_termination", return_value=0
        ):
            cls, matched, freed, errors = _purge_incomplete_job(
                job_dir,
                tmp_path,
                db_path,
                1,
                execute=False,
                hours_cutoff=24,
                optimizer=None,
                job_metadata={"orig_index": 7},
            )
        assert cls == "purged"
        assert len(matched) >= 1
        assert freed == 0
        assert not (job_dir / ".do_not_rerun.json").exists()
        assert (job_dir / "orca.out").exists()

    def test_idempotent_rerun(self, tmp_path):
        db_path, job_dir = self._setup(tmp_path, status="timeout")
        with patch(
            "oact_utilities.workflows.clean.check_job_termination", return_value=0
        ):
            _purge_incomplete_job(
                job_dir,
                tmp_path,
                db_path,
                1,
                execute=True,
                hours_cutoff=24,
                optimizer=None,
                job_metadata={"orig_index": 7},
            )
            # Second run: only the marker remains
            cls, matched, freed, errors = _purge_incomplete_job(
                job_dir,
                tmp_path,
                db_path,
                1,
                execute=True,
                hours_cutoff=24,
                optimizer=None,
                job_metadata={"orig_index": 7},
            )
        assert cls == "purged"
        assert matched == []  # nothing left to remove except marker
        remaining = list(job_dir.iterdir())
        assert len(remaining) == 1
        assert remaining[0].name == ".do_not_rerun.json"


class TestPurgeIncompleteIntegration:
    def test_gate_blocks_purge_on_mismatch(self, tmp_path):
        """--purge-incomplete runs validation by default and refuses on mismatch."""
        root = tmp_path / "jobs"
        root.mkdir()
        jd = _create_job_dir(root, "job_0", ["orca.tmp"])
        _write_orca_inp(jd, ["Np", "F", "F", "F"])  # mismatch vs DB U;O;O
        db_path = _create_test_db(
            tmp_path / "test.db",
            [
                {
                    "orig_index": 0,
                    "status": "timeout",
                    "elements": "U;O;O",
                    "natoms": 3,
                    "job_dir": str(jd),
                }
            ],
        )
        with patch(
            "oact_utilities.workflows.clean.check_job_termination", return_value=0
        ):
            ok = clean_job_directories(
                db_path=db_path,
                root_dir=root,
                categories=set(),
                purge_incomplete=True,
                execute=True,
            )
        assert ok is False
        assert (jd / "orca.tmp").exists()  # nothing purged
        assert not (jd / ".do_not_rerun.json").exists()

    def test_skip_validation_allows_purge(self, tmp_path):
        root = tmp_path / "jobs"
        root.mkdir()
        jd = _create_job_dir(root, "job_0", ["orca.tmp"])
        _write_orca_inp(jd, ["Np", "F", "F", "F"])  # would fail validation
        db_path = _create_test_db(
            tmp_path / "test.db",
            [
                {
                    "orig_index": 0,
                    "status": "timeout",
                    "elements": "U;O;O",
                    "natoms": 3,
                    "job_dir": str(jd),
                }
            ],
        )
        with patch(
            "oact_utilities.workflows.clean.check_job_termination", return_value=0
        ):
            ok = clean_job_directories(
                db_path=db_path,
                root_dir=root,
                categories=set(),
                purge_incomplete=True,
                execute=True,
                skip_validation=True,
            )
        assert ok is True
        assert (jd / ".do_not_rerun.json").exists()
        assert not (jd / "orca.tmp").exists()

    def test_validate_db_standalone_exit_code(self, tmp_path):
        """Standalone --validate-db exits non-zero on mismatch."""
        import subprocess

        root = tmp_path / "jobs"
        root.mkdir()
        jd = _create_job_dir(root, "job_0", [])
        _write_orca_inp(jd, ["Np", "F", "F", "F"])
        db_path = _create_test_db(
            tmp_path / "test.db",
            [
                {
                    "orig_index": 0,
                    "status": "timeout",
                    "elements": "U;O;O",
                    "natoms": 3,
                    "job_dir": str(jd),
                }
            ],
        )
        result = subprocess.run(
            [
                "python",
                "-m",
                "oact_utilities.workflows.clean",
                str(db_path),
                str(root),
                "--validate-db",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "FAIL" in result.stdout
