"""Tests for architector workflow manager."""

import os
import sqlite3
import time

import pandas as pd
import pytest

from oact_utilities.utils.architector import create_workflow_db
from oact_utilities.workflows import (
    ArchitectorWorkflow,
    JobStatus,
    create_workflow,
    update_job_status,
)


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample architector CSV file."""
    csv_file = tmp_path / "sample.csv"

    # Create sample XYZ strings (without atom count and comment lines)
    xyz1 = """H 0.0 0.0 0.0
H 0.0 0.0 0.74"""

    xyz2 = """O 0.0 0.0 0.0
H 0.757 0.586 0.0
H -0.757 0.586 0.0"""

    data = {
        "aligned_csd_core": [xyz1, xyz2],
        "charge": [0, 0],
        "uhf": [1, 1],  # spin multiplicity (1 = singlet)
        "other_col": ["A", "B"],
    }

    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)

    return csv_file


def test_create_workflow(sample_csv, tmp_path):
    """Test workflow creation from CSV."""
    db_path = tmp_path / "workflow.db"

    db_path_ret, workflow = create_workflow(
        csv_path=sample_csv,
        db_path=db_path,
        geometry_column="aligned_csd_core",
        charge_column="charge",
        spin_column="uhf",
    )

    assert db_path_ret.exists()

    # Check that jobs were created
    jobs = workflow.get_jobs_by_status(JobStatus.TO_RUN)
    assert len(jobs) == 2

    # Check job details
    job1 = jobs[0]
    assert job1.natoms == 2
    assert "H" in job1.elements

    job2 = jobs[1]
    assert job2.natoms == 3
    assert "O" in job2.elements

    workflow.close()


def test_workflow_status_updates(tmp_path):
    """Test updating job statuses."""
    db_path = tmp_path / "test.db"

    # Use new db init
    from oact_utilities.utils.architector import _init_db, _insert_row

    conn = _init_db(db_path)

    # Insert test rows
    _insert_row(
        conn,
        orig_index=0,
        elements="H;H",
        natoms=2,
        geometry="H 0 0 0\nH 0 0 0.74",
        status="to_run",
    )
    _insert_row(
        conn,
        orig_index=1,
        elements="O;H;H",
        natoms=3,
        geometry="O 0 0 0\nH 0.757 0.586 0\nH -0.757 0.586 0",
        status="to_run",
    )
    conn.commit()
    conn.close()

    # Test workflow
    with ArchitectorWorkflow(db_path) as workflow:
        # Check initial state
        ready = workflow.get_jobs_by_status(JobStatus.TO_RUN)
        assert len(ready) == 2

        # Update one job
        workflow.update_status(1, JobStatus.RUNNING)

        # Check updated state
        ready = workflow.get_jobs_by_status(JobStatus.TO_RUN)
        running = workflow.get_jobs_by_status(JobStatus.RUNNING)
        assert len(ready) == 1
        assert len(running) == 1

        # Bulk update
        workflow.update_status_bulk([1, 2], JobStatus.COMPLETED)

        completed = workflow.get_jobs_by_status(JobStatus.COMPLETED)
        assert len(completed) == 2


def test_workflow_metrics(tmp_path):
    """Test updating job metrics."""
    db_path = tmp_path / "test.db"

    # Use new db init
    from oact_utilities.utils.architector import _init_db, _insert_row

    conn = _init_db(db_path)

    _insert_row(
        conn,
        orig_index=0,
        elements="H;H",
        natoms=2,
        geometry="H 0 0 0\nH 0 0 0.74",
        status="running",
    )
    conn.commit()
    conn.close()

    with ArchitectorWorkflow(db_path) as workflow:
        # Update metrics
        workflow.update_job_metrics(
            job_id=1,
            job_dir="/path/to/job",
            max_forces=0.00123,
            scf_steps=15,
            final_energy=-1.23456,
        )

        # Retrieve and check
        jobs = workflow.get_jobs_by_status(JobStatus.RUNNING)
        assert len(jobs) == 1
        job = jobs[0]
        assert job.job_dir == "/path/to/job"
        assert job.max_forces == pytest.approx(0.00123)
        assert job.scf_steps == 15
        assert job.final_energy == pytest.approx(-1.23456)


def test_count_by_status(tmp_path):
    """Test status counting."""
    db_path = tmp_path / "test.db"

    # Use new db init
    from oact_utilities.utils.architector import _init_db, _insert_row

    conn = _init_db(db_path)

    for i in range(10):
        if i < 5:
            status = "to_run"
        elif i < 8:
            status = "running"
        elif i < 9:
            status = "completed"
        else:
            status = "failed"

        _insert_row(
            conn,
            orig_index=i,
            elements="H;H",
            natoms=2,
            geometry="H 0 0 0\nH 0 0 0.74",
            status=status,
        )
    conn.commit()
    conn.close()

    with ArchitectorWorkflow(db_path) as workflow:
        counts = workflow.count_by_status()

        assert counts[JobStatus.TO_RUN] == 5
        assert counts[JobStatus.RUNNING] == 3
        assert counts[JobStatus.COMPLETED] == 1
        assert counts[JobStatus.FAILED] == 1


def test_create_workflow_db_directly(sample_csv, tmp_path):
    """Test direct database creation."""
    db_path = tmp_path / "test.db"

    result_path = create_workflow_db(
        csv_path=sample_csv,
        db_path=db_path,
        geometry_column="aligned_csd_core",
        charge_column="charge",
        spin_column="uhf",
    )

    assert result_path.exists()

    # Check database contents
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM structures")
    count = cur.fetchone()[0]
    assert count == 2

    cur.execute("SELECT elements, natoms, charge, spin FROM structures WHERE id = 1")
    row = cur.fetchone()
    assert row[0] == "H;H"  # elements
    assert row[1] == 2  # natoms
    assert row[2] == 0  # charge
    assert row[3] == 1  # spin multiplicity (singlet)

    conn.close()


def test_timeout_status(tmp_path):
    """Test timeout status handling."""
    db_path = tmp_path / "test.db"

    # Use new db init
    from oact_utilities.utils.architector import _init_db, _insert_row

    conn = _init_db(db_path)

    # Insert test rows with different statuses
    _insert_row(
        conn,
        orig_index=0,
        elements="H;H",
        natoms=2,
        geometry="H 0 0 0\nH 0 0 0.74",
        status="running",
    )
    _insert_row(
        conn,
        orig_index=1,
        elements="O;H;H",
        natoms=3,
        geometry="O 0 0 0\nH 0.757 0.586 0\nH -0.757 0.586 0",
        status="timeout",
    )
    _insert_row(
        conn,
        orig_index=2,
        elements="H;H",
        natoms=2,
        geometry="H 0 0 0\nH 0 0 0.74",
        status="failed",
    )
    conn.commit()
    conn.close()

    with ArchitectorWorkflow(db_path) as workflow:
        # Check timeout jobs
        timeout = workflow.get_jobs_by_status(JobStatus.TIMEOUT)
        assert len(timeout) == 1
        assert timeout[0].orig_index == 1

        # Update to timeout
        workflow.update_status(1, JobStatus.TIMEOUT)
        timeout = workflow.get_jobs_by_status(JobStatus.TIMEOUT)
        assert len(timeout) == 2


def test_to_run_status(tmp_path):
    """Test TO_RUN status handling."""
    db_path = tmp_path / "test.db"

    from oact_utilities.utils.architector import _init_db, _insert_row

    conn = _init_db(db_path)

    # Insert one row as legacy "ready" and one as "to_run"
    _insert_row(
        conn,
        orig_index=0,
        elements="H;H",
        natoms=2,
        geometry="H 0 0 0\nH 0 0 0.74",
        status="ready",
    )
    _insert_row(
        conn,
        orig_index=1,
        elements="O;H;H",
        natoms=3,
        geometry="O 0 0 0\nH 0.757 0.586 0\nH -0.757 0.586 0",
        status="to_run",
    )
    conn.commit()
    conn.close()

    # Opening the DB triggers _ensure_schema() which migrates ready -> to_run
    with ArchitectorWorkflow(db_path) as workflow:
        to_run = workflow.get_jobs_by_status(JobStatus.TO_RUN)
        assert len(to_run) == 2  # Both rows are now to_run

        # Verify no "ready" rows remain
        ready = workflow.get_jobs_by_status(JobStatus.READY)
        assert len(ready) == 0


def test_reset_timeout_jobs(tmp_path):
    """Test resetting timeout jobs."""
    db_path = tmp_path / "test.db"

    from oact_utilities.utils.architector import _init_db, _insert_row

    conn = _init_db(db_path)

    # Insert test rows with timeout status
    _insert_row(
        conn,
        orig_index=0,
        elements="H;H",
        natoms=2,
        geometry="H 0 0 0\nH 0 0 0.74",
        status="timeout",
        fail_count=0,
    )
    _insert_row(
        conn,
        orig_index=1,
        elements="O;H;H",
        natoms=3,
        geometry="O 0 0 0\nH 0.757 0.586 0\nH -0.757 0.586 0",
        status="timeout",
        fail_count=2,
    )
    conn.commit()
    conn.close()

    with ArchitectorWorkflow(db_path) as workflow:
        # Reset timeout jobs
        workflow.reset_timeout_jobs()

        # Check that all timeout jobs are now TO_RUN
        timeout = workflow.get_jobs_by_status(JobStatus.TIMEOUT)
        to_run = workflow.get_jobs_by_status(JobStatus.TO_RUN)
        assert len(timeout) == 0
        assert len(to_run) == 2

        # Check that fail_count was incremented
        jobs = workflow.get_jobs_by_status(JobStatus.TO_RUN)
        assert jobs[0].fail_count == 1
        assert jobs[1].fail_count == 3


def test_reset_timeout_jobs_with_limit(tmp_path):
    """Test resetting timeout jobs with max_fail_count limit."""
    db_path = tmp_path / "test.db"

    from oact_utilities.utils.architector import _init_db, _insert_row

    conn = _init_db(db_path)

    # Insert test rows with timeout status and different fail counts
    _insert_row(
        conn,
        orig_index=0,
        elements="H;H",
        natoms=2,
        geometry="H 0 0 0\nH 0 0 0.74",
        status="timeout",
        fail_count=1,
    )
    _insert_row(
        conn,
        orig_index=1,
        elements="O;H;H",
        natoms=3,
        geometry="O 0 0 0\nH 0.757 0.586 0\nH -0.757 0.586 0",
        status="timeout",
        fail_count=3,
    )
    conn.commit()
    conn.close()

    with ArchitectorWorkflow(db_path) as workflow:
        # Reset timeout jobs, but only those with fail_count < 3
        workflow.reset_timeout_jobs(max_fail_count=3)

        # Check that only the first job was reset
        timeout = workflow.get_jobs_by_status(JobStatus.TIMEOUT)
        to_run = workflow.get_jobs_by_status(JobStatus.TO_RUN)
        assert len(timeout) == 1  # Job with fail_count=3 stays in timeout
        assert len(to_run) == 1  # Job with fail_count=1 was reset


def test_reset_failed_with_timeout(tmp_path):
    """Test resetting failed jobs with include_timeout=True."""
    db_path = tmp_path / "test.db"

    from oact_utilities.utils.architector import _init_db, _insert_row

    conn = _init_db(db_path)

    # Insert test rows with failed and timeout status
    _insert_row(
        conn,
        orig_index=0,
        elements="H;H",
        natoms=2,
        geometry="H 0 0 0\nH 0 0 0.74",
        status="failed",
        fail_count=0,
    )
    _insert_row(
        conn,
        orig_index=1,
        elements="O;H;H",
        natoms=3,
        geometry="O 0 0 0\nH 0.757 0.586 0\nH -0.757 0.586 0",
        status="timeout",
        fail_count=0,
    )
    conn.commit()
    conn.close()

    with ArchitectorWorkflow(db_path) as workflow:
        # Reset both failed and timeout jobs
        workflow.reset_failed_jobs(include_timeout=True)

        # Check that both are now TO_RUN
        failed = workflow.get_jobs_by_status(JobStatus.FAILED)
        timeout = workflow.get_jobs_by_status(JobStatus.TIMEOUT)
        to_run = workflow.get_jobs_by_status(JobStatus.TO_RUN)
        assert len(failed) == 0
        assert len(timeout) == 0
        assert len(to_run) == 2


def test_update_status_increment_fail_count(tmp_path):
    """Test that increment_fail_count atomically increments fail_count with status."""
    db_path = tmp_path / "test.db"

    from oact_utilities.utils.architector import _init_db, _insert_row

    conn = _init_db(db_path)
    _insert_row(
        conn,
        orig_index=0,
        elements="H;H",
        natoms=2,
        geometry="H 0 0 0\nH 0 0 0.74",
        status="running",
        fail_count=0,
    )
    conn.commit()
    conn.close()

    with ArchitectorWorkflow(db_path) as workflow:
        workflow.update_status(1, JobStatus.FAILED, increment_fail_count=True)

        jobs = workflow.get_jobs_by_status(JobStatus.FAILED)
        assert len(jobs) == 1
        assert jobs[0].fail_count == 1

        # Increment again
        workflow.update_status(1, JobStatus.FAILED, increment_fail_count=True)
        jobs = workflow.get_jobs_by_status(JobStatus.FAILED)
        assert jobs[0].fail_count == 2


def test_update_status_increment_fail_count_with_error(tmp_path):
    """Test increment_fail_count works together with error_message."""
    db_path = tmp_path / "test.db"

    from oact_utilities.utils.architector import _init_db, _insert_row

    conn = _init_db(db_path)
    _insert_row(
        conn,
        orig_index=0,
        elements="H;H",
        natoms=2,
        geometry="H 0 0 0\nH 0 0 0.74",
        status="running",
        fail_count=1,
    )
    conn.commit()
    conn.close()

    with ArchitectorWorkflow(db_path) as workflow:
        workflow.update_status(
            1,
            JobStatus.FAILED,
            error_message="ORCA crashed",
            increment_fail_count=True,
        )

        jobs = workflow.get_jobs_by_status(JobStatus.FAILED)
        assert len(jobs) == 1
        assert jobs[0].fail_count == 2
        assert jobs[0].error_message == "ORCA crashed"


def test_update_status_default_no_increment(tmp_path):
    """Test that default update_status does NOT change fail_count."""
    db_path = tmp_path / "test.db"

    from oact_utilities.utils.architector import _init_db, _insert_row

    conn = _init_db(db_path)
    _insert_row(
        conn,
        orig_index=0,
        elements="H;H",
        natoms=2,
        geometry="H 0 0 0\nH 0 0 0.74",
        status="running",
        fail_count=3,
    )
    conn.commit()
    conn.close()

    with ArchitectorWorkflow(db_path) as workflow:
        workflow.update_status(1, JobStatus.FAILED)

        jobs = workflow.get_jobs_by_status(JobStatus.FAILED)
        assert len(jobs) == 1
        assert jobs[0].fail_count == 3  # Unchanged


def test_update_job_status_timeout_detection(tmp_path):
    """Test timeout detection: only stale files without termination signal are timeout.

    A file with "ORCA TERMINATED NORMALLY" should be COMPLETED even if old.
    A stale file without a termination signal should be TIMEOUT.
    """
    from oact_utilities.utils.architector import _init_db, _insert_row

    # --- Case 1: Old file WITH normal termination → COMPLETED ---
    db_path = tmp_path / "test_completed.db"
    conn = _init_db(db_path)
    _insert_row(
        conn,
        orig_index=0,
        elements="H;H",
        natoms=2,
        geometry="H 0 0 0\nH 0 0 0.74",
        status="running",
    )
    conn.commit()
    conn.close()

    job_dir = tmp_path / "job_completed_old"
    job_dir.mkdir()
    output_file = job_dir / "orca.out"
    output_file.write_text(
        "ORCA CALCULATION\n"
        "...\n"
        "...\n"
        "...\n"
        "****ORCA TERMINATED NORMALLY****\n"
    )
    eight_hours_ago = time.time() - (8 * 3600)
    os.utime(output_file, (eight_hours_ago, eight_hours_ago))

    with ArchitectorWorkflow(db_path) as workflow:
        status = update_job_status(
            workflow=workflow,
            job_dir=job_dir,
            job_id=1,
            extract_metrics=True,
            unzip=False,
        )
        assert status == JobStatus.COMPLETED, "Old completed file should stay COMPLETED"
        jobs = workflow.get_jobs_by_status(JobStatus.COMPLETED)
        assert len(jobs) == 1

    # --- Case 2: Old file WITHOUT termination signal → TIMEOUT ---
    db_path2 = tmp_path / "test_timeout.db"
    conn = _init_db(db_path2)
    _insert_row(
        conn,
        orig_index=0,
        elements="H;H",
        natoms=2,
        geometry="H 0 0 0\nH 0 0 0.74",
        status="running",
    )
    conn.commit()
    conn.close()

    job_dir2 = tmp_path / "job_timeout"
    job_dir2.mkdir()
    output_file2 = job_dir2 / "orca.out"
    output_file2.write_text(
        "ORCA CALCULATION\n" "SCF CONVERGED AFTER 15 CYCLES\n" "...\n"
    )
    os.utime(output_file2, (eight_hours_ago, eight_hours_ago))

    with ArchitectorWorkflow(db_path2) as workflow:
        status = update_job_status(
            workflow=workflow,
            job_dir=job_dir2,
            job_id=1,
            extract_metrics=True,
            unzip=False,
        )
        assert status == JobStatus.TIMEOUT, "Old inconclusive file should be TIMEOUT"
        jobs = workflow.get_jobs_by_status(JobStatus.TIMEOUT)
        assert len(jobs) == 1


def test_update_job_status_error_detection(tmp_path):
    """Test that update_job_status correctly detects error jobs."""
    db_path = tmp_path / "test.db"

    from oact_utilities.utils.architector import _init_db, _insert_row

    conn = _init_db(db_path)
    _insert_row(
        conn,
        orig_index=0,
        elements="H;H",
        natoms=2,
        geometry="H 0 0 0\nH 0 0 0.74",
        status="running",
    )
    conn.commit()
    conn.close()

    # Create a job directory with an error output file
    job_dir = tmp_path / "job_error"
    job_dir.mkdir()
    output_file = job_dir / "orca.out"

    # Write output with error in last 5 lines
    output_file.write_text(
        "ORCA CALCULATION\n"
        "...\n"
        "...\n"
        "SCF NOT CONVERGED\n"
        "Error: SCF failed to converge\n"
    )

    with ArchitectorWorkflow(db_path) as workflow:
        # Update job status - should detect error
        status = update_job_status(
            workflow=workflow,
            job_dir=job_dir,
            job_id=1,
            extract_metrics=True,
            unzip=False,
        )

        # Should be marked as FAILED
        assert status == JobStatus.FAILED, "Error file should be detected as failed"

        # Verify in database
        jobs = workflow.get_jobs_by_status(JobStatus.FAILED)
        assert len(jobs) == 1


def test_update_job_status_normal_termination(tmp_path):
    """Test that update_job_status correctly detects normal termination."""
    db_path = tmp_path / "test.db"

    from oact_utilities.utils.architector import _init_db, _insert_row

    conn = _init_db(db_path)
    _insert_row(
        conn,
        orig_index=0,
        elements="H;H",
        natoms=2,
        geometry="H 0 0 0\nH 0 0 0.74",
        status="running",
    )
    conn.commit()
    conn.close()

    # Create a job directory with successful output
    job_dir = tmp_path / "job_success"
    job_dir.mkdir()
    output_file = job_dir / "orca.out"

    # Write output with normal termination (recent file)
    output_file.write_text(
        "ORCA CALCULATION\n"
        "...\n"
        "...\n"
        "...\n"
        "****ORCA TERMINATED NORMALLY****\n"
    )

    with ArchitectorWorkflow(db_path) as workflow:
        # Update job status - should detect success
        status = update_job_status(
            workflow=workflow,
            job_dir=job_dir,
            job_id=1,
            extract_metrics=True,
            unzip=False,
        )

        # Should be marked as COMPLETED
        assert status == JobStatus.COMPLETED, "Normal termination should be completed"

        # Verify in database
        jobs = workflow.get_jobs_by_status(JobStatus.COMPLETED)
        assert len(jobs) == 1


def test_reset_missing_jobs(tmp_path):
    """Test that jobs with missing directories are reset to TO_RUN."""
    from oact_utilities.utils.architector import _init_db, _insert_row
    from oact_utilities.workflows.dashboard import reset_missing_jobs

    db_path = tmp_path / "test.db"
    conn = _init_db(db_path)

    # Insert jobs with various statuses
    _insert_row(
        conn,
        orig_index=0,
        elements="H;H",
        natoms=2,
        geometry="H 0 0 0\nH 0 0 0.74",
        status="running",
        fail_count=0,
    )
    _insert_row(
        conn,
        orig_index=1,
        elements="H;H",
        natoms=2,
        geometry="H 0 0 0\nH 0 0 0.74",
        status="failed",
        fail_count=1,
    )
    _insert_row(
        conn,
        orig_index=2,
        elements="H;H",
        natoms=2,
        geometry="H 0 0 0\nH 0 0 0.74",
        status="running",
        fail_count=0,
    )
    _insert_row(
        conn,
        orig_index=3,
        elements="H;H",
        natoms=2,
        geometry="H 0 0 0\nH 0 0 0.74",
        status="completed",
        fail_count=0,
    )
    conn.commit()
    conn.close()

    # Create directories for jobs 0 and 3 only — jobs 1 and 2 are "missing"
    root_dir = tmp_path / "jobs"
    root_dir.mkdir()
    (root_dir / "job_0").mkdir()
    (root_dir / "job_3").mkdir()

    with ArchitectorWorkflow(db_path) as workflow:
        count = reset_missing_jobs(workflow, root_dir)

        # Jobs 1 (failed) and 2 (running) should be reset — their dirs don't exist
        assert count == 2

        # Verify they are now TO_RUN
        to_run = workflow.get_jobs_by_status(JobStatus.TO_RUN)
        reset_indices = sorted(j.orig_index for j in to_run)
        assert reset_indices == [1, 2]

        # Verify fail_count was incremented
        for job in to_run:
            if job.orig_index == 1:
                assert job.fail_count == 2  # was 1, now 2
            elif job.orig_index == 2:
                assert job.fail_count == 1  # was 0, now 1

        # Job 0 (running, dir exists) should stay running
        running = workflow.get_jobs_by_status(JobStatus.RUNNING)
        assert len(running) == 1
        assert running[0].orig_index == 0

        # Job 3 (completed) should not be touched
        completed = workflow.get_jobs_by_status(JobStatus.COMPLETED)
        assert len(completed) == 1
        assert completed[0].orig_index == 3


def test_reset_missing_jobs_no_missing(tmp_path):
    """Test reset_missing_jobs when all directories exist."""
    from oact_utilities.utils.architector import _init_db, _insert_row
    from oact_utilities.workflows.dashboard import reset_missing_jobs

    db_path = tmp_path / "test.db"
    conn = _init_db(db_path)

    _insert_row(
        conn,
        orig_index=0,
        elements="H;H",
        natoms=2,
        geometry="H 0 0 0\nH 0 0 0.74",
        status="running",
        fail_count=0,
    )
    conn.commit()
    conn.close()

    root_dir = tmp_path / "jobs"
    root_dir.mkdir()
    (root_dir / "job_0").mkdir()

    with ArchitectorWorkflow(db_path) as workflow:
        count = reset_missing_jobs(workflow, root_dir)
        assert count == 0


def test_ensure_schema_adds_optimizer_column(tmp_path):
    """_ensure_schema auto-migrates old DBs by adding the optimizer column."""
    from oact_utilities.utils.architector import _init_db, _insert_row

    db_path = tmp_path / "test.db"
    conn = _init_db(db_path)
    _insert_row(
        conn,
        orig_index=0,
        elements="H;H",
        natoms=2,
        geometry="H 0 0 0\nH 0 0 0.74",
        status="to_run",
    )
    conn.commit()
    conn.close()

    # Opening the workflow should auto-add the optimizer column
    with ArchitectorWorkflow(db_path) as workflow:
        # Verify the column exists by querying it
        cur = workflow.conn.execute("SELECT optimizer FROM structures WHERE id = 1")
        row = cur.fetchone()
        assert row is not None
        assert row[0] is None  # Default should be NULL


def test_worker_id_set_and_cleared(tmp_path):
    """worker_id is set on mark_jobs_as_running and cleared on status change."""
    from oact_utilities.utils.architector import _init_db, _insert_row

    db_path = tmp_path / "test.db"
    conn = _init_db(db_path)
    _insert_row(
        conn,
        orig_index=0,
        elements="H;H",
        natoms=2,
        geometry="H 0 0 0\nH 0 0 0.74",
        status="to_run",
    )
    conn.commit()
    conn.close()

    with ArchitectorWorkflow(db_path) as workflow:
        # Mark as running with a worker_id
        workflow.mark_jobs_as_running([1], worker_id="slurm_12345")
        jobs = workflow.get_jobs_by_status(JobStatus.RUNNING)
        assert len(jobs) == 1
        assert jobs[0].worker_id == "slurm_12345"

        # Complete the job -- worker_id should be cleared
        workflow.update_status(1, JobStatus.COMPLETED, worker_id=None)
        jobs = workflow.get_jobs_by_status(JobStatus.COMPLETED)
        assert len(jobs) == 1
        assert jobs[0].worker_id is None


def test_worker_id_bulk_reset(tmp_path):
    """update_status_bulk clears worker_id for multiple jobs at once."""
    from oact_utilities.utils.architector import _init_db, _insert_row

    db_path = tmp_path / "test.db"
    conn = _init_db(db_path)
    for i in range(5):
        _insert_row(
            conn,
            orig_index=i,
            elements="H;H",
            natoms=2,
            geometry="H 0 0 0\nH 0 0 0.74",
            status="to_run",
        )
    conn.commit()
    conn.close()

    with ArchitectorWorkflow(db_path) as workflow:
        job_ids = [1, 2, 3, 4, 5]
        workflow.mark_jobs_as_running(job_ids, worker_id="flux_abc123")

        # Verify all have worker_id set
        running = workflow.get_jobs_by_status(JobStatus.RUNNING)
        assert all(j.worker_id == "flux_abc123" for j in running)

        # Bulk reset orphans to TO_RUN, clearing worker_id
        workflow.update_status_bulk([1, 2, 3], JobStatus.TO_RUN, worker_id=None)

        to_run = workflow.get_jobs_by_status(JobStatus.TO_RUN)
        assert len(to_run) == 3
        assert all(j.worker_id is None for j in to_run)

        # Remaining jobs still have worker_id
        still_running = workflow.get_jobs_by_status(JobStatus.RUNNING)
        assert len(still_running) == 2
        assert all(j.worker_id == "flux_abc123" for j in still_running)


def test_get_running_jobs_by_worker(tmp_path):
    """get_running_jobs_by_worker filters by worker_id."""
    from oact_utilities.utils.architector import _init_db, _insert_row

    db_path = tmp_path / "test.db"
    conn = _init_db(db_path)
    for i in range(4):
        _insert_row(
            conn,
            orig_index=i,
            elements="H;H",
            natoms=2,
            geometry="H 0 0 0\nH 0 0 0.74",
            status="to_run",
        )
    conn.commit()
    conn.close()

    with ArchitectorWorkflow(db_path) as workflow:
        # Two jobs owned by worker A, two by worker B
        workflow.mark_jobs_as_running([1, 2], worker_id="slurm_100")
        workflow.mark_jobs_as_running([3, 4], worker_id="slurm_200")

        worker_a = workflow.get_running_jobs_by_worker("slurm_100")
        assert len(worker_a) == 2
        assert {j.id for j in worker_a} == {1, 2}

        worker_b = workflow.get_running_jobs_by_worker("slurm_200")
        assert len(worker_b) == 2
        assert {j.id for j in worker_b} == {3, 4}

        # Non-existent worker returns empty
        none = workflow.get_running_jobs_by_worker("slurm_999")
        assert len(none) == 0


def test_ensure_schema_migrates_worker_id(tmp_path):
    """_ensure_schema adds worker_id column to old databases."""
    import sqlite3

    db_path = tmp_path / "old.db"
    # Create a DB without worker_id column (simulating old schema)
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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.execute(
        "INSERT INTO structures (orig_index, elements, natoms, status, geometry) "
        "VALUES (0, 'H;H', 2, 'ready', 'H 0 0 0')"
    )
    conn.commit()
    conn.close()

    # Opening triggers _ensure_schema which adds worker_id and migrates ready->to_run
    with ArchitectorWorkflow(db_path) as workflow:
        jobs = workflow.get_jobs_by_status(JobStatus.TO_RUN)
        assert len(jobs) == 1
        assert jobs[0].worker_id is None
        assert jobs[0].status == JobStatus.TO_RUN

        # Verify worker_id column works
        workflow.mark_jobs_as_running([1], worker_id="test_123")
        running = workflow.get_running_jobs_by_worker("test_123")
        assert len(running) == 1


def test_sigterm_handler_sets_flag():
    """SIGTERM handler sets the shutdown flag without raising."""
    import signal

    _shutdown_requested = False
    _original = signal.getsignal(signal.SIGTERM)

    def _handler(signum, frame):
        nonlocal _shutdown_requested
        _shutdown_requested = True

    try:
        signal.signal(signal.SIGTERM, _handler)
        # Send SIGTERM to ourselves
        os.kill(os.getpid(), signal.SIGTERM)
        assert _shutdown_requested is True
    finally:
        signal.signal(signal.SIGTERM, _original)


# --- Phase 3: Scheduler liveness checks and orphan recovery ---


def test_get_active_scheduler_jobs_slurm_mock(monkeypatch):
    """get_active_scheduler_jobs returns a set of job IDs for SLURM."""
    from unittest.mock import MagicMock

    from oact_utilities.utils.scheduler import get_active_scheduler_jobs

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "12345\n12346\n12347\n"

    monkeypatch.setattr(
        "oact_utilities.utils.scheduler.subprocess.run", lambda *a, **kw: mock_result
    )

    active = get_active_scheduler_jobs("slurm")
    assert active == {"12345", "12346", "12347"}


def test_get_active_scheduler_jobs_flux_mock(monkeypatch):
    """get_active_scheduler_jobs returns compact Flux IDs."""
    from unittest.mock import MagicMock

    from oact_utilities.utils.scheduler import get_active_scheduler_jobs

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "f2xgUVYLJs27\nf2xgUVdyx8JK\n"

    monkeypatch.setattr(
        "oact_utilities.utils.scheduler.subprocess.run", lambda *a, **kw: mock_result
    )

    active = get_active_scheduler_jobs("flux")
    assert active == {"f2xgUVYLJs27", "f2xgUVdyx8JK"}


def test_get_active_scheduler_jobs_empty(monkeypatch):
    """Empty scheduler queue returns empty set (not None)."""
    from unittest.mock import MagicMock

    from oact_utilities.utils.scheduler import get_active_scheduler_jobs

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""

    monkeypatch.setattr(
        "oact_utilities.utils.scheduler.subprocess.run", lambda *a, **kw: mock_result
    )

    active = get_active_scheduler_jobs("slurm")
    assert active == set()
    assert active is not None


def test_get_active_scheduler_jobs_unreachable(monkeypatch):
    """Unreachable scheduler returns None (conservative: skip recovery)."""
    import subprocess as sp

    from oact_utilities.utils.scheduler import get_active_scheduler_jobs

    monkeypatch.setattr(
        "oact_utilities.utils.scheduler.subprocess.run",
        lambda *a, **kw: (_ for _ in ()).throw(sp.TimeoutExpired("squeue", 30)),
    )

    active = get_active_scheduler_jobs("slurm")
    assert active is None


def test_get_active_scheduler_jobs_command_not_found(monkeypatch):
    """Missing scheduler command returns None."""
    from oact_utilities.utils.scheduler import get_active_scheduler_jobs

    monkeypatch.setattr(
        "oact_utilities.utils.scheduler.subprocess.run",
        lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError("squeue")),
    )

    active = get_active_scheduler_jobs("slurm")
    assert active is None


def test_recover_orphaned_jobs_dead_scheduler(tmp_path, monkeypatch):
    """Orphaned RUNNING jobs from a dead scheduler job are recovered."""
    from oact_utilities.utils.architector import _init_db, _insert_row
    from oact_utilities.workflows.dashboard import recover_orphaned_jobs

    db_path = tmp_path / "test.db"
    conn = _init_db(db_path)
    # Insert 3 jobs
    for i in range(3):
        _insert_row(
            conn,
            orig_index=i,
            elements="H;H",
            natoms=2,
            geometry="H 0 0 0\nH 0 0 0.74",
            status="to_run",
        )
    conn.commit()
    conn.close()

    with ArchitectorWorkflow(db_path) as workflow:
        # Mark all as running under scheduler job "slurm_100"
        workflow.mark_jobs_as_running([1, 2, 3], worker_id="slurm_100")

        # Mock scheduler: slurm_100 is NOT in active set (it's dead)
        monkeypatch.setattr(
            "oact_utilities.utils.scheduler.get_active_scheduler_jobs",
            lambda sched: set(),  # empty = nothing active
        )

        # Mock check_job_termination: job 1 completed, job 2 failed, job 3 inconclusive
        def mock_check(job_dir, **kwargs):
            if job_dir and "job_0" in str(job_dir):
                return 1  # completed
            elif job_dir and "job_1" in str(job_dir):
                return -1  # failed
            return 0  # inconclusive

        monkeypatch.setattr(
            "oact_utilities.workflows.dashboard.check_job_termination",
            mock_check,
        )

        # Set job_dirs so the check function has something to work with
        workflow.update_job_metrics(1, job_dir=str(tmp_path / "job_0"))
        workflow.update_job_metrics(2, job_dir=str(tmp_path / "job_1"))
        workflow.update_job_metrics(3, job_dir=str(tmp_path / "job_2"))

        result = recover_orphaned_jobs(workflow, scheduler="slurm")

        assert result["dead_jobs"] == 1
        assert result["completed"] == 1
        assert result["failed"] == 1
        assert result["reset"] == 1
        assert result["recovered"] == 3

        # Verify final statuses
        jobs = {j.id: j for j in workflow.get_jobs_by_status()}
        assert jobs[1].status == JobStatus.COMPLETED
        assert jobs[1].worker_id is None
        assert jobs[2].status == JobStatus.FAILED
        assert jobs[2].worker_id is None
        assert jobs[3].status == JobStatus.TO_RUN
        assert jobs[3].worker_id is None


def test_recover_orphans_scheduler_unreachable(tmp_path, monkeypatch):
    """When scheduler is unreachable, no jobs are modified (conservative)."""
    from oact_utilities.utils.architector import _init_db, _insert_row
    from oact_utilities.workflows.dashboard import recover_orphaned_jobs

    db_path = tmp_path / "test.db"
    conn = _init_db(db_path)
    _insert_row(
        conn,
        orig_index=0,
        elements="H;H",
        natoms=2,
        geometry="H 0 0 0\nH 0 0 0.74",
        status="to_run",
    )
    conn.commit()
    conn.close()

    with ArchitectorWorkflow(db_path) as workflow:
        workflow.mark_jobs_as_running([1], worker_id="slurm_100")

        # Mock scheduler: returns None (unreachable)
        monkeypatch.setattr(
            "oact_utilities.utils.scheduler.get_active_scheduler_jobs",
            lambda sched: None,
        )

        result = recover_orphaned_jobs(workflow, scheduler="slurm")

        assert result["skipped"] == 1
        assert result["recovered"] == 0

        # Job should still be RUNNING (not modified)
        jobs = workflow.get_jobs_by_status(JobStatus.RUNNING)
        assert len(jobs) == 1
        assert jobs[0].worker_id == "slurm_100"


def test_recover_orphans_all_active(tmp_path, monkeypatch):
    """When all scheduler jobs are active, no orphans are detected."""
    from oact_utilities.utils.architector import _init_db, _insert_row
    from oact_utilities.workflows.dashboard import recover_orphaned_jobs

    db_path = tmp_path / "test.db"
    conn = _init_db(db_path)
    _insert_row(
        conn,
        orig_index=0,
        elements="H;H",
        natoms=2,
        geometry="H 0 0 0\nH 0 0 0.74",
        status="to_run",
    )
    conn.commit()
    conn.close()

    with ArchitectorWorkflow(db_path) as workflow:
        workflow.mark_jobs_as_running([1], worker_id="slurm_100")

        # Mock scheduler: slurm_100 IS active
        monkeypatch.setattr(
            "oact_utilities.utils.scheduler.get_active_scheduler_jobs",
            lambda sched: {"slurm_100"},
        )

        result = recover_orphaned_jobs(workflow, scheduler="slurm")

        assert result["recovered"] == 0
        assert result["dead_jobs"] == 0

        # Job should still be RUNNING
        jobs = workflow.get_jobs_by_status(JobStatus.RUNNING)
        assert len(jobs) == 1
