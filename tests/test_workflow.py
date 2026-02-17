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
        "uhf": [0, 0],  # unpaired electrons
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
    jobs = workflow.get_jobs_by_status(JobStatus.READY)
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
        status="ready",
    )
    _insert_row(
        conn,
        orig_index=1,
        elements="O;H;H",
        natoms=3,
        geometry="O 0 0 0\nH 0.757 0.586 0\nH -0.757 0.586 0",
        status="ready",
    )
    conn.commit()
    conn.close()

    # Test workflow
    with ArchitectorWorkflow(db_path) as workflow:
        # Check initial state
        ready = workflow.get_jobs_by_status(JobStatus.READY)
        assert len(ready) == 2

        # Update one job
        workflow.update_status(1, JobStatus.RUNNING)

        # Check updated state
        ready = workflow.get_jobs_by_status(JobStatus.READY)
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
            status = "ready"
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

        assert counts[JobStatus.READY] == 5
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
    assert row[3] == 1  # spin (uhf=0 -> 2S+1 = 1)

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

    # Insert test rows with TO_RUN status
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
        status="ready",
    )
    conn.commit()
    conn.close()

    with ArchitectorWorkflow(db_path) as workflow:
        # Check TO_RUN jobs
        to_run = workflow.get_jobs_by_status(JobStatus.TO_RUN)
        assert len(to_run) == 1

        # Check that we can get both TO_RUN and READY jobs
        ready_jobs = workflow.get_jobs_by_status([JobStatus.TO_RUN, JobStatus.READY])
        assert len(ready_jobs) == 2


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
    """Test that update_job_status correctly detects timeout jobs."""
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

    # Create a job directory with an old output file (timeout scenario)
    job_dir = tmp_path / "job_timeout"
    job_dir.mkdir()
    output_file = job_dir / "orca.out"

    # Write output with normal termination
    output_file.write_text(
        "ORCA CALCULATION\n"
        "...\n"
        "...\n"
        "...\n"
        "****ORCA TERMINATED NORMALLY****\n"
    )

    # Make file old (> 6 hours)
    eight_hours_ago = time.time() - (8 * 3600)
    os.utime(output_file, (eight_hours_ago, eight_hours_ago))

    with ArchitectorWorkflow(db_path) as workflow:
        # Update job status - should detect timeout
        status = update_job_status(
            workflow=workflow,
            job_dir=job_dir,
            job_id=1,
            extract_metrics=True,
            unzip=False,
        )

        # Should be marked as TIMEOUT
        assert status == JobStatus.TIMEOUT, "Old file should be detected as timeout"

        # Verify in database
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
