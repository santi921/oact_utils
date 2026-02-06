"""Tests for status checking utilities."""

import os
import tempfile
import time
from pathlib import Path

import pytest

from oact_utilities.utils.status import (
    check_file_termination,
    check_job_termination,
    pull_log_file,
)


def test_check_file_termination_gzipped():
    """Test that check_file_termination works with gzipped ORCA output files."""
    HERE = Path(__file__).resolve().parent
    quacc_dir = HERE / "files" / "quacc_example"

    # Test with gzipped file
    gz_file = quacc_dir / "orca.out.gz"
    assert gz_file.exists(), f"Test file {gz_file} not found"

    status = check_file_termination(str(gz_file))
    assert status == 1, f"Expected status 1 (success), got {status}"


def test_check_job_termination_quacc_output():
    """Test that check_job_termination works with quacc output directory."""
    HERE = Path(__file__).resolve().parent
    quacc_dir = HERE / "files" / "quacc_example"

    assert quacc_dir.exists(), f"Test directory {quacc_dir} not found"

    status = check_job_termination(str(quacc_dir))
    assert status == 1, f"Expected status 1 (success), got {status}"


def test_pull_log_file_gzipped():
    """Test that pull_log_file can find gzipped output files."""
    HERE = Path(__file__).resolve().parent
    quacc_dir = HERE / "files" / "quacc_example"

    log_file = pull_log_file(str(quacc_dir))
    assert log_file.endswith(".out.gz"), f"Expected .out.gz file, got {log_file}"
    assert Path(log_file).exists(), f"Log file {log_file} not found"


def test_check_file_termination_regular():
    """Test that check_file_termination still works with regular ORCA output files."""
    HERE = Path(__file__).resolve().parent
    orca_dir = HERE / "files" / "orca_direct_example"

    # Find a regular .out file in the test directory
    out_files = list(orca_dir.glob("*.out"))
    if not out_files:
        pytest.skip("No regular .out files found in test directory")

    status = check_file_termination(str(out_files[0]))
    # Status can be 0, 1, -1, or -2 depending on the test file
    assert status in [0, 1, -1, -2], f"Unexpected status value: {status}"


def test_check_job_termination_regular():
    """Test that check_job_termination still works with regular ORCA output."""
    HERE = Path(__file__).resolve().parent
    orca_dir = HERE / "files" / "orca_direct_example"

    if not orca_dir.exists():
        pytest.skip("Regular ORCA test directory not found")

    status = check_job_termination(str(orca_dir))
    # Status can be 0, 1, -1, or -2 depending on the test file
    assert status in [0, 1, -1, -2], f"Unexpected status value: {status}"


def test_check_file_termination_timeout():
    """Test that check_file_termination detects timeout (file not modified in 6+ hours)."""
    # Create a temporary file with old modification time
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".out") as f:
        f.write("Test ORCA output\nSome calculation output\n")
        temp_file = f.name

    try:
        # Set file modification time to 7 hours ago
        seven_hours_ago = time.time() - (7 * 3600)
        os.utime(temp_file, (seven_hours_ago, seven_hours_ago))

        # Check that it's detected as timeout
        status = check_file_termination(temp_file)
        assert status == -2, f"Expected status -2 (timeout), got {status}"

    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_check_file_termination_not_timeout():
    """Test that check_file_termination does NOT timeout for recently modified files."""
    # Create a temporary file with recent modification time
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".out") as f:
        f.write("Test ORCA output\nSome calculation output\n")
        temp_file = f.name

    try:
        # File should have recent modification time by default
        # Check that it's NOT detected as timeout (should be 0 = still running)
        status = check_file_termination(temp_file)
        assert status == 0, f"Expected status 0 (still running), got {status}"

    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_check_file_termination_success():
    """Test that check_file_termination correctly detects successful termination."""
    # Create a temporary file with ORCA success message
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".out") as f:
        f.write("Test ORCA output\n")
        f.write("Some calculation output\n")
        f.write("ORCA TERMINATED NORMALLY\n")
        temp_file = f.name

    try:
        status = check_file_termination(temp_file)
        assert status == 1, f"Expected status 1 (success), got {status}"

    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_check_file_termination_abort():
    """Test that check_file_termination correctly detects aborted jobs."""
    # Create a temporary file with ORCA abort message
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".out") as f:
        f.write("Test ORCA output\n")
        f.write("Some calculation output\n")
        f.write("Error: aborting the run\n")
        temp_file = f.name

    try:
        status = check_file_termination(temp_file)
        assert status == -1, f"Expected status -1 (abort), got {status}"

    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.unlink(temp_file)
