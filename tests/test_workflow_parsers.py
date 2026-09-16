"""Tests for workflow parsers using real ORCA output files."""

import gzip
import tempfile
from pathlib import Path

import pytest

from oact_utilities.utils.analysis import (
    parse_final_energy,
    parse_job_metrics,
    parse_max_forces,
    parse_scf_steps,
)


@pytest.fixture
def orca_direct_dir():
    """Path to direct ORCA run example."""
    return Path("tests/files/orca_direct_example")


@pytest.fixture
def quacc_dir():
    """Path to quacc run example."""
    return Path("tests/files/quacc_example")


def test_parse_max_forces_direct(orca_direct_dir):
    """Test max forces extraction from direct ORCA output."""
    if not orca_direct_dir.exists():
        pytest.skip("Test data directory not found")

    log_file = orca_direct_dir / "logs"
    if not log_file.exists():
        pytest.skip("Log file not found")

    max_forces = parse_max_forces(str(log_file))

    # Should find a max gradient value
    assert max_forces is not None
    assert isinstance(max_forces, float)
    assert max_forces >= 0  # Forces should be non-negative


def test_parse_scf_steps_direct(orca_direct_dir):
    """Test SCF steps extraction from direct ORCA output."""
    if not orca_direct_dir.exists():
        pytest.skip("Test data directory not found")

    log_file = orca_direct_dir / "logs"
    if not log_file.exists():
        pytest.skip("Log file not found")

    scf_steps = parse_scf_steps(str(log_file))

    # Should find SCF iterations (27 + 17 + 10 + 2 = 56 from 4 geo opt steps)
    assert scf_steps == 56


def test_parse_final_energy_direct(orca_direct_dir):
    """Test final energy extraction from direct ORCA output."""
    if not orca_direct_dir.exists():
        pytest.skip("Test data directory not found")

    log_file = orca_direct_dir / "logs"
    if not log_file.exists():
        pytest.skip("Log file not found")

    energy = parse_final_energy(str(log_file))

    # Should find energy
    assert energy is not None
    assert isinstance(energy, float)
    # Energy should be negative for bound systems
    assert energy < 0


def test_parse_job_metrics_direct(orca_direct_dir):
    """Test complete metrics extraction from direct ORCA run."""
    if not orca_direct_dir.exists():
        pytest.skip("Test data directory not found")

    metrics = parse_job_metrics(orca_direct_dir, unzip=False)

    assert "max_forces" in metrics
    assert "scf_steps" in metrics
    assert "final_energy" in metrics
    assert "success" in metrics

    # Check that values were extracted
    assert metrics["max_forces"] is not None or metrics["success"] is True
    assert metrics["scf_steps"] is not None
    assert metrics["final_energy"] is not None


def test_parse_max_forces_quacc_gzipped(quacc_dir):
    """Test max forces from gzipped quacc ORCA output."""
    if not quacc_dir.exists():
        pytest.skip("Test data directory not found")

    gz_file = quacc_dir / "orca.out.gz"
    if not gz_file.exists():
        pytest.skip("Gzipped output not found")

    # Manually unzip and test
    with gzip.open(gz_file, "rt") as f_in:
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".out"
        ) as f_out:
            f_out.write(f_in.read())
            temp_path = f_out.name

    try:
        max_forces = parse_max_forces(temp_path)

        assert max_forces is not None
        assert isinstance(max_forces, float)
        assert max_forces >= 0
    finally:
        import os

        os.unlink(temp_path)


def test_parse_job_metrics_quacc(quacc_dir):
    """Test complete metrics extraction from quacc run."""
    if not quacc_dir.exists():
        pytest.skip("Test data directory not found")

    metrics = parse_job_metrics(quacc_dir, unzip=True)

    assert "max_forces" in metrics
    assert "scf_steps" in metrics
    assert "final_energy" in metrics
    assert "success" in metrics

    # Quacc job should have succeeded
    assert metrics["success"] is True

    # Check values
    if metrics["max_forces"] is not None:
        assert isinstance(metrics["max_forces"], float)
        assert metrics["max_forces"] >= 0

    if metrics["scf_steps"] is not None:
        assert isinstance(metrics["scf_steps"], int)
        assert metrics["scf_steps"] > 0

    if metrics["final_energy"] is not None:
        assert isinstance(metrics["final_energy"], float)
        assert metrics["final_energy"] < 0


def test_parse_job_metrics_missing_dir():
    """Test that parser handles missing directories gracefully."""
    metrics = parse_job_metrics("/nonexistent/directory", unzip=False)

    assert metrics["max_forces"] is None
    assert metrics["scf_steps"] is None
    assert metrics["final_energy"] is None
    assert metrics["success"] is False


def test_parse_max_forces_missing_file():
    """Test that parser handles missing files gracefully."""
    max_forces = parse_max_forces("/nonexistent/file.out")

    assert max_forces is None


def test_parse_scf_steps_missing_file():
    """Test that parser handles missing files gracefully."""
    scf_steps = parse_scf_steps("/nonexistent/file.out")

    assert scf_steps is None


def test_parse_final_energy_missing_file():
    """Test that parser handles missing files gracefully."""
    energy = parse_final_energy("/nonexistent/file.out")

    assert energy is None


def test_num_electrons_scf_is_ecp_aware_direct(orca_direct_dir):
    """NEL is parsed from the output, not derived from the geometry.

    AmO sums to Z=103 but the Am ECP replaces 60 core electrons, so the SCF
    runs with 43. Any geometry-derived count would be wrong by the ECP size.
    """
    if not (orca_direct_dir / "logs").exists():
        pytest.skip("Test data directory not found")

    metrics = parse_job_metrics(str(orca_direct_dir), recompute=True)
    assert metrics["num_electrons_scf"] == 43


def test_num_electrons_scf_is_ecp_aware_quacc(quacc_dir):
    """NpF3 sums to Z=120; def-ECP on Np leaves 60 electrons in the SCF."""
    if not (quacc_dir / "orca.out.gz").exists():
        pytest.skip("Test data directory not found")

    metrics = parse_job_metrics(str(quacc_dir), unzip=True, recompute=True)
    assert metrics["num_electrons_scf"] == 60


def test_force_max_is_per_atom_norm_not_component(quacc_dir):
    """force_max (engrad per-atom norm) is a different quantity to max_forces.

    max_forces here is ORCA's MAX gradient *component* from the output; the
    norm is always >= the largest component.
    """
    if not (quacc_dir / "orca.engrad.gz").exists():
        pytest.skip("Test data directory not found")

    metrics = parse_job_metrics(str(quacc_dir), unzip=True, recompute=True)

    assert metrics["force_max"] == pytest.approx(0.004533063634215315)
    assert metrics["max_forces"] == pytest.approx(0.0045012181)
    assert metrics["force_max"] > metrics["max_forces"]


def test_force_max_found_with_prefixed_engrad(orca_direct_dir):
    """Direct runs name the engrad after the input (AmO_orca.engrad)."""
    if not (orca_direct_dir / "AmO_orca.engrad").exists():
        pytest.skip("Test data directory not found")

    metrics = parse_job_metrics(str(orca_direct_dir), recompute=True)
    assert metrics["force_max"] == pytest.approx(0.00012826976843802347)
