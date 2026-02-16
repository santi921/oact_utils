import gzip
import tempfile
from pathlib import Path

import numpy as np

from oact_utilities.utils.analysis import (
    get_rmsd_start_final,
    parse_job_metrics,
    parse_mulliken_population,
)


def test_get_rmsd_start_final():
    test_dir = Path(__file__).parent / "files"
    res_no_traj = get_rmsd_start_final(str(test_dir / "no_traj"))
    res_traj = get_rmsd_start_final(str(test_dir / "traj"))
    np.testing.assert_array_almost_equal(
        res_no_traj["energies_frames"],
        res_traj["energies_frames"],
        decimal=5,
        err_msg="Energies from no_traj and traj do not match",
    )


def test_parse_mulliken_population():
    """Test parsing Mulliken population analysis from ORCA output."""
    # Test with gzipped quacc example
    test_dir = Path(__file__).parent / "files"
    gz_file = test_dir / "quacc_example" / "orca.out.gz"

    # Unzip to temp file for testing
    with gzip.open(gz_file, "rt") as f_in:
        content = f_in.read()
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".out"
        ) as f_out:
            f_out.write(content)
            temp_path = f_out.name

    try:
        result = parse_mulliken_population(temp_path)

        # Check that we got results
        assert result is not None, "Should parse Mulliken data from test file"

        # Check structure
        assert "mulliken_charges" in result
        assert "mulliken_spins" in result
        assert "elements" in result
        assert "indices" in result

        # Check that we have data
        assert len(result["mulliken_charges"]) > 0, "Should have Mulliken charges"
        assert len(result["mulliken_spins"]) > 0, "Should have Mulliken spins"
        assert len(result["elements"]) > 0, "Should have element symbols"
        assert len(result["indices"]) > 0, "Should have atomic indices"

        # Check that all lists have the same length
        n_atoms = len(result["elements"])
        assert len(result["mulliken_charges"]) == n_atoms
        assert len(result["mulliken_spins"]) == n_atoms
        assert len(result["indices"]) == n_atoms

        # Check specific values from the test file
        # From the grep output above: 0 Np:    1.648611    4.000610
        assert result["elements"][0] == "Np", "First element should be Np"
        assert abs(result["mulliken_charges"][0] - 1.648611) < 0.001
        assert abs(result["mulliken_spins"][0] - 4.000610) < 0.001

        # Check that we have 4 atoms total (Np + 3 F)
        assert n_atoms == 4, "Should have 4 atoms in NpF3"

        # Check sum of charges is close to zero
        charge_sum = sum(result["mulliken_charges"])
        assert abs(charge_sum) < 0.001, "Sum of charges should be ~0"

        # Check sum of spins is close to 4
        spin_sum = sum(result["mulliken_spins"])
        assert abs(spin_sum - 4.0) < 0.001, "Sum of spins should be ~4"

    finally:
        Path(temp_path).unlink()


def test_parse_job_metrics_with_mulliken():
    """Test that parse_job_metrics includes Mulliken population analysis."""
    test_dir = Path(__file__).parent / "files"
    job_dir = test_dir / "quacc_example"

    result = parse_job_metrics(job_dir, unzip=True)

    # Check that we got the basic metrics
    assert result is not None
    assert "mulliken_population" in result

    # Check that Mulliken data was parsed
    mulliken = result["mulliken_population"]
    if mulliken is not None:
        assert "mulliken_charges" in mulliken
        assert "mulliken_spins" in mulliken
        assert len(mulliken["mulliken_charges"]) > 0
