"""Tests for submit_jobs module."""

from unittest.mock import MagicMock

import pytest

from oact_utilities.workflows.submit_jobs import (
    DEFAULT_ORCA_CONFIG,
    DEFAULT_ORCA_PATHS,
    OrcaConfig,
    prepare_job_directory,
    write_flux_job_file,
    write_slurm_job_file,
)


@pytest.fixture
def mock_job_record():
    """Create a mock job record."""
    record = MagicMock()
    record.id = 1
    record.orig_index = 42
    record.geometry = """H 0.0 0.0 0.0
H 0.0 0.0 0.74"""
    record.charge = 0
    record.spin = 1
    return record


@pytest.fixture
def mock_actinide_job_record():
    """Create a mock job record with actinide."""
    record = MagicMock()
    record.id = 2
    record.orig_index = 100
    record.geometry = """U 0.0 0.0 0.0
O 1.8 0.0 0.0
O -1.8 0.0 0.0"""
    record.charge = 2
    record.spin = 3
    return record


@pytest.fixture
def orca_config_with_path():
    """ORCA config with a fake orca_path to avoid which() returning None."""
    return {"orca_path": "/fake/path/to/orca"}


class TestPrepareJobDirectory:
    """Tests for prepare_job_directory function."""

    def test_creates_directory(self, mock_job_record, tmp_path, orca_config_with_path):
        """Test that job directory is created."""
        job_dir = prepare_job_directory(
            mock_job_record, tmp_path, orca_config=orca_config_with_path
        )

        assert job_dir.exists()
        assert job_dir.is_dir()
        assert job_dir.name == "job_42"

    def test_creates_orca_inp(self, mock_job_record, tmp_path, orca_config_with_path):
        """Test that orca.inp file is created."""
        job_dir = prepare_job_directory(
            mock_job_record, tmp_path, orca_config=orca_config_with_path
        )

        orca_inp = job_dir / "orca.inp"
        assert orca_inp.exists()

    def test_orca_inp_contains_geometry(
        self, mock_job_record, tmp_path, orca_config_with_path
    ):
        """Test that orca.inp contains the geometry."""
        job_dir = prepare_job_directory(
            mock_job_record, tmp_path, orca_config=orca_config_with_path
        )

        orca_inp = job_dir / "orca.inp"
        content = orca_inp.read_text()

        # Should contain H atoms
        assert "H" in content
        # Should contain coordinates
        assert "0.74" in content

    def test_charge_spin_in_orca_inp(
        self, mock_actinide_job_record, tmp_path, orca_config_with_path
    ):
        """Test that charge and spin are correctly set in orca.inp."""
        job_dir = prepare_job_directory(
            mock_actinide_job_record, tmp_path, orca_config=orca_config_with_path
        )

        orca_inp = job_dir / "orca.inp"
        content = orca_inp.read_text()

        # Should contain charge=2, mult=3 line (format: "* xyz charge mult")
        assert "2 3" in content

    def test_custom_job_dir_pattern(
        self, mock_job_record, tmp_path, orca_config_with_path
    ):
        """Test custom job directory pattern."""
        job_dir = prepare_job_directory(
            mock_job_record,
            tmp_path,
            job_dir_pattern="calc_{id}_{orig_index}",
            orca_config=orca_config_with_path,
        )

        assert job_dir.name == "calc_1_42"

    def test_default_orca_config_applied(
        self, mock_job_record, tmp_path, orca_config_with_path
    ):
        """Test that default ORCA config is applied."""
        job_dir = prepare_job_directory(
            mock_job_record, tmp_path, orca_config=orca_config_with_path
        )

        orca_inp = job_dir / "orca.inp"
        content = orca_inp.read_text()

        # Should contain default functional
        assert DEFAULT_ORCA_CONFIG["functional"] in content

    def test_custom_orca_config(self, mock_job_record, tmp_path):
        """Test that custom ORCA config is applied."""
        orca_config: OrcaConfig = {
            "functional": "B3LYP",
            "opt": True,
            "orca_path": "/fake/path/to/orca",
        }
        job_dir = prepare_job_directory(
            mock_job_record,
            tmp_path,
            orca_config=orca_config,
        )

        orca_inp = job_dir / "orca.inp"
        content = orca_inp.read_text()

        assert "B3LYP" in content
        # Opt keyword should be present
        assert "Opt" in content

    def test_actinide_basis_applied(
        self, mock_actinide_job_record, tmp_path, orca_config_with_path
    ):
        """Test that actinide basis set is applied for U."""
        job_dir = prepare_job_directory(
            mock_actinide_job_record, tmp_path, orca_config=orca_config_with_path
        )

        orca_inp = job_dir / "orca.inp"
        content = orca_inp.read_text()

        # Should contain actinide-specific basis set
        assert DEFAULT_ORCA_CONFIG["actinide_basis"] in content

    def test_setup_func_called(self, mock_job_record, tmp_path, orca_config_with_path):
        """Test that setup_func is called."""
        called_with = []

        def setup_func(job_dir, job_record):
            called_with.append((job_dir, job_record))
            (job_dir / "custom_file.txt").write_text("custom")

        job_dir = prepare_job_directory(
            mock_job_record,
            tmp_path,
            orca_config=orca_config_with_path,
            setup_func=setup_func,
        )

        assert len(called_with) == 1
        assert called_with[0][0] == job_dir
        assert called_with[0][1] == mock_job_record
        assert (job_dir / "custom_file.txt").exists()

    def test_no_geometry_still_creates_dir(self, tmp_path):
        """Test that directory is created even without geometry."""
        record = MagicMock()
        record.id = 1
        record.orig_index = 1
        record.geometry = None

        job_dir = prepare_job_directory(record, tmp_path)

        assert job_dir.exists()
        # No orca.inp should be created without geometry
        assert not (job_dir / "orca.inp").exists()


class TestWriteFluxJobFile:
    """Tests for write_flux_job_file function."""

    def test_creates_flux_script(self, tmp_path):
        """Test that flux script is created."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        flux_script = write_flux_job_file(job_dir)

        assert flux_script.exists()
        assert flux_script.name == "flux_job.flux"

    def test_flux_script_executable(self, tmp_path):
        """Test that flux script is executable."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        flux_script = write_flux_job_file(job_dir)

        assert flux_script.stat().st_mode & 0o755

    def test_flux_script_contains_orca_command(self, tmp_path):
        """Test that flux script runs ORCA directly."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        orca_path = "/path/to/orca"
        flux_script = write_flux_job_file(job_dir, orca_path=orca_path)

        content = flux_script.read_text()

        # Should run ORCA directly, not python orca.py
        assert orca_path in content
        assert "orca.inp" in content
        assert "python orca.py" not in content

    def test_flux_script_cores(self, tmp_path):
        """Test that cores are set in flux script."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        flux_script = write_flux_job_file(job_dir, n_cores=8)

        content = flux_script.read_text()
        assert "#flux: -n 8" in content

    def test_flux_script_hours(self, tmp_path):
        """Test that hours are converted to minutes."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        flux_script = write_flux_job_file(job_dir, n_hours=3)

        content = flux_script.read_text()
        assert "#flux: -t 180m" in content

    def test_flux_script_conda_env(self, tmp_path):
        """Test that conda environment is configurable."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        flux_script = write_flux_job_file(job_dir, conda_env="myenv")

        content = flux_script.read_text()
        assert "conda activate myenv" in content


class TestWriteSlurmJobFile:
    """Tests for write_slurm_job_file function."""

    def test_creates_slurm_script(self, tmp_path):
        """Test that SLURM script is created."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        slurm_script = write_slurm_job_file(job_dir)

        assert slurm_script.exists()
        assert slurm_script.name == "slurm_job.sh"

    def test_slurm_script_executable(self, tmp_path):
        """Test that SLURM script is executable."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        slurm_script = write_slurm_job_file(job_dir)

        assert slurm_script.stat().st_mode & 0o755

    def test_slurm_script_contains_orca_command(self, tmp_path):
        """Test that SLURM script runs ORCA directly."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        orca_path = "/path/to/orca"
        slurm_script = write_slurm_job_file(job_dir, orca_path=orca_path)

        content = slurm_script.read_text()

        assert orca_path in content
        assert "orca.inp" in content
        assert "python orca.py" not in content

    def test_slurm_script_cores(self, tmp_path):
        """Test that cores are set in SLURM script."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        slurm_script = write_slurm_job_file(job_dir, n_cores=8)

        content = slurm_script.read_text()
        assert "#SBATCH --ntasks-per-node 8" in content


class TestDefaultOrcaPaths:
    """Tests for default ORCA path configuration."""

    def test_flux_default_path(self):
        """Test that flux has a default ORCA path."""
        assert "flux" in DEFAULT_ORCA_PATHS
        assert DEFAULT_ORCA_PATHS["flux"].endswith("/orca")

    def test_slurm_default_path(self):
        """Test that slurm has a default ORCA path."""
        assert "slurm" in DEFAULT_ORCA_PATHS
