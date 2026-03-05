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
            "optimizer": "orca",
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

    def test_ks_method_uks(self, mock_job_record, tmp_path):
        """UKS keyword appears in generated orca.inp."""
        orca_config: OrcaConfig = {
            "functional": "wB97M-V",
            "ks_method": "uks",
            "orca_path": "/fake/path/to/orca",
        }
        job_dir = prepare_job_directory(
            mock_job_record, tmp_path, orca_config=orca_config
        )
        content = (job_dir / "orca.inp").read_text()
        assert "UKS" in content

    def test_ks_method_rks(self, mock_job_record, tmp_path):
        """RKS keyword appears in generated orca.inp."""
        orca_config: OrcaConfig = {
            "ks_method": "rks",
            "orca_path": "/fake/path/to/orca",
        }
        job_dir = prepare_job_directory(
            mock_job_record, tmp_path, orca_config=orca_config
        )
        content = (job_dir / "orca.inp").read_text()
        assert "RKS" in content

    def test_ks_method_roks(self, mock_job_record, tmp_path):
        """ROKS keyword appears in generated orca.inp."""
        orca_config: OrcaConfig = {
            "ks_method": "roks",
            "orca_path": "/fake/path/to/orca",
        }
        job_dir = prepare_job_directory(
            mock_job_record, tmp_path, orca_config=orca_config
        )
        content = (job_dir / "orca.inp").read_text()
        assert "ROKS" in content

    def test_ks_method_none_by_default(self, mock_job_record, tmp_path):
        """No KS keyword when ks_method is not set."""
        orca_config: OrcaConfig = {
            "orca_path": "/fake/path/to/orca",
        }
        job_dir = prepare_job_directory(
            mock_job_record, tmp_path, orca_config=orca_config
        )
        content = (job_dir / "orca.inp").read_text()
        # Should not contain any standalone KS keywords
        # (UKS may appear for actinide singlets via symmetry breaking)
        assert " UKS" not in content or " RKS" not in content
        assert " ROKS" not in content

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

    def test_optimizer_orca_tight(self, mock_job_record, tmp_path):
        """TightOpt keyword appears when optimizer=orca, opt_level=tight."""
        orca_config: OrcaConfig = {
            "optimizer": "orca",
            "opt_level": "tight",
            "orca_path": "/fake/path/to/orca",
        }
        job_dir = prepare_job_directory(
            mock_job_record, tmp_path, orca_config=orca_config
        )
        content = (job_dir / "orca.inp").read_text()
        assert "TightOpt" in content

    def test_optimizer_sella_creates_shim(self, mock_job_record, tmp_path):
        """Sella optimizer generates run_sella.py shim and sella_config.json."""
        import json

        orca_config: OrcaConfig = {
            "optimizer": "sella",
            "orca_path": "/fake/path/to/orca",
        }
        job_dir = prepare_job_directory(
            mock_job_record, tmp_path, orca_config=orca_config
        )
        shim = job_dir / "run_sella.py"
        assert shim.exists()
        content = shim.read_text()
        assert "run_sella_optimization" in content
        assert "sella_config.json" in content

        # Verify JSON config contains correct parameters
        config_file = job_dir / "sella_config.json"
        assert config_file.exists()
        config = json.loads(config_file.read_text())
        assert config["fmax"] == 0.05
        assert config["orca_cmd"] == "/fake/path/to/orca"
        assert config["charge"] == 0
        assert config["mult"] == 1
        assert config["save_all_steps"] is False

    def test_sella_save_all_steps_in_config(self, mock_job_record, tmp_path):
        """save_all_steps=True is serialized into sella_config.json."""
        import json

        orca_config: OrcaConfig = {
            "optimizer": "sella",
            "orca_path": "/fake/path/to/orca",
            "save_all_steps": True,
        }
        job_dir = prepare_job_directory(
            mock_job_record, tmp_path, orca_config=orca_config
        )
        config = json.loads((job_dir / "sella_config.json").read_text())
        assert config["save_all_steps"] is True

    def test_optimizer_none_no_opt(self, mock_job_record, tmp_path):
        """No Opt keyword when optimizer is None (single-point)."""
        orca_config: OrcaConfig = {
            "optimizer": None,
            "orca_path": "/fake/path/to/orca",
        }
        job_dir = prepare_job_directory(
            mock_job_record, tmp_path, orca_config=orca_config
        )
        content = (job_dir / "orca.inp").read_text()
        # Should not contain Opt/TightOpt/etc. keyword (single-point = EnGrad only)
        simple_line = [ln for ln in content.splitlines() if ln.startswith("!")][0]
        assert "Opt" not in simple_line
        assert not (job_dir / "run_sella.py").exists()


class TestFluxSellaJobFile:
    """Tests for Flux job file with Sella optimizer."""

    def test_flux_sella_runs_python(self, tmp_path):
        """Flux script runs python run_sella.py for Sella jobs."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        flux_script = write_flux_job_file(job_dir, optimizer="sella")
        content = flux_script.read_text()

        assert "python run_sella.py" in content
        assert "orca orca.inp" not in content

    def test_flux_orca_runs_orca(self, tmp_path):
        """Flux script runs ORCA directly for ORCA optimizer."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        orca_path = "/path/to/orca"
        flux_script = write_flux_job_file(
            job_dir, orca_path=orca_path, optimizer="orca"
        )
        content = flux_script.read_text()

        assert orca_path in content
        assert "orca.inp" in content
        assert "python run_sella.py" not in content


class TestSlurmSellaJobFile:
    """Tests for SLURM job file with Sella optimizer."""

    def test_slurm_sella_runs_python(self, tmp_path):
        """SLURM script runs python run_sella.py for Sella jobs."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        slurm_script = write_slurm_job_file(job_dir, optimizer="sella")
        content = slurm_script.read_text()

        assert "python run_sella.py" in content
        assert "orca orca.inp" not in content

    def test_slurm_orca_runs_orca(self, tmp_path):
        """SLURM script runs ORCA directly for non-Sella jobs."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        orca_path = "/path/to/orca"
        slurm_script = write_slurm_job_file(
            job_dir, orca_path=orca_path, optimizer="orca"
        )
        content = slurm_script.read_text()

        assert orca_path in content
        assert "orca.inp" in content
        assert "python run_sella.py" not in content


class TestSaveStepOutputs:
    """Tests for _save_step_outputs callback."""

    def test_copies_orca_files_to_step_dir(self, tmp_path):
        """Callback copies all orca.* files into numbered step directories."""
        from oact_utilities.core.orca.sella_runner import _save_step_outputs

        # Create fake ORCA output files
        (tmp_path / "orca.out").write_text("ORCA output")
        (tmp_path / "orca.engrad").write_text("gradient data")
        (tmp_path / "orca.gbw").write_bytes(b"\x00\x01\x02")
        (tmp_path / "orca.property.txt").write_text("properties")
        # Non-orca file should NOT be copied
        (tmp_path / "sella.log").write_text("sella log")

        step_counter: list[int] = [0]

        # First call -> step_000
        _save_step_outputs(job_path=tmp_path, step_counter=step_counter)
        assert step_counter[0] == 1
        step_000 = tmp_path / "step_000"
        assert step_000.is_dir()
        assert (step_000 / "orca.out").read_text() == "ORCA output"
        assert (step_000 / "orca.engrad").read_text() == "gradient data"
        assert (step_000 / "orca.gbw").read_bytes() == b"\x00\x01\x02"
        assert (step_000 / "orca.property.txt").exists()
        assert not (step_000 / "sella.log").exists()

        # Modify orca.out and call again -> step_001
        (tmp_path / "orca.out").write_text("ORCA output step 2")
        _save_step_outputs(job_path=tmp_path, step_counter=step_counter)
        assert step_counter[0] == 2
        step_001 = tmp_path / "step_001"
        assert step_001.is_dir()
        assert (step_001 / "orca.out").read_text() == "ORCA output step 2"

    def test_skips_directories_matching_orca_glob(self, tmp_path):
        """Directories named orca.* are not copied."""
        from oact_utilities.core.orca.sella_runner import _save_step_outputs

        (tmp_path / "orca.out").write_text("output")
        (tmp_path / "orca.tmpdir").mkdir()

        step_counter: list[int] = [0]
        _save_step_outputs(job_path=tmp_path, step_counter=step_counter)

        step_000 = tmp_path / "step_000"
        assert (step_000 / "orca.out").exists()
        assert not (step_000 / "orca.tmpdir").exists()


class TestDefaultOrcaPaths:
    """Tests for default ORCA path configuration."""

    def test_flux_default_path(self):
        """Test that flux has a default ORCA path."""
        assert "flux" in DEFAULT_ORCA_PATHS
        assert DEFAULT_ORCA_PATHS["flux"].endswith("/orca")

    def test_slurm_default_path(self):
        """Test that slurm has a default ORCA path."""
        assert "slurm" in DEFAULT_ORCA_PATHS
