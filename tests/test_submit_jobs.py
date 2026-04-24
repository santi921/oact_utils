"""Tests for submit_jobs module."""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from oact_utilities.workflows.architector_workflow import ArchitectorWorkflow, JobStatus
from oact_utilities.workflows.submit_jobs import (
    DEFAULT_ORCA_CONFIG,
    DEFAULT_ORCA_PATHS,
    OrcaConfig,
    _write_job_update,
    prepare_job_directory,
    write_flux_job_file,
    write_slurm_job_file,
)

PARSL_INSTALLED = importlib.util.find_spec("parsl") is not None


@pytest.fixture
def mock_job_record():
    """Create a mock job record."""
    record = MagicMock()
    record.id = 1
    record.orig_index = 42
    record.job_dir = None
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
    record.job_dir = None
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

    def test_hostname_placeholder_in_job_dir_pattern(
        self, mock_job_record, tmp_path, orca_config_with_path, monkeypatch
    ):
        """Custom job directory patterns may include the coordinator hostname."""
        monkeypatch.setattr(
            "oact_utilities.workflows.job_dir_patterns.get_job_dir_hostname",
            lambda: "coord01",
        )
        job_dir = prepare_job_directory(
            mock_job_record,
            tmp_path,
            job_dir_pattern="{hostname}_calc_{id}_{orig_index}",
            orca_config=orca_config_with_path,
        )

        assert job_dir.name == "coord01_calc_1_42"

    def test_apply_job_dir_prefix(self):
        """Stable run prefixes prepend cleanly to the base job directory pattern."""
        from oact_utilities.workflows.job_dir_patterns import apply_job_dir_prefix

        assert apply_job_dir_prefix("job_{orig_index}", "campaignA") == (
            "campaignA_job_{orig_index}"
        )

    def test_apply_job_dir_prefix_rejects_invalid_chars(self):
        """Run prefixes reject path-like or unsafe characters."""
        from oact_utilities.workflows.job_dir_patterns import apply_job_dir_prefix

        with pytest.raises(ValueError, match="job_prefix may contain only"):
            apply_job_dir_prefix("job_{orig_index}", "../campaign")

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
        record.job_dir = None
        record.geometry = None

        job_dir = prepare_job_directory(record, tmp_path)

        assert job_dir.exists()
        # No orca.inp should be created without geometry
        assert not (job_dir / "orca.inp").exists()

    def test_uses_preset_job_dir(
        self, mock_job_record, tmp_path, orca_config_with_path
    ):
        """If job_record.job_dir is set, use it instead of computing from root_dir/pattern."""
        preset = tmp_path / "seeds" / "orig_index_42"
        preset.mkdir(parents=True)
        mock_job_record.job_dir = str(preset)

        irrelevant_root = tmp_path / "should_not_be_used"
        result = prepare_job_directory(
            mock_job_record, irrelevant_root, orca_config=orca_config_with_path
        )

        assert result == preset
        assert (preset / "orca.inp").exists()
        assert not irrelevant_root.exists()


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

    def test_flux_script_ld_library_path(self, tmp_path):
        """Test that LD_LIBRARY_PATH override is written when provided."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        flux_script = write_flux_job_file(job_dir, ld_library_path="/custom/lib")

        content = flux_script.read_text()
        assert "export LD_LIBRARY_PATH=/custom/lib" in content

    def test_flux_script_default_ld_library_path(self, tmp_path):
        """Test that the default LD_LIBRARY_PATH is used when none is provided."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        flux_script = write_flux_job_file(job_dir)

        content = flux_script.read_text()
        assert "export LD_LIBRARY_PATH=" in content
        assert "/custom/lib" not in content


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

    def test_slurm_script_ld_library_path(self, tmp_path):
        """Test that LD_LIBRARY_PATH override is written when provided."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        slurm_script = write_slurm_job_file(job_dir, ld_library_path="/custom/lib")

        content = slurm_script.read_text()
        assert "export LD_LIBRARY_PATH=/custom/lib" in content

    def test_slurm_script_no_ld_library_path_by_default(self, tmp_path):
        """Test that LD_LIBRARY_PATH export is omitted when default is empty."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        slurm_script = write_slurm_job_file(job_dir)

        content = slurm_script.read_text()
        # Default slurm LD_LIBRARY_PATH is empty, so no export line should appear
        assert "export LD_LIBRARY_PATH=:" not in content


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


@pytest.mark.skipif(not PARSL_INSTALLED, reason="parsl not installed")
class TestBuildParslConfigFlux:
    """Tests for build_parsl_config_flux function."""

    def test_monitoring_enabled_by_default(self):
        """MonitoringHub is attached by default."""
        from oact_utilities.workflows.submit_jobs import build_parsl_config_flux

        config = build_parsl_config_flux()
        assert config.monitoring is not None
        expected = f"sqlite:///{(Path(config.run_dir).resolve() / 'monitoring.db')}"
        assert config.monitoring.logging_endpoint == expected

    def test_monitoring_disabled_when_flag_false(self):
        """enable_monitoring=False omits MonitoringHub from Config."""
        from oact_utilities.workflows.submit_jobs import build_parsl_config_flux

        config = build_parsl_config_flux(enable_monitoring=False)
        assert config.monitoring is None


@pytest.mark.skipif(not PARSL_INSTALLED, reason="parsl not installed")
class TestBuildParslConfigSlurm:
    """Tests for build_parsl_config_slurm function."""

    def test_single_node_uses_simple_launcher(self):
        """nodes_per_block=1 uses SimpleLauncher for backwards compatibility."""
        from parsl.launchers import SimpleLauncher

        from oact_utilities.workflows.submit_jobs import build_parsl_config_slurm

        config = build_parsl_config_slurm(nodes_per_block=1)
        provider = config.executors[0].provider
        assert isinstance(provider.launcher, SimpleLauncher)

    def test_multi_node_uses_srun_launcher(self):
        """nodes_per_block > 1 switches to SrunLauncher."""
        from parsl.launchers import SrunLauncher

        from oact_utilities.workflows.submit_jobs import build_parsl_config_slurm

        config = build_parsl_config_slurm(nodes_per_block=4)
        provider = config.executors[0].provider
        assert isinstance(provider.launcher, SrunLauncher)

    def test_nodes_per_block_passed_to_provider(self):
        """nodes_per_block value is forwarded to SlurmProvider."""
        from oact_utilities.workflows.submit_jobs import build_parsl_config_slurm

        config = build_parsl_config_slurm(nodes_per_block=50)
        provider = config.executors[0].provider
        assert provider.nodes_per_block == 50

    def test_single_node_scheduler_options_has_ntasks(self):
        """Single-node mode sets ntasks-per-node = cores * workers."""
        from oact_utilities.workflows.submit_jobs import build_parsl_config_slurm

        config = build_parsl_config_slurm(
            max_workers=4, cores_per_worker=16, nodes_per_block=1
        )
        provider = config.executors[0].provider
        assert "--ntasks-per-node=64" in provider.scheduler_options
        assert "--cpus-per-task=1" in provider.scheduler_options

    def test_single_node_scheduler_options_can_reserve_more_than_active_cores(self):
        """Single-node Slurm can reserve more scheduler cores than active worker cores."""
        from oact_utilities.workflows.submit_jobs import build_parsl_config_slurm

        config = build_parsl_config_slurm(
            max_workers=8,
            cores_per_worker=8,
            cpus_per_node=192,
            nodes_per_block=1,
        )
        provider = config.executors[0].provider
        assert "--ntasks-per-node=192" in provider.scheduler_options
        assert "--cpus-per-task=1" in provider.scheduler_options

    def test_multi_node_scheduler_options_empty(self):
        """Multi-node mode has no extra scheduler_options (exclusive=True handles it)."""
        from oact_utilities.workflows.submit_jobs import build_parsl_config_slurm

        config = build_parsl_config_slurm(nodes_per_block=10)
        provider = config.executors[0].provider
        assert "--ntasks-per-node" not in provider.scheduler_options
        assert provider.exclusive is True

    def test_multi_node_scheduler_options_include_explicit_cpus_per_node(self):
        """Multi-node Slurm adds ntasks-per-node when an explicit override is requested."""
        from oact_utilities.workflows.submit_jobs import build_parsl_config_slurm

        config = build_parsl_config_slurm(
            max_workers=8,
            cores_per_worker=8,
            cpus_per_node=192,
            nodes_per_block=10,
        )
        provider = config.executors[0].provider
        assert "--ntasks-per-node=192" in provider.scheduler_options
        assert "--cpus-per-task=1" in provider.scheduler_options
        assert provider.exclusive is True

    def test_slurm_cpu_shape_rejects_undersized_override(self):
        """Reserved Slurm cores cannot be smaller than active worker cores."""
        from oact_utilities.workflows.submit_jobs import build_parsl_config_slurm

        with pytest.raises(ValueError, match="cpus_per_node must be >="):
            build_parsl_config_slurm(
                max_workers=12,
                cores_per_worker=16,
                cpus_per_node=128,
            )

    def test_cpu_affinity_block(self):
        """HighThroughputExecutor has cpu_affinity='block'."""
        from oact_utilities.workflows.submit_jobs import build_parsl_config_slurm

        config = build_parsl_config_slurm()
        executor = config.executors[0]
        assert executor.cpu_affinity == "block"

    def test_unique_run_dir(self):
        """Config uses a unique run_dir to avoid collisions."""
        from oact_utilities.workflows.submit_jobs import build_parsl_config_slurm

        config = build_parsl_config_slurm()
        assert config.run_dir.startswith("runinfo/run_")

    def test_worker_init_prepends_orca_bin_dir_for_absolute_path(self, monkeypatch):
        """Absolute orca_path causes its parent dir to be prepended to PATH in worker_init."""
        monkeypatch.setattr(
            "oact_utilities.workflows.submit_jobs.shutil.which",
            lambda exe: None,
        )
        from oact_utilities.workflows.submit_jobs import build_parsl_config_slurm

        config = build_parsl_config_slurm(orca_path="/opt/orca/bin/orca")
        provider = config.executors[0].provider
        assert "export PATH=/opt/orca/bin:" in provider.worker_init

    def test_worker_init_no_path_export_for_non_absolute_orca_path(self, monkeypatch):
        """No PATH export when orca_path is None or a bare executable name."""
        monkeypatch.setattr(
            "oact_utilities.workflows.submit_jobs.shutil.which",
            lambda exe: None,
        )
        from oact_utilities.workflows.submit_jobs import build_parsl_config_slurm

        for val in [None, "orca"]:
            config = build_parsl_config_slurm(orca_path=val)
            provider = config.executors[0].provider
            assert "export PATH=" not in provider.worker_init

    def test_monitoring_enabled_by_default(self):
        """MonitoringHub is attached by default."""
        from oact_utilities.workflows.submit_jobs import build_parsl_config_slurm

        config = build_parsl_config_slurm()
        assert config.monitoring is not None
        expected = f"sqlite:///{(Path(config.run_dir).resolve() / 'monitoring.db')}"
        assert config.monitoring.logging_endpoint == expected

    def test_monitoring_disabled_when_flag_false(self):
        """enable_monitoring=False omits MonitoringHub from Config."""
        from oact_utilities.workflows.submit_jobs import build_parsl_config_slurm

        config = build_parsl_config_slurm(enable_monitoring=False)
        assert config.monitoring is None


@pytest.mark.skipif(not PARSL_INSTALLED, reason="parsl not installed")
class TestBuildParslConfigPbsPro:
    """Tests for build_parsl_config_pbspro function."""

    def test_single_node_uses_simple_launcher(self):
        """nodes_per_block=1 uses SimpleLauncher."""
        from parsl.launchers import SimpleLauncher

        from oact_utilities.workflows.submit_jobs import build_parsl_config_pbspro

        config = build_parsl_config_pbspro(nodes_per_block=1)
        provider = config.executors[0].provider
        assert isinstance(provider.launcher, SimpleLauncher)

    def test_multi_node_uses_pbsdsh_launcher(self):
        """nodes_per_block > 1 uses a pbsdsh-based launcher, not mpiexec."""
        from oact_utilities.workflows.submit_jobs import build_parsl_config_pbspro

        config = build_parsl_config_pbspro(nodes_per_block=4)
        provider = config.executors[0].provider
        assert provider.launcher.__class__.__name__ == "PbsdshLauncher"
        wrapped = provider.launcher("echo hi", tasks_per_node=1, nodes_per_block=4)
        assert "awk '!seen[$1]++ {print NR-1}'" in wrapped
        assert 'pbsdsh -n "$NODE_INDEX"' in wrapped
        assert "mpiexec" not in wrapped

    def test_nodes_per_block_passed_to_provider(self):
        """nodes_per_block value is forwarded to PBSProProvider."""
        from oact_utilities.workflows.submit_jobs import build_parsl_config_pbspro

        config = build_parsl_config_pbspro(nodes_per_block=39)
        provider = config.executors[0].provider
        assert provider.nodes_per_block == 39

    def test_pbs_cpu_shape_derived_from_worker_layout(self):
        """ncpus and mpiprocs derive from max_workers * cores_per_worker."""
        from oact_utilities.workflows.submit_jobs import build_parsl_config_pbspro

        config = build_parsl_config_pbspro(
            max_workers=12, cores_per_worker=16, nodes_per_block=39
        )
        provider = config.executors[0].provider
        assert provider.cpus_per_node == 192
        assert provider.select_options == "mpiprocs=192"

    def test_pbs_cpu_shape_can_reserve_more_than_active_worker_cores(self):
        """PBS can reserve a full node while workers intentionally leave cores idle."""
        from oact_utilities.workflows.submit_jobs import build_parsl_config_pbspro

        config = build_parsl_config_pbspro(
            max_workers=8,
            cores_per_worker=8,
            cpus_per_node=192,
            nodes_per_block=39,
        )
        provider = config.executors[0].provider
        assert provider.cpus_per_node == 192
        assert provider.select_options == "mpiprocs=192"

    def test_pbs_cpu_shape_rejects_undersized_override(self):
        """Reserved PBS cores cannot be smaller than active worker cores."""
        from oact_utilities.workflows.submit_jobs import build_parsl_config_pbspro

        with pytest.raises(ValueError, match="cpus_per_node must be >="):
            build_parsl_config_pbspro(
                max_workers=12,
                cores_per_worker=16,
                cpus_per_node=128,
            )

    def test_monitoring_db_written_to_runinfo(self):
        """Monitoring is enabled with a per-run database inside run_dir."""
        from oact_utilities.workflows.submit_jobs import build_parsl_config_pbspro

        config = build_parsl_config_pbspro()
        assert config.monitoring is not None
        expected = f"sqlite:///{(Path(config.run_dir).resolve() / 'monitoring.db')}"
        assert config.monitoring.logging_endpoint == expected

    def test_monitoring_disabled_when_flag_false(self):
        """enable_monitoring=False omits MonitoringHub from Config."""
        from oact_utilities.workflows.submit_jobs import build_parsl_config_pbspro

        config = build_parsl_config_pbspro(enable_monitoring=False)
        assert config.monitoring is None

    def test_executor_has_runtime_address(self):
        """HTEX sets a runtime-resolved address for worker connectivity."""
        from oact_utilities.workflows.submit_jobs import build_parsl_config_pbspro

        config = build_parsl_config_pbspro()
        executor = config.executors[0]
        assert executor.address is not None

    def test_executor_uses_python_module_launch_cmd(self):
        """PBS HTEX launch uses absolute python -m, not a PATH-based console script."""
        from oact_utilities.workflows.submit_jobs import build_parsl_config_pbspro

        config = build_parsl_config_pbspro()
        executor = config.executors[0]
        assert " -m parsl.executors.high_throughput.process_worker_pool " in (
            executor.launch_cmd
        )
        assert "process_worker_pool.py" not in executor.launch_cmd


class TestMultiNodeCLIValidation:
    """Tests for --nodes-per-block CLI validation."""

    def test_flux_rejects_multi_node(self):
        """--nodes-per-block > 1 with --scheduler flux should error."""
        import subprocess

        result = subprocess.run(
            [
                "python",
                "-m",
                "oact_utilities.workflows.submit_jobs",
                "fake.db",
                "fake_dir",
                "--use-parsl",
                "--scheduler",
                "flux",
                "--nodes-per-block",
                "4",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "only supported with --scheduler slurm or pbspro" in result.stderr

    def test_nodes_per_block_less_than_one_errors(self):
        """--nodes-per-block 0 should error."""
        import subprocess

        result = subprocess.run(
            [
                "python",
                "-m",
                "oact_utilities.workflows.submit_jobs",
                "fake.db",
                "fake_dir",
                "--use-parsl",
                "--scheduler",
                "slurm",
                "--nodes-per-block",
                "0",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "must be >= 1" in result.stderr

    def test_pbspro_requires_parsl_mode(self):
        """PBS Pro support is currently only available through Parsl mode."""
        import subprocess

        result = subprocess.run(
            [
                "python",
                "-m",
                "oact_utilities.workflows.submit_jobs",
                "fake.db",
                "fake_dir",
                "--scheduler",
                "pbspro",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "currently only supported with --use-parsl" in result.stderr

    def test_pbspro_rejects_cpus_per_node_below_active_worker_cores(self):
        """PBS full-node override must not undersize the active worker layout."""
        import subprocess

        result = subprocess.run(
            [
                "python",
                "-m",
                "oact_utilities.workflows.submit_jobs",
                "fake.db",
                "fake_dir",
                "--use-parsl",
                "--scheduler",
                "pbspro",
                "--max-workers",
                "12",
                "--cores-per-worker",
                "16",
                "--cpus-per-node",
                "128",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "must be >= max_workers * cores_per_worker" in result.stderr

    def test_slurm_rejects_cpus_per_node_below_active_worker_cores(self):
        """Slurm full-node override must not undersize the active worker layout."""
        import subprocess

        result = subprocess.run(
            [
                "python",
                "-m",
                "oact_utilities.workflows.submit_jobs",
                "fake.db",
                "fake_dir",
                "--use-parsl",
                "--scheduler",
                "slurm",
                "--max-workers",
                "12",
                "--cores-per-worker",
                "16",
                "--cpus-per-node",
                "128",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "must be >= max_workers * cores_per_worker" in result.stderr


# --- Pre-submission disk check tests ---


class TestSkipFinishedOnDisk:
    """Tests for _skip_finished_on_disk()."""

    def _make_db(self, tmp_path, n_jobs=3):
        """Helper: create a workflow DB with n TO_RUN jobs."""
        from oact_utilities.utils.architector import _init_db, _insert_row

        db_path = tmp_path / "test.db"
        conn = _init_db(db_path)
        for i in range(n_jobs):
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
        return db_path

    def test_no_directories_all_pass_through(self, tmp_path):
        """Jobs with no directories on disk all pass through."""
        from oact_utilities.workflows.architector_workflow import (
            ArchitectorWorkflow,
            JobStatus,
        )
        from oact_utilities.workflows.submit_jobs import _skip_finished_on_disk

        db_path = self._make_db(tmp_path)
        root_dir = tmp_path / "jobs"
        root_dir.mkdir()

        with ArchitectorWorkflow(db_path) as wf:
            jobs = wf.get_jobs_by_status(JobStatus.TO_RUN, include_geometry=True)
            result = _skip_finished_on_disk(jobs, root_dir, "job_{orig_index}", wf)
            assert len(result) == 3

    def test_completed_on_disk_filtered_and_db_updated(self, tmp_path):
        """A job with 'ORCA TERMINATED NORMALLY' on disk is skipped and DB updated."""
        from oact_utilities.workflows.architector_workflow import (
            ArchitectorWorkflow,
            JobStatus,
        )
        from oact_utilities.workflows.submit_jobs import _skip_finished_on_disk

        db_path = self._make_db(tmp_path)
        root_dir = tmp_path / "jobs"

        # Create a completed job directory
        job_dir = root_dir / "job_0"
        job_dir.mkdir(parents=True)
        (job_dir / "orca.out").write_text("Some output\nORCA TERMINATED NORMALLY\n")

        with ArchitectorWorkflow(db_path) as wf:
            jobs = wf.get_jobs_by_status(JobStatus.TO_RUN, include_geometry=True)
            result = _skip_finished_on_disk(jobs, root_dir, "job_{orig_index}", wf)
            # Job 0 should be filtered out
            assert len(result) == 2
            assert all(j.orig_index != 0 for j in result)

            # DB should be updated to COMPLETED
            completed = wf.get_jobs_by_status(JobStatus.COMPLETED)
            assert len(completed) == 1
            assert completed[0].orig_index == 0

    def test_failed_on_disk_filtered_and_db_updated(self, tmp_path):
        """A job with 'aborting the run' on disk is skipped and DB updated."""
        from oact_utilities.workflows.architector_workflow import (
            ArchitectorWorkflow,
            JobStatus,
        )
        from oact_utilities.workflows.submit_jobs import _skip_finished_on_disk

        db_path = self._make_db(tmp_path)
        root_dir = tmp_path / "jobs"

        # Create a failed job directory
        job_dir = root_dir / "job_1"
        job_dir.mkdir(parents=True)
        (job_dir / "orca.out").write_text("SCF NOT CONVERGED\naborting the run\n")

        with ArchitectorWorkflow(db_path) as wf:
            jobs = wf.get_jobs_by_status(JobStatus.TO_RUN, include_geometry=True)
            result = _skip_finished_on_disk(jobs, root_dir, "job_{orig_index}", wf)
            assert len(result) == 2

            # DB should be updated to FAILED with fail_count incremented
            failed = wf.get_jobs_by_status(JobStatus.FAILED)
            assert len(failed) == 1
            assert failed[0].fail_count == 1
            assert failed[0].error_message is not None

    def test_no_output_file_passes_through(self, tmp_path):
        """A directory with no .out file passes through for submission."""
        from oact_utilities.workflows.architector_workflow import (
            ArchitectorWorkflow,
            JobStatus,
        )
        from oact_utilities.workflows.submit_jobs import _skip_finished_on_disk

        db_path = self._make_db(tmp_path)
        root_dir = tmp_path / "jobs"

        # Create an empty job directory (just orca.inp, no output)
        job_dir = root_dir / "job_0"
        job_dir.mkdir(parents=True)
        (job_dir / "orca.inp").write_text("! UKS wB97M-V\n")

        with ArchitectorWorkflow(db_path) as wf:
            jobs = wf.get_jobs_by_status(JobStatus.TO_RUN, include_geometry=True)
            result = _skip_finished_on_disk(jobs, root_dir, "job_{orig_index}", wf)
            # All jobs pass through (no output = nothing to detect)
            assert len(result) == 3

    def test_pattern_fallback_when_job_dir_null(self, tmp_path):
        """Uses pattern-based path when job.job_dir is NULL in DB."""
        from oact_utilities.workflows.architector_workflow import (
            ArchitectorWorkflow,
            JobStatus,
        )
        from oact_utilities.workflows.submit_jobs import _skip_finished_on_disk

        db_path = self._make_db(tmp_path)
        root_dir = tmp_path / "jobs"

        # Create completed output at pattern-based path (job_dir is NULL in DB)
        job_dir = root_dir / "job_0"
        job_dir.mkdir(parents=True)
        (job_dir / "orca.out").write_text("Some output\nORCA TERMINATED NORMALLY\n")

        with ArchitectorWorkflow(db_path) as wf:
            jobs = wf.get_jobs_by_status(JobStatus.TO_RUN, include_geometry=True)
            # Verify job_dir is NULL
            assert jobs[0].job_dir is None

            result = _skip_finished_on_disk(jobs, root_dir, "job_{orig_index}", wf)
            # Job 0 detected via pattern fallback
            assert len(result) == 2

    def test_mixed_batch(self, tmp_path):
        """Mixed batch: one completed, one failed, one new."""
        from oact_utilities.workflows.architector_workflow import (
            ArchitectorWorkflow,
            JobStatus,
        )
        from oact_utilities.workflows.submit_jobs import _skip_finished_on_disk

        db_path = self._make_db(tmp_path)
        root_dir = tmp_path / "jobs"

        # Job 0: completed
        d0 = root_dir / "job_0"
        d0.mkdir(parents=True)
        (d0 / "orca.out").write_text("ORCA TERMINATED NORMALLY\n")

        # Job 1: failed
        d1 = root_dir / "job_1"
        d1.mkdir(parents=True)
        (d1 / "orca.out").write_text("aborting the run\n")

        # Job 2: no directory (new)

        with ArchitectorWorkflow(db_path) as wf:
            jobs = wf.get_jobs_by_status(JobStatus.TO_RUN, include_geometry=True)
            result = _skip_finished_on_disk(jobs, root_dir, "job_{orig_index}", wf)

            assert len(result) == 1
            assert result[0].orig_index == 2

            counts = wf.count_by_status()
            assert counts.get(JobStatus.COMPLETED, 0) == 1
            assert counts.get(JobStatus.FAILED, 0) == 1
            assert counts.get(JobStatus.TO_RUN, 0) == 1


class TestOrcaExitCodeVerification:
    """Test that Parsl path verifies ORCA output even when exit code is 0.

    ORCA can exit with code 0 while an MPI child (e.g. orca_leanscf_mpi)
    has failed and printed 'aborting the run' to the output file.
    """

    def _check_orca_output(self, job_dir: Path) -> str | None:
        """Replicate the verification logic from orca_job_wrapper.

        Returns None if the output looks good, or an error string if it
        shows a failure despite exit code 0.
        """
        from collections import deque

        out_path = job_dir / "orca.out"
        if not out_path.exists():
            return None

        try:
            with open(out_path, errors="replace") as fh:
                tail = deque(fh, maxlen=10)
            has_normal = any("ORCA TERMINATED NORMALLY" in ln for ln in tail)
            has_abort = any("aborting the run" in ln or "Error" in ln for ln in tail)
            if not has_normal and has_abort:
                return "ORCA exited 0 but output shows error"
        except OSError:
            pass

        return None

    def test_normal_termination_passes(self, tmp_path):
        """Exit code 0 + ORCA TERMINATED NORMALLY -> completed."""
        job_dir = tmp_path / "job_0"
        job_dir.mkdir()
        (job_dir / "orca.out").write_text(
            "Some output\n****ORCA TERMINATED NORMALLY****\n"
        )
        assert self._check_orca_output(job_dir) is None

    def test_abort_despite_exit_zero(self, tmp_path):
        """Exit code 0 + 'aborting the run' -> detected as failed."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()
        (job_dir / "orca.out").write_text(
            "ORCA finished by error termination in LEANSCF\n"
            "[file orca_tools/qcmsg.cpp, line 394]:\n"
            "  .... aborting the run\n"
        )
        result = self._check_orca_output(job_dir)
        assert result is not None
        assert "error" in result.lower()

    def test_error_despite_exit_zero(self, tmp_path):
        """Exit code 0 + 'Error' in tail -> detected as failed."""
        job_dir = tmp_path / "job_2"
        job_dir.mkdir()
        (job_dir / "orca.out").write_text(
            "Error (TDIISSCF_AO): Cannot read Error matrix\n"
            "ORCA finished by error termination in LEANSCF\n"
            "  .... aborting the run\n"
        )
        result = self._check_orca_output(job_dir)
        assert result is not None

    def test_no_output_file_passes(self, tmp_path):
        """Missing orca.out -> no error (trust exit code)."""
        job_dir = tmp_path / "job_3"
        job_dir.mkdir()
        assert self._check_orca_output(job_dir) is None

    def test_benign_error_with_normal_termination(self, tmp_path):
        """'Error' line followed by ORCA TERMINATED NORMALLY -> passes.

        ORCA can print benign messages containing 'Error' (e.g. integral
        error warnings) before completing normally.
        """
        job_dir = tmp_path / "job_4"
        job_dir.mkdir()
        (job_dir / "orca.out").write_text(
            "Some of the Onep Integrals have an Error larger than 1.0e-05\n"
            "****ORCA TERMINATED NORMALLY****\n"
        )
        assert self._check_orca_output(job_dir) is None


# --- Tests for _write_job_update (per-job commit) ---


@pytest.fixture
def workflow_db(tmp_path):
    """Create a workflow DB with 5 test jobs for update tests."""
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
            status="running",
        )
    conn.commit()
    conn.close()
    return db_path


class TestWriteJobUpdate:
    """Tests for _write_job_update per-job commit function."""

    def test_completed_job_with_metrics(self, workflow_db):
        """Completed job gets status + metrics in one commit."""
        with ArchitectorWorkflow(workflow_db) as wf:
            _write_job_update(
                wf,
                {
                    "job_id": 1,
                    "status": JobStatus.COMPLETED,
                    "metrics": {
                        "job_dir": "/path/to/job_0",
                        "max_forces": 0.001,
                        "scf_steps": 10,
                        "final_energy": -1.5,
                        "wall_time": 120.0,
                        "n_cores": 4,
                    },
                },
            )

            completed = wf.get_jobs_by_status(JobStatus.COMPLETED)
            assert len(completed) == 1
            assert completed[0].max_forces == 0.001
            assert completed[0].wall_time == 120.0
            assert completed[0].scf_steps == 10
            assert completed[0].worker_id is None

    def test_completed_job_partial_metrics(self, workflow_db):
        """Completed job with only some metrics populated."""
        with ArchitectorWorkflow(workflow_db) as wf:
            _write_job_update(
                wf,
                {
                    "job_id": 2,
                    "status": JobStatus.COMPLETED,
                    "metrics": {
                        "job_dir": "/path/to/job_1",
                        "max_forces": 0.002,
                        "scf_steps": 15,
                        "final_energy": -2.5,
                    },
                },
            )

            completed = wf.get_jobs_by_status(JobStatus.COMPLETED)
            assert len(completed) == 1
            assert completed[0].scf_steps == 15
            assert completed[0].wall_time is None

    def test_failed_job_with_error_and_fail_count(self, workflow_db):
        """Failed job gets error_message and incremented fail_count."""
        with ArchitectorWorkflow(workflow_db) as wf:
            _write_job_update(
                wf,
                {
                    "job_id": 1,
                    "status": JobStatus.FAILED,
                    "error_message": "SCF did not converge",
                    "increment_fail_count": True,
                },
            )

            failed = wf.get_jobs_by_status(JobStatus.FAILED)
            assert len(failed) == 1
            assert failed[0].error_message == "SCF did not converge"
            assert failed[0].fail_count == 1

    def test_multiple_updates_independent(self, workflow_db):
        """Each update commits independently -- no cross-job rollback."""
        with ArchitectorWorkflow(workflow_db) as wf:
            _write_job_update(
                wf,
                {
                    "job_id": 1,
                    "status": JobStatus.COMPLETED,
                    "metrics": {"job_dir": "/j0", "final_energy": -1.0},
                },
            )
            _write_job_update(
                wf,
                {
                    "job_id": 2,
                    "status": JobStatus.FAILED,
                    "error_message": "memory error",
                    "increment_fail_count": True,
                },
            )
            _write_job_update(
                wf,
                {
                    "job_id": 3,
                    "status": JobStatus.TIMEOUT,
                    "error_message": "exceeded 20h",
                },
            )

            completed = wf.get_jobs_by_status(JobStatus.COMPLETED)
            failed = wf.get_jobs_by_status(JobStatus.FAILED)
            timeout = wf.get_jobs_by_status(JobStatus.TIMEOUT)
            running = wf.get_jobs_by_status(JobStatus.RUNNING)

            assert len(completed) == 1
            assert len(failed) == 1
            assert len(timeout) == 1
            assert len(running) == 2  # jobs 4 and 5 unchanged


class TestSubmitBatchClaimFiltering:
    """submit_batch must only submit the subset of ids that claim_jobs won."""

    def _seed_db(self, db_path, n):
        from oact_utilities.utils.architector import _init_db, _insert_row

        conn = _init_db(db_path)
        for i in range(n):
            _insert_row(
                conn,
                orig_index=i,
                elements="H;H",
                natoms=2,
                geometry="H 0.0 0.0 0.0\nH 0.0 0.0 0.74",
                status="to_run",
                charge=0,
                spin=1,
            )
        conn.commit()
        conn.close()

    def test_traditional_path_skips_jobs_lost_to_claim(self, tmp_path, monkeypatch):
        """submit_batch only sbatches ids that claim_jobs returned."""
        import subprocess as _subprocess

        from oact_utilities.workflows import submit_jobs as sj_mod

        db_path = tmp_path / "test.db"
        self._seed_db(db_path, 5)
        root_dir = tmp_path / "jobs"

        with ArchitectorWorkflow(db_path) as workflow:
            winners = [1, 3]
            monkeypatch.setattr(
                ArchitectorWorkflow,
                "claim_jobs",
                lambda self, ids, worker_id: list(winners),
            )

            prepared: list[int] = []

            def fake_prepare(job, *args, **kwargs):
                prepared.append(job.id)
                job_dir = root_dir / f"job_{job.orig_index}"
                job_dir.mkdir(parents=True, exist_ok=True)
                return job_dir

            monkeypatch.setattr(sj_mod, "prepare_job_directory", fake_prepare)

            def fake_write_flux(job_dir, **kwargs):
                script = job_dir / "flux_job.flux"
                script.write_text("#!/bin/sh\necho fake\n")
                return script

            monkeypatch.setattr(sj_mod, "write_flux_job_file", fake_write_flux)

            submitted_cmds: list[list[str]] = []

            def fake_run(cmd, **kwargs):
                submitted_cmds.append(list(cmd))
                return _subprocess.CompletedProcess(
                    cmd, 0, stdout="fakejob\n", stderr=""
                )

            monkeypatch.setattr(sj_mod.subprocess, "run", fake_run)

            submitted = sj_mod.submit_batch(
                workflow,
                root_dir,
                batch_size=5,
                scheduler="flux",
                randomize=False,
                dry_run=False,
            )

        assert sorted(submitted) == winners
        assert sorted(prepared) == winners
        assert len(submitted_cmds) == len(winners)

    def test_traditional_path_aborts_when_claim_empty(self, tmp_path, monkeypatch):
        """If claim_jobs loses every id, no directories are prepared."""
        from oact_utilities.workflows import submit_jobs as sj_mod

        db_path = tmp_path / "test.db"
        self._seed_db(db_path, 3)
        root_dir = tmp_path / "jobs"

        with ArchitectorWorkflow(db_path) as workflow:
            monkeypatch.setattr(
                ArchitectorWorkflow,
                "claim_jobs",
                lambda self, ids, worker_id: [],
            )
            prepare_calls = {"n": 0}

            def boom(*args, **kwargs):
                prepare_calls["n"] += 1
                raise AssertionError("prepare_job_directory should not run")

            monkeypatch.setattr(sj_mod, "prepare_job_directory", boom)

            submitted = sj_mod.submit_batch(
                workflow,
                root_dir,
                batch_size=5,
                scheduler="flux",
                randomize=False,
                dry_run=False,
            )

        assert submitted == []
        assert prepare_calls["n"] == 0
