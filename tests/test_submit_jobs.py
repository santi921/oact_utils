"""Tests for submit_jobs module."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from oact_utilities.workflows.architector_workflow import ArchitectorWorkflow, JobStatus
from oact_utilities.workflows.submit_jobs import (
    DEFAULT_ORCA_CONFIG,
    DEFAULT_ORCA_PATHS,
    DRAC_DEFAULT_MODULE_LOAD,
    SANDIA_DEFAULT_ACCOUNT,
    SANDIA_DEFAULT_OPENMPI_MODULE,
    SANDIA_DEFAULT_PARTITION,
    SANDIA_DEFAULT_QOS,
    OrcaConfig,
    _build_parsl_drac_worker_init,
    _build_parsl_sandia_worker_init,
    _build_parsl_sandia_worker_init_multi_block,
    _resolve_scheduler_job_id,
    _write_job_update,
    _write_prefailure_marker,
    prepare_job_directory,
    write_flux_job_file,
    write_slurm_drac_job_file,
    write_slurm_job_file,
    write_slurm_sandia_job_file,
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


class TestWriteSlurmSandiaJobFile:
    """Tests for write_slurm_sandia_job_file (CTS1/TLCC2)."""

    # Sandia has no shared ORCA install; tests must always pass an explicit path.
    SANDIA_TEST_ORCA = "/opt/orca-test/orca_6_1_0_linux_x86-64_shared/orca"

    def test_creates_script_with_correct_name(self, tmp_path):
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        script = write_slurm_sandia_job_file(job_dir, orca_path=self.SANDIA_TEST_ORCA)

        assert script.exists()
        assert script.name == "slurm_job.sh"

    def test_script_is_executable(self, tmp_path):
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        script = write_slurm_sandia_job_file(job_dir, orca_path=self.SANDIA_TEST_ORCA)

        assert script.stat().st_mode & 0o755

    def test_uses_partition_not_constraint(self, tmp_path):
        """Sandia writer must use --partition (not --constraint=standard)."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        script = write_slurm_sandia_job_file(
            job_dir, partition="attaway", orca_path=self.SANDIA_TEST_ORCA
        )
        content = script.read_text()

        assert "#SBATCH --partition=attaway" in content
        assert "--constraint" not in content

    def test_uses_module_load_not_conda(self, tmp_path):
        """Sandia writer must module-load OpenMPI; conda activation is wrong here."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        script = write_slurm_sandia_job_file(job_dir, orca_path=self.SANDIA_TEST_ORCA)
        content = script.read_text()

        assert f"module load {SANDIA_DEFAULT_OPENMPI_MODULE}" in content
        assert "conda activate" not in content

    def test_sets_ompi_mca_env(self, tmp_path):
        """Sandia writer forces TCP/vader transport via OMPI_MCA env."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        script = write_slurm_sandia_job_file(job_dir, orca_path=self.SANDIA_TEST_ORCA)
        content = script.read_text()

        assert "export OMPI_MCA_pml=ob1" in content
        assert "export OMPI_MCA_mtl=^psm2" in content
        assert "export OMPI_MCA_btl=tcp,self,vader" in content

    def test_derives_mpi_root_at_runtime(self, tmp_path):
        """LD_LIBRARY_PATH is derived from $(which mpirun) after module load."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        script = write_slurm_sandia_job_file(job_dir, orca_path=self.SANDIA_TEST_ORCA)
        content = script.read_text()

        assert "export MPI_ROOT=$(dirname $(dirname $(which mpirun)))" in content
        assert "export LD_LIBRARY_PATH=${MPI_ROOT}/lib:$LD_LIBRARY_PATH" in content

    def test_default_ntasks_per_node_is_36(self, tmp_path):
        """CTS1 nodes have 36 cores; default reflects that."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        script = write_slurm_sandia_job_file(job_dir, orca_path=self.SANDIA_TEST_ORCA)
        content = script.read_text()

        assert "#SBATCH --ntasks-per-node=36" in content

    def test_custom_ntasks_per_node(self, tmp_path):
        """TLCC2 has 16 cores; n_cores override should propagate."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        script = write_slurm_sandia_job_file(
            job_dir, n_cores=16, orca_path=self.SANDIA_TEST_ORCA
        )
        content = script.read_text()

        assert "#SBATCH --ntasks-per-node=16" in content

    def test_account_and_qos_in_script(self, tmp_path):
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        script = write_slurm_sandia_job_file(
            job_dir,
            account="fy250086",
            qos="normal",
            orca_path=self.SANDIA_TEST_ORCA,
        )
        content = script.read_text()

        assert "#SBATCH --account=fy250086" in content
        assert "#SBATCH --qos=normal" in content

    def test_no_default_sandia_orca_path(self):
        """Sandia intentionally has no DEFAULT_ORCA_PATHS entry: it would
        hardcode one developer's homedir as every teammate's default."""
        assert "sandia" not in DEFAULT_ORCA_PATHS

    def test_requires_orca_path(self, tmp_path):
        """Omitting orca_path must raise; there is no Sandia-wide shared install."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        with pytest.raises(ValueError, match="orca_path"):
            write_slurm_sandia_job_file(job_dir)

    def test_orca_command_appears_with_input(self, tmp_path):
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        script = write_slurm_sandia_job_file(job_dir, orca_path=self.SANDIA_TEST_ORCA)
        content = script.read_text()

        assert f"{self.SANDIA_TEST_ORCA} orca.inp" in content

    def test_sella_optimizer_runs_python_shim(self, tmp_path):
        """When optimizer=sella, the script runs the Sella shim instead of ORCA."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        script = write_slurm_sandia_job_file(
            job_dir, optimizer="sella", orca_path=self.SANDIA_TEST_ORCA
        )
        content = script.read_text()

        assert "python run_sella.py" in content
        assert (
            " orca.inp" not in content.split("python run_sella.py")[0].split("\n")[-1]
        )

    def test_sandia_defaults_match_constants(self, tmp_path):
        """Defaults written into the script match the exported constants."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        script = write_slurm_sandia_job_file(job_dir, orca_path=self.SANDIA_TEST_ORCA)
        content = script.read_text()

        assert f"#SBATCH --partition={SANDIA_DEFAULT_PARTITION}" in content
        assert f"#SBATCH --qos={SANDIA_DEFAULT_QOS}" in content
        assert f"#SBATCH --account={SANDIA_DEFAULT_ACCOUNT}" in content


class TestWriteSlurmDracJobFile:
    """Tests for write_slurm_drac_job_file (Fir/Narval/Nibi/Rorqual/Trillium)."""

    def test_creates_executable_script(self, tmp_path):
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        script = write_slurm_drac_job_file(job_dir, account="def-yqw")

        assert script.exists()
        assert script.name == "slurm_job.sh"
        assert script.stat().st_mode & 0o755

    def test_omits_qos_partition_constraint(self, tmp_path):
        """DRAC auto-assigns the partition; emitting these breaks submission."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        content = write_slurm_drac_job_file(job_dir, account="def-yqw").read_text()

        assert "--qos" not in content
        assert "--partition" not in content
        assert "--constraint" not in content

    def test_no_ompi_mca_overrides(self, tmp_path):
        """DRAC InfiniBand uses OpenMPI defaults; Sandia's TCP/PSM2 env is wrong."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        content = write_slurm_drac_job_file(job_dir, account="def-yqw").read_text()

        assert "OMPI_MCA" not in content

    def test_module_load_and_module_orca_binary(self, tmp_path):
        """Loads the module chain and runs ORCA via $EBROOTORCA, not conda/srun."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        content = write_slurm_drac_job_file(job_dir, account="def-yqw").read_text()

        assert f"module load {DRAC_DEFAULT_MODULE_LOAD}" in content
        assert "$EBROOTORCA/orca orca.inp" in content
        assert "conda activate" not in content
        assert "srun" not in content
        assert "mpirun" not in content

    def test_account_and_ntasks_propagate(self, tmp_path):
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        content = write_slurm_drac_job_file(
            job_dir, account="def-yqw", n_cores=8
        ).read_text()

        assert "#SBATCH --account=def-yqw" in content
        assert "#SBATCH --ntasks-per-node=8" in content

    def test_no_venv_activation_by_default(self, tmp_path):
        """Plain ORCA runs need no Python env, so no venv is sourced."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        content = write_slurm_drac_job_file(job_dir, account="def-yqw").read_text()

        assert "bin/activate" not in content

    def test_venv_activation_when_given(self, tmp_path):
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        content = write_slurm_drac_job_file(
            job_dir, account="def-yqw", venv_path="/home/u/oact-env"
        ).read_text()

        assert "source /home/u/oact-env/bin/activate" in content

    def test_sella_requires_venv(self, tmp_path):
        """The Sella driver needs the venv; omitting venv_path must error."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        with pytest.raises(ValueError, match="venv_path"):
            write_slurm_drac_job_file(job_dir, account="def-yqw", optimizer="sella")

    def test_sella_runs_driver_with_venv(self, tmp_path):
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        content = write_slurm_drac_job_file(
            job_dir,
            account="def-yqw",
            optimizer="sella",
            venv_path="/home/u/oact-env",
        ).read_text()

        assert "python run_sella.py" in content
        assert "source /home/u/oact-env/bin/activate" in content

    def test_optional_mem_per_cpu(self, tmp_path):
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        with_mem = write_slurm_drac_job_file(
            job_dir, account="def-yqw", mem_per_cpu="3900M"
        ).read_text()
        assert "#SBATCH --mem-per-cpu=3900M" in with_mem

        job_dir2 = tmp_path / "job_2"
        job_dir2.mkdir()
        without_mem = write_slurm_drac_job_file(job_dir2, account="def-yqw").read_text()
        assert "--mem-per-cpu" not in without_mem


class TestBuildParslSandiaWorkerInit:
    """Tests for the minimal Sandia Parsl worker_init.

    Workers inherit module-loaded MPI / OMPI MCA env from the salloc'd
    coordinator shell, so the worker_init must NOT re-do module load (that
    would risk Lmod-not-defined errors and LD_LIBRARY_PATH duplication).
    """

    def test_no_module_load(self):
        out = _build_parsl_sandia_worker_init(orca_path=None)
        assert "module load" not in out

    def test_no_conda_activate(self):
        out = _build_parsl_sandia_worker_init(orca_path=None)
        assert "conda activate" not in out

    def test_no_ld_library_path_munging(self):
        """Worker must not stomp the inherited LD_LIBRARY_PATH."""
        out = _build_parsl_sandia_worker_init(orca_path=None)
        assert "LD_LIBRARY_PATH" not in out

    def test_no_ompi_mca_overrides(self):
        """OMPI MCA settings come from the salloc shell, not worker_init."""
        out = _build_parsl_sandia_worker_init(orca_path=None)
        assert "OMPI_MCA" not in out

    def test_includes_jax_and_omp_defaults(self):
        out = _build_parsl_sandia_worker_init(orca_path=None)
        assert "JAX_PLATFORMS=cpu" in out
        assert "OMP_NUM_THREADS=1" in out

    def test_orca_bin_dir_prepended_to_path(self):
        out = _build_parsl_sandia_worker_init(
            orca_path="/home/svargas/orca_6_1_0_linux_x86-64_shared_openmpi418/orca"
        )
        assert (
            "export PATH=/home/svargas/orca_6_1_0_linux_x86-64_shared_openmpi418:$PATH"
            in out
        )

    def test_no_path_export_when_orca_path_relative(self):
        out = _build_parsl_sandia_worker_init(orca_path="orca")
        assert "export PATH=" not in out


class TestBuildParslDracWorkerInit:
    """Tests for the minimal DRAC Parsl worker_init.

    Like Sandia, workers inherit the module-loaded ORCA chain and activated venv
    from the coordinator's allocation shell, so the worker_init must not re-do
    module load or conda/venv activation.
    """

    def test_no_module_load_or_activate(self):
        out = _build_parsl_drac_worker_init(orca_path=None)
        assert "module load" not in out
        assert "conda activate" not in out
        assert "bin/activate" not in out

    def test_no_ld_or_mca_munging(self):
        out = _build_parsl_drac_worker_init(orca_path=None)
        assert "LD_LIBRARY_PATH" not in out
        assert "OMPI_MCA" not in out

    def test_includes_jax_and_omp_defaults(self):
        out = _build_parsl_drac_worker_init(orca_path=None)
        assert "JAX_PLATFORMS=cpu" in out
        assert "OMP_NUM_THREADS=1" in out

    def test_orca_bin_dir_prepended_to_path(self):
        out = _build_parsl_drac_worker_init(
            orca_path="/cvmfs/restricted/easybuild/orca/6.1.0/orca"
        )
        assert "export PATH=/cvmfs/restricted/easybuild/orca/6.1.0:$PATH" in out


@pytest.mark.skipif(not PARSL_INSTALLED, reason="parsl not installed")
class TestBuildParslConfigDracLocal:
    """Tests for build_parsl_config_drac_local (DRAC single-node LocalProvider)."""

    def test_uses_local_provider(self):
        from parsl.providers import LocalProvider

        from oact_utilities.workflows.submit_jobs import build_parsl_config_drac_local

        config = build_parsl_config_drac_local(orca_path="/cvmfs/x/orca/6.1.0/orca")
        assert isinstance(config.executors[0].provider, LocalProvider)

    def test_worker_and_core_counts_propagate(self):
        from oact_utilities.workflows.submit_jobs import build_parsl_config_drac_local

        config = build_parsl_config_drac_local(
            max_workers=12,
            cores_per_worker=16,
            orca_path="/cvmfs/x/orca/6.1.0/orca",
        )
        ex = config.executors[0]
        assert ex.label == "drac_local_htex"
        assert ex.cores_per_worker == 16
        assert ex.max_workers_per_node == 12

    def test_worker_init_has_no_module_load_or_mca(self):
        from oact_utilities.workflows.submit_jobs import build_parsl_config_drac_local

        config = build_parsl_config_drac_local(orca_path="/cvmfs/x/orca/6.1.0/orca")
        wi = config.executors[0].provider.worker_init
        assert "module load" not in wi
        assert "OMPI_MCA" not in wi
        assert "/cvmfs/x/orca/6.1.0" in wi  # orca bin dir on PATH


class TestBuildParslSandiaWorkerInitMultiBlock:
    """Tests for the Sandia Parsl SlurmProvider worker_init.

    Each Parsl block is a fresh sbatch allocation with no inherited modules,
    so the worker_init must do the full bootstrap itself: module load,
    MPI_ROOT derivation, LD_LIBRARY_PATH, OMPI MCA settings.
    """

    def test_module_load_present(self):
        out = _build_parsl_sandia_worker_init_multi_block(
            openmpi_module="aue/openmpi/4.1.6-gcc-12.3.0"
        )
        assert "module load aue/openmpi/4.1.6-gcc-12.3.0\n" in out

    def test_mpi_root_derivation(self):
        out = _build_parsl_sandia_worker_init_multi_block()
        assert "export MPI_ROOT=$(dirname $(dirname $(which mpirun)))\n" in out

    def test_default_ld_library_path(self):
        out = _build_parsl_sandia_worker_init_multi_block()
        assert "export LD_LIBRARY_PATH=${MPI_ROOT}/lib:$LD_LIBRARY_PATH\n" in out

    def test_ld_library_path_override(self):
        out = _build_parsl_sandia_worker_init_multi_block(ld_library_path="/foo/lib")
        assert "export LD_LIBRARY_PATH=/foo/lib:$LD_LIBRARY_PATH\n" in out
        assert "${MPI_ROOT}/lib" not in out

    def test_ompi_mca_exports(self):
        out = _build_parsl_sandia_worker_init_multi_block()
        assert "export OMPI_MCA_pml=ob1\n" in out
        assert "export OMPI_MCA_mtl=^psm2\n" in out
        assert "export OMPI_MCA_btl=tcp,self,vader\n" in out

    def test_module_load_before_mpi_root(self):
        """which mpirun only resolves after module load -- ordering matters."""
        out = _build_parsl_sandia_worker_init_multi_block()
        assert out.index("module load") < out.index("MPI_ROOT")

    def test_orca_bin_dir_prepended_to_path(self):
        out = _build_parsl_sandia_worker_init_multi_block(
            orca_path="/home/svargas/orca_6_1_0_linux_x86-64_shared_openmpi418/orca"
        )
        assert (
            "export PATH=/home/svargas/orca_6_1_0_linux_x86-64_shared_openmpi418:$PATH"
            in out
        )

    def test_no_path_export_when_orca_path_none(self):
        out = _build_parsl_sandia_worker_init_multi_block(orca_path=None)
        assert "export PATH=" not in out

    def test_no_path_export_when_orca_path_relative(self):
        out = _build_parsl_sandia_worker_init_multi_block(orca_path="orca")
        assert "export PATH=" not in out

    def test_jax_and_omp_defaults(self):
        out = _build_parsl_sandia_worker_init_multi_block()
        assert "export JAX_PLATFORMS=cpu\n" in out
        assert "export OMP_NUM_THREADS=1\n" in out


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


class TestSubmitBatchSandiaDeprecation:
    """Per-job sbatch path on Sandia must surface a FutureWarning."""

    def test_warning_fires_once_per_batch(self, monkeypatch):
        from oact_utilities.workflows import submit_jobs as mod

        mock_jobs = [MagicMock(id=i) for i in (1, 2, 3)]
        for job in mock_jobs:
            job.fail_count = 0

        monkeypatch.setattr(
            mod, "filter_jobs_for_submission", lambda *a, **k: list(mock_jobs)
        )
        monkeypatch.setattr(mod, "_filter_marker_jobs", lambda jobs, *a, **k: jobs)
        # Halt before per-job loop work so we only exercise the warn site.
        monkeypatch.setattr(
            mod,
            "prepare_job_directory",
            lambda *a, **k: (_ for _ in ()).throw(StopIteration()),
        )

        wf = MagicMock()

        with pytest.warns(FutureWarning, match="Per-job sbatch on Sandia"):
            mod.submit_batch(
                workflow=wf,
                root_dir="/tmp/oact_test_warn",
                batch_size=3,
                scheduler="slurm",
                site="sandia",
                orca_config={"orca_path": "/opt/orca-test/orca"},
                dry_run=True,
            )

    def test_warning_does_not_fire_for_non_sandia(self, monkeypatch):
        import warnings as warnings_mod

        from oact_utilities.workflows import submit_jobs as mod

        mock_jobs = [MagicMock(id=1, fail_count=0)]
        monkeypatch.setattr(
            mod, "filter_jobs_for_submission", lambda *a, **k: list(mock_jobs)
        )
        monkeypatch.setattr(mod, "_filter_marker_jobs", lambda jobs, *a, **k: jobs)
        monkeypatch.setattr(
            mod,
            "prepare_job_directory",
            lambda *a, **k: (_ for _ in ()).throw(StopIteration()),
        )

        wf = MagicMock()

        with warnings_mod.catch_warnings(record=True) as captured:
            warnings_mod.simplefilter("always")
            try:
                mod.submit_batch(
                    workflow=wf,
                    root_dir="/tmp/oact_test_warn",
                    batch_size=1,
                    scheduler="flux",
                    site="default",
                    dry_run=True,
                )
            except StopIteration:
                pass
        assert not any(
            issubclass(w.category, FutureWarning)
            and "Per-job sbatch on Sandia" in str(w.message)
            for w in captured
        )


@pytest.mark.skipif(not PARSL_INSTALLED, reason="parsl not installed")
class TestBuildParslConfigSandia:
    """Tests for build_parsl_config_sandia (Sandia SLURM multi-block path)."""

    def test_single_node_uses_simple_launcher(self):
        from parsl.launchers import SimpleLauncher

        from oact_utilities.workflows.submit_jobs import build_parsl_config_sandia

        config = build_parsl_config_sandia(nodes_per_block=1)
        provider = config.executors[0].provider
        assert isinstance(provider.launcher, SimpleLauncher)

    def test_multi_node_uses_srun_launcher(self):
        from parsl.launchers import SrunLauncher

        from oact_utilities.workflows.submit_jobs import build_parsl_config_sandia

        config = build_parsl_config_sandia(nodes_per_block=4)
        provider = config.executors[0].provider
        assert isinstance(provider.launcher, SrunLauncher)

    def test_partition_always_in_scheduler_options(self):
        from oact_utilities.workflows.submit_jobs import build_parsl_config_sandia

        for nodes in (1, 4):
            config = build_parsl_config_sandia(
                nodes_per_block=nodes, partition="attaway"
            )
            provider = config.executors[0].provider
            assert "#SBATCH --partition=attaway" in provider.scheduler_options

    def test_single_node_emits_ntasks_default(self):
        from oact_utilities.workflows.submit_jobs import build_parsl_config_sandia

        config = build_parsl_config_sandia(
            max_workers=3, cores_per_worker=12, nodes_per_block=1
        )
        provider = config.executors[0].provider
        assert "--ntasks-per-node=36" in provider.scheduler_options
        assert "--cpus-per-task=1" in provider.scheduler_options

    def test_multi_node_omits_ntasks_unless_override(self):
        from oact_utilities.workflows.submit_jobs import build_parsl_config_sandia

        config = build_parsl_config_sandia(nodes_per_block=4)
        provider = config.executors[0].provider
        assert "--ntasks-per-node" not in provider.scheduler_options

    def test_multi_node_includes_ntasks_when_cpus_per_node_override(self):
        from oact_utilities.workflows.submit_jobs import build_parsl_config_sandia

        config = build_parsl_config_sandia(
            max_workers=3,
            cores_per_worker=12,
            cpus_per_node=72,
            nodes_per_block=4,
        )
        provider = config.executors[0].provider
        assert "--ntasks-per-node=72" in provider.scheduler_options
        assert "--cpus-per-task=1" in provider.scheduler_options

    def test_undersized_cpus_per_node_raises(self):
        from oact_utilities.workflows.submit_jobs import build_parsl_config_sandia

        with pytest.raises(ValueError, match="cpus_per_node must be >="):
            build_parsl_config_sandia(
                max_workers=3, cores_per_worker=12, cpus_per_node=12
            )

    def test_worker_init_contents(self):
        from oact_utilities.workflows.submit_jobs import build_parsl_config_sandia

        config = build_parsl_config_sandia(
            openmpi_module="aue/openmpi/4.1.6-gcc-12.3.0"
        )
        worker_init = config.executors[0].provider.worker_init
        assert "module load aue/openmpi/4.1.6-gcc-12.3.0" in worker_init
        assert "OMPI_MCA_pml=ob1" in worker_init
        assert "OMPI_MCA_mtl=^psm2" in worker_init
        assert "OMPI_MCA_btl=tcp,self,vader" in worker_init
        assert "MPI_ROOT=$(dirname $(dirname $(which mpirun)))" in worker_init
        assert "JAX_PLATFORMS=cpu" in worker_init
        assert "OMP_NUM_THREADS=1" in worker_init

    def test_walltime_format(self):
        """Both single- and double-digit hours must zero-pad correctly.

        Guards against someone simplifying ``{:02d}`` to ``{}``.
        """
        from oact_utilities.workflows.submit_jobs import build_parsl_config_sandia

        assert (
            build_parsl_config_sandia(walltime_hours=2).executors[0].provider.walltime
            == "02:00:00"
        )
        assert (
            build_parsl_config_sandia(walltime_hours=10).executors[0].provider.walltime
            == "10:00:00"
        )


@pytest.mark.skipif(not PARSL_INSTALLED, reason="parsl not installed")
class TestSubmitBatchParslSandiaRouting:
    """Tests that submit_batch_parsl dispatches to the right Sandia builder."""

    def _run(self, monkeypatch, *, nodes_per_block, max_blocks):
        """Drive submit_batch_parsl through dispatch.

        The routing decision happens before parsl.load. We stub parsl.load
        to raise so the function exits early via its existing try/except
        path -- it catches the RuntimeError, resets claimed jobs, and
        returns []. Either way the builder call has already been recorded.
        """
        import parsl

        from oact_utilities.workflows import submit_jobs as mod

        slurm_calls: list[dict] = []
        local_calls: list[dict] = []

        mock_job = MagicMock()
        mock_job.id = 1
        wf = MagicMock()

        monkeypatch.setattr(
            mod, "filter_jobs_for_submission", lambda *a, **k: [mock_job]
        )
        monkeypatch.setattr(mod, "_filter_marker_jobs", lambda jobs, *a, **k: jobs)
        monkeypatch.setattr(
            mod, "prepare_job_directory", lambda *a, **k: Path("/tmp/x")
        )
        monkeypatch.setattr(
            mod,
            "build_parsl_config_sandia",
            lambda **k: slurm_calls.append(k) or MagicMock(),
        )
        monkeypatch.setattr(
            mod,
            "build_parsl_config_sandia_local",
            lambda **k: local_calls.append(k) or MagicMock(),
        )
        monkeypatch.setattr(parsl, "clear", lambda: None)
        monkeypatch.setattr(
            parsl,
            "load",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop")),
        )

        mod.submit_batch_parsl(
            workflow=wf,
            root_dir="/tmp/oact_test_routing",
            num_jobs=1,
            nodes_per_block=nodes_per_block,
            max_blocks=max_blocks,
            site="sandia",
            orca_config={"orca_path": "/opt/orca-test/orca"},
        )
        return slurm_calls, local_calls

    def test_multi_node_routes_to_slurm_builder(self, monkeypatch):
        slurm_calls, local_calls = self._run(
            monkeypatch, nodes_per_block=2, max_blocks=1
        )
        assert len(slurm_calls) == 1
        assert len(local_calls) == 0

    def test_multi_block_routes_to_slurm_builder(self, monkeypatch):
        slurm_calls, local_calls = self._run(
            monkeypatch, nodes_per_block=1, max_blocks=4
        )
        assert len(slurm_calls) == 1
        assert len(local_calls) == 0

    def test_single_block_single_node_routes_to_local_builder(self, monkeypatch):
        slurm_calls, local_calls = self._run(
            monkeypatch, nodes_per_block=1, max_blocks=1
        )
        assert len(local_calls) == 1
        assert len(slurm_calls) == 0


# ---------------------------------------------------------------------------
# Inline cleanup hooks (--clean-on-complete / --purge-on-fail)
# ---------------------------------------------------------------------------


class TestInlineCleanupHooks:
    """Tests for the Parsl-mode inline cleanup hooks in submit_jobs."""

    @staticmethod
    def _make_completed_job_dir(root: Path, name: str) -> Path:
        """Create a job_dir with the file mix --clean-all targets."""
        job_dir = root / name
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "orca.out").write_text("Some output\nORCA TERMINATED NORMALLY\n")
        (job_dir / "orca.inp").write_text("! wB97M-V")
        (job_dir / "orca.engrad").write_text("0 1")
        (job_dir / "orca.tmp").write_text("scratch")
        (job_dir / "orca.tmp.7").write_text("scratch")
        (job_dir / "orca.bas").write_text("basis")
        (job_dir / "orca.bas0").write_text("basis")
        (job_dir / "orca_tmp_abc123").mkdir()
        (job_dir / "orca_tmp_abc123" / "guess.gbw").write_text("x")
        return job_dir

    def test_cleanup_completed_removes_scratch_keeps_critical(self, tmp_path):
        """--clean-on-complete strips tmp/bas/orca_tmp_* and preserves outputs."""
        from oact_utilities.workflows.submit_jobs import (
            _cleanup_completed_job_inline,
        )

        job_dir = self._make_completed_job_dir(tmp_path, "job_1")

        _cleanup_completed_job_inline(str(job_dir), tmp_path, optimizer=None)

        # Scratch + basis removed
        assert not (job_dir / "orca.tmp").exists()
        assert not (job_dir / "orca.tmp.7").exists()
        assert not (job_dir / "orca.bas").exists()
        assert not (job_dir / "orca.bas0").exists()
        assert not (job_dir / "orca_tmp_abc123").exists()
        # Critical outputs preserved
        assert (job_dir / "orca.out").exists()
        assert (job_dir / "orca.inp").exists()
        assert (job_dir / "orca.engrad").exists()

    def test_cleanup_completed_swallows_errors(self, tmp_path, capsys):
        """A cleanup failure must not raise; it must only print a warning."""
        from unittest.mock import patch

        from oact_utilities.workflows.submit_jobs import (
            _cleanup_completed_job_inline,
        )

        job_dir = self._make_completed_job_dir(tmp_path, "job_2")

        with patch(
            "oact_utilities.workflows.submit_jobs._process_job",
            side_effect=RuntimeError("boom"),
        ):
            _cleanup_completed_job_inline(str(job_dir), tmp_path, optimizer=None)

        assert "cleanup error" in capsys.readouterr().out

    def test_cleanup_skips_revalidation(self, tmp_path):
        """Inline cleanup passes skip_revalidation=True (Parsl already verified)."""
        from unittest.mock import patch

        from oact_utilities.workflows.submit_jobs import (
            _cleanup_completed_job_inline,
        )

        job_dir = self._make_completed_job_dir(tmp_path, "job_3")

        with patch(
            "oact_utilities.workflows.submit_jobs._process_job",
            return_value=([], 0, []),
        ) as mock_proc:
            _cleanup_completed_job_inline(str(job_dir), tmp_path, optimizer="sella")

        _, kwargs = mock_proc.call_args
        assert kwargs["skip_revalidation"] is True
        assert kwargs["categories"] == {"tmp", "bas"}
        assert kwargs["execute"] is True
        assert kwargs["optimizer"] == "sella"

    @staticmethod
    def _make_failed_db(tmp_path: Path, job_id: int) -> Path:
        import sqlite3

        db_path = tmp_path / "wf.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE structures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO structures (id, status) VALUES (?, ?)",
            (job_id, "failed"),
        )
        conn.commit()
        conn.close()
        return db_path

    def test_purge_failed_writes_marker_and_deletes(self, tmp_path):
        """--purge-on-fail writes .do_not_rerun.json and removes other files."""
        from unittest.mock import MagicMock

        from oact_utilities.workflows.clean import MARKER_FILENAME
        from oact_utilities.workflows.submit_jobs import (
            _purge_failed_job_inline,
        )

        job_id = 17
        db_path = self._make_failed_db(tmp_path, job_id)

        job_dir = tmp_path / "job_17"
        job_dir.mkdir()
        (job_dir / "orca.out").write_text("aborting the run\n")
        (job_dir / "orca.inp").write_text("! BP86")
        (job_dir / "orca.tmp").write_text("scratch")
        (job_dir / "orca_tmp_xyz").mkdir()
        (job_dir / "orca_tmp_xyz" / "guess.gbw").write_text("x")

        job_record = MagicMock()
        job_record.id = job_id
        job_record.orig_index = 99
        job_record.elements = "U;O;O"
        job_record.charge = 0
        job_record.spin = 1
        job_record.fail_count = 1  # pre-increment value

        _purge_failed_job_inline(
            str(job_dir),
            tmp_path,
            db_path,
            job_record,
            error_message="ORCA exited with code 1",
        )

        # Only marker remains
        remaining = {p.name for p in job_dir.iterdir()}
        assert remaining == {MARKER_FILENAME}, remaining

        # Marker content reflects the increment + the new error message
        marker = json.loads((job_dir / MARKER_FILENAME).read_text())
        assert marker["orig_index"] == 99
        assert marker["elements"] == "U;O;O"
        assert marker["charge"] == 0
        assert marker["spin"] == 1
        assert marker["fail_count"] == 2  # +1 over the pre-increment record
        assert marker["error_message"] == "ORCA exited with code 1"

    def test_purge_failed_skips_when_db_status_mismatch(self, tmp_path):
        """TOCTOU re-check: if DB says not failed, contents are preserved."""
        import sqlite3
        from unittest.mock import MagicMock

        from oact_utilities.workflows.submit_jobs import (
            _purge_failed_job_inline,
        )

        job_id = 5
        db_path = tmp_path / "wf.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE structures (id INTEGER PRIMARY KEY, status TEXT)")
        # status != "failed" -- TOCTOU should bail
        conn.execute(
            "INSERT INTO structures (id, status) VALUES (?, ?)",
            (job_id, "completed"),
        )
        conn.commit()
        conn.close()

        job_dir = tmp_path / "job_5"
        job_dir.mkdir()
        (job_dir / "orca.out").write_text("ORCA TERMINATED NORMALLY\n")
        (job_dir / "orca.tmp").write_text("scratch")

        job_record = MagicMock()
        job_record.id = job_id
        job_record.orig_index = 1
        job_record.elements = "U"
        job_record.charge = 0
        job_record.spin = 1
        job_record.fail_count = 0

        _purge_failed_job_inline(
            str(job_dir),
            tmp_path,
            db_path,
            job_record,
            error_message="should not be purged",
        )

        # Nothing was deleted, no marker written
        assert (job_dir / "orca.out").exists()
        assert (job_dir / "orca.tmp").exists()
        assert not (job_dir / ".do_not_rerun.json").exists()


class TestResolveSchedulerJobId:
    """Tests for the worker_id resolution helper (covers B1 review finding)."""

    def test_slurm_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.setenv("FLUX_JOB_ID", "fAAA")
        monkeypatch.setenv("PBS_JOBID", "99.server")
        assert _resolve_scheduler_job_id() == "12345"

    def test_flux_when_no_slurm(self, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.setenv("FLUX_JOB_ID", "fAAA")
        monkeypatch.setenv("PBS_JOBID", "99.server")
        assert _resolve_scheduler_job_id() == "fAAA"

    def test_pbs_jobid_picked_up_when_others_absent(self, monkeypatch):
        """PBS Pro orphan recovery depends on PBS_JOBID being captured here."""
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("FLUX_JOB_ID", raising=False)
        monkeypatch.setenv("PBS_JOBID", "12345.pbs-server.fqdn")
        assert _resolve_scheduler_job_id() == "12345.pbs-server.fqdn"

    def test_pid_sentinel_when_no_scheduler(self, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("FLUX_JOB_ID", raising=False)
        monkeypatch.delenv("PBS_JOBID", raising=False)
        result = _resolve_scheduler_job_id()
        assert result.startswith("pid_")
        assert result[4:].isdigit()


class TestPrefailureMarker:
    """Tests for the pre-commit marker write (covers B5 review finding)."""

    def test_marker_written_with_metadata(self, tmp_path):
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        job = MagicMock()
        job.orig_index = 42
        job.elements = "U;O;O"
        job.charge = 0
        job.spin = 3
        job.fail_count = 1

        _write_prefailure_marker(
            str(job_dir), job, error_message="SCF did not converge"
        )

        marker_path = job_dir / ".do_not_rerun.json"
        assert marker_path.exists()
        data = json.loads(marker_path.read_text())
        assert data["orig_index"] == 42
        assert data["elements"] == "U;O;O"
        assert data["fail_count"] == 2  # pre-increment + 1 to match SQL-side
        assert data["error_message"] == "SCF did not converge"

    def test_marker_fail_count_handles_none(self, tmp_path):
        """job.fail_count is NULL on first failure; (None or 0) + 1 == 1."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        job = MagicMock()
        job.orig_index = 1
        job.elements = "H"
        job.charge = 0
        job.spin = 1
        job.fail_count = None

        _write_prefailure_marker(str(job_dir), job, error_message=None)

        data = json.loads((job_dir / ".do_not_rerun.json").read_text())
        assert data["fail_count"] == 1

    def test_marker_write_failure_does_not_raise(self, tmp_path, capsys):
        """Marker write must be best-effort: a missing job_dir cannot block
        the FAILED DB commit that follows."""
        nonexistent = tmp_path / "nope"  # parent dir does not exist
        job = MagicMock()
        job.orig_index = 1
        job.elements = "H"
        job.charge = 0
        job.spin = 1
        job.fail_count = 0

        # Should not raise.
        _write_prefailure_marker(str(nonexistent), job, error_message="x")
        captured = capsys.readouterr()
        assert "pre-failure marker write failed" in captured.out


class TestMaxAtomsFilter:
    """Tests for the --max-atoms submission size cap (filter_jobs_for_submission)."""

    def _make_db(self, tmp_path, natoms_values):
        """Helper: create a workflow DB with one TO_RUN job per atom count."""
        from oact_utilities.utils.architector import _init_db, _insert_row

        db_path = tmp_path / "test.db"
        conn = _init_db(db_path)
        for i, n in enumerate(natoms_values):
            _insert_row(
                conn,
                orig_index=i,
                elements=";".join(["H"] * n),
                natoms=n,
                geometry="H 0 0 0",
                status="to_run",
            )
        conn.commit()
        conn.close()
        return db_path

    def test_cap_excludes_larger_molecules(self, tmp_path):
        from oact_utilities.workflows.submit_jobs import filter_jobs_for_submission

        db_path = self._make_db(tmp_path, [2, 50, 100])
        with ArchitectorWorkflow(db_path) as wf:
            jobs = filter_jobs_for_submission(
                wf, num_jobs=10, max_atoms=50, randomize=False
            )
        assert sorted(j.natoms for j in jobs) == [2, 50]

    def test_no_cap_returns_all(self, tmp_path):
        from oact_utilities.workflows.submit_jobs import filter_jobs_for_submission

        db_path = self._make_db(tmp_path, [2, 50, 100])
        with ArchitectorWorkflow(db_path) as wf:
            jobs = filter_jobs_for_submission(
                wf, num_jobs=10, max_atoms=None, randomize=False
            )
        assert len(jobs) == 3

    def test_null_natoms_excluded(self, tmp_path, capsys):
        """A NULL natoms row is dropped rather than raising TypeError, and is
        reported separately from over-cap skips."""
        import sqlite3

        from oact_utilities.workflows.submit_jobs import filter_jobs_for_submission

        db_path = self._make_db(tmp_path, [2, 100])
        conn = sqlite3.connect(db_path)
        conn.execute("UPDATE structures SET natoms = NULL WHERE orig_index = 0")
        conn.commit()
        conn.close()

        with ArchitectorWorkflow(db_path) as wf:
            jobs = filter_jobs_for_submission(
                wf, num_jobs=10, max_atoms=50, randomize=False
            )
        # orig_index 0 (now NULL) and orig_index 1 (natoms 100) both excluded.
        assert jobs == []
        out = capsys.readouterr().out
        # NULL row is not misreported as "natoms > 50"; the two reasons are split.
        assert "Skipped 1 jobs with natoms > 50" in out
        assert "Skipped 1 jobs with missing (NULL) natoms" in out

    def test_over_cap_jobs_left_to_run(self, tmp_path):
        """Filtering does not mutate status: over-cap jobs stay TO_RUN for a
        later (longer wall-time) batch and are never claimed."""
        from oact_utilities.workflows.submit_jobs import filter_jobs_for_submission

        db_path = self._make_db(tmp_path, [2, 100])
        with ArchitectorWorkflow(db_path) as wf:
            submitted = filter_jobs_for_submission(
                wf, num_jobs=10, max_atoms=50, randomize=False
            )
            still_ready = wf.get_jobs_by_status(
                JobStatus.TO_RUN, include_geometry=False
            )
        assert [j.natoms for j in submitted] == [2]
        assert sorted(j.natoms for j in still_ready) == [2, 100]


class TestMaxAtomsCLI:
    """Tests that --max-atoms reaches the submission functions and is validated."""

    def _make_db(self, tmp_path):
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
        return db_path

    def test_forwarded_traditional(self, tmp_path, monkeypatch):
        import sys

        from oact_utilities.workflows import submit_jobs as mod

        db_path = self._make_db(tmp_path)
        captured = {}
        monkeypatch.setattr(mod, "submit_batch", lambda **k: captured.update(k) or [])
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "submit_jobs",
                str(db_path),
                str(tmp_path / "jobs"),
                "--scheduler",
                "flux",
                "--max-atoms",
                "40",
            ],
        )
        mod.main()
        assert captured["max_atoms"] == 40

    def test_forwarded_parsl(self, tmp_path, monkeypatch):
        import sys

        from oact_utilities.workflows import submit_jobs as mod

        db_path = self._make_db(tmp_path)
        captured = {}
        monkeypatch.setattr(
            mod, "submit_batch_parsl", lambda **k: captured.update(k) or []
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "submit_jobs",
                str(db_path),
                str(tmp_path / "jobs"),
                "--use-parsl",
                "--scheduler",
                "flux",
                "--max-atoms",
                "40",
            ],
        )
        mod.main()
        assert captured["max_atoms"] == 40

    def test_default_is_none(self, tmp_path, monkeypatch):
        import sys

        from oact_utilities.workflows import submit_jobs as mod

        db_path = self._make_db(tmp_path)
        captured = {}
        monkeypatch.setattr(mod, "submit_batch", lambda **k: captured.update(k) or [])
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "submit_jobs",
                str(db_path),
                str(tmp_path / "jobs"),
                "--scheduler",
                "flux",
            ],
        )
        mod.main()
        assert captured["max_atoms"] is None

    def test_rejects_non_positive(self, tmp_path, monkeypatch):
        import sys

        from oact_utilities.workflows import submit_jobs as mod

        db_path = self._make_db(tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "submit_jobs",
                str(db_path),
                str(tmp_path / "jobs"),
                "--max-atoms",
                "0",
            ],
        )
        with pytest.raises(SystemExit):
            mod.main()
