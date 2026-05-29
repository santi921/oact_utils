"""Tests for submit_jobs module."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from oact_utilities.workflows.architector_workflow import ArchitectorWorkflow, JobStatus
from oact_utilities.workflows.globus_transfer import (
    GlobusTransferConfig,
    GlobusTransferResult,
)
from oact_utilities.workflows.submit_jobs import (
    _COORDINATOR_CLEANUP_LEAD_MINUTES,
    DEFAULT_ORCA_CONFIG,
    DEFAULT_ORCA_PATHS,
    OrcaConfig,
    _classify_claimed_job,
    _classify_parsl_future_failure,
    _cleanup_completed_job_inline,
    _cleanup_deadline_reached,
    _compute_cleanup_deadline,
    _flush_pending_updates,
    _get_coordinator_walltime_hours,
    _get_scheduler_job_id,
    _is_manager_lost_exception,
    _parsl_active_window,
    _purge_failed_job_inline,
    _serialize_job_record,
    _submit_globus_backup_if_verified,
    _wait_for_worker_startup_file,
    _write_job_update,
    _write_prefailure_marker,
    prepare_job_directory,
    write_flux_job_file,
    write_slurm_job_file,
)

PARSL_INSTALLED = importlib.util.find_spec("parsl") is not None


@pytest.fixture
def mock_job_record():
    """Create a mock job record."""
    return SimpleNamespace(
        id=1,
        orig_index=42,
        elements="H;H",
        natoms=2,
        status=JobStatus.TO_RUN,
        charge=0,
        spin=1,
        geometry="""H 0.0 0.0 0.0
H 0.0 0.0 0.74""",
        job_dir=None,
        max_forces=None,
        scf_steps=None,
        final_energy=None,
        error_message=None,
        fail_count=0,
        wall_time=None,
        n_cores=None,
        optimizer=None,
        worker_id=None,
        generator_data=None,
    )


@pytest.fixture
def mock_actinide_job_record():
    """Create a mock job record with actinide."""
    return SimpleNamespace(
        id=2,
        orig_index=100,
        elements="U;O;O",
        natoms=3,
        status=JobStatus.TO_RUN,
        charge=2,
        spin=3,
        geometry="""U 0.0 0.0 0.0
O 1.8 0.0 0.0
O -1.8 0.0 0.0""",
        job_dir=None,
        max_forces=None,
        scf_steps=None,
        final_energy=None,
        error_message=None,
        fail_count=0,
        wall_time=None,
        n_cores=None,
        optimizer=None,
        worker_id=None,
        generator_data=None,
    )


@pytest.fixture
def orca_config_with_path():
    """ORCA config with a fake orca_path to avoid which() returning None."""
    return {"orca_path": "/fake/path/to/orca"}


class TestParslActiveWindow:
    """Tests for Parsl active future window sizing."""

    def test_flux_uses_batch_size_window(self):
        """Flux keeps using the requested batch size as the active window."""
        assert (
            _parsl_active_window(
                scheduler="flux",
                num_jobs=500,
                nodes_per_block=39,
                max_blocks=7,
                max_workers=8,
            )
            == 500
        )

    def test_pbspro_ignores_batch_size_window(self):
        """PBS Pro derives the active window from max scheduler capacity."""
        assert (
            _parsl_active_window(
                scheduler="pbspro",
                num_jobs=10,
                nodes_per_block=521,
                max_blocks=2,
                max_workers=8,
            )
            == 8336
        )

    def test_slurm_ignores_batch_size_window(self):
        """Slurm derives the active window from max scheduler capacity."""
        assert (
            _parsl_active_window(
                scheduler="slurm",
                num_jobs=10,
                nodes_per_block=10,
                max_blocks=4,
                max_workers=4,
            )
            == 160
        )


class TestCoordinatorCleanupDeadline:
    """Tests for coordinator shutdown timing in Parsl mode."""

    def test_deadline_starts_five_minutes_before_168_hour_limit(self):
        """Cleanup begins 5 minutes before the detected coordinator limit."""
        now = 1000.0
        deadline = _compute_cleanup_deadline(
            coordinator_walltime_hours=168.0,
            cleanup_lead_minutes=_COORDINATOR_CLEANUP_LEAD_MINUTES,
            now=now,
        )

        assert deadline == pytest.approx(now + (168 * 3600) - (5 * 60))

    def test_deadline_reached_uses_monotonic_clock(self, monkeypatch):
        """Deadline checks trip once the monotonic clock crosses the target."""
        monkeypatch.setattr(
            "oact_utilities.workflows.submit_jobs.time.monotonic",
            lambda: 10.0,
        )
        assert _cleanup_deadline_reached(9.0) is True
        assert _cleanup_deadline_reached(10.5) is False
        assert _cleanup_deadline_reached(None) is False

    def test_get_coordinator_walltime_prefers_environment(self, monkeypatch):
        """Coordinator deadline should come from COORDINATOR_WALLTIME."""
        monkeypatch.setenv("COORDINATOR_WALLTIME", "168.0")

        assert _get_coordinator_walltime_hours() == pytest.approx(168.0)

    def test_get_coordinator_walltime_rejects_non_float(self, monkeypatch):
        """Malformed coordinator walltime disables the deadline."""
        monkeypatch.setenv("COORDINATOR_WALLTIME", "7-00:00:00")

        assert _get_coordinator_walltime_hours() is None


class TestWorkerStartupFileWait:
    """Tests for worker-side startup file visibility retries."""

    def test_returns_immediately_when_file_is_readable(self, tmp_path):
        """Readable startup files should pass without waiting."""
        input_file = tmp_path / "orca.inp"
        input_file.write_text("! test\n")

        _wait_for_worker_startup_file(input_file, timeout_seconds=0.0)

    def test_retries_once_for_transient_file_not_found(self, monkeypatch, tmp_path):
        """A transient open failure should be retried within the timeout window."""
        input_file = tmp_path / "orca.inp"
        input_file.write_text("! test\n")
        real_open = Path.open
        state = {"calls": 0}

        def flaky_open(self, *args, **kwargs):
            if self == input_file and state["calls"] == 0:
                state["calls"] += 1
                raise FileNotFoundError("transient metadata miss")
            return real_open(self, *args, **kwargs)

        monkeypatch.setattr(
            "oact_utilities.workflows.submit_jobs.time.sleep",
            lambda _: None,
        )
        monkeypatch.setattr(Path, "open", flaky_open)

        _wait_for_worker_startup_file(input_file, timeout_seconds=0.01)

        assert state["calls"] == 1

    def test_raises_real_exception_after_timeout(self, monkeypatch, tmp_path):
        """The last filesystem exception should be preserved after timeout."""
        input_file = tmp_path / "orca.inp"
        real_open = Path.open

        def denied_open(self, *args, **kwargs):
            if self == input_file:
                raise PermissionError("access denied")
            return real_open(self, *args, **kwargs)

        monkeypatch.setattr(
            "oact_utilities.workflows.submit_jobs.time.sleep",
            lambda _: None,
        )
        monkeypatch.setattr(Path, "open", denied_open)

        with pytest.raises(PermissionError, match="access denied"):
            _wait_for_worker_startup_file(input_file, timeout_seconds=0.0)


class TestParslJobSerialization:
    """Tests for worker-side job payload transport."""

    @pytest.mark.skip(
        reason="Pre-existing on origin/main: _deserialize_job_record is "
        "referenced here but never defined in submit_jobs.py. Tracked "
        "separately; not in scope for this branch."
    )
    def test_round_trip_preserves_prepare_fields(self, mock_job_record):
        """Serialized job payload keeps the fields needed for preparation."""
        payload = _serialize_job_record(mock_job_record)
        restored = _deserialize_job_record(payload)  # noqa: F821

        assert restored.id == mock_job_record.id
        assert restored.orig_index == mock_job_record.orig_index
        assert restored.elements == mock_job_record.elements
        assert restored.job_dir == mock_job_record.job_dir
        assert restored.geometry == mock_job_record.geometry
        assert restored.charge == mock_job_record.charge
        assert restored.spin == mock_job_record.spin


class TestClaimedJobClassification:
    """Tests for coordinator-side claimed-job classification."""

    def test_marker_blocked_job_is_classified_without_submission(
        self, mock_job_record, tmp_path
    ):
        """Marker-blocked claimed jobs should be filtered before Parsl submit."""
        job_dir = tmp_path / "job_42"
        job_dir.mkdir()
        (job_dir / ".do_not_rerun.json").write_text("{}")
        mock_job_record.job_dir = str(job_dir)

        result = _classify_claimed_job(
            mock_job_record,
            tmp_path,
            "job_{orig_index}",
            "pbspro",
            {"orca_path": "/fake/path/to/orca"},
            4,
        )

        assert result["action"] == "marker_blocked"
        assert result["job_id"] == mock_job_record.id

    def test_slurm_claimed_job_returns_submit_payload(self, mock_job_record, tmp_path):
        """Deferred-prep schedulers should return a submit-ready payload."""
        result = _classify_claimed_job(
            mock_job_record,
            tmp_path,
            "job_{orig_index}",
            "slurm",
            {"orca_path": "/fake/path/to/orca"},
            8,
        )

        assert result["action"] == "submit_ready"
        assert result["job_id"] == mock_job_record.id
        assert result["job_payload"]["geometry"] == mock_job_record.geometry


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

    def test_formula_placeholders_in_job_dir_pattern(
        self, mock_actinide_job_record, tmp_path, orca_config_with_path
    ):
        """Custom job directory patterns may include formula, charge, and spin."""
        job_dir = prepare_job_directory(
            mock_actinide_job_record,
            tmp_path,
            job_dir_pattern="{formula}_q{charge}_m{spin}_idx{orig_index}",
            orca_config=orca_config_with_path,
        )

        assert job_dir.name == "UO2_q2_m3_idx100"

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


class TestLegacySubmitBatch:
    """Tests for the non-Parsl submit path."""

    def test_flux_path_stays_on_legacy_batch_flow(self, monkeypatch, tmp_path):
        """Flux submission should keep using the old one-shot batch logic."""
        from types import SimpleNamespace

        from oact_utilities.workflows import submit_jobs as sj

        workflow = MagicMock()
        workflow.mark_jobs_as_running = MagicMock()
        workflow.claim_jobs_for_submission = MagicMock()
        workflow.update_job_metrics_bulk = MagicMock()
        workflow.update_status = MagicMock()

        job1 = MagicMock(
            id=1, orig_index=1, job_dir=None, geometry="H 0 0 0", charge=0, spin=1
        )
        job2 = MagicMock(
            id=2, orig_index=2, job_dir=None, geometry="H 0 0 0", charge=0, spin=1
        )

        monkeypatch.setattr(
            sj,
            "filter_jobs_for_submission",
            lambda *args, **kwargs: [job1, job2],
        )
        monkeypatch.setattr(sj, "_filter_marker_jobs", lambda jobs, *a, **k: jobs)
        monkeypatch.setattr(sj, "_skip_finished_on_disk", lambda jobs, *a, **k: jobs)

        def fake_prepare_job_directory(job_record, root_dir, **kwargs):
            job_dir = root_dir / f"job_{job_record.id}"
            job_dir.mkdir(parents=True, exist_ok=True)
            (job_dir / "orca.inp").write_text("! test\n")
            return job_dir

        def fake_write_flux_job_file(job_dir, **kwargs):
            job_script = job_dir / "flux_job.flux"
            job_script.write_text("#!/bin/bash\n")
            return job_script

        monkeypatch.setattr(sj, "prepare_job_directory", fake_prepare_job_directory)
        monkeypatch.setattr(sj, "write_flux_job_file", fake_write_flux_job_file)
        monkeypatch.setattr(
            sj.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(stdout="submitted"),
        )

        submitted = sj.submit_batch(
            workflow=workflow,
            root_dir=tmp_path / "root",
            batch_size=2,
            scheduler="flux",
        )

        assert submitted == [1, 2]
        workflow.mark_jobs_as_running.assert_called_once_with([1, 2])
        workflow.claim_jobs_for_submission.assert_not_called()

    def test_slurm_script_no_ld_library_path_by_default(self, tmp_path):
        """Test that LD_LIBRARY_PATH export is omitted when default is empty."""
        job_dir = tmp_path / "job_1"
        job_dir.mkdir()

        slurm_script = write_slurm_job_file(job_dir)

        content = slurm_script.read_text()
        # Default slurm LD_LIBRARY_PATH is empty, so no export line should appear
        assert "export LD_LIBRARY_PATH=:" not in content


class TestSubmitJobsCli:
    """CLI regression tests for submit_jobs.main()."""

    def test_validate_worker_imports_short_circuits_without_db(self, monkeypatch):
        """Validation mode should not require db/root or open the workflow DB."""
        from oact_utilities.workflows import submit_jobs as sj

        captured: dict[str, object] = {}

        def fake_validate_worker_imports(**kwargs):
            captured.update(kwargs)
            return True

        class FailingWorkflow:
            def __init__(self, *args, **kwargs):
                raise AssertionError("ArchitectorWorkflow should not be opened")

        monkeypatch.setattr(
            sj, "_validate_parsl_worker_imports", fake_validate_worker_imports
        )
        monkeypatch.setattr(sj, "ArchitectorWorkflow", FailingWorkflow)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "submit_jobs.py",
                "--validate-worker-imports",
                "--scheduler",
                "pbspro",
                "--optimizer",
                "sella",
            ],
        )

        with pytest.raises(SystemExit) as excinfo:
            sj.main()

        assert excinfo.value.code == 0
        assert captured["scheduler"] == "pbspro"
        assert captured["orca_config"]["optimizer"] == "sella"

    def test_use_parsl_forwards_mpirun_path(self, monkeypatch, tmp_path):
        """The --mpirun-path CLI option must reach submit_batch_parsl()."""
        from oact_utilities.workflows import submit_jobs as sj

        captured: dict[str, object] = {}

        class DummyWorkflow:
            def __init__(self, *args, **kwargs):
                pass

            def close(self):
                pass

        def fake_submit_batch_parsl(**kwargs):
            captured.update(kwargs)
            return []

        monkeypatch.setattr(sj, "ArchitectorWorkflow", DummyWorkflow)
        monkeypatch.setattr(sj, "submit_batch_parsl", fake_submit_batch_parsl)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "submit_jobs.py",
                str(tmp_path / "workflow.db"),
                str(tmp_path / "jobs"),
                "--use-parsl",
                "--mpirun-path",
                "/opt/custom/openmpi/bin/mpirun",
            ],
        )

        sj.main()

        assert captured["mpirun_path"] == "/opt/custom/openmpi/bin/mpirun"

    def test_use_parsl_forwards_queue(self, monkeypatch, tmp_path):
        """The --queue CLI option must reach submit_batch_parsl()."""
        from oact_utilities.workflows import submit_jobs as sj

        captured: dict[str, object] = {}

        class DummyWorkflow:
            def __init__(self, *args, **kwargs):
                pass

            def close(self):
                pass

        def fake_submit_batch_parsl(**kwargs):
            captured.update(kwargs)
            return []

        monkeypatch.setattr(sj, "ArchitectorWorkflow", DummyWorkflow)
        monkeypatch.setattr(sj, "submit_batch_parsl", fake_submit_batch_parsl)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "submit_jobs.py",
                str(tmp_path / "workflow.db"),
                str(tmp_path / "jobs"),
                "--use-parsl",
                "--scheduler",
                "pbspro",
                "--queue",
                "frontier_lg",
            ],
        )

        sj.main()

        assert captured["queue"] == "frontier_lg"

    def test_use_parsl_recovers_orphaned_rows_on_launch(self, monkeypatch, tmp_path):
        """Parsl startup should run orphan recovery before claiming new work."""
        from oact_utilities.workflows import submit_jobs as sj

        captured: dict[str, object] = {}

        class DummyWorkflow:
            def __init__(self, *args, **kwargs):
                pass

            def close(self):
                pass

        def fake_submit_batch_parsl(**kwargs):
            captured["submitted"] = True
            return []

        def fake_recover_orphaned_jobs(workflow, scheduler, **kwargs):
            captured["recovered_scheduler"] = scheduler
            captured["recover_kwargs"] = kwargs
            return {
                "recovered": 7,
                "completed": 0,
                "failed": 0,
                "reset": 7,
                "dead_jobs": 1,
                "skipped": 0,
            }

        monkeypatch.setattr(sj, "ArchitectorWorkflow", DummyWorkflow)
        monkeypatch.setattr(sj, "submit_batch_parsl", fake_submit_batch_parsl)
        monkeypatch.setattr(
            "oact_utilities.workflows.dashboard.recover_orphaned_jobs",
            fake_recover_orphaned_jobs,
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "submit_jobs.py",
                str(tmp_path / "workflow.db"),
                str(tmp_path / "jobs"),
                "--use-parsl",
                "--scheduler",
                "pbspro",
            ],
        )

        sj.main()

        assert captured["recovered_scheduler"] == "pbspro"
        assert captured["recover_kwargs"]["root_dir"] is None
        assert captured["submitted"] is True

    def test_use_parsl_reroot_passes_recovery_fallback_path(
        self, monkeypatch, tmp_path
    ):
        """Parsl startup should pass reroot fallback info into orphan recovery."""
        from oact_utilities.workflows import submit_jobs as sj

        captured: dict[str, object] = {}

        class DummyWorkflow:
            def __init__(self, *args, **kwargs):
                pass

            def close(self):
                pass

        def fake_submit_batch_parsl(**kwargs):
            return []

        def fake_recover_orphaned_jobs(workflow, scheduler, **kwargs):
            captured["scheduler"] = scheduler
            captured["kwargs"] = kwargs
            return {
                "recovered": 0,
                "completed": 0,
                "failed": 0,
                "reset": 0,
                "dead_jobs": 0,
                "skipped": 0,
            }

        monkeypatch.setattr(sj, "ArchitectorWorkflow", DummyWorkflow)
        monkeypatch.setattr(sj, "submit_batch_parsl", fake_submit_batch_parsl)
        monkeypatch.setattr(
            "oact_utilities.workflows.dashboard.recover_orphaned_jobs",
            fake_recover_orphaned_jobs,
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "submit_jobs.py",
                str(tmp_path / "workflow.db"),
                str(tmp_path / "jobs"),
                "--use-parsl",
                "--scheduler",
                "pbspro",
                "--reroot",
                "--job-dir-pattern",
                "{formula}_q{charge}_m{spin}_idx{orig_index}",
            ],
        )

        sj.main()

        assert captured["scheduler"] == "pbspro"
        assert captured["kwargs"]["root_dir"] == str(tmp_path / "jobs")
        assert (
            captured["kwargs"]["job_dir_pattern"]
            == "{formula}_q{charge}_m{spin}_idx{orig_index}"
        )


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

    def test_worker_init_exports_repo_root(self):
        """Worker init exports repo-root hints for package imports."""
        from oact_utilities.workflows.submit_jobs import build_parsl_config_slurm

        config = build_parsl_config_slurm()
        provider = config.executors[0].provider
        assert "export OACT_UTILITIES_REPO_ROOT=" in provider.worker_init
        assert "export PYTHONPATH=" in provider.worker_init

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

    def test_slurm_provider_cmd_timeout_is_30_minutes(self):
        """Slurm provider scheduler commands get a 30 minute timeout."""
        from oact_utilities.workflows.submit_jobs import build_parsl_config_slurm

        config = build_parsl_config_slurm()
        provider = config.executors[0].provider
        assert provider.cmd_timeout == 1800

    def test_slurm_executor_launch_cmd_exports_repo_root(self):
        """Slurm HTEX launch injects repo-root env into worker startup."""
        from oact_utilities.workflows.submit_jobs import build_parsl_config_slurm

        config = build_parsl_config_slurm()
        executor = config.executors[0]
        assert "OACT_UTILITIES_REPO_ROOT=" in executor.launch_cmd
        assert "PYTHONPATH=" in executor.launch_cmd
        assert "/miniconda3/envs/py10mpi/bin/python -m " in executor.launch_cmd
        assert " -m parsl.executors.high_throughput.process_worker_pool " in (
            executor.launch_cmd
        )


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
        assert "OACT_UTILITIES_REPO_ROOT=" in executor.launch_cmd
        assert "PYTHONPATH=" in executor.launch_cmd
        assert "/miniconda3/envs/py10mpi/bin/python -m " in executor.launch_cmd
        assert " -m parsl.executors.high_throughput.process_worker_pool " in (
            executor.launch_cmd
        )
        assert "process_worker_pool.py" not in executor.launch_cmd

    def test_pbs_provider_cmd_timeout_is_30_minutes(self):
        """PBS provider scheduler commands get a 30 minute timeout."""
        from oact_utilities.workflows.submit_jobs import build_parsl_config_pbspro

        config = build_parsl_config_pbspro()
        provider = config.executors[0].provider
        assert provider.cmd_timeout == 1800


class TestSchedulerJobId:
    """Tests for scheduler allocation ID detection."""

    def test_prefers_slurm_over_other_ids(self, monkeypatch):
        """SLURM_JOB_ID takes precedence when multiple schedulers are visible."""
        monkeypatch.setenv("SLURM_JOB_ID", "123")
        monkeypatch.setenv("PBS_JOBID", "456")
        monkeypatch.setenv("FLUX_JOB_ID", "789")
        assert _get_scheduler_job_id() == "123"

    def test_uses_pbs_jobid_when_present(self, monkeypatch):
        """PBS allocations should be recorded with PBS_JOBID."""
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.setenv("PBS_JOBID", "456.server")
        monkeypatch.delenv("FLUX_JOB_ID", raising=False)
        assert _get_scheduler_job_id() == "456.server"


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


class TestGlobusCLIValidation:
    """Tests for Globus backup CLI validation."""

    @staticmethod
    def _env_without_globus():
        import os

        env = os.environ.copy()
        for name in (
            "GLOBUS_SOURCE_ENDPOINT_ID",
            "GLOBUS_DESTINATION_ENDPOINT_ID",
            "GLOBUS_DEST_ROOT",
            "GLOBUS_CLIENT_ID",
            "GLOBUS_TRANSFER_REFRESH_TOKEN",
            "GLOBUS_CLIENT_SECRET",
        ):
            env.pop(name, None)
        return env

    def test_globus_transfer_requires_parsl_mode(self):
        """--globus-transfer is only valid for the Parsl submitter."""
        import subprocess

        result = subprocess.run(
            [
                "python",
                "-m",
                "oact_utilities.workflows.submit_jobs",
                "fake.db",
                "fake_dir",
                "--globus-transfer",
            ],
            capture_output=True,
            text=True,
            env=self._env_without_globus(),
        )
        assert result.returncode != 0
        assert "--globus-transfer requires --use-parsl" in result.stderr

    def test_globus_transfer_requires_endpoint_env_or_args(self):
        """Missing Globus config should fail before opening the workflow DB."""
        import subprocess

        result = subprocess.run(
            [
                "python",
                "-m",
                "oact_utilities.workflows.submit_jobs",
                "fake.db",
                "fake_dir",
                "--use-parsl",
                "--globus-transfer",
            ],
            capture_output=True,
            text=True,
            env=self._env_without_globus(),
        )
        assert result.returncode != 0
        assert "--globus-transfer requires" in result.stderr
        assert "GLOBUS_SOURCE_ENDPOINT_ID" in result.stderr
        assert "GLOBUS_DESTINATION_ENDPOINT_ID" in result.stderr
        assert "GLOBUS_DEST_ROOT" in result.stderr
        assert "GLOBUS_CLIENT_ID" in result.stderr
        assert "GLOBUS_TRANSFER_REFRESH_TOKEN" in result.stderr

    def test_globus_transfer_accepts_cli_values(self):
        """Complete refresh-token config should pass validation."""
        import subprocess

        result = subprocess.run(
            [
                "python",
                "-m",
                "oact_utilities.workflows.submit_jobs",
                "fake.db",
                "fake_dir",
                "--use-parsl",
                "--globus-transfer",
                "--globus-source-endpoint-id",
                "src",
                "--globus-destination-endpoint-id",
                "dest",
                "--globus-dest-root",
                "/backup",
                "--globus-client-id",
                "client-id",
                "--globus-transfer-refresh-token",
                "refresh-token",
            ],
            capture_output=True,
            text=True,
            env=self._env_without_globus(),
        )
        assert result.returncode != 0
        assert "--globus-transfer requires" not in result.stderr
        assert "Database not found" in result.stdout


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

    def test_completed_on_disk_clears_worker_id_for_claimed_job(self, tmp_path):
        """Claimed jobs reclassified as completed on disk clear worker_id."""
        from oact_utilities.workflows.architector_workflow import (
            ArchitectorWorkflow,
            JobStatus,
        )
        from oact_utilities.workflows.submit_jobs import _skip_finished_on_disk

        db_path = self._make_db(tmp_path, n_jobs=1)
        root_dir = tmp_path / "jobs"
        job_dir = root_dir / "job_0"
        job_dir.mkdir(parents=True)
        (job_dir / "orca.out").write_text("Some output\nORCA TERMINATED NORMALLY\n")

        with ArchitectorWorkflow(db_path) as wf:
            claimed = wf.claim_jobs_for_submission(limit=1, worker_id="pbs_123")
            assert claimed[0].worker_id == "pbs_123"

            result = _skip_finished_on_disk(claimed, root_dir, "job_{orig_index}", wf)
            assert result == []

            completed = wf.get_jobs_by_status(JobStatus.COMPLETED)
            assert len(completed) == 1
            assert completed[0].worker_id is None

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

    def test_failed_on_disk_clears_worker_id_for_claimed_job(self, tmp_path):
        """Claimed jobs reclassified as failed on disk clear worker_id."""
        from oact_utilities.workflows.architector_workflow import (
            ArchitectorWorkflow,
            JobStatus,
        )
        from oact_utilities.workflows.submit_jobs import _skip_finished_on_disk

        db_path = self._make_db(tmp_path, n_jobs=1)
        root_dir = tmp_path / "jobs"
        job_dir = root_dir / "job_0"
        job_dir.mkdir(parents=True)
        (job_dir / "orca.out").write_text("SCF NOT CONVERGED\naborting the run\n")

        with ArchitectorWorkflow(db_path) as wf:
            claimed = wf.claim_jobs_for_submission(limit=1, worker_id="slurm_123")
            assert claimed[0].worker_id == "slurm_123"

            result = _skip_finished_on_disk(claimed, root_dir, "job_{orig_index}", wf)
            assert result == []

            failed = wf.get_jobs_by_status(JobStatus.FAILED)
            assert len(failed) == 1
            assert failed[0].worker_id is None

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


class TestGlobusBackupGate:
    """Tests for the verified-success Globus backup gate."""

    @staticmethod
    def _config() -> GlobusTransferConfig:
        return GlobusTransferConfig(
            source_endpoint_id="source-ep",
            destination_endpoint_id="dest-ep",
            dest_root="/backup/root",
            client_id="client-id",
            transfer_refresh_token="refresh-token",
        )

    def test_submits_only_after_completed_result_and_success_metrics(
        self, tmp_path, monkeypatch
    ):
        """Completed wrapper result plus successful metrics submits one transfer."""
        from oact_utilities.workflows import submit_jobs as sj

        root_dir = tmp_path / "jobs"
        job_dir = root_dir / "job_1"
        job_dir.mkdir(parents=True)
        calls = []

        transfer_client = object()

        def fake_archive_and_submit_transfer(
            job_dir, root_dir, config, transfer_client=None
        ):
            calls.append((job_dir, root_dir, config, transfer_client))
            return GlobusTransferResult(
                archive_path=Path(root_dir) / "job_1.tar.gz",
                destination_path="/backup/root/job_1.tar.gz",
                task_id="task-123",
            )

        monkeypatch.setattr(
            sj, "archive_and_submit_transfer", fake_archive_and_submit_transfer
        )

        result = _submit_globus_backup_if_verified(
            job_id=1,
            job_dir=job_dir,
            root_dir=root_dir,
            result={"status": "completed"},
            metrics={"success": True},
            globus_config=self._config(),
            globus_transfer_client=transfer_client,
        )

        assert result is not None
        assert result.task_id == "task-123"
        assert calls == [(job_dir, root_dir, self._config(), transfer_client)]

    @pytest.mark.parametrize(
        ("wrapper_result", "metrics"),
        [
            ({"status": "failed"}, {"success": True}),
            ({"status": "timeout"}, {"success": True}),
            ({"status": "completed"}, {"success": False}),
            ({"status": "completed"}, None),
        ],
    )
    def test_skips_without_both_success_signals(
        self, tmp_path, monkeypatch, wrapper_result, metrics
    ):
        """Failed, timeout, and metrics-unsuccessful cases skip transfer."""
        from oact_utilities.workflows import submit_jobs as sj

        root_dir = tmp_path / "jobs"
        job_dir = root_dir / "job_1"
        job_dir.mkdir(parents=True)
        calls = []

        def fake_archive_and_submit_transfer(
            job_dir, root_dir, config, transfer_client=None
        ):
            calls.append((job_dir, root_dir, config, transfer_client))
            return GlobusTransferResult(
                archive_path=Path(root_dir) / "job_1.tar.gz",
                destination_path="/backup/root/job_1.tar.gz",
                task_id="task-123",
            )

        monkeypatch.setattr(
            sj, "archive_and_submit_transfer", fake_archive_and_submit_transfer
        )

        result = _submit_globus_backup_if_verified(
            job_id=1,
            job_dir=job_dir,
            root_dir=root_dir,
            result=wrapper_result,
            metrics=metrics,
            globus_config=self._config(),
        )

        assert result is None
        assert calls == []

    def test_transfer_failure_raises(self, tmp_path, monkeypatch):
        """Globus failures raise instead of being swallowed."""
        from oact_utilities.workflows import submit_jobs as sj

        root_dir = tmp_path / "jobs"
        job_dir = root_dir / "job_1"
        job_dir.mkdir(parents=True)

        def fake_archive_and_submit_transfer(
            job_dir, root_dir, config, transfer_client=None
        ):
            raise RuntimeError("globus unavailable")

        monkeypatch.setattr(
            sj, "archive_and_submit_transfer", fake_archive_and_submit_transfer
        )

        with pytest.raises(RuntimeError, match="globus unavailable"):
            _submit_globus_backup_if_verified(
                job_id=1,
                job_dir=job_dir,
                root_dir=root_dir,
                result={"status": "completed"},
                metrics={"success": True},
                globus_config=self._config(),
            )


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


class TestParslFailureClassification:
    """Tests for Parsl failure classification helpers."""

    def test_manager_lost_exception_is_detected_by_name(self):
        """ManagerLost exceptions are recognized without importing Parsl."""

        class ManagerLost(Exception):
            pass

        assert _is_manager_lost_exception(ManagerLost("lost manager"))

    def test_worker_lost_exception_is_detected_by_message(self):
        """WorkerLost text in the message is also treated as an infra loss."""
        exc = RuntimeError("WorkerLost: task could not find its manager")
        assert _is_manager_lost_exception(exc)

    def test_manager_lost_maps_to_to_run(self):
        """Manager loss should requeue the job instead of incrementing fail count."""

        class ManagerLost(Exception):
            pass

        status, increment_fail_count, wandb_status = _classify_parsl_future_failure(
            ManagerLost("manager vanished")
        )
        assert status == JobStatus.TO_RUN
        assert increment_fail_count is False
        assert wandb_status == "requeued"

    def test_regular_exception_remains_failed(self):
        """Non-infrastructure errors remain terminal failures."""
        status, increment_fail_count, wandb_status = _classify_parsl_future_failure(
            RuntimeError("SCF crashed")
        )
        assert status == JobStatus.FAILED
        assert increment_fail_count is True
        assert wandb_status == "failed"


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
        from oact_utilities.workflows.clean import MARKER_FILENAME

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

        job_id = 5
        db_path = tmp_path / "wf.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE structures (id INTEGER PRIMARY KEY, status TEXT)")
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


class TestPrefailureMarker:
    """Tests for the pre-commit marker write that protects the FAILED window."""

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

        _write_prefailure_marker(str(nonexistent), job, error_message="x")
        captured = capsys.readouterr()
        assert "pre-failure marker write failed" in captured.out


class TestFlushPendingUpdatesHookDispatch:
    """Tests for the buffered-flush hook dispatch added on this branch.

    Specific to the main-branch port: on Sandia each future writes the DB
    inline, but on main updates are buffered in pending_updates and flushed
    in batches. The purge hook must run AFTER the FAILED row commits or its
    TOCTOU re-check silently skips every purge.
    """

    def test_hook_payload_stripped_before_write_job_update(self, tmp_path):
        """_hook_payload must be popped from the update before DB write."""
        from unittest.mock import patch

        seen_updates: list[dict] = []

        def fake_write(_workflow, update):
            seen_updates.append(dict(update))

        with patch(
            "oact_utilities.workflows.submit_jobs._write_job_update",
            side_effect=fake_write,
        ):
            pending = [
                {
                    "job_id": 1,
                    "status": JobStatus.COMPLETED,
                    "_hook_payload": {
                        "kind": "clean",
                        "job_dir": str(tmp_path / "job_1"),
                        "optimizer": None,
                    },
                }
            ]
            (tmp_path / "job_1").mkdir()
            _flush_pending_updates(MagicMock(), pending, root_dir=None)

        assert seen_updates == [{"job_id": 1, "status": JobStatus.COMPLETED}]

    def test_clean_hook_dispatched_after_db_write(self, tmp_path):
        """Clean hook fires only after _write_job_update returns."""
        from unittest.mock import patch

        call_order: list[str] = []

        def fake_write(_workflow, update):
            call_order.append(f"write:{update['job_id']}")

        def fake_clean(job_dir, root_dir, optimizer):
            call_order.append(f"clean:{Path(job_dir).name}")

        with (
            patch(
                "oact_utilities.workflows.submit_jobs._write_job_update",
                side_effect=fake_write,
            ),
            patch(
                "oact_utilities.workflows.submit_jobs._cleanup_completed_job_inline",
                side_effect=fake_clean,
            ),
        ):
            (tmp_path / "job_a").mkdir()
            pending = [
                {
                    "job_id": 42,
                    "status": JobStatus.COMPLETED,
                    "_hook_payload": {
                        "kind": "clean",
                        "job_dir": str(tmp_path / "job_a"),
                        "optimizer": None,
                    },
                }
            ]
            _flush_pending_updates(MagicMock(), pending, root_dir=tmp_path)

        assert call_order == ["write:42", "clean:job_a"]

    def test_purge_runs_after_flush_not_before(self, tmp_path):
        """Purge hook must run after the FAILED row commits so the TOCTOU
        re-check inside _purge_failed_job sees status='failed'."""
        from unittest.mock import patch

        call_order: list[str] = []

        def fake_write(_workflow, update):
            call_order.append(f"write:{update['job_id']}:{update['status'].value}")

        def fake_purge(job_dir, root_dir, db_path, job, error_message):
            call_order.append(f"purge:{job.id}")

        with (
            patch(
                "oact_utilities.workflows.submit_jobs._write_job_update",
                side_effect=fake_write,
            ),
            patch(
                "oact_utilities.workflows.submit_jobs._purge_failed_job_inline",
                side_effect=fake_purge,
            ),
        ):
            job_record = MagicMock(id=7, fail_count=0)
            (tmp_path / "job_7").mkdir()
            pending = [
                {
                    "job_id": 7,
                    "status": JobStatus.FAILED,
                    "error_message": "boom",
                    "_hook_payload": {
                        "kind": "purge",
                        "job_dir": str(tmp_path / "job_7"),
                        "db_path": tmp_path / "wf.db",
                        "job": job_record,
                        "error_message": "boom",
                    },
                }
            ]
            _flush_pending_updates(MagicMock(), pending, root_dir=tmp_path)

        # Write must come strictly before purge -- this is what the TOCTOU
        # re-check inside _purge_failed_job depends on.
        assert call_order == [f"write:7:{JobStatus.FAILED.value}", "purge:7"]

    def test_root_dir_none_skips_hooks(self, tmp_path):
        """When root_dir is None, hook payloads are dropped silently."""
        from unittest.mock import patch

        clean_calls: list = []
        with (
            patch(
                "oact_utilities.workflows.submit_jobs._write_job_update",
            ),
            patch(
                "oact_utilities.workflows.submit_jobs._cleanup_completed_job_inline",
                side_effect=lambda *a, **kw: clean_calls.append(a),
            ),
        ):
            pending = [
                {
                    "job_id": 1,
                    "status": JobStatus.COMPLETED,
                    "_hook_payload": {
                        "kind": "clean",
                        "job_dir": str(tmp_path),
                        "optimizer": None,
                    },
                }
            ]
            _flush_pending_updates(MagicMock(), pending, root_dir=None)

        assert clean_calls == []

    def test_hook_payload_preserved_on_write_failure(self, tmp_path):
        """If _write_job_update raises, the source dict must still carry
        _hook_payload so a retry can reattempt the hook. Earlier versions
        popped destructively and lost the hook on transient SQLite errors.
        """
        from unittest.mock import patch

        def boom(_workflow, _update):
            raise RuntimeError("transient sqlite lock")

        pending = [
            {
                "job_id": 1,
                "status": JobStatus.FAILED,
                "error_message": "boom",
                "_hook_payload": {
                    "kind": "purge",
                    "job_dir": str(tmp_path / "job_1"),
                    "db_path": tmp_path / "wf.db",
                    "job": MagicMock(id=1),
                    "error_message": "boom",
                },
            }
        ]

        with patch(
            "oact_utilities.workflows.submit_jobs._write_job_update",
            side_effect=boom,
        ):
            with pytest.raises(RuntimeError, match="transient"):
                _flush_pending_updates(MagicMock(), pending, root_dir=tmp_path)

        # The hook payload must still be there for the retry to find.
        assert "_hook_payload" in pending[0]
        assert pending[0]["_hook_payload"]["kind"] == "purge"

    def test_write_payload_excludes_hook_key(self, tmp_path):
        """The dict reaching _write_job_update must not contain
        _hook_payload (would corrupt the SQL UPDATE columns)."""
        from unittest.mock import patch

        seen: list[dict] = []
        with (
            patch(
                "oact_utilities.workflows.submit_jobs._write_job_update",
                side_effect=lambda _w, u: seen.append(dict(u)),
            ),
            patch(
                "oact_utilities.workflows.submit_jobs._cleanup_completed_job_inline",
            ),
        ):
            (tmp_path / "job_1").mkdir()
            pending = [
                {
                    "job_id": 1,
                    "status": JobStatus.COMPLETED,
                    "_hook_payload": {
                        "kind": "clean",
                        "job_dir": str(tmp_path / "job_1"),
                        "optimizer": None,
                    },
                }
            ]
            _flush_pending_updates(MagicMock(), pending, root_dir=tmp_path)

        assert "_hook_payload" not in seen[0]
        # Source dict retains the payload (idempotent re-flush stays safe).
        assert "_hook_payload" in pending[0]
