"""Tests for qtaim_generator ORCA output parsing (parse_generator_data)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from oact_utilities.utils.analysis import (
    GENERATOR_AVAILABLE,
    _sanitize_for_json,
    parse_generator_data,
)

# no_traj/logs is a UO2 geometry optimization step with ALLPOP analysis
# (Mulliken, Loewdin, Mayer charges/bond orders, gradients).
# find_orca_output_file() only recognises orca.out/output.out, so the fixture
# symlinks logs -> orca.out inside tmp_path.
NO_TRAJ_LOGS = Path(__file__).parent / "files" / "no_traj" / "logs"

pytestmark = pytest.mark.skipif(
    not GENERATOR_AVAILABLE,
    reason="qtaim_generator not installed",
)


@pytest.fixture()
def job_dir(tmp_path: Path) -> Path:
    """Temp job directory with orca.out symlink pointing to the real fixture."""
    (tmp_path / "orca.out").symlink_to(NO_TRAJ_LOGS)
    return tmp_path


class TestSanitizeForJson:
    def test_integer(self):
        assert _sanitize_for_json(np.int32(42)) == 42
        assert isinstance(_sanitize_for_json(np.int32(42)), int)

    def test_float(self):
        val = _sanitize_for_json(np.float64(3.14))
        assert abs(val - 3.14) < 1e-6

    def test_nan_becomes_none(self):
        assert _sanitize_for_json(np.float64(float("nan"))) is None

    def test_inf_becomes_none(self):
        assert _sanitize_for_json(np.float64(float("inf"))) is None

    def test_plain_float_nan_becomes_none(self):
        assert _sanitize_for_json(float("nan")) is None

    def test_ndarray(self):
        arr = np.array([1.0, 2.0, 3.0])
        assert _sanitize_for_json(arr) == [1.0, 2.0, 3.0]

    def test_nested_dict(self):
        d = {"a": np.float64(1.0), "b": {"c": np.int32(2)}}
        result = _sanitize_for_json(d)
        assert result == {"a": 1.0, "b": {"c": 2}}
        assert json.dumps(result)  # must be serializable


class TestParseGeneratorData:
    def test_returns_json_string(self, job_dir: Path):
        result = parse_generator_data(job_dir)
        assert result is not None
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_contains_energy(self, job_dir: Path):
        parsed = json.loads(parse_generator_data(job_dir))
        assert "final_energy_eh" in parsed
        assert abs(parsed["final_energy_eh"] - (-626.605298577892)) < 1e-6

    def test_scf_converged(self, job_dir: Path):
        parsed = json.loads(parse_generator_data(job_dir))
        assert parsed.get("scf_converged") is True

    def test_homo_lumo(self, job_dir: Path):
        parsed = json.loads(parse_generator_data(job_dir))
        assert abs(parsed["homo_eh"] - (-0.976831)) < 1e-6
        assert abs(parsed["lumo_eh"] - (-0.554888)) < 1e-6
        assert abs(parsed["homo_lumo_gap_eh"] - 0.42194299999999996) < 1e-6

    def test_mulliken_charges(self, job_dir: Path):
        parsed = json.loads(parse_generator_data(job_dir))
        assert "mulliken_charges" in parsed
        charges = parsed["mulliken_charges"]
        assert "1_U" in charges
        assert abs(charges["1_U"] - 2.391968) < 1e-5
        assert abs(charges["2_O"] - (-0.195975)) < 1e-5
        assert abs(charges["3_O"] - (-0.195994)) < 1e-5

    def test_loewdin_charges(self, job_dir: Path):
        parsed = json.loads(parse_generator_data(job_dir))
        assert "loewdin_charges" in parsed
        charges = parsed["loewdin_charges"]
        assert abs(charges["1_U"] - 2.437017) < 1e-5
        assert abs(charges["2_O"] - (-0.218521)) < 1e-5

    def test_mayer_bond_orders(self, job_dir: Path):
        parsed = json.loads(parse_generator_data(job_dir))
        assert "mayer_bond_orders" in parsed
        bonds = parsed["mayer_bond_orders"]
        assert "1_U_to_2_O" in bonds
        assert abs(bonds["1_U_to_2_O"] - 2.3437) < 1e-4
        assert abs(bonds["1_U_to_3_O"] - 2.3437) < 1e-4

    def test_gradient(self, job_dir: Path):
        parsed = json.loads(parse_generator_data(job_dir))
        assert "gradient" in parsed
        assert "gradient_norm" in parsed
        assert "gradient_max" in parsed
        assert abs(parsed["gradient_max"] - 0.000137424) < 1e-9
        assert abs(parsed["gradient_norm"] - 0.0001896753) < 1e-9

    def test_warnings_present(self, job_dir: Path):
        parsed = json.loads(parse_generator_data(job_dir))
        assert "warnings" in parsed
        assert len(parsed["warnings"]) == 5

    def test_energy_components(self, job_dir: Path):
        parsed = json.loads(parse_generator_data(job_dir))
        ec = parsed["energy_components"]
        assert abs(ec["nuclear_repulsion_eh"] - 171.580507691624) < 1e-6
        assert abs(ec["electronic_energy_eh"] - (-798.363186173683)) < 1e-6

    def test_cache_written(self, job_dir: Path):
        parse_generator_data(job_dir)
        assert (job_dir / "generator_metrics.json").exists()

    def test_cache_hit_skips_reparse(self, job_dir: Path):
        first = parse_generator_data(job_dir)
        # Replace the symlink with a blank file; cache should still serve data
        (job_dir / "orca.out").unlink()
        (job_dir / "orca.out").write_text("")
        second = parse_generator_data(job_dir)
        assert first == second

    def test_recompute_ignores_cache(self, job_dir: Path):
        parse_generator_data(job_dir)
        # Poison the cache
        (job_dir / "generator_metrics.json").write_text('{"poisoned": true}')
        result = parse_generator_data(job_dir, recompute=True)
        parsed = json.loads(result)
        assert "poisoned" not in parsed
        assert "final_energy_eh" in parsed

    def test_missing_out_file_returns_none(self, tmp_path: Path):
        # Directory exists but has no orca.out
        result = parse_generator_data(tmp_path)
        assert result is None

    def test_unavailable_returns_none_without_package(self, job_dir: Path, monkeypatch):
        import oact_utilities.utils.analysis as analysis_mod

        monkeypatch.setattr(analysis_mod, "GENERATOR_AVAILABLE", False)
        result = parse_generator_data(job_dir)
        assert result is None

    def test_schema_migration_adds_column(self, tmp_path: Path):
        """generator_data column is added to existing DBs via _ensure_schema."""
        import sqlite3

        db_path = tmp_path / "workflow.db"

        # Create DB the old way (without generator_data column)
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """CREATE TABLE structures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                orig_index INTEGER,
                elements TEXT,
                natoms INTEGER,
                status TEXT DEFAULT 'to_run',
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
                n_cores INTEGER
            )"""
        )
        conn.execute(
            "INSERT INTO structures (orig_index, elements, natoms, status, charge, spin) "
            "VALUES (0, 'U;O;O', 3, 'to_run', 0, 1)"
        )
        conn.commit()
        conn.close()

        # Opening via ArchitectorWorkflow should migrate the schema
        from oact_utilities.workflows.architector_workflow import ArchitectorWorkflow

        with ArchitectorWorkflow(db_path) as wf:
            cur = wf._execute_with_retry("PRAGMA table_info(structures)")
            cols = {row[1] for row in cur.fetchall()}
            assert "generator_data" in cols
