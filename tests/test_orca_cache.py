"""Tests for orca_metrics.json cache functionality."""

import json
import time
from pathlib import Path

import pytest

from oact_utilities.utils.analysis import (
    parse_job_metrics,
    read_orca_cache,
    write_orca_cache,
)
from oact_utilities.utils.status import check_job_termination, pull_log_file

# --- Fixtures ---


@pytest.fixture
def tmp_job_dir(tmp_path: Path) -> Path:
    """Create a temporary job directory with a minimal ORCA output file."""
    job_dir = tmp_path / "job_0"
    job_dir.mkdir()

    # Write a minimal ORCA output file that passes parsing
    orca_output = """\
                            *     ORCA      *

Program Version 6.0.0

nprocs                     4

SCF CONVERGED AFTER   12 CYCLES

FINAL SINGLE POINT ENERGY      -1234.567890

ORCA TERMINATED NORMALLY
"""
    (job_dir / "orca.out").write_text(orca_output)
    return job_dir


@pytest.fixture
def sample_metrics() -> dict:
    """Return a sample metrics dict like parse_job_metrics returns."""
    return {
        "max_forces": 0.00042,
        "scf_steps": 12,
        "final_energy": -1234.567890,
        "success": True,
        "is_timeout": False,
        "termination_status": 1,
        "mulliken_population": None,
        "nprocs": 4,
        "wall_time": None,
        "time_dict": None,
        "sella_steps": None,
    }


# --- write_orca_cache tests ---


class TestWriteOrcaCache:
    def test_creates_valid_json(self, tmp_path: Path, sample_metrics: dict) -> None:
        """write_orca_cache produces valid JSON with _source_mtime."""
        cache_path = tmp_path / "orca_metrics.json"
        write_orca_cache(cache_path, sample_metrics, 1234567890.0)

        assert cache_path.exists()
        data = json.loads(cache_path.read_text())
        assert data["_source_mtime"] == 1234567890.0
        assert data["scf_steps"] == 12
        assert data["final_energy"] == -1234.567890

    def test_atomic_write_no_partial_file(self, tmp_path: Path) -> None:
        """Tmp file is cleaned up even on non-OSError scenarios."""
        cache_path = tmp_path / "orca_metrics.json"
        write_orca_cache(cache_path, {"test": True}, 100.0)
        assert cache_path.exists()
        # No .tmp file should remain
        assert not cache_path.with_suffix(".tmp").exists()

    def test_silent_on_readonly_directory(self, tmp_path: Path) -> None:
        """write_orca_cache does not raise on write failure."""
        # Use a non-existent parent directory to trigger OSError
        cache_path = tmp_path / "nonexistent_dir" / "orca_metrics.json"
        # Should not raise
        write_orca_cache(cache_path, {"test": True}, 100.0)
        assert not cache_path.exists()

    def test_does_not_mutate_input_dict(
        self, tmp_path: Path, sample_metrics: dict
    ) -> None:
        """write_orca_cache does not add _source_mtime to the input dict."""
        cache_path = tmp_path / "orca_metrics.json"
        original_keys = set(sample_metrics.keys())
        write_orca_cache(cache_path, sample_metrics, 100.0)
        assert set(sample_metrics.keys()) == original_keys


# --- read_orca_cache tests ---


class TestReadOrcaCache:
    def test_returns_dict_for_valid_cache(
        self, tmp_path: Path, sample_metrics: dict
    ) -> None:
        """read_orca_cache returns metrics dict for a valid, fresh cache."""
        cache_path = tmp_path / "orca_metrics.json"
        write_orca_cache(cache_path, sample_metrics, 100.0)

        result = read_orca_cache(cache_path, 100.0)
        assert result is not None
        assert result["scf_steps"] == 12
        assert result["final_energy"] == -1234.567890
        # _source_mtime should be stripped
        assert "_source_mtime" not in result

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        """read_orca_cache returns None when cache file does not exist."""
        cache_path = tmp_path / "orca_metrics.json"
        result = read_orca_cache(cache_path, 100.0)
        assert result is None

    def test_returns_none_for_stale_cache(
        self, tmp_path: Path, sample_metrics: dict
    ) -> None:
        """read_orca_cache returns None when source is newer than cache."""
        cache_path = tmp_path / "orca_metrics.json"
        write_orca_cache(cache_path, sample_metrics, 100.0)

        # Source mtime is newer than what's in the cache
        result = read_orca_cache(cache_path, 200.0)
        assert result is None

    def test_returns_none_for_corrupted_json(self, tmp_path: Path) -> None:
        """read_orca_cache returns None for invalid JSON content."""
        cache_path = tmp_path / "orca_metrics.json"
        cache_path.write_text("not valid json {{{")

        result = read_orca_cache(cache_path, 100.0)
        assert result is None

    def test_returns_none_for_missing_mtime_field(self, tmp_path: Path) -> None:
        """read_orca_cache returns None when _source_mtime is missing (always stale)."""
        cache_path = tmp_path / "orca_metrics.json"
        cache_path.write_text(json.dumps({"scf_steps": 10}))

        # Source mtime > 0, cache has no _source_mtime (defaults to 0)
        result = read_orca_cache(cache_path, 1.0)
        assert result is None

    def test_fresh_cache_with_equal_mtime(
        self, tmp_path: Path, sample_metrics: dict
    ) -> None:
        """Cache is considered fresh when source mtime equals cached mtime."""
        cache_path = tmp_path / "orca_metrics.json"
        write_orca_cache(cache_path, sample_metrics, 100.0)

        result = read_orca_cache(cache_path, 100.0)
        assert result is not None


# --- parse_job_metrics cache integration tests ---


class TestParseJobMetricsCache:
    def test_creates_cache_on_first_call(self, tmp_job_dir: Path) -> None:
        """parse_job_metrics creates orca_metrics.json after parsing."""
        cache_path = tmp_job_dir / "orca_metrics.json"
        assert not cache_path.exists()

        parse_job_metrics(tmp_job_dir)

        assert cache_path.exists()
        data = json.loads(cache_path.read_text())
        assert "_source_mtime" in data
        assert data["scf_steps"] == 12

    def test_uses_cache_on_second_call(self, tmp_job_dir: Path) -> None:
        """parse_job_metrics returns cached result without re-parsing."""
        # First call: creates cache
        result1 = parse_job_metrics(tmp_job_dir)

        # Second call: should use cache (and include _cache_hit marker)
        result2 = parse_job_metrics(tmp_job_dir)

        assert result2.get("_cache_hit") is True
        # Core metrics should match
        assert result1["scf_steps"] == result2["scf_steps"]
        assert result1["final_energy"] == result2["final_energy"]

    def test_recompute_ignores_cache(self, tmp_job_dir: Path) -> None:
        """parse_job_metrics with recompute=True skips cache read."""
        # First call: creates cache
        parse_job_metrics(tmp_job_dir)

        # Recompute: should NOT have _cache_hit
        result = parse_job_metrics(tmp_job_dir, recompute=True)
        assert result.get("_cache_hit") is not True

    def test_gzipped_output_creates_cache(self, tmp_path: Path) -> None:
        """parse_job_metrics creates cache for gzipped quacc output."""
        import gzip

        job_dir = tmp_path / "quacc_job"
        job_dir.mkdir()

        orca_output = """\
nprocs                     8

SCF CONVERGED AFTER   20 CYCLES

FINAL SINGLE POINT ENERGY      -5678.123456

ORCA TERMINATED NORMALLY
"""
        # Write gzipped output
        with gzip.open(job_dir / "orca.out.gz", "wt") as f:
            f.write(orca_output)

        # First call with unzip=True
        result = parse_job_metrics(job_dir, unzip=True)
        cache_path = job_dir / "orca_metrics.json"
        assert cache_path.exists()
        assert result["scf_steps"] == 20

        # Second call should use cache
        result2 = parse_job_metrics(job_dir, unzip=True)
        assert result2.get("_cache_hit") is True
        assert result2["scf_steps"] == 20

    def test_stale_cache_triggers_regeneration(self, tmp_job_dir: Path) -> None:
        """When source file is touched, stale cache is regenerated."""
        # First call: creates cache
        parse_job_metrics(tmp_job_dir)
        cache_path = tmp_job_dir / "orca_metrics.json"
        old_content = cache_path.read_text()

        # Touch the source file to make cache stale
        time.sleep(0.05)  # Ensure mtime differs
        orca_out = tmp_job_dir / "orca.out"
        orca_out.write_text(orca_out.read_text())

        # Second call: should re-parse (not use stale cache)
        result = parse_job_metrics(tmp_job_dir)
        assert result.get("_cache_hit") is not True

        # Cache should be regenerated with new mtime
        new_data = json.loads(cache_path.read_text())
        old_data = json.loads(old_content)
        assert new_data["_source_mtime"] > old_data["_source_mtime"]


# --- File discovery exclusion tests ---


class TestCacheFileDiscovery:
    def test_pull_log_file_ignores_cache(self, tmp_job_dir: Path) -> None:
        """pull_log_file does not return orca_metrics.json."""
        # Create cache file
        (tmp_job_dir / "orca_metrics.json").write_text("{}")

        log_file = pull_log_file(str(tmp_job_dir))
        assert log_file is not None
        assert "orca_metrics.json" not in log_file
        assert log_file.endswith(".out") or log_file.endswith(".gz")

    def test_check_job_termination_ignores_cache(self, tmp_job_dir: Path) -> None:
        """check_job_termination does not pick up orca_metrics.json."""
        # Create cache file (make it the newest file)
        time.sleep(0.05)
        (tmp_job_dir / "orca_metrics.json").write_text("{}")

        # Should still detect completion from orca.out, not read cache
        status = check_job_termination(str(tmp_job_dir))
        assert status == 1  # COMPLETED (from orca.out content)


# --- schema version tests ---


class TestCacheSchemaVersion:
    def test_write_stamps_schema_version(
        self, tmp_path: Path, sample_metrics: dict
    ) -> None:
        """write_orca_cache records the schema version alongside the mtime."""
        from oact_utilities.utils.analysis import _ORCA_CACHE_SCHEMA_VERSION

        cache_path = tmp_path / "orca_metrics.json"
        write_orca_cache(cache_path, sample_metrics, 100.0)

        data = json.loads(cache_path.read_text())
        assert data["_schema_version"] == _ORCA_CACHE_SCHEMA_VERSION

    def test_read_strips_schema_version(
        self, tmp_path: Path, sample_metrics: dict
    ) -> None:
        """The version marker is not returned to callers."""
        cache_path = tmp_path / "orca_metrics.json"
        write_orca_cache(cache_path, sample_metrics, 100.0)

        result = read_orca_cache(cache_path, 100.0)
        assert result is not None
        assert "_schema_version" not in result

    def test_rejects_unversioned_cache(
        self, tmp_path: Path, sample_metrics: dict
    ) -> None:
        """A pre-versioning cache is rejected even when its mtime is fresh.

        Without this, warm caches written before force_max / num_electrons_scf
        existed would be served forever with those keys silently absent.
        """
        cache_path = tmp_path / "orca_metrics.json"
        cache_path.write_text(json.dumps({**sample_metrics, "_source_mtime": 100.0}))

        assert read_orca_cache(cache_path, 100.0) is None

    def test_rejects_other_schema_version(
        self, tmp_path: Path, sample_metrics: dict
    ) -> None:
        """A cache from a different schema version is re-parsed."""
        cache_path = tmp_path / "orca_metrics.json"
        cache_path.write_text(
            json.dumps(
                {**sample_metrics, "_source_mtime": 100.0, "_schema_version": 999}
            )
        )

        assert read_orca_cache(cache_path, 100.0) is None

    def test_stale_cache_regenerated_with_new_keys(self, tmp_job_dir: Path) -> None:
        """An unversioned on-disk cache is replaced by a full parse."""
        cache_path = tmp_job_dir / "orca_metrics.json"
        source_mtime = (tmp_job_dir / "orca.out").stat().st_mtime
        cache_path.write_text(
            json.dumps({"scf_steps": 999, "_source_mtime": source_mtime})
        )

        result = parse_job_metrics(tmp_job_dir)

        assert result.get("_cache_hit") is not True
        assert result["scf_steps"] == 12
        assert "force_max" in result
        assert "num_electrons_scf" in result


# --- with_engrad tests ---


class TestWithEngrad:
    @pytest.fixture
    def job_with_engrad(self, tmp_job_dir: Path) -> Path:
        """Add an .engrad whose single atom carries a 3-4-0 gradient (norm 5)."""
        (tmp_job_dir / "orca.engrad").write_text(
            "#\n# Number of atoms\n#\n 1\n"
            "#\n# The current total energy in Eh\n#\n  -1234.567890\n"
            "#\n# The current gradient in Eh/bohr\n#\n"
            "  3.000000000\n  4.000000000\n  0.000000000\n"
            "#\n# The atomic numbers and current coordinates in Bohr\n#\n"
            "   8  0.000000 0.000000 0.000000\n"
        )
        return tmp_job_dir

    def test_engrad_read_by_default(self, job_with_engrad: Path) -> None:
        metrics = parse_job_metrics(job_with_engrad, recompute=True)
        assert metrics["force_max"] == pytest.approx(5.0)

    def test_with_engrad_false_skips_the_file(self, job_with_engrad: Path) -> None:
        """Census parses the .engrad itself and opts out of a second read."""
        metrics = parse_job_metrics(job_with_engrad, recompute=True, with_engrad=False)
        assert metrics["force_max"] is None

    def test_cache_hit_backfills_a_skipped_force_max(
        self, job_with_engrad: Path
    ) -> None:
        """A cache written with with_engrad=False must not starve later callers.

        Census writes such caches. A dashboard run over the same corpus would
        otherwise see force_max as permanently absent.
        """
        parse_job_metrics(job_with_engrad, recompute=True, with_engrad=False)
        cache_path = job_with_engrad / "orca_metrics.json"
        assert json.loads(cache_path.read_text())["force_max"] is None

        metrics = parse_job_metrics(job_with_engrad)

        assert metrics.get("_cache_hit") is True
        assert metrics["force_max"] == pytest.approx(5.0)
        # Written back, so the corpus converges on complete caches.
        assert json.loads(cache_path.read_text())["force_max"] == pytest.approx(5.0)

    def test_cache_hit_without_engrad_does_not_loop(self, tmp_job_dir: Path) -> None:
        """A job with no .engrad at all stays a clean cache hit."""
        parse_job_metrics(tmp_job_dir, recompute=True)
        metrics = parse_job_metrics(tmp_job_dir)

        assert metrics.get("_cache_hit") is True
        assert metrics["force_max"] is None
