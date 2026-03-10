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
