"""Tests for prepare_seeded_sella_db.py."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from oact_utilities.workflows.prepare_seeded_sella_db import (
    _safe_copy,
    prepare_seeded_sella_db,
    scan_for_job_dir,
    validate_seed_files,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

MINIMAL_ENGRAD = """\
#
# Number of atoms
#
 2
#
# The current total energy in Eh
#
   -1.234567890000
#
# The current gradient in Eh/bohr
#
       0.000000100000
       0.000000200000
       0.000123456789
      -0.000000100000
      -0.000000200000
      -0.000123456789
#
# The atomic numbers and current coordinates in Bohr
#
   1     0.0000000    0.0000000   -0.7000000
   1     0.0000000    0.0000000    0.7000000
"""

MINIMAL_GEOMETRY = "2\nH2\nH  0.0  0.0  0.0\nH  0.0  0.0  0.74\n"


def _make_sp_db(db_path: Path, rows: list[dict]) -> None:
    """Create a minimal SP workflow DB with the given rows (all status=completed)."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """CREATE TABLE structures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            orig_index INTEGER,
            elements TEXT,
            natoms INTEGER,
            status TEXT,
            charge INTEGER,
            spin INTEGER,
            geometry TEXT,
            job_dir TEXT
        )"""
    )
    for row in rows:
        conn.execute(
            "INSERT INTO structures (orig_index, elements, natoms, status, charge, spin, geometry)"
            " VALUES (?, ?, ?, 'completed', ?, ?, ?)",
            (
                row["orig_index"],
                row.get("elements", "H;H"),
                row.get("natoms", 2),
                row.get("charge", 0),
                row.get("spin", 1),
                row.get("geometry", MINIMAL_GEOMETRY),
            ),
        )
    conn.commit()
    conn.close()


def _make_sp_job_dir(parent: Path, name: str, valid: bool = True) -> Path:
    """Create a fake SP job directory with seed files."""
    job_dir = parent / name
    job_dir.mkdir(parents=True)
    if valid:
        (job_dir / "orca.engrad").write_text(MINIMAL_ENGRAD)
        (job_dir / "orca.gbw").write_bytes(b"\x00" * 8)
        (job_dir / "orca.out").write_text("ORCA TERMINATED NORMALLY\n")
    return job_dir


# ---------------------------------------------------------------------------
# scan_for_job_dir
# ---------------------------------------------------------------------------


class TestScanForJobDir:
    def test_exact_match(self, tmp_path):
        """Finds job_42 by exact name."""
        (tmp_path / "job_42").mkdir()
        result = scan_for_job_dir(tmp_path, 42)
        assert result == tmp_path / "job_42"

    def test_fallback_match_underscore(self, tmp_path):
        """Finds task_run_42 via regex fallback when no exact job_42 exists."""
        (tmp_path / "task_run_42").mkdir()
        result = scan_for_job_dir(tmp_path, 42)
        assert result == tmp_path / "task_run_42"

    def test_fallback_match_dash(self, tmp_path):
        """Regex fallback matches dirs ending with -42."""
        (tmp_path / "campaign-42").mkdir()
        result = scan_for_job_dir(tmp_path, 42)
        assert result == tmp_path / "campaign-42"

    def test_multiple_matches_raises(self, tmp_path):
        """Raises ValueError when multiple dirs match the fallback pattern."""
        (tmp_path / "run_42").mkdir()
        (tmp_path / "batch_42").mkdir()
        with pytest.raises(
            ValueError, match="Multiple directories match orig_index=42"
        ):
            scan_for_job_dir(tmp_path, 42)

    def test_no_match_returns_none(self, tmp_path):
        """Returns None when no directory matches."""
        (tmp_path / "job_99").mkdir()
        result = scan_for_job_dir(tmp_path, 42)
        assert result is None

    def test_no_substring_false_positive(self, tmp_path):
        """index=1 does NOT match job_10 or job_21 (substring containment excluded)."""
        (tmp_path / "job_10").mkdir()
        (tmp_path / "job_21").mkdir()
        result = scan_for_job_dir(tmp_path, 1)
        assert result is None

    def test_exact_takes_priority_over_fallback(self, tmp_path):
        """Exact job_42 is returned even when fallback matches also exist."""
        (tmp_path / "job_42").mkdir()
        (tmp_path / "run_42").mkdir()
        result = scan_for_job_dir(tmp_path, 42)
        assert result == tmp_path / "job_42"


# ---------------------------------------------------------------------------
# validate_seed_files
# ---------------------------------------------------------------------------


class TestValidateSeedFiles:
    def test_valid_dir_passes(self, tmp_path):
        job_dir = _make_sp_job_dir(tmp_path, "job_0")
        ok, reason = validate_seed_files(job_dir)
        assert ok
        assert reason == ""

    def test_missing_engrad(self, tmp_path):
        job_dir = _make_sp_job_dir(tmp_path, "job_0")
        (job_dir / "orca.engrad").unlink()
        ok, reason = validate_seed_files(job_dir)
        assert not ok
        assert "engrad" in reason

    def test_missing_gbw(self, tmp_path):
        job_dir = _make_sp_job_dir(tmp_path, "job_0")
        (job_dir / "orca.gbw").unlink()
        ok, reason = validate_seed_files(job_dir)
        assert not ok
        assert "gbw" in reason

    def test_missing_out(self, tmp_path):
        job_dir = _make_sp_job_dir(tmp_path, "job_0")
        (job_dir / "orca.out").unlink()
        ok, reason = validate_seed_files(job_dir)
        assert not ok
        assert "orca.out" in reason

    def test_corrupt_engrad(self, tmp_path):
        job_dir = _make_sp_job_dir(tmp_path, "job_0")
        (job_dir / "orca.engrad").write_text("not a valid engrad file\n")
        ok, reason = validate_seed_files(job_dir)
        assert not ok
        assert "empty gradient" in reason or "parse error" in reason


# ---------------------------------------------------------------------------
# _safe_copy
# ---------------------------------------------------------------------------


class TestSafeCopy:
    def test_normal_copy(self, tmp_path):
        src = tmp_path / "src" / "orca.engrad"
        src.parent.mkdir()
        src.write_text("data")
        dst = tmp_path / "dst" / "orca.engrad"
        dst.parent.mkdir()

        _safe_copy(src, dst, src.parent)

        assert dst.read_text() == "data"

    def test_symlink_escape_raises(self, tmp_path):
        """A symlink pointing outside job_dir is rejected."""
        outside = tmp_path / "outside.txt"
        outside.write_text("secret")

        job_dir = tmp_path / "job"
        job_dir.mkdir()
        link = job_dir / "orca.engrad"
        link.symlink_to(outside)

        dst = tmp_path / "dst" / "orca.engrad"
        dst.parent.mkdir()

        with pytest.raises(ValueError, match="Symlink escape"):
            _safe_copy(link, dst, job_dir)


# ---------------------------------------------------------------------------
# prepare_seeded_sella_db (integration)
# ---------------------------------------------------------------------------


class TestPrepareSeededSellaDb:
    def _setup(self, tmp_path, n_jobs: int = 3):
        """Create a SP DB and job dirs, return (sp_db, scan_dir, out_db, seed_dir)."""
        scan_dir = tmp_path / "sp_jobs"
        scan_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for i in range(n_jobs):
            _make_sp_job_dir(scan_dir, f"job_{i}")
            rows.append({"orig_index": i})

        sp_db = tmp_path / "sp.db"
        _make_sp_db(sp_db, rows)

        return sp_db, scan_dir, tmp_path / "sella.db", tmp_path / "seeds"

    def test_dry_run_copies_nothing(self, tmp_path):
        sp_db, scan_dir, out_db, seed_dir = self._setup(tmp_path)

        prepare_seeded_sella_db(sp_db, scan_dir, out_db, seed_dir, execute=False)

        assert not out_db.exists()
        assert not seed_dir.exists()

    def test_execute_copies_three_files(self, tmp_path):
        sp_db, scan_dir, out_db, seed_dir = self._setup(tmp_path, n_jobs=1)

        prepare_seeded_sella_db(sp_db, scan_dir, out_db, seed_dir, execute=True)

        seed_folder = seed_dir / "orig_index_0"
        assert (seed_folder / "orca.engrad").exists()
        assert (seed_folder / "orca.gbw").exists()
        assert (seed_folder / "orca.out").exists()

    def test_execute_creates_output_db(self, tmp_path):
        sp_db, scan_dir, out_db, seed_dir = self._setup(tmp_path)

        prepare_seeded_sella_db(sp_db, scan_dir, out_db, seed_dir, execute=True)

        assert out_db.exists()

    def test_execute_sets_job_dir_in_db(self, tmp_path):
        sp_db, scan_dir, out_db, seed_dir = self._setup(tmp_path, n_jobs=1)

        prepare_seeded_sella_db(sp_db, scan_dir, out_db, seed_dir, execute=True)

        conn = sqlite3.connect(str(out_db))
        row = conn.execute(
            "SELECT job_dir FROM structures WHERE orig_index=0"
        ).fetchone()
        conn.close()

        expected = str((seed_dir / "orig_index_0").resolve())
        assert row[0] == expected

    def test_execute_status_is_to_run(self, tmp_path):
        sp_db, scan_dir, out_db, seed_dir = self._setup(tmp_path)

        prepare_seeded_sella_db(sp_db, scan_dir, out_db, seed_dir, execute=True)

        conn = sqlite3.connect(str(out_db))
        statuses = {
            r[0] for r in conn.execute("SELECT DISTINCT status FROM structures")
        }
        conn.close()
        assert statuses == {"to_run"}

    def test_missing_job_dirs_excluded(self, tmp_path):
        """Jobs with no matching directory are excluded from the output DB."""
        scan_dir = tmp_path / "sp_jobs"
        scan_dir.mkdir()
        _make_sp_job_dir(scan_dir, "job_0")

        sp_db = tmp_path / "sp.db"
        _make_sp_db(sp_db, [{"orig_index": i} for i in range(3)])
        out_db = tmp_path / "sella.db"
        seed_dir = tmp_path / "seeds"

        prepare_seeded_sella_db(sp_db, scan_dir, out_db, seed_dir, execute=True)

        conn = sqlite3.connect(str(out_db))
        count = conn.execute("SELECT COUNT(*) FROM structures").fetchone()[0]
        conn.close()
        assert count == 1

    def test_invalid_seed_files_excluded(self, tmp_path):
        """Jobs with corrupt engrad are excluded."""
        scan_dir = tmp_path / "sp_jobs"
        scan_dir.mkdir()
        _make_sp_job_dir(scan_dir, "job_0")
        bad = _make_sp_job_dir(scan_dir, "job_1")
        (bad / "orca.engrad").write_text("garbage")

        sp_db = tmp_path / "sp.db"
        _make_sp_db(sp_db, [{"orig_index": 0}, {"orig_index": 1}])
        out_db = tmp_path / "sella.db"
        seed_dir = tmp_path / "seeds"

        prepare_seeded_sella_db(sp_db, scan_dir, out_db, seed_dir, execute=True)

        conn = sqlite3.connect(str(out_db))
        count = conn.execute("SELECT COUNT(*) FROM structures").fetchone()[0]
        conn.close()
        assert count == 1

    def test_geometry_carried_over(self, tmp_path):
        sp_db, scan_dir, out_db, seed_dir = self._setup(tmp_path, n_jobs=1)

        prepare_seeded_sella_db(sp_db, scan_dir, out_db, seed_dir, execute=True)

        conn = sqlite3.connect(str(out_db))
        row = conn.execute(
            "SELECT geometry FROM structures WHERE orig_index=0"
        ).fetchone()
        conn.close()
        assert row[0] == MINIMAL_GEOMETRY

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def test_fraction_reduces_row_count(self, tmp_path):
        sp_db, scan_dir, out_db, seed_dir = self._setup(tmp_path, n_jobs=10)

        prepare_seeded_sella_db(
            sp_db, scan_dir, out_db, seed_dir, execute=True, fraction=0.5
        )

        conn = sqlite3.connect(str(out_db))
        count = conn.execute("SELECT COUNT(*) FROM structures").fetchone()[0]
        conn.close()
        assert 4 <= count <= 6

    def test_fraction_reproducible_with_same_seed(self, tmp_path):
        # Run A
        sp_db_a, scan_dir_a, out_db_a, seed_dir_a = self._setup(
            tmp_path / "a", n_jobs=20
        )
        # Run B (independent copy)
        sp_db_b, scan_dir_b, out_db_b, seed_dir_b = self._setup(
            tmp_path / "b", n_jobs=20
        )

        prepare_seeded_sella_db(
            sp_db_a,
            scan_dir_a,
            out_db_a,
            seed_dir_a,
            execute=True,
            fraction=0.25,
            random_seed=7,
        )
        prepare_seeded_sella_db(
            sp_db_b,
            scan_dir_b,
            out_db_b,
            seed_dir_b,
            execute=True,
            fraction=0.25,
            random_seed=7,
        )

        conn_a = sqlite3.connect(str(out_db_a))
        conn_b = sqlite3.connect(str(out_db_b))
        indices_a = {r[0] for r in conn_a.execute("SELECT orig_index FROM structures")}
        indices_b = {r[0] for r in conn_b.execute("SELECT orig_index FROM structures")}
        conn_a.close()
        conn_b.close()

        assert indices_a == indices_b

    def test_fraction_different_seed_gives_different_sample(self, tmp_path):
        sp_db_a, scan_dir_a, out_db_a, seed_dir_a = self._setup(
            tmp_path / "a", n_jobs=20
        )
        sp_db_b, scan_dir_b, out_db_b, seed_dir_b = self._setup(
            tmp_path / "b", n_jobs=20
        )

        prepare_seeded_sella_db(
            sp_db_a,
            scan_dir_a,
            out_db_a,
            seed_dir_a,
            execute=True,
            fraction=0.5,
            random_seed=1,
        )
        prepare_seeded_sella_db(
            sp_db_b,
            scan_dir_b,
            out_db_b,
            seed_dir_b,
            execute=True,
            fraction=0.5,
            random_seed=2,
        )

        conn_a = sqlite3.connect(str(out_db_a))
        conn_b = sqlite3.connect(str(out_db_b))
        indices_a = {r[0] for r in conn_a.execute("SELECT orig_index FROM structures")}
        indices_b = {r[0] for r in conn_b.execute("SELECT orig_index FROM structures")}
        conn_a.close()
        conn_b.close()

        assert indices_a != indices_b
