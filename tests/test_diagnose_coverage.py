"""Tests for the clean.py coverage diagnostic."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from oact_utilities.workflows.diagnose_coverage import (
    _DIR_MISSING,
    _ESCAPES,
    _EXISTS,
    _NULL,
    diagnose_coverage,
)


def _create_test_db(db_path: Path, jobs: list[dict]) -> Path:
    """Create a minimal workflow SQLite database for testing.

    ArchitectorWorkflow migrates optimizer/worker_id/generator_data columns on
    open, so this minimal schema is sufficient.
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE structures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            orig_index INTEGER,
            elements TEXT,
            natoms INTEGER,
            status TEXT,
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
            n_cores INTEGER,
            optimizer TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    for job in jobs:
        conn.execute(
            """
            INSERT INTO structures
                (orig_index, elements, natoms, status, charge, spin, job_dir)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job.get("orig_index", 1),
                job.get("elements", "U;O"),
                job.get("natoms", 2),
                job.get("status", "completed"),
                job.get("charge", 0),
                job.get("spin", 1),
                job.get("job_dir"),
            ),
        )
    conn.commit()
    conn.close()
    return db_path


def test_coverage_bins_and_orphans(tmp_path: Path) -> None:
    """End-to-end: every bin, reroot-recoverability, orphans, running excluded."""
    root = tmp_path / "jobs"
    root.mkdir()
    # On-disk dirs: two referenced + one orphan that matches an escaping row.
    (root / "job_1").mkdir()
    (root / "job_2").mkdir()
    (root / "orphan_99").mkdir()

    db = tmp_path / "wf.db"
    _create_test_db(
        db,
        [
            # exists_processed (referenced -> job_1)
            {"status": "completed", "job_dir": str(root / "job_1")},
            # exists_processed (referenced -> job_2)
            {"status": "failed", "job_dir": str(root / "job_2")},
            # dir_missing: resolves under root, but job_3 absent on disk
            {"status": "timeout", "job_dir": str(root / "job_3")},
            # null_job_dir
            {"status": "to_run", "job_dir": None},
            # escapes_root, basename orphan_99 exists under root -> reroot-recoverable
            {"status": "completed", "job_dir": "/elsewhere/orphan_99"},
            # running is NOT in the clean set -> excluded entirely
            {"status": "running", "job_dir": str(root / "job_1")},
        ],
    )

    report = diagnose_coverage(db, root, workers=4)

    # running excluded -> 5 candidate rows, not 6
    assert report.total_rows == 5
    assert report.bins.get(_EXISTS, 0) == 2
    assert report.bins.get(_DIR_MISSING, 0) == 1
    assert report.bins.get(_ESCAPES, 0) == 1
    assert report.bins.get(_NULL, 0) == 1

    # Only the escaping orphan_99 row has a basename present under root.
    assert report.reroot_recoverable == 1

    # On-disk vs DB
    assert report.dirs_on_disk == 3
    assert report.dirs_referenced == 2
    assert report.orphan_dirs == 1  # orphan_99
    assert report.orphans_reroot_recoverable == 1  # orphan_99 is a DB basename
    assert "orphan_99" in report.orphan_samples


def test_no_gap_when_all_resolve(tmp_path: Path) -> None:
    """When every clean-set row resolves to an existing dir, there is no gap."""
    root = tmp_path / "jobs"
    root.mkdir()
    (root / "job_1").mkdir()
    (root / "job_2").mkdir()

    db = tmp_path / "wf.db"
    _create_test_db(
        db,
        [
            {"status": "completed", "job_dir": str(root / "job_1")},
            {"status": "to_run", "job_dir": str(root / "job_2")},
        ],
    )

    report = diagnose_coverage(db, root, workers=2)

    assert report.total_rows == 2
    assert report.bins.get(_EXISTS, 0) == 2
    assert report.bins.get(_DIR_MISSING, 0) == 0
    assert report.bins.get(_ESCAPES, 0) == 0
    assert report.bins.get(_NULL, 0) == 0
    assert report.orphan_dirs == 0
    assert report.reroot_recoverable == 0
