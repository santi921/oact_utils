"""Regression tests for .do_not_rerun.json marker handling in dashboard --update.

Covers:
- Basic TO_RUN + marker -> FAILED transition with correct error_message.
- CAS guard: concurrent writer flipping the row to FAILED does not cause
  dashboard's marker update to double-bump fail_count.
- Idempotent re-run: a second --update on an already-FAILED marker row
  does not re-increment fail_count.
- Single-commit invariant: marker-blocked rows and normal transitions
  share one _commit_with_retry call per update_all_statuses invocation.
"""

from __future__ import annotations

import sqlite3

import pytest

from oact_utilities.utils.architector import create_workflow_db
from oact_utilities.workflows.architector_workflow import (
    ArchitectorWorkflow,
    JobStatus,
)
from oact_utilities.workflows.clean import MARKER_ERROR_MESSAGE, MARKER_FILENAME
from oact_utilities.workflows.dashboard import update_all_statuses
from oact_utilities.workflows.job_dir_patterns import (
    DEFAULT_JOB_DIR_PATTERN,
    render_job_dir_pattern,
)


@pytest.fixture
def marker_workflow(tmp_path):
    """One-row workflow DB with a purged-style job dir (marker only, no orca.out)."""
    csv = tmp_path / "in.csv"
    csv.write_text("orig_index,charge,spin,aligned_csd_core\n0,0,1,Am 0 0 0\n")
    db = create_workflow_db(
        csv_path=str(csv),
        db_path=str(tmp_path / "wf.db"),
        geometry_column="aligned_csd_core",
    )
    jobs_root = tmp_path / "jobs"
    jd = jobs_root / render_job_dir_pattern(
        DEFAULT_JOB_DIR_PATTERN, orig_index=0, job_id=1
    )
    jd.mkdir(parents=True)
    (jd / MARKER_FILENAME).write_text('{"failure_reason": "SCF not converged"}')
    return db, jobs_root, jd


def _seed_to_run(db_path, job_dir):
    with ArchitectorWorkflow(db_path) as wf:
        wf.update_status_bulk([1], JobStatus.TO_RUN)
        wf._execute_with_retry(
            "UPDATE structures SET job_dir = ? WHERE id = 1", (str(job_dir),)
        )
        wf._commit_with_retry()


def test_marker_update_transitions_to_failed(marker_workflow):
    db, jobs_root, jd = marker_workflow
    _seed_to_run(db, jd)

    with ArchitectorWorkflow(db) as wf:
        update_all_statuses(wf, root_dir=jobs_root)
        row = wf._execute_with_retry(
            "SELECT status, fail_count, error_message FROM structures WHERE id = 1"
        ).fetchone()

    assert row[0] == "failed"
    assert row[1] == 1
    assert row[2] == MARKER_ERROR_MESSAGE


def test_marker_update_is_idempotent(marker_workflow):
    db, jobs_root, jd = marker_workflow
    _seed_to_run(db, jd)

    with ArchitectorWorkflow(db) as wf:
        update_all_statuses(wf, root_dir=jobs_root)
        update_all_statuses(wf, root_dir=jobs_root)
        row = wf._execute_with_retry(
            "SELECT fail_count FROM structures WHERE id = 1"
        ).fetchone()

    # Second run finds row already FAILED and no-ops via snapshot guard.
    assert row[0] == 1


def test_marker_update_cas_prevents_double_increment(marker_workflow, monkeypatch):
    """Concurrent writer flips the row to FAILED between snapshot and commit.

    Dashboard's marker update uses only_if_status=<snapshot_old_status> so
    the racing write no-ops, leaving fail_count at the value set by the
    concurrent writer (1), not 2.
    """
    db, jobs_root, jd = marker_workflow
    _seed_to_run(db, jd)

    original = ArchitectorWorkflow.update_status_bulk_multi

    def racing_update(self, status_groups, additional=None):
        # Simulate submit_jobs or a second dashboard racing ahead.
        with sqlite3.connect(db, timeout=5.0) as conn2:
            conn2.execute("PRAGMA journal_mode=DELETE")
            conn2.execute(
                "UPDATE structures SET status='failed', "
                "fail_count = COALESCE(fail_count,0) + 1, "
                "error_message='concurrent writer' WHERE id = 1"
            )
            conn2.commit()
        return original(self, status_groups, additional)

    monkeypatch.setattr(ArchitectorWorkflow, "update_status_bulk_multi", racing_update)

    with ArchitectorWorkflow(db) as wf:
        update_all_statuses(wf, root_dir=jobs_root)

    with ArchitectorWorkflow(db) as wf:
        row = wf._execute_with_retry(
            "SELECT status, fail_count, error_message FROM structures WHERE id = 1"
        ).fetchone()

    assert row[0] == "failed"
    assert row[1] == 1, f"CAS should have no-opped; fail_count bumped to {row[1]}"
    assert row[2] == "concurrent writer"


def test_marker_update_uses_single_commit(marker_workflow, monkeypatch):
    """update_all_statuses issues exactly one _commit_with_retry even when
    both normal status transitions and marker-blocked transitions occur."""
    db, jobs_root, jd = marker_workflow
    _seed_to_run(db, jd)

    counter = {"n": 0}
    original = ArchitectorWorkflow._commit_with_retry

    def counting_commit(self):
        counter["n"] += 1
        return original(self)

    with ArchitectorWorkflow(db) as wf:
        monkeypatch.setattr(ArchitectorWorkflow, "_commit_with_retry", counting_commit)
        counter["n"] = 0
        update_all_statuses(wf, root_dir=jobs_root)

    assert (
        counter["n"] == 1
    ), f"expected exactly one commit per update_all_statuses, got {counter['n']}"
