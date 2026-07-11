"""Tests for the one-time A -> B job-root merge script."""

import json
import os
import time
from pathlib import Path

from oact_utilities.scripts.merge_job_roots import (
    ACT_DISCARD_SRC,
    ACT_DROP_BOTH,
    ACT_DROP_SRC,
    ACT_KEEP_DEST,
    ACT_MOVE,
    ACT_REPLACE_DEST,
    ACT_SKIP_MISMATCH,
    build_plan,
    execute_plan,
    scan_root,
)
from oact_utilities.workflows.clean import MARKER_FILENAME

_NOW = time.time()


_DEFAULT_INP = "* xyz 0 1\nH 0.0 0.0 0.0\n*\n"
_OTHER_INP = "* xyz 0 1\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n*\n"


def _set_times(job_dir: Path, mtime: float) -> None:
    for p in job_dir.rglob("*"):
        os.utime(p, (mtime, mtime))
    os.utime(job_dir, (mtime, mtime))


def _make_completed(
    root: Path, name: str, mtime: float = _NOW, inp: str = _DEFAULT_INP
) -> Path:
    jd = root / name
    jd.mkdir(parents=True)
    (jd / "orca.inp").write_text(inp)
    (jd / "orca.out").write_text("stuff\n" * 20 + "ORCA TERMINATED NORMALLY\n")
    _set_times(jd, mtime)
    return jd


def _make_failed(
    root: Path, name: str, mtime: float = _NOW, inp: str = _DEFAULT_INP
) -> Path:
    jd = root / name
    jd.mkdir(parents=True)
    (jd / "orca.inp").write_text(inp)
    (jd / "orca.out").write_text("stuff\n" * 20 + "aborting the run\n")
    _set_times(jd, mtime)
    return jd


def _make_to_run(
    root: Path, name: str, mtime: float = _NOW, inp: str = _DEFAULT_INP
) -> Path:
    jd = root / name
    jd.mkdir(parents=True)
    (jd / "orca.inp").write_text(inp)
    _set_times(jd, mtime)
    return jd


def _scan_both(root_a: Path, root_b: Path):
    jobs_a = scan_root(root_a, hours_cutoff=24, workers=2, label="A")
    jobs_b = scan_root(root_b, hours_cutoff=24, workers=2, label="B")
    return jobs_a, jobs_b


def test_status_classification(tmp_path):
    root = tmp_path / "r"
    _make_completed(root, "done")
    _make_failed(root, "dead")
    _make_to_run(root, "queued")
    jobs = scan_root(root, hours_cutoff=24, workers=2, label="A")
    assert jobs["done"].status == "completed"
    assert jobs["dead"].status == "failed"
    assert jobs["queued"].status == "to_run"


def test_plan_actions(tmp_path):
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    _make_completed(root_a, "a_only_done")
    _make_to_run(root_a, "a_only_queued")
    _make_completed(root_b, "b_only_done")
    # Collision: both to_run, A newer -> A wins.
    _make_to_run(root_a, "both_queued", mtime=_NOW)
    _make_to_run(root_b, "both_queued", mtime=_NOW - 5000)
    # Collision: A completed vs B to_run with newer files -> A still wins.
    _make_completed(root_a, "tier_beats_time", mtime=_NOW - 5000)
    _make_to_run(root_b, "tier_beats_time", mtime=_NOW)
    # Collision: both completed, B newer -> B wins.
    _make_completed(root_a, "b_newer_done", mtime=_NOW - 5000)
    _make_completed(root_b, "b_newer_done", mtime=_NOW)

    jobs_a, jobs_b = _scan_both(root_a, root_b)
    plan = {e.name: e for e in build_plan(jobs_a, jobs_b, drop_incomplete=False)}

    assert plan["a_only_done"].action == ACT_MOVE
    assert plan["a_only_queued"].action == ACT_MOVE  # kept without flag
    assert plan["b_only_done"].action == ACT_KEEP_DEST
    assert plan["both_queued"].action == ACT_REPLACE_DEST
    assert plan["tier_beats_time"].action == ACT_REPLACE_DEST
    assert plan["b_newer_done"].action == ACT_DISCARD_SRC


def test_plan_drop_incomplete(tmp_path):
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    _make_to_run(root_a, "a_queued")
    _make_to_run(root_b, "b_queued")
    _make_to_run(root_a, "both_queued")
    _make_to_run(root_b, "both_queued")
    _make_completed(root_a, "a_done")

    jobs_a, jobs_b = _scan_both(root_a, root_b)
    plan = {e.name: e for e in build_plan(jobs_a, jobs_b, drop_incomplete=True)}

    assert plan["a_queued"].action == ACT_DROP_SRC
    assert plan["b_queued"].action == "drop_dest"
    assert plan["both_queued"].action == ACT_DROP_BOTH
    assert plan["a_done"].action == ACT_MOVE


def test_hash_mismatch_collision_skipped(tmp_path):
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    # Same name, DIFFERENT molecules: nothing may be deleted, even though A
    # is completed and newer.
    _make_completed(root_a, "clash", mtime=_NOW, inp=_DEFAULT_INP)
    _make_completed(root_b, "clash", mtime=_NOW - 5000, inp=_OTHER_INP)
    # Same name, hash unverifiable (B copy has no orca.inp).
    _make_completed(root_a, "noinp", inp=_DEFAULT_INP)
    jd = root_b / "noinp"
    jd.mkdir(parents=True)
    (jd / "orca.out").write_text("stuff\n" * 20 + "ORCA TERMINATED NORMALLY\n")

    jobs_a, jobs_b = _scan_both(root_a, root_b)
    plan = {e.name: e for e in build_plan(jobs_a, jobs_b, drop_incomplete=False)}
    assert plan["clash"].action == ACT_SKIP_MISMATCH
    assert plan["noinp"].action == ACT_SKIP_MISMATCH

    counts, markers, errors = execute_plan(
        list(plan.values()), root_b, root_a, execute=True
    )
    assert not errors
    assert markers == 0
    # All four copies untouched.
    assert (root_a / "clash").is_dir() and (root_b / "clash").is_dir()
    assert (root_a / "noinp").is_dir() and (root_b / "noinp").is_dir()


def test_hash_mismatch_incomplete_still_dropped(tmp_path):
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    # Different molecules under one name, but both incomplete: with
    # --drop-incomplete the removal is status-based, so both still go.
    _make_to_run(root_a, "clash", inp=_DEFAULT_INP)
    _make_to_run(root_b, "clash", inp=_OTHER_INP)

    jobs_a, jobs_b = _scan_both(root_a, root_b)
    plan = build_plan(jobs_a, jobs_b, drop_incomplete=True)
    assert plan[0].action == ACT_DROP_BOTH
    execute_plan(plan, root_b, root_a, execute=True)
    assert not (root_a / "clash").exists()
    assert not (root_b / "clash").exists()


def test_dry_run_touches_nothing(tmp_path):
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    _make_completed(root_a, "done")
    _make_to_run(root_a, "queued")
    _make_failed(root_b, "dead")

    jobs_a, jobs_b = _scan_both(root_a, root_b)
    plan = build_plan(jobs_a, jobs_b, drop_incomplete=True)
    counts, markers, errors = execute_plan(plan, root_b, root_a, execute=False)

    assert not errors
    assert markers == 1  # only the failed job would be markered
    assert (root_a / "done").is_dir()
    assert (root_a / "queued").is_dir()
    assert not (root_b / "done").exists()
    assert not (root_b / "dead" / MARKER_FILENAME).exists()


def test_execute_moves_markers_and_drops(tmp_path):
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    _make_completed(root_a, "done")
    _make_failed(root_a, "dead")
    _make_to_run(root_a, "queued")
    _make_completed(root_b, "already_home")

    jobs_a, jobs_b = _scan_both(root_a, root_b)
    plan = build_plan(jobs_a, jobs_b, drop_incomplete=True)
    counts, markers, errors = execute_plan(plan, root_b, root_a, execute=True)

    assert not errors
    # Moved into B; only the failed job is markered, completed jobs get none.
    assert (root_b / "done" / "orca.out").exists()
    assert not (root_b / "done" / MARKER_FILENAME).exists()
    assert (root_b / "dead" / MARKER_FILENAME).exists()
    assert (root_b / "dead" / "orca.out").exists()  # contents kept
    assert not (root_b / "already_home" / MARKER_FILENAME).exists()
    # Source emptied; incomplete dropped entirely.
    assert not (root_a / "done").exists()
    assert not (root_a / "dead").exists()
    assert not (root_a / "queued").exists()
    assert not (root_b / "queued").exists()

    marker = json.loads((root_b / "dead" / MARKER_FILENAME).read_text())
    assert marker["detected_status"] == "failed"
    assert "failure_reason" in marker


def test_execute_collision_replace_and_discard(tmp_path):
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    # A newer to_run wins -> replaces B copy.
    _make_to_run(root_a, "newer_in_a", mtime=_NOW)
    jd_b = _make_to_run(root_b, "newer_in_a", mtime=_NOW - 5000)
    (jd_b / "stale.txt").write_text("old")
    os.utime(jd_b / "stale.txt", (_NOW - 5000, _NOW - 5000))
    # B completed wins -> A copy discarded.
    _make_to_run(root_a, "b_is_done", mtime=_NOW)
    _make_completed(root_b, "b_is_done", mtime=_NOW - 5000)

    jobs_a, jobs_b = _scan_both(root_a, root_b)
    plan = build_plan(jobs_a, jobs_b, drop_incomplete=False)
    counts, markers, errors = execute_plan(plan, root_b, root_a, execute=True)

    assert not errors
    assert counts[ACT_REPLACE_DEST] == 1
    assert counts[ACT_DISCARD_SRC] == 1
    assert not (root_b / "newer_in_a" / "stale.txt").exists()  # B copy replaced
    assert (root_b / "newer_in_a" / "orca.inp").exists()
    assert not (root_a / "newer_in_a").exists()
    assert not (root_a / "b_is_done").exists()
    assert not (root_b / "b_is_done" / MARKER_FILENAME).exists()


def test_marker_idempotent(tmp_path):
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    root_a.mkdir()
    jd = _make_failed(root_b, "dead")
    (jd / MARKER_FILENAME).write_text("{}")

    jobs_a, jobs_b = _scan_both(root_a, root_b)
    plan = build_plan(jobs_a, jobs_b, drop_incomplete=False)
    _counts, markers, errors = execute_plan(plan, root_b, root_a, execute=True)
    assert not errors
    assert markers == 0  # already markered, not rewritten
    assert (jd / MARKER_FILENAME).read_text() == "{}"
