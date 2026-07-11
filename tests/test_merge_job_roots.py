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


def _echo(inp: str) -> str:
    """Render input text as the INPUT FILE echo block ORCA prints in .out."""
    lines = [
        "=" * 80,
        " " * 39 + "INPUT FILE",
        "=" * 80,
        "NAME = orca.inp",
    ]
    for i, line in enumerate(inp.splitlines(), start=1):
        lines.append(f"|{i:3d}> {line}")
    lines.append(
        f"|{len(inp.splitlines()) + 1:3d}> " + " " * 26 + "****END OF INPUT****"
    )
    return "\n".join(lines) + "\n"


def _set_times(job_dir: Path, mtime: float) -> None:
    for p in job_dir.rglob("*"):
        os.utime(p, (mtime, mtime))
    os.utime(job_dir, (mtime, mtime))


def _make_completed(
    root: Path,
    name: str,
    mtime: float = _NOW,
    inp: str = _DEFAULT_INP,
    echo_inp: str | None = None,
) -> Path:
    jd = root / name
    jd.mkdir(parents=True)
    (jd / "orca.inp").write_text(inp)
    echo = _echo(echo_inp if echo_inp is not None else inp)
    (jd / "orca.out").write_text(echo + "stuff\n" * 20 + "ORCA TERMINATED NORMALLY\n")
    _set_times(jd, mtime)
    return jd


def _make_failed(
    root: Path, name: str, mtime: float = _NOW, inp: str = _DEFAULT_INP
) -> Path:
    jd = root / name
    jd.mkdir(parents=True)
    (jd / "orca.inp").write_text(inp)
    (jd / "orca.out").write_text(_echo(inp) + "stuff\n" * 20 + "aborting the run\n")
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


def test_echo_hash_and_states(tmp_path):
    root = tmp_path / "r"
    _make_completed(root, "good")
    # Empty inp but valid out with echo: hash recovered from the echo.
    jd = _make_completed(root, "empty_inp")
    (jd / "orca.inp").write_text("")
    # inp swapped in from another structure: inp/out hashes disagree.
    jd2 = _make_completed(root, "swapped", inp=_OTHER_INP, echo_inp=_DEFAULT_INP)
    # Entirely empty folder.
    (root / "hollow").mkdir()

    jobs = scan_root(root, hours_cutoff=24, workers=2, label="A")
    good, empty_inp, swapped, hollow = (
        jobs["good"],
        jobs["empty_inp"],
        jobs["swapped"],
        jobs["hollow"],
    )
    assert good.hash_source == "out"
    assert good.inp_geom_key == good.out_geom_key
    assert not good.self_inconsistent

    assert empty_inp.inp_state == "empty"
    assert empty_inp.empty_inp_otherwise_valid
    assert empty_inp.hash_source == "out"
    assert empty_inp.geom_key == good.geom_key  # recovered via echo

    assert swapped.self_inconsistent
    assert swapped.geom_key == swapped.out_geom_key  # out echo wins

    assert hollow.is_empty
    assert hollow.geom_key is None
    assert jd2.name == "swapped"  # silence unused warning


def test_empty_inp_collision_resolved_via_echo(tmp_path):
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    # A copy is complete with a good inp; B copy is complete but its inp got
    # truncated to 0 bytes -- the out echo proves same structure, so the
    # collision resolves (B newer wins here).
    _make_completed(root_a, "clash", mtime=_NOW - 5000)
    jd_b = _make_completed(root_b, "clash", mtime=_NOW)
    (jd_b / "orca.inp").write_text("")
    os.utime(jd_b / "orca.inp", (_NOW, _NOW))

    jobs_a, jobs_b = _scan_both(root_a, root_b)
    plan = build_plan(jobs_a, jobs_b, drop_incomplete=False)
    assert plan[0].action == ACT_DISCARD_SRC
    assert "same structure hash" in plan[0].reason


def test_self_inconsistent_collision_skipped(tmp_path):
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    # B's inp disagrees with its own out echo (misplaced input): never
    # auto-resolve, even though the echo matches A.
    _make_completed(root_a, "clash", mtime=_NOW)
    _make_completed(
        root_b, "clash", mtime=_NOW - 5000, inp=_OTHER_INP, echo_inp=_DEFAULT_INP
    )

    jobs_a, jobs_b = _scan_both(root_a, root_b)
    plan = build_plan(jobs_a, jobs_b, drop_incomplete=False)
    assert plan[0].action == ACT_SKIP_MISMATCH
    assert "DISAGREE in B" in plan[0].reason


def test_empty_folder_superseded(tmp_path):
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    # Non-empty side supersedes regardless of hash availability or mtime.
    _make_completed(root_a, "b_hollow", mtime=_NOW - 5000)
    (root_b / "b_hollow").mkdir(parents=True)
    (root_a / "a_hollow").mkdir(parents=True)
    _make_to_run(root_b, "a_hollow")
    (root_a / "both_hollow").mkdir()
    (root_b / "both_hollow").mkdir(parents=True)

    jobs_a, jobs_b = _scan_both(root_a, root_b)
    plan = {e.name: e for e in build_plan(jobs_a, jobs_b, drop_incomplete=False)}
    assert plan["b_hollow"].action == ACT_REPLACE_DEST
    assert plan["a_hollow"].action == ACT_DISCARD_SRC
    assert plan["both_hollow"].action == ACT_DISCARD_SRC

    counts, _markers, errors = execute_plan(
        list(plan.values()), root_b, root_a, execute=True
    )
    assert not errors
    assert (root_b / "b_hollow" / "orca.out").exists()  # A's content moved in
    assert not (root_a / "a_hollow").exists()
    assert (root_b / "a_hollow" / "orca.inp").exists()
    assert not (root_a / "both_hollow").exists()


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
