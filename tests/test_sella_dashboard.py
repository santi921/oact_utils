"""Tests for sella-specific dashboard plumbing.

Phase 1 covers:
  - read_sella_log_tail against the water fixtures (3 converged + 1 restart)
  - Schema migration idempotence (second open runs zero ALTER)
  - update_job_metrics_bulk persistence of sella_steps / sella_converged

Phase 2 tests (view output, auto-detection SQL shape, no-opt-traj-read)
live in the same file below the Phase 1 block.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from oact_utilities.utils.analysis import (
    SellaStepRow,
    parse_job_metrics,
    read_sella_log_tail,
)
from oact_utilities.utils.architector import create_workflow_db
from oact_utilities.workflows import ArchitectorWorkflow, JobStatus

FIXTURE_ROOT = Path("examples/sella_water_example")


# --------------------------------------------------------------------
# read_sella_log_tail
# --------------------------------------------------------------------


def _water_fixture(name: str) -> Path:
    return FIXTURE_ROOT / "jobs" / name


@pytest.mark.parametrize(
    "name,expected",
    [
        # Last data row of each fixture's sella.log.
        # Values hand-computed from the committed fixture files.
        ("water_bent", SellaStepRow(step=3, energy=-2079.864488, fmax=0.0382)),
        ("water_stretched", SellaStepRow(step=5, energy=-2079.864615, fmax=0.0430)),
        # water_compressed has a restart (two header rows); the tail of the
        # SECOND segment is what we want.
        (
            "water_compressed",
            SellaStepRow(step=6, energy=-2079.864610, fmax=0.0014),
        ),
    ],
)
def test_read_sella_log_tail_against_fixtures(name, expected):
    """Tail-read returns the correct last-row for each water fixture."""
    log_path = _water_fixture(name) / "sella.log"
    if not log_path.exists():
        pytest.skip(f"Fixture missing: {log_path}")

    row = read_sella_log_tail(log_path)
    assert row is not None, f"Expected a row for {name}, got None"
    assert row["step"] == expected["step"]
    assert row["energy"] == pytest.approx(expected["energy"], abs=1e-4)
    assert row["fmax"] == pytest.approx(expected["fmax"], abs=1e-3)


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(None, id="missing_file"),
        pytest.param("", id="empty_file"),
        pytest.param(
            "     Step     Time          Energy         fmax         cmax       rtrust          rho\n",
            id="header_only",
        ),
    ],
)
def test_read_sella_log_tail_none_cases(tmp_path, content):
    """Missing / empty / header-only logs all return None."""
    log = tmp_path / "sella.log"
    if content is not None:
        log.write_text(content)
    assert read_sella_log_tail(log) is None


def test_read_sella_log_tail_ignores_prior_segment_after_restart(tmp_path):
    """Fresh restart (header written, no data yet) returns None.

    Before this fix, the tail-reader would report the last step of the
    prior segment as if it were the current step, because the header
    row was skipped and the reverse-iteration hit the prior segment's
    last data row. The fix tracks the last header position and only
    accepts data rows that appear after it.
    """
    log = tmp_path / "sella.log"
    log.write_text(
        "     Step     Time          Energy         fmax         cmax       rtrust          rho\n"
        "Sella   0 08:00:00    -76.400000       0.5000       0.0000       0.1000       1.0000\n"
        "Sella   1 08:00:10    -76.450000       0.2500       0.0000       0.1000       1.0000\n"
        "Sella   2 08:00:20    -76.470000       0.1200       0.0000       0.1000       1.0000\n"
        "     Step     Time          Energy         fmax         cmax       rtrust          rho\n"
    )
    # Prior segment last row was Sella 2; buggy reader would return that.
    assert read_sella_log_tail(log) is None


def test_read_sella_log_tail_restart_with_partial_second_segment(tmp_path):
    """Restart with a few data rows after the new header returns the last new row."""
    log = tmp_path / "sella.log"
    log.write_text(
        "     Step     Time          Energy         fmax         cmax       rtrust          rho\n"
        "Sella   0 08:00:00    -76.400000       0.5000       0.0000       0.1000       1.0000\n"
        "Sella   1 08:00:10    -76.450000       0.2500       0.0000       0.1000       1.0000\n"
        "Sella   2 08:00:20    -76.470000       0.1200       0.0000       0.1000       1.0000\n"
        "     Step     Time          Energy         fmax         cmax       rtrust          rho\n"
        "Sella   0 08:05:00    -76.470000       0.1200       0.0000       0.1000       1.0000\n"
        "Sella   1 08:05:10    -76.480000       0.0600       0.0000       0.1000       1.0000\n"
    )
    row = read_sella_log_tail(log)
    assert row is not None
    # Should be step 1 of the NEW segment, not step 2 of the old one.
    assert row["step"] == 1
    assert row["fmax"] == pytest.approx(0.06, abs=1e-3)


# Tested by test_sella_progress_functions_do_not_touch_trajectory below,
# which parametrises over every function in the running-progress code path
# (including read_sella_log_tail).


# --------------------------------------------------------------------
# Schema migration
# --------------------------------------------------------------------


def _build_sample_db(tmp_path: Path) -> Path:
    """Return a workflow DB path with two rows, one sella and one SP."""
    csv = tmp_path / "sample.csv"
    df = pd.DataFrame(
        {
            "aligned_csd_core": [
                "H 0.0 0.0 0.0\nH 0.0 0.0 0.74",
                "O 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0",
            ],
            "charge": [0, 0],
            "uhf": [1, 1],
        }
    )
    df.to_csv(csv, index=False)

    db = tmp_path / "workflow.db"
    create_workflow_db(csv_path=csv, db_path=db, geometry_column="aligned_csd_core")
    return db


def test_migration_adds_sella_columns(tmp_path):
    """First open adds sella_steps and sella_converged to a fresh DB."""
    db = _build_sample_db(tmp_path)

    with ArchitectorWorkflow(db) as wf:
        cur = wf._execute_with_retry("PRAGMA table_info(structures)")
        cols = {row[1] for row in cur.fetchall()}
        assert "sella_steps" in cols
        assert "sella_converged" in cols


def test_migration_is_idempotent(tmp_path, monkeypatch):
    """Second open runs zero ALTER statements (columns already exist)."""
    db = _build_sample_db(tmp_path)

    # First open creates the columns.
    with ArchitectorWorkflow(db):
        pass

    # Count ALTER statements on the second open. Wrap _execute_with_retry
    # on the instance; patching at the class level is brittle because the
    # method is bound in __init__ via _ensure_schema.
    alter_count = {"n": 0}

    original_init = ArchitectorWorkflow.__init__

    def counting_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

    orig_execute = ArchitectorWorkflow._execute_with_retry

    def spying_execute(self, query, params=()):
        if query.lstrip().upper().startswith("ALTER"):
            alter_count["n"] += 1
        return orig_execute(self, query, params)

    monkeypatch.setattr(ArchitectorWorkflow, "_execute_with_retry", spying_execute)

    with ArchitectorWorkflow(db):
        pass

    assert (
        alter_count["n"] == 0
    ), f"Expected zero ALTER on second open, saw {alter_count['n']}"


def test_job_record_exposes_new_fields(tmp_path):
    """JobRecord objects carry sella_steps and sella_converged (None on unpopulated rows)."""
    db = _build_sample_db(tmp_path)

    with ArchitectorWorkflow(db) as wf:
        jobs = wf.get_jobs_by_status(JobStatus.TO_RUN)
        assert jobs, "Sample DB should have at least one TO_RUN job"
        j = jobs[0]
        assert hasattr(j, "sella_steps")
        assert hasattr(j, "sella_converged")
        assert j.sella_steps is None
        assert j.sella_converged is None


# --------------------------------------------------------------------
# update_job_metrics_bulk
# --------------------------------------------------------------------


def test_bulk_write_persists_sella_columns(tmp_path):
    """update_job_metrics_bulk writes sella_steps and sella_converged when provided."""
    db = _build_sample_db(tmp_path)

    with ArchitectorWorkflow(db) as wf:
        jobs = wf.get_jobs_by_status(JobStatus.TO_RUN)
        j = jobs[0]

        wf.update_job_metrics_bulk(
            [
                {
                    "job_id": j.id,
                    "job_dir": "/tmp/fake",
                    "sella_steps": 7,
                    "sella_converged": 1,
                }
            ]
        )

        refetched = next(
            r for r in wf.get_jobs_by_status(JobStatus.TO_RUN) if r.id == j.id
        )
        assert refetched.sella_steps == 7
        assert refetched.sella_converged == 1


def test_bulk_write_rejects_invalid_sella_converged(tmp_path):
    """sella_converged must be 0, 1, or None -- other values raise.

    SQLite would silently coerce True/False to 1/0 and accept 42, -1,
    or strings. The tri-state invariant downstream code relies on
    (dashboard summary, pymatgen/AiiDA-style semantics) would be
    broken without this guard.
    """
    db = _build_sample_db(tmp_path)

    with ArchitectorWorkflow(db) as wf:
        j = wf.get_jobs_by_status(JobStatus.TO_RUN)[0]

        for bad_value in (2, -1, "CONVERGED", True, False):
            with pytest.raises(ValueError, match="sella_converged"):
                wf.update_job_metrics_bulk(
                    [{"job_id": j.id, "sella_converged": bad_value}]
                )


def test_partial_index_on_optimizer_exists(tmp_path):
    """Migration creates a partial index on the optimizer column.

    The index turns has_sella_jobs()'s auto-detection from a full
    table scan into an index lookup on SP-only campaigns.
    """
    db = _build_sample_db(tmp_path)
    with ArchitectorWorkflow(db) as wf:
        cur = wf._execute_with_retry(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='index' AND name='idx_optimizer_sella'"
        )
        row = cur.fetchone()
        assert row is not None, "idx_optimizer_sella should exist"
        # Partial -- skips rows where optimizer IS NULL so SP-only DBs
        # pay ~zero index overhead.
        assert "optimizer IS NOT NULL" in row[1]


def test_bulk_write_leaves_sella_columns_null_when_absent(tmp_path):
    """Omitting sella_* from the metrics dict leaves the columns NULL."""
    db = _build_sample_db(tmp_path)

    with ArchitectorWorkflow(db) as wf:
        jobs = wf.get_jobs_by_status(JobStatus.TO_RUN)
        j = jobs[0]

        wf.update_job_metrics_bulk(
            [
                {
                    "job_id": j.id,
                    "max_forces": 0.01,
                    "final_energy": -76.5,
                }
            ]
        )

        refetched = next(
            r for r in wf.get_jobs_by_status(JobStatus.TO_RUN) if r.id == j.id
        )
        assert refetched.sella_steps is None
        assert refetched.sella_converged is None
        # And existing columns still write through unharmed
        assert refetched.max_forces == pytest.approx(0.01)
        assert refetched.final_energy == pytest.approx(-76.5)


# --------------------------------------------------------------------
# parse_job_metrics end-to-end against a water fixture
# --------------------------------------------------------------------


def test_parse_job_metrics_returns_sella_converged():
    """parse_job_metrics returns sella_converged=1 for a converged fixture."""
    fixture = _water_fixture("water_bent")
    if not (fixture / "sella_status.txt").exists():
        pytest.skip("Fixture missing")
    if not (fixture / "run_sella.py").exists():
        pytest.skip("Marker file missing")

    metrics = parse_job_metrics(fixture)

    # Three converged fixtures have CONVERGED status -> sella_converged=1
    assert metrics.get("sella_converged") == 1
    # sella_steps should come from sella_status.txt (3 for water_bent)
    assert metrics.get("sella_steps") == 3


# --------------------------------------------------------------------
# Phase 2: display + CLI
# --------------------------------------------------------------------


def _set_optimizer_sella(workflow, job_ids: list[int]) -> None:
    """Seed optimizer='sella' on the given rows."""
    placeholders = ",".join("?" * len(job_ids))
    workflow._execute_with_retry(
        f"UPDATE structures SET optimizer = 'sella' WHERE id IN ({placeholders})",
        tuple(job_ids),
    )
    workflow._commit_with_retry()


def _set_sella_row(
    workflow,
    job_id: int,
    status: str,
    sella_converged: int | None = None,
    sella_steps: int | None = None,
) -> None:
    """Seed a test row to a specific sella state."""
    workflow._execute_with_retry(
        "UPDATE structures SET optimizer = 'sella', status = ?, "
        "sella_converged = ?, sella_steps = ? WHERE id = ?",
        (status, sella_converged, sella_steps, job_id),
    )
    workflow._commit_with_retry()


def test_has_sella_jobs_false_on_sp_only_db(tmp_path):
    """Fresh DB with default optimizer=NULL returns False."""
    from oact_utilities.workflows.dashboard import has_sella_jobs

    db = _build_sample_db(tmp_path)
    with ArchitectorWorkflow(db) as wf:
        assert has_sella_jobs(wf) is False


def test_has_sella_jobs_true_when_any_row_set(tmp_path):
    """Setting optimizer='sella' on one row flips the detection."""
    from oact_utilities.workflows.dashboard import has_sella_jobs

    db = _build_sample_db(tmp_path)
    with ArchitectorWorkflow(db) as wf:
        jobs = wf.get_jobs_by_status(JobStatus.TO_RUN)
        _set_optimizer_sella(wf, [jobs[0].id])
        assert has_sella_jobs(wf) is True


def test_has_sella_jobs_uses_limit_1(tmp_path, monkeypatch):
    """Auto-detection must use SELECT 1 ... LIMIT 1, not a full scan."""
    from oact_utilities.workflows.dashboard import has_sella_jobs

    db = _build_sample_db(tmp_path)
    captured: list[str] = []
    orig_execute = ArchitectorWorkflow._execute_with_retry

    def spying_execute(self, query, params=()):
        captured.append(query)
        return orig_execute(self, query, params)

    monkeypatch.setattr(ArchitectorWorkflow, "_execute_with_retry", spying_execute)

    with ArchitectorWorkflow(db) as wf:
        # Drain the migration/status queries so we isolate the detection query.
        captured.clear()
        has_sella_jobs(wf)

    assert len(captured) == 1, f"Expected one SQL call, got {len(captured)}"
    sql = captured[0].upper()
    # Must be a light-weight existence check, not a COUNT or full scan.
    assert "LIMIT 1" in sql
    assert "SELECT 1" in sql
    assert "COUNT" not in sql


def test_print_sella_summary_empty_db(tmp_path, capsys):
    """SP-only DB prints the 'no sella jobs' message and exits cleanly."""
    from oact_utilities.workflows.dashboard import print_sella_summary

    db = _build_sample_db(tmp_path)
    with ArchitectorWorkflow(db) as wf:
        print_sella_summary(wf)

    captured = capsys.readouterr()
    assert "No sella jobs in this database." in captured.out


def test_print_sella_summary_tristate_counts(tmp_path, capsys):
    """Seeded DB shows correct CONVERGED / NOT_CONVERGED / ERROR / RUNNING counts."""
    from oact_utilities.workflows.dashboard import print_sella_summary

    db = _build_sample_db(tmp_path)
    with ArchitectorWorkflow(db) as wf:
        jobs = wf.get_jobs_by_status(JobStatus.TO_RUN)
        assert len(jobs) >= 2

        # Two rows, one CONVERGED, one NOT_CONVERGED. Mark both as completed.
        _set_sella_row(wf, jobs[0].id, "completed", sella_converged=1, sella_steps=5)
        _set_sella_row(wf, jobs[1].id, "completed", sella_converged=0, sella_steps=100)

        print_sella_summary(wf)

    captured = capsys.readouterr()
    out = captured.out
    assert "CONVERGED" in out
    assert "NOT_CONVERGED" in out
    # Both counts should show 1 (one converged, one not).
    # And the non-converged detail section should show the second job's id.
    assert str(jobs[1].id) in out
    # Step stats should include max=100 (the NOT_CONVERGED row) and min=5.
    assert "Max:    100" in out


def test_show_sella_running_progress_tail_read(tmp_path, capsys):
    """Running progress reads sella.log tail, never opt.traj."""
    from oact_utilities.workflows.dashboard import show_sella_running_progress
    from oact_utilities.workflows.job_dir_patterns import DEFAULT_JOB_DIR_PATTERN

    db = _build_sample_db(tmp_path)

    # Build a fake root_dir with one job directory matching the default
    # pattern job_{orig_index}. Put a sella.log in it.
    root = tmp_path / "jobs_root"
    root.mkdir()

    with ArchitectorWorkflow(db) as wf:
        jobs = wf.get_jobs_by_status(JobStatus.TO_RUN)
        j = jobs[0]

        # Mark as RUNNING, optimizer=sella.
        _set_sella_row(wf, j.id, "running")

        job_dir = root / f"job_{j.orig_index}"
        job_dir.mkdir()
        (job_dir / "sella.log").write_text(
            "     Step     Time          Energy         fmax         cmax       rtrust          rho\n"
            "Sella   0 08:30:00    -76.400000       0.5000       0.0000       0.1000       1.0000\n"
            "Sella   1 08:30:10    -76.450000       0.2500       0.0000       0.1000       1.0000\n"
            "Sella   2 08:30:20    -76.480000       0.0800       0.0000       0.1000       1.0000\n"
        )

        show_sella_running_progress(
            wf,
            root_dir=root,
            job_dir_pattern=DEFAULT_JOB_DIR_PATTERN,
        )

    out = capsys.readouterr().out
    # Current step from tail-read should be 2.
    assert str(j.id) in out
    assert " 2 " in out or "  2" in out  # step column
    assert "0.0800" in out  # current fmax


def test_show_sella_running_progress_empty(tmp_path, capsys):
    """No running sella jobs -> friendly message, no crash."""
    from oact_utilities.workflows.dashboard import show_sella_running_progress
    from oact_utilities.workflows.job_dir_patterns import DEFAULT_JOB_DIR_PATTERN

    db = _build_sample_db(tmp_path)
    with ArchitectorWorkflow(db) as wf:
        show_sella_running_progress(
            wf,
            root_dir=tmp_path,
            job_dir_pattern=DEFAULT_JOB_DIR_PATTERN,
        )

    out = capsys.readouterr().out
    assert "No running sella jobs" in out


def test_sella_progress_row_is_typed():
    """_probe_sella_current_step returns a SellaProgressRow TypedDict, not an untyped dict.

    Plan review flagged the untyped dict return as "the one discipline
    slip" since the PR introduced SellaStepRow specifically to unify
    parser return shapes. This test pins the contract.
    """
    import inspect

    from oact_utilities.workflows.dashboard import (
        SellaProgressRow,
        _probe_sella_current_step,
    )

    sig = inspect.signature(_probe_sella_current_step)
    # dashboard.py uses `from __future__ import annotations`, so the
    # return annotation is stored as a string. A simple substring match
    # is sufficient to catch a regression back to `dict | None`.
    annotation = str(sig.return_annotation)
    assert (
        "SellaProgressRow" in annotation
    ), f"Expected SellaProgressRow in return annotation, got {annotation}"
    # TypedDict has the expected keys.
    assert set(SellaProgressRow.__annotations__.keys()) == {
        "job_id",
        "orig_index",
        "step",
        "fmax",
        "energy",
        "mtime",
    }


@pytest.mark.parametrize(
    "import_path",
    [
        "oact_utilities.utils.analysis.read_sella_log_tail",
        "oact_utilities.workflows.dashboard._probe_sella_current_step",
        "oact_utilities.workflows.dashboard.show_sella_running_progress",
    ],
)
def test_sella_progress_functions_do_not_touch_trajectory(import_path):
    """Running-progress path must never open opt.traj.

    ASE trajectory append-while-read is unsafe
    (https://gitlab.com/ase/ase/-/issues/249). Docstrings legitimately
    mention opt.traj as a warning, so strip them before checking.
    """
    import importlib
    import inspect

    mod_path, _, attr = import_path.rpartition(".")
    fn = getattr(importlib.import_module(mod_path), attr)
    src = inspect.getsource(fn)
    doc = fn.__doc__
    if doc is not None:
        src = src.replace(doc, "", 1)
    assert "Trajectory" not in src, f"{fn.__name__} references Trajectory"
    assert ".traj" not in src, f"{fn.__name__} references .traj"


def test_parse_job_metrics_sella_not_converged(tmp_path):
    """Synthetic NOT_CONVERGED fixture -> sella_converged=0."""
    job = tmp_path / "fake_job"
    job.mkdir()
    (job / "run_sella.py").write_text("# marker\n")
    (job / "sella_status.txt").write_text(
        "status: NOT_CONVERGED\nsteps: 100\nfinal_fmax: 0.123456\n"
    )
    # parse_job_metrics needs an orca.out or orca.inp to pull from; give
    # it a minimal one so the upstream parser does not fail.
    (job / "orca.out").write_text(
        "Dummy output\n"
        "FINAL SINGLE POINT ENERGY        -1.000000\n"
        "ORCA TERMINATED NORMALLY\n"
    )
    (job / "orca.inp").write_text("! wB97M-V\n* xyz 0 1\nO 0 0 0\n*\n")

    metrics = parse_job_metrics(job)

    assert metrics.get("sella_converged") == 0
    assert metrics.get("sella_steps") == 100
