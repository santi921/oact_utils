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


def test_read_sella_log_tail_missing_file(tmp_path):
    """Non-existent sella.log returns None without raising."""
    assert read_sella_log_tail(tmp_path / "nope.log") is None


def test_read_sella_log_tail_header_only(tmp_path):
    """A log with only the header row returns None."""
    log = tmp_path / "sella.log"
    log.write_text(
        "     Step     Time          Energy         fmax         cmax       rtrust          rho\n"
    )
    assert read_sella_log_tail(log) is None


def test_read_sella_log_tail_empty_file(tmp_path):
    """Empty file returns None."""
    log = tmp_path / "sella.log"
    log.write_text("")
    assert read_sella_log_tail(log) is None


def test_read_sella_log_tail_does_not_touch_trajectory():
    """Verify the tail-reader does not read opt.traj.

    ASE trajectory append-while-read is documented unsafe
    (https://gitlab.com/ase/ase/-/issues/249). The tail-reader must
    only read sella.log.

    Inspects the function body post-docstring via AST so docstring
    references to opt.traj (which are legitimate warnings in prose)
    do not false-trigger the guard.
    """
    import ast
    import inspect

    src = inspect.getsource(read_sella_log_tail)
    tree = ast.parse(src)
    func = tree.body[0]
    assert isinstance(func, ast.FunctionDef)

    # Strip the docstring and re-serialize the function body.
    body = func.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
    ):
        body = body[1:]
    func.body = body or [ast.Pass()]
    code_only = ast.unparse(func)

    # No Trajectory class/function references in the code path.
    assert "Trajectory" not in code_only
    # No .traj file reads either.
    assert ".traj" not in code_only


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
