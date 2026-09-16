"""Tests for dashboard CLI behavior."""

from __future__ import annotations

import sys

import pytest


@pytest.mark.parametrize("scheduler", ["slurm", "pbspro"])
def test_dashboard_recover_orphans_dispatches_supported_scheduler(
    monkeypatch, tmp_path, scheduler
):
    """--recover-orphans should dispatch Slurm and PBS Pro symmetrically."""
    from oact_utilities.workflows import dashboard as dash

    captured: dict[str, object] = {}

    class DummyWorkflow:
        def __init__(self, db_path):
            captured["db_path"] = db_path

        def count_by_status(self):
            return {}

        def close(self):
            captured["closed"] = True

    def fake_recover_orphaned_jobs(workflow, scheduler, **kwargs):
        captured["workflow"] = workflow
        captured["scheduler"] = scheduler
        captured["kwargs"] = kwargs
        return {
            "recovered": 0,
            "completed": 0,
            "failed": 0,
            "reset": 0,
            "dead_jobs": 0,
            "skipped": 0,
        }

    monkeypatch.setattr(dash, "ArchitectorWorkflow", DummyWorkflow)
    monkeypatch.setattr(dash, "recover_orphaned_jobs", fake_recover_orphaned_jobs)
    monkeypatch.setattr(dash, "print_summary", lambda workflow: None)
    monkeypatch.setattr(dash, "print_progress_bar", lambda *args, **kwargs: None)
    monkeypatch.setattr(dash, "finish_wandb_run", lambda run: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dashboard.py",
            str(tmp_path / "workflow.db"),
            "--recover-orphans",
            "--scheduler",
            scheduler,
        ],
    )

    dash.main()

    assert captured["db_path"] == str(tmp_path / "workflow.db")
    assert captured["scheduler"] == scheduler
    assert captured["kwargs"] == {
        "hours_cutoff": 24,
        "verbose": False,
        "workers": 4,
    }
    assert captured["closed"] is True


# ---------------------------------------------------------------------------
# Quality readout
# ---------------------------------------------------------------------------


def _quality_db(tmp_path, rows):
    """Build a workflow DB of completed jobs carrying the given quality scalars.

    Each entry in *rows* is ``(elements, spin, quality_kwargs)``.
    """
    import sqlite3

    from oact_utilities.workflows.architector_workflow import (
        ArchitectorWorkflow,
        JobStatus,
    )

    db_path = tmp_path / "wf.db"
    conn = sqlite3.connect(db_path)
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
            wall_time REAL,
            n_cores INTEGER,
            error_message TEXT,
            fail_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    for i, (elements, spin, _) in enumerate(rows):
        conn.execute(
            "INSERT INTO structures (orig_index, elements, natoms, status, "
            "charge, spin) VALUES (?, ?, ?, ?, ?, ?)",
            (i, elements, len(elements.split(";")), JobStatus.COMPLETED.value, 0, spin),
        )
    conn.commit()
    conn.close()

    workflow = ArchitectorWorkflow(str(db_path))  # migration adds the columns
    for job_id, (_, _, quality) in enumerate(rows, start=1):
        workflow.update_job_metrics(job_id, **quality)
    return workflow


def test_migration_adds_quality_columns(tmp_path):
    """Opening a pre-quality database backfills the schema."""
    workflow = _quality_db(tmp_path, [("Am;O", 8, {})])
    cur = workflow._execute_with_retry("PRAGMA table_info(structures)")
    cols = {row[1] for row in cur.fetchall()}
    workflow.close()

    assert {
        "force_max",
        "num_electrons_scf",
        "s_squared",
        "n_alpha",
        "n_beta",
        "homo_lumo_gap_alpha",
        "homo_lumo_gap_beta",
        "exchange_deviation",
        "orca_parser_version",
    } <= cols


def test_update_job_metrics_writes_quality_scalars(tmp_path):
    workflow = _quality_db(
        tmp_path,
        [
            (
                "Am;O",
                8,
                {"force_max": 0.001, "s_squared": 15.8, "orca_parser_version": 2},
            )
        ],
    )
    cur = workflow._execute_with_retry(
        "SELECT force_max, s_squared, orca_parser_version FROM structures"
    )
    row = cur.fetchone()
    workflow.close()

    assert tuple(row) == (0.001, 15.8, 2)


def test_update_job_metrics_rejects_unknown_keyword(tmp_path):
    workflow = _quality_db(tmp_path, [("Am;O", 8, {})])
    with pytest.raises(TypeError, match="unexpected keyword"):
        workflow.update_job_metrics(1, s_squarred=1.0)
    workflow.close()


def test_print_quality_summary_reports_both_distributions(tmp_path, capsys):
    """Forces in eV/A and spin contamination, with the over-cutoff counts."""
    from oact_utilities.workflows.census import EH_BOHR_TO_EV_ANG
    from oact_utilities.workflows.dashboard import print_quality_summary

    over = 60.0 / EH_BOHR_TO_EV_ANG  # 60 eV/A, above the 50 default
    under = 1.0 / EH_BOHR_TO_EV_ANG
    workflow = _quality_db(
        tmp_path,
        [
            # Am, mult 8 -> S(S+1) = 15.75. Deviation 0.05, under the 1.1 cutoff.
            ("Am;O", 8, {"force_max": under, "s_squared": 15.80}),
            # Deviation 2.0, over the cutoff, and fmax over the threshold.
            ("Am;O", 8, {"force_max": over, "s_squared": 17.75}),
        ],
    )
    print_quality_summary(workflow)
    workflow.close()

    out = capsys.readouterr().out
    assert "Max force per atom" in out
    assert "Above 50 eV/A: 1 (50.00%)" in out
    assert "Over cutoff: 1 (50.00%)" in out


def test_print_quality_summary_counts_ungraded(tmp_path, capsys):
    from oact_utilities.workflows.dashboard import print_quality_summary

    workflow = _quality_db(
        tmp_path,
        [("Am;O", 8, {"force_max": 0.001}), ("Am;O", 8, {"s_squared": 15.8})],
    )
    print_quality_summary(workflow)
    workflow.close()

    out = capsys.readouterr().out
    assert "1 without force_max, 1 without s_squared" in out


def test_print_quality_summary_flags_v1_parser_rows(tmp_path, capsys):
    from oact_utilities.workflows.dashboard import print_quality_summary

    workflow = _quality_db(
        tmp_path,
        [("Am;O", 8, {"s_squared": 15.8, "orca_parser_version": 1})],
    )
    print_quality_summary(workflow)
    workflow.close()

    assert "orca_parser_version 1" in capsys.readouterr().out


def test_print_quality_summary_splits_by_metal_class(tmp_path, capsys):
    from oact_utilities.workflows.dashboard import print_quality_summary

    workflow = _quality_db(
        tmp_path,
        [
            ("Am;O", 8, {"s_squared": 15.80}),
            # Fe gets the stricter 0.5 cutoff; deviation 0.6 clears it.
            ("Fe;O", 8, {"s_squared": 16.35}),
        ],
    )
    print_quality_summary(workflow)
    workflow.close()

    out = capsys.readouterr().out
    assert "actinide" in out
    assert "non_actinide" in out
