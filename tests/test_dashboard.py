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
