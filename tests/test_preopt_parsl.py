"""Parsl wrapper runs the pre-opt shim before ORCA (underlying function, no Parsl runtime)."""

from __future__ import annotations

import importlib.util
import os
import signal
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from oact_utilities.workflows.submit_jobs import prepare_job_directory

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("parsl") is None, reason="parsl not installed"
)


def _job(tmp_path: Path, fake_xtb: str, on_fail: str = "fail") -> tuple[Path, dict]:
    fake_orca = tmp_path / "orca_bin"
    fake_orca.write_text("#!/bin/sh\necho ran-orca\n")
    fake_orca.chmod(0o755)
    rec = MagicMock()
    rec.id, rec.orig_index, rec.job_dir = 1, 42, None
    rec.geometry = "O 0.0 0.0 0.12\nH 0.0 0.76 -0.48\nH 0.0 -0.76 -0.48"
    rec.charge, rec.spin = 0, 1
    cfg = {
        "orca_path": str(fake_orca),
        "simple_input": "pm3",
        "preopt": "gxtb",
        "gxtb_path": fake_xtb,
        "preopt_on_fail": on_fail,
    }
    job = prepare_job_directory(rec, tmp_path / "jobs", orca_config=cfg, n_cores=1)
    return job, cfg


def _run(job: Path, cfg: dict) -> dict:
    from oact_utilities.workflows.submit_jobs import orca_job_wrapper

    return orca_job_wrapper.func(1, str(job), cfg, timeout_seconds=60)


def test_preopt_runs_before_orca(fake_xtb: str, tmp_path: Path) -> None:
    job, cfg = _job(tmp_path, fake_xtb)
    result = _run(job, cfg)
    assert result["status"] == "completed", result
    assert (job / "preopt.xyz").exists()
    assert (job / "preopt_driver.log").exists()
    assert "ran-orca" in (job / "orca.out").read_text()


def test_preopt_failure_stops_before_orca(
    fake_xtb: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_XTB_MODE", "bad_exit")
    job, cfg = _job(tmp_path, fake_xtb)
    result = _run(job, cfg)
    assert result["status"] == "failed"
    assert "Pre-optimization failed: RuntimeError" in result["error"]
    assert not (job / "orca.out").exists()


def test_preopt_failure_continue_runs_orca(
    fake_xtb: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_XTB_MODE", "bad_exit")
    job, cfg = _job(tmp_path, fake_xtb, on_fail="continue")
    result = _run(job, cfg)
    assert result["status"] == "completed", result
    assert "ran-orca" in (job / "orca.out").read_text()
    assert not (job / "preopt.xyz").exists()


def test_preopt_timeout_kills_xtb_child(
    fake_xtb: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pidfile = tmp_path / "xtb.pid"
    monkeypatch.setenv("FAKE_XTB_MODE", "hang")
    monkeypatch.setenv("FAKE_XTB_PIDFILE", str(pidfile))
    job, cfg = _job(tmp_path, fake_xtb)
    from oact_utilities.workflows.submit_jobs import orca_job_wrapper

    result = orca_job_wrapper.func(1, str(job), cfg, timeout_seconds=5)
    assert result["status"] == "failed" and "timed out" in result["error"]
    assert not (job / "orca.out").exists()
    pid = int(pidfile.read_text())
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        os.kill(pid, signal.SIGKILL)
        pytest.fail("xtb kept running after the pre-opt timeout")


def test_missing_shim_fails_cleanly(fake_xtb: str, tmp_path: Path) -> None:
    job, cfg = _job(tmp_path, fake_xtb)
    (job / "run_preopt.py").unlink()
    result = _run(job, cfg)
    assert (
        result["status"] == "failed" and "Pre-opt script not found" in result["error"]
    )
    assert not (job / "orca.out").exists()
