import os
import time

from oact_utilities.scripts.multi_spin.multi_spin_from_converged_calcs import (
    wrapper_write_job_folder,
)


def test_wrapper_skips_when_recent_flux_out(tmp_path, monkeypatch, capsys):
    out = tmp_path / "job"
    out.mkdir()
    # create flux_job and a matching flux-1.out
    (out / "flux_job.flux").write_text("x")
    f = out / "flux-1.out"
    f.write_text("log")
    now = time.time()
    os.utime(f, (now, now))

    called = {"wrote": False}

    # monkeypatch write_orca_inputs (should not be called)
    def fake_write_orca_inputs(**kwargs):
        called["wrote"] = True

    monkeypatch.setattr(
        "oact_utilities.scripts.multi_spin.multi_spin_from_converged_calcs.write_orca_inputs",
        fake_write_orca_inputs,
    )

    wrapper_write_job_folder(str(out), atoms=None, tf_sella=False, skip_running=True)
    captured = capsys.readouterr()
    assert "Skipping" in captured.out
    assert not called["wrote"]


def test_wrapper_does_not_skip_for_non_matching_recent_files(tmp_path, monkeypatch):
    out = tmp_path / "job2"
    out.mkdir()
    (out / "flux_job.flux").write_text("x")
    f = out / "recent_other.out"
    f.write_text("log")
    now = time.time()
    os.utime(f, (now, now))

    called = {"wrote": False}

    def fake_write_orca_inputs(**kwargs):
        called["wrote"] = True

    monkeypatch.setattr(
        "oact_utilities.scripts.multi_spin.multi_spin_from_converged_calcs.write_orca_inputs",
        fake_write_orca_inputs,
    )

    # call wrapper; since no flux-*.out exists, it should proceed and call write_orca_inputs
    wrapper_write_job_folder(str(out), atoms=None, tf_sella=False, skip_running=True)
    assert called["wrote"]
