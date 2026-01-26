import os
import subprocess
import time

from oact_utilities.scripts.multi_spin import run_multi_spin as rms


def test_skip_running_respects_flux_out(tmp_path, monkeypatch):
    d = tmp_path / "jobdir"
    d.mkdir()
    # create a flux_job file so the script finds it
    (d / "flux_job.flux").write_text("dummy")

    # create some other files, including a matching flux-1.out
    (d / "other.txt").write_text("x")
    f_match = d / "flux-1.out"
    f_match.write_text("log")

    # set mtime of flux-1.out to now-30s (recent)
    now = time.time()
    os.utime(f_match, (now, now))

    # monkeypatch subprocess.run so it would be invoked if not skipped
    called = {"ran": False}

    def fake_run(cmd, shell=True):
        called["ran"] = True

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)

    launched = rms.find_and_launch_flux(
        str(tmp_path), max_depth=1, dry_run=False, skip_running=True
    )
    # since flux-1.out is recent, it should skip and not launch
    assert launched == 0
    assert not called["ran"]


def test_skip_running_ignores_non_matching_files(tmp_path, monkeypatch):
    d = tmp_path / "jobdir2"
    d.mkdir()
    (d / "flux_job.flux").write_text("dummy")

    # create a recently modified file that DOES NOT match flux-*.out
    f_other = d / "some_recent.out"
    f_other.write_text("log")
    now = time.time()
    os.utime(f_other, (now, now))

    called = {"ran": False}

    def fake_run(cmd, shell=True):
        called["ran"] = True

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)

    launched = rms.find_and_launch_flux(
        str(tmp_path), max_depth=1, dry_run=False, skip_running=True
    )
    # since no flux-*.out exists, it should NOT skip and should attempt launch
    assert launched == 1
    assert called["ran"]
