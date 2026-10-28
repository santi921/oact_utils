"""Tests for the --job-list / --shard / --generator-only paths of extract_metrics_manual."""

import json
import shutil
import sys
from pathlib import Path

import pytest

from oact_utilities.scripts import extract_metrics_manual as emm
from oact_utilities.utils.analysis import GENERATOR_AVAILABLE

FILES = Path(__file__).parent / "files"


def _run(monkeypatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["extract_metrics_manual", *argv])
    emm.main()


def test_read_job_list_skips_comments_and_dedups(tmp_path):
    listing = tmp_path / "jobs.txt"
    listing.write_text("# header\n/b/job_2\n\n/a/job_1\n/b/job_2\n")
    assert emm._read_job_list(listing) == [Path("/a/job_1"), Path("/b/job_2")]


def test_generator_only_reports_gzip_only_dir(tmp_path):
    job = tmp_path / "job_gz"
    shutil.copytree(FILES / "quacc_example", job)
    result = emm.process_job(job, tmp_path, None, False, True, False, 6.0, True)
    assert result["generator_ok"] is False
    assert "no orca.out" in result["error"]
    assert not (job / "generator_metrics.json").exists()


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["root_dir", "--job-list", "jobs.txt"],
        ["--job-list", "jobs.txt", "--output-root", "mirror"],
        ["--job-list", "jobs.txt", "--shard", "2/2"],
    ],
)
def test_rejects_bad_arguments(monkeypatch, argv):
    with pytest.raises(SystemExit):
        _run(monkeypatch, *argv)


@pytest.mark.skipif(not GENERATOR_AVAILABLE, reason="qtaim_gen not installed")
def test_generator_only_job_list_shard(monkeypatch, tmp_path, capsys):
    jobs = []
    for i in range(4):
        job = tmp_path / f"job_{i}"
        shutil.copytree(FILES / "orca_direct_example", job)
        jobs.append(job)
    listing = tmp_path / "jobs.txt"
    listing.write_text("".join(f"{j}\n" for j in jobs))

    _run(monkeypatch, "--job-list", str(listing), "--generator-only", "--shard", "1/2")

    written = [j for j in jobs if (j / "generator_metrics.json").exists()]
    assert written == [jobs[1], jobs[3]]
    gen = json.loads((jobs[1] / "generator_metrics.json").read_text())
    assert gen["orca_parser_version"] >= 2
    assert "generator_metrics.json   : 2" in capsys.readouterr().out
