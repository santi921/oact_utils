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


def test_generator_only_records_exception_instead_of_raising(monkeypatch, tmp_path):
    def boom(_):
        raise OSError("EIO")

    monkeypatch.setattr(emm, "find_orca_output_file", boom)
    result = emm.process_job(tmp_path, tmp_path, None, False, True, False, 6.0, True)
    assert result["generator_ok"] is False
    assert result["error"] == "generator: EIO"


def test_generator_only_honours_output_root(monkeypatch, tmp_path):
    src, mirror = tmp_path / "src", tmp_path / "mirror"
    job = src / "job_0"
    job.mkdir(parents=True)

    def write_dest(job_dir, dest_file, recompute):
        dest_file.write_text("{}")
        return True

    def no_source_write(*args, **kwargs):
        raise AssertionError("wrote into the source tree")

    monkeypatch.setattr(emm, "_generator_to_path", write_dest)
    monkeypatch.setattr(emm, "parse_generator_data", no_source_write)
    result = emm.process_job(job, src, mirror, False, True, False, 6.0, True)
    assert result["generator_ok"] is True and result["error"] is None
    assert (mirror / "job_0" / "generator_metrics.json").exists()
    assert list(job.iterdir()) == []


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
