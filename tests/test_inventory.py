"""Tests for the on-disk job-directory inventory (workflows/inventory.py)."""

from __future__ import annotations

import csv
from pathlib import Path

from oact_utilities.workflows.inventory import (
    _CLASS_ESSENTIAL,
    _CLASS_OTHER,
    _CLASS_SCRATCH_BAS,
    _CLASS_SCRATCH_TMP,
    _classify_file,
    _ext_key,
    inventory_root,
    scan_job_dir,
)


def _write(path: Path, size: int) -> None:
    """Create a file of exactly ``size`` bytes."""
    path.write_bytes(b"x" * size)


def _build_corpus(root: Path) -> None:
    """Build a small synthetic corpus with two job directories.

    job_0: essential outputs + a stray scratch .tmp + a nested orca_tmp_* dir.
    job_1: essential output + a .bas scratch file + an unknown .weird file.
    """
    j0 = root / "job_0"
    j0.mkdir()
    _write(j0 / "orca.out", 1000)
    _write(j0 / "orca.inp", 200)
    _write(j0 / "orca.tmp", 500)  # scratch_tmp
    tmp_dir = j0 / "orca_tmp_abc123"
    tmp_dir.mkdir()
    _write(tmp_dir / "scratch.0", 4000)  # inside nested tmp -> scratch_tmp
    _write(tmp_dir / "scratch.1", 1000)

    j1 = root / "job_1"
    j1.mkdir()
    _write(j1 / "orca.out", 800)
    _write(j1 / "orca.bas1", 300)  # scratch_bas
    _write(j1 / "data.weird", 50)  # other


def test_ext_key_normalizes_numbered_and_gz() -> None:
    assert _ext_key("orca.tmp") == ".tmp"
    assert _ext_key("orca.tmp.3") == ".tmp"
    assert _ext_key("orca.bas") == ".bas"
    assert _ext_key("orca.bas12") == ".bas"
    assert _ext_key("core") == "core"
    assert _ext_key("core.4711") == "core"
    assert _ext_key("orca.out.gz") == ".out.gz"
    assert _ext_key("orca.gbw") == ".gbw"
    assert _ext_key("README") == "(no ext)"


def test_classify_file_buckets() -> None:
    assert _classify_file("orca.out", in_tmp_dir=False) == _CLASS_ESSENTIAL
    assert _classify_file("orca.tmp", in_tmp_dir=False) == _CLASS_SCRATCH_TMP
    assert _classify_file("orca.bas1", in_tmp_dir=False) == _CLASS_SCRATCH_BAS
    assert _classify_file("data.weird", in_tmp_dir=False) == _CLASS_OTHER
    # Anything inside an orca_tmp_* dir is scratch regardless of its own name.
    assert _classify_file("orca.out", in_tmp_dir=True) == _CLASS_SCRATCH_TMP


def test_scan_job_dir_counts_sizes_and_nested_tmp(tmp_path: Path) -> None:
    _build_corpus(tmp_path)
    inv = scan_job_dir(tmp_path / "job_0")

    # 3 top-level files + 2 inside the nested tmp dir.
    assert inv.n_files == 5
    assert inv.n_subdirs == 1
    assert inv.total_bytes == 1000 + 200 + 500 + 4000 + 1000
    assert inv.n_nested_tmp_dirs == 1
    assert inv.nested_tmp_bytes == 5000

    assert inv.class_bytes[_CLASS_ESSENTIAL] == 1200
    # stray orca.tmp (500) + both files inside the nested tmp dir (5000)
    assert inv.class_bytes[_CLASS_SCRATCH_TMP] == 5500
    assert inv.reclaimable_bytes == inv.total_bytes - 1200

    # Largest file is the 4000-byte scratch inside the nested tmp dir.
    assert inv.largest_files[0][1] == 4000


def test_inventory_root_aggregates_and_csv(tmp_path: Path) -> None:
    _build_corpus(tmp_path)
    corpus = inventory_root(tmp_path, workers=2)

    assert len(corpus.jobs) == 2
    assert corpus.total_files == 5 + 3
    assert corpus.jobs_with_nested_tmp == 1
    assert corpus.total_nested_tmp_dirs == 1
    assert corpus.class_bytes[_CLASS_SCRATCH_BAS] == 300
    assert corpus.class_bytes[_CLASS_OTHER] == 50
    # Jobs sorted largest-first; job_0 (6700 B) dominates job_1 (1150 B).
    assert corpus.jobs[0].path.name == "job_0"

    csv_path = tmp_path / "out.csv"
    from oact_utilities.workflows.inventory import _write_csv

    _write_csv(corpus, csv_path)
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 2
    job0 = next(r for r in rows if r["job_dir"] == "job_0")
    assert int(job0["total_bytes"]) == 6700
    assert int(job0["n_nested_tmp_dirs"]) == 1
    assert int(job0["essential_bytes"]) == 1200


def test_loose_files_at_root_counted_separately(tmp_path: Path) -> None:
    _build_corpus(tmp_path)
    _write(tmp_path / "submit_batch.sh", 123)  # loose file at root
    corpus = inventory_root(tmp_path, workers=2)
    assert corpus.loose_files == 1
    assert corpus.loose_bytes == 123
    assert len(corpus.jobs) == 2  # the loose file is not treated as a job dir


def test_clean_tmp_dry_run_matches_but_keeps_files(tmp_path: Path) -> None:
    _build_corpus(tmp_path)
    j0 = tmp_path / "job_0"
    corpus = inventory_root(
        tmp_path, workers=2, clean_categories=frozenset({"tmp"}), execute=False
    )
    # job_0 has a stray orca.tmp (500) + the orca_tmp_abc123/ dir (5000).
    assert corpus.clean_matched == 2
    assert corpus.clean_bytes == 5500
    assert corpus.clean_freed == 0  # dry run deletes nothing
    assert (j0 / "orca.tmp").exists()
    assert (j0 / "orca_tmp_abc123").is_dir()
    assert (j0 / "orca.out").exists()


def test_clean_tmp_execute_deletes_scratch_only(tmp_path: Path) -> None:
    _build_corpus(tmp_path)
    j0 = tmp_path / "job_0"
    j1 = tmp_path / "job_1"
    corpus = inventory_root(
        tmp_path, workers=2, clean_categories=frozenset({"tmp"}), execute=True
    )
    assert corpus.clean_freed == 5500
    # tmp scratch gone
    assert not (j0 / "orca.tmp").exists()
    assert not (j0 / "orca_tmp_abc123").exists()
    # essential + non-tmp scratch preserved
    assert (j0 / "orca.out").exists()
    assert (j0 / "orca.inp").exists()
    assert (j1 / "orca.bas1").exists()  # bas not requested
    assert (j1 / "data.weird").exists()  # unknown file never matched


def test_clean_all_execute_removes_tmp_and_bas(tmp_path: Path) -> None:
    _build_corpus(tmp_path)
    j1 = tmp_path / "job_1"
    corpus = inventory_root(
        tmp_path, workers=2, clean_categories=frozenset({"tmp", "bas"}), execute=True
    )
    # tmp (5500) + bas (300) freed; the unknown .weird file is left behind.
    assert corpus.clean_freed == 5800
    assert not (j1 / "orca.bas1").exists()
    assert (j1 / "data.weird").exists()
    assert (j1 / "orca.out").exists()
