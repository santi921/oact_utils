import os
import sqlite3
import time

import pytest

from oact_utilities.scripts.multi_spin import check_multi_spin as cms


def test_parse_info_from_path_spin_style():
    path = os.path.join("/root", "lotA", "categoryX", "Molecule123", "spin_5")
    info = cms.parse_info_from_path(path)
    assert info["lot"] == "lotA"
    assert info["cat"] == "categoryX"
    assert info["name"] == "Molecule123"
    assert info["spin"] == "spin_5"


def test_find_and_get_status_inserts_into_db(tmp_path, monkeypatch):
    # create sample folder structure with flux_job.flux files
    base = tmp_path
    # Molecule A with two spin dirs
    molA_spin1 = base / "lot1" / "catA" / "MolA" / "spin_1"
    molA_spin2 = base / "lot1" / "catA" / "MolA" / "spin_2"
    molB_spin1 = base / "lot2" / "catB" / "MolB" / "spin_1"
    for p in (molA_spin1, molA_spin2, molB_spin1):
        p.mkdir(parents=True, exist_ok=True)
        (p / "flux_job.flux").write_text("dummy")

    # Mock status checks to produce variety of statuses depending on path
    def fake_check_sella_complete(path):
        # mark molA spin_2 as sella_complete
        return "spin_2" in path

    def fake_check_job_termination(path):
        # molA spin_1 -> running (False/0)
        # molA spin_2 -> completed -> return True
        # molB spin_1 -> failed -> return -1
        if "MolA" in path and "spin_1" in path:
            return 0
        if "MolA" in path and "spin_2" in path:
            return 1
        if "MolB" in path and "spin_1" in path:
            return -1
        return 0

    monkeypatch.setattr(cms, "check_sella_complete", fake_check_sella_complete)
    monkeypatch.setattr(cms, "check_job_termination", fake_check_job_termination)

    processed = cms.find_and_get_status(str(base), max_depth=6, verbose=False)
    assert processed == 3

    # verify DB contents
    db_path = base / "multi_spin_jobs.sqlite3"
    assert db_path.exists()

    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM jobs")
    (count,) = c.fetchone()
    assert count == 3

    # check statuses
    c.execute("SELECT name, spin, status, note FROM jobs ORDER BY name, spin")
    rows = c.fetchall()
    # find corresponding rows
    status_map = {(r[0], r[1]): (r[2], r[3]) for r in rows}

    assert status_map.get(("MolA", "spin_1"))[0] == 0
    assert status_map.get(("MolA", "spin_2"))[0] == 1
    assert status_map.get(("MolB", "spin_1"))[0] == -1

    conn.close()


def test_find_and_get_status_prints_table(tmp_path, monkeypatch, capsys):
    base = tmp_path
    molA_spin1 = base / "lot1" / "catA" / "MolA" / "spin_1"
    molA_spin2 = base / "lot1" / "catA" / "MolA" / "spin_2"
    for p in (molA_spin1, molA_spin2):
        p.mkdir(parents=True, exist_ok=True)
        (p / "flux_job.flux").write_text("dummy")

    def fake_check_sella_complete(path):
        return "spin_2" in path

    def fake_check_job_termination(path):
        if "spin_1" in path:
            return 0
        if "spin_2" in path:
            return 1
        return 0

    monkeypatch.setattr(cms, "check_sella_complete", fake_check_sella_complete)
    monkeypatch.setattr(cms, "check_job_termination", fake_check_job_termination)

    processed = cms.find_and_get_status(
        str(base), max_depth=6, verbose=False, print_table=True
    )
    assert processed == 2
    captured = capsys.readouterr()
    assert "Full jobs table" in captured.out
    # category should be included in the summary line
    assert "Molecule: MolA (category: catA)" in captured.out
    # legend should be present
    assert "Status legend" in captured.out
    # ensure legend appears after the "Full jobs table" header (i.e., above the table)
    full_idx = captured.out.index("Full jobs table")
    assert captured.out.find("Status legend", full_idx) > full_idx


def test_find_and_get_status_detects_running(tmp_path, monkeypatch):
    base = tmp_path
    molA_spin1 = base / "lot1" / "catA" / "MolA" / "spin_1"
    molA_spin1.mkdir(parents=True, exist_ok=True)
    (molA_spin1 / "flux_job.flux").write_text("dummy")
    # create a recent flux-1.out
    out = molA_spin1 / "flux-1.out"
    out.write_text("log")
    now = time.time()
    os.utime(out, (now, now))

    def fake_check_sella_complete(path):
        return False

    def fake_check_job_termination(path):
        return 0

    monkeypatch.setattr(cms, "check_sella_complete", fake_check_sella_complete)
    monkeypatch.setattr(cms, "check_job_termination", fake_check_job_termination)

    processed = cms.find_and_get_status(str(base), max_depth=6, verbose=False)
    assert processed == 1

    db_path = base / "multi_spin_jobs.sqlite3"
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.execute("SELECT name, spin, status, note FROM jobs")
    rows = c.fetchall()
    assert len(rows) == 1
    name, spin, status, note = rows[0]
    assert name == "MolA"
    assert spin == "spin_1"
    assert status == 2
    assert note.startswith("running_recent")
    conn.close()


def test_running_age_threshold(tmp_path, monkeypatch):
    base = tmp_path
    mol = base / "lot1" / "catA" / "MolA" / "spin_1"
    mol.mkdir(parents=True, exist_ok=True)
    (mol / "flux_job.flux").write_text("dummy")
    out = mol / "flux-1.out"
    out.write_text("log")
    now = time.time()
    # set mtime to now - 10 seconds
    os.utime(out, (now - 10, now - 10))

    def fake_check_sella_complete(path):
        return False

    def fake_check_job_termination(path):
        return 0

    monkeypatch.setattr(cms, "check_sella_complete", fake_check_sella_complete)
    monkeypatch.setattr(cms, "check_job_termination", fake_check_job_termination)

    # running_age_seconds = 20 should mark it as running
    processed = cms.find_and_get_status(
        str(base), max_depth=6, verbose=False, running_age_seconds=20
    )
    assert processed == 1
    conn = sqlite3.connect(str(base / "multi_spin_jobs.sqlite3"))
    c = conn.cursor()
    c.execute("SELECT status FROM jobs")
    status = c.fetchone()[0]
    assert status == 2
    conn.close()

    # remove DB and run with small threshold so it should NOT be running
    os.remove(str(base / "multi_spin_jobs.sqlite3"))
    processed = cms.find_and_get_status(
        str(base), max_depth=6, verbose=False, running_age_seconds=5
    )
    assert processed == 1
    conn = sqlite3.connect(str(base / "multi_spin_jobs.sqlite3"))
    c = conn.cursor()
    c.execute("SELECT status FROM jobs")
    status = c.fetchone()[0]
    assert status == 0
    conn.close()


def test_validate_charge_spin_list():
    """Test validation of charge/spin lists (Issue #009)."""
    # Valid list of floats
    result = cms.validate_charge_spin_list([1.5, -0.5, 0.0], "test_field")
    assert result == [1.5, -0.5, 0.0]

    # Valid with integers (should convert to float)
    result = cms.validate_charge_spin_list([1, 2, 3], "test_field")
    assert result == [1.0, 2.0, 3.0]

    # None returns empty list
    result = cms.validate_charge_spin_list(None, "test_field")
    assert result == []

    # Length validation
    result = cms.validate_charge_spin_list([1.0, 2.0], "test_field", expected_length=2)
    assert len(result) == 2

    # Length mismatch should raise
    try:
        cms.validate_charge_spin_list([1.0, 2.0], "test_field", expected_length=3)
        pytest.fail("Should have raised ValueError")
    except ValueError as e:
        assert "length mismatch" in str(e)

    # Non-list should raise
    try:
        cms.validate_charge_spin_list("not a list", "test_field")
        pytest.fail("Should have raised ValueError")
    except ValueError as e:
        assert "must be a list" in str(e)

    # Non-numeric element should raise
    try:
        cms.validate_charge_spin_list([1.0, "invalid", 3.0], "test_field")
        pytest.fail("Should have raised ValueError")
    except ValueError as e:
        assert "must be numeric" in str(e)


def test_safe_load_json_field():
    """Test safe JSON loading with validation (Issue #009)."""
    # Valid JSON list
    result = cms.safe_load_json_field("[1.5, -0.5, 0.0]", "test_field")
    assert result == [1.5, -0.5, 0.0]

    # None returns empty list
    result = cms.safe_load_json_field(None, "test_field")
    assert result == []

    # Empty string returns empty list
    result = cms.safe_load_json_field("", "test_field")
    assert result == []

    # Invalid JSON should raise
    try:
        cms.safe_load_json_field("invalid json", "test_field")
        pytest.fail("Should have raised ValueError")
    except ValueError as e:
        assert "Invalid JSON" in str(e)

    # JSON with wrong type should raise
    try:
        cms.safe_load_json_field('{"key": "value"}', "test_field")
        pytest.fail("Should have raised ValueError")
    except ValueError as e:
        assert "must be a list" in str(e)

    # JSON with non-numeric elements should raise
    try:
        cms.safe_load_json_field('[1.0, "invalid", 3.0]', "test_field")
        pytest.fail("Should have raised ValueError")
    except ValueError as e:
        assert "must be numeric" in str(e)
