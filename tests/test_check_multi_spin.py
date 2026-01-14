import os
import sqlite3
from pathlib import Path
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
