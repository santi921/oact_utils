"""Tests for the per-atom gradient columns in scripts/extract_populations.py."""

from __future__ import annotations

import logging
import shutil
import sqlite3
from pathlib import Path

import pytest

from oact_utilities.scripts.extract_populations import (
    _CREATE_ATOMS,
    _CREATE_STRUCTURES,
    JobRow,
    _prepare_output,
    extract_job,
    read_geometry,
)
from oact_utilities.workflows.census import parse_engrad

FILES = Path(__file__).parent / "files"
AMO_DIR = FILES / "orca_direct_example"
NPF3_DIR = FILES / "quacc_example"
LOGGER = logging.getLogger("test_extract_populations")


def _grad_triples(engrad: Path) -> list[tuple[float, float, float]]:
    flat = parse_engrad(engrad)["gradient"]
    return [tuple(flat[i : i + 3]) for i in range(0, len(flat), 3)]


@pytest.mark.parametrize(
    "job_dir, engrad_name",
    [(AMO_DIR, "AmO_orca.engrad"), (NPF3_DIR, "orca.engrad.gz")],
)
def test_read_geometry_returns_per_atom_gradient(job_dir: Path, engrad_name: str):
    symbols, coords, gradient = read_geometry(job_dir)
    assert symbols
    assert len(coords) == len(gradient) == len(symbols)
    assert gradient == _grad_triples(job_dir / engrad_name)


def test_read_geometry_without_gradient_block_keeps_coords(tmp_path: Path):
    source = (AMO_DIR / "AmO_orca.engrad").read_text().splitlines(keepends=True)
    start = next(i for i, line in enumerate(source) if "current gradient" in line)
    # The block is: header, "#", one value per line, then a closing "#".
    end = next(i for i in range(start + 2, len(source)) if source[i].strip() == "#")
    (tmp_path / "orca.engrad").write_text("".join(source[:start] + source[end + 1 :]))

    symbols, coords, gradient = read_geometry(tmp_path)
    assert symbols == ["Am", "O"]
    assert len(coords) == 2
    assert gradient == []


def test_read_geometry_missing_engrad(tmp_path: Path):
    assert read_geometry(tmp_path) == ([], [], [])


def test_extract_job_writes_gradient_columns(tmp_path: Path):
    job_dir = tmp_path / "job_1"
    shutil.copytree(AMO_DIR, job_dir)
    job = JobRow(
        id=1,
        orig_index=1,
        elements="Am;O",
        natoms=2,
        charge=0,
        spin=6,
        job_dir=str(job_dir),
        final_energy=None,
        max_forces=None,
    )
    row, atoms = extract_job(
        job, job_dir, s_squared=None, force_max=None, unzip=False, hours_cutoff=24
    )
    assert row[-1] is None, f"unexpected parse_note {row[-1]}"
    assert len(atoms) == 2
    expected = _grad_triples(job_dir / "AmO_orca.engrad")
    assert [atom[-3:] for atom in atoms] == expected

    # The tuple must fit the declared table exactly, gradient columns last.
    conn = sqlite3.connect(":memory:")
    conn.execute(_CREATE_ATOMS)
    names = [r[1] for r in conn.execute("PRAGMA table_info(atoms)")]
    assert len(names) == len(atoms[0])
    assert names[-3:] == ["grad_x", "grad_y", "grad_z"]


_OLD_ATOMS = """
CREATE TABLE atoms (
    job_id INTEGER, atom_index INTEGER, element TEXT, x REAL, y REAL, z REAL,
    mulliken_charge REAL, mulliken_spin REAL, loewdin_charge REAL, loewdin_spin REAL,
    PRIMARY KEY (job_id, atom_index)
)
"""


def _old_output(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute(_CREATE_STRUCTURES)
    conn.execute(_OLD_ATOMS)
    conn.execute("INSERT INTO structures (job_id) VALUES (1)")
    conn.commit()
    return conn


def test_prepare_output_rebuilds_pre_gradient_atoms_table(tmp_path: Path):
    conn = _old_output(tmp_path / "out.db")
    _prepare_output(conn, tmp_path / "out.db", append=False, logger=LOGGER)
    names = {r[1] for r in conn.execute("PRAGMA table_info(atoms)")}
    assert {"grad_x", "grad_y", "grad_z"} <= names


def test_prepare_output_refuses_append_into_pre_gradient_table(tmp_path: Path):
    conn = _old_output(tmp_path / "out.db")
    with pytest.raises(ValueError, match="older version"):
        _prepare_output(conn, tmp_path / "out.db", append=True, logger=LOGGER)


def test_prepare_output_keeps_current_schema_on_append(tmp_path: Path):
    conn = sqlite3.connect(str(tmp_path / "out.db"))
    _prepare_output(conn, tmp_path / "out.db", append=False, logger=LOGGER)
    conn.execute("INSERT INTO structures (job_id) VALUES (7)")
    conn.commit()
    _prepare_output(conn, tmp_path / "out.db", append=True, logger=LOGGER)
    assert conn.execute("SELECT job_id FROM structures").fetchall() == [(7,)]
