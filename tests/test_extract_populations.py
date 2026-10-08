"""Tests for scripts/extract_populations.py."""

from __future__ import annotations

import logging
import shutil
import sqlite3
from pathlib import Path

import pytest

from oact_utilities.scripts.extract_populations import (
    _CREATE_ATOMS,
    _CREATE_STRUCTURES,
    _SOURCE_MARKER,
    JobRow,
    _prepare_output,
    copy_job_dirs,
    extract_job,
    extract_populations,
    print_summary,
    read_completed_jobs,
    read_geometry,
    select_jobs,
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


# --- copy_job_dirs ---------------------------------------------------------


def _job_dir(root: Path, name: str, marker: str) -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / "orca.out").write_text(marker)
    return path


def test_copy_keeps_same_named_jobs_apart(tmp_path: Path):
    a = _job_dir(tmp_path / "chunk00", "job_12", "from chunk00")
    b = _job_dir(tmp_path / "chunk01", "job_12", "from chunk01")
    dest = copy_job_dirs([(1, a), (2, b)], tmp_path / "copies", logger=LOGGER)

    assert dest[1] != dest[2]
    for job_id, src in ((1, a), (2, b)):
        copy = Path(dest[job_id])
        assert (copy / "orca.out").read_text() == (src / "orca.out").read_text()
        assert (copy / _SOURCE_MARKER).read_text().strip() == str(src)


def test_copy_does_not_reuse_another_sources_copy(tmp_path: Path):
    a = _job_dir(tmp_path / "chunk00", "job_12", "from chunk00")
    b = _job_dir(tmp_path / "chunk01", "job_12", "from chunk01")
    first = copy_job_dirs([(1, a)], tmp_path / "copies", logger=LOGGER)
    second = copy_job_dirs([(2, b)], tmp_path / "copies", logger=LOGGER)

    assert Path(second[2]).name == "job_12__id2"
    assert (Path(first[1]) / "orca.out").read_text() == "from chunk00"
    assert (Path(second[2]) / "orca.out").read_text() == "from chunk01"


def test_copy_overwrite_replaces_instead_of_merging(tmp_path: Path):
    src = _job_dir(tmp_path / "jobs", "job_1", "out")
    dest = Path(copy_job_dirs([(1, src)], tmp_path / "copies", logger=LOGGER)[1])
    (dest / "stale.tmp").write_text("left over from an earlier copy")

    copy_job_dirs([(1, src)], tmp_path / "copies", logger=LOGGER)
    assert (dest / "stale.tmp").exists(), "without overwrite the copy is left alone"

    copy_job_dirs([(1, src)], tmp_path / "copies", overwrite=True, logger=LOGGER)
    assert not (dest / "stale.tmp").exists()
    assert (dest / "orca.out").read_text() == "out"


def test_copy_skip_scratch_matches_clean(tmp_path: Path):
    src = _job_dir(tmp_path / "jobs", "job_1", "out")
    (src / "orca.tmp").write_text("x")
    (src / "core").write_text("x")
    (src / "orca.bas").write_text("x")
    (src / "orca_tmp_abc").mkdir()
    (src / "orca_tmp_abc" / "f").write_text("x")
    # clean.py treats these as data: the scratch-dir regex never matches a
    # file, and ^core$ never matches a directory.
    (src / "orca_tmp_notadir").write_text("keep")
    (src / "sub" / "core").mkdir(parents=True)
    (src / "sub" / "core" / "f").write_text("keep")

    dest = Path(
        copy_job_dirs(
            [(1, src)], tmp_path / "copies", skip_scratch=True, logger=LOGGER
        )[1]
    )
    kept = {p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()}
    assert kept == {"orca.out", "orca_tmp_notadir", "sub/core/f", _SOURCE_MARKER}


# --- reading and selecting -------------------------------------------------


def _workflow_db(path: Path, rows: list[tuple]) -> Path:
    """Minimal workflow DB: (id, orig_index, job_dir, status, spin, elements, s2)."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE structures (id INTEGER PRIMARY KEY, orig_index INTEGER, "
        "elements TEXT, natoms INTEGER, charge INTEGER, spin INTEGER, job_dir TEXT, "
        "final_energy REAL, max_forces REAL, status TEXT, s_squared REAL, "
        "force_max REAL)"
    )
    conn.executemany(
        "INSERT INTO structures (id, orig_index, job_dir, status, spin, elements, "
        "natoms, charge, s_squared) VALUES (?, ?, ?, ?, ?, ?, 2, 0, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    return path


def test_read_completed_jobs_is_ordered_by_id(tmp_path: Path):
    db = _workflow_db(
        tmp_path / "wf.db",
        [(i, 40 - i, None, "completed", 1, "H;H", None) for i in (1, 2, 3)],
    )
    # An index on (status, orig_index) lets SQLite answer WHERE status = ? in
    # orig_index order, which runs opposite to id here.
    conn = sqlite3.connect(db)
    conn.execute("CREATE INDEX idx_status_orig ON structures(status, orig_index)")
    conn.commit()
    conn.close()
    assert [job.id for job in read_completed_jobs(db)] == [1, 2, 3]


@pytest.mark.parametrize(
    "flags",
    [
        {"over_cutoff": True},
        {"min_contamination": 0.1},
        {"max_contamination": 5.0},
        {"min_force_ev_ang": 1.0},
    ],
)
def test_filters_count_missing_scalars_as_ungraded(flags: dict, caplog):
    job = JobRow(1, 1, "U;O", 2, 0, 3, None, None, None)
    options = {
        "min_contamination": None,
        "max_contamination": None,
        "over_cutoff": False,
        "min_force_ev_ang": None,
        **flags,
    }
    caplog.set_level(logging.INFO, logger=LOGGER.name)
    kept = select_jobs(
        [job],
        {1: {"s_squared": None, "force_max": None}},
        n_samples=None,
        select="random",
        seed=0,
        failing_quality=None,
        force_thresh_ev_ang=50.0,
        logger=LOGGER,
        **options,
    )
    assert kept == []
    assert "1 jobs skipped for want of a scalar" in caplog.text


def test_summary_counts_deviation_equal_to_cutoff(tmp_path: Path, capsys):
    out = tmp_path / "out.db"
    conn = sqlite3.connect(out)
    _prepare_output(conn, out, append=False, logger=LOGGER)
    conn.execute(
        "INSERT INTO structures (job_id, spin_contamination, contamination_cutoff) "
        "VALUES (1, 0.5, 0.5), (2, 0.4, 0.5)"
    )
    conn.commit()
    conn.close()
    print_summary(out)
    assert "at or over element-dependent cutoff: 1" in capsys.readouterr().out


# --- --append --------------------------------------------------------------


def test_append_clears_stale_atoms_and_keeps_copied_to(tmp_path: Path):
    job_dir = tmp_path / "jobs" / "job_1"
    shutil.copytree(AMO_DIR, job_dir)
    db = _workflow_db(
        tmp_path / "wf.db", [(1, 1, str(job_dir), "completed", 6, "Am;O", None)]
    )
    out = tmp_path / "out.db"

    extract_populations(
        db, out, n_samples=None, copy_jobs=tmp_path / "copies", logger=LOGGER
    )
    conn = sqlite3.connect(out)
    assert (
        conn.execute("SELECT COUNT(*) FROM atoms WHERE job_id = 1").fetchone()[0] == 2
    )
    copied_to = conn.execute("SELECT copied_to FROM structures").fetchone()[0]
    assert copied_to
    conn.close()

    # The job loses its populations; re-extract it into the same output. Keep
    # only the engrad: this fixture's AmO_orca_atom95.out is a full ORCA
    # output too, and the parser falls back to it once orca.out is gone.
    for path in job_dir.iterdir():
        if path.name != "AmO_orca.engrad":
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    extract_populations(
        db, out, n_samples=None, append=True, recompute=True, logger=LOGGER
    )

    conn = sqlite3.connect(out)
    assert (
        conn.execute("SELECT COUNT(*) FROM atoms WHERE job_id = 1").fetchone()[0] == 0
    )
    note, kept_copy = conn.execute(
        "SELECT parse_note, copied_to FROM structures WHERE job_id = 1"
    ).fetchone()
    conn.close()
    assert note == "no_populations"
    assert kept_copy == copied_to
