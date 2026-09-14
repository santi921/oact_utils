"""Tests for joining an extxyz with filtered_structures.csv into a labelled DB."""

from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write as ase_write

from oact_utilities.utils.architector import xyz_string_to_atoms
from oact_utilities.workflows.pull_filtered import (
    BUCKET_ALL,
    BUCKET_FOLDER,
    BUCKET_REASON,
    FORCE_CLASS_METAL,
    FORCE_CLASS_NEIGHBOR,
    FORCE_CLASS_OTHER,
    KEPT,
    NEIGHBOR_CUTOFF_ANG,
    Label,
    Reservoir,
    Row,
    build_row,
    collect_rows,
    folder_of,
    force_breakdown,
    load_labels,
    main,
    normalise_job_path,
    write_db,
)

ROOT = "/lus/eagle/projects/BLASTNet"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _frame(
    job_path: str,
    symbols: list[str],
    charge: int = 0,
    spin: int = 1,
    fmax: float = 0.3,
    energy: float = -123.5,
) -> Atoms:
    """One extxyz frame carrying the info keys build_dataset.py filters on."""
    positions = np.arange(len(symbols) * 3, dtype=float).reshape(-1, 3)
    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.calc = SinglePointCalculator(
        atoms, energy=energy, forces=np.zeros((len(symbols), 3))
    )
    atoms.info.update(
        job_path=job_path,
        charge=charge,
        spin=spin,
        fmax=fmax,
        s_squared_expectation=2.1,
        homo_lumo_gap=[1.4, 0.9],
        num_electrons_scf=42.0,
    )
    return atoms


def _write_csv(path: Path, records: list[tuple[str, str, str, str, str]]) -> Path:
    lines = ["job_path,job_id,folder,stage,reason"]
    lines += [",".join(r) for r in records]
    path.write_text("\n".join(lines) + "\n")
    return path


def _row(
    frame_index: int = 0, reason: str | None = None, folder: str = "act_531"
) -> Row:
    return Row(
        frame_index=frame_index,
        job_path=f"{ROOT}/{folder}/jobs_parsl/job_{frame_index}",
        job_id=f"job_{frame_index}",
        folder=folder,
        filtered=1 if reason else 0,
        filter_stage="quality" if reason else None,
        filter_reason=reason,
        symbols=["U", "F"],
        geometry="U 0 0 0\nF 1 1 1",
        charge=0,
        spin=4,
        metal="U",
        metal_class="actinide",
        formula="FU",
        fmax_ev_ang=0.3,
        s_squared=2.1,
        homo_lumo_gap_min=0.9,
        n_electrons_scf=42.0,
        energy_ev=-1.0,
    )


# ---------------------------------------------------------------------------
# folder_of -- must mirror build_dataset.py:_folder_of exactly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "job_path,expected",
    [
        (f"{ROOT}/act_222_santi/jobs_parsl/job_3991", "act_222_santi"),
        (f"{ROOT}/act_531_chunk12/jobs_parsl/job_1", "act_531"),
        (f"{ROOT}/nonact_531_chunk_0/jobs_parsl/job_1", "nonact_531"),
        (f"{ROOT}/entropy_grad/entropy_grad_0/jobs_parsl/job_7", "entropy_grad"),
        (f"{ROOT}/afir_v1/afir_v1_02/jobs_parsl_backup/job_2", "afir_v1"),
        (f"{ROOT}/wave1/job_5", "wave1"),
        ("", "unknown"),
        (None, "unknown"),
    ],
)
def test_folder_of(job_path, expected):
    assert folder_of(job_path) == expected


def test_normalise_job_path_strips_trailing_slash():
    assert normalise_job_path(f"{ROOT}/wave1/job_5/ ") == f"{ROOT}/wave1/job_5"


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------


def test_load_labels(tmp_path):
    csv_path = _write_csv(
        tmp_path / "f.csv",
        [
            (
                f"{ROOT}/act_531_chunk12/jobs_parsl/job_1",
                "job_1",
                "act_531",
                "quality",
                "spin contamination",
            ),
            (
                f"{ROOT}/wave1/job_5",
                "job_5",
                "wave1",
                "energy",
                "energy deviates too much from reference",
            ),
        ],
    )
    labels = load_labels(csv_path)
    assert len(labels) == 2
    label = labels[f"{ROOT}/wave1/job_5"]
    assert label == Label(
        "job_5", "wave1", "energy", "energy deviates too much from reference"
    )


def test_load_labels_rejects_csv_without_job_path(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("id,reason\n1,nope\n")
    with pytest.raises(ValueError, match="job_path"):
        load_labels(path)


# ---------------------------------------------------------------------------
# Frame -> Row
# ---------------------------------------------------------------------------


def test_build_row_labels_a_filtered_frame():
    job_path = f"{ROOT}/act_531_chunk12/jobs_parsl/job_1"
    labels = {job_path: Label("job_1", "act_531", "quality", "fmax >= 50.0")}
    row = build_row(
        _frame(job_path, ["U", "F", "F", "F"], charge=0, spin=4, fmax=61.0), 0, labels
    )
    assert row.filtered == 1
    assert (row.filter_stage, row.filter_reason) == ("quality", "fmax >= 50.0")
    assert row.folder == "act_531"
    assert (row.metal, row.metal_class) == ("U", "actinide")
    assert (row.charge, row.spin) == (0, 4)
    assert row.fmax_ev_ang == pytest.approx(61.0)
    assert row.homo_lumo_gap_min == pytest.approx(0.9)
    assert row.reason_bucket == "fmax >= 50.0"


def test_build_row_labels_an_unlisted_frame_as_kept():
    row = build_row(_frame(f"{ROOT}/wave1/job_5", ["Fe", "C", "N"]), 3, {})
    assert row.filtered == 0
    assert row.filter_stage is None and row.filter_reason is None
    assert row.job_id == "job_5"
    assert row.reason_bucket == KEPT
    assert row.metal_class == "non_actinide"


def test_build_row_geometry_is_a_headerless_block_ase_can_read():
    """A comment line in the geometry column has historically broken the readers."""
    frame = _frame(f"{ROOT}/wave1/job_5", ["U", "O", "H"])
    row = build_row(frame, 0, {})
    assert len(row.geometry.splitlines()) == 3
    atoms = xyz_string_to_atoms(row.geometry)
    assert atoms.get_chemical_symbols() == ["U", "O", "H"]
    np.testing.assert_allclose(atoms.positions, frame.positions)


def test_build_row_without_geometry():
    row = build_row(
        _frame(f"{ROOT}/wave1/job_5", ["U", "O"]), 0, {}, include_geometry=False
    )
    assert row.geometry == ""


def test_build_row_survives_a_frame_with_no_job_path():
    atoms = Atoms(symbols=["U", "F"], positions=np.zeros((2, 3)))
    row = build_row(atoms, 7, {})
    assert (row.job_path, row.job_id, row.folder) == ("", "", "unknown")
    assert row.filtered == 0
    assert row.energy_ev is None


# ---------------------------------------------------------------------------
# Reservoir
# ---------------------------------------------------------------------------


def test_reservoir_keeps_everything_without_a_cap():
    res = Reservoir(cap=None, mode=BUCKET_ALL)
    for i in range(50):
        res.add(_row(i))
    assert len(res.rows()) == 50


def test_reservoir_caps_globally_and_returns_extxyz_order():
    res = Reservoir(cap=5, mode=BUCKET_ALL, seed=0)
    for i in range(200):
        res.add(_row(i))
    rows = res.rows()
    assert len(rows) == 5
    assert [r.frame_index for r in rows] == sorted(r.frame_index for r in rows)


def test_reservoir_caps_each_reason_separately():
    res = Reservoir(cap=3, mode=BUCKET_REASON, seed=0)
    for i in range(30):
        res.add(_row(i, reason="spin contamination"))
    for i in range(30, 40):
        res.add(_row(i, reason="fmax >= 50.0"))
    for i in range(40, 100):
        res.add(_row(i))
    buckets = Counter(row.reason_bucket for row in res.rows())
    assert buckets == {"spin contamination": 3, "fmax >= 50.0": 3, KEPT: 3}


def test_reservoir_caps_each_folder_separately():
    res = Reservoir(cap=2, mode=BUCKET_FOLDER, seed=0)
    for i in range(40):
        res.add(_row(i, folder="act_531"))
    for i in range(40, 45):
        res.add(_row(i, folder="wave1"))
    res.add(_row(99, folder="afir_v1"))
    buckets = Counter(row.folder for row in res.rows())
    # A folder with fewer rows than the cap contributes everything it has.
    assert buckets == {"act_531": 2, "wave1": 2, "afir_v1": 1}


def test_reservoir_is_deterministic_for_a_seed():
    def draw():
        res = Reservoir(cap=4, mode=BUCKET_ALL, seed=7)
        for i in range(100):
            res.add(_row(i))
        return [r.frame_index for r in res.rows()]

    assert draw() == draw()


def test_reservoir_samples_the_whole_stream_not_just_the_head():
    """Algorithm R must reach past the first `cap` rows."""
    res = Reservoir(cap=10, mode=BUCKET_ALL, seed=1)
    for i in range(1000):
        res.add(_row(i))
    assert max(r.frame_index for r in res.rows()) > 10


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


@pytest.fixture
def corpus(tmp_path):
    """A four-frame extxyz with two of its jobs listed in the CSV."""
    specs = [
        (f"{ROOT}/act_531_chunk12/jobs_parsl/job_1", ["U", "F", "F", "F"], 0, 4, 61.0),
        (f"{ROOT}/act_531_chunk12/jobs_parsl/job_2", ["Np", "O", "H", "H"], 1, 5, 0.4),
        (
            f"{ROOT}/entropy_grad/entropy_grad_0/jobs_parsl/job_7",
            ["Fe", "C", "N"],
            -1,
            3,
            0.2,
        ),
        (f"{ROOT}/nonact_4_06/jobs_parsl/job_9", ["Cu", "Cl", "Cl"], 2, 2, 0.1),
    ]
    frames = [_frame(jp, syms, q, s, fmax) for jp, syms, q, s, fmax in specs]
    extxyz = tmp_path / "all_structures.extxyz"
    ase_write(str(extxyz), frames, format="extxyz")
    csv_path = _write_csv(
        tmp_path / "filtered_structures.csv",
        [
            (specs[0][0], "job_1", "act_531", "quality", "fmax >= 50.0"),
            (specs[3][0], "job_9", "nonact_4_06", "quality", "spin contamination"),
        ],
    )
    return extxyz, csv_path


def test_collect_rows_labels_every_frame(corpus):
    extxyz, csv_path = corpus
    rows, by_reason, by_folder = collect_rows(extxyz, load_labels(csv_path))
    assert len(rows) == 4
    assert sum(r.filtered for r in rows) == 2
    assert by_reason == {KEPT: 2, "fmax >= 50.0": 1, "spin contamination": 1}
    assert by_folder == {"act_531": 2, "entropy_grad": 1, "nonact_4_06": 1}


def test_collect_rows_only_filtered(corpus):
    extxyz, csv_path = corpus
    rows, _, _ = collect_rows(extxyz, load_labels(csv_path), only="filtered")
    assert [r.job_id for r in rows] == ["job_1", "job_9"]


def test_collect_rows_only_kept(corpus):
    extxyz, csv_path = corpus
    rows, _, _ = collect_rows(extxyz, load_labels(csv_path), only="kept")
    assert all(r.filtered == 0 for r in rows)
    assert len(rows) == 2


def test_collect_rows_folder_filter(corpus):
    extxyz, csv_path = corpus
    rows, _, _ = collect_rows(
        extxyz, load_labels(csv_path), folders=frozenset({"act_531"})
    )
    assert {r.folder for r in rows} == {"act_531"}
    assert len(rows) == 2


def test_collect_rows_per_reason_cap(corpus):
    extxyz, csv_path = corpus
    rows, by_reason, _ = collect_rows(extxyz, load_labels(csv_path), per_reason=1)
    assert len(rows) == 3  # two reasons + the kept bucket
    # The population count is unaffected by the sampling cap.
    assert by_reason[KEPT] == 2


def test_collect_rows_per_folder_cap(corpus):
    extxyz, csv_path = corpus
    rows, _, by_folder = collect_rows(extxyz, load_labels(csv_path), per_folder=1)
    assert Counter(r.folder for r in rows) == {
        "act_531": 1,
        "entropy_grad": 1,
        "nonact_4_06": 1,
    }
    # act_531 holds two frames; the cap does not change the population count.
    assert by_folder["act_531"] == 2


def test_collect_rows_per_folder_respects_only_filtered(corpus):
    extxyz, csv_path = corpus
    rows, _, by_folder = collect_rows(
        extxyz, load_labels(csv_path), only="filtered", per_folder=5
    )
    assert {r.folder for r in rows} == {"act_531", "nonact_4_06"}
    assert all(r.filtered == 1 for r in rows)
    assert "entropy_grad" not in by_folder


def test_collect_rows_debug_limit(corpus):
    extxyz, csv_path = corpus
    rows, _, _ = collect_rows(extxyz, load_labels(csv_path), max_frames=2)
    assert len(rows) == 2


def test_write_db_round_trip(corpus, tmp_path):
    extxyz, csv_path = corpus
    rows, _, _ = collect_rows(extxyz, load_labels(csv_path))
    out = tmp_path / "out.db"
    write_db(rows, out)

    with sqlite3.connect(out) as conn:
        conn.row_factory = sqlite3.Row
        got = {r["job_id"]: dict(r) for r in conn.execute("SELECT * FROM structures")}

    assert set(got) == {"job_1", "job_2", "job_7", "job_9"}
    filtered = got["job_1"]
    assert filtered["filtered"] == 1
    assert filtered["filter_reason"] == "fmax >= 50.0"
    assert filtered["elements"] == "U;F;F;F"
    assert filtered["natoms"] == 4
    assert filtered["metal_class"] == "actinide"
    assert filtered["status"] == "completed"
    assert filtered["n_basis"] > 0  # derived from elements by _insert_row
    # Schema units differ from the extxyz, so those columns stay NULL.
    assert filtered["final_energy"] is None and filtered["max_forces"] is None
    assert filtered["energy_ev"] == pytest.approx(-123.5)
    assert xyz_string_to_atoms(filtered["geometry"]).get_chemical_symbols() == [
        "U",
        "F",
        "F",
        "F",
    ]
    assert got["job_2"]["filtered"] == 0
    assert got["job_2"]["filter_reason"] is None


def test_write_db_replaces_an_existing_file(corpus, tmp_path):
    extxyz, csv_path = corpus
    rows, _, _ = collect_rows(extxyz, load_labels(csv_path))
    out = tmp_path / "out.db"
    write_db(rows, out)
    write_db(rows[:1], out)
    with sqlite3.connect(out) as conn:
        assert conn.execute("SELECT COUNT(*) FROM structures").fetchone()[0] == 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_main_writes_a_db(corpus, tmp_path, capsys):
    extxyz, csv_path = corpus
    out = tmp_path / "cli.db"
    assert main([str(extxyz), str(csv_path), "-o", str(out)]) == 0
    with sqlite3.connect(out) as conn:
        assert conn.execute("SELECT COUNT(*) FROM structures").fetchone()[0] == 4
    assert "filtered=2" in capsys.readouterr().out


def test_main_no_geometry(corpus, tmp_path):
    extxyz, csv_path = corpus
    out = tmp_path / "cli.db"
    assert main([str(extxyz), str(csv_path), "-o", str(out), "--no-geometry"]) == 0
    with sqlite3.connect(out) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM structures WHERE geometry != ''"
            ).fetchone()[0]
            == 0
        )


def test_main_rejects_conflicting_selectors(corpus, tmp_path):
    extxyz, csv_path = corpus
    args = [str(extxyz), str(csv_path), "-o", str(tmp_path / "x.db")]
    with pytest.raises(SystemExit):
        main(args + ["--only-filtered", "--only-kept"])
    with pytest.raises(SystemExit):
        main(args + ["--per-reason", "2", "-n", "3"])
    with pytest.raises(SystemExit):
        main(args + ["--per-reason", "2", "--per-folder", "3"])
    with pytest.raises(SystemExit):
        main(args + ["--per-folder", "2", "-n", "3"])


def test_main_per_folder(corpus, tmp_path):
    extxyz, csv_path = corpus
    out = tmp_path / "cli.db"
    assert main([str(extxyz), str(csv_path), "-o", str(out), "--per-folder", "1"]) == 0
    with sqlite3.connect(out) as conn:
        folders = [r[0] for r in conn.execute("SELECT folder FROM structures")]
    assert sorted(folders) == ["act_531", "entropy_grad", "nonact_4_06"]


def test_main_returns_nonzero_when_nothing_matches(corpus, tmp_path, capsys):
    extxyz, csv_path = corpus
    out = tmp_path / "empty.db"
    code = main([str(extxyz), str(csv_path), "-o", str(out), "--folder", "nope"])
    assert code == 1
    assert not out.exists()


# ---------------------------------------------------------------------------
# force_breakdown -- which atom carries fmax, relative to the metal centre
# ---------------------------------------------------------------------------


def _force_frame(symbols: list[str], positions, forces) -> Atoms:
    """A frame whose per-atom forces are set explicitly."""
    atoms = Atoms(symbols=symbols, positions=np.asarray(positions, dtype=float))
    atoms.calc = SinglePointCalculator(
        atoms, energy=-1.0, forces=np.asarray(forces, dtype=float)
    )
    return atoms


def test_force_breakdown_peak_on_the_metal() -> None:
    frame = _force_frame(
        ["U", "O", "C"],
        [(0, 0, 0), (2.0, 0, 0), (12.0, 0, 0)],
        [(9.0, 0, 0), (1.0, 0, 0), (2.0, 0, 0)],
    )
    out = force_breakdown(frame, "U")
    assert out["fmax_atom_class"] == FORCE_CLASS_METAL
    assert out["fmax_atom_element"] == "U"
    assert out["fmax_atom_index"] == 0
    assert out["force_max_ev_ang"] == pytest.approx(9.0)
    assert out["force_max_metal"] == pytest.approx(9.0)
    assert out["force_max_neighbor"] == pytest.approx(1.0)
    assert out["force_max_other"] == pytest.approx(2.0)
    assert out["n_metal_neighbors"] == 1


def test_force_breakdown_peak_on_a_neighbor() -> None:
    frame = _force_frame(
        ["U", "O", "C"],
        [(0, 0, 0), (2.0, 0, 0), (12.0, 0, 0)],
        [(1.0, 0, 0), (7.0, 0, 0), (2.0, 0, 0)],
    )
    out = force_breakdown(frame, "U")
    assert out["fmax_atom_class"] == FORCE_CLASS_NEIGHBOR
    assert out["fmax_atom_element"] == "O"
    assert out["force_max_neighbor"] == pytest.approx(7.0)


def test_force_breakdown_peak_on_a_distant_atom() -> None:
    frame = _force_frame(
        ["U", "O", "C"],
        [(0, 0, 0), (2.0, 0, 0), (12.0, 0, 0)],
        [(1.0, 0, 0), (2.0, 0, 0), (8.0, 0, 0)],
    )
    out = force_breakdown(frame, "U")
    assert out["fmax_atom_class"] == FORCE_CLASS_OTHER
    assert out["fmax_atom_element"] == "C"
    assert out["force_max_other"] == pytest.approx(8.0)


def test_force_breakdown_uses_norms_not_components() -> None:
    """A 3-4-5 vector on the neighbour beats a larger single component."""
    frame = _force_frame(
        ["U", "O"],
        [(0, 0, 0), (2.0, 0, 0)],
        [(4.5, 0, 0), (3.0, 4.0, 0)],
    )
    out = force_breakdown(frame, "U")
    assert out["force_max_neighbor"] == pytest.approx(5.0)
    assert out["fmax_atom_class"] == FORCE_CLASS_NEIGHBOR


def test_force_breakdown_cutoff_moves_atoms_between_classes() -> None:
    positions = [(0, 0, 0), (3.0, 0, 0)]
    forces = [(1.0, 0, 0), (6.0, 0, 0)]
    frame = _force_frame(["U", "O"], positions, forces)

    near = force_breakdown(frame, "U", cutoff=4.0)
    assert near["fmax_atom_class"] == FORCE_CLASS_NEIGHBOR
    assert near["n_metal_neighbors"] == 1

    far = force_breakdown(frame, "U", cutoff=2.0)
    assert far["fmax_atom_class"] == FORCE_CLASS_OTHER
    assert far["n_metal_neighbors"] == 0
    assert far["force_max_neighbor"] is None


def test_force_breakdown_without_a_metal_is_all_other() -> None:
    frame = _force_frame(
        ["C", "O"], [(0, 0, 0), (1.2, 0, 0)], [(1.0, 0, 0), (3.0, 0, 0)]
    )
    out = force_breakdown(frame, None)
    assert out["fmax_atom_class"] == FORCE_CLASS_OTHER
    assert out["force_max_metal"] is None
    assert out["force_max_neighbor"] is None
    assert out["force_max_other"] == pytest.approx(3.0)
    assert out["n_metal_neighbors"] == 0


def test_force_breakdown_without_forces_is_all_none() -> None:
    frame = Atoms(symbols=["U", "O"], positions=[(0, 0, 0), (2.0, 0, 0)])
    out = force_breakdown(frame, "U")
    assert set(out.values()) == {None}


def test_build_row_records_the_force_breakdown() -> None:
    frame = _force_frame(
        ["U", "O", "C"],
        [(0, 0, 0), (2.0, 0, 0), (12.0, 0, 0)],
        [(1.0, 0, 0), (9.0, 0, 0), (2.0, 0, 0)],
    )
    frame.info["job_path"] = f"{ROOT}/act_531/jobs_parsl/job_1"
    row = build_row(frame, 0, {})
    assert row.fmax_atom_class == FORCE_CLASS_NEIGHBOR
    assert row.fmax_atom_element == "O"
    assert row.neighbor_cutoff_ang == NEIGHBOR_CUTOFF_ANG


def test_write_db_round_trips_the_force_columns(tmp_path: Path) -> None:
    frame = _force_frame(
        ["U", "O", "C"],
        [(0, 0, 0), (2.0, 0, 0), (12.0, 0, 0)],
        [(1.0, 0, 0), (9.0, 0, 0), (2.0, 0, 0)],
    )
    frame.info["job_path"] = f"{ROOT}/act_531/jobs_parsl/job_1"
    out = tmp_path / "forces.db"
    write_db([build_row(frame, 0, {})], out)

    with sqlite3.connect(out) as conn:
        record = conn.execute(
            "SELECT fmax_atom_class, fmax_atom_element, fmax_atom_index, "
            "force_max_ev_ang, force_max_metal, force_max_neighbor, "
            "force_max_other, n_metal_neighbors, neighbor_cutoff_ang "
            "FROM structures"
        ).fetchone()

    assert record[0] == FORCE_CLASS_NEIGHBOR
    assert record[1] == "O"
    assert record[2] == 1
    assert record[3] == pytest.approx(9.0)
    assert record[4] == pytest.approx(1.0)
    assert record[5] == pytest.approx(9.0)
    assert record[6] == pytest.approx(2.0)
    assert record[7] == 1
    assert record[8] == pytest.approx(NEIGHBOR_CUTOFF_ANG)
