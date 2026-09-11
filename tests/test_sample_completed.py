"""Tests for sampling completed structures out of a job-directory root."""

from __future__ import annotations

import gzip
import shutil
import sqlite3
from pathlib import Path

import pytest
from ase.io import read as ase_read

from oact_utilities.utils.architector import xyz_string_to_atoms
from oact_utilities.workflows.census import BOHR_TO_ANG
from oact_utilities.workflows.sample_completed import (
    EXTRA_COLUMNS,
    REASON_METAL,
    REASON_NO_INPUT,
    REASON_NOT_COMPLETED,
    REASON_OK,
    REASON_SIZE,
    Candidate,
    Sample,
    collect_candidates,
    final_geometry,
    main,
    read_inp_geometry,
    read_xyz_geometry,
    scan_candidate,
    select_samples,
    write_extxyz,
    write_workflow_db,
)

FILES = Path(__file__).parent / "files"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OUT_DONE = "SCF CONVERGED AFTER 12 CYCLES\n****ORCA TERMINATED NORMALLY****\n"
_OUT_FAILED = "something broke\n[file orca_main] ... aborting the run\n"


def _inp(symbols: list[str], charge: int = 0, spin: int = 1) -> str:
    """Minimal ORCA input with one atom per Angstrom along x."""
    coords = "\n".join(
        f"{s}  {i:.6f}  0.000000  0.000000" for i, s in enumerate(symbols)
    )
    return f"! wB97M-V RIJCOSX\n%pal\n  nprocs 4\nend\n\n* xyz {charge} {spin}\n{coords}\n*\n"


def _xyz(symbols: list[str], shift: float, energy: float | None = None) -> str:
    """Standard XYZ with atoms shifted by ``shift`` along y, ORCA-style comment."""
    comment = "Coordinates from ORCA-job orca"
    if energy is not None:
        comment += f" E {energy:.12f}"
    lines = [str(len(symbols)), comment]
    lines += [f"  {s}  {i:.6f}  {shift:.6f}  0.000000" for i, s in enumerate(symbols)]
    return "\n".join(lines) + "\n"


_Z = {"H": 1, "C": 6, "O": 8, "F": 9, "Fe": 26, "Th": 90, "U": 92, "Np": 93, "Am": 95}


def _engrad(symbols: list[str], shift_ang: float, energy: float = -1.0) -> str:
    """ORCA .engrad with atoms at (i, 0, shift) Angstrom, written in Bohr."""
    lines = ["#", "# Number of atoms", "#", f" {len(symbols)}"]
    lines += ["#", "# The current total energy in Eh", "#", f"  {energy:.9f}"]
    lines += ["#", "# The current gradient in Eh/bohr", "#"]
    lines += ["  0.001000000"] * (3 * len(symbols))
    lines += ["#", "# The atomic numbers and current coordinates in Bohr", "#"]
    for i, s in enumerate(symbols):
        x = i / BOHR_TO_ANG
        z = shift_ang / BOHR_TO_ANG
        lines.append(f"{_Z[s]:4d}  {x:.7f} 0.0000000 {z:.7f}")
    lines.append("")
    return "\n".join(lines)


def _write_job(
    root: Path,
    name: str,
    symbols: list[str],
    charge: int = 0,
    spin: int = 1,
    out: str | None = _OUT_DONE,
    xyz: str | None = None,
    engrad: str | None = None,
    extra: dict[str, str] | None = None,
) -> Path:
    job = root / name
    job.mkdir(parents=True)
    (job / "orca.inp").write_text(_inp(symbols, charge, spin))
    if out is not None:
        (job / "orca.out").write_text(out)
    if xyz is not None:
        (job / "orca.xyz").write_text(xyz)
    if engrad is not None:
        (job / "orca.engrad").write_text(engrad)
    for filename, content in (extra or {}).items():
        (job / filename).write_text(content)
    return job


def _corpus(tmp_path: Path) -> Path:
    """Completed: U x3 (3, 5, 9 atoms), Np x2, Am x1, Fe x2. Plus failed / to_run."""
    root = tmp_path / "jobs"
    root.mkdir()
    _write_job(root, "job_1", ["U", "O", "O"], spin=1)
    _write_job(root, "job_2", ["U", "O", "O", "H", "H"], spin=3)
    _write_job(root, "job_3", ["U"] + ["F"] * 8, charge=-2, spin=1)
    _write_job(root, "job_4", ["Np", "F", "F", "F"], spin=5)
    _write_job(root, "job_5", ["Np", "O", "O"], spin=2)
    _write_job(root, "job_6", ["Am", "O"], spin=8)
    _write_job(root, "job_7", ["Fe", "O", "O", "O"], spin=5)
    _write_job(root, "job_8", ["Fe", "C", "O"], spin=1)
    _write_job(root, "job_9", ["Th", "O", "O"], out=_OUT_FAILED)
    _write_job(root, "job_10", ["Th", "O"], out=None)
    return root


# ---------------------------------------------------------------------------
# Geometry readers
# ---------------------------------------------------------------------------


def test_read_xyz_geometry_parses_orca_energy_comment(tmp_path):
    path = tmp_path / "orca.xyz"
    path.write_text(_xyz(["Am", "O"], shift=0.5, energy=-670.5))
    symbols, coords, energy = read_xyz_geometry(path)
    assert symbols == ["Am", "O"]
    assert coords == [[0.0, 0.5, 0.0], [1.0, 0.5, 0.0]]
    assert energy == pytest.approx(-670.5)


def test_read_xyz_geometry_gzipped_and_no_energy(tmp_path):
    path = tmp_path / "orca.xyz.gz"
    with gzip.open(path, "wt") as f:
        f.write(_xyz(["U", "O"], shift=0.0))
    symbols, _, energy = read_xyz_geometry(path)
    assert symbols == ["U", "O"]
    assert energy is None


def test_read_xyz_geometry_rejects_truncated_body(tmp_path):
    path = tmp_path / "orca.xyz"
    path.write_text("3\ncomment\nU 0 0 0\nO 1 0 0\n")
    assert read_xyz_geometry(path) == ([], [], None)


def test_read_inp_geometry(tmp_path):
    path = tmp_path / "orca.inp"
    path.write_text(_inp(["Np", "F", "F"], charge=0, spin=4))
    symbols, coords = read_inp_geometry(path)
    assert symbols == ["Np", "F", "F"]
    assert coords[1] == [1.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Final geometry priority
# ---------------------------------------------------------------------------


def test_final_geometry_prefers_xyz_then_engrad_then_inp(tmp_path):
    symbols = ["U", "O", "O"]
    job = _write_job(
        tmp_path,
        "job",
        symbols,
        xyz=_xyz(symbols, shift=1.0, energy=-2.0),
        engrad=_engrad(symbols, shift_ang=2.0, energy=-3.0),
    )

    source, coords, energy = final_geometry(job, symbols)
    assert source == "xyz"
    assert coords[0][1] == pytest.approx(1.0)
    # The engrad energy wins over the xyz comment when both are present.
    assert energy == pytest.approx(-3.0)

    (job / "orca.xyz").unlink()
    source, coords, energy = final_geometry(job, symbols)
    assert source == "engrad"
    assert coords[0][2] == pytest.approx(2.0)
    assert coords[1][0] == pytest.approx(1.0)
    assert energy == pytest.approx(-3.0)

    (job / "orca.engrad").unlink()
    source, coords, energy = final_geometry(job, symbols)
    assert source == "inp"
    assert coords == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    assert energy is None


def test_final_geometry_skips_source_with_mismatched_atoms(tmp_path):
    symbols = ["U", "O", "O"]
    job = _write_job(
        tmp_path,
        "job",
        symbols,
        xyz=_xyz(["U", "O"], shift=1.0),  # stale file from another structure
        engrad=_engrad(symbols, shift_ang=2.0),
    )
    source, _, _ = final_geometry(job, symbols)
    assert source == "engrad"


def test_final_geometry_ignores_trajectory_and_stray_xyz(tmp_path):
    symbols = ["U", "O", "O"]
    job = _write_job(
        tmp_path,
        "job",
        symbols,
        extra={
            "orca_trj.xyz": _xyz(symbols, shift=9.0),
            "start.xyz": _xyz(symbols, shift=8.0),
        },
    )
    source, coords, _ = final_geometry(job, symbols)
    assert source == "inp"
    assert coords[0][1] == 0.0


def test_final_geometry_uses_input_stem_xyz(tmp_path):
    """ORCA names the final xyz after the input: AmO_orca.inp -> AmO_orca.xyz."""
    job = tmp_path / "job"
    job.mkdir()
    (job / "AmO_orca.inp").write_text(_inp(["Am", "O"], spin=8))
    (job / "AmO_orca.xyz").write_text(_xyz(["Am", "O"], shift=0.3))
    source, coords, _ = final_geometry(job, ["Am", "O"])
    assert source == "xyz"
    assert coords[0][1] == pytest.approx(0.3)


def test_final_geometry_none_when_nothing_agrees(tmp_path):
    job = _write_job(tmp_path, "job", ["U", "O"])
    assert final_geometry(job, ["Np", "O"]) is None


# ---------------------------------------------------------------------------
# Candidate scan
# ---------------------------------------------------------------------------


def test_scan_candidate_reasons(tmp_path):
    root = _corpus(tmp_path)

    cand, reason = scan_candidate(root / "job_2")
    assert reason == REASON_OK
    assert cand is not None
    assert cand.symbols == ["U", "O", "O", "H", "H"]
    assert cand.natoms == 5
    assert cand.charge == 0
    assert cand.spin == 3
    assert cand.metal == "U"
    assert cand.metal_class == "actinide"
    assert cand.source_orig_index == 2

    assert scan_candidate(root / "job_9") == (None, REASON_NOT_COMPLETED)
    assert scan_candidate(root / "job_10") == (None, REASON_NOT_COMPLETED)
    assert scan_candidate(root / "job_2", min_atoms=6) == (None, REASON_SIZE)
    assert scan_candidate(root / "job_2", max_atoms=4) == (None, REASON_SIZE)
    assert scan_candidate(root / "job_2", metals=frozenset({"Np"})) == (
        None,
        REASON_METAL,
    )
    assert scan_candidate(root / "job_7", actinides_only=True) == (None, REASON_METAL)

    empty = root / "job_11"
    empty.mkdir()
    assert scan_candidate(empty) == (None, REASON_NO_INPUT)


def test_scan_candidate_filters_run_before_status(tmp_path):
    """A failed job outside the size window is counted as filtered, not failed."""
    root = _corpus(tmp_path)
    assert scan_candidate(root / "job_9", max_atoms=2) == (None, REASON_SIZE)


def test_scan_candidate_sella_requires_convergence(tmp_path):
    symbols = ["U", "O", "O"]
    converged = _write_job(
        tmp_path,
        "sella_ok",
        symbols,
        extra={
            "run_sella.py": "",
            "sella_status.txt": "status: CONVERGED\nsteps: 5\n",
        },
    )
    stalled = _write_job(
        tmp_path,
        "sella_bad",
        symbols,
        extra={
            "run_sella.py": "",
            "sella_status.txt": "status: NOT_CONVERGED\nsteps: 100\n",
        },
    )
    assert scan_candidate(converged)[1] == REASON_OK
    # orca.out says TERMINATED NORMALLY (last ORCA step), but Sella did not converge.
    assert scan_candidate(stalled) == (None, REASON_NOT_COMPLETED)


def test_collect_candidates_tallies_every_directory(tmp_path):
    root = _corpus(tmp_path)
    candidates, reasons = collect_candidates(root, workers=2)
    assert sum(reasons.values()) == 10
    assert reasons[REASON_OK] == 8
    assert reasons[REASON_NOT_COMPLETED] == 2
    assert {c.job_dir.name for c in candidates} == {f"job_{i}" for i in range(1, 9)}


def test_collect_candidates_size_and_metal_filters(tmp_path):
    root = _corpus(tmp_path)
    candidates, reasons = collect_candidates(
        root, min_atoms=3, max_atoms=5, actinides_only=True, workers=2
    )
    assert {c.job_dir.name for c in candidates} == {"job_1", "job_2", "job_4", "job_5"}
    assert reasons[REASON_SIZE] == 3  # job_3 (9 atoms), job_6 (2), job_10 (2)
    assert reasons[REASON_METAL] == 2  # both Fe jobs


def test_collect_candidates_limit(tmp_path):
    root = _corpus(tmp_path)
    _, reasons = collect_candidates(root, limit=3)
    assert sum(reasons.values()) == 3


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _cand(name: str, metal: str, natoms: int = 3) -> Candidate:
    return Candidate(
        job_dir=Path(name),
        symbols=[metal] + ["O"] * (natoms - 1),
        charge=0,
        spin=1,
        metal=metal,
        metal_class="actinide" if metal in {"U", "Np", "Am", "Th"} else "non_actinide",
        source_orig_index=None,
    )


def _fake_resolve(candidate: Candidate) -> Sample | None:
    if candidate.job_dir.name.startswith("bad"):
        return None
    coords = [[float(i), 0.0, 0.0] for i in range(candidate.natoms)]
    return Sample(**vars(candidate), coords=coords, geometry_source="test")


def test_select_samples_plain_draws_n_and_is_seeded():
    pool = [_cand(f"c{i}", "U") for i in range(20)]
    a, unresolved = select_samples(pool, 5, seed=7, resolve=_fake_resolve)
    b, _ = select_samples(pool, 5, seed=7, resolve=_fake_resolve)
    c, _ = select_samples(pool, 5, seed=8, resolve=_fake_resolve)
    assert len(a) == 5
    assert unresolved == 0
    assert [s.job_dir for s in a] == [s.job_dir for s in b]
    assert [s.job_dir for s in a] != [s.job_dir for s in c]
    assert len({s.job_dir for s in a}) == 5


def test_select_samples_none_takes_everything():
    pool = [_cand(f"c{i}", "U") for i in range(6)]
    samples, _ = select_samples(pool, None, resolve=_fake_resolve)
    assert len(samples) == 6


def test_select_samples_replaces_unresolvable_picks():
    pool = [
        _cand("bad1", "U"),
        _cand("bad2", "U"),
        _cand("ok1", "U"),
        _cand("ok2", "U"),
    ]
    samples, unresolved = select_samples(pool, 2, seed=0, resolve=_fake_resolve)
    assert {s.job_dir.name for s in samples} == {"ok1", "ok2"}
    assert unresolved == 2


def test_select_samples_even_actinides_round_robin():
    pool = (
        [_cand(f"u{i}", "U") for i in range(6)]
        + [_cand(f"np{i}", "Np") for i in range(6)]
        + [_cand(f"am{i}", "Am") for i in range(6)]
        + [_cand(f"fe{i}", "Fe") for i in range(6)]
    )
    samples, _ = select_samples(pool, 9, even_actinides=True, resolve=_fake_resolve)
    counts = {}
    for s in samples:
        counts[s.metal] = counts.get(s.metal, 0) + 1
    assert counts == {"U": 3, "Np": 3, "Am": 3}


def test_select_samples_even_actinides_redistributes_shortfall():
    pool = (
        [_cand(f"u{i}", "U") for i in range(10)]
        + [_cand(f"np{i}", "Np") for i in range(10)]
        + [_cand("am0", "Am")]
    )
    samples, _ = select_samples(pool, 9, even_actinides=True, resolve=_fake_resolve)
    counts = {}
    for s in samples:
        counts[s.metal] = counts.get(s.metal, 0) + 1
    assert len(samples) == 9
    assert counts["Am"] == 1
    assert counts["U"] + counts["Np"] == 8
    assert abs(counts["U"] - counts["Np"]) <= 1


def test_select_samples_even_actinides_exhausts_pool_short():
    pool = [_cand("u0", "U"), _cand("np0", "Np")]
    samples, _ = select_samples(pool, 5, even_actinides=True, resolve=_fake_resolve)
    assert len(samples) == 2


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _samples() -> list[Sample]:
    a = _fake_resolve(_cand("job_17", "U", natoms=3))
    b = _fake_resolve(_cand("job_5", "Np", natoms=4))
    assert a is not None and b is not None
    a.charge, a.spin, a.ref_final_energy = -1, 3, -1.5
    a.source_orig_index = 17
    b.geometry_source = "engrad"
    return [a, b]


def test_write_workflow_db_schema_and_rows(tmp_path):
    out = tmp_path / "sample.db"
    write_workflow_db(_samples(), out)

    conn = sqlite3.connect(out)
    conn.row_factory = sqlite3.Row
    cols = {r[1] for r in conn.execute("PRAGMA table_info(structures)")}
    assert set(EXTRA_COLUMNS) <= cols
    rows = conn.execute("SELECT * FROM structures ORDER BY orig_index").fetchall()
    conn.close()

    assert [r["orig_index"] for r in rows] == [0, 1]
    assert all(r["status"] == "to_run" for r in rows)
    assert all(r["job_dir"] is None for r in rows)

    first = rows[0]
    assert first["elements"] == "U;O;O"
    assert first["natoms"] == 3
    assert first["charge"] == -1
    assert first["spin"] == 3
    assert first["metal"] == "U"
    assert first["metal_class"] == "actinide"
    assert first["source_job_dir"] == "job_17"
    assert first["source_orig_index"] == 17
    assert first["ref_final_energy"] == pytest.approx(-1.5)
    assert first["n_basis"] is not None and first["n_basis"] > 0
    atoms = xyz_string_to_atoms(first["geometry"])
    assert atoms.get_chemical_symbols() == ["U", "O", "O"]
    assert atoms.get_positions()[2][0] == pytest.approx(2.0)

    assert rows[1]["geometry_source"] == "engrad"
    assert rows[1]["ref_final_energy"] is None


def test_write_extxyz_frames_and_info(tmp_path):
    out = tmp_path / "sample.xyz"
    write_extxyz(_samples(), out)
    frames = ase_read(out, index=":")
    assert len(frames) == 2
    first = frames[0]
    assert first.get_chemical_symbols() == ["U", "O", "O"]
    assert first.info["sample_index"] == 0
    assert first.info["charge"] == -1
    assert first.info["spin"] == 3
    assert first.info["metal"] == "U"
    assert first.info["source_job_dir"] == "job_17"
    assert first.info["ref_final_energy_eh"] == pytest.approx(-1.5)
    # None values are dropped rather than written as the string "None".
    assert "ref_final_energy_eh" not in frames[1].info
    assert "source_orig_index" not in frames[1].info


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_main_writes_db(tmp_path, capsys):
    root = _corpus(tmp_path)
    out = tmp_path / "sample.db"
    rc = main([str(root), "-o", str(out), "-n", "4", "--min-atoms", "3", "--seed", "1"])
    assert rc == 0
    conn = sqlite3.connect(out)
    rows = conn.execute("SELECT natoms, geometry_source FROM structures").fetchall()
    conn.close()
    assert len(rows) == 4
    assert all(n >= 3 for n, _ in rows)
    assert all(src == "inp" for _, src in rows)
    report = capsys.readouterr().out
    assert "Scanned 10 job directories" in report
    assert "Selected 4 structures" in report


def test_main_even_actinides_xyz(tmp_path):
    root = _corpus(tmp_path)
    out = tmp_path / "sample.xyz"
    rc = main([str(root), "-o", str(out), "-n", "3", "--even-actinides"])
    assert rc == 0
    frames = ase_read(out, index=":")
    assert sorted(f.info["metal"] for f in frames) == ["Am", "Np", "U"]


def test_main_metals_filter_and_shortfall_warning(tmp_path, capsys):
    root = _corpus(tmp_path)
    out = tmp_path / "sample.db"
    rc = main([str(root), "-o", str(out), "-n", "5", "--metals", "Fe"])
    assert rc == 0
    conn = sqlite3.connect(out)
    metals = {r[0] for r in conn.execute("SELECT metal FROM structures")}
    n = conn.execute("SELECT COUNT(*) FROM structures").fetchone()[0]
    conn.close()
    assert metals == {"Fe"}
    assert n == 2
    assert "only 2 of the requested 5" in capsys.readouterr().err


def test_main_no_match_returns_error(tmp_path, capsys):
    root = _corpus(tmp_path)
    out = tmp_path / "sample.db"
    rc = main([str(root), "-o", str(out), "--min-atoms", "50"])
    assert rc == 1
    assert not out.exists()
    assert "no completed structures" in capsys.readouterr().err


def test_main_refuses_to_clobber_without_overwrite(tmp_path):
    root = _corpus(tmp_path)
    out = tmp_path / "sample.db"
    out.write_text("")
    with pytest.raises(SystemExit):
        main([str(root), "-o", str(out)])
    assert main([str(root), "-o", str(out), "--overwrite"]) == 0


def test_main_rejects_unknown_output_suffix(tmp_path):
    root = _corpus(tmp_path)
    with pytest.raises(SystemExit):
        main([str(root), "-o", str(tmp_path / "sample.csv")])


def test_main_on_real_orca_fixtures(tmp_path):
    """AmO (direct ORCA OPT, .xyz present) and NpF3 (gzipped quacc, engrad only)."""
    root = tmp_path / "jobs"
    root.mkdir()
    shutil.copytree(FILES / "orca_direct_example", root / "job_1")
    shutil.copytree(FILES / "quacc_example", root / "job_2")

    out = tmp_path / "sample.db"
    assert main([str(root), "-o", str(out)]) == 0

    conn = sqlite3.connect(out)
    conn.row_factory = sqlite3.Row
    rows = {r["metal"]: r for r in conn.execute("SELECT * FROM structures")}
    conn.close()

    amo = rows["Am"]
    assert amo["geometry_source"] == "xyz"
    assert amo["spin"] == 8
    assert amo["ref_final_energy"] == pytest.approx(-670.534993289315)
    pos = xyz_string_to_atoms(amo["geometry"]).get_positions()
    assert pos[0][2] == pytest.approx(0.01412029303496, abs=1e-8)
    assert pos[1][2] == pytest.approx(1.84484970696504, abs=1e-8)

    npf3 = rows["Np"]
    assert npf3["geometry_source"] == "engrad"
    assert npf3["elements"] == "Np;F;F;F"
    assert npf3["charge"] == 0
    assert npf3["spin"] == 5
    assert npf3["n_basis"] == 225
    assert npf3["ref_final_energy"] == pytest.approx(-814.120925544656)
    pos = xyz_string_to_atoms(npf3["geometry"]).get_positions()
    assert pos[1][2] == pytest.approx(3.8938185 * BOHR_TO_ANG, abs=1e-6)
