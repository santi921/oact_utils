"""Tests for the DB-free job-directory census."""

from __future__ import annotations

import csv
import gzip
import sqlite3
from pathlib import Path

import pytest

from oact_utilities.workflows.census import (
    _FIELD_NAMES,
    CONV_NORMAL,
    EH_BOHR_TO_EV_ANG,
    _parse_orig_index,
    force_stats,
    hill_formula,
    metal_class,
    parse_engrad,
    parse_inp,
    pick_metal,
    run_census,
)

FILES = Path(__file__).parent / "files"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_job(
    root: Path,
    name: str,
    inp: str | None = None,
    out: str | None = None,
    engrad: str | None = None,
    extra: dict[str, str] | None = None,
) -> Path:
    """Create a synthetic job directory with the given file contents."""
    job = root / name
    job.mkdir(parents=True)
    if inp is not None:
        (job / "orca.inp").write_text(inp)
    if out is not None:
        (job / "orca.out").write_text(out)
    if engrad is not None:
        (job / "orca.engrad").write_text(engrad)
    for filename, content in (extra or {}).items():
        (job / filename).write_text(content)
    return job


_INP_AMO = """! wB97M-V RIJCOSX NoUseSym DEFGRID2 OPT
%pal
  nprocs 6
end
%basis
  NewGTO Am "ma-def-TZVP" end
end

* xyz -1 8
Am	0.000000	0.000000	0.000000
O	0.000000	0.000000	1.858970
*
"""

_OUT_DONE = "SCF CONVERGED AFTER 12 CYCLES\n****ORCA TERMINATED NORMALLY****\n"
_OUT_FAILED = "something broke\n[file orca_main] ... aborting the run\n"


def _engrad(energy: float, gradient: list[float], zs_coords: list[tuple]) -> str:
    """Build a minimal but format-faithful ORCA .engrad file."""
    lines = [
        "#",
        "# Number of atoms",
        "#",
        f" {len(zs_coords)}",
        "#",
        "# The current total energy in Eh",
        "#",
        f"  {energy:.9f}",
        "#",
        "# The current gradient in Eh/bohr",
        "#",
    ]
    lines += [f"  {g:.9f}" for g in gradient]
    lines += [
        "#",
        "# The atomic numbers and current coordinates in Bohr",
        "#",
    ]
    for z, x, y, zc in zs_coords:
        lines.append(f"{z:4d}  {x:.6f} {y:.6f} {zc:.6f}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Element helpers
# ---------------------------------------------------------------------------


def test_metal_class_two_way():
    assert metal_class("U") == "actinide"
    assert metal_class("Cm") == "actinide"
    assert metal_class("Po") == "non_actinide"
    assert metal_class("Fe") == "non_actinide"
    assert metal_class(None) is None


def test_pick_metal_prefers_actinide_over_heavier_ligand():
    # Bi (Z=83) is heavier than Th (Z=90)? No -- but Bi outranks every d-block
    # metal by Z, so a plain highest-Z scan would mis-pick it as the centre.
    assert pick_metal(["Fe", "Bi", "N", "C", "H"]) == "Fe"
    assert pick_metal(["U", "Bi", "O"]) == "U"
    assert pick_metal(["Po", "Fe", "O"]) == "Po"


def test_pick_metal_none_for_organics():
    assert pick_metal(["C", "H", "H", "O"]) is None
    assert pick_metal([]) is None


def test_pick_metal_highest_z_within_tier():
    assert pick_metal(["Th", "U", "O"]) == "U"


def test_hill_formula():
    assert hill_formula(["C", "H", "H", "O"]) == "CH2O"
    assert hill_formula(["Am", "O"]) == "AmO"
    assert hill_formula(["Np", "F", "F", "F"]) == "F3Np"
    assert hill_formula([]) == ""


@pytest.mark.parametrize(
    "name,expected",
    [
        ("job_17", 17),
        ("host1_job_17", 17),
        ("campaign_job_17_extra", 17),
        ("AmO_0_1_42", 42),
        ("no_digits_here", None),
        # Real corpus naming: {host}_{formula}_q{charge}_m{spin}_idx{orig_index}
        ("barfoot_BN4C6H8O2_q2_m1_idx8761", 8761),
        ("carpenter_UO2C4H6_q0_m5_idx123", 123),
        # Same pattern without the idx field: the trailing number is the spin,
        # not an index, so guessing it would corrupt a join back to the DB.
        ("BN4C6H8O2_q2_m1", None),
        ("AmO_q0_m8", None),
        ("UO2_q-1", None),
    ],
)
def test_parse_orig_index(name, expected):
    assert _parse_orig_index(name) == expected


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------


def test_parse_inp(tmp_path):
    inp = tmp_path / "orca.inp"
    inp.write_text(_INP_AMO)
    parsed = parse_inp(inp)
    assert parsed["symbols"] == ["Am", "O"]
    assert parsed["charge"] == -1
    assert parsed["spin"] == 8
    assert parsed["nprocs_requested"] == 6
    assert parsed["functional"] == "wB97M-V"
    assert "DEFGRID2" in parsed["simple_input"]


def test_parse_inp_gzipped(tmp_path):
    inp = tmp_path / "orca.inp.gz"
    with gzip.open(inp, "wt") as f:
        f.write(_INP_AMO)
    assert parse_inp(inp)["symbols"] == ["Am", "O"]


def test_parse_inp_missing_file(tmp_path):
    parsed = parse_inp(tmp_path / "nope.inp")
    assert parsed["symbols"] == []
    assert parsed["charge"] is None


def test_parse_engrad_matches_get_engrad():
    """The census engrad reader must agree with analysis.get_engrad."""
    from oact_utilities.utils.analysis import get_engrad

    path = FILES / "orca_direct_example" / "AmO_orca.engrad"
    mine = parse_engrad(path)
    theirs = get_engrad(str(path))
    assert mine["energy"] == pytest.approx(theirs["total_energy_Eh"])
    assert mine["gradient"] == pytest.approx(theirs["gradient_Eh_per_bohr"])
    assert mine["symbols"] == ["Am", "O"]
    assert len(mine["coords_bohr"]) == 6


def test_parse_engrad_gzipped(tmp_path):
    """Gzipped quacc engrad is read without a temp file."""
    src = (FILES / "orca_direct_example" / "AmO_orca.engrad").read_text()
    path = tmp_path / "orca.engrad.gz"
    with gzip.open(path, "wt") as f:
        f.write(src)
    assert parse_engrad(path)["symbols"] == ["Am", "O"]


def test_parse_engrad_corrupt_returns_empty(tmp_path):
    path = tmp_path / "orca.engrad"
    path.write_text("# The current gradient in Eh/bohr\n#\nnot-a-float\n")
    assert parse_engrad(path) == {}


# ---------------------------------------------------------------------------
# Force statistics
# ---------------------------------------------------------------------------


def test_force_stats_metal_and_neighbor_split():
    # U at the origin, O at 2 Bohr (~1.06 A, a neighbor), F at 12 Bohr
    # (~6.35 A, outside the 4.0 A cutoff).
    symbols = ["U", "O", "F"]
    coords = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 12.0, 0.0, 0.0]
    gradient = [0.1, 0.0, 0.0, 0.02, 0.0, 0.0, 0.003, 0.0, 0.0]
    stats = force_stats(symbols, coords, gradient, "U")

    assert stats["force_max"] == pytest.approx(0.1)
    assert stats["metal_force"] == pytest.approx(0.1)
    assert stats["ligand_force_max"] == pytest.approx(0.02)
    assert stats["ligand_force_mean"] == pytest.approx(0.0115)
    assert stats["n_neighbors"] == 1
    assert stats["neighbor_force_max"] == pytest.approx(0.02)
    assert stats["force_mean"] == pytest.approx(0.041)
    assert stats["force_median"] == pytest.approx(0.02)


def test_force_stats_convergence_fractions():
    symbols = ["U", "O", "F", "F"]
    coords = [0.0] * 12
    # Norms: 2e-4 (tight), 5e-4 (normal), 2e-3 (loose), 1e-1 (none).
    gradient = [2e-4, 0, 0, 5e-4, 0, 0, 2e-3, 0, 0, 1e-1, 0, 0]
    stats = force_stats(symbols, coords, gradient, "U")
    assert stats["frac_conv_tight"] == pytest.approx(0.25)
    assert stats["frac_conv_normal"] == pytest.approx(0.50)
    assert stats["frac_conv_loose"] == pytest.approx(0.75)


def test_force_stats_rejects_mismatched_gradient():
    assert force_stats(["U", "O"], [0.0] * 6, [0.1, 0.0, 0.0], "U") == {}
    assert force_stats([], [], [], "U") == {}


def test_force_stats_without_metal_skips_split():
    stats = force_stats(["C", "O"], [0.0] * 6, [0.1, 0, 0, 0.2, 0, 0], None)
    assert "force_max" in stats
    assert "metal_force" not in stats
    assert "n_neighbors" not in stats


def test_eh_bohr_conversion_matches_notebooks():
    """Both source notebooks hard-code this factor; keep them in agreement."""
    assert EH_BOHR_TO_EV_ANG == pytest.approx(51.42206313)
    assert CONV_NORMAL * EH_BOHR_TO_EV_ANG == pytest.approx(0.05142206313)


# ---------------------------------------------------------------------------
# End-to-end census
# ---------------------------------------------------------------------------


def _corpus(tmp_path: Path) -> Path:
    """Four job dirs covering completed / failed / to_run and both metal classes."""
    root = tmp_path / "jobs"
    root.mkdir(parents=True)
    grad = [0.001, 0.0, 0.0, 0.0005, 0.0, 0.0]
    _write_job(
        root,
        "job_1",
        inp=_INP_AMO,
        out=_OUT_DONE,
        engrad=_engrad(-670.5, grad, [(95, 0.0, 0.0, 0.0), (8, 3.5, 0.0, 0.0)]),
    )
    _write_job(root, "job_2", inp=_INP_AMO, out=_OUT_FAILED)
    _write_job(root, "job_3", inp=_INP_AMO)  # never ran
    _write_job(
        root,
        "job_4",
        inp=_INP_AMO.replace("Am", "Fe"),
        out=_OUT_DONE,
    )
    return root


def test_run_census_end_to_end(tmp_path):
    root = _corpus(tmp_path)
    out = tmp_path / "census.csv"
    summary, written = run_census([root], out, fmt="csv")

    assert written == out
    assert summary.total == 4
    assert summary.status["completed"] == 2
    assert summary.status["failed"] == 1
    assert summary.status["to_run"] == 1

    rows = {r["job_name"]: r for r in csv.DictReader(open(out))}
    assert set(rows) == {"job_1", "job_2", "job_3", "job_4"}
    assert list(csv.DictReader(open(out)).fieldnames) == list(_FIELD_NAMES)

    done = rows["job_1"]
    assert done["status"] == "completed"
    assert done["metal"] == "Am"
    assert done["metal_class"] == "actinide"
    assert done["elements"] == "Am;O"
    assert done["formula"] == "AmO"
    assert done["natoms"] == "2"
    assert done["charge"] == "-1"
    assert done["spin"] == "8"
    assert done["ligand_elements"] == "O"
    assert done["orig_index"] == "1"
    assert float(done["engrad_energy"]) == pytest.approx(-670.5)
    assert float(done["force_max"]) == pytest.approx(0.001)
    assert float(done["max_forces"]) == pytest.approx(0.001)
    assert done["n_neighbors"] == "1"

    assert rows["job_2"]["status"] == "failed"
    assert "aborting the run" in rows["job_2"]["failure_reason"]
    assert rows["job_3"]["status"] == "to_run"
    assert rows["job_3"]["final_energy"] == ""
    assert rows["job_4"]["metal"] == "Fe"
    assert rows["job_4"]["metal_class"] == "non_actinide"


def test_run_census_no_forces_skips_engrad(tmp_path):
    root = _corpus(tmp_path)
    out = tmp_path / "census.csv"
    run_census([root], out, fmt="csv", with_forces=False)
    row = next(r for r in csv.DictReader(open(out)) if r["job_name"] == "job_1")
    assert row["engrad_energy"] == ""
    assert row["force_max"] == ""
    assert row["n_neighbors"] == ""
    # Status and composition still come through.
    assert row["status"] == "completed"
    assert row["metal"] == "Am"


def test_run_census_no_metrics_still_gets_engrad_energy(tmp_path):
    """--no-metrics drops the orca.out read but keeps the cheap engrad tier."""
    root = _corpus(tmp_path)
    out = tmp_path / "census.csv"
    run_census([root], out, fmt="csv", with_metrics=False)
    row = next(r for r in csv.DictReader(open(out)) if r["job_name"] == "job_1")
    assert row["scf_steps"] == ""
    assert row["wall_time"] == ""
    assert float(row["final_energy"]) == pytest.approx(-670.5)
    assert float(row["force_max"]) == pytest.approx(0.001)


def test_run_census_sqlite_output(tmp_path):
    root = _corpus(tmp_path)
    out = tmp_path / "census.db"
    run_census([root], out, fmt="sqlite", chunk_size=2)

    conn = sqlite3.connect(out)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(census)")]
    assert cols == list(_FIELD_NAMES)
    counts = dict(conn.execute("SELECT status, COUNT(*) FROM census GROUP BY status"))
    assert counts == {"completed": 2, "failed": 1, "to_run": 1}
    (metal,) = conn.execute(
        "SELECT metal FROM census WHERE job_name = 'job_4'"
    ).fetchone()
    assert metal == "Fe"
    conn.close()


def test_run_census_multiple_roots_are_labelled(tmp_path):
    root_a = _corpus(tmp_path / "a")
    root_b = _corpus(tmp_path / "b")
    out = tmp_path / "census.csv"
    summary, _ = run_census([root_a, root_b], out, fmt="csv")

    assert summary.total == 8
    assert summary.by_root[str(root_a)] == 4
    assert summary.by_root[str(root_b)] == 4
    roots = {r["root"] for r in csv.DictReader(open(out))}
    assert roots == {str(root_a), str(root_b)}


def test_run_census_chunked_writes_lose_nothing(tmp_path):
    """A chunk size below the row count must not drop or duplicate rows."""
    root = _corpus(tmp_path)
    out = tmp_path / "census.csv"
    run_census([root], out, fmt="csv", chunk_size=1, workers=2)
    assert len(list(csv.DictReader(open(out)))) == 4


def test_run_census_debug_limit(tmp_path):
    root = _corpus(tmp_path)
    out = tmp_path / "census.csv"
    summary, _ = run_census([root], out, fmt="csv", limit=2)
    assert summary.total == 2


def test_run_census_handles_empty_dir(tmp_path):
    root = tmp_path / "jobs"
    (root / "job_9").mkdir(parents=True)
    out = tmp_path / "census.csv"
    summary, _ = run_census([root], out, fmt="csv")
    assert summary.total == 1
    row = next(iter(csv.DictReader(open(out))))
    assert row["status"] == "to_run"
    assert row["metal"] == ""
    assert row["natoms"] == ""


def test_run_census_on_real_orca_fixtures(tmp_path):
    """Full pipeline against the checked-in ORCA and quacc outputs."""
    import shutil

    root = tmp_path / "jobs"
    root.mkdir()
    shutil.copytree(FILES / "orca_direct_example", root / "job_1")
    shutil.copytree(FILES / "quacc_example", root / "job_2")

    out = tmp_path / "census.csv"
    summary, _ = run_census([root], out, fmt="csv")
    assert summary.status["completed"] == 2

    rows = {r["job_name"]: r for r in csv.DictReader(open(out))}

    amo = rows["job_1"]
    assert amo["metal"] == "Am"
    assert amo["metal_class"] == "actinide"
    assert amo["natoms"] == "2"
    assert amo["spin"] == "8"
    assert float(amo["final_energy"]) == pytest.approx(-670.534993289315)

    npf3 = rows["job_2"]  # gzipped quacc output
    assert npf3["metal"] == "Np"
    assert npf3["elements"] == "Np;F;F;F"
    assert npf3["natoms"] == "4"
    assert npf3["n_basis"] == "225"
    assert npf3["scf_steps"] == "59"
    assert float(npf3["final_energy"]) == pytest.approx(-814.120925544656)
    # Mulliken metal populations for NpF3: charge ~1.65, spin ~4.0 (quintet).
    assert float(npf3["metal_mulliken_charge"]) == pytest.approx(1.65, abs=0.01)
    assert float(npf3["metal_mulliken_spin"]) == pytest.approx(4.0, abs=0.01)
    assert npf3["charge_conserved"] in ("True", "1")
