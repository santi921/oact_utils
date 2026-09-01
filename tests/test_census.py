"""Tests for the DB-free job-directory census."""

from __future__ import annotations

import csv
import gzip
import json
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


# ---------------------------------------------------------------------------
# Parallel shards and merge
# ---------------------------------------------------------------------------


def test_census_schema_is_declared_not_inferred(tmp_path):
    """Shards must share one schema even when a column is all-null."""
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    from oact_utilities.workflows.census import census_schema

    # Corpus A has engrad data; corpus B has none, so every force column is null.
    root_a = _corpus(tmp_path / "a")
    root_b = tmp_path / "b" / "jobs"
    _write_job(root_b, "job_9", inp=_INP_AMO, out=_OUT_DONE)

    out_a, out_b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    run_census([root_a], out_a, fmt="parquet")
    run_census([root_b], out_b, fmt="parquet")

    schema_a = pq.read_schema(out_a)
    assert schema_a.equals(pq.read_schema(out_b))
    assert schema_a.equals(census_schema())


def test_merge_census_concatenates_shards(tmp_path):
    pytest.importorskip("pyarrow")
    import pandas as pd

    from oact_utilities.workflows.census import merge_census

    shards = []
    for name in ("a", "b", "c"):
        root = _corpus(tmp_path / name)
        out = tmp_path / f"{name}.parquet"
        run_census([root], out, fmt="parquet")
        shards.append(out)

    combined = tmp_path / "combined.parquet"
    summary, written = merge_census(shards, combined)

    assert written == combined
    assert summary.total == 12  # 3 shards x 4 jobs
    assert summary.duplicate_job_dirs == 0
    assert summary.status["completed"] == 6
    assert summary.status["failed"] == 3

    want = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    got = pd.read_parquet(combined)
    sort_on = ["root", "job_name"]
    pd.testing.assert_frame_equal(
        want.sort_values(sort_on).reset_index(drop=True),
        got.sort_values(sort_on).reset_index(drop=True),
    )


def test_merge_census_preserves_root_column(tmp_path):
    """A merged table must still be groupable by source root."""
    pytest.importorskip("pyarrow")
    import pandas as pd

    from oact_utilities.workflows.census import merge_census

    roots, shards = [], []
    for name in ("a", "b"):
        root = _corpus(tmp_path / name)
        out = tmp_path / f"{name}.parquet"
        run_census([root], out, fmt="parquet")
        roots.append(str(root))
        shards.append(out)

    combined = tmp_path / "combined.parquet"
    summary, _ = merge_census(shards, combined)

    assert {summary.by_root[r] for r in roots} == {4}
    df = pd.read_parquet(combined)
    assert set(df["root"]) == set(roots)
    assert dict(df.groupby("root").size()) == {roots[0]: 4, roots[1]: 4}


def test_merge_census_flags_duplicate_job_dirs(tmp_path):
    """Re-ingesting a shard double-counts; the summary must say so."""
    pytest.importorskip("pyarrow")

    from oact_utilities.workflows.census import merge_census

    root = _corpus(tmp_path)
    shard = tmp_path / "a.parquet"
    run_census([root], shard, fmt="parquet")

    combined = tmp_path / "combined.parquet"
    summary, _ = merge_census([shard, shard], combined)
    # expand_parquet_inputs de-duplicates, but a direct call must still detect it
    assert summary.total == 8
    assert summary.duplicate_job_dirs == 4


def test_run_census_flags_overlapping_roots(tmp_path):
    """The same root twice (or a nested root) inflates every count."""
    root = _corpus(tmp_path)
    out = tmp_path / "census.csv"
    summary, _ = run_census([root, root], out, fmt="csv")
    assert summary.total == 8
    assert summary.duplicate_job_dirs == 4


def test_merge_census_rejects_foreign_parquet(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    from oact_utilities.workflows.census import merge_census

    bogus = tmp_path / "bogus.parquet"
    pq.write_table(pa.table({"x": [1], "y": ["a"]}), bogus)
    with pytest.raises(ValueError, match="not a census table"):
        merge_census([bogus], tmp_path / "out.parquet")


def test_merge_census_rejects_output_as_input(tmp_path):
    pytest.importorskip("pyarrow")

    from oact_utilities.workflows.census import merge_census

    root = _corpus(tmp_path)
    shard = tmp_path / "a.parquet"
    run_census([root], shard, fmt="parquet")
    with pytest.raises(ValueError, match="also an input"):
        merge_census([shard.resolve()], shard)


def test_merge_census_to_sqlite(tmp_path):
    """Shards can be merged into a different output format."""
    pytest.importorskip("pyarrow")

    from oact_utilities.workflows.census import merge_census

    shards = []
    for name in ("a", "b"):
        root = _corpus(tmp_path / name)
        out = tmp_path / f"{name}.parquet"
        run_census([root], out, fmt="parquet")
        shards.append(out)

    combined = tmp_path / "combined.db"
    summary, _ = merge_census(shards, combined, fmt="sqlite")
    assert summary.total == 8

    conn = sqlite3.connect(combined)
    (n,) = conn.execute("SELECT COUNT(*) FROM census").fetchone()
    assert n == 8
    counts = dict(conn.execute("SELECT status, COUNT(*) FROM census GROUP BY status"))
    assert counts["completed"] == 4
    conn.close()


def test_expand_parquet_inputs(tmp_path):
    from oact_utilities.workflows.census import expand_parquet_inputs

    (tmp_path / "a.parquet").touch()
    (tmp_path / "b.parquet").touch()
    (tmp_path / "notes.txt").touch()

    # directory picks up only .parquet
    got = expand_parquet_inputs([str(tmp_path)])
    assert [p.name for p in got] == ["a.parquet", "b.parquet"]

    # glob works
    assert len(expand_parquet_inputs([str(tmp_path / "*.parquet")])) == 2

    # a directory plus one of its own files must not double-count
    got = expand_parquet_inputs([str(tmp_path), str(tmp_path / "a.parquet")])
    assert [p.name for p in got] == ["a.parquet", "b.parquet"]

    # explicit file
    assert expand_parquet_inputs([str(tmp_path / "a.parquet")])[0].name == "a.parquet"


# ---------------------------------------------------------------------------
# Purge markers (.do_not_rerun.json)
# ---------------------------------------------------------------------------


def _marker(job: Path, **fields) -> None:
    """Write a .do_not_rerun.json purge marker into a job directory."""
    (job / ".do_not_rerun.json").write_text(json.dumps(fields))


def test_purge_marker_makes_job_failed_and_recovers_reason(tmp_path):
    """A purged job's output is gone; the marker is the only record of why."""
    root = tmp_path / "jobs"
    job = _write_job(root, "carpenter_job_1")
    _marker(
        job,
        purge_type="failed",
        scf_steps=412,
        failure_reason="SCF NOT CONVERGED AFTER 600 CYCLES",
    )

    out = tmp_path / "census.csv"
    summary, _ = run_census([root], out, fmt="csv")
    row = next(iter(csv.DictReader(open(out))))

    # Without the marker this would read to_run and overstate pending work.
    assert row["status"] == "failed"
    assert row["marker"] == "True"
    assert row["purge_type"] == "failed"
    assert row["scf_steps"] == "412"
    assert row["failure_reason"] == "SCF NOT CONVERGED AFTER 600 CYCLES"
    assert summary.status["failed"] == 1
    assert summary.status["to_run"] == 0


def test_purge_marker_without_reason_still_reports_failed(tmp_path):
    root = tmp_path / "jobs"
    job = _write_job(root, "job_2")
    _marker(job, purge_type="incomplete_archive")

    out = tmp_path / "census.csv"
    run_census([root], out, fmt="csv")
    row = next(iter(csv.DictReader(open(out))))
    assert row["status"] == "failed"
    assert row["purge_type"] == "incomplete_archive"
    assert row["failure_reason"] == "purged, no reason recorded"


def test_purge_marker_never_overrides_a_completed_job(tmp_path):
    """A normally-terminated orca.out is its own proof; the marker loses."""
    root = tmp_path / "jobs"
    job = _write_job(root, "job_3", inp=_INP_AMO, out=_OUT_DONE)
    _marker(job, purge_type="failed", failure_reason="stale marker")

    out = tmp_path / "census.csv"
    run_census([root], out, fmt="csv")
    row = next(iter(csv.DictReader(open(out))))
    assert row["status"] == "completed"
    assert row["marker"] == "True"
    assert row["purge_type"] == "failed"
    assert row["failure_reason"] == ""


def test_atom_scf_guess_files_are_not_job_output(tmp_path):
    """orca_atom<N>.out is an initial-guess artefact, not a finished job."""
    root = tmp_path / "jobs"
    job = _write_job(root, "carpenter_job_0")
    for ext in ("bibtex", "densities", "out", "property.txt"):
        (job / f"orca_atom90.{ext}").write_text("atomic SCF guess scratch\n")

    out = tmp_path / "census.csv"
    run_census([root], out, fmt="csv")
    row = next(iter(csv.DictReader(open(out))))
    assert row["status"] == "to_run"
    assert row["metal"] == ""
    assert row["natoms"] == ""
    assert row["final_energy"] == ""


def test_corrupt_purge_marker_is_survivable(tmp_path):
    from oact_utilities.workflows.census import read_purge_marker

    root = tmp_path / "jobs"
    job = _write_job(root, "job_4")
    (job / ".do_not_rerun.json").write_text("{not json")
    assert read_purge_marker(job) == {}

    out = tmp_path / "census.csv"
    run_census([root], out, fmt="csv")
    row = next(iter(csv.DictReader(open(out))))
    # Marker present but unreadable: still a give-up signal, no purge_type.
    assert row["status"] == "failed"
    assert row["purge_type"] == ""


def test_mixed_purged_and_live_corpus_counts(tmp_path):
    """The carpenter/08052026 shape: mostly purged, a real minority alive."""
    root = tmp_path / "jobs"
    for i in range(8):
        job = _write_job(root, f"carpenter_job_p{i}")
        _marker(job, purge_type="failed", failure_reason="aborting the run")
    for i in range(3):
        _write_job(root, f"carpenter_job_ok{i}", inp=_INP_AMO, out=_OUT_DONE)

    out = tmp_path / "census.csv"
    summary, _ = run_census([root], out, fmt="csv")
    assert summary.total == 11
    assert summary.status["failed"] == 8
    assert summary.status["completed"] == 3
    assert summary.status["to_run"] == 0
    assert summary.metal_status[("Am", "completed")] == 3


# ---------------------------------------------------------------------------
# SQLite shards (the no-pyarrow path) and cross-format merges
# ---------------------------------------------------------------------------


def test_merge_sqlite_shards_into_sqlite(tmp_path):
    from oact_utilities.workflows.census import merge_census

    shards = []
    for name in ("a", "b"):
        root = _corpus(tmp_path / name)
        out = tmp_path / f"{name}.db"
        run_census([root], out, fmt="sqlite")
        shards.append(out)

    combined = tmp_path / "combined.db"
    summary, _ = merge_census(shards, combined, fmt="sqlite")
    assert summary.total == 8
    assert summary.duplicate_job_dirs == 0
    conn = sqlite3.connect(combined)
    (n,) = conn.execute("SELECT COUNT(*) FROM census").fetchone()
    assert n == 8
    conn.close()


def test_merge_sqlite_shard_into_parquet_restores_bools(tmp_path):
    """SQLite flattens bools to 0/1; parquet's bool column rejects raw ints."""
    pytest.importorskip("pyarrow")
    import pandas as pd

    from oact_utilities.workflows.census import merge_census

    root = tmp_path / "jobs"
    job = _write_job(root, "job_1", inp=_INP_AMO, out=_OUT_DONE)
    _marker(job, purge_type="failed")  # marker=True exercises a real bool

    shard = tmp_path / "a.db"
    run_census([root], shard, fmt="sqlite")

    combined = tmp_path / "combined.parquet"
    summary, _ = merge_census([shard], combined, fmt="parquet")
    assert summary.total == 1

    df = pd.read_parquet(combined)
    assert df["marker"].dtype == bool
    assert bool(df["marker"].iloc[0]) is True


def test_merge_mixed_parquet_and_sqlite_shards(tmp_path):
    pytest.importorskip("pyarrow")
    import pandas as pd

    from oact_utilities.workflows.census import merge_census

    root_a = _corpus(tmp_path / "a")
    root_b = _corpus(tmp_path / "b")
    shard_a, shard_b = tmp_path / "a.parquet", tmp_path / "b.db"
    run_census([root_a], shard_a, fmt="parquet")
    run_census([root_b], shard_b, fmt="sqlite")

    combined = tmp_path / "combined.parquet"
    summary, _ = merge_census([shard_a, shard_b], combined)
    assert summary.total == 8
    df = pd.read_parquet(combined)
    assert df["root"].nunique() == 2
    assert df["job_dir"].nunique() == 8


def test_expand_inputs_picks_up_both_shard_types(tmp_path):
    from oact_utilities.workflows.census import expand_parquet_inputs

    (tmp_path / "a.parquet").touch()
    (tmp_path / "b.db").touch()
    (tmp_path / "notes.txt").touch()
    got = [p.name for p in expand_parquet_inputs([str(tmp_path)])]
    assert got == ["a.parquet", "b.db"]


def test_merge_rejects_unknown_shard_extension(tmp_path):
    from oact_utilities.workflows.census import merge_census

    bogus = tmp_path / "shard.txt"
    bogus.write_text("nope")
    with pytest.raises(ValueError, match="unsupported shard type"):
        merge_census([bogus], tmp_path / "out.db", fmt="sqlite")


def test_symlinked_job_dirs_are_discovered(tmp_path):
    """A corpus assembled by linking must not silently under-count.

    inventory.py and clean.py both discover job dirs with a plain is_dir(),
    which follows symlinks; census has to match or the same root yields two
    different job counts depending on which tool you ask.
    """
    real = tmp_path / "real"
    _write_job(real, "job_1", inp=_INP_AMO, out=_OUT_DONE)
    _write_job(real, "job_2", inp=_INP_AMO, out=_OUT_DONE)

    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "job_1").symlink_to(real / "job_1")
    (linked / "job_2").symlink_to(real / "job_2")
    (linked / "dangling").symlink_to(tmp_path / "does_not_exist")

    out = tmp_path / "census.csv"
    summary, _ = run_census([linked], out, fmt="csv")

    # Both links resolved; the dangling one contributes nothing.
    assert summary.total == 2
    assert summary.status["completed"] == 2
    rows = {r["job_name"]: r for r in csv.DictReader(open(out))}
    assert set(rows) == {"job_1", "job_2"}
    assert rows["job_1"]["metal"] == "Am"


# ---------------------------------------------------------------------------
# Wall time on the cheap path
# ---------------------------------------------------------------------------


def test_parse_total_run_time():
    from oact_utilities.workflows.census import parse_total_run_time

    lines = [
        "                             ****ORCA TERMINATED NORMALLY****",
        "TOTAL RUN TIME: 0 days 0 hours 1 minutes 19 seconds 893 msec",
    ]
    assert parse_total_run_time(lines) == pytest.approx(79.893)
    assert parse_total_run_time(
        ["TOTAL RUN TIME: 2 days 3 hours 4 minutes 5 seconds 6 msec"]
    ) == pytest.approx(2 * 86400 + 3 * 3600 + 4 * 60 + 5 + 0.006)
    assert parse_total_run_time(["no timing here"]) is None
    assert parse_total_run_time([]) is None


def test_no_metrics_still_reports_wall_time(tmp_path):
    """--no-metrics must keep status AND time-to-completion.

    TOTAL RUN TIME is the last line ORCA writes, so the tail read that
    determines status already carries it -- no full-file read needed.
    """
    root = tmp_path / "jobs"
    _write_job(
        root,
        "job_1",
        inp=_INP_AMO,
        out="SCF CONVERGED AFTER 12 CYCLES\n"
        "Timings for individual modules:\n"
        "Sum of individual times          ...       79.073 sec (=   1.318 min)\n"
        "                             ****ORCA TERMINATED NORMALLY****\n"
        "TOTAL RUN TIME: 0 days 0 hours 1 minutes 19 seconds 893 msec\n",
    )
    out = tmp_path / "census.csv"
    run_census([root], out, fmt="csv", with_metrics=False)
    row = next(iter(csv.DictReader(open(out))))

    assert row["status"] == "completed"
    assert float(row["wall_time"]) == pytest.approx(79.893)
    # the expensive columns are genuinely gone
    assert row["scf_steps"] == ""
    # but the inp still supplies the requested core count
    assert row["nprocs_requested"] == "6"


def test_metrics_path_wall_time_is_not_clobbered(tmp_path):
    """With metrics on, the parsed value wins; the tail fallback must not override."""
    root = tmp_path / "jobs"
    _write_job(
        root,
        "job_1",
        inp=_INP_AMO,
        out="SCF CONVERGED AFTER 12 CYCLES\n"
        "Timings for individual modules:\n"
        "Sum of individual times          ...       79.073 sec (=   1.318 min)\n"
        "                             ****ORCA TERMINATED NORMALLY****\n"
        "TOTAL RUN TIME: 0 days 0 hours 1 minutes 19 seconds 893 msec\n",
    )
    out = tmp_path / "census.csv"
    run_census([root], out, fmt="csv", recompute=True)
    row = next(iter(csv.DictReader(open(out))))
    assert float(row["wall_time"]) == pytest.approx(79.073)
