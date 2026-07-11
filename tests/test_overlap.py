"""Tests for the two-root ORCA input overlap checker."""

import gzip
from pathlib import Path

from oact_utilities.workflows.overlap import (
    canonicalize,
    find_input_file,
    parse_coordinate_block,
    parse_job_dir,
    scan_root,
)

_INP_TEMPLATE = """! wB97M-V RIJCOSX NoUseSym DIIS NormalConv DEFGRID3
%scf
  MaxIter 1000
end
%pal
  nprocs 6
end

* xyz {charge} {mult}
{atoms}
*
"""

_AMF3_ATOMS = """Am\t0.000000\t0.000000\t0.000000
F\t0.000000\t0.000000\t2.061070
F\t0.000000\t1.960687\t-0.630113
F\t-1.636177\t-0.973355\t-0.792575"""

_AMO_ATOMS = """Am\t0.000000\t0.000000\t0.000000
O\t0.000000\t0.000000\t1.858970"""


def _make_job(
    root: Path,
    name: str,
    atoms: str,
    charge: int = 0,
    mult: int = 7,
    inp_name: str = "orca.inp",
    gz: bool = False,
) -> Path:
    job_dir = root / name
    job_dir.mkdir(parents=True)
    content = _INP_TEMPLATE.format(charge=charge, mult=mult, atoms=atoms)
    if gz:
        with gzip.open(job_dir / (inp_name + ".gz"), "wt") as fh:
            fh.write(content)
    else:
        (job_dir / inp_name).write_text(content)
    return job_dir


def test_parse_coordinate_block():
    text = _INP_TEMPLATE.format(charge=-1, mult=8, atoms=_AMF3_ATOMS)
    parsed = parse_coordinate_block(text)
    assert parsed is not None
    charge, mult, atoms = parsed
    assert charge == -1
    assert mult == 8
    assert len(atoms) == 4
    assert atoms[0][0] == "Am"
    assert atoms[1] == ("F", 0.0, 0.0, 2.061070)


def test_parse_coordinate_block_unclosed():
    text = "* xyz 0 1\nH 0.0 0.0 0.0\n"  # no closing '*'
    assert parse_coordinate_block(text) is None


def test_canonicalize_translation_and_order_invariant():
    atoms = [("Am", 0.0, 0.0, 0.0), ("O", 0.0, 0.0, 1.85897)]
    shifted = [("O", 1.0, 2.0, 3.85897), ("Am", 1.0, 2.0, 2.0)]
    key1, fkey1, formula1 = canonicalize(0, 7, atoms, decimals=3)
    key2, fkey2, formula2 = canonicalize(0, 7, shifted, decimals=3)
    assert key1 == key2
    assert fkey1 == fkey2
    assert formula1 == "AmO"


def test_canonicalize_distinguishes_charge_spin_geometry():
    atoms = [("Am", 0.0, 0.0, 0.0), ("O", 0.0, 0.0, 1.85897)]
    stretched = [("Am", 0.0, 0.0, 0.0), ("O", 0.0, 0.0, 2.5)]
    base, _, _ = canonicalize(0, 7, atoms, decimals=3)
    assert canonicalize(1, 7, atoms, decimals=3)[0] != base
    assert canonicalize(0, 5, atoms, decimals=3)[0] != base
    assert canonicalize(0, 7, stretched, decimals=3)[0] != base


def test_find_input_file_prefers_orca_inp(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    (job / "aaa.inp").write_text("x")
    (job / "orca.inp").write_text("x")
    assert find_input_file(job).name == "orca.inp"
    assert find_input_file(tmp_path / "missing") is None


def test_parse_job_dir_gzipped(tmp_path):
    job = _make_job(tmp_path, "gzjob", _AMO_ATOMS, gz=True)
    result = parse_job_dir(job, decimals=3)
    assert not isinstance(result, str)
    assert result.formula == "AmO"
    assert result.natoms == 2


def test_parse_job_dir_skip_reasons(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert parse_job_dir(empty, decimals=3) == "no_inp"

    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "orca.inp").write_text("! wB97M-V\n%pal nprocs 6 end\n")
    assert parse_job_dir(bad, decimals=3) == "unparseable_block"


def test_scan_root_overlap(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    # Shared structure (translated + gzipped in B), plus one unique per root
    # and an internal duplicate in A.
    _make_job(root_a, "job_amf3", _AMF3_ATOMS)
    _make_job(root_a, "job_amf3_dup", _AMF3_ATOMS)
    _make_job(root_a, "job_amo", _AMO_ATOMS)
    shifted = "\n".join(
        f"{line.split()[0]}\t{float(line.split()[1]) + 1.5:.6f}\t{float(line.split()[2]) - 2.0:.6f}\t{float(line.split()[3]) + 0.25:.6f}"
        for line in _AMF3_ATOMS.splitlines()
    )
    _make_job(root_b, "job_amf3_moved", shifted, gz=True)
    _make_job(root_b, "job_amo_charged", _AMO_ATOMS, charge=1)

    scan_a = scan_root(root_a, decimals=3, workers=2, limit=None, label="A")
    scan_b = scan_root(root_b, decimals=3, workers=2, limit=None, label="B")

    geom_a, geom_b = scan_a.by_geom, scan_b.by_geom
    assert len(scan_a.structures) == 3
    assert len(geom_a) == 2  # AmF3 dup collapses
    assert len(geom_a[[k for k in geom_a if len(geom_a[k]) > 1][0]]) == 2

    overlap = set(geom_a) & set(geom_b)
    assert len(overlap) == 1  # AmF3 matches despite translation + gzip
    only_b = set(geom_b) - overlap
    assert len(only_b) == 1  # AmO with charge=1 does not match A's neutral AmO
