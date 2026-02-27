"""Tests for rerun_singlets_uks.py utility functions."""

import json

from oact_utilities.scripts.multi_spin.rerun_singlets_uks import (
    atoms_from_db_row,
    extract_charge_from_orca_inp,
)

# ---------------------------------------------------------------------------
# extract_charge_from_orca_inp
# ---------------------------------------------------------------------------


def test_extract_charge_standard_format(tmp_path):
    """Parse charge from standard ORCA '* xyz <charge> <mult>' line."""
    inp = tmp_path / "orca.inp"
    inp.write_text(
        "! wB97M-V def2-TZVPD UKS\n"
        "* xyz -2 1\n"
        "U 0.0 0.0 0.0\n"
        "O 0.0 0.0 1.8\n"
        "*\n"
    )
    assert extract_charge_from_orca_inp(str(tmp_path)) == -2


def test_extract_charge_positive(tmp_path):
    """Parse positive charge."""
    inp = tmp_path / "orca.inp"
    inp.write_text("! EnGrad\n* xyz 3 5\nU 0 0 0\n*\n")
    assert extract_charge_from_orca_inp(str(tmp_path)) == 3


def test_extract_charge_zero(tmp_path):
    """Parse zero charge."""
    inp = tmp_path / "orca.inp"
    inp.write_text("! EnGrad\n* xyz 0 1\nH 0 0 0\n*\n")
    assert extract_charge_from_orca_inp(str(tmp_path)) == 0


def test_extract_charge_no_file(tmp_path):
    """Return None if orca.inp doesn't exist."""
    assert extract_charge_from_orca_inp(str(tmp_path)) is None


def test_extract_charge_no_xyz_line(tmp_path):
    """Return None if no '* xyz' line found."""
    inp = tmp_path / "orca.inp"
    inp.write_text("! EnGrad\n%scf maxiter 500\nend\n")
    assert extract_charge_from_orca_inp(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# atoms_from_db_row
# ---------------------------------------------------------------------------


def test_atoms_from_db_row_symbols():
    """Reconstruct Atoms from element symbols."""
    coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [0.0, 0.0, -1.8]]
    elements = ["U", "O", "O"]
    atoms = atoms_from_db_row(json.dumps(coords), json.dumps(elements))
    assert atoms is not None
    assert len(atoms) == 3
    assert atoms.get_chemical_formula() == "O2U"


def test_atoms_from_db_row_atomic_numbers():
    """Reconstruct Atoms from atomic numbers."""
    coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8]]
    elements = [92, 8]  # U, O
    atoms = atoms_from_db_row(json.dumps(coords), json.dumps(elements))
    assert atoms is not None
    assert len(atoms) == 2
    assert "U" in atoms.get_chemical_symbols()
    assert "O" in atoms.get_chemical_symbols()


def test_atoms_from_db_row_string_atomic_numbers():
    """Reconstruct Atoms from string atomic numbers (common DB format)."""
    coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8]]
    elements = ["92", "8"]  # U, O as strings
    atoms = atoms_from_db_row(json.dumps(coords), json.dumps(elements))
    assert atoms is not None
    assert "U" in atoms.get_chemical_symbols()


def test_atoms_from_db_row_none_coords():
    """Return None if coords are None."""
    assert atoms_from_db_row(None, json.dumps(["U", "O"])) is None


def test_atoms_from_db_row_none_elements():
    """Return None if elements are None."""
    assert atoms_from_db_row(json.dumps([[0, 0, 0]]), None) is None


def test_atoms_from_db_row_invalid_json():
    """Return None for invalid JSON."""
    assert atoms_from_db_row("not json", json.dumps(["U"])) is None
