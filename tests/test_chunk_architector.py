import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ase.io import read

from oact_utilities.utils.architector import chunk_architector_csv, xyz_string_to_atoms


def test_chunk_architector_writes_xyz_and_manifest(tmp_path: Path):
    # prepare a small CSV with three XYZ blocks
    xyz1 = """2
mol1
H 0.0 0.0 0.0
H 0.0 0.0 0.74
"""
    xyz2 = """3
mol2
O 0.0 0.0 0.0
H 0.0 0.76 0.58
H 0.0 -0.76 0.58
"""
    xyz3 = """1
mol3
He 0.0 0.0 0.0
"""

    df = pd.DataFrame({"aligned_csd_core": [xyz1, xyz2, xyz3]})
    csv_path = tmp_path / "arch.csv"
    df.to_csv(csv_path, index=False)

    outdir = tmp_path / "out"
    db_path = outdir / "manifest.db"
    manifest = chunk_architector_csv(csv_path, outdir, chunk_size=2, db_path=db_path)

    assert manifest.exists()

    # chunk_0 should have 2 frames, chunk_1 should have 1 frame
    chunk0 = outdir / "chunk_0.xyz"
    chunk1 = outdir / "chunk_1.xyz"
    assert chunk0.exists()
    assert chunk1.exists()

    structures0 = list(read(chunk0, index=":", format="xyz"))
    structures1 = list(read(chunk1, index=":", format="xyz"))

    assert len(structures0) == 2
    assert len(structures1) == 1

    # manifest should contain three entries mapping to chunk files
    man_df = pd.read_csv(manifest)
    assert len(man_df) == 3
    assert set(man_df["chunk_file"]) == {"chunk_0.xyz", "chunk_1.xyz"}

    # verify DB was created and has three rows with status 'ready'
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM structures")
    count = cur.fetchone()[0]
    assert count == 3
    cur.execute("SELECT DISTINCT(status) FROM structures")
    statuses = {r[0] for r in cur.fetchall()}
    assert statuses == {"ready"}
    conn.close()


class TestXyzStringToAtoms:
    """Tests for xyz_string_to_atoms function."""

    def test_standard_xyz_format(self):
        """Test parsing standard XYZ format with atom count and comment."""
        xyz_str = """2
H2 molecule
H 0.0 0.0 0.0
H 0.0 0.0 0.74
"""
        atoms = xyz_string_to_atoms(xyz_str)

        assert len(atoms) == 2
        assert list(atoms.get_chemical_symbols()) == ["H", "H"]
        assert np.allclose(atoms.get_positions()[0], [0.0, 0.0, 0.0])
        assert np.allclose(atoms.get_positions()[1], [0.0, 0.0, 0.74])

    def test_architector_format_no_header(self):
        """Test parsing architector CSV format (no header)."""
        xyz_str = """H 0.0 0.0 0.0
H 0.0 0.0 0.74"""
        atoms = xyz_string_to_atoms(xyz_str)

        assert len(atoms) == 2
        assert list(atoms.get_chemical_symbols()) == ["H", "H"]
        assert np.allclose(atoms.get_positions()[1], [0.0, 0.0, 0.74])

    def test_water_molecule(self):
        """Test parsing a water molecule."""
        xyz_str = """O 0.0 0.0 0.0
H 0.757 0.586 0.0
H -0.757 0.586 0.0"""
        atoms = xyz_string_to_atoms(xyz_str)

        assert len(atoms) == 3
        assert list(atoms.get_chemical_symbols()) == ["O", "H", "H"]

    def test_actinide_element(self):
        """Test parsing structure with actinide element."""
        xyz_str = """U 0.0 0.0 0.0
O 1.8 0.0 0.0
O -1.8 0.0 0.0"""
        atoms = xyz_string_to_atoms(xyz_str)

        assert len(atoms) == 3
        assert "U" in atoms.get_chemical_symbols()
        assert atoms.get_chemical_symbols().count("O") == 2

    def test_empty_string_raises(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="Empty XYZ string"):
            xyz_string_to_atoms("")

    def test_whitespace_only_raises(self):
        """Test that whitespace-only string raises ValueError."""
        with pytest.raises(ValueError, match="Empty XYZ string"):
            xyz_string_to_atoms("   \n\n  ")

    def test_no_valid_atoms_raises(self):
        """Test that string with no valid coordinate lines raises ValueError."""
        xyz_str = """2
comment line only"""
        with pytest.raises(ValueError, match="No atoms found"):
            xyz_string_to_atoms(xyz_str)
