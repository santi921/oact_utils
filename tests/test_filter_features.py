"""Tests for the v4 filter classifier's feature construction.

These functions are the contract between training and serving: the notebook builds its
training matrix with them and the production scorer builds its inputs with them, so a
change here silently changes what a trained model means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oact_utilities.utils.filter_features import (
    COVALENT_RADII,
    GEOM_CATEGORICAL,
    GEOM_NUMERIC,
    NEIGHBOR_CUTOFF,
    WEIRD_BOND_SCALE,
    build_features,
    categorical_features,
    comp_numeric,
    feature_columns,
    geometric_features,
    infer_metal,
    ligand_composition,
    parse_xyz,
)

# Np-O at 1.80 Ang is an ordinary neptunyl bond; the H sits well outside the first shell.
NEPTUNYL = "Np 0.0 0.0 0.0\nO 1.80 0.0 0.0\nO -1.80 0.0 0.0\nH 3.00 0.0 0.0"
WATER = "O 0.000 0.000 0.000\nH 0.758 0.587 0.000\nH -0.758 0.587 0.000"


def _frame(xyz: str, metal: str, **overrides) -> pd.DataFrame:
    symbols, coords = parse_xyz(xyz)
    row = {
        "elements": ";".join(symbols),
        "natoms": len(symbols),
        "charge": 0,
        "spin": 1,
        "metal": metal,
        "n_basis": 100,
        **geometric_features(symbols, coords, metal),
    }
    row.update(overrides)
    return pd.DataFrame([row])


class TestParseXyz:
    def test_bare_block(self):
        symbols, coords = parse_xyz(NEPTUNYL)
        assert symbols == ["Np", "O", "O", "H"]
        assert coords.shape == (4, 3)

    def test_header_and_bare_agree(self):
        """A 2-line XYZ header must be stripped, not read as coordinates."""
        bare_symbols, bare_coords = parse_xyz(NEPTUNYL)
        headed_symbols, headed_coords = parse_xyz(f"4\nsome comment\n{NEPTUNYL}")
        assert headed_symbols == bare_symbols
        assert np.allclose(headed_coords, bare_coords)

    def test_blank_lines_ignored(self):
        symbols, _ = parse_xyz(f"\n{NEPTUNYL}\n\n")
        assert symbols == ["Np", "O", "O", "H"]

    def test_extra_columns_ignored(self):
        """extXYZ rows carry forces after the coordinates; only x/y/z are read."""
        symbols, coords = parse_xyz("Np 0.0 0.0 0.0 0.1 0.2 0.3")
        assert symbols == ["Np"]
        assert np.allclose(coords, [[0.0, 0.0, 0.0]])


class TestCovalentRadii:
    def test_covers_all_118_elements(self):
        assert len(COVALENT_RADII) == 118

    def test_actinides_beyond_cordero_are_real_values(self):
        """Cordero stops at Cm; ASE pads Bk-Og with a flat 2.00 Ang placeholder.

        This corpus contains Bk, Cf, Es and No, so a placeholder would distort exactly the
        rows that matter. Pyykko gives each a distinct, physical value.
        """
        beyond = {COVALENT_RADII[s] for s in ("Bk", "Cf", "Es", "Fm", "Md", "No", "Lr")}
        assert 2.0 not in beyond
        assert len(beyond) > 1


class TestGeometricFeatures:
    def test_declared_keys_are_all_produced(self):
        symbols, coords = parse_xyz(NEPTUNYL)
        feats = geometric_features(symbols, coords, "Np")
        assert set(feats) == set(GEOM_NUMERIC) | set(GEOM_CATEGORICAL)

    def test_metal_neighbours(self):
        symbols, coords = parse_xyz(NEPTUNYL)
        feats = geometric_features(symbols, coords, "Np")
        assert feats["metal_nn1_ang"] == pytest.approx(1.80)
        assert feats["metal_nn2_ang"] == pytest.approx(1.80)
        assert feats["metal_nn1_elem"] == "O"
        assert feats["metal_coord_n"] == 3  # both O and the H at 3.0 Ang are within 4.0

    def test_actinyl_bond_is_not_a_clash_at_the_current_scale(self):
        """The reason WEIRD_BOND_SCALE is 0.70 and not 0.80.

        Single-bond radii give r_Np + r_O = 2.34 Ang, so 0.80 x 2.34 = 1.87 Ang would flag
        a real Np=O bond at 1.80 Ang. 0.70 x 2.34 = 1.64 Ang does not.
        """
        symbols, coords = parse_xyz(NEPTUNYL)
        feats = geometric_features(symbols, coords, "Np")
        expected = COVALENT_RADII["Np"] + COVALENT_RADII["O"]
        assert feats["min_cov_ratio"] == pytest.approx(1.80 / expected, rel=1e-6)
        assert feats["min_cov_ratio"] > WEIRD_BOND_SCALE
        assert feats["n_weird_bonds"] == 0

    def test_real_clash_is_flagged(self):
        clash = "O 0.000 0.000 0.000\nH 0.300 0.230 0.000\nH -0.758 0.587 0.000"
        symbols, coords = parse_xyz(clash)
        feats = geometric_features(symbols, coords, "O")
        assert feats["min_cov_ratio"] < WEIRD_BOND_SCALE
        assert feats["n_weird_bonds"] == 1
        assert feats["weird_bond_frac"] == pytest.approx(1 / 3)

    def test_missing_metal_degrades_to_nan_not_exception(self):
        """A row whose named metal is absent must not blow up the whole batch."""
        symbols, coords = parse_xyz(WATER)
        feats = geometric_features(symbols, coords, "Np")
        assert np.isnan(feats["metal_nn1_ang"])
        assert feats["metal_nn1_elem"] is None
        assert not np.isnan(
            feats["shortest_bond_ang"]
        )  # non-metal features still valid

    def test_single_atom(self):
        feats = geometric_features(["Np"], np.zeros((1, 3)), "Np")
        assert np.isnan(feats["shortest_bond_ang"])
        assert feats["metal_nn1_elem"] is None

    def test_scale_is_a_parameter(self):
        symbols, coords = parse_xyz(NEPTUNYL)
        assert (
            geometric_features(symbols, coords, "Np", scale=0.70)["n_weird_bonds"] == 0
        )
        assert (
            geometric_features(symbols, coords, "Np", scale=0.80)["n_weird_bonds"] == 2
        )

    def test_cutoff_is_a_parameter(self):
        symbols, coords = parse_xyz(NEPTUNYL)
        assert (
            geometric_features(symbols, coords, "Np", cutoff=2.0)["metal_coord_n"] == 2
        )
        assert (
            geometric_features(symbols, coords, "Np", cutoff=NEIGHBOR_CUTOFF)[
                "metal_coord_n"
            ]
            == 3
        )


class TestInferMetal:
    @pytest.mark.parametrize(
        "xyz,expected",
        [
            (NEPTUNYL, "Np"),
            ("Gd 0 0 0\nO 2.0 0 0", "Gd"),
            ("U 0 0 0\nBr 2.5 0 0", "U"),  # heavy ligand must not outrank the centre
            (WATER, ""),  # purely organic
        ],
    )
    def test_picks_the_centre(self, xyz, expected):
        assert infer_metal(parse_xyz(xyz)[0]) == expected


class TestBuildFeatures:
    def test_produces_every_declared_column(self):
        ligands = ["C", "N", "O"]
        feats = build_features(_frame(NEPTUNYL, "Np"), ligands)
        assert set(feature_columns(ligands)) <= set(feats.columns)

    def test_composition_values(self):
        feats = build_features(_frame(NEPTUNYL, "Np"), ["C", "N", "O"]).iloc[0]
        assert feats["n_unique_elements"] == 3  # Np, O, H
        assert feats["n_ligand_types"] == 1  # O only; metal and H excluded
        assert feats["h_fraction"] == pytest.approx(0.25)
        assert feats["has_O"] == 1
        assert feats["has_C"] == 0

    def test_index_is_preserved(self):
        frame = _frame(NEPTUNYL, "Np")
        frame.index = pd.Index([77])
        assert build_features(frame, ["O"]).index.tolist() == [77]

    def test_categorical_features_are_declared(self):
        assert categorical_features() == {"metal", "metal_nn1_elem"}

    def test_comp_numeric_tracks_the_vocabulary(self):
        assert "has_Cl" in comp_numeric(["Cl"])
        assert "has_Cl" not in comp_numeric(["O"])


class TestLigandComposition:
    def test_excludes_metal_and_hydrogen(self):
        assert ligand_composition(_frame(NEPTUNYL, "Np")).tolist() == ["O"]

    def test_sorted_and_deduplicated(self):
        frame = _frame(NEPTUNYL, "Np", elements="Np;O;C;O;N;H")
        assert ligand_composition(frame).tolist() == ["C,N,O"]

    def test_bare_metal_is_labelled_not_blank(self):
        """An empty group key would silently merge unrelated rows into one CV fold."""
        frame = _frame(NEPTUNYL, "Np", elements="Np;H")
        assert ligand_composition(frame).tolist() == ["(none)"]
