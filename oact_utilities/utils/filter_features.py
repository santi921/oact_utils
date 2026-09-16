"""Feature construction for the v4 filter classifier.

Single source of truth: ``classifier_v4_filter.ipynb`` imports this module to build its
training matrix, and ``v4_filter_predict.py`` imports it to score new structures. There is
no generated copy, so the training path and the serving path cannot drift apart.

Every feature here is computable from an input structure before ORCA runs. The filter's own
inputs (fmax, <S^2>, HOMO-LUMO gap, SCF electron count, energy) are deliberately absent:
training on them gives AUC near 1.0 and learns nothing.

Radii are single-bond covalent radii for all 118 elements from Pyykko and Atsumi,
*Chem. Eur. J.* **15** (2009) 186, converted pm -> Angstrom. Cordero (2008), which
``ase.data.covalent_radii`` ships, stops at Cm (Z = 96) and ASE pads the remaining 22
elements with a flat 2.00 Ang placeholder; this corpus contains Bk, Cf, Es and No, so that
padding would distort exactly the rows that matter most here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

COVALENT_RADII: dict[str, float] = {
    "H": 0.32,
    "He": 0.46,
    "Li": 1.33,
    "Be": 1.02,
    "B": 0.85,
    "C": 0.75,
    "N": 0.71,
    "O": 0.63,
    "F": 0.64,
    "Ne": 0.67,
    "Na": 1.55,
    "Mg": 1.39,
    "Al": 1.26,
    "Si": 1.16,
    "P": 1.11,
    "S": 1.03,
    "Cl": 0.99,
    "Ar": 0.96,
    "K": 1.96,
    "Ca": 1.71,
    "Sc": 1.48,
    "Ti": 1.36,
    "V": 1.34,
    "Cr": 1.22,
    "Mn": 1.19,
    "Fe": 1.16,
    "Co": 1.11,
    "Ni": 1.1,
    "Cu": 1.12,
    "Zn": 1.18,
    "Ga": 1.24,
    "Ge": 1.21,
    "As": 1.21,
    "Se": 1.16,
    "Br": 1.14,
    "Kr": 1.17,
    "Rb": 2.1,
    "Sr": 1.85,
    "Y": 1.63,
    "Zr": 1.54,
    "Nb": 1.47,
    "Mo": 1.38,
    "Tc": 1.28,
    "Ru": 1.25,
    "Rh": 1.25,
    "Pd": 1.2,
    "Ag": 1.28,
    "Cd": 1.36,
    "In": 1.42,
    "Sn": 1.4,
    "Sb": 1.4,
    "Te": 1.36,
    "I": 1.33,
    "Xe": 1.31,
    "Cs": 2.32,
    "Ba": 1.96,
    "La": 1.8,
    "Ce": 1.63,
    "Pr": 1.76,
    "Nd": 1.74,
    "Pm": 1.73,
    "Sm": 1.72,
    "Eu": 1.68,
    "Gd": 1.69,
    "Tb": 1.68,
    "Dy": 1.67,
    "Ho": 1.66,
    "Er": 1.65,
    "Tm": 1.64,
    "Yb": 1.7,
    "Lu": 1.62,
    "Hf": 1.52,
    "Ta": 1.46,
    "W": 1.37,
    "Re": 1.31,
    "Os": 1.29,
    "Ir": 1.22,
    "Pt": 1.23,
    "Au": 1.24,
    "Hg": 1.33,
    "Tl": 1.44,
    "Pb": 1.44,
    "Bi": 1.51,
    "Po": 1.45,
    "At": 1.47,
    "Rn": 1.42,
    "Fr": 2.23,
    "Ra": 2.01,
    "Ac": 1.86,
    "Th": 1.75,
    "Pa": 1.69,
    "U": 1.7,
    "Np": 1.71,
    "Pu": 1.72,
    "Am": 1.66,
    "Cm": 1.66,
    "Bk": 1.68,
    "Cf": 1.68,
    "Es": 1.65,
    "Fm": 1.67,
    "Md": 1.73,
    "No": 1.76,
    "Lr": 1.61,
    "Rf": 1.57,
    "Db": 1.49,
    "Sg": 1.43,
    "Bh": 1.41,
    "Hs": 1.34,
    "Mt": 1.29,
    "Ds": 1.28,
    "Rg": 1.21,
    "Cn": 1.22,
    "Nh": 1.36,
    "Fl": 1.43,
    "Mc": 1.62,
    "Lv": 1.75,
    "Ts": 1.65,
    "Og": 1.57,
}

# A contact below this fraction of (r_A + r_B) is a clash. 0.70 rather than the more usual
# 0.80 because these are single-bond radii: at 0.80 an ordinary actinyl Np=O near 1.80 Ang
# trips the test, and 48.6% of structures that survived the filter get flagged. At 0.70 that
# falls to 6.4% while fmax recall only drops from 0.998 to 0.977.
WEIRD_BOND_SCALE = 0.70

# Metal coordination radius, matching census.py.
NEIGHBOR_CUTOFF = 4.0

# Organic/main-group elements that are never treated as the metal centre when inferring one.
_NON_METAL_HINTS = ("H", "C", "N", "O", "F", "S", "P", "Cl")

GEOM_NUMERIC = [
    "shortest_bond_ang",  # closest contact anywhere in the structure
    "min_cov_ratio",  # that contact as a fraction of (r_A + r_B)
    "n_weird_bonds",  # pairs below WEIRD_BOND_SCALE x (r_A + r_B)
    "weird_bond_frac",  # the same, per atom
    "metal_nn1_ang",  # metal to its nearest neighbour
    "metal_nn2_ang",  # metal to its second nearest neighbour
    "metal_nn1_ratio",  # covalent-normalized, so comparable across elements
    "metal_nn2_ratio",
    "metal_nn_gap_ang",  # nn2 - nn1, how distinct the first shell is
    "metal_coord_n",  # atoms within NEIGHBOR_CUTOFF of the metal
    "radius_gyration_ang",  # overall compactness
    "max_extent_ang",  # largest interatomic distance
]
GEOM_CATEGORICAL = ["metal_nn1_elem"]

BASE_COMP_NUMERIC = [
    "natoms",
    "charge",
    "spin",
    "n_basis",
    "n_unique_elements",
    "n_ligand_types",
    "h_fraction",
]
COMP_CATEGORICAL = ["metal"]


def comp_numeric(ligand_elems: list[str]) -> list[str]:
    """Composition numeric column names for a given ligand-element vocabulary."""
    return BASE_COMP_NUMERIC + [f"has_{elem}" for elem in ligand_elems]


def feature_columns(ligand_elems: list[str]) -> list[str]:
    """The full combined feature order the model expects."""
    return (
        comp_numeric(ligand_elems) + COMP_CATEGORICAL + GEOM_NUMERIC + GEOM_CATEGORICAL
    )


def categorical_features() -> set[str]:
    """Columns the preprocessor one-hot encodes rather than scales."""
    return set(COMP_CATEGORICAL) | set(GEOM_CATEGORICAL)


def parse_xyz(text: str) -> tuple[list[str], np.ndarray]:
    """Symbols and an (natoms, 3) array in Angstrom, from standard XYZ or a bare block."""
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if lines and lines[0].split()[0].isdigit():
        lines = lines[2:]
    symbols, coords = [], []
    for line in lines:
        parts = line.split()
        symbols.append(parts[0])
        coords.append([float(value) for value in parts[1:4]])
    return symbols, np.asarray(coords, dtype=float)


def infer_metal(symbols: list[str]) -> str:
    """Heaviest non-organic element present, correct for the mononuclear complexes here.

    Returns an empty string when the structure is purely organic.
    """
    order = list(COVALENT_RADII)
    candidates = [s for s in symbols if s not in _NON_METAL_HINTS]
    return max(candidates, key=order.index) if candidates else ""


def geometric_features(
    symbols: list[str],
    coords: np.ndarray,
    metal: str,
    *,
    scale: float = WEIRD_BOND_SCALE,
    cutoff: float = NEIGHBOR_CUTOFF,
) -> dict[str, Any]:
    """All of GEOM_NUMERIC + GEOM_CATEGORICAL for one structure.

    Metal-centred entries come back as NaN / None when the named metal is absent from
    the geometry, so a bad row degrades to a missing value rather than an exception.
    """
    natoms = len(symbols)
    if natoms < 2:
        empty: dict[str, Any] = {name: np.nan for name in GEOM_NUMERIC}
        empty["metal_nn1_elem"] = None
        return empty

    radii = np.array([COVALENT_RADII[symbol] for symbol in symbols])
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    expected = radii[:, None] + radii[None, :]
    ratio = distances / expected
    np.fill_diagonal(distances, np.inf)
    np.fill_diagonal(ratio, np.inf)

    n_weird = int(np.triu(ratio < scale, k=1).sum())
    centroid = coords.mean(axis=0)
    finite = distances[np.isfinite(distances)]

    feats: dict[str, Any] = {
        "shortest_bond_ang": float(distances.min()),
        "min_cov_ratio": float(ratio.min()),
        "n_weird_bonds": n_weird,
        "weird_bond_frac": n_weird / natoms,
        "radius_gyration_ang": float(
            np.sqrt(((coords - centroid) ** 2).sum(axis=1).mean())
        ),
        "max_extent_ang": float(finite.max()),
    }

    if metal not in symbols:
        feats |= {
            "metal_nn1_ang": np.nan,
            "metal_nn2_ang": np.nan,
            "metal_nn1_ratio": np.nan,
            "metal_nn2_ratio": np.nan,
            "metal_nn_gap_ang": np.nan,
            "metal_coord_n": np.nan,
            "metal_nn1_elem": None,
        }
        return feats

    centre = symbols.index(metal)
    order = np.argsort(distances[centre])
    first = int(order[0])
    second = int(order[1]) if natoms > 2 else None
    feats |= {
        "metal_nn1_ang": float(distances[centre, first]),
        "metal_nn1_ratio": float(ratio[centre, first]),
        "metal_nn2_ang": (
            float(distances[centre, second]) if second is not None else np.nan
        ),
        "metal_nn2_ratio": (
            float(ratio[centre, second]) if second is not None else np.nan
        ),
        "metal_nn_gap_ang": (
            float(distances[centre, second] - distances[centre, first])
            if second is not None
            else np.nan
        ),
        "metal_coord_n": int((distances[centre] <= cutoff).sum()),
        "metal_nn1_elem": symbols[first],
    }
    return feats


def build_features(df: pd.DataFrame, ligand_elems: list[str]) -> pd.DataFrame:
    """Composition + geometry feature matrix. Nothing here needs ORCA to have run.

    Args:
        df: rows carrying ``elements`` (semicolon-separated), ``natoms``, ``charge``,
            ``spin``, ``n_basis``, ``metal``, and every GEOM_NUMERIC / GEOM_CATEGORICAL
            column, as produced by :func:`geometric_features`.
        ligand_elems: the ``has_<elem>`` vocabulary, fixed at training time.
    """
    elems = [row.split(";") for row in df.elements]
    metals = df.metal.tolist()

    comp = pd.DataFrame(
        {
            "n_unique_elements": [len(set(e)) for e in elems],
            "n_ligand_types": [
                len({x for x in e if x not in (m, "H")}) for e, m in zip(elems, metals)
            ],
            "h_fraction": [e.count("H") / len(e) for e in elems],
        },
        index=df.index,
    )
    for elem in ligand_elems:
        comp[f"has_{elem}"] = [int(elem in set(e)) for e in elems]

    return pd.concat(
        [
            df[["natoms", "charge", "spin", "n_basis", "metal"]],
            comp,
            df[GEOM_NUMERIC + GEOM_CATEGORICAL],
        ],
        axis=1,
    )


def ligand_composition(df: pd.DataFrame) -> pd.Series:
    """Sorted non-metal, non-H element set per row, used as the CV grouping key.

    Folds never split one ligand environment across train and validation, so the score
    reflects generalization to unseen chemistry rather than memorized compositions.
    """
    return pd.Series(
        [
            ",".join(sorted({x for x in row.split(";") if x not in (metal, "H")}))
            for row, metal in zip(df.elements, df.metal)
        ],
        index=df.index,
    ).replace("", "(none)")
