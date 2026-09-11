"""Morfeus feature computation worker for multiprocessing.

Extracted from classifier_homoleptics.ipynb so that ProcessPoolExecutor
with mp_context='spawn' can import and pickle the worker function.
Uses batch processing to minimize IPC overhead.
"""

import warnings
from io import StringIO

import numpy as np
from ase.io import read as ase_read
from morfeus import SASA, BuriedVolume

MORFEUS_FEAT_NAMES = [
    "bv_free_vol",
    "sasa_metal",
]


def get_metal_idx(atoms, metal_symbol: str) -> int:
    """Return 0-based index of metal atom. Raises AssertionError if not exactly one."""
    syms = atoms.get_chemical_symbols()
    idxs = [i for i, s in enumerate(syms) if s == metal_symbol]
    assert len(idxs) == 1, f"Expected 1 {metal_symbol} atom, found {len(idxs)}"
    return idxs[0]


def compute_morfeus_features(xyz_str: str, metal_symbol: str) -> dict:
    """Compute trimmed Morfeus + coordination geometry descriptors.

    Dispersion and VisibleVolume removed for speed. Warnings from morfeus
    are suppressed -- NaN results are handled by SimpleImputer.
    """
    atoms = ase_read(StringIO(xyz_str), format="xyz")
    elements = list(atoms.get_chemical_symbols())
    coords = atoms.get_positions()
    feats = {k: np.nan for k in MORFEUS_FEAT_NAMES}

    try:
        midx = get_metal_idx(atoms, metal_symbol)
        midx_1 = midx + 1  # Morfeus is 1-indexed
    except AssertionError:
        return feats

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            bv = BuriedVolume(elements, coords, midx_1)
            feats["bv_free_vol"] = bv.free_volume
        except Exception:
            pass

        try:
            sasa_obj = SASA(elements, coords)
            try:
                feats["sasa_metal"] = sasa_obj.atom_areas[midx_1]
            except (KeyError, IndexError):
                pass
        except Exception:
            pass

    return feats


def morfeus_batch_worker(batch: list) -> list:
    """Process a batch of (df_idx, xyz_str, metal_symbol) tuples.

    Returns list of dicts with features + _df_idx key.
    Batch processing minimizes IPC overhead -- one pickle round-trip
    per batch (~500 items) instead of per item.
    """
    results = []
    for df_idx, xyz_str, metal_symbol in batch:
        feats = compute_morfeus_features(xyz_str, metal_symbol)
        feats["_df_idx"] = df_idx
        results.append(feats)
    return results
