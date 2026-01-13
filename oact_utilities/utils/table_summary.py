import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .create import atomic_numbers_to_elements, elements_to_atomic_numbers


BOHR_TO_ANGSTROM = 0.529177210903


def _parse_npy_into_table(path: str, sella: bool = False) -> Optional[pd.DataFrame]:
    """Load an npy run-summary and return a normalized pandas DataFrame.

    Columns: name, elements_numbers, elements_names, coords, status, delta_energy
    coords are in Angstroms (converts from bohr when needed).
    """
    if path is None or not os.path.exists(path):
        return None
    arr = np.load(path, allow_pickle=True).item()
    table = []
    for key, val in arr.items():
        if sella:
            status = 1 if val.get("status", None) is not None else 0
            coords_bohr = val.get("coords_final_bohr", None)
            #print("coords bohr", coords_bohr)
            if coords_bohr not in (None, []):
                coords = np.array(coords_bohr) * BOHR_TO_ANGSTROM
            else:
                coords = None
            energy_list = val.get("sella_energy_frames", None)
            if energy_list == []:
                energy_list = None
            delta_energy = (energy_list[-1] - energy_list[0]) if energy_list is not None else None
            elements_numbers = val.get("elements_engrad", None)
            elements_names = (
                atomic_numbers_to_elements(elements_numbers)
                if elements_numbers is not None
                else None
            )
        else:
            coords = val.get("coords_final", None)
            elems = val.get("elements_final", None)
            status = 1 if coords is not None else 0
            delta_energy = None
            if status == 1:
                energies = val.get("energies_opt", None)
                if energies is not None and len(energies) > 0:
                    delta_energy = (energies[-1] - energies[0]) * 27.2114
            elements_numbers = (
                elements_to_atomic_numbers(elems) if elems is not None else None
            )
            elements_names = elems

        table.append(
            {
                "name": key,
                "elements_numbers": elements_numbers,
                "elements_names": elements_names,
                "coords": coords,
                "status": int(status),
                "delta_energy": float(delta_energy) if delta_energy is not None else None,
            }
        )

    return pd.DataFrame(table)


def summarize_category_status(
    category: str,
    omol_lot: str,
    x2c_lot: str,
    omol_sella_lot: str,
    x2c_sella_lot: str,
) -> pd.DataFrame:
    """Return a DataFrame summarizing status/delta_energy across four tables for a category.

    The resulting DataFrame is indexed by structure name and contains columns:
      - omol_status, omol_delta, x2c_status, x2c_delta, omol_sella_status, omol_sella_delta, x2c_sella_status, x2c_sella_delta

    Missing files or missing rows will result in NaNs.
    """
    p_omol = os.path.join(omol_lot, f"omol_{category}.npy")
    p_x2c = os.path.join(x2c_lot, f"x2c_{category}.npy")
    p_omol_sella = os.path.join(omol_sella_lot, f"{category}omol_sella_.npy")
    p_x2c_sella = os.path.join(x2c_sella_lot, f"{category}x2c_sella_.npy")

    t_omol = _parse_npy_into_table(p_omol, sella=False)
    t_x2c = _parse_npy_into_table(p_x2c, sella=False)
    t_omol_sella = _parse_npy_into_table(p_omol_sella, sella=True)
    t_x2c_sella = _parse_npy_into_table(p_x2c_sella, sella=True)

    # helper to create a series from table
    def _series_from_table(df: Optional[pd.DataFrame], prefix: str):
        if df is None:
            return pd.DataFrame()
        df2 = df.set_index("name")[ ["status", "delta_energy"] ].copy()
        df2.columns = [f"{prefix}_status", f"{prefix}_delta"]
        return df2

    s_omol = _series_from_table(t_omol, "omol")
    s_x2c = _series_from_table(t_x2c, "x2c")
    s_omol_sella = _series_from_table(t_omol_sella, "omol_sella")
    s_x2c_sella = _series_from_table(t_x2c_sella, "x2c_sella")

    # union index
    idx = set()
    for s in (s_omol, s_x2c, s_omol_sella, s_x2c_sella):
        if isinstance(s, pd.DataFrame) and not s.empty:
            idx |= set(s.index.tolist())
    if len(idx) == 0:
        return pd.DataFrame()

    idx = sorted(idx)
    out = pd.DataFrame(index=idx)
    for s in (s_omol, s_x2c, s_omol_sella, s_x2c_sella):
        if s.empty:
            continue
        out = out.join(s, how="left")

    out.index.name = "name"
    out.insert(0, "category", category)
    return out.reset_index()


def summarize_all_categories(
    categories: Iterable[str],
    omol_lot: str,
    x2c_lot: str,
    omol_sella_lot: str,
    x2c_sella_lot: str,
) -> pd.DataFrame:
    """Summarize status for multiple categories and return concatenated DataFrame."""
    frames = []
    for cat in categories:
        df = summarize_category_status(cat, omol_lot, x2c_lot, omol_sella_lot, x2c_sella_lot)
        if df is None or df.empty:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
