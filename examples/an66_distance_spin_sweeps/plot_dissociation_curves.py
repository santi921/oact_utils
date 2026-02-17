#!/usr/bin/env python3
"""Plot dissociation energy and force curves for AN66 distance-spin sweeps.

Overlays OMol and X2C LOT results for each unique molecule, with separate
curves per spin state. Produces two subplots per molecule: energy and forces
vs. dissociation distance.

Usage:
    python plot_dissociation_curves.py
    python plot_dissociation_curves.py --omol-db path/to/omol.db --x2c-db path/to/x2c.db
    python plot_dissociation_curves.py --output-dir plots/ --format pdf
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTINIDES = {
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
}

EH_TO_KCALMOL = 627.5094740631  # 1 Hartree in kcal/mol

# Spin-state color cycle (consistent across all molecules)
SPIN_COLORS = {
    1: "#1f77b4",
    2: "#ff7f0e",
    3: "#2ca02c",
    4: "#d62728",
    5: "#9467bd",
    6: "#8c564b",
    7: "#e377c2",
    8: "#7f7f7f",
    9: "#bcbd22",
    10: "#17becf",
    11: "#aec7e8",
    12: "#ffbb78",
    13: "#98df8a",
    14: "#ff9896",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_db(db_path: Path) -> pd.DataFrame:
    """Load completed structures from a workflow database.

    Args:
        db_path: Path to the SQLite database.

    Returns:
        DataFrame with columns: orig_index, elements, spin, charge,
        final_energy, max_forces, geometry.
    """
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(
        """
        SELECT orig_index, elements, spin, charge,
               final_energy, max_forces, geometry
        FROM structures
        WHERE status = 'completed'
          AND final_energy IS NOT NULL
        """,
        conn,
    )
    conn.close()
    return df


def load_mapping(mapping_path: Path) -> dict:
    """Load the molecule-to-row mapping JSON."""
    with open(mapping_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def parse_geometry(geom_str: str) -> tuple[list[str], np.ndarray]:
    """Parse an XYZ geometry string into element symbols and positions.

    Args:
        geom_str: XYZ format string (natoms, comment, coordinates).

    Returns:
        Tuple of (element_symbols, positions_array).
    """
    lines = [ln for ln in geom_str.strip().splitlines() if ln.strip()]
    symbols: list[str] = []
    positions: list[list[float]] = []
    # Skip first two lines (natoms + comment)
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 4:
            symbols.append(parts[0])
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return symbols, np.array(positions)


def _get_an_ligand_distances(geom_str: str) -> tuple[list[str], list[float]]:
    """Compute all An-ligand distances from a geometry.

    Returns:
        Tuple of (ligand_symbols, distances) for each non-actinide atom.
    """
    symbols, positions = parse_geometry(geom_str)

    an_idx = None
    for i, sym in enumerate(symbols):
        if sym in ACTINIDES:
            an_idx = i
            break
    if an_idx is None:
        raise ValueError(f"No actinide found in geometry: {symbols}")

    an_pos = positions[an_idx]
    lig_symbols: list[str] = []
    lig_dists: list[float] = []
    for i, sym in enumerate(symbols):
        if i != an_idx:
            lig_symbols.append(sym)
            lig_dists.append(float(np.linalg.norm(positions[i] - an_pos)))
    return lig_symbols, lig_dists


def detect_dissociating_atom_index(geometries: list[str]) -> int:
    """Detect which ligand atom is being dissociated across a sweep.

    Compares An-ligand distances across all geometries and returns the
    ligand index (0-based, excluding the actinide) with the largest
    variance — i.e., the atom whose distance is actually changing.

    Args:
        geometries: List of XYZ geometry strings from the sweep.

    Returns:
        Index of the dissociating ligand atom (among non-actinide atoms).
    """
    all_dists: list[list[float]] = []
    for geom in geometries:
        _, dists = _get_an_ligand_distances(geom)
        all_dists.append(dists)

    dist_array = np.array(all_dists)  # shape: (n_geoms, n_ligands)
    variances = np.var(dist_array, axis=0)
    return int(np.argmax(variances))


def compute_dissociation_distance(geom_str: str, dissoc_atom_idx: int) -> float:
    """Compute the An-L distance for the dissociating atom.

    Args:
        geom_str: XYZ format geometry string.
        dissoc_atom_idx: Index of the dissociating atom among non-actinide atoms.

    Returns:
        Distance in Angstroms from actinide to the dissociating ligand.
    """
    _, dists = _get_an_ligand_distances(geom_str)
    return dists[dissoc_atom_idx]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _relative_energy_kcalmol(energy_series: pd.Series) -> pd.Series:
    """Convert absolute energies to relative energies in kcal/mol.

    Shifts so the minimum energy in the series is 0.
    """
    return (energy_series - energy_series.min()) * EH_TO_KCALMOL


def _estimate_equilibrium_distance(
    omol_data: pd.DataFrame | None,
    x2c_data: pd.DataFrame | None,
) -> float | None:
    """Estimate the equilibrium An-L distance from the sweep data.

    The fine sweep starts at min_shift=-0.2 relative to the optimized GS
    geometry, so R_eq = min(sweep distances) + 0.2 Å.
    """
    min_d = np.inf
    for df in (omol_data, x2c_data):
        if df is not None and len(df) > 0:
            min_d = min(min_d, df["distance"].min())
    if np.isinf(min_d):
        return None
    return float(min_d + 0.2)


def plot_molecule(
    mol_name: str,
    ligand_type: str,
    omol_data: pd.DataFrame | None,
    x2c_data: pd.DataFrame | None,
    output_dir: Path,
    fmt: str = "png",
    dpi: int = 150,
) -> Path | None:
    """Create energy and force plots for a single molecule.

    Energy is plotted as relative energy (kcal/mol) within each LOT+spin
    curve, so OMol and X2C can be compared on the same axis despite
    different absolute energy scales.

    Args:
        mol_name: Molecule name (e.g., "AmF3").
        ligand_type: Element symbol of the dissociating ligand.
        omol_data: DataFrame with OMol results (may be None/empty).
        x2c_data: DataFrame with X2C results (may be None/empty).
        output_dir: Directory for saving figures.
        fmt: Output format (png, pdf, svg).
        dpi: Resolution for raster formats.

    Returns:
        Path to the saved figure, or None if no data.
    """
    has_omol = omol_data is not None and len(omol_data) > 0
    has_x2c = x2c_data is not None and len(x2c_data) > 0

    if not has_omol and not has_x2c:
        return None

    fig, (ax_energy, ax_force) = plt.subplots(
        2,
        1,
        figsize=(8, 10),
        sharex=True,
    )
    fig.suptitle(
        f"{mol_name} — Dissociation of {ligand_type}",
        fontsize=14,
        fontweight="bold",
    )

    # Collect all spin states across both LOTs
    all_spins: set[int] = set()
    if has_omol:
        all_spins.update(omol_data["spin"].unique())
    if has_x2c:
        all_spins.update(x2c_data["spin"].unique())

    for spin in sorted(all_spins):
        color = SPIN_COLORS.get(spin, "#333333")

        # OMol: solid lines with circles
        if has_omol:
            subset = omol_data[omol_data["spin"] == spin].sort_values("distance")
            if len(subset) > 0:
                rel_e = _relative_energy_kcalmol(subset["final_energy"])
                ax_energy.plot(
                    subset["distance"],
                    rel_e,
                    "o-",
                    color=color,
                    label=f"OMol S={spin}",
                    markersize=4,
                    linewidth=1.5,
                )
                ax_force.plot(
                    subset["distance"],
                    subset["max_forces"],
                    "o-",
                    color=color,
                    label=f"OMol S={spin}",
                    markersize=4,
                    linewidth=1.5,
                )

        # X2C: dashed lines with triangles
        if has_x2c:
            subset = x2c_data[x2c_data["spin"] == spin].sort_values("distance")
            if len(subset) > 0:
                rel_e = _relative_energy_kcalmol(subset["final_energy"])
                ax_energy.plot(
                    subset["distance"],
                    rel_e,
                    "v--",
                    color=color,
                    label=f"X2C S={spin}",
                    markersize=4,
                    linewidth=1.5,
                )
                ax_force.plot(
                    subset["distance"],
                    subset["max_forces"],
                    "v--",
                    color=color,
                    label=f"X2C S={spin}",
                    markersize=4,
                    linewidth=1.5,
                )

    # Equilibrium distance vertical line
    r_eq = _estimate_equilibrium_distance(omol_data, x2c_data)
    if r_eq is not None:
        for ax in (ax_energy, ax_force):
            ax.axvline(
                r_eq,
                color="black",
                linestyle=":",
                linewidth=1.2,
                alpha=0.7,
            )
        ax_energy.text(
            r_eq,
            ax_energy.get_ylim()[1] * 0.95,
            f" R$_{{eq}}$={r_eq:.2f} Å",
            fontsize=8,
            va="top",
            ha="left",
        )

    # Format energy subplot
    ax_energy.set_ylabel("Relative Energy (kcal/mol)", fontsize=12)
    ax_energy.legend(fontsize=8, ncol=2, loc="best")
    ax_energy.grid(True, which="major", alpha=0.3)
    ax_energy.grid(True, which="minor", alpha=0.15, linestyle=":")
    ax_energy.minorticks_on()
    ax_energy.tick_params(labelsize=10)

    # Format force subplot
    ax_force.set_xlabel(f"An–{ligand_type} Distance (Å)", fontsize=12)
    ax_force.set_ylabel("Max Force (Eh/Bohr)", fontsize=12)
    ax_force.set_yscale("log")
    ax_force.legend(fontsize=8, ncol=2, loc="best")
    ax_force.grid(True, which="major", alpha=0.3)
    ax_force.grid(True, which="minor", alpha=0.15, linestyle=":")
    ax_force.tick_params(labelsize=10)

    plt.tight_layout()

    output_path = output_dir / f"{mol_name}_dissociation.{fmt}"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path


def add_distance_column(
    df: pd.DataFrame,
    mapping: dict,
    mol_name: str,
) -> tuple[pd.DataFrame, str]:
    """Add a 'distance' column to the DataFrame for a given molecule.

    Detects which atom is actually being dissociated by finding the
    An-ligand bond with the largest variance across the sweep geometries,
    rather than trusting the mapping's ligand_type (which may disagree
    with Architector's choice of which bond to pull).

    Args:
        df: DataFrame with orig_index and geometry columns.
        mapping: Molecule mapping dict.
        mol_name: Molecule name key in mapping.

    Returns:
        Tuple of (filtered DataFrame with 'distance' column,
        element symbol of the actual dissociating ligand).
    """
    info = mapping[mol_name]
    row_ids = set(info["row_ids"])
    fallback_ligand = info["ligand_types"][0]

    mol_df = df[df["orig_index"].isin(row_ids)].copy()
    if mol_df.empty:
        return mol_df, fallback_ligand

    # Detect which atom is actually being dissociated from the geometries
    geom_list = mol_df["geometry"].tolist()
    try:
        dissoc_idx = detect_dissociating_atom_index(geom_list)
        # Get the element symbol of the dissociating atom
        lig_symbols, _ = _get_an_ligand_distances(geom_list[0])
        actual_ligand = lig_symbols[dissoc_idx]
    except (ValueError, IndexError):
        dissoc_idx = None
        actual_ligand = fallback_ligand

    distances = []
    for _, row in mol_df.iterrows():
        try:
            if dissoc_idx is not None:
                d = compute_dissociation_distance(row["geometry"], dissoc_idx)
            else:
                d = np.nan
            distances.append(d)
        except (ValueError, IndexError):
            distances.append(np.nan)

    mol_df["distance"] = distances
    mol_df = mol_df.dropna(subset=["distance"])

    # Drop rows with positive energies (SCF convergence failures)
    mol_df = mol_df[mol_df["final_energy"] < 0]

    # Drop energy outliers per spin state: points that deviate from the
    # median by more than `energy_outlier_eh` Hartrees are almost certainly
    # converged to the wrong electronic state.
    energy_outlier_eh = 1.0
    filtered_indices: list[int] = []
    for _spin, group in mol_df.groupby("spin"):
        median_e = group["final_energy"].median()
        keep = (group["final_energy"] - median_e).abs() <= energy_outlier_eh
        filtered_indices.extend(group[keep].index.tolist())
    mol_df = mol_df.loc[filtered_indices]

    return mol_df, actual_ligand


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate dissociation curve plots for all AN66 molecules."""
    parser = argparse.ArgumentParser(
        description="Plot AN66 dissociation energy and force curves",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--omol-db",
        type=Path,
        default=Path("results/an66_dissociation.db"),
        help="Path to OMol LOT results database",
    )
    parser.add_argument(
        "--x2c-db",
        type=Path,
        default=Path("results/an66_dissociation_x2c.db"),
        help="Path to X2C LOT results database",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("an66_dissociation.mapping.json"),
        help="Path to molecule mapping JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Output directory for plots",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output figure format",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution for raster formats",
    )
    parser.add_argument(
        "--molecules",
        type=str,
        nargs="*",
        default=None,
        help="Specific molecule names to plot (default: all)",
    )
    args = parser.parse_args()

    # Load mapping
    mapping = load_mapping(args.mapping)
    print(f"Loaded mapping with {len(mapping)} molecules")

    # Load databases
    omol_df = None
    x2c_df = None

    if args.omol_db.exists():
        omol_df = load_db(args.omol_db)
        print(f"OMol DB: {len(omol_df)} completed rows")
    else:
        print(f"WARNING: OMol DB not found: {args.omol_db}")

    if args.x2c_db.exists():
        x2c_df = load_db(args.x2c_db)
        print(f"X2C DB:  {len(x2c_df)} completed rows")
    else:
        print(f"WARNING: X2C DB not found: {args.x2c_db}")

    if omol_df is None and x2c_df is None:
        print("ERROR: No databases found. Nothing to plot.")
        return

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Select molecules
    mol_names = args.molecules if args.molecules else sorted(mapping.keys())
    print(f"\nPlotting {len(mol_names)} molecules...\n")

    plotted = 0
    skipped = 0

    for mol_name in mol_names:
        if mol_name not in mapping:
            print(f"  WARNING: {mol_name} not found in mapping, skipping")
            skipped += 1
            continue

        # Build per-molecule DataFrames with distances
        # The actual dissociating ligand is detected from geometry variance,
        # which may differ from the mapping's ligand_type.
        omol_mol = None
        x2c_mol = None
        ligand_type = mapping[mol_name]["ligand_types"][0]  # fallback

        if omol_df is not None:
            omol_mol, ligand_type = add_distance_column(omol_df, mapping, mol_name)

        if x2c_df is not None:
            x2c_mol, lt = add_distance_column(x2c_df, mapping, mol_name)
            if x2c_mol is not None and len(x2c_mol) > 0:
                ligand_type = lt

        result = plot_molecule(
            mol_name=mol_name,
            ligand_type=ligand_type,
            omol_data=omol_mol,
            x2c_data=x2c_mol,
            output_dir=args.output_dir,
            fmt=args.format,
            dpi=args.dpi,
        )

        if result:
            print(f"  {mol_name}: saved -> {result.name}")
            plotted += 1
        else:
            print(f"  {mol_name}: no completed data, skipped")
            skipped += 1

    print(f"\nDone. Plotted: {plotted}, Skipped: {skipped}")
    print(f"Output directory: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
