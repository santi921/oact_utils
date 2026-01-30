"""Example script to create a sample actinide first-neighbor distance plot."""

from __future__ import annotations

import pandas as pd

from oact_utilities.utils.plotting import plot_element_vs_lot


def build_sample_df():
    # synthetic example: three elements, two LOTs
    data = {
        "element": ["U", "U", "Np", "Np", "Pu", "Pu"],
        "lot": ["omol", "x2c", "omol", "x2c", "omol", "x2c"],
        "distance": [2.32, 2.30, 2.25, 2.28, 2.40, 2.35],
    }
    return pd.DataFrame(data)


def main():
    df = build_sample_df()
    ax = plot_element_vs_lot(df, value_col="distance")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig("actinide_first_neighbor_example.png", dpi=200)


if __name__ == "__main__":
    main()
