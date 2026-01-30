import matplotlib
import numpy as np
import pandas as pd


def test_actinide_neighbor_mean_distances_basic():
    from oact_utilities.utils.analysis import actinide_first_neighbor_distances

    # create a simple geometry: U at origin, two O at 2.3 and 2.4 Angstrom on x-axis
    elements = ["U", "O", "O", "C"]
    coords = np.array([[0.0, 0.0, 0.0], [2.3, 0, 0], [2.4, 0, 0], [5.0, 0, 0]])

    res = actinide_first_neighbor_distances(elements, coords, max_distance=3.0)
    assert len(res) == 1
    entry = res[0]
    assert entry["center_symbol"] == "U"
    # nearest neighbor should be 2.3
    assert abs(entry["first_distance"] - 2.3) < 1e-6
    assert entry["n_neighbors_within_cutoff"] == 2
    assert entry["neighbor_symbol"] == "O"


def test_plot_element_vs_lot_runs():
    # Use non-interactive backend to avoid display
    matplotlib.use("Agg")

    from oact_utilities.utils.plotting import plot_element_vs_lot

    df = pd.DataFrame(
        {
            "element": ["U", "U", "Np", "Np"],
            "lot": ["omol", "x2c", "omol", "x2c"],
            "distance": [2.32, 2.31, 2.22, 2.24],
        }
    )

    ax = plot_element_vs_lot(df, value_col="distance")
    assert ax.get_xlabel() == "Element"
    assert ax.get_ylabel() == "distance"
    legend = ax.get_legend()
    assert legend is not None
