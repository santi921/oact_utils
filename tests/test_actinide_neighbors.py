import numpy as np


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
