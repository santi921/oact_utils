from spyrmsd.rmsd import rmsd

import numpy as np
from oact_utilities.utils.analysis import get_full_info_all_jobs, get_rmsd_start_final


def test_get_rmsd_start_final():
    res_no_traj = get_rmsd_start_final(
        "/Users/santiagovargas/dev/oact_utils/tests/files/no_traj"
    )
    res_traj = get_rmsd_start_final(
        "/Users/santiagovargas/dev/oact_utils/tests/files/traj/"
    )
    np.testing.assert_array_almost_equal(
        res_no_traj["energies_frames"],
        res_traj["energies_frames"],
        decimal=5,
        err_msg="Energies from no_traj and traj do not match",
    )


test_get_rmsd_start_final()
