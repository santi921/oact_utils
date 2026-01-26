import os

import numpy as np

from oact_utilities.utils.analysis import get_sp_info_all_jobs


def save_analysis_data(
    folder: str,
    save_dir: str = "./analysis/",
    prefix_save: str = None,
    flux_tf: bool = True,
) -> None:
    """
    Simple script to save analysis data for all jobs in a folder.
    Args:
        folder (str): Path to the root folder containing job subfolders.
        prefix_save (str, optional): Prefix to add to the saved file name. Defaults to None.
        flux_tf (bool, optional): Whether to check for flux output files. Defaults to True.
    Returns:
        None
    """
    root_name = folder.split("/")[-2]
    json_data = get_sp_info_all_jobs(folder, flux_tf=flux_tf)
    np.save(
        "{}/{}{}_sp.npy".format(
            save_dir, root_name, prefix_save if prefix_save else ""
        ),
        json_data,
    )


if __name__ == "__main__":

    dry = True
    skip_done = True
    skip_failed = True
    verbose = False
    flux_tf = True

    hard_chalc = "Hard_Donors/Chalcogenides/"
    hard_nitrates = "Hard_Donors/Nitrates/"
    hard_crown_cryptands = "Hard_Donors/Crown-Cryptands/"
    organic_COT = "Organic/COT/"
    organic_carbenes = "Organic/carbenes/"
    organic_tris_cp = "Organic/tris-Cp/"
    soft_ethers = "Soft_Donors/ChalcogenEthers/"
    soft_chalcogenides = "Soft_Donors/Chalcogenides/"
    soft_dithiocarbamates = "Soft_Donors/Dithiocarbamates-dithiophosphates-dithiolates/"

    list_of_folders = [
        hard_chalc,
        hard_nitrates,
        hard_crown_cryptands,
        organic_COT,
        organic_carbenes,
        organic_tris_cp,
        soft_ethers,
        soft_chalcogenides,
        soft_dithiocarbamates,
        # radical_semiquinones
    ]

    ##############################################################################
    # Ritwik - Things to modify for your system
    root = "/usr/workspace/vargas58/orca_test/maria_benchmarks/wave_2_x2c_opt_filtered/"
    ##############################################################################

    for folder in list_of_folders:
        root_directory = os.path.join(root, folder)
        save_analysis_data(root_directory, flux_tf=flux_tf, prefix_save="x2c_")
