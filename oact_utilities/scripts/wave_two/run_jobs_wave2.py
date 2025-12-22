import os
from oact_utilities.utils.jobs import launch_flux_jobs, launch_slurm_jobs


if __name__ == "__main__":

    ##############################################################################
    # Ritwik - Things to modify for your system
    dry = True
    root = "/Users/santiagovargas/dev/oact_utils/data/baselines/jobs/"
    ##############################################################################

    skip_done = True

    hard_chalc = "Hard_Donors/Chalcogenides/"
    hard_nitrates = "Hard_Donors/Nitrates/"
    hard_crown_cryptands = "Hard_Donors/Crown-Cryptands/"
    organic_COT = "Organic/COT/"
    organic_carbenes = "Organic/carbenes/"
    organic_tris_cp = "Organic/tris-Cp/"
    soft_ethers = "Soft_Donors/ChalcogenidesEthers/"
    soft_chalcogenides = "Soft_Donors/Chalcogenides/"
    soft_dithiocarbamates = "Soft_Donors/Dithiocarbamates-dithiophosphates-dithiolates/"
    radical_semiquinones = "Radical/Semiquinones/"

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
        radical_semiquinones,
    ]

    for folder in list_of_folders:
        root_directory = os.path.join(root, folder)
        # launch_flux_jobs(root_dir=root_directory, dry=dry, skip_done=skip_done)
        # Ritwik - use this to launch SLURM jobs
        launch_slurm_jobs(root_dir=root_directory, dry=dry, skip_done=skip_done)
