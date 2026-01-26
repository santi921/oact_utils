import os

from oact_utilities.utils.status import check_sucessful_jobs_sella

if __name__ == "__main__":

    check_many = False
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

    ##############################################################################
    # Ritwik - Things to modify for your system
    root = "/Users/santiagovargas/dev/oact_utils/data/baselines/jobs/"
    flux_tf = True
    ##############################################################################

    for folder in list_of_folders:
        root_directory = os.path.join(root, folder)
        check_sucessful_jobs_sella(
            root_dir=root_directory,
        )
