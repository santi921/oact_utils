import os
from oact_utilities.utils.jobs import launch_flux_jobs


if __name__ == "__main__":

    dry = True
    skip_done = False
    root = "/Users/santiagovargas/dev/oact_utils/data/baselines/jobs/"

    hard_chalc = "Hard_Donors/Chalcogenides/"
    organic_COT = "Organic/COT/"
    soft_ethers = "Soft_Donors/ChalcogenidesEthers/"
    soft_chalcogenides = "Soft_Donors/Chalcogenides/"
    radical_semiquinones = "Radical/Semiquinones/"

    root_directory = os.path.join(root, hard_chalc)
    launch_flux_jobs(root_dir=root_directory, dry=dry, skip_done=skip_done)
    root_directory = os.path.join(root, organic_COT)
    launch_flux_jobs(root_dir=root_directory, dry=dry, skip_done=skip_done)
    root_directory = os.path.join(root, soft_ethers)
    launch_flux_jobs(root_dir=root_directory, dry=dry, skip_done=skip_done)
    root_directory = os.path.join(root, soft_chalcogenides)
    launch_flux_jobs(root_dir=root_directory, dry=dry, skip_done=skip_done)
    root_directory = os.path.join(root, radical_semiquinones)
    launch_flux_jobs(root_dir=root_directory, dry=dry, skip_done=skip_done)
