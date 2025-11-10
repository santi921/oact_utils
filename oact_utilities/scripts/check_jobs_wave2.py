import os
from oact_utilities.utils.status import check_sucessful_jobs


if __name__ == "__main__":

    flux_tf = True
    check_many = False

    root = "/Users/santiagovargas/dev/oact_utils/data/baselines/jobs/"

    hard_chalc = "Hard_Donors/Chalcogenides/"
    organic_COT = "Organic/COT/"
    soft_ethers = "Soft_Donors/ChalcogenidesEthers/"
    soft_chalcogenides = "Soft_Donors/Chalcogenides/"
    radical_semiquinones = "Radical/Semiquinones/"

    root_directory = os.path.join(root, hard_chalc)
    check_sucessful_jobs(
        root_dir=root_directory, check_many=check_many, flux_tf=flux_tf
    )
    root_directory = os.path.join(root, organic_COT)
    check_sucessful_jobs(
        root_dir=root_directory, check_many=check_many, flux_tf=flux_tf
    )
    root_directory = os.path.join(root, soft_ethers)
    check_sucessful_jobs(
        root_dir=root_directory, check_many=check_many, flux_tf=flux_tf
    )
    root_directory = os.path.join(root, soft_chalcogenides)
    check_sucessful_jobs(
        root_dir=root_directory, check_many=check_many, flux_tf=flux_tf
    )
    root_directory = os.path.join(root, radical_semiquinones)
    check_sucessful_jobs(
        root_dir=root_directory, check_many=check_many, flux_tf=flux_tf
    )
