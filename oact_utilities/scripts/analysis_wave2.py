import os
import numpy as np 
from oact_utilities.utils.analysis import get_full_info_all_jobs

def save_analysis_data(folder, prefix_save=None, flux_tf=True):
    root_name = folder.split("/")[-2]
    json_data = get_full_info_all_jobs(folder, flux_tf=flux_tf)
    #print(root_name)
    #print(json_data)
    print("Saving analysis data for {}...".format(root_name))
    print("Data keys: {}".format(list(json_data.keys())))
    # count the number of successful jobs, ie ones that are not all none values in the key 
    num_successful_jobs = sum(1 for v in json_data.values() if any(val is not None for val in v.values()))
    print(f"Number of successful jobs: {num_successful_jobs}")
    np.save('./analysis/{}{}.npy'.format(root_name, prefix_save if prefix_save else ""), json_data)


if __name__ == "__main__":

    dry = True
    skip_done = True 
    skip_failed = True
    verbose=False
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
        #radical_semiquinones
    ]

    root = "/usr/workspace/vargas58/orca_test/maria_benchmarks/wave_2_x2c_opt_filtered/"
    for folder in list_of_folders:
        root_directory = os.path.join(root, folder)
        #launch_flux_jobs(root_dir=root_directory, dry=dry, skip_done=skip_done, skip_failed=skip_failed, verbose=verbose)
        save_analysis_data(root_directory, flux_tf=flux_tf, prefix_save="x2c_")

    root = "/usr/workspace/vargas58/orca_test/maria_benchmarks/wave_2_omol_opt_filtered/"
    for folder in list_of_folders:
        root_directory = os.path.join(root, folder)
        #launch_flux_jobs(root_dir=root_directory, dry=dry, skip_done=skip_done, skip_failed=skip_failed, verbose=verbose)
        save_analysis_data(root_directory, flux_tf=flux_tf, prefix_save="omol_")




