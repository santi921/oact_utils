import os 

def check_job_termination(dir, check_many=False):
    # sweep folder file for flux*out files 
    files = os.listdir(dir)
    #print("files: ", files)
    files_out = [f for f in files if f.startswith("flux") and f.endswith("out")]
    
    if len(files_out) > 1:
        files_out.sort(key=lambda x: os.path.getmtime(os.path.join(dir, x)), reverse=True)
        if not check_many:
            files_out = files_out[:1]
        

    if len(files_out) == 0:
        return False
    if check_many and len(files_out) > 1:
        status_list = []
        for file_out in files_out:
            output_file = dir + "/" + file_out

            # read last line of output_file
            with open(output_file, "r") as f:
                lines = f.readlines()
            # if last line contains "ORCA TERMINATED NORMALLY", then get geometry forces
            #ORCA TERMINATED NORMALLY
            file_status = 0
            for line in lines[-5:]:      
                if "ORCA TERMINATED NORMALLY" in line:
                    file_status = 1
                if "aborting the run" in line:
                    file_status = -1
            status_list.append(file_status)
        if all(status == 1 for status in status_list):
            return 1
        elif any(status == -1 for status in status_list):
            return -1
        else:
            return 0
        
    else:
        output_file = dir + "/" + files_out[0]

        # read last line of output_file
        with open(output_file, "r") as f:
            lines = f.readlines()
        # if last line contains "ORCA TERMINATED NORMALLY", then get geometry forces
        #ORCA TERMINATED NORMALLY
        for line in lines[-5:]:      
            if "ORCA TERMINATED NORMALLY" in line:
                return 1
            if "aborting the run" in line:
                return -1
        return 0

def check_sucessful_jobs(
    root_dir, check_many=False
):
    count_folder = 0
    count_success = 0
    count_still_running = 0
    # iterate through every subfolder in root_dir
    for folder in os.listdir(root_dir):
        folder_to_use = os.path.join(root_dir, folder)
        if os.path.isdir(folder_to_use):
            count_folder += 1
            # check if folder has successful flux job 
            if check_job_termination(folder_to_use, check_many=check_many) == 1:
                print(f"Job in {folder_to_use} completed successfully.")
                count_success += 1
            elif check_job_termination(folder_to_use, check_many=check_many) == 0:
                count_still_running += 1
                print(f"Job in {folder_to_use} is still running or incomplete.")
            else:
                print(f"Job in {folder_to_use} did not complete successfully.")

    
    print(f"Total successful jobs: {count_success} / {count_folder}")
    print(f"Total still running jobs: {count_still_running} / {count_folder}")


def check_job_termination_whole(root_dir, df_multiplicity):
    
    job_list = df_multiplicity['molecule'].tolist()

    for job in job_list:
        folder_results = f"{root_dir}/{job}_done"
        # check if folder_results exists
        if os.path.exists(folder_results):
            # iterate and see if there is a file ending in .out or .log
            for file in os.listdir(folder_results):
                if file.endswith("logs"):
                    output_file = os.path.join(folder_results, file)
                    break

            # read last line of output_file
            with open(output_file, "r") as f:
                lines = f.readlines()
            # if last line contains "ORCA TERMINATED NORMALLY", then get geometry forces
            if "ORCA TERMINATED NORMALLY" not in lines[-2]:
                print(f"Job {job} did not terminate normally.")
