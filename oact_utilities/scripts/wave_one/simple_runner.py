import subprocess
import os




def run_orca_and_rename(
    root_dir, orca_exe="~/Documents/orca_6_1_0_macosx_arm64_openmpi411/orca"
):
    orca_exe = os.path.expanduser(orca_exe)
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        # Skip if not a directory or already processed
        if not os.path.isdir(folder_path):
            continue
        # Find .inp file
        inp_files = [f for f in os.listdir(folder_path) if f.endswith("reload.inp")]
        if not inp_files:
            print(f"No .inp file found in {folder_path}")
            continue
        inp_file = inp_files[0]
        print(f"Running ORCA for {inp_file} in {folder_path}")
        # Change working directory for ORCA run
        cwd = os.getcwd()
        try:
            os.chdir(folder_path)
            subprocess.run(f'{orca_exe} "{inp_file}" > logs', shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running ORCA in {folder_path}: {e}")
            os.chdir(cwd)
            continue
        os.chdir(cwd)
        # Rename folder
        new_folder_path = folder_path + "_done"
        os.rename(folder_path, new_folder_path)
        print(f"Renamed {folder_path} to {new_folder_path}")


run_orca_and_rename("/Users/santiagovargas/dev/phase_two/")
