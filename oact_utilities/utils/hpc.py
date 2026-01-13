import os 

def write_flux_no_template_sella_ase(
    root_dir: str,
    two_step: bool = False,
    n_cores: int = 4,
    n_hours: int = 2,
    queue: str = "pbatch",
    allocation: str = "dnn-sim",
) -> None:

    base_lines = [
        "#!/bin/sh\n",
        "#flux: -N 1\n",
        f"#flux: -n {n_cores}\n",
        f"#flux: -q {queue}\n",
        f"#flux: -B {allocation}\n",
        f"#flux: -t {n_hours*60}m\n",
        "\n",
        "source ~/.bashrc\n",
        "conda activate py10mpi\n",
        "export LD_LIBRARY_PATH=/usr/WS1/vargas58/miniconda3/envs/py10mpi/lib:$LD_LIBRARY_PATH\n",
        "export JAX_PLATFORMS=cpu\n",
        "python orca.py\n",
    ]

    # create folder if it does not exist
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # go through each subfolder in root_directory and write a flux job for each, scan for .inp files and add them to last line of template
    for folder in os.listdir(root_dir):
        folder_to_use = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_to_use):
            continue

        inp_files = [
            os.path.join(folder_to_use, f)
            for f in os.listdir(folder_to_use)
            if f.endswith(".inp")
        ]
        if not inp_files:
            continue
        out_lines = base_lines.copy()
        with open(os.path.join(folder_to_use, "flux_job.flux"), "w") as fh:
            fh.writelines(out_lines)


def write_flux_no_template(
    root_dir: str,
    two_step: bool = False,
    n_cores: int = 4,
    n_hours: int = 2,
    queue: str = "pbatch",
    allocation: str = "dnn-sim",
) -> None:

    base_lines = [
        "#!/bin/sh\n",
        "#flux: -N 1\n",
        f"#flux: -n {n_cores}\n",
        f"#flux: -q {queue}\n",
        f"#flux: -B {allocation}\n",
        f"#flux: -t {n_hours*60}m\n",
        "\n",
        "source ~/.bashrc\n",
        "conda activate py10mpi\n",
        "export LD_LIBRARY_PATH=/usr/WS1/vargas58/miniconda3/envs/py10mpi/lib:$LD_LIBRARY_PATH\n",
        "/usr/workspace/vargas58/orca-6.1.0-f.0_linux_x86-64/bin/orca",
    ]

    # create folder if it does not exist
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # go through each subfolder in root_directory and write a flux job for each, scan for .inp files and add them to last line of template
    for folder in os.listdir(root_dir):
        folder_to_use = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_to_use):
            continue

        if two_step:
            loose_files = [
                os.path.join(folder_to_use, f)
                for f in os.listdir(folder_to_use)
                if f.endswith("omol_loose.inp")
            ]
            tight_files = [
                os.path.join(folder_to_use, f)
                for f in os.listdir(folder_to_use)
                if f.endswith("omol_tight.inp")
            ]

            # skip writing if no files found for that step
            if loose_files:
                out_lines = base_lines.copy()
                # ensure last line ends without newline so we can append input list
                if out_lines[-1].endswith("\n"):
                    out_lines[-1] = out_lines[-1][:-1]
                out_lines[-1] = out_lines[-1] + " " + " ".join(loose_files) + "\n"
                with open(
                    os.path.join(folder_to_use, "flux_job_loose.flux"), "w"
                ) as fh:
                    fh.writelines(out_lines)

            if tight_files:
                out_lines = base_lines.copy()
                if out_lines[-1].endswith("\n"):
                    out_lines[-1] = out_lines[-1][:-1]
                out_lines[-1] = out_lines[-1] + " " + " ".join(tight_files) + "\n"
                with open(
                    os.path.join(folder_to_use, "flux_job_tight.flux"), "w"
                ) as fh:
                    fh.writelines(out_lines)
        else:
            inp_files = [
                os.path.join(folder_to_use, f)
                for f in os.listdir(folder_to_use)
                if f.endswith(".inp")
            ]
            if not inp_files:
                continue
            out_lines = base_lines.copy()
            if out_lines[-1].endswith("\n"):
                out_lines[-1] = out_lines[-1][:-1]
            out_lines[-1] = out_lines[-1] + " " + " ".join(inp_files) + "\n"
            with open(os.path.join(folder_to_use, "flux_job.flux"), "w") as fh:
                fh.writelines(out_lines)


def write_slurm_no_template(
    root_dir: str,
    two_step: bool = False,
    n_cores: int = 4,
    n_hours: int = 2,
    queue: str = "pbatch",
    allocation: str = "dnn-sim",
    source_bashrc: str = "source ~/.bashrc",
    conda_env: str = "py10mpi",
    LD_LIBRARY_PATH: str = "/usr/WS1/vargas58/miniconda3/envs/py10mpi/lib",
    orca_command: str = "/usr/workspace/vargas58/orca-6.1.0-f.0_linux_x86-64/bin/orca",
) -> None:

    base_lines = [
        "#!/bin/sh\n",
        "#SBATCH -N 1\n",
        # f"#SBATCH -n {n_cores}\n", # optional, remove with None
        f"#SBATCH --constraint standard\n",
        f"#SBATCH --qos {queue}\n",
        f"#SBATCH --account {allocation}\n",
        f"#SBATCH -t {n_hours}:00:00\n",
        "\n",
        f"{source_bashrc}\n",
        f"conda activate {conda_env}\n",
        f"export LD_LIBRARY_PATH={LD_LIBRARY_PATH}:$LD_LIBRARY_PATH\n",
        orca_command,
    ]

    # create folder if it does not exist
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # go through each subfolder in root_directory and write a flux job for each, scan for .inp files and add them to last line of template
    for folder in os.listdir(root_dir):
        folder_to_use = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_to_use):
            continue

        if two_step:
            loose_files = [
                os.path.join(folder_to_use, f)
                for f in os.listdir(folder_to_use)
                if f.endswith("omol_loose.inp")
            ]
            tight_files = [
                os.path.join(folder_to_use, f)
                for f in os.listdir(folder_to_use)
                if f.endswith("omol_tight.inp")
            ]

            # skip writing if no files found for that step
            if loose_files:
                out_lines = base_lines.copy()
                # ensure last line ends without newline so we can append input list
                if out_lines[-1].endswith("\n"):
                    out_lines[-1] = out_lines[-1][:-1]
                out_lines[-1] = out_lines[-1] + " " + " ".join(loose_files) + "\n"
                with open(os.path.join(folder_to_use, "slurm_job_loose.sh"), "w") as fh:
                    fh.writelines(out_lines)

            if tight_files:
                out_lines = base_lines.copy()
                if out_lines[-1].endswith("\n"):
                    out_lines[-1] = out_lines[-1][:-1]
                out_lines[-1] = out_lines[-1] + " " + " ".join(tight_files) + "\n"
                with open(os.path.join(folder_to_use, "slurm_job_tight.sh"), "w") as fh:
                    fh.writelines(out_lines)
        else:
            inp_files = [
                os.path.join(folder_to_use, f)
                for f in os.listdir(folder_to_use)
                if f.endswith(".inp")
            ]
            if not inp_files:
                continue
            out_lines = base_lines.copy()
            if out_lines[-1].endswith("\n"):
                out_lines[-1] = out_lines[-1][:-1]
            out_lines[-1] = out_lines[-1] + " " + " ".join(inp_files) + "\n"
            with open(os.path.join(folder_to_use, "slurm_job.sh"), "w") as fh:
                fh.writelines(out_lines)


def write_flux(template_file: str, root_dir: str, two_step: bool = False) -> None:

    with open(template_file, "r") as f:
        lines = f.readlines()

    # remove lines that start with #* and * and lines in between
    lines_cleaned_template = []
    for line in lines:
        if not line.startswith("#") and not line.startswith("\n"):
            lines_cleaned_template.append(line)

    # create folder if it does not exist
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # go through each subfolder in root_directory and write a flux job for each, scan for .inp files and add them to last line of template
    for folder in os.listdir(root_dir):
        folder_to_use = os.path.join(root_dir, folder)
        if os.path.isdir(folder_to_use):
            if two_step:
                inp_files_loose = [
                    f for f in os.listdir(folder_to_use) if f.endswith("omol_loose.inp")
                ]
                inp_files_tight = [
                    f for f in os.listdir(folder_to_use) if f.endswith("omol_tight.inp")
                ]
                inp_files_loose_full_path = [
                    os.path.join(folder_to_use, f) for f in inp_files_loose
                ]
                inp_files_tight_full_path = [
                    os.path.join(folder_to_use, f) for f in inp_files_tight
                ]
                inp_files_loose_line = " ".join(inp_files_loose_full_path)
                inp_files_tight_line = " ".join(inp_files_tight_full_path)
                lines_cleaned_template_modified = lines_cleaned_template.copy()
                # rm \n from last line if present
                if lines_cleaned_template_modified[-1].endswith("\n"):
                    lines_cleaned_template_modified[-1] = (
                        lines_cleaned_template_modified[-1][:-1]
                    )
                lines_cleaned_template_modified_loose = (
                    lines_cleaned_template_modified.copy()
                )
                lines_cleaned_template_modified_tight = (
                    lines_cleaned_template_modified.copy()
                )

                lines_cleaned_template_modified_loose[-1] = (
                    lines_cleaned_template_modified_loose[-1]
                    + f" {inp_files_loose_line}\n"
                )
                lines_cleaned_template_modified_tight[-1] = (
                    lines_cleaned_template_modified_tight[-1]
                    + f" {inp_files_tight_line}\n"
                )
                file_name_loose = f"{folder_to_use}/flux_job_loose.flux"
                file_name_tight = f"{folder_to_use}/flux_job_tight.flux"
                with open(file_name_loose, "w") as f:
                    for line in lines_cleaned_template_modified_loose:
                        f.write(line)
                with open(file_name_tight, "w") as f:
                    for line in lines_cleaned_template_modified_tight:
                        f.write(line)
            else:

                inp_files = [f for f in os.listdir(folder_to_use) if f.endswith(".inp")]
                inp_files_full_path = [
                    os.path.join(folder_to_use, f) for f in inp_files
                ]
                inp_files_line = " ".join(inp_files_full_path)
                lines_cleaned_template_modified = lines_cleaned_template.copy()
                # rm \n from last line if present
                if lines_cleaned_template_modified[-1].endswith("\n"):
                    lines_cleaned_template_modified[-1] = (
                        lines_cleaned_template_modified[-1][:-1]
                    )
                lines_cleaned_template_modified[-1] = (
                    lines_cleaned_template_modified[-1] + f" {inp_files_line}\n"
                )

                file_name = f"{folder_to_use}/flux_job.flux"
                with open(file_name, "w") as f:
                    for line in lines_cleaned_template_modified:
                        f.write(line)

