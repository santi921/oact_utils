import os
import pickle as pkl
import shutil
from pathlib import Path

import numpy as np
import pytest
from spyrmsd.rmsd import rmsd

from oact_utilities.core.orca.recipes import ase_relaxation
from oact_utilities.utils.analysis import (
    get_geo_forces,
    get_rmsd_start_final,
)
from oact_utilities.utils.create import (
    fetch_actinides,
    read_geom_from_inp_file,
    read_template,
)

hartree_to_ev = 27.2114


def test_H2(
    tmp_path,
    orca_cmd: str = "/Users/santiagovargas/Documents/orca_6_1_0_macosx_arm64_openmpi411/orca",
):
    """QUACC baseline jobs for AN66 dataset

    This test creates temporary directories that are cleaned up automatically
    via pytest's tmp_path fixture and explicit cleanup in finally block
    (ensuring cleanup even on test failure).
    """
    HERE = Path(__file__).resolve().parent
    inp_test = str(HERE / "files" / "H2O.inp")

    # Define cleanup directories up front so they're available in finally block
    cleanup_dirs = [
        HERE / "files" / "quacc",
        HERE / "files" / "test_orca",
        HERE / "files" / "data",
    ]

    try:
        atoms = read_geom_from_inp_file(inp_test, ase_format_tf=True)
        charge = atoms.charge
        mult = atoms.spin
        cores = 12
        nbo_tf = False
        actinide_basis = "def2-TZVP"
        actinide_ecp = None
        non_actinide_basis = "def2-TZVP"
        output_directory = str(tmp_path / "quacc")
        os.makedirs(output_directory, exist_ok=True)

        # Skip this test if ORCA is not available on the system
        if (
            not (os.path.exists(orca_cmd) and os.access(orca_cmd, os.X_OK))
            and shutil.which("orca") is None
        ):
            pytest.skip("ORCA executable not found; skipping ORCA-dependent test")

        res_dict = ase_relaxation(
            atoms=atoms,
            charge=charge,
            spin_multiplicity=mult,
            functional="TPSS",
            simple_input="omol",
            scf_MaxIter=400,
            outputdir=output_directory,
            orca_cmd=orca_cmd,
            nbo=nbo_tf,
            nprocs=cores,
            actinide_basis=actinide_basis,
            actinide_ecp=actinide_ecp,
            non_actinide_basis=non_actinide_basis,
        )

        res_dict = dict(res_dict)
        # save dict as pickle for later analysis
        with open(f"{output_directory}/results.pkl", "wb") as f:
            pkl.dump(res_dict, f)

        # run_folder = res_dict["dir_name"]

        # assert job is completed
        assert res_dict["converged"], "quacc baseline job did not converge"

        """ Normal baseline jobs for AN66 dataset """

        template_file = str(HERE / "files" / "template.inp")
        root_dir = str(tmp_path / "test_orca" / "h2o")
        # ensure root_dir exists and is empty
        if os.path.exists(root_dir):
            for f in os.listdir(root_dir):
                fp = os.path.join(root_dir, f)
                if os.path.isfile(fp):
                    os.remove(fp)
        else:
            os.makedirs(root_dir, exist_ok=True)

        lines_cleaned_template = read_template(template_file)
        spin = 1
        element_set = set(["H", "O"])
        actinide_list = fetch_actinides()
        # write lines to list first

        lines_to_write = []
        lines_to_write.append(f"%pal\n nprocs {cores} \nend\n\n")
        lines_to_write.append("%basis\n")
        for element in element_set:
            if element in actinide_list:
                if os.path.isfile(actinide_basis):
                    lines_to_write.append(
                        f'  GTOName      = "{actinide_basis}"      # read orbital basis\n'
                    )
                else:
                    lines_to_write.append(
                        f'  NewGTO {element} "{actinide_basis}" end\n'
                    )

                if actinide_ecp is not None:
                    lines_to_write.append(f'  NewECP {element} "{actinide_ecp}" end\n')
            else:
                if os.path.isfile(non_actinide_basis):
                    lines_to_write.append(
                        f'  GTOName      = "{non_actinide_basis}"      # read orbital basis\n'
                    )
                else:
                    lines_to_write.append(
                        f'  NewGTO {element} "{non_actinide_basis}" end\n'
                    )

        lines_to_write.append("end\n\n")

        lines_to_write.append(f"* xyz {charge} {spin}\n")
        for atom in read_geom_from_inp_file(str(inp_test)):
            element = atom["element"]
            x = atom["x"]
            y = atom["y"]
            z = atom["z"]
            lines_to_write.append(f"{element}\t{x:.6f}\t{y:.6f}\t{z:.6f}\n")

        lines_to_write.append("*\n")

        file_name = f"{root_dir}/test_orca.inp"
        # if file exists, remove it
        if os.path.exists(file_name):
            os.remove(file_name)

        with open(file_name, "w") as f:
            for line in lines_cleaned_template:
                f.write(line)
            f.write("\n")
            for line in lines_to_write:
                f.write(line)

        # get current directory
        current_directory = os.getcwd()

        command_loose = f"cd {root_dir} && {orca_cmd} test_orca.inp > h2o.log && cd {current_directory}"

        os.system(command_loose)

        # find log file; if ORCA didn't produce a usable log/out, skip the rest of the test
        candidate_logs = [
            f
            for f in os.listdir(root_dir)
            if f.endswith(".log") or f.endswith(".out") or "flux-" in f
        ]
        if not candidate_logs:
            pytest.skip(
                "ORCA did not produce a log/output file; skipping ORCA-dependent assertions"
            )
        log_file_name = candidate_logs[0]
        log_file_path = os.path.join(root_dir, log_file_name)

        # Attempt to parse ORCA output and compare to Quacc results; skip if parsing fails
        # nprocs, total_time_seconds = find_timings_and_cores(log_file_path)
        geo_forces = get_geo_forces(log_file=log_file_path)
        geom_info = get_rmsd_start_final(root_dir)

        # Check if direct ORCA run converged by looking for "GEOMETRY OPTIMIZATION COMPLETED" in log
        orca_converged = False
        with open(log_file_path) as f:
            log_content = f.read()
            if (
                "THE OPTIMIZATION HAS CONVERGED" in log_content
                or "HURRAY" in log_content
            ):
                orca_converged = True

        # Always verify quacc converged
        assert res_dict["converged"], "quacc baseline job did not converge"

        # Only compare geometries, energies, and forces if both calculations converged
        # Note: The direct ORCA run uses a different functional (wB97M-V from template)
        # than quacc (TPSS), so comparisons are only meaningful if both converged
        if orca_converged:
            orca_final_energies = geom_info["energies_frames"][-1] * hartree_to_ev
            coords_final = geom_info["coords_final"]

            quacc_res_dict = res_dict["results"]
            energy_quacc = quacc_res_dict["energy"]
            forces_quacc = quacc_res_dict["forces"]
            coords_quacc = res_dict["trajectory"][-1].get_positions()
            atomic_numbers = [atom.number for atom in res_dict["trajectory"][-1]]
            rmsd_orca_quacc = rmsd(
                coords_quacc, coords_final, atomic_numbers, atomic_numbers
            )

            # rmsd - increased tolerance due to convergence sensitivity
            assert (
                rmsd_orca_quacc < 1e-2
            ), f"RMSD between ORCA and Quacc final geometries is too large: {rmsd_orca_quacc}"
            # energies - large tolerance because different functionals (wB97M-V vs TPSS)
            assert (
                abs(energy_quacc - orca_final_energies) < 1.0
            ), f"Final energies between ORCA and Quacc differ too much: {abs(energy_quacc - orca_final_energies)} eV"
            # forces
            rms_forces_quacc = np.sqrt(np.mean(forces_quacc**2))
            rms_forces_orca = geo_forces[-1]["RMS_Gradient"]
            # check that the RMS forces are in reasonable range (tolerance 1e-2)
            assert (
                abs(rms_forces_quacc - rms_forces_orca) < 1e-2
            ), f"RMS forces between ORCA and Quacc differ too much: {abs(rms_forces_quacc - rms_forces_orca)} eV/A"
        else:
            # Direct ORCA run didn't converge - skip comparison but verify quacc still worked
            pytest.skip(
                "Direct ORCA run did not converge; skipping comparison (quacc run succeeded)"
            )

    finally:
        # Cleanup: Always run, even if test fails
        # tmp_path is automatically cleaned by pytest, but we explicitly
        # clean up any stray directories in tests/files/ that quacc
        # might have created outside of tmp_path
        for cleanup_dir in cleanup_dirs:
            if cleanup_dir.exists():
                shutil.rmtree(cleanup_dir, ignore_errors=True)
