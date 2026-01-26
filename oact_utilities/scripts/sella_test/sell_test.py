import os
import time

from oact_utilities.core.orca.recipes import pure_ase_relaxation
from oact_utilities.utils.create import read_geom_from_inp_file
from oact_utilities.utils.status import check_job_termination, check_sella_complete


def main():
    os.environ["JAX_PLATFORMS"] = "cpu"
    inp_test = "/Users/santiagovargas/dev/oact_utils/oact_utilities/scripts/sella_test/orca.inp"
    atoms_orca = read_geom_from_inp_file(inp_test, ase_format_tf=True)
    charge = atoms_orca.charge
    mult = atoms_orca.spin
    orca_path = (
        "/Users/santiagovargas/Documents/orca_6_1_0_macosx_arm64_openmpi411/orca"
    )
    nbo_tf = False
    cores = 12
    actinide_basis = "ma-def-TZVP"
    actinide_ecp = "def-ECP"
    non_actinide_basis = "def2-TZVPD"
    time_start = time.time()
    output_directory = (
        "/Users/santiagovargas/dev/oact_utils/oact_utilities/scripts/sella_test/"
    )
    print(check_sella_complete(output_directory, fmax=0.05))
    print(check_job_termination(output_directory))

    _ = pure_ase_relaxation(
        atoms=atoms_orca,
        charge=charge,
        spin_multiplicity=mult,
        functional="wB97M-V",
        simple_input="omol",
        scf_MaxIter=1000,
        outputdir=output_directory,
        orca_cmd=orca_path,
        nbo=nbo_tf,
        nprocs=cores,
        actinide_basis=actinide_basis,
        actinide_ecp=actinide_ecp,
        traj_file="/Users/santiagovargas/dev/oact_utils/oact_utilities/scripts/sella_test/opt.traj",
        non_actinide_basis=non_actinide_basis,
        restart=True,
    )
    time_end = time.time()
    print("Total time (s): ", time_end - time_start)


if __name__ == "__main__":
    main()
