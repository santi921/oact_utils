import os

from oact_utilities.core.orca.calc import write_orca_inputs
from oact_utilities.utils.an66 import process_geometry_file, process_multiplicity_file
from oact_utilities.utils.hpc import write_flux_no_template


def write_flux_orca_an66(
    actinide_basis: str,
    actinide_ecp: str,
    non_actinide_basis: str,
    orca_exe: str,
    root_dir: str,
    cores: int,
    safety: bool = True,
    n_hours: int = 24,
    allocation: str = "dnn-sim",
    two_step: bool = False,
    queue: str = "pbatch",
    ref_geom_file: str = "/Users/santiagovargas/dev/oact_utils/data/data/ref_geoms.txt",
    ref_multiplicity_file: str = "/Users/santiagovargas/dev/oact_utils/data/data/ref_multiplicity.txt",
):

    # throw fit if ref geom or multiplicity files don't exist
    if not os.path.exists(ref_geom_file):
        raise FileNotFoundError(
            f"Reference geometry file {ref_geom_file} does not exist."
        )
    if not os.path.exists(ref_multiplicity_file):
        raise FileNotFoundError(
            f"Reference multiplicity file {ref_multiplicity_file} does not exist."
        )
    # make folder if not there

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    dict_geoms = process_geometry_file(ref_geom_file, ase_format_tf=True)
    df_multiplicity = process_multiplicity_file(ref_multiplicity_file)
    print("Combining geometry and multiplicity data...")
    print(f"Number of geometries found: {len(dict_geoms)}")
    print(f"Geometries keys: {list(dict_geoms.keys())[:5]}")
    print(f"Number of multiplicities found: {len(df_multiplicity)}")
    print(f"Multiplicity keys: {list(df_multiplicity.molecule.tolist())[:5]}")
    dict_unified = {
        k: {
            "geometry": dict_geoms[k],
            "multiplicity": df_multiplicity[df_multiplicity.molecule == k][
                "multiplicity"
            ].values[0],
        }
        for k in dict_geoms.keys()
        if k in df_multiplicity.molecule.tolist()
    }
    print(f"Number of unified entries: {len(dict_unified)}")

    count = 0
    for mol_name in dict_unified.keys():
        print(f"Preparing job for molecule: {mol_name}")
        folder_to_use = os.path.join(root_dir, mol_name)
        if not os.path.exists(folder_to_use):
            os.mkdir(folder_to_use)
        print(f"Using calculation folder: {folder_to_use}")

        write_orca_inputs(
            atoms=dict_unified[mol_name]["geometry"],
            output_directory=folder_to_use,
            charge=0,
            mult=dict_unified[mol_name]["multiplicity"],
            nbo=False,
            cores=cores,
            functional="wB97M-V",
            scf_MaxIter=600,
            simple_input="omol",
            orca_path=orca_exe,
            actinide_basis=actinide_basis,
            actinide_ecp=actinide_ecp,
            non_actinide_basis=non_actinide_basis,
        )
        count += 1

    if safety:
        cores += 2

    write_flux_no_template(
        root_dir=root_dir,
        two_step=two_step,
        n_cores=cores,
        n_hours=n_hours,
        queue=queue,
        allocation=allocation,
    )


if __name__ == "__main__":

    cores = 10
    two_step = None
    n_hours = 4
    ref_geom_file = "/Users/santiagovargas/dev/oact_utils/data/data/ref_geoms.txt"
    ref_multiplicity_file = (
        "/Users/santiagovargas/dev/oact_utils/data/data/ref_multiplicity.txt"
    )
    ################################## OMOL BLOCK ##################################

    # 1) baseline omol
    actinide_basis = "ma-def-TZVP"
    actinide_ecp = "def-ECP"
    non_actinide_basis = "def2-TZVPD"
    root_directory = "/usr/workspace/vargas58/orca_test/an66_omol_sweep/omol/"
    root_directory = "/Users/santiagovargas/dev/oact_utils/data/an66_new"
    orca_exe = "/Users/santiagovargas/Documents/orca_6_1_0_macosx_arm64_openmpi411/orca"

    write_flux_orca_an66(
        actinide_basis=actinide_basis,
        actinide_ecp=actinide_ecp,
        non_actinide_basis=non_actinide_basis,
        ref_geom_file=ref_geom_file,
        ref_multiplicity_file=ref_multiplicity_file,
        two_step=two_step,
        cores=cores,
        safety=False,
        n_hours=n_hours,
        allocation="dnn-sim",
        queue="pbatch",
        root_dir=root_directory,
        orca_exe=orca_exe,
    )


# root_dirs
# root_directory = "/usr/workspace/vargas58/orca_test/an66_omol_sweep/omol/"
# root_directory = "/usr/workspace/vargas58/orca_test/an66_omol_sweep/omol_600_pmodel_shift/"
# root_directory = "/usr/workspace/vargas58/orca_test/an66_omol_sweep/omol_600_pmodel_shift_non_ma/"
# root_directory = "/usr/workspace/vargas58/orca_test/an66_benchmarks/x2c_pvtz_omol_M062x/"
# root_directory = "/usr/workspace/vargas58/orca_test/an66_benchmarks/x2c_pvtz_omol_tpss/"
# root_directory = "/usr/workspace/vargas58/orca_test/an66_benchmarks/x2c_pvtz_omol_pbe0/"
# root_directory = "/usr/workspace/vargas58/orca_test/an66_benchmarks/x2c_pwcvtz_omol_M062x/"
# root_directory = "/usr/workspace/vargas58/orca_test/an66_benchmarks/dk3_omol_M062x/"
# root_directory = "/usr/workspace/vargas58/orca_test/an66_benchmarks/dk3_omol_pbe0/"
# root_directory = "/usr/workspace/vargas58/orca_test/an66_benchmarks/dk3_omol_tpss/"
