import os
import numpy as np
import pandas as pd
from ase import Atoms
import time 
import glob

from oact_utilities.utils.create import write_inputs_ase
from oact_utilities.core.orca.calc import write_orca_inputs
from oact_utilities.utils.hpc import (
    write_flux_no_template_sella_ase,
    write_flux_no_template,
)
from oact_utilities.utils.table_summary import _parse_npy_into_table
from oact_utilities.utils.status import check_job_termination, check_sella_complete


actinides_list = [
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
]
actinide_number = [89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103]
dict_number_to_symbol = {
    str(num): sym for num, sym in zip(actinide_number, actinides_list)
}
dict_symbol_to_number = {sym: num for num, sym in zip(actinide_number, actinides_list)}


def read_and_process_data(data_xlsx):
    # ensure to get all tabs of the excel file
    data_df = pd.read_excel(data_xlsx, sheet_name=None)
    # flatten dataframe to a single table
    data_df = pd.concat(data_df.values(), ignore_index=True)
    # remove all rows with only Nans
    data_df = data_df.dropna(how="all")
    # remove unnamed columns
    data_df = data_df.loc[:, ~data_df.columns.str.contains("^Unnamed")]
    # clean rows with missing multiplicity,
    list_data = [
        "CCDC ID",
        "Actinide",
        "Formula",
        "Charge ",
        "Multiplicity",
        "Multiplicity_List",
    ]
    data_df = data_df.dropna(subset=list_data)
    # convert multiplicity to int
    data_df["Multiplicity"] = data_df["Multiplicity"].astype(int)
    return data_df


def _has_valid_status(*rows):
    for row in rows:
        if row is None:
            continue
        if row.get("status", 0) in (1, True):
            return True
    return False


# helper: map element numbers/strings to element symbols
def elems_to_symbols(elems):
    syms = []
    for e in elems:
        try:
            if isinstance(e, (int, np.integer)):
                syms.append(dict_number_to_symbol.get(str(e), str(e)))
            else:
                try:
                    syms.append(dict_number_to_symbol.get(str(int(e)), str(e)))
                except Exception:
                    syms.append(str(e))
        except Exception:
            syms.append(str(e))
    return syms


# ensure elements are atomic numbers
def _ensure_atomic_numbers(elems):
    if elems is None:
        return None
    try:
        if len(elems) == 0:
            return elems
    except Exception:
        return None
    # convert strings to numbers if needed
    if isinstance(elems[0], (int, np.integer)):
        return elems
    try:
        return [dict_symbol_to_number.get(str(e), int(e)) for e in elems]
    except Exception:
        return None


def wrapper_write_job_folder(
    output_folder: str,
    atoms: Atoms,
    tf_sella: bool = False,
    n_cores: int = 8,
    n_hours: int = 24,
    queue: str = "pbatch",
    allocation: str = "default_allocation",
    charge: int = 0,
    mult: int = 1,
    functional: str = "wB97M-V",
    max_scf_iterations: int = 1000,
    lot: str = "omol",
    orca_exe: str = "/path/to/orca",
    actinide_basis: str = "ma-def-TZVP",
    actinide_ecp: str = "def-ECP",
    non_actinide_basis: str = "def2-TZVPD",
    error_code: int = 0,
    tight_two_e_int: bool = False,
    skip_done: bool = True,
    skip_running: bool = True,
) -> None:
    """
    Create job folders and write input files for either Sella/ASE or ORCA jobs.

    Args:
        output_folder (str): Path to the output folder where job files will be written.
        tf_sella (bool): Whether to use Sella/ASE for job creation. Defaults to False (ORCA).
        n_cores (int): Number of cores to use for the job. Defaults to 8.
        n_hours (int): Number of hours to allocate for the job. Defaults to 24.
        queue (str): Queue name for job submission. Defaults to "pbatch".
        allocation (str): Allocation name for job submission. Defaults to "default_allocation".
        atoms (Atoms): ASE Atoms object representing the molecular structure.
        charge (int): Molecular charge. Defaults to 0.
        mult (int): Spin multiplicity. Defaults to 1.
        functional (str): Functional to use for the calculation. Defaults to "wB97M-V".
        max_scf_iterations (int): Maximum number of SCF iterations. Defaults to 1000.
        lot (str): Level of theory (e.g., "omol" or "x2c"). Defaults to "omol".
        orca_exe (str): Path to the ORCA executable. Defaults to "/path/to/orca".
        actinide_basis (str): Basis set for actinides. Defaults to "ma-def-TZVP".
        actinide_ecp (str): Effective core potential for actinides. Defaults to "def-ECP".
        non_actinide_basis (str): Basis set for non-actinides. Defaults to "def2-TZVPD".
        error_code (int): Error handling code. Defaults to 0.
        tight_two_e_int (bool): Whether to use tight two-electron integrals. Defaults to False.

    Returns:
        None
    """
    # make sure output folder exists, don't overwrite if it already exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)


    # if skip_running is True, check if there is already a flux job running in the folder
    if skip_running:
        flux_file = os.path.join(output_folder, "flux_job.flux")
        if os.path.exists(flux_file):
            # only consider files matching flux-*.out when deciding to skip

            out_files = glob.glob(os.path.join(output_folder, "flux-*.out"))
            if out_files:
                latest_file = max(out_files, key=os.path.getmtime)
                latest_mtime = os.path.getmtime(latest_file)
                if (time.time() - latest_mtime) < 3600:  # 1 hour in seconds
                    print(
                        f"Skipping {output_folder} because a recent flux-*.out was modified within the last hour"
                    )
                    return


    if tf_sella:
        if skip_done:
            # check if folder has successful flux job
            if check_sella_complete(output_folder):
                print(f"Skipping {output_folder} - succcessful job found.")
                return

            if check_job_termination(output_folder) == -1:
                print(f"Skipping {output_folder} - failed job found.")
                return

        write_orca_inputs(
            atoms=atoms,
            output_directory=output_folder,
            charge=charge,
            mult=mult,
            nbo=False,
            cores=n_cores,
            functional=functional,
            scf_MaxIter=max_scf_iterations,
            simple_input=lot,
            orca_path=orca_exe,
            actinide_basis=actinide_basis,
            actinide_ecp=actinide_ecp,
            non_actinide_basis=non_actinide_basis,
            opt=False,
            error_handle=True,
            error_code=error_code,
            tight_two_e_int=tight_two_e_int,
        )

        traj_file = os.path.join(output_folder, "opt.traj")
        if not os.path.exists(traj_file):
            traj_file = None

        write_inputs_ase(
            output_directory=output_folder,
            cores=n_cores,
            restart=True,
            mult=mult,
            charge=charge,
            functional=functional,
            scf_MaxIter=max_scf_iterations,
            actinide_basis=actinide_basis,
            actinide_ecp=actinide_ecp,
            non_actinide_basis=non_actinide_basis,
            error_handle=True,
            error_code=error_code,
            tight_two_e_int=tight_two_e_int,
            nbo=False,
            simple_input=lot,
            orca_path=orca_exe,
            traj_file=traj_file,
        )

        write_flux_no_template_sella_ase(
            root_dir=output_folder,
            n_cores=n_cores,
            n_hours=n_hours,
            queue=queue,
            allocation=allocation,
            two_step=False,
        )

    else:
        if skip_done:
            if check_job_termination(output_folder):
                print(f"Skipping {output_folder} as it has a completed job.")
                return

        write_orca_inputs(
            atoms=atoms,
            output_directory=output_folder,
            charge=charge,
            mult=mult,
            nbo=False,
            cores=n_cores,
            functional=functional,
            scf_MaxIter=max_scf_iterations,
            simple_input=lot,
            orca_path=orca_exe,
            actinide_basis=actinide_basis,
            actinide_ecp=actinide_ecp,
            non_actinide_basis=non_actinide_basis,
            opt=True,
            error_handle=True,
            error_code=error_code,
            tight_two_e_int=tight_two_e_int,
        )

        write_flux_no_template(
            root_dir=output_folder,
            n_cores=n_cores,
            n_hours=n_hours,
            queue=queue,
            allocation=allocation,
            two_step=False,
        )


def _is_sella_path(path):
    return path is not None and "sella" in os.path.basename(path)


def _choose_best(main_row, alt_row):
    # main_row / alt_row are dict-like with keys: 'coords', 'elements_numbers', 'delta_energy', 'status'
    if main_row is None and alt_row is None:
        return None, None, None

    if main_row is None:
        return alt_row.get("coords", None), alt_row.get("elements_numbers", None), "alt"
    if alt_row is None:
        return (
            main_row.get("coords", None),
            main_row.get("elements_numbers", None),
            "main",
        )

    def _valid(r):
        if r is None:
            return False
        coords = r.get("coords", None)
        elems = r.get("elements_numbers", None)
        if coords is None or elems is None:
            return False
        try:
            return len(elems) > 0 and np.array(coords).size > 0
        except Exception:
            return False

    valid_main = _valid(main_row)
    valid_alt = _valid(alt_row)

    # Prefer a valid row
    if valid_main and not valid_alt:
        return main_row["coords"], main_row["elements_numbers"], "main"
    if valid_alt and not valid_main:
        return alt_row["coords"], alt_row["elements_numbers"], "alt"

    # If both valid, use delta_energy when available
    dm = main_row.get("delta_energy", None)
    da = alt_row.get("delta_energy", None)
    if dm is not None and da is not None:
        if da < dm:
            return alt_row["coords"], alt_row["elements_numbers"], "alt"
        else:
            return main_row["coords"], main_row["elements_numbers"], "main"

    # If neither has delta_energy or one missing, prefer the one that has coords
    if main_row.get("coords", None) is not None and alt_row.get("coords", None) is None:
        return main_row["coords"], main_row["elements_numbers"], "main"
    if alt_row.get("coords", None) is not None and main_row.get("coords", None) is None:
        return alt_row["coords"], alt_row["elements_numbers"], "alt"

    # final fallback: prefer main
    return main_row.get("coords", None), main_row.get("elements_numbers", None), "main"


def sanitize_key(key):
    # remove - or ECP from the key, and replace spaces with underscores
    key = key.replace("-", "").replace("ECP", "").replace("ZORA", "")
    return key


def reopt_for_different_spins(
    path_omol,
    path_x2c,
    path_omol_alt=None,
    path_x2c_alt=None,
    data_file_spin_lists=None,
    orca_exe="orca",
    output_dir="./output",
    **kwargs,
):

    dataframe_spin_lists = read_and_process_data(data_file_spin_lists)
    # Use parse_npy_into_table to build normalized tables for each path
    table_a_main = (
        _parse_npy_into_table(path_omol, sella=_is_sella_path(path_omol))
        if path_omol is not None
        else None
    )
    table_b_main = (
        _parse_npy_into_table(path_x2c, sella=_is_sella_path(path_x2c))
        if path_x2c is not None
        else None
    )
    table_a_alt = (
        _parse_npy_into_table(path_omol_alt, sella=_is_sella_path(path_omol_alt))
        if path_omol_alt is not None
        else None
    )
    table_b_alt = (
        _parse_npy_into_table(path_x2c_alt, sella=_is_sella_path(path_x2c_alt))
        if path_x2c_alt is not None
        else None
    )

    # index by name for quick lookup
    if table_a_main is not None:
        table_a_main = table_a_main.set_index("name")
    if table_b_main is not None:
        table_b_main = table_b_main.set_index("name")
    if table_a_alt is not None:
        table_a_alt = table_a_alt.set_index("name")
    if table_b_alt is not None:
        table_b_alt = table_b_alt.set_index("name")

    keys = set()
    for t in (table_a_main, table_b_main, table_a_alt, table_b_alt):
        if t is not None:
            keys |= set(t.index.tolist())

    name = kwargs.get("name", "category")
    print("-" * 60)
    print(f"Actinide bond distances for {name}:")

    missing_list = []
    for key in sorted(keys):
        # print('-'*40)
        print(f"Processing {key}...")

        main_a = (
            table_a_main.loc[key].to_dict()
            if (table_a_main is not None and key in table_a_main.index)
            else None
        )
        alt_a = (
            table_a_alt.loc[key].to_dict()
            if (table_a_alt is not None and key in table_a_alt.index)
            else None
        )
        main_b = (
            table_b_main.loc[key].to_dict()
            if (table_b_main is not None and key in table_b_main.index)
            else None
        )
        alt_b = (
            table_b_alt.loc[key].to_dict()
            if (table_b_alt is not None and key in table_b_alt.index)
            else None
        )

        invalid_reasons = []
        if not _has_valid_status(main_a, alt_a):
            invalid_reasons.append("omol")
        if not _has_valid_status(main_b, alt_b):
            invalid_reasons.append("x2c")

        if invalid_reasons:
            which = " and ".join(invalid_reasons)
            print(f"Skipping {key} due to no valid {which} run(s) (main/alt).")
            missing_list.append(key)
            continue

        coords_a, elems_a, which_a = _choose_best(main_a, alt_a)
        coords_b, elems_b, which_b = _choose_best(main_b, alt_b)

        if coords_a is None or coords_b is None:
            print(
                f"Skipping {key} due to missing coordinates. (omol: {which_a}, x2c: {which_b})"
            )
            missing_list.append(key)
            continue

        elems_a = _ensure_atomic_numbers(elems_a)
        elems_b = _ensure_atomic_numbers(elems_b)

        if elems_a is None or elems_b is None:
            print(f"Skipping {key} due to missing/invalid element numbering.")
            missing_list.append(key)
            continue

        # compute actinide positions and distances using elements_numbers
        try:
            pos_actindes_a = [
                np.array(coords_a)[i]
                for i, elem in enumerate(elems_a)
                if elem in actinide_number
            ]
            pos_actindes_b = [
                np.array(coords_b)[i]
                for i, elem in enumerate(elems_b)
                if elem in actinide_number
            ]

        except Exception as e:
            print("Error extracting actinide positions for", key, e)
            missing_list.append(key)
            continue

        if len(pos_actindes_a) == 0 or len(pos_actindes_b) == 0:
            print(f"No actinide atom found for {key}.")
            missing_list.append(key)
            continue

        # Determine which row and path were selected for omol/x2c

        used_omol_path = path_omol if which_a == "main" else path_omol_alt
        used_x2c_path = path_x2c if which_b == "main" else path_x2c_alt

        tf_sella_omol = _is_sella_path(used_omol_path)
        tf_sella_x2c = _is_sella_path(used_x2c_path)

        pos_a = np.array(coords_a)
        pos_b = np.array(coords_b)
        print(f"Selected OMOL for {key}: {which_a} (Sella: {tf_sella_omol})")
        print(f"Selected X2C for {key}: {which_b} (Sella: {tf_sella_x2c})")
        # print(f"Number of atoms for {key }: OMOL={len(sym_a)}, X2C={len(sym_b)}")

        atoms_a = Atoms(symbols=elems_a, positions=pos_a)
        atoms_b = Atoms(symbols=elems_b, positions=pos_b)

        # dummy charge/multiplicity for now (will be read from file later)
        charge_dummy = kwargs.get("charge", 0)
        mult_dummy = kwargs.get("mult", 1)
        mults_to_try = [mult_dummy]

        # get charge/multiplicity and multiplicity list from the data file if available
        if data_file_spin_lists is not None:
            df = dataframe_spin_lists
            row = df[df["Formula"] == sanitize_key(key)]
            if not row.empty:
                charge_dummy = int(row["Charge "].values[0])
                mult_dummy = int(row["Multiplicity"].values[0])
                raw_mult_list = row["Multiplicity_List"].values[0]

                # parse multiplicity list which may be stored as string like '1,3,5' or as a single integer
                if isinstance(raw_mult_list, (int, np.integer)):
                    mults_to_try = [int(raw_mult_list)]
                elif isinstance(raw_mult_list, str):
                    parts = [p.strip() for p in raw_mult_list.split(",") if p.strip()]
                    try:
                        mults_to_try = [int(p) for p in parts]
                    except Exception:
                        mults_to_try = [mult_dummy]
                elif hasattr(raw_mult_list, "__iter__"):
                    try:
                        mults_to_try = [int(x) for x in list(raw_mult_list)]
                    except Exception:
                        mults_to_try = [mult_dummy]
                else:
                    mults_to_try = [mult_dummy]

                print(
                    f"Using charge={charge_dummy} and multiplicity={mult_dummy} from data file for {sanitize_key(key)}"
                )
                print(f"Multiplicity list: {mults_to_try}")
            else:
                # skip this molecule if not found in the data file
                print(
                    f"...> No charge/multiplicity found in data file for {sanitize_key(key)}, using defaults."
                )
                continue

        # output structure: <output_dir>/(omol|x2c)/<category>/<molecule>/<spin>
        base_omol_out = os.path.join(output_dir, "omol", name, key)
        base_x2c_out = os.path.join(output_dir, "x2c", name, key)
        os.makedirs(base_omol_out, exist_ok=True)
        os.makedirs(base_x2c_out, exist_ok=True)

        actinide_basis = kwargs.get("actinide_basis", "ma-def-TZVP")
        actinide_ecp = kwargs.get("actinide_ecp", "def-ECP")
        non_actinide_basis = kwargs.get("non_actinide_basis", "def2-TZVPD")
        max_scf_iterations = kwargs.get("max_scf_iterations", 600)

        for spin in mults_to_try:
            # skip the spin that is equal to the original (we already have that run)
            if spin == mult_dummy:
                print(
                    f"Skipping spin {spin} for {key} (same as original multiplicity)."
                )
                continue

            omol_out = os.path.join(base_omol_out, f"spin_{spin}")
            x2c_out = os.path.join(base_x2c_out, f"spin_{spin}")
            os.makedirs(omol_out, exist_ok=True)
            os.makedirs(x2c_out, exist_ok=True)

            try:
                print(
                    f"Writing {'Sella' if tf_sella_omol else 'ORCA'} job for OMOL {key} (spin {spin}) -> {omol_out}"
                )
                wrapper_write_job_folder(
                    output_folder=omol_out,
                    atoms=atoms_a,
                    tf_sella=tf_sella_omol,
                    n_cores=kwargs.get("cores", 24),
                    n_hours=kwargs.get("n_hours", 24),
                    queue=kwargs.get("queue", "pbatch"),
                    allocation=kwargs.get("allocation", "dnn-sim"),
                    charge=charge_dummy,
                    mult=spin,
                    functional=kwargs.get("functional", "wB97M-V"),
                    max_scf_iterations=max_scf_iterations,
                    lot="omol",
                    orca_exe=orca_exe,
                    actinide_basis=actinide_basis,
                    actinide_ecp=actinide_ecp,
                    non_actinide_basis=non_actinide_basis,
                    error_code=0,
                    tight_two_e_int=kwargs.get("tight_two_e_int", False),
                    skip_done=kwargs.get("skip_done", True),
                    skip_running=kwargs.get("skip_running", True),
                )
            except Exception as e:
                print(f"Failed to write OMOL job for {key} (spin {spin}): {e}")

            try:
                print(
                    f"Writing {'Sella' if tf_sella_x2c else 'ORCA'} job for X2C {key} (spin {spin}) -> {x2c_out}"
                )

                # allow overriding X2C-specific basis/ECP via kwargs, otherwise keep defaults
                x2c_actinide_ecp = kwargs.get("actinide_ecp_x2c", None)
                x2c_non_actinide_basis = kwargs.get(
                    "non_actinide_basis_x2c", "X2C-TZVPPall"
                )
                x2c_actinide_basis = kwargs.get(
                    "actinide_basis_x2c",
                    "/usr/workspace/vargas58/orca_test/basis_sets/cc_pvtz_x2c.bas",
                )

                wrapper_write_job_folder(
                    output_folder=x2c_out,
                    atoms=atoms_b,
                    tf_sella=tf_sella_x2c,
                    n_cores=kwargs.get("cores", 24),
                    n_hours=kwargs.get("n_hours", 24),
                    queue=kwargs.get("queue", "pbatch"),
                    allocation=kwargs.get("allocation", "dnn-sim"),
                    charge=charge_dummy,
                    mult=spin,
                    functional=kwargs.get("functional_x2c", "PBE0"),
                    max_scf_iterations=max_scf_iterations,
                    lot="x2c",
                    orca_exe=orca_exe,
                    actinide_basis=x2c_actinide_basis,
                    actinide_ecp=x2c_actinide_ecp,
                    non_actinide_basis=x2c_non_actinide_basis,
                    error_code=0,
                    tight_two_e_int=kwargs.get("tight_two_e_int", False),
                    skip_done=kwargs.get("skip_done", True),
                    skip_running=kwargs.get("skip_running", True),
                )
            except Exception as e:
                print(f"Failed to write X2C job for {key} (spin {spin}): {e}")


def main():

    list_cats = [
        "Chalcogenides",
        "COT",
        "Nitrates",
        "tris-Cp",
        "carbenes",
        "Crown-Cryptands",
        "ChalcogenEthers",
        "Chalcogenides_soft",
        "Dithiocarbamates-dithiophosphates-dithiolates",
    ]

    omol_lot = "/usr/workspace/vargas58/multi_spin_data/omol/"
    x2c_lot = "/usr/workspace/vargas58/multi_spin_data/x2c/"
    omol_sella_lot = (
        "/usr/workspace/vargas58/multi_spin_data/omol_sella/"
    )
    x2c_sella_lot = "/usr/workspace/vargas58/multi_spin_data/x2c_sella/"
    data_file_spin_lists = (
        "/usr/workspace/vargas58/multi_spin_data/dataset.xlsx"
    )
    orca_exe = (
        "/usr/workspace/vargas58/orca_test/orca_6_2_1_linux_x86-64_openmpi411/orca"
    )
    output_dir = "/p/lustre5/vargas58/maria_benchmarks/multi_spin/"

    for cat in list_cats[1:]:
        test_x2c = os.path.join(x2c_lot, "x2c_{}.npy".format(cat))
        test_omol = os.path.join(omol_lot, "omol_{}.npy".format(cat))
        test_x2c_sella = os.path.join(x2c_sella_lot, "{}x2c_sella_.npy".format(cat))
        test_omol_sella = os.path.join(omol_sella_lot, "{}omol_sella_.npy".format(cat))

        reopt_for_different_spins(
            path_omol=test_omol,
            path_x2c=test_x2c,
            path_omol_alt=test_omol_sella,
            path_x2c_alt=test_x2c_sella,
            data_file_spin_lists=data_file_spin_lists,
            output_dir=output_dir,
            actinide_basis="ma-def-TZVP",
            actinide_ecp="def-ECP",
            non_actinide_basis="def2-TZVPD",
            functional="wB97M-V",
            functional_x2c="PBE0",
            max_scf_iterations=600,
            tight_two_e_int=False,
            cores=24,
            n_hours=24,
            orca_exe=orca_exe,
            job_handler="flux",
            name=cat,
            skip_done=True,
            skip_running=True,
        )


main()
