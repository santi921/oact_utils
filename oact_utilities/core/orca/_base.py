"""Base jobs for ORCA."""

from __future__ import annotations

from typing import TYPE_CHECKING
from enum import Enum
import os 

from ase.calculators.orca import ORCA, OrcaProfile, OrcaTemplate

from oact_utilities.core.orca import calc
from quacc import get_settings
from quacc.runners.ase import Runner
from quacc.schemas.ase import Summarize
from quacc.utils.dicts import recursive_dict_merge

from oact_utilities.core.orca.calc import (
    read_xyz_from_orca, 
    get_mem_estimate, 
    get_orca_blocks,
    Vertical
)


if TYPE_CHECKING:
    from typing import Any

    from ase.atoms import Atoms

    from quacc.types import Filenames, OptParams, OptSchema, RunSchema, SourceDirectory

_LABEL = OrcaTemplate()._label  # skipcq: PYL-W0212
GEOM_FILE = f"{_LABEL}.xyz"


def run_and_summarize(
    atoms: Atoms,
    charge: int = 0,
    spin_multiplicity: int = 1,
    default_inputs: list[str] | None = None,
    default_blocks: list[str] | None = None,
    input_swaps: list[str] | None = None,
    block_swaps: list[str] | None = None,
    additional_fields: dict[str, Any] | None = None,
    copy_files: SourceDirectory | dict[SourceDirectory, Filenames] | None = None,
    **calc_kwargs,
) -> RunSchema:
    """
    Base job function for ORCA recipes.

    Parameters
    ----------
    atoms
        Atoms object
    charge
        Charge of the system.
    spin_multiplicity
        Multiplicity of the system.
    default_inputs
        Default input parameters.
    default_blocks
        Default block input parameters.
    input_swaps
        List of orcasimpleinput swaps for the calculator. To remove entries
        from the defaults, put a `#` in front of the name.
    block_swaps
        List of orcablock swaps for the calculator. To remove entries
        from the defaults, put a `#` in front of the name.
    additional_fields
        Any additional fields to supply to the summarizer.
    copy_files
        Files to copy (and decompress) from source to the runtime directory.
    **calc_kwargs
        Any other keyword arguments to pass to the `ORCA` calculator.

    Returns
    -------
    RunSchema
        Dictionary of results
    """
    calc = prep_calculator(
        charge=charge,
        spin_multiplicity=spin_multiplicity,
        default_inputs=default_inputs,
        default_blocks=default_blocks,
        input_swaps=input_swaps,
        block_swaps=block_swaps,
        **calc_kwargs,
    )

    final_atoms = Runner(atoms, calc, copy_files=copy_files).run_calc()

    return Summarize(additional_fields=additional_fields).run(final_atoms, atoms)


def run_and_summarize_opt(
    atoms: Atoms,
    charge: int = 0,
    spin_multiplicity: int = 1,
    default_inputs: list[str] | None = None,
    default_blocks: list[str] | None = None,
    input_swaps: list[str] | None = None,
    block_swaps: list[str] | None = None,
    opt_defaults: dict[str, Any] | None = None,
    opt_params: OptParams | None = None,
    additional_fields: dict[str, Any] | None = None,
    copy_files: SourceDirectory | dict[SourceDirectory, Filenames] | None = None,
    **calc_kwargs,
) -> OptSchema:
    """
    Base job function for ORCA recipes with ASE optimizer.

    Parameters
    ----------
    atoms
        Atoms object
    charge
        Charge of the system.
    spin_multiplicity
        Multiplicity of the system.
    default_inputs
        Default input parameters.
    default_blocks
        Default block input parameters.
    input_swaps
        List of orcasimpleinput swaps for the calculator. To remove entries
        from the defaults, put a `#` in front of the name.
    block_swaps
        List of orcablock swaps for the calculator. To remove entries
        from the defaults, put a `#` in front of the name.
    opt_defaults
        Default arguments for the ASE optimizer.
    opt_params
        Dictionary of custom kwargs for [quacc.runners.ase.Runner.run_opt][]
    additional_fields
        Any additional fields to supply to the summarizer.
    copy_files
        Files to copy (and decompress) from source to the runtime directory.
    **calc_kwargs
        Any other keyword arguments to pass to the `ORCA` calculator.

    Returns
    -------
    OptSchema
        Dictionary of results
    """
    calc = prep_calculator(
        charge=charge,
        spin_multiplicity=spin_multiplicity,
        input_swaps=input_swaps,
        block_swaps=block_swaps,
        **calc_kwargs,
    )

    opt_flags = recursive_dict_merge(opt_defaults, opt_params)
    dyn = Runner(atoms, calc, copy_files=copy_files).run_opt(**opt_flags)
    return Summarize(additional_fields=additional_fields).opt(dyn)


def prep_calculator(
    charge: int = 0,
    spin_multiplicity: int = 1,
    input_swaps: list[str] | None = None,
    block_swaps: list[str] | None = None,
    **calc_kwargs,
) -> ORCA:
    """
    Prepare the ORCA calculator.

    Parameters
    ----------
    charge
        Charge of the system.
    spin_multiplicity
        Multiplicity of the system.
    input_swaps
        List of orcasimpleinput swaps for the calculator. To remove entries
        from the defaults, put a `#` in front of the name.
    block_swaps
        List of orcablock swaps for the calculator. To remove entries
        from the defaults, put a `#` in front of the name.
    **calc_kwargs
        Any other keyword arguments to pass to the `ORCA` calculator.

    Returns
    -------
    ORCA
        The ORCA calculator
    """

    orcasimpleinput = input_swaps
    orcablocks = "\n".join(block_swaps)
    settings = get_settings()

    return ORCA(
        profile=OrcaProfile(command=settings.ORCA_CMD),
        charge=charge,
        mult=spin_multiplicity,
        orcasimpleinput=orcasimpleinput,
        orcablocks=orcablocks,
        **calc_kwargs,
    )

#### WAVE 2 BENCHMARK SPECIFIC FUNCTIONS BELOW ####

def run_and_summarize_opt_wave2(
    atoms: Atoms,
    output_directory,
    charge: int = 0,
    mult: int = 1,
    nbo: bool = False,
    cores: int = 12,
    opt: bool = False,
    functional: str = "wB97M-V",
    simple_input: str = "omol",
    orca_path: str = None,
    vertical: Enum = Vertical.Default,
    scf_MaxIter: int = None,
    actinide_basis: str = "ma-def-TZVP",
    actinide_ecp: str | None = None,
    non_actinide_basis: str = "def2-TZVPD",
    tight_two_e_int: bool = False,
    error_handle: bool = False,
    error_code: int = 0,
    #spin_multiplicity: int = 1,
    opt_defaults: dict[str, Any] | None = None,
    opt_params: OptParams | None = None,
    additional_fields: dict[str, Any] | None = None,
    copy_files: SourceDirectory | dict[SourceDirectory, Filenames] | None = None,
    **calc_kwargs,
) -> OptSchema:
    """
    Base job function for ORCA recipes with ASE optimizer.

    Parameters
    ----------
    atoms
        Atoms object
    charge
        Charge of the system.
    spin_multiplicity
        Multiplicity of the system.
    default_inputs
        Default input parameters.
    default_blocks
        Default block input parameters.
    input_swaps
        List of orcasimpleinput swaps for the calculator. To remove entries
        from the defaults, put a `#` in front of the name.
    block_swaps
        List of orcablock swaps for the calculator. To remove entries
        from the defaults, put a `#` in front of the name.
    opt_defaults
        Default arguments for the ASE optimizer.
    opt_params
        Dictionary of custom kwargs for [quacc.runners.ase.Runner.run_opt][]
    additional_fields
        Any additional fields to supply to the summarizer.
    copy_files
        Files to copy (and decompress) from source to the runtime directory.
    **calc_kwargs
        Any other keyword arguments to pass to the `ORCA` calculator.

    Returns
    -------
    OptSchema
        Dictionary of results
    """
    calc = prep_calculator_wave2(
        atoms=atoms,
        output_directory=output_directory,
        charge=charge,
        mult=mult,
        nbo=nbo,
        cores=cores,
        opt=opt,
        functional=functional,
        simple_input=simple_input,
        orca_path=orca_path,
        vertical=vertical,
        scf_MaxIter=scf_MaxIter,
        actinide_basis=actinide_basis,
        actinide_ecp=actinide_ecp,
        non_actinide_basis=non_actinide_basis,
        tight_two_e_int=tight_two_e_int,
        error_handle=error_handle,
        error_code=error_code,
    )

    opt_flags = recursive_dict_merge(opt_defaults, opt_params)
    dyn = Runner(atoms, calc, copy_files=copy_files).run_opt(**opt_flags)
    return Summarize(additional_fields=additional_fields).opt(dyn)


def prep_calculator_wave2(
    atoms: Atoms,
    output_directory,
    charge: int = 0,
    mult: int = 1,
    nbo: bool = False,
    cores: int = 12,
    opt: bool = False,
    functional: str = "wB97M-V",
    simple_input: str = "omol",
    orca_path: str = None,
    vertical: Enum = Vertical.Default,
    scf_MaxIter: int = None,
    actinide_basis: str = "ma-def-TZVP",
    actinide_ecp: str | None = None,
    non_actinide_basis: str = "def2-TZVPD",
    tight_two_e_int: bool = False,
    error_handle: bool = False,
    error_code: int = 0,
):
    """
    One-off method to be used if you wanted to write inputs for an arbitrary
    system. Primarily used for debugging.
    """
    if error_handle:
        if (
            error_code == 0
        ):  # assume this is not a fresh calc and we need to pull atoms from orca.xyz
            # read in atoms from orca.xyz in output_directory
            if os.path.exists(os.path.join(output_directory, "orca.xyz")):
                print("Reading atoms from existing orca.xyz!")
                atoms, comment = read_xyz_from_orca(
                    os.path.join(output_directory, "orca.xyz")
                )

    orcasimpleinput, orcablocks = get_orca_blocks(
        atoms=atoms,
        nbo=nbo,
        cores=cores,
        opt=opt,
        vertical=vertical,
        scf_MaxIter=scf_MaxIter,
        mult=mult,
        functional=functional,
        simple_input=simple_input,
        actinide_basis=actinide_basis,
        actinide_ecp=actinide_ecp,
        non_actinide_basis=non_actinide_basis,
        error_handle=error_handle,
        error_code=error_code,
        tight_two_e_int=tight_two_e_int
    )

    # print(orcablocks)

    mem_est = get_mem_estimate(atoms, vertical, mult)

    if orca_path is not None:
        MyOrcaProfile = OrcaProfile(command=orca_path)
    else:
        MyOrcaProfile = OrcaProfile([which("orca")])

    orca_blocks_as_str = "\n".join(orcablocks)
    # print(orca_blocks_as_str)

    return ORCA(
        charge=charge,
        mult=mult,
        profile=MyOrcaProfile,
        orcasimpleinput=orcasimpleinput,
        orcablocks=orca_blocks_as_str,
        directory=output_directory,
    )
