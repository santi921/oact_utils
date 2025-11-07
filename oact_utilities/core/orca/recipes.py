# modified from om-data: https://github.com/Open-Catalyst-Project/om-data

from __future__ import annotations

import os

from oact_utilities.core.orca._base import run_and_summarize, run_and_summarize_opt
from quacc import change_settings

from oact_utilities.core.orca.calc import (
    Vertical,
    get_orca_blocks
)


def single_point_calculation(
    atoms,
    charge,
    spin_multiplicity,
    functional: str = "wB97M-V",
    simple_input: str = "omol",
    scf_MaxIter: int = None,
    actinide_basis: str = "ma-def-TZVP",
    actinide_ecp: str | None = None,
    non_actinide_basis: str = "def2-TZVPD",
    nprocs=12,
    outputdir=os.getcwd(),
    vertical=Vertical.Default,
    nbo=False,
    orca_cmd: str = "orca",
    copy_files=None,
    **calc_kwargs,
):
    """
    Wrapper around QUACC's static job to standardize single-point calculations.
    See github.com/Quantum-Accelerators/quacc/blob/main/src/quacc/recipes/orca/core.py#L22
    for more details.

    Arguments
    ---------

    atoms: Atoms
        Atoms object
    charge: int
        Charge of system
    spin_multiplicity: int
        Multiplicity of the system
    xc: str
        Exchange-correlaction functional
    basis: str
        Basis set
    orcasimpleinput: list
        List of `orcasimpleinput` settings for the calculator
    orcablocks: list
        List of `orcablocks` swaps for the calculator
    nprocs: int
        Number of processes to parallelize across
    nbo: bool
        Run NBO as part of the Orca calculation
    outputdir: str
        Directory to move results to upon completion
    calc_kwargs:
        Additional kwargs for the custom Orca calculator
    """

    default_inputs = [functional, non_actinide_basis, "engrad"]
    default_blocks = [f"%pal nprocs {nprocs} end"]
    
    orcasimpleinput, orcablocks = get_orca_blocks(
        atoms,
        nbo=nbo,
        cores=nprocs,
        vertical=vertical,
        scf_MaxIter=scf_MaxIter,
        mult=spin_multiplicity,
        functional=functional,
        simple_input=simple_input,
        actinide_basis=actinide_basis,
        actinide_ecp=actinide_ecp,
        non_actinide_basis=non_actinide_basis,
    )
    #print(orcasimpleinput)
    #print(orcablocks)
    
    with change_settings(
        {
            "ORCA_CMD": orca_cmd,
            "RESULTS_DIR": outputdir,
            "SCRATCH_DIR": outputdir,
        }
    ):
        doc = run_and_summarize(
            atoms,
            charge=charge,
            spin_multiplicity=spin_multiplicity,
            default_inputs=default_inputs,
            default_blocks=default_blocks,
            input_swaps=orcasimpleinput,
            block_swaps=orcablocks,
            copy_files=copy_files,
            **calc_kwargs,
        )

    return doc


#ase_relax_job / relax_job
def ase_relaxation(
    atoms,
    charge,
    spin_multiplicity,
    functional: str = "wB97M-V",
    simple_input: str = "omol",
    scf_MaxIter: int = None,
    actinide_basis: str = "ma-def-TZVP",
    actinide_ecp: str | None = None,
    non_actinide_basis: str = "def2-TZVPD",
    nprocs=12,
    opt_params=None,
    outputdir=os.getcwd(),
    vertical=Vertical.Default,
    copy_files=None,
    nbo=False,
    orca_cmd: str = "orca",
    step_counter_start=0,
    **calc_kwargs,
):
    """
    Wrapper around QUACC's ase_relax_job to standardize geometry optimizations.
    See github.com/Quantum-Accelerators/quacc/blob/main/src/quacc/recipes/orca/core.py#L22
    for more details.

    Arguments
    ---------

    atoms: Atoms
        Atoms object
    charge: int
        Charge of system
    spin_multiplicity: int
        Multiplicity of the system
    xc: str
        Exchange-correlaction functional
    basis: str
        Basis set
    orcasimpleinput: list
        List of `orcasimpleinput` settings for the calculator
    orcablocks: list
        List of `orcablocks` swaps for the calculator
    nprocs: int
        Number of processes to parallelize across
    opt_params: dict
        Dictionary of optimizer parameters
    nbo: bool
        Run NBO as part of the Orca calculation
    step_counter_start: int
        Index to start step counter from (used for optimization restarts)
    outputdir: str
        Directory to move results to upon completion
    calc_kwargs:
        Additional kwargs for the custom Orca calculator
    """


    orcasimpleinput, orcablocks = get_orca_blocks(
        atoms,
        nbo=nbo,
        cores=nprocs,
        vertical=vertical,
        scf_MaxIter=scf_MaxIter,
        mult=spin_multiplicity,
        functional=functional,
        simple_input=simple_input,
        actinide_basis=actinide_basis,
        actinide_ecp=actinide_ecp,
        non_actinide_basis=non_actinide_basis,
    )

    with change_settings(
        {
            "ORCA_CMD": orca_cmd,
            "RESULTS_DIR": outputdir,
            "SCRATCH_DIR": outputdir,
        }
    ):

        doc = run_and_summarize_opt(
            atoms,
            charge=charge,
            spin_multiplicity=spin_multiplicity,
            input_swaps=orcasimpleinput,
            block_swaps=orcablocks,
            opt_params=opt_params,
            copy_files=copy_files,
            step_counter_start=step_counter_start,
            **calc_kwargs,
        )

    return doc
