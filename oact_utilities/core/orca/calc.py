# modified from om-data: https://github.com/Open-Catalyst-Project/om-data

import re
import os
from enum import Enum
from shutil import which

from ase import Atoms
from ase.calculators.orca import ORCA, OrcaProfile
from ase.optimize import LBFGS
from sella import Sella

from oact_utilities.utils.create import read_xyz_from_orca


# ECP sizes taken from Table 6.5 in the Orca 5.0.3 manual
ECP_SIZE = {
    **{i: 28 for i in range(37, 55)},
    **{i: 46 for i in range(55, 58)},
    **{i: 28 for i in range(58, 72)},
    **{i: 60 for i in range(72, 87)},
}

# SV - change from def2 to def
ECP_SIZE = {
    **{i: 60 for i in range(87, 88)},
    **{i: 78 for i in range(89, 103)},
}

# number of basis sets
BASIS_DICT = {
    "H": 9,
    "He": 9,
    "Li": 17,
    "Be": 22,
    "B": 37,
    "C": 37,
    "N": 37,
    "O": 40,
    "F": 40,
    "Ne": 40,
    "Na": 35,
    "Mg": 35,
    "Al": 43,
    "Si": 43,
    "P": 43,
    "S": 46,
    "Cl": 46,
    "Ar": 46,
    "K": 36,
    "Ca": 36,
    "Sc": 48,
    "Ti": 48,
    "V": 48,
    "Cr": 48,
    "Mn": 48,
    "Fe": 48,
    "Co": 48,
    "Ni": 48,
    "Cu": 48,
    "Zn": 51,
    "Ga": 54,
    "Ge": 54,
    "As": 54,
    "Se": 57,
    "Br": 57,
    "Kr": 57,
    "Rb": 33,
    "Sr": 33,
    "Y": 40,
    "Zr": 40,
    "Nb": 40,
    "Mo": 40,
    "Tc": 40,
    "Ru": 40,
    "Rh": 40,
    "Pd": 40,
    "Ag": 40,
    "Cd": 40,
    "In": 56,
    "Sn": 56,
    "Sb": 56,
    "Te": 59,
    "I": 59,
    "Xe": 59,
    "Cs": 32,
    "Ba": 40,
    "La": 43,
    "Ce": 105,
    "Pr": 105,
    "Nd": 98,
    "Pm": 98,
    "Sm": 98,
    "Eu": 93,
    "Gd": 98,
    "Tb": 98,
    "Dy": 98,
    "Ho": 98,
    "Er": 101,
    "Tm": 101,
    "Yb": 96,
    "Lu": 96,
    "Hf": 43,
    "Ta": 43,
    "W": 43,
    "Re": 43,
    "Os": 43,
    "Ir": 43,
    "Pt": 43,
    "Au": 43,
    "Hg": 46,
    "Tl": 56,
    "Pb": 56,
    "Bi": 56,
    "Po": 59,
    "At": 59,
    "Rn": 59,
    "Ac": 105,
    "Th": 105,
    "Pa": 105,
    "U": 105,
    "Np": 105,
    "Pu": 105,
    "Am": 105,
    "Cm": 105,
    "Bk": 105,
    "Cf": 105,
    "Es": 105,
    "Fm": 105,
    "Md": 105,
    "No": 105,
    "Lr": 105,
}

ACTINIDE_LIST = [
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

# ORCA_FUNCTIONAL = ""
ORCA_BASIS = "def2-TZVPD"
ORCA_SIMPLE_INPUT = [
    "RIJCOSX",
    "def2/J",
    "NoUseSym",
    "DIIS",
    "NOSOSCF",
    "NormalConv",
    "DEFGRID3",
    "ALLPOP",
    "NoTRAH",  # SV - add for TM complexes
]

ORCA_SIMPLE_INPUT_X2C = [
    "DLU-X2C",
    "RIJCOSX",
    "AutoAux",
    "NoUseSym",
    "DIIS",
    "NOSOSCF",
    "NormalConv",
    "DEFGRID3",
    "ALLPOP",
    "NoTRAH",  # SV - add for TM complexes
]

ORCA_SIMPLE_INPUT_DK3 = [
    "DKH",
    "RIJCOSX",
    "SARC/J",
    "NoUseSym",
    "DIIS",
    "NOSOSCF",
    "NormalConv",
    "DEFGRID3",
    "ALLPOP",
    "NoTRAH",  # SV - add for TM complexes
]

ORCA_BLOCKS = [
    "%scf \n  Convergence Tight\n  maxiter 500\n  THRESH 1e-12\n  TCUT 1e-13\n  Shift Shift 0.1 ErrOff 0.1 end\n  DIISMaxEq   7\n  Guess PModel\nend",
    "%elprop Dipole true Quadrupole true end",
    "%output Print[P_ReducedOrbPopMO_L] 1 Print[P_ReducedOrbPopMO_M] 1 Print[P_BondOrder_L] 1 Print[P_BondOrder_M] 1 Print[P_Fockian] 1 Print[P_OrbEn] 2 end",
]

ORCA_BLOCKS_X2C = [
    "%rel \n  FiniteNuc     true\n  DLU         true\n  LightAtomThresh 0\nend",
    "%scf \n  Convergence Tight\n  maxiter 500\n  THRESH 1e-12\n  TCUT 1e-13\nShift Shift 0.1 ErrOff 0.1 end\nend",
    "%elprop Dipole true Quadrupole true end",
    "%output Print[P_ReducedOrbPopMO_L] 1 Print[P_ReducedOrbPopMO_M] 1 Print[P_BondOrder_L] 1 Print[P_BondOrder_M] 1 Print[P_Fockian] 1 Print[P_OrbEn] 2 end",
]

ORCA_BLOCKS_DK3 = [
    "%rel \n  FiniteNuc     true\nend\n",
    "%scf \n  Convergence Tight\n  maxiter 500\n  THRESH 1e-12\n  TCUT 1e-13\n. Shift Shift 0.1 ErrOff 0.1 end\nend",
    "%elprop Dipole true Quadrupole true end",
    "%output Print[P_ReducedOrbPopMO_L] 1 Print[P_ReducedOrbPopMO_M] 1 Print[P_BondOrder_L] 1 Print[P_BondOrder_M] 1 Print[P_Fockian] 1 Print[P_OrbEn] 2 end",
]

NBO_FLAGS = '%nbo NBOKEYLIST = "$NBO NPA NBO E2PERT 0.1 $END" end'  # SV - turn off??


LOOSE_OPT_PARAMETERS = {
    "optimizer": Sella,
    "store_intermediate_results": True,
    "fmax": 0.1,
    "max_steps": 100,
    "optimizer_kwargs": {
        "order": 0,
        "internal": True,
    },
}
OPT_PARAMETERS = {
    "optimizer": Sella,
    "store_intermediate_results": True,
    "fmax": 0.05,
    "max_steps": 100,
    "optimizer_kwargs": {
        "order": 0,
        "internal": True,
    },
}
TIGHT_OPT_PARAMETERS = {
    "optimizer": Sella,
    "store_intermediate_results": True,
    "fmax": 0.01,
    "max_steps": 100,
    "optimizer_kwargs": {
        "order": 0,
        "internal": True,
    },
}
EVAL_OPT_PARAMETERS = {
    "optimizer": LBFGS,
    "store_intermediate_results": True,
    "fmax": 0.01,
    "max_steps": 300,
    "optimizer_kwargs": {},
}

TS_OPT_PARAMETERS = {
    "optimizer": Sella,
    "store_intermediate_results": True,
    "fmax": 0.05,
    "max_steps": 200,
    "optimizer_kwargs": {
        "order": 1,
        "internal": True,
    },
}


class Vertical(Enum):
    Default = "default"
    MetalOrganics = "metal-organics"
    Oss = "open-shell-singlet"


def get_symm_break_block(atoms: Atoms, charge: int) -> str:
    """
    Determine the ORCA Rotate block needed to break symmetry in a singlet

    This is determined by taking the sum of atomic numbers less any charge (because
    electrons are negatively charged) and removing any electrons that are in an ECP
    and dividing by 2. This gives the number of occupied orbitals, but since ORCA is
    0-indexed, it gives the index of the LUMO.

    We use a rotation angle of 20 degrees or about a 12% mixture of LUMO into HOMO.
    This is somewhat arbitrary but similar to the default setting in Q-Chem, and seemed
    to perform well in tests of open-shell singlets.
    """
    n_electrons = sum(atoms.get_atomic_numbers()) - charge
    ecp_electrons = sum(
        ECP_SIZE.get(at_num, 0) for at_num in atoms.get_atomic_numbers()
    )
    n_electrons -= ecp_electrons
    lumo = n_electrons // 2
    return f"%scf rotate {{{lumo-1}, {lumo}, 20, 1, 1}} end end"


def get_n_basis(atoms: Atoms) -> int:
    """
    Get the number of basis functions that will be used for the given input.

    We assume our basis is def2-tzvpd. The number of basis functions is used
    to estimate the memory requirments of a given job.

    :param atoms: atoms to compute the number of basis functions of
    :return: number of basis functions as printed by Orca
    """
    nbasis = 0
    for elt in atoms.get_chemical_symbols():
        nbasis += BASIS_DICT[elt]
    return nbasis


def get_mem_estimate(
    atoms: Atoms, vertical: Enum = Vertical.Default, mult: int = 1
) -> int:
    """
    Get an estimate of the memory requirement for given input in MB.

    If the estimate is less than 1000MB, we return 1000MB.

    :param atoms: atoms to compute the number of basis functions of
    :param vertical: Which vertical this is for (all metal-organics are
                     UKS, as are all regular open-shell calcs)
    :param mult: spin multiplicity of input
    :return: estimated (upper-bound) to the memory requirement of this Orca job
    """
    nbasis = get_n_basis(atoms)
    if vertical == Vertical.Default and mult == 1:
        # Default RKS scaling as determined by PDB-ligand pockets in Orca6
        a = 0.0076739752343756434
        b = 361.4745947062764
    else:
        # Default UKS scaling as determined by metal-organics in Orca5
        a = 0.016460518374501867
        b = -320.38502508802776
    mem_est = int(max(a * nbasis**1.5 + b, 2000))
    return mem_est


def get_orca_blocks(
    atoms: Atoms,
    mult: int = 1,
    nbo: bool = True,
    cores: int = 12,
    opt: bool = False,
    simple_input: str = "omol",
    vertical: Enum = Vertical.Default,
    scf_MaxIter: int = None,
    functional: str = "wB97M-V",
    basis: str = None,
    actinide_basis: str = "ma-def-TZVP",
    actinide_ecp: str | None = None,
    non_actinide_basis: str = "def2-TZVPD",
    error_handle: bool = False,
    error_code: int = 0,

):


    if opt:
        job = 'Opt'
    else: 
        job = 'EnGrad'

    if simple_input == "omol":
        simple = ORCA_SIMPLE_INPUT.copy()
        # add opt or engrad at the start
        simple.insert(0, job)
        orcablocks = ORCA_BLOCKS.copy()
        if error_handle:
            # replace with looser scf settings if error code indicates scf failure
            if error_code == -1:
                print("Using looser SCF settings due to previous SCF failure.")
                # we know it's the first block we need to modify
                orcablocks[0] = re.sub(
                    r"Convergence Tight", f"Convergence Medium", orcablocks[0]
                )
                    
    elif simple_input == "x2c":
        simple = ORCA_SIMPLE_INPUT_X2C.copy()
        # add as second item in list after x2c
        simple.insert(1, job)
        orcablocks = ORCA_BLOCKS_X2C.copy()
        
        if error_handle:
            # replace with looser scf settings and pmodel guess if error code indicates scf failure
            if error_code == -1:
                print("Using looser SCF settings due to previous SCF failure + PModel guess.")
                # we know it's the second block we need to modify
                orcablocks[1] = "%scf \n  Convergence Medium\n  maxiter 500\n  THRESH 1e-12\n  TCUT 1e-13\n  DIISMaxEq   7\n  Guess PModel\n Shift Shift 0.1 ErrOff 0.1 end\nend",
                
            

    elif simple_input == "dk3":
        simple = ORCA_SIMPLE_INPUT_DK3.copy()
        simple.insert(1, job)
        
        orcablocks = ORCA_BLOCKS_DK3.copy()
        if error_handle:
            # replace with looser scf settings and pmodel guess if error code indicates scf failure
            if error_code == -1:
                print("Using looser SCF settings due to previous SCF failure + PModel guess.")
                # we know it's the second block we need to modify
                orcablocks[1] = "%scf \n  Convergence Medium\n  maxiter 500\n  THRESH 1e-12\n  TCUT 1e-13\n  DIISMaxEq   7\n  Guess PModel\n Shift Shift 0.1 ErrOff 0.1 end\nend",
                

    if basis is not None:
        orcasimpleinput = " ".join([functional] + [basis] + simple)
    else:
        orcasimpleinput = " ".join([functional] + [ORCA_BASIS] + simple)

    elem_set = set(atoms.get_chemical_symbols())

    orcablocks.append("%basis")

    for elem in elem_set:
        if elem in ACTINIDE_LIST:
            if os.path.isfile(actinide_basis):
                orcablocks.append(
                    f'  GTOName      = "{actinide_basis}"      # read orbital basis'
                )

            else:
                orcablocks.append(f'  NewGTO {elem} "{actinide_basis}" end')

            if actinide_ecp is not None:
                orcablocks.append(f'  NewECP {elem} "{actinide_ecp}" end')
        else:
            if os.path.isfile(non_actinide_basis):
                orcablocks.append(f'  GTOName      = "{non_actinide_basis}"')

            else:
                orcablocks.append(f'  NewGTO {elem} "{non_actinide_basis}" end')

    orcablocks.append(f"end")
    orcablocks.append(f"%pal\n nprocs " + str(cores) + "\nend")

    # Include estimate of memory needs
    mem_est = get_mem_estimate(atoms, vertical, mult)
    orcablocks.append(f"%maxcore {mem_est}")

    if not nbo:
        orcasimpleinput += " NONBO NONPA"
    else:
        orcablocks.append(f"{NBO_FLAGS}")


    if vertical in {Vertical.MetalOrganics, Vertical.Oss} and mult == 1:
        orcasimpleinput += " UKS"
        orcablocks.append(get_symm_break_block(atoms, charge=0))

    if scf_MaxIter:
        for block_line in orcablocks:
            if "maxiter 500" in block_line:
                index = orcablocks.index(block_line)
                orcablocks[index] = re.sub(
                    r"maxiter \d+", f"maxiter {scf_MaxIter}", block_line
                )
                break

    # print("orca_blocks: ", orcablocks)
    # print("orca_simple: ", orcasimpleinput)
    return orcasimpleinput, orcablocks


def write_orca_inputs(
    atoms: Atoms,
    output_directory,
    charge: int = 0,
    mult: int = 1,
    nbo: bool = False,
    cores: int = 12,
    opt: bool=False,
    functional: str = "wB97M-V",
    simple_input: str = "omol",
    orca_path: str = None,
    vertical: Enum = Vertical.Default,
    scf_MaxIter: int = None,
    actinide_basis: str = "ma-def-TZVP",
    actinide_ecp: str | None = None,
    non_actinide_basis: str = "def2-TZVPD",
    error_handle: bool = False,
    error_code: int = 0,
):
    """
    One-off method to be used if you wanted to write inputs for an arbitrary
    system. Primarily used for debugging.
    """

    if error_handle:
        if error_code == 0: # assume this is not a fresh calc and we need to pull atoms from orca.xyz
            # read in atoms from orca.xyz in output_directory
            print("Reading atoms from existing orca.xyz!")
            atoms, comment = read_xyz_from_orca(os.path.join(output_directory, "orca.xyz"))


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
        error_code=error_code
    )

    # print(orcablocks)

    mem_est = get_mem_estimate(atoms, vertical, mult)

    if orca_path is not None:
        MyOrcaProfile = OrcaProfile(command=orca_path)
    else:
        MyOrcaProfile = OrcaProfile([which("orca")])

    orca_blocks_as_str = "\n".join(orcablocks)
    # print(orca_blocks_as_str)

    calc = ORCA(
        charge=charge,
        mult=mult,
        profile=MyOrcaProfile,
        orcasimpleinput=orcasimpleinput,
        orcablocks=orca_blocks_as_str,
        directory=output_directory,
    )

    calc.write_inputfiles(atoms, ["energy", "forces"])
