"""Basis-set size utilities.

Holds the per-element basis-function table and the helper that sums it for a
structure. This lives in ``utils`` rather than ``core.orca.calc`` so that
lightweight callers (workflow DB creation) can count basis functions without
importing ``calc``, which pulls in ``sella`` and ``ase.calculators.orca`` at
module import time.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, overload

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
    # Period 7 s-block (analogues of Cs=32, Ba=40)
    "Fr": 32,
    "Ra": 40,
    # Actinides
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
    # Transactinides (6d: Rf-Cn analogues of Hf-Hg; 7p: Nh-Og analogues of Tl-Rn)
    "Rf": 43,
    "Db": 43,
    "Sg": 43,
    "Bh": 43,
    "Hs": 43,
    "Mt": 43,
    "Ds": 43,
    "Rg": 43,
    "Cn": 46,
    "Nh": 56,
    "Fl": 56,
    "Mc": 56,
    "Lv": 59,
    "Ts": 59,
    "Og": 59,
}


@overload
def count_basis_functions(
    symbols: Iterable[str], strict: Literal[True] = ...
) -> int: ...


@overload
def count_basis_functions(
    symbols: Iterable[str], strict: Literal[False]
) -> int | None: ...


def count_basis_functions(symbols: Iterable[str], strict: bool = True) -> int | None:
    """Sum the per-element basis-function counts for a set of atoms.

    Args:
        symbols: Element symbols, e.g. ``["O", "H", "H"]``.
        strict: If True, raise on a symbol missing from ``BASIS_DICT``. If
            False, return None instead so that a single unparseable structure
            cannot abort a bulk conversion.

    Returns:
        Total number of basis functions, or None when ``strict`` is False and
        a symbol is not in the table.

    Raises:
        KeyError: If ``strict`` is True and a symbol is not in the table.
    """
    total = 0
    for symbol in symbols:
        try:
            total += BASIS_DICT[symbol]
        except KeyError:
            if strict:
                raise
            return None
    return total
