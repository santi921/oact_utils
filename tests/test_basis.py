"""Tests for basis-function counting (oact_utilities/utils/basis.py)."""

import pytest
from ase import Atoms

from oact_utilities.core.orca.calc import BASIS_DICT as CALC_BASIS_DICT
from oact_utilities.core.orca.calc import get_n_basis
from oact_utilities.utils.basis import BASIS_DICT, count_basis_functions


def _atoms(symbols):
    """Build an Atoms object with arbitrary (unused) positions."""
    return Atoms(
        symbols=symbols, positions=[(i, 0.0, 0.0) for i in range(len(symbols))]
    )


class TestCountBasisFunctions:
    """Counting behavior and agreement with the calc.py entry point."""

    def test_water(self):
        # O=40, H=9, H=9
        assert count_basis_functions(["O", "H", "H"]) == 58

    def test_empty(self):
        assert count_basis_functions([]) == 0

    def test_calc_reexport_is_same_object(self):
        """calc.BASIS_DICT must stay importable for existing callers."""
        assert CALC_BASIS_DICT is BASIS_DICT

    @pytest.mark.parametrize(
        "symbols",
        [
            ["O", "H", "H"],
            ["Np", "F", "F", "F"],
            ["Am"],
            ["U"] + ["C"] * 20 + ["H"] * 19,
        ],
    )
    def test_agrees_with_get_n_basis(self, symbols):
        """get_n_basis delegates here, so the two must never diverge."""
        assert get_n_basis(_atoms(symbols)) == count_basis_functions(symbols)

    def test_covers_every_element_in_table(self):
        """The whole table sums without a KeyError (guards a truncated move)."""
        symbols = sorted(BASIS_DICT)
        assert count_basis_functions(symbols) == sum(BASIS_DICT.values())

    def test_strict_raises_on_unknown_symbol(self):
        with pytest.raises(KeyError):
            count_basis_functions(["O", "Xx"])

    def test_lenient_returns_none_on_unknown_symbol(self):
        assert count_basis_functions(["O", "Xx"], strict=False) is None

    def test_lenient_still_counts_known_symbols(self):
        assert count_basis_functions(["O", "H", "H"], strict=False) == 58


class TestGroundTruthAgainstOrcaOutputs:
    """Pin the table against basis counts real ORCA runs actually reported.

    ``get_mem_estimate`` models memory as ``a * n_basis**1.5 + b``, so an error
    in the table propagates straight into %maxcore and worker sizing. These
    reference values are the "Number of basis functions" lines from the ORCA
    outputs committed under tests/files/, both using the default basis config
    (actinide ma-def-TZVP + def-ECP, everything else def2-TZVPD).
    """

    # tests/files/quacc_example/orca.out.gz -- NpF3, reports 223.
    def test_npf3_matches_real_orca_count(self):
        assert count_basis_functions(["Np", "F", "F", "F"]) == pytest.approx(
            223, rel=0.05
        )

    # tests/files/orca_direct_example/AmO_orca_atom95.out -- lone Am, reports 94.
    # The table's actinide entries are a uniform 105, which runs ~12% high on a
    # bare actinide (conservative: it over-allocates memory).
    def test_lone_actinide_matches_real_orca_count(self):
        assert count_basis_functions(["Am"]) == pytest.approx(94, rel=0.15)
