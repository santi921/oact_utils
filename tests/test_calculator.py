"""Tests for ORCA calculator setup and UKS symmetry breaking."""

from ase import Atoms

from oact_utilities.core.orca.calc import (
    ACTINIDE_LIST,
    ECP_SIZE,
    get_orca_blocks,
    get_symm_break_block,
)

# ---------------------------------------------------------------------------
# ECP_SIZE dict integrity
# ---------------------------------------------------------------------------


def test_ecp_size_covers_all_ranges():
    """ECP_SIZE must include entries for Z=37-86 (def2) and Z=87-102 (heavy)."""
    # def2 range
    for z in range(37, 87):
        assert z in ECP_SIZE, f"Missing ECP_SIZE entry for Z={z}"
    # heavy elements
    for z in range(87, 103):
        assert z in ECP_SIZE, f"Missing ECP_SIZE entry for Z={z}"


# ---------------------------------------------------------------------------
# get_symm_break_block
# ---------------------------------------------------------------------------


def test_symm_break_block_simple_molecule():
    """Verify LUMO index for a simple molecule (H2, charge=0)."""
    h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    block = get_symm_break_block(h2, charge=0)
    # H2: 2 electrons, no ECP → LUMO index = 1
    assert "rotate {0, 1, 20, 1, 1}" in block


def test_symm_break_block_respects_charge():
    """Charge should reduce electron count and shift LUMO index."""
    # Li2 neutral: 6 electrons → LUMO = 3
    li2 = Atoms("Li2", positions=[[0, 0, 0], [0, 0, 2.67]])
    block_neutral = get_symm_break_block(li2, charge=0)
    assert "rotate {2, 3, 20, 1, 1}" in block_neutral

    # Li2 with charge +2: 4 electrons → LUMO = 2
    block_charged = get_symm_break_block(li2, charge=2)
    assert "rotate {1, 2, 20, 1, 1}" in block_charged


def test_symm_break_block_all_electron_elements():
    """All-electron elements should NOT have ECP electrons subtracted."""
    # UO2: U (Z=92) + 2×O (Z=8) = 108 electrons
    # Without all-electron flag: U has ECP_SIZE=78 → 108-78=30 → LUMO=15
    # With all-electron flag for U: 108-0=108 → LUMO=54
    uo2 = Atoms("UO2", positions=[[0, 0, 0], [0, 0, 1.8], [0, 0, -1.8]])

    block_with_ecp = get_symm_break_block(uo2, charge=0)
    block_all_electron = get_symm_break_block(
        uo2, charge=0, all_electron_elements={"U"}
    )

    # These should differ because U's 78 ECP electrons change the LUMO index
    assert block_with_ecp != block_all_electron


# ---------------------------------------------------------------------------
# get_orca_blocks — actinide singlet detection
# ---------------------------------------------------------------------------


def _make_uo2(charge: int = 0) -> Atoms:
    """Helper: create a UO2 molecule."""
    return Atoms("UO2", positions=[[0, 0, 0], [0, 0, 1.8], [0, 0, -1.8]])


def test_actinide_singlet_gets_uks():
    """Actinide singlet (mult=1) should automatically get UKS + rotate block."""
    atoms = _make_uo2()
    simple, blocks = get_orca_blocks(atoms, mult=1, charge=0)
    assert "UKS" in simple
    assert any("rotate" in b for b in blocks)


def test_actinide_non_singlet_no_uks():
    """Actinide with mult > 1 should NOT get automatic UKS symmetry breaking."""
    atoms = _make_uo2()
    simple, blocks = get_orca_blocks(atoms, mult=3, charge=0)
    assert "UKS" not in simple
    assert not any("rotate" in b for b in blocks)


def test_non_actinide_singlet_no_uks():
    """Non-actinide singlet should NOT get UKS."""
    h2o = Atoms("OH2", positions=[[0, 0, 0], [0, 0, 0.96], [0.93, 0, -0.24]])
    simple, blocks = get_orca_blocks(h2o, mult=1, charge=0)
    assert "UKS" not in simple
    assert not any("rotate" in b for b in blocks)


def test_actinide_singlet_charge_passed_to_rotate():
    """Charge should propagate into the rotate block LUMO calculation."""
    atoms = _make_uo2()
    _, blocks_neutral = get_orca_blocks(atoms, mult=1, charge=0)
    _, blocks_charged = get_orca_blocks(atoms, mult=1, charge=2)

    # Extract rotate blocks
    rotate_neutral = [b for b in blocks_neutral if "rotate" in b]
    rotate_charged = [b for b in blocks_charged if "rotate" in b]

    assert len(rotate_neutral) == 1
    assert len(rotate_charged) == 1
    # Different charge → different LUMO index → different rotate block
    assert rotate_neutral[0] != rotate_charged[0]


def test_all_actinides_detected():
    """Every element in ACTINIDE_LIST should trigger UKS for singlets."""
    for elem in ACTINIDE_LIST:
        atoms = Atoms(elem, positions=[[0, 0, 0]])
        simple, blocks = get_orca_blocks(atoms, mult=1, charge=0)
        assert "UKS" in simple, f"UKS not added for singlet {elem}"
        assert any(
            "rotate" in b for b in blocks
        ), f"Rotate block missing for singlet {elem}"


# ---------------------------------------------------------------------------
# opt_level tests
# ---------------------------------------------------------------------------


def test_opt_level_normal():
    """opt_level='normal' produces 'Opt' keyword."""
    h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    simple, _ = get_orca_blocks(h2, opt=True, opt_level="normal")
    assert " Opt " in simple or simple.endswith(" Opt") or "Opt " in simple


def test_opt_level_tight():
    """opt_level='tight' produces 'TightOpt' keyword."""
    h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    simple, _ = get_orca_blocks(h2, opt=True, opt_level="tight")
    assert "TightOpt" in simple


def test_opt_level_verytight():
    """opt_level='verytight' produces 'VeryTightOpt' keyword."""
    h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    simple, _ = get_orca_blocks(h2, opt=True, opt_level="verytight")
    assert "VeryTightOpt" in simple


def test_opt_level_loose():
    """opt_level='loose' produces 'LooseOpt' keyword."""
    h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    simple, _ = get_orca_blocks(h2, opt=True, opt_level="loose")
    assert "LooseOpt" in simple


def test_opt_false_ignores_level():
    """opt=False should not include any Opt keyword regardless of opt_level."""
    h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    simple, _ = get_orca_blocks(h2, opt=False, opt_level="tight")
    assert "Opt" not in simple
