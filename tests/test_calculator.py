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
# Double-UKS prevention (ks_method + actinide singlet auto-detection)
# ---------------------------------------------------------------------------


def test_explicit_uks_no_duplicate():
    """Explicit ks_method='uks' on actinide singlet should NOT produce 'UKS UKS'."""
    atoms = _make_uo2()
    simple, blocks = get_orca_blocks(atoms, mult=1, charge=0, ks_method="uks")
    # Count occurrences of UKS in the simple input
    uks_count = simple.upper().split().count("UKS")
    assert uks_count == 1, f"Expected 1 UKS, got {uks_count}: {simple}"
    # Rotate block should still be present (symmetry breaking still needed)
    assert any("rotate" in b for b in blocks)


def test_explicit_rks_still_gets_uks_for_actinide_singlet():
    """Explicit ks_method='rks' on actinide singlet should still add UKS (auto-detection)."""
    atoms = _make_uo2()
    simple, blocks = get_orca_blocks(atoms, mult=1, charge=0, ks_method="rks")
    # Auto-detection should add UKS even though RKS was requested
    assert "UKS" in simple
    assert "RKS" in simple
    # Rotate block should be present
    assert any("rotate" in b for b in blocks)


def test_ks_method_appended_to_simple_input():
    """Explicit ks_method should appear in the simple input line."""
    atoms = _make_uo2()
    # Non-singlet with explicit UKS (no auto-detection)
    simple, _ = get_orca_blocks(atoms, mult=3, charge=0, ks_method="uks")
    assert "UKS" in simple

    # Non-actinide with explicit ROKS
    h2o = Atoms("OH2", positions=[[0, 0, 0], [0, 0, 0.96], [0.93, 0, -0.24]])
    simple, _ = get_orca_blocks(h2o, mult=3, charge=0, ks_method="roks")
    assert "ROKS" in simple
