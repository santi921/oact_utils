"""Element-coverage checks for pre-optimizers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from oact_utilities.core.preopt.elements import (
    UnsupportedElementsError,
    check_elements,
    get_supported_elements,
    gxtb_supported_elements,
    parse_elements,
    supported_from_reference_keys,
    supported_from_reference_list,
    unsupported_elements,
)

REPO = Path(__file__).resolve().parents[1]
OAC_CKPT = Path(
    os.environ.get(
        "OACT_TEST_CKPT", REPO / "examples" / "benchmarks" / "v4_main_inference.pt"
    )
)
REGISTRY_MODEL = os.environ.get(
    "OACT_TEST_REGISTRY_MODEL", "esen-sm-conserving-all-omol"
)


def test_parse_elements_semicolon_and_iterable() -> None:
    assert parse_elements("U;O;O; H") == ["U", "O", "O", "H"]
    assert parse_elements(["U", " O "]) == ["U", "O"]


def test_gxtb_range() -> None:
    sup = gxtb_supported_elements()
    assert {"H", "U", "Lr"} <= sup
    assert "Rf" not in sup
    assert len(sup) == 103


def test_reference_list_zero_means_absent() -> None:
    refs = [0.0] * 120
    refs[1] = -0.5
    refs[8] = -75.0
    refs[92] = -1.0
    refs[86] = 1e-11  # fit noise, as Rn in the v3/v4 OAct checkpoints
    assert supported_from_reference_list(refs) == {"H", "O", "U"}


def test_reference_keys() -> None:
    assert supported_from_reference_keys([1, 6, "8"]) == {"H", "C", "O"}


def test_unsupported_sorted_by_z_and_error_message() -> None:
    sup = frozenset({"H", "O"})
    assert unsupported_elements("Pu;U;O", sup) == ["U", "Pu"]
    with pytest.raises(UnsupportedElementsError) as exc:
        check_elements("Pu;U;O", sup, "mlip", job_id=7)
    assert exc.value.elements == ["U", "Pu"]
    assert "job 7" in str(exc.value)
    assert isinstance(exc.value, ValueError)


def test_check_elements_passes() -> None:
    check_elements(["H", "O"], frozenset({"H", "O", "C"}), "mlip")


@pytest.mark.mlip
@pytest.mark.skipif(not OAC_CKPT.exists(), reason=f"no OAct checkpoint at {OAC_CKPT}")
def test_oact_checkpoint_covers_actinides_not_rn() -> None:
    sup = get_supported_elements(str(OAC_CKPT))
    assert {"U", "Pu", "O", "H"} <= sup
    assert "Rn" not in sup


@pytest.mark.mlip
def test_registry_omol_model_has_no_actinides() -> None:
    pytest.importorskip("fairchem.core")
    try:
        sup = get_supported_elements(REGISTRY_MODEL)
    except Exception as exc:  # noqa: BLE001 -- offline / gated / not cached
        pytest.skip(f"cannot load {REGISTRY_MODEL}: {type(exc).__name__}")
    assert {"H", "C", "O", "Bi"} <= sup
    assert not ({"U", "Pu", "Th"} & sup)
