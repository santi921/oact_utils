"""MLIP pre-optimization: mock (ASE EMT) tests plus native tests that need real models."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from oact_utilities.core.preopt import mlip
from oact_utilities.core.preopt._common import parse_xyz
from oact_utilities.core.preopt.elements import UnsupportedElementsError

REPO = Path(__file__).resolve().parents[1]
OAC_CKPT = Path(
    os.environ.get(
        "OACT_TEST_CKPT", REPO / "examples" / "benchmarks" / "v4_main_inference.pt"
    )
)
REGISTRY_MODEL = os.environ.get(
    "OACT_TEST_REGISTRY_MODEL", "esen-sm-conserving-all-omol"
)
WATER = "3\n\nO 0 0 0.12\nH 0 0.76 -0.48\nH 0 -0.76 -0.48\n"
CU2 = "2\n\nCu 0 0 0\nCu 0 0 2.4\n"


@pytest.fixture()
def emt_mlip(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fairchem.core")
    import fairchem.core
    from ase.calculators.emt import EMT

    monkeypatch.setattr(
        fairchem.core, "FAIRChemCalculator", lambda unit, task_name: EMT()
    )
    monkeypatch.setattr(
        mlip, "load_mlip", lambda ckpt, device: (None, frozenset({"Cu", "H"}))
    )


def test_mock_relaxes_cu_dimer(emt_mlip: None) -> None:
    xyz, res = mlip.run_mlip_preopt(CU2, 0, 1, "fake.pt", fmax=0.02)
    assert res.ok and res.converged
    assert res.method == "mlip" and "fake.pt" in res.detail
    assert res.fmax_final <= 0.02
    _, p = parse_xyz(xyz)
    assert 2.0 < abs(p[1, 2] - p[0, 2]) < 2.8


def test_mock_max_steps_not_converged(emt_mlip: None) -> None:
    _, res = mlip.run_mlip_preopt(CU2, 0, 1, "fake.pt", fmax=1e-9, max_steps=2)
    assert res.ok and not res.converged and res.n_steps <= 3


def test_mock_unsupported_element_raises_before_calc(emt_mlip: None) -> None:
    pu = "2\n\nPu 0 0 0\nCu 0 0 2.5\n"
    with pytest.raises(UnsupportedElementsError) as exc:
        mlip.run_mlip_preopt(pu, 0, 1, "fake.pt", job_id=12)
    assert exc.value.elements == ["Pu"] and "job 12" in str(exc.value)


@pytest.mark.mlip
def test_native_omol_only_esen_rejects_actinide_runs_water() -> None:
    pytest.importorskip("fairchem.core")
    try:
        mlip.load_mlip(REGISTRY_MODEL, "cpu")
    except Exception as exc:  # noqa: BLE001 -- offline / gated / not cached
        pytest.skip(f"cannot load {REGISTRY_MODEL}: {type(exc).__name__}")
    with pytest.raises(UnsupportedElementsError):
        mlip.run_mlip_preopt(
            "3\n\nU 0 0 0\nO 0 0 1.9\nO 0 0 -1.9\n", 2, 1, REGISTRY_MODEL
        )
    xyz, res = mlip.run_mlip_preopt(WATER, 0, 1, REGISTRY_MODEL, fmax=0.05)
    assert res.ok and res.converged and res.fmax_final <= 0.05
    assert (
        res.energy_ev < -2000
    )  # eV, water total energy in an omol-referenced model scale


@pytest.mark.mlip
@pytest.mark.skipif(not OAC_CKPT.exists(), reason=f"no OAct checkpoint at {OAC_CKPT}")
def test_native_oact_checkpoint_relaxes_uo2() -> None:
    pytest.importorskip("fairchem.core")
    uo2 = "3\n\nU 0 0 0\nO 0 0 1.9\nO 0 0 -1.9\n"
    xyz, res = mlip.run_mlip_preopt(uo2, 2, 1, str(OAC_CKPT), fmax=0.05)
    assert res.ok, res.error
    assert res.converged
    _, p = parse_xyz(xyz)
    assert 1.5 < abs(p[1, 2] - p[0, 2]) < 2.2
