"""CPU tests for the batched-optimizer step math (no GPU / no fairchem).

``batched_optimizer`` has no fairchem import, so ``segment_normalized_step`` and
``per_structure_min_dist`` are pure-torch and testable on CPU.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from oact_utilities.scripts.entropy_downselect.batched_optimizer import (  # noqa: E402
    per_structure_min_dist,
    segment_normalized_step,
)


def test_segment_step_moves_max_atom_by_step_size():
    # structure 0: atoms 0,1 ; structure 1: atoms 2,3,4
    batch_idx = torch.tensor([0, 0, 1, 1, 1])
    R0 = torch.zeros(5, 3, dtype=torch.float64)
    pos = R0.clone()
    grad = torch.zeros(5, 3, dtype=torch.float64)
    grad[0, 0] = 2.0  # struct 0 max norm 2
    grad[1, 0] = 1.0
    grad[2, 0] = 4.0  # struct 1 max norm 4
    active = torch.tensor([True, True])

    new = segment_normalized_step(
        pos, grad, R0, batch_idx, 2, step_size=0.1, max_disp=1.0, active=active
    )
    disp = (new - R0).norm(dim=1)
    # most-responsive atom of each structure moves exactly step_size
    assert disp[0].item() == pytest.approx(0.1, abs=1e-9)
    assert disp[2].item() == pytest.approx(0.1, abs=1e-9)
    # the weaker atom in struct 0 moves proportionally (norm 1 vs 2 -> half)
    assert disp[1].item() == pytest.approx(0.05, abs=1e-9)


def test_segment_step_trust_region_clamps():
    batch_idx = torch.tensor([0, 0])
    R0 = torch.zeros(2, 3, dtype=torch.float64)
    grad = torch.zeros(2, 3, dtype=torch.float64)
    grad[0, 0] = 1.0
    grad[1, 0] = 1.0
    active = torch.tensor([True])
    # step_size 0.5 would move 0.5, but max_disp caps at 0.03
    new = segment_normalized_step(
        R0.clone(), grad, R0, batch_idx, 1, step_size=0.5, max_disp=0.03, active=active
    )
    disp = (new - R0).norm(dim=1)
    assert disp.max().item() == pytest.approx(0.03, abs=1e-9)


def test_segment_step_inactive_frozen():
    batch_idx = torch.tensor([0, 1])
    R0 = torch.zeros(2, 3, dtype=torch.float64)
    grad = torch.ones(2, 3, dtype=torch.float64)
    active = torch.tensor([True, False])
    new = segment_normalized_step(
        R0.clone(), grad, R0, batch_idx, 2, step_size=0.1, max_disp=1.0, active=active
    )
    assert (new[1] == R0[1]).all()  # inactive structure did not move
    assert (new[0] != R0[0]).any()  # active structure moved


def test_per_structure_min_dist():
    # struct 0: two atoms 1.5 apart; struct 1: three atoms, closest pair 0.5
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    offsets = torch.tensor([0, 2])
    natoms = torch.tensor([2, 3])
    md = per_structure_min_dist(pos, offsets, natoms)
    assert md[0].item() == pytest.approx(1.5, abs=1e-9)
    assert md[1].item() == pytest.approx(0.5, abs=1e-9)


def test_per_structure_min_dist_single_atom_is_inf():
    pos = torch.zeros(1, 3, dtype=torch.float64)
    md = per_structure_min_dist(pos, torch.tensor([0]), torch.tensor([1]))
    assert np.isinf(md[0].item())
