"""Batched structural optimization against a frozen covariance.

Optimizes many selected structures in parallel in a single forward+backward per step.
This is valid because the covariance barely changes within a batch (each rank-1 update
scales by ``1/(n+1)``), so the whole batch can ascend against the batch-start covariance
-- the same approximation that lets batch-greedy freeze the candidate pool.

The Jacobian from positions to features is block-diagonal across structures, so one
batched backward yields every structure's gradient. Per-structure normalized step,
trust region, clash guard, and best-tracking are done with segment ops keyed by the
collated batch index.

No fairchem import here: this operates on tensors produced by ``DifferentiableFeaturizer``
(``build_batch`` / ``featurize_batch``), so the step math is unit-testable on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class OptParams:
    max_steps: int
    max_disp: float
    step_size: float
    min_dist: float


def segment_normalized_step(
    pos: torch.Tensor,
    grad: torch.Tensor,
    R0: torch.Tensor,
    batch_idx: torch.Tensor,
    n_structs: int,
    step_size: float,
    max_disp: float,
    active: torch.Tensor,
) -> torch.Tensor:
    """One normalized ascent step for a batch of structures (pure torch).

    The most-responsive atom of each structure moves ``step_size`` Angstrom along the
    gradient; the whole structure is then clamped so no atom exceeds ``max_disp`` from
    its origin ``R0``. Inactive structures (``active[m]`` False) do not move.

    Args:
        pos, grad, R0: (A, 3) current positions, gradient, and origin.
        batch_idx: (A,) structure index of each atom.
        n_structs: M, number of structures.
        active: (M,) bool mask of structures still being optimized.

    Returns:
        (A, 3) new positions (detached).
    """
    gnorm = grad.norm(dim=1)  # (A,)
    max_gn = torch.zeros(n_structs, device=pos.device, dtype=pos.dtype)
    max_gn = max_gn.scatter_reduce(
        0, batch_idx, gnorm, reduce="amax", include_self=True
    )
    scale = step_size / max_gn.clamp(min=1e-12)  # (M,)
    scale = torch.where(active, scale, torch.zeros_like(scale))
    pos_new = pos + scale[batch_idx].unsqueeze(1) * grad
    disp = pos_new - R0
    norms = disp.norm(dim=1, keepdim=True).clamp(min=1e-12)
    factor = torch.clamp(max_disp / norms, max=1.0)
    return (R0 + disp * factor).detach()


def per_structure_min_dist(
    pos: torch.Tensor, offsets: torch.Tensor, natoms: torch.Tensor
) -> torch.Tensor:
    """Minimum interatomic distance within each structure -> (M,) tensor."""
    m = len(offsets)
    out = torch.full((m,), float("inf"), device=pos.device, dtype=pos.dtype)
    for i in range(m):
        s = int(offsets[i])
        k = int(natoms[i])
        if k < 2:
            continue
        block = pos[s : s + k]
        d = torch.cdist(block, block)
        d = d + torch.eye(k, device=pos.device, dtype=pos.dtype) * 1e9
        out[i] = d.min()
    return out


class BatchedStructureOptimizer:
    """Optimize a list of structures against a frozen covariance, in atom-budget batches.

    Each ``optimize`` call groups the structures into sub-batches whose total atom count
    is <= ``max_atoms``, runs the bounded ascent on each sub-batch (one forward+backward
    per step over all its structures), and returns per-structure results aligned with the
    input list.
    """

    def __init__(
        self,
        featurizer,
        mu_seed: np.ndarray,
        W: np.ndarray,
        opt: OptParams,
        max_atoms: int = 1024,
    ) -> None:
        self.f = featurizer
        dev = featurizer.device
        self.mu_seed_t = torch.as_tensor(mu_seed, dtype=torch.float64, device=dev)
        self.W_t = torch.as_tensor(W, dtype=torch.float64, device=dev)
        self.opt = opt
        self.max_atoms = max_atoms

    def _bin(self, atoms_list: list) -> list[list[int]]:
        bins: list[list[int]] = []
        cur: list[int] = []
        cur_atoms = 0
        for i, a in enumerate(atoms_list):
            na = len(a)
            if cur and cur_atoms + na > self.max_atoms:
                bins.append(cur)
                cur, cur_atoms = [], 0
            cur.append(i)
            cur_atoms += na
        if cur:
            bins.append(cur)
        return bins

    def optimize(
        self,
        atoms_list: list,
        C_inv: np.ndarray,
        mu: np.ndarray,
        n_current: int,
        x_orig: np.ndarray | None = None,
    ) -> list[tuple[np.ndarray, object, dict]]:
        """Optimize all structures; returns (feature, atoms_out, diag) per input index."""
        dev = self.f.device
        C_inv_t = torch.as_tensor(C_inv, dtype=torch.float64, device=dev)
        mu_t = torch.as_tensor(mu, dtype=torch.float64, device=dev)
        results: list = [None] * len(atoms_list)
        for bin_idx in self._bin(atoms_list):
            sub = [atoms_list[i] for i in bin_idx]
            xo = None if x_orig is None else x_orig[bin_idx]
            sub_res = self._optimize_subbatch(sub, C_inv_t, mu_t, n_current, xo)
            for j, i in enumerate(bin_idx):
                results[i] = sub_res[j]
        return results

    def _optimize_subbatch(
        self,
        sub_atoms: list,
        C_inv_t: torch.Tensor,
        mu_t: torch.Tensor,
        n_current: int,
        x_orig_sub: np.ndarray | None,
    ) -> list[tuple[np.ndarray, object, dict]]:
        f = self.f
        opt = self.opt
        denom = float(n_current + 1)
        d_const = float(len(mu_t)) * float(np.log(n_current / (n_current + 1)))

        data, offsets, natoms, batch_idx = f.build_batch(sub_atoms)
        m = len(sub_atoms)
        R0 = data.pos.detach().clone()

        pos = R0.clone().requires_grad_(True)
        neg_inf = torch.full((m,), -float("inf"), device=R0.device, dtype=torch.float64)
        best_score = neg_inf.clone()
        best_pos = R0.clone()
        best_step = torch.zeros(m, dtype=torch.long, device=R0.device)
        score0 = None
        x_step0 = None

        for step in range(opt.max_steps + 1):
            metal = f.featurize_batch(data, pos, offsets)  # (M, D)
            x = (metal.to(torch.float64) - self.mu_seed_t) @ self.W_t  # (M, D)
            d = x - mu_t
            q = (d @ C_inv_t * d).sum(1)  # (M,)
            score = torch.log1p(q / denom)  # (M,) f64
            sc = score.detach()
            if step == 0:
                score0 = sc.clone()
                x_step0 = x.detach().clone()

            md = per_structure_min_dist(pos.detach(), offsets, natoms)
            valid = (md >= opt.min_dist) & torch.isfinite(sc)
            improve = valid & (sc > best_score)
            best_score = torch.where(improve, sc, best_score)
            best_step = torch.where(
                improve, torch.full_like(best_step, step), best_step
            )
            if improve.any():
                amask = improve[batch_idx].unsqueeze(1)
                best_pos = torch.where(amask, pos.detach(), best_pos)

            if step == opt.max_steps:
                break

            active = valid
            if not active.any():
                break
            total = score[active].sum()
            (grad,) = torch.autograd.grad(total, pos)
            grad = torch.nan_to_num(grad)
            pos = segment_normalized_step(
                pos.detach(),
                grad,
                R0,
                batch_idx,
                m,
                opt.step_size,
                opt.max_disp,
                active,
            ).requires_grad_(True)

        with torch.no_grad():
            metal_best = f.featurize_batch(data, best_pos, offsets)
            x_best = (metal_best.to(torch.float64) - self.mu_seed_t) @ self.W_t

        improved = torch.isfinite(best_score)
        # structures that never found a valid step: keep original geometry + feature
        x_commit = torch.where(improved.unsqueeze(1), x_best, x_step0)
        delta_opt = torch.where(improved, best_score, score0)

        R0_c = R0.cpu().numpy()
        best_pos_c = best_pos.cpu().numpy()
        x_commit_c = x_commit.cpu().numpy()
        score0_c = score0.cpu().numpy()
        delta_opt_c = delta_opt.cpu().numpy()
        best_step_c = best_step.cpu().numpy()
        md_before = per_structure_min_dist(R0, offsets, natoms).cpu().numpy()
        md_after = per_structure_min_dist(best_pos, offsets, natoms).cpu().numpy()
        x_step0_c = x_step0.cpu().numpy()

        out = []
        for i in range(m):
            s = int(offsets[i])
            k = int(natoms[i])
            atoms_out = sub_atoms[i].copy()
            atoms_out.set_positions(best_pos_c[s : s + k].astype(np.float64))
            disp = np.linalg.norm(best_pos_c[s : s + k] - R0_c[s : s + k], axis=1).max()
            drift = float("nan")
            if x_orig_sub is not None:
                drift = float(np.linalg.norm(x_step0_c[i] - x_orig_sub[i]))
            diag = {
                "natoms": k,
                "optimized": True,
                "fallback": not bool(improved[i]),
                "delta_logdet_orig": d_const + float(score0_c[i]),
                "delta_logdet_opt": d_const + float(delta_opt_c[i]),
                "n_steps": int(best_step_c[i]),
                "max_disp": float(disp),
                "min_dist_before": float(md_before[i]),
                "min_dist_after": float(md_after[i]),
                "feature_drift": drift,
                "stop_reason": "batched" if bool(improved[i]) else "no_valid_step",
            }
            out.append((x_commit_c[i], atoms_out, diag))
        return out
