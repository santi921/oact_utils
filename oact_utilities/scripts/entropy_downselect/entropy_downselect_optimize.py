"""Entropy downselect with in-loop structure optimization.

Runs the exact greedy entropy downselect of ``entropy_downselect.py``, but the moment
a structure is selected it is handed to a short, bounded gradient ascent that perturbs
its atomic positions (through the frozen fairchem GNN) to increase its marginal
``delta log det`` against the live covariance. The optimized feature is committed to the
covariance in place of the original, and the optimized geometry is written to the output
LMDB.

The selection mechanics are unchanged -- the only difference is the ``commit_hook``
passed to ``run_selection`` (see ``oact_utilities.utils.entropy_selection``).

Optimization is bounded by a trust region (max per-atom displacement), a clash guard,
a max step count, a monotonic-improvement stop, and a graceful fallback that commits the
original feature/geometry on any failure.

Usage:
    python -m oact_utilities.scripts.entropy_downselect.entropy_downselect_optimize \
        --features-dir /path/to/candidate_features/ \
        --seed-features /path/to/seed_features.npy \
        --lmdb-dir /path/to/source_lmdbs/ \
        --model-path /path/to/inference_ckpt.pt \
        --output-dir /path/to/output/ \
        --n-select 2000 --limit 200000 \
        --opt-max-steps 5 --opt-max-disp 0.3 --opt-step-size 0.05 --opt-min-dist 0.7
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from oact_utilities.scripts.entropy_downselect.batched_optimizer import (
    BatchedStructureOptimizer,
    OptParams,
)
from oact_utilities.scripts.entropy_downselect.featurizer import (
    DifferentiableFeaturizer,
)
from oact_utilities.scripts.entropy_downselect.run_lmdb_inference import MODEL_PATH
from oact_utilities.utils.entropy_selection import (
    compute_single_score,
    debug_log,
    load_features,
    make_index_resolver,
    run_selection,
    whiten,
)


def _min_pair_dist(pos: torch.Tensor) -> float:
    """Minimum interatomic distance (clash metric) for an (N,3) position tensor."""
    P = pos.detach()
    if P.shape[0] < 2:
        return float("inf")
    d = torch.cdist(P, P)
    d = d + torch.eye(P.shape[0], device=P.device, dtype=d.dtype) * 1e9
    return float(d.min())


def optimize_structure(
    atoms,
    C_inv: np.ndarray,
    mu: np.ndarray,
    n_current: int,
    featurizer: DifferentiableFeaturizer,
    mu_seed_t: torch.Tensor,
    W_t: torch.Tensor,
    x_orig_whitened: np.ndarray,
    opt: OptParams,
) -> tuple[np.ndarray, object, dict]:
    """Bounded gradient ascent on a single structure's positions.

    Maximizes ``log1p(q/(n+1))`` (monotone in delta_log_det) where
    ``q = (x-mu) C_inv (x-mu)`` and ``x`` is the whitened metal-atom feature, w.r.t.
    atomic positions, within a per-atom displacement trust region.

    Steps follow the *normalized* gradient: the raw marginal-gain gradient is scaled
    by ``1/(n+1)`` (tiny when the background set is large, and shrinking as ``n``
    grows), so each step instead moves the most-responsive atom a fixed
    ``step_size`` Angstrom along the gradient direction, bounded by the trust region.

    Returns:
        x_committed: (D,) whitened feature to commit (best valid geometry found).
        atoms_out: ASE Atoms with optimized positions.
        diag: per-structure diagnostics.
    """
    dev = featurizer.device
    C_inv_t = torch.as_tensor(C_inv, dtype=torch.float64, device=dev)
    mu_t = torch.as_tensor(mu, dtype=torch.float64, device=dev)
    denom = float(n_current + 1)

    data = featurizer.build_data(atoms)
    R0 = featurizer.initial_pos(data)

    def whitened_feature(pos: torch.Tensor) -> torch.Tensor:
        x_raw = featurizer.featurize(data, pos)
        return (x_raw.to(torch.float64) - mu_seed_t) @ W_t

    def log1p_score(x_w: torch.Tensor) -> torch.Tensor:
        d = x_w - mu_t
        q = d @ C_inv_t @ d
        return torch.log1p(q / denom)

    min_dist_before = _min_pair_dist(R0)

    pos = R0.clone().requires_grad_(True)
    best_score = -float("inf")
    best_pos = R0.clone()
    best_x = None
    drift = float("nan")
    n_steps = 0
    stop_reason = "max_steps"

    for step in range(opt.max_steps + 1):
        x_w = whitened_feature(pos)
        score_t = log1p_score(x_w)
        score = float(score_t.detach())

        if step == 0:
            drift = float(np.linalg.norm(x_w.detach().cpu().numpy() - x_orig_whitened))

        if not np.isfinite(score):
            stop_reason = "nonfinite"
            break
        if _min_pair_dist(pos) < opt.min_dist:
            stop_reason = "clash"
            break

        if score > best_score:
            best_score = score
            best_pos = pos.detach().clone()
            best_x = x_w.detach()
        else:
            stop_reason = "no_improve"
            break

        if step == opt.max_steps:
            break

        (grad,) = torch.autograd.grad(score_t, pos)
        if not torch.isfinite(grad).all():
            stop_reason = "nonfinite_grad"
            break

        with torch.no_grad():
            max_atom_gnorm = float(grad.norm(dim=1).max())
            if max_atom_gnorm < 1e-12:
                stop_reason = "zero_grad"
                break
            # normalized step: most-responsive atom moves step_size A along the gradient
            pos_new = pos + (opt.step_size / max_atom_gnorm) * grad
            disp = pos_new - R0
            norms = disp.norm(dim=1, keepdim=True).clamp(min=1e-12)
            factor = torch.clamp(opt.max_disp / norms, max=1.0)
            pos_new = R0 + disp * factor
        pos = pos_new.detach().requires_grad_(True)
        n_steps = step + 1

    best_x_np = best_x.cpu().numpy() if best_x is not None else x_orig_whitened.copy()
    best_pos_np = best_pos.detach().cpu().numpy().astype(np.float64)

    atoms_out = atoms.copy()
    atoms_out.set_positions(best_pos_np)

    max_disp = float(np.linalg.norm(best_pos_np - R0.cpu().numpy(), axis=1).max())
    diag = {
        "natoms": len(atoms_out),
        "optimized": True,
        "fallback": False,
        "delta_logdet_orig": compute_single_score(
            x_orig_whitened, C_inv, mu, n_current
        ),
        "delta_logdet_opt": compute_single_score(best_x_np, C_inv, mu, n_current),
        "n_steps": n_steps,
        "max_disp": max_disp,
        "min_dist_before": min_dist_before,
        "min_dist_after": _min_pair_dist(best_pos),
        "feature_drift": drift,
        "stop_reason": stop_reason,
    }
    return best_x_np, atoms_out, diag


def _jsonable(d: dict) -> dict:
    """Coerce numpy scalars in a diagnostics dict to JSON-serializable Python types."""
    out: dict = {}
    for k, v in d.items():
        if isinstance(v, np.floating):
            out[k] = float(v)
        elif isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, np.bool_):
            out[k] = bool(v)
        else:
            out[k] = v
    return out


class _StreamingLmdbWriter:
    """Incrementally write rank-keyed Atoms to an LMDB, flushing in batches.

    Keeps memory bounded (only ``flush_every`` pickled blobs buffered). Supports resume:
    with ``resume=True`` the existing LMDB is opened and appended to; re-written ranks
    overwrite, so re-emitting ranks after a restart is idempotent. ``finalize`` writes
    the ``length`` key.
    """

    def __init__(
        self,
        path: Path,
        resume: bool,
        flush_every: int = 2000,
        map_size: int = 50 * (1 << 30),
    ) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not resume and path.exists():
            path.unlink()
            Path(str(path) + "-lock").unlink(missing_ok=True)
        self.env = lmdb.open(
            str(path), map_size=map_size, subdir=False, meminit=False, map_async=True
        )
        self.flush_every = flush_every
        self._buf: list[tuple[int, bytes]] = []

    def put(self, rank: int, atoms) -> None:
        self._buf.append((rank, pickle.dumps(atoms, protocol=-1)))
        if len(self._buf) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        txn = self.env.begin(write=True)
        for rank, blob in self._buf:
            txn.put(f"{rank}".encode("ascii"), blob)
        txn.commit()
        self._buf.clear()

    def finalize(self, n_total: int) -> None:
        self.flush()
        txn = self.env.begin(write=True)
        txn.put(b"length", pickle.dumps(n_total, protocol=-1))
        txn.commit()
        self.env.sync()

    def close(self) -> None:
        self.env.close()


class _DiagnosticsLog:
    """Append-only JSONL of per-rank diagnostics; the source of truth for the report.

    Persists across restarts (``resume=True`` keeps existing lines). ``load_deduped``
    reads all records and keeps the last occurrence per rank, so ranks re-emitted after
    a restart collapse to their final values.
    """

    def __init__(self, path: Path, resume: bool, flush_every: int = 2000) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not resume and path.exists():
            path.unlink()
        self._fh = open(path, "a")
        self.flush_every = flush_every
        self._since_flush = 0

    def append(self, diag: dict) -> None:
        self._fh.write(json.dumps(_jsonable(diag)) + "\n")
        self._since_flush += 1
        if self._since_flush >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._since_flush = 0

    def close(self) -> None:
        self.flush()
        self._fh.close()

    def load_deduped(self) -> list[dict]:
        by_rank: dict[int, dict] = {}
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    by_rank[int(d["rank"])] = d
        return [by_rank[r] for r in sorted(by_rank)]


class _StreamingHookBase:
    """Shared streaming/output machinery for the optimizing hooks.

    Owns the source-LMDB readers, the streaming optimized-structure LMDB writer, and the
    per-rank diagnostics JSONL, plus checkpoint flushing and finalization (write the LMDB
    ``length``, ``metadata.npz``, and the report). Subclasses implement the actual
    optimization in ``__call__``.
    """

    def __init__(
        self,
        X: np.ndarray,
        file_boundaries: list[tuple[str, int, int]],
        lmdb_dir: str,
        output_dir: Path,
        resume: bool,
    ) -> None:
        self.X = X
        self.resolve = make_index_resolver(file_boundaries)
        self.lmdb_dir = Path(lmdb_dir)
        self.writer = _StreamingLmdbWriter(
            output_dir / "optimized_structures.lmdb", resume=resume
        )
        self.diaglog = _DiagnosticsLog(
            output_dir / "optimization_diag.jsonl", resume=resume
        )
        self._env_cache: dict[str, lmdb.Environment] = {}

    def _load_atoms(self, global_idx: int):
        stem, local = self.resolve(global_idx)
        env = self._env_cache.get(stem)
        if env is None:
            env = lmdb.open(
                str(self.lmdb_dir / stem / "data.lmdb"),
                readonly=True,
                lock=False,
                subdir=False,
            )
            self._env_cache[stem] = env
        with env.begin() as txn:
            atoms = pickle.loads(txn.get(f"{local}".encode("ascii")))
        return atoms

    def _emit(self, rank: int, atoms, diag: dict) -> None:
        atoms.info["selection_rank"] = rank
        self.writer.put(rank, atoms)
        self.diaglog.append(diag)

    def on_checkpoint(self, step: int) -> None:
        """Flush streamed outputs in lockstep with the covariance checkpoint."""
        self.writer.flush()
        self.diaglog.flush()

    def finalize(self, output_dir: Path, n_total: int) -> None:
        """Finalize the optimized LMDB, write metadata.npz and the report."""
        self.writer.finalize(n_total)
        self.diaglog.close()
        records = self.diaglog.load_deduped()
        if len(records) != n_total:
            debug_log(
                f"WARNING: {len(records)} diagnostics records but {n_total} selected"
            )
        natoms = np.array([int(r["natoms"]) for r in records], dtype=np.int32)
        np.savez(str(output_dir / "metadata.npz"), natoms=natoms)
        _save_report(output_dir, records)

    def close(self) -> None:
        for env in self._env_cache.values():
            env.close()
        self._env_cache.clear()
        self.writer.close()


class OptimizingCommitHook(_StreamingHookBase):
    """Per-pick commit hook: optimize each selected structure one at a time.

    Returns the optimized whitened feature for the rank-1 covariance update and streams
    the optimized Atoms/diagnostics. Falls back to the original geometry/feature on
    optimization failure.
    """

    def __init__(
        self,
        X: np.ndarray,
        featurizer: DifferentiableFeaturizer,
        file_boundaries: list[tuple[str, int, int]],
        lmdb_dir: str,
        mu_seed: np.ndarray,
        W: np.ndarray,
        opt: OptParams,
        output_dir: Path,
        resume: bool,
        opt_top_n: int | None = None,
    ) -> None:
        super().__init__(X, file_boundaries, lmdb_dir, output_dir, resume)
        self.featurizer = featurizer
        dev = featurizer.device
        self.mu_seed_t = torch.as_tensor(mu_seed, dtype=torch.float64, device=dev)
        self.W_t = torch.as_tensor(W, dtype=torch.float64, device=dev)
        self.opt = opt
        self.opt_top_n = opt_top_n

    def _commit_original(self, atoms, rank, gidx, reason, **extra) -> dict:
        atoms = atoms.copy()
        diag = {
            "rank": rank,
            "global_index": gidx,
            "natoms": len(atoms),
            "optimized": False,
            "fallback": reason == "fallback",
            "delta_logdet_orig": np.nan,
            "delta_logdet_opt": np.nan,
            "n_steps": 0,
            "max_disp": 0.0,
            "min_dist_before": np.nan,
            "min_dist_after": np.nan,
            "feature_drift": np.nan,
            "stop_reason": reason,
            **extra,
        }
        self._emit(rank, atoms, diag)
        return diag

    def __call__(self, best_global, rank, C, C_inv, mu, n_current):
        x_orig = self.X[best_global]
        try:
            atoms = self._load_atoms(int(best_global))
        except Exception as e:
            raise RuntimeError(
                f"Failed to load source structure rank={rank} gidx={best_global}: {e}"
            ) from e

        if self.opt_top_n is not None and rank >= self.opt_top_n:
            return x_orig, self._commit_original(
                atoms, rank, int(best_global), "skipped"
            )

        try:
            x_opt, atoms_opt, diag = optimize_structure(
                atoms,
                C_inv,
                mu,
                n_current,
                self.featurizer,
                self.mu_seed_t,
                self.W_t,
                x_orig,
                self.opt,
            )
        except Exception as e:  # noqa: BLE001 - any opt error -> graceful fallback
            debug_log(f"opt failed rank={rank} gidx={best_global}: {e}")
            return x_orig, self._commit_original(
                atoms, rank, int(best_global), "fallback", error=str(e)
            )

        diag.update({"rank": rank, "global_index": int(best_global)})
        self._emit(rank, atoms_opt, diag)
        return x_opt, diag


class BatchedOptimizingHook(_StreamingHookBase):
    """Per-batch optimizer: optimize a whole batch of selected structures in parallel.

    ``run_selection`` calls this once per batch with the batch's ``(rank, global_index)``
    items and the frozen batch-start covariance; it loads the structures, runs the
    ``BatchedStructureOptimizer`` (one forward+backward per step over the batch), streams
    the optimized Atoms/diagnostics, and returns ``{rank: optimized_feature}``.
    """

    def __init__(
        self,
        X: np.ndarray,
        optimizer: BatchedStructureOptimizer,
        file_boundaries: list[tuple[str, int, int]],
        lmdb_dir: str,
        output_dir: Path,
        resume: bool,
    ) -> None:
        super().__init__(X, file_boundaries, lmdb_dir, output_dir, resume)
        self.optimizer = optimizer

    def __call__(self, items, C_inv, mu, n_current):
        gidxs = [int(g) for _r, g in items]
        atoms_list = [self._load_atoms(g) for g in gidxs]
        x_orig = self.X[gidxs]
        results = self.optimizer.optimize(
            atoms_list, C_inv, mu, n_current, x_orig=x_orig
        )
        out: dict[int, np.ndarray] = {}
        for (rank, gidx), (feat, atoms_opt, diag) in zip(items, results):
            diag = {**diag, "rank": rank, "global_index": gidx}
            self._emit(rank, atoms_opt, diag)
            out[rank] = feat
        return out


class MultiGPUBatchOptimizingHook(_StreamingHookBase):
    """Per-batch optimizer that shards each batch across persistent per-GPU workers.

    Spawns one worker per GPU (each loads the model on its device). Per batch, the items
    are round-robin sharded across workers; each worker loads its structures and runs the
    batched optimization on its GPU, and the coordinator gathers and streams the results.
    Optimizations within a batch are independent given the frozen covariance, so this is
    embarrassingly parallel across GPUs.
    """

    def __init__(
        self,
        X: np.ndarray,
        file_boundaries: list[tuple[str, int, int]],
        lmdb_dir: str,
        output_dir: Path,
        resume: bool,
        model_path: str,
        mu_seed: np.ndarray,
        W: np.ndarray,
        opt: OptParams,
        max_atoms: int,
        n_gpus: int,
    ) -> None:
        super().__init__(X, file_boundaries, lmdb_dir, output_dir, resume)
        import torch.multiprocessing as mp

        from oact_utilities.scripts.entropy_downselect.multi_gpu_optimizer import (
            worker_main,
        )

        self.n_gpus = n_gpus
        ctx = mp.get_context("spawn")
        self.out_q = ctx.Queue()
        self.in_qs = [ctx.Queue() for _ in range(n_gpus)]
        self.procs = []
        for g in range(n_gpus):
            p = ctx.Process(
                target=worker_main,
                args=(
                    g,
                    model_path,
                    mu_seed,
                    W,
                    opt,
                    max_atoms,
                    file_boundaries,
                    lmdb_dir,
                    self.in_qs[g],
                    self.out_q,
                ),
                daemon=True,
            )
            p.start()
            self.procs.append(p)

        ready = 0
        while ready < n_gpus:
            gid, status, payload = self.out_q.get()
            if status == "error":
                raise RuntimeError(f"GPU worker {gid} failed at init:\n{payload}")
            ready += 1
        debug_log(f"Multi-GPU: {n_gpus} workers ready")
        self._batch_id = 0

    def __call__(self, items, C_inv, mu, n_current):
        if not items:
            return {}
        self._batch_id += 1
        bid = self._batch_id
        chunks: list[list] = [[] for _ in range(self.n_gpus)]
        for k, it in enumerate(items):
            chunks[k % self.n_gpus].append(it)
        for g in range(self.n_gpus):
            ch = chunks[g]
            if ch:
                xo = self.X[[int(gi) for _r, gi in ch]]
            else:
                xo = np.empty((0, self.X.shape[1]), dtype=self.X.dtype)
            self.in_qs[g].put((bid, ch, xo, C_inv, mu, n_current))

        out: dict[int, np.ndarray] = {}
        received = 0
        while received < self.n_gpus:
            gid, rbid, payload = self.out_q.get()
            if rbid == "error":
                raise RuntimeError(f"GPU worker {gid} failed:\n{payload}")
            for rank, gidx, feat, atoms_opt, diag in payload:
                diag = {**diag, "rank": rank, "global_index": gidx}
                self._emit(rank, atoms_opt, diag)
                out[rank] = feat
            received += 1
        return out

    def close(self) -> None:
        for q in self.in_qs:
            q.put(None)
        for p in self.procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
        super().close()


def save_outputs(
    output_dir: Path,
    results: dict,
    features_dir: str,
    file_boundaries: list[tuple[str, int, int]],
    hook: _StreamingHookBase,
) -> None:
    """Write indices, history, metadata, optimized LMDB, and optimization report."""
    selected = np.array(results["selected_indices"], dtype=np.int64)

    np.save(str(output_dir / "selected_indices.npy"), selected)
    debug_log(f"Saved selected_indices.npy: {selected.shape}")

    np.savez(
        str(output_dir / "selection_history.npz"),
        delta_log_dets=np.array(results["delta_log_dets"], dtype=np.float64),
        log_dets=np.array(results["log_dets"], dtype=np.float64),
        seed_indices=results["seed_indices"],
    )
    debug_log("Saved selection_history.npz")

    feat_dir = Path(features_dir)
    metadata_frames = []
    for stem, start, count in tqdm(file_boundaries, desc="Gathering metadata"):
        meta_path = feat_dir / f"{stem}_metadata.parquet"
        if not meta_path.exists():
            debug_log(f"WARNING: {meta_path} not found, skipping")
            continue
        mask = (selected >= start) & (selected < start + count)
        if not mask.any():
            continue
        local_idx = selected[mask] - start
        meta = pd.read_parquet(str(meta_path)).iloc[local_idx].copy()
        meta["global_index"] = selected[mask]
        meta["source_file"] = stem
        meta["local_index"] = local_idx
        metadata_frames.append(meta)
    if metadata_frames:
        selected_meta = pd.concat(metadata_frames, ignore_index=True)
        selected_meta.sort_values("global_index", inplace=True)
        selected_meta.to_parquet(
            str(output_dir / "selected_metadata.parquet"),
            index=False,
        )
        debug_log(f"Saved selected_metadata.parquet: {len(selected_meta):,} rows")

    # Finalize the streamed optimized LMDB + metadata.npz + optimization report.
    hook.finalize(output_dir, len(selected))

    checkpoint_path = output_dir / "checkpoint.npz"
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        debug_log("Removed checkpoint (selection complete)")


def _save_report(output_dir: Path, diagnostics: list[dict]) -> None:
    if not diagnostics:
        return
    keys_f = [
        "rank",
        "global_index",
        "delta_logdet_orig",
        "delta_logdet_opt",
        "n_steps",
        "max_disp",
        "min_dist_before",
        "min_dist_after",
        "feature_drift",
    ]
    cols = {
        k: np.array([d.get(k, np.nan) for d in diagnostics], dtype=np.float64)
        for k in keys_f
    }
    cols["optimized"] = np.array(
        [bool(d.get("optimized", False)) for d in diagnostics], dtype=bool
    )
    cols["fallback"] = np.array(
        [bool(d.get("fallback", False)) for d in diagnostics], dtype=bool
    )
    np.savez(str(output_dir / "optimization_report.npz"), **cols)

    opt_mask = cols["optimized"]
    n_opt = int(opt_mask.sum())
    gains = cols["delta_logdet_opt"][opt_mask] - cols["delta_logdet_orig"][opt_mask]
    debug_log(
        f"Optimization report: {n_opt}/{len(diagnostics)} optimized, "
        f"{int(cols['fallback'].sum())} fallback. "
        f"mean delta_logdet gain={np.nanmean(gains) if n_opt else 0:.5f}, "
        f"mean max_disp={np.nanmean(cols['max_disp'][opt_mask]) if n_opt else 0:.3f} A, "
        f"min(min_dist_after)={np.nanmin(cols['min_dist_after'][opt_mask]) if n_opt else 0:.3f} A"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entropy downselect with in-loop structure optimization.",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        required=True,
        help="Directory with *_features.npy and *_metadata.parquet.",
    )
    parser.add_argument(
        "--seed-features",
        type=str,
        required=True,
        help="Seed features .npy (initial covariance).",
    )
    parser.add_argument(
        "--lmdb-dir",
        type=str,
        required=True,
        help="Parent dir of source <stem>/data.lmdb structure files.",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory for output files."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help="fairchem inference checkpoint for the featurizer.",
    )
    parser.add_argument(
        "--n-select",
        type=int,
        default=2000,
        help="Number of structures to select (default: 2000).",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--regularization", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--pool-factor", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only first N candidate structures (for testing).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--opt-max-steps",
        type=int,
        default=5,
        help="Max gradient-ascent steps per structure (default: 5).",
    )
    parser.add_argument(
        "--opt-max-disp",
        type=float,
        default=0.3,
        help="Max per-atom displacement in Angstrom (default: 0.3).",
    )
    parser.add_argument(
        "--opt-step-size",
        type=float,
        default=0.05,
        help="Per-step max per-atom displacement in Angstrom along the normalized "
        "gradient (default: 0.05).",
    )
    parser.add_argument(
        "--opt-min-dist",
        type=float,
        default=0.7,
        help="Clash guard: reject geometries with any pair closer than "
        "this (Angstrom, default: 0.7).",
    )
    parser.add_argument(
        "--opt-top-n",
        type=int,
        default=None,
        help="Only optimize the first N selections (by rank). Non-batched mode only.",
    )
    parser.add_argument(
        "--batched",
        action="store_true",
        help="Batch each selection batch's optimizations into one forward+backward per "
        "step against the frozen batch-start covariance (much higher GPU utilization).",
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=1024,
        help="Atom budget per optimization sub-batch in --batched mode (default: 1024). "
        "The backward pass retains activations, so this is much smaller than an "
        "inference batch; raise it on large-memory GPUs.",
    )
    parser.add_argument(
        "--n-gpus",
        type=int,
        default=1,
        help="Number of GPUs. >1 shards each batch's optimizations across that many "
        "per-GPU worker processes (implies batched). Default: 1.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint. The covariance/selection resume from "
        "checkpoint.npz and the optimized LMDB + diagnostics JSONL are appended to "
        "(re-emitted ranks overwrite, so resume is consistent).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    debug_log(f"Loading seed features from {args.seed_features}...")
    seed_X = np.load(args.seed_features).astype(np.float32)
    debug_log(f"Seed features shape: {seed_X.shape}")

    debug_log("Loading candidate features...")
    X, file_boundaries = load_features(args.features_dir, limit=args.limit)

    debug_log("Whitening (transform computed from seed features)...")
    seed_mean, W = whiten(X, ref=seed_X, reg=args.regularization)
    seed_X -= seed_mean.astype(np.float32)
    seed_X = seed_X @ W.astype(np.float32)
    debug_log("Seed features whitened")

    opt = OptParams(
        max_steps=args.opt_max_steps,
        max_disp=args.opt_max_disp,
        step_size=args.opt_step_size,
        min_dist=args.opt_min_dist,
    )

    run_kwargs: dict = {}
    if args.n_gpus and args.n_gpus > 1:
        # Multi-GPU: no featurizer/CUDA in the parent -- the spawned workers own the GPUs.
        debug_log(f"Mode: multi-GPU batched optimization ({args.n_gpus} GPUs)")
        hook = MultiGPUBatchOptimizingHook(
            X=X,
            file_boundaries=file_boundaries,
            lmdb_dir=args.lmdb_dir,
            output_dir=output_dir,
            resume=args.resume,
            model_path=args.model_path,
            mu_seed=seed_mean,
            W=W,
            opt=opt,
            max_atoms=args.max_atoms,
            n_gpus=args.n_gpus,
        )
        run_kwargs["batch_optimize_fn"] = hook
    elif args.batched:
        debug_log(f"Mode: single-GPU batched optimization (model on {args.device})")
        featurizer = DifferentiableFeaturizer(args.model_path, device=args.device)
        optimizer = BatchedStructureOptimizer(
            featurizer, seed_mean, W, opt, max_atoms=args.max_atoms
        )
        hook = BatchedOptimizingHook(
            X=X,
            optimizer=optimizer,
            file_boundaries=file_boundaries,
            lmdb_dir=args.lmdb_dir,
            output_dir=output_dir,
            resume=args.resume,
        )
        run_kwargs["batch_optimize_fn"] = hook
    else:
        debug_log(f"Mode: per-structure optimization (model on {args.device})")
        featurizer = DifferentiableFeaturizer(args.model_path, device=args.device)
        hook = OptimizingCommitHook(
            X=X,
            featurizer=featurizer,
            file_boundaries=file_boundaries,
            lmdb_dir=args.lmdb_dir,
            mu_seed=seed_mean,
            W=W,
            opt=opt,
            output_dir=output_dir,
            resume=args.resume,
            opt_top_n=args.opt_top_n,
        )
        run_kwargs["commit_hook"] = hook

    debug_log(f"Starting selection+optimization: {args.n_select:,} from {X.shape[0]:,}")
    results = run_selection(
        X=X,
        n_select=args.n_select,
        seed_features=seed_X,
        reg=args.regularization,
        batch_size=args.batch_size,
        pool_factor=args.pool_factor,
        checkpoint_every=args.checkpoint_every,
        output_dir=output_dir,
        resume=args.resume,
        random_seed=args.random_seed,
        **run_kwargs,
    )

    debug_log("Saving outputs...")
    save_outputs(output_dir, results, args.features_dir, file_boundaries, hook)
    hook.close()

    elapsed = time.time() - t_start
    n_sel = len(results["selected_indices"])
    debug_log(
        f"Done. Selected+optimized {n_sel:,} structures in {elapsed:.1f}s. "
        f"Final log-det: {results['log_dets'][-1]:.4f}"
    )


if __name__ == "__main__":
    main()
