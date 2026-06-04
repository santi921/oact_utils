"""Per-GPU worker process for multi-GPU batched structure optimization.

A pool of these workers (one per GPU) is driven by ``MultiGPUBatchOptimizingHook`` in
``entropy_downselect_optimize``. Each worker loads the model once on its GPU and, per
batch, optimizes its shard of the batch's structures (which it loads from the source
LMDBs) against the frozen covariance sent by the coordinator.

This module imports only the featurizer/optimizer/core (NOT the optimize script), so it
is safe to use as a ``torch.multiprocessing`` spawn target without re-importing the main
program. The coordinator owns the output streaming; workers only compute.

Worker protocol (over multiprocessing queues):
  - On startup the worker emits ``(gpu_id, "ready", None)`` once the model is loaded.
  - Each task is ``(batch_id, items, x_orig, C_inv, mu, n_current)`` where ``items`` is a
    list of ``(rank, global_index)``; the worker replies
    ``(gpu_id, batch_id, [(rank, global_index, feature, atoms, diag), ...])``.
  - A ``None`` task is the shutdown sentinel.
  - On any exception the worker emits ``(gpu_id, "error", traceback_str)`` and exits.
"""

from __future__ import annotations

import pickle
import traceback
from pathlib import Path

import lmdb

from oact_utilities.scripts.entropy_downselect.batched_optimizer import (
    BatchedStructureOptimizer,
    OptParams,
)
from oact_utilities.scripts.entropy_downselect.featurizer import (
    DifferentiableFeaturizer,
)
from oact_utilities.utils.entropy_selection import debug_log, make_index_resolver


def worker_main(
    gpu_id: int,
    model_path: str,
    mu_seed,
    W,
    opt: OptParams,
    max_atoms: int,
    file_boundaries: list,
    lmdb_dir: str,
    in_q,
    out_q,
) -> None:
    """Entry point for a single per-GPU optimization worker (spawn target)."""
    try:
        import torch

        # fairchem accepts only the literal "cuda"; pin this worker to its GPU via
        # set_device so "cuda" resolves to the right physical device.
        torch.cuda.set_device(gpu_id)
        featurizer = DifferentiableFeaturizer(model_path, device="cuda")
        debug_log(
            f"worker {gpu_id} on cuda:{torch.cuda.current_device()} "
            f"({torch.cuda.get_device_name(torch.cuda.current_device())})"
        )
        optimizer = BatchedStructureOptimizer(
            featurizer, mu_seed, W, opt, max_atoms=max_atoms
        )
        resolve = make_index_resolver(file_boundaries)
        lmdb_base = Path(lmdb_dir)
        env_cache: dict[str, lmdb.Environment] = {}

        def load(global_idx: int):
            stem, local = resolve(int(global_idx))
            env = env_cache.get(stem)
            if env is None:
                env = lmdb.open(
                    str(lmdb_base / stem / "data.lmdb"),
                    readonly=True,
                    lock=False,
                    subdir=False,
                )
                env_cache[stem] = env
            with env.begin() as txn:
                return pickle.loads(txn.get(f"{local}".encode("ascii")))

        out_q.put((gpu_id, "ready", None))

        while True:
            task = in_q.get()
            if task is None:
                break
            batch_id, items, x_orig, C_inv, mu, n_current = task
            if not items:
                out_q.put((gpu_id, batch_id, []))
                continue
            atoms_list = [load(g) for _r, g in items]
            res = optimizer.optimize(atoms_list, C_inv, mu, n_current, x_orig=x_orig)
            payload = [
                (items[k][0], int(items[k][1]), res[k][0], res[k][1], res[k][2])
                for k in range(len(items))
            ]
            out_q.put((gpu_id, batch_id, payload))

        for env in env_cache.values():
            env.close()
    except Exception:  # noqa: BLE001 - report any failure to the coordinator
        out_q.put((gpu_id, "error", traceback.format_exc()))
