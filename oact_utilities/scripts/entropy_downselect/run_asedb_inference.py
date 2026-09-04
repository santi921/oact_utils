"""Extract pre-activation features from a fairchem model using AseDBDataset-format structures.

Reads structures from a fairchem-native ``ase_db`` directory (``*.aselmdb`` shards +
``metadata.npz``, e.g. OMOL/act/nonact training data), runs batched inference with the same
forward-hook feature-capture technique as ``run_lmdb_inference.py``, and writes a single
``<stem>_features.npy`` (no metadata parquet — this is meant for building seed feature arrays
for entropy downselect, which need only a flat feature array, not per-structure identity).

Distributed: run one process per GPU. Each process (rank ``SLURM_PROCID`` of ``SLURM_NTASKS``,
both overridable) processes a strided shard of the dataset (``indices = rank, rank+world_size,
...``) and writes its own ``<stem>_features.npy``. Unlike ``run_lmdb_inference.py``, there is no
memmap/resume machinery: chunks here are small enough (at most ~1M/6 structures) to accumulate
in memory and save once at the end, and seed-feature row order has no downstream meaning.

Usage (per rank):
    python -m oact_utilities.scripts.entropy_downselect.run_asedb_inference \
        /pscratch/sd/i/ishan_a/OMOL/4M/train_1M \
        -o /pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/seed_features_v3 \
        --stem omol_rank0 --model-path <ckpt> --rank 0 --world-size 6
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fairchem.core.datasets.ase_datasets import AseDBDataset
from fairchem.core.datasets.collaters.simple_collater import data_list_collater
from fairchem.core.datasets.samplers.max_atom_distributed_sampler import get_batches

from oact_utilities.scripts.entropy_downselect.run_lmdb_inference import (
    MODEL_PATH,
    IndexTrackingBatchSampler,
    _find_last_linear,
    _get_torch_model_from_predictor,
    create_predictor,
)


def debug_log(message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [pid={os.getpid()}] {message}", flush=True)


def build_dataset(src: str, task_name: str, target_dtype) -> AseDBDataset:
    """Construct an AseDBDataset matching the training a2g config.

    ``src`` must always be the full source directory (never a partial list of shard files) —
    BaseDataset._metadata looks up metadata.npz once per config["src"] entry and concatenates
    whatever it finds, so a partial file list re-reads and concatenates the same directory-level
    metadata.npz multiple times, producing a wrong-length natoms array.
    """
    config = {
        "src": str(src),
        "a2g_args": {
            "task_name": task_name,
            "molecule_cell_size": 120.0,
            "r_energy": True,
            "r_forces": True,
            "r_edges": False,
            "r_data_keys": ["spin", "charge"],
            "radius": 6.0,
            "target_dtype": target_dtype,
        },
        "key_mapping": {"energy": "energy", "forces": "forces"},
    }
    return AseDBDataset(config)


def load_natoms(src: str, dataset: AseDBDataset, total: int) -> np.ndarray:
    metadata_npz = Path(src) / "metadata.npz"
    if metadata_npz.exists():
        return np.load(str(metadata_npz))["natoms"][:total].astype(np.int64)
    debug_log(f"No metadata.npz in {src}; scanning natoms...")
    return np.array(
        [int(dataset.get_metadata("natoms", i)) for i in tqdm(range(total), desc="Scanning natoms")],
        dtype=np.int64,
    )


def run_inference(
    src: str,
    output_dir: str,
    stem: str,
    model_path: str,
    max_atoms: int,
    num_workers: int,
    device: str,
    rank: int,
    world_size: int,
    limit: int | None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    debug_log(f"[rank {rank}/{world_size}] Loading model {model_path}")
    predictor = create_predictor(model_path=model_path, device=device)
    torch_model = _get_torch_model_from_predictor(predictor)
    hook_name, hook_layer = _find_last_linear(torch_model)
    feature_dim = hook_layer.in_features
    debug_log(f"[rank {rank}] Hook target: {hook_name}, feature dim: {feature_dim}")

    avail_tasks = list(predictor.dataset_to_tasks.keys())
    task_name = avail_tasks[0]
    debug_log(f"[rank {rank}] available tasks: {avail_tasks}, using: {task_name}")

    dataset = build_dataset(src, task_name, predictor.inference_settings.base_precision_dtype)
    total = len(dataset) if limit is None else min(limit, len(dataset))
    debug_log(f"[rank {rank}] Dataset: {total} structures")

    natoms_all = load_natoms(src, dataset, total)
    assigned = np.arange(rank, total, world_size, dtype=np.int64)
    batches, atom_counts, filtered = get_batches(
        natoms_all[assigned], assigned, max_atoms=max_atoms, min_atoms=0
    )
    debug_log(
        f"[rank {rank}/{world_size}] {len(assigned)} structs assigned -> "
        f"{len(batches)} batches, mean atoms/batch: {np.mean(atom_counts):.0f}, "
        f"filtered: {filtered}"
    )

    loader = DataLoader(
        dataset,
        batch_sampler=IndexTrackingBatchSampler(batches),
        collate_fn=partial(data_list_collater, otf_graph=True),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    last_pre_activation = None

    def hook_fn(_mod, inp, _out):
        nonlocal last_pre_activation
        if isinstance(inp, (tuple, list)) and len(inp) > 0 and torch.is_tensor(inp[0]):
            last_pre_activation = inp[0].detach()
        elif torch.is_tensor(inp):
            last_pre_activation = inp.detach()
        else:
            last_pre_activation = None

    handle = hook_layer.register_forward_hook(hook_fn)

    all_feats: list[np.ndarray] = []
    oom_skipped = 0
    t0 = time.time()
    for batch_idx, batch_data in enumerate(
        tqdm(loader, desc=f"Inference rank{rank}", total=len(batches))
    ):
        try:
            with torch.no_grad():
                predictor.predict(batch_data)
        except torch.OutOfMemoryError:
            # A structure's spatial extent occasionally blows up radius_graph_pbc_v2's grid
            # tensor (independent of atom count) — rare and data-dependent. Seed features don't
            # need full coverage, so skip this batch rather than crash the whole worker.
            debug_log(
                f"[rank {rank}] WARNING: OOM on batch {batch_idx} "
                f"(dataset indices {batches[batch_idx]}), skipping"
            )
            oom_skipped += len(batches[batch_idx])
            torch.cuda.empty_cache()
            last_pre_activation = None
            continue

        if last_pre_activation is None:
            debug_log(f"[rank {rank}] WARNING: batch produced no pre-activation, skipping")
            continue

        x = last_pre_activation.cpu().numpy()
        natoms_batch = batch_data["natoms"].cpu().numpy()
        first_atom_offsets = np.zeros(len(natoms_batch), dtype=np.int64)
        np.cumsum(natoms_batch[:-1], out=first_atom_offsets[1:])
        all_feats.append(x[first_atom_offsets])

    handle.remove()

    if oom_skipped > 0:
        debug_log(f"[rank {rank}] WARNING: {oom_skipped} structures skipped due to OOM")

    processed = sum(f.shape[0] for f in all_feats)
    elapsed = time.time() - t0
    debug_log(
        f"[rank {rank}] Inference done: {processed} structures in {elapsed:.1f}s "
        f"({processed / max(elapsed, 1e-9):.0f} struct/s)"
    )
    if filtered > 0:
        debug_log(f"[rank {rank}] WARNING: {filtered} structures filtered (exceeded max-atoms)")

    X = np.concatenate(all_feats, axis=0).astype(np.float32)
    features_path = output_dir / f"{stem}_features.npy"
    np.save(str(features_path), X)
    debug_log(f"[rank {rank}] Written: {features_path} ({X.shape})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract pre-activation features from an AseDBDataset via fairchem model."
    )
    parser.add_argument("src", type=str, help="Path to the ase_db source directory.")
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        required=True,
        help="Directory for the output NPY file.",
    )
    parser.add_argument(
        "--stem",
        type=str,
        required=True,
        help="Output file stem (written as <stem>_features.npy).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help="Path to fairchem inference checkpoint.",
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=1024,
        help=(
            "Maximum atoms per batch for balanced sampler (default: 1024, matching "
            "run_lmdb_eval.sh). Keep this small for molecule_cell_size-based PBC sources: "
            "large combined-batch atom counts blow up radius_graph_pbc's spatial grid tensor "
            "(see radius_graph_pbc.py's own 'boxsize is large and PBC is on' OOM comment)."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes (default: 4).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device (default: cuda).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=int(os.environ.get("SLURM_PROCID", 0)),
        help="Distributed rank (default: SLURM_PROCID).",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=int(os.environ.get("SLURM_NTASKS", 1)),
        help="Distributed world size (default: SLURM_NTASKS).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N structures (for testing).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        src=args.src,
        output_dir=args.output_dir,
        stem=args.stem,
        model_path=args.model_path,
        max_atoms=args.max_atoms,
        num_workers=args.num_workers,
        device=args.device,
        rank=args.rank,
        world_size=args.world_size,
        limit=args.limit,
    )
