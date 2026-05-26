"""Extract pre-activation features from a fairchem model using LMDB-stored structures.

Reads ASE Atoms from raw LMDB (created by pkl_to_lmdb.py), runs batched inference
with a balanced batch sampler (grouped by natoms), and writes features to a
memory-mapped NPY file + metadata Parquet.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from collections import defaultdict
from datetime import datetime
from functools import partial
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.collaters.simple_collater import data_list_collater
from fairchem.core.datasets.samplers.max_atom_distributed_sampler import get_batches
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

MODEL_PATH = "/pscratch/sd/i/ishan_a/open_actinides/runs/202605-0213-3947-5676/checkpoints/final/inference_ckpt.pt"

METADATA_KEYS = [
    "name",
    "gen_unique_name",
    "is_distorted",
    "distortion_variant_index",
    "metal",
    "met_chrg_spn",
]


def debug_log(message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [pid={os.getpid()}] {message}", flush=True)


class LmdbAtomicDataset(Dataset):
    """PyTorch Dataset that reads pickled ASE Atoms from raw LMDB."""

    def __init__(self, lmdb_path: str, a2g_fn):
        self.lmdb_path = lmdb_path
        self.a2g_fn = a2g_fn
        self._env = None
        # Read length from a temporary env (closed immediately)
        env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)
        with env.begin() as txn:
            self.length = pickle.loads(txn.get(b"length"))
        env.close()

    @property
    def env(self):
        # Lazy open: safe across DataLoader fork boundaries
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path, readonly=True, lock=False, readahead=False,
                subdir=False,
            )
        return self._env

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> AtomicData:
        with self.env.begin() as txn:
            atoms = pickle.loads(txn.get(f"{idx}".encode("ascii")))
        return self.a2g_fn(atoms)

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None


class IndexTrackingBatchSampler:
    """Batch sampler that yields pre-computed index lists from get_batches."""

    def __init__(self, batches: list[list[int]]):
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


def _get_torch_model_from_predictor(predictor):
    for attr in ("module", "model", "inference_model", "_model"):
        model = getattr(predictor, attr, None)
        if isinstance(model, torch.nn.Module):
            return model
    for _, value in vars(predictor).items():
        if isinstance(value, torch.nn.Module):
            return value
    raise AttributeError(
        "Could not find underlying torch.nn.Module on predictor. "
        "Inspect `dir(predictor)` and update _get_torch_model_from_predictor()."
    )


def _find_last_linear(module: torch.nn.Module) -> tuple[str, torch.nn.Linear]:
    last_name = None
    last_layer = None
    for name, submodule in module.named_modules():
        if isinstance(submodule, torch.nn.Linear):
            last_name, last_layer = name, submodule
    if last_layer is None:
        raise RuntimeError("No torch.nn.Linear found in model.")
    return last_name, last_layer


def _disable_forces_and_stress(predictor) -> None:
    """Disable force/stress so conditional_grad won't override torch.no_grad()."""
    hydra = predictor.model.module
    if hasattr(hydra.backbone, "regress_config"):
        hydra.backbone.regress_config.forces = False
        hydra.backbone.regress_config.stress = False
    hydra._tasks = {
        k: v for k, v in hydra._tasks.items()
        if v.property not in ("forces", "stress")
    }
    dset_map = defaultdict(list)
    for task in hydra._tasks.values():
        for ds in task.datasets:
            dset_map[ds].append(task)
    hydra._dataset_to_tasks = dict(dset_map)


def create_predictor(model_path: str, device: str = "cuda"):
    """Load fairchem predict unit with inference settings."""
    inference_settings = InferenceSettings(
        tf32=False,
        activation_checkpointing=False,
        merge_mole=False,
        compile=True,
        external_graph_gen=False,
    )
    predictor = load_predict_unit(
        path=model_path, device=device, inference_settings=inference_settings
    )
    _disable_forces_and_stress(predictor)
    return predictor


def extract_metadata(lmdb_path: str, length: int) -> pd.DataFrame:
    """Read metadata from all LMDB entries (fast sequential scan)."""
    env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)
    rows = []
    with env.begin() as txn:
        for idx in tqdm(range(length), desc="Reading metadata"):
            atoms = pickle.loads(txn.get(f"{idx}".encode("ascii")))
            row = {}
            for k in METADATA_KEYS:
                v = atoms.info.get(k)
                row[k] = v.item() if hasattr(v, "item") else v
            rows.append(row)
    env.close()
    return pd.DataFrame(rows)


def run_inference(
    lmdb_path: str,
    output_dir: str,
    model_path: str = MODEL_PATH,
    max_atoms: int = 4096,
    num_workers: int = 4,
    device: str = "cuda",
    limit: int | None = None,
    resume: bool = False,
) -> None:
    """Main inference loop: LMDB -> balanced batches -> GPU -> NPY + Parquet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Use parent directory name as stem (files are now <dataset>/data.lmdb)
    lmdb_file = Path(lmdb_path)
    stem = lmdb_file.parent.name if lmdb_file.stem == "data" else lmdb_file.stem

    # Load model
    debug_log(f"Loading model from {model_path}")
    predictor = create_predictor(model_path=model_path, device=device)
    torch_model = _get_torch_model_from_predictor(predictor)
    hook_name, hook_layer = _find_last_linear(torch_model)
    debug_log(f"Hook target: {hook_name}")

    # Set up a2g function
    task_name = list(predictor.dataset_to_tasks.keys())[0]
    a2g_fn = partial(
        AtomicData.from_ase,
        task_name=task_name,
        r_edges=False,
        r_data_keys=["spin", "charge"],
        radius=6.0,
        target_dtype=predictor.inference_settings.base_precision_dtype,
    )

    # Open dataset
    dataset = LmdbAtomicDataset(lmdb_path, a2g_fn)
    total_structures = len(dataset) if limit is None else min(limit, len(dataset))
    debug_log(f"Dataset: {total_structures} structures")

    # Load natoms for batch sampler (look in same directory as the .lmdb)
    lmdb_dir = Path(lmdb_path).parent
    metadata_npz = lmdb_dir / "metadata.npz"
    if not metadata_npz.exists():
        metadata_npz = lmdb_dir / f"{stem}_metadata.npz"
    if metadata_npz.exists():
        natoms_array = np.load(str(metadata_npz))["natoms"][:total_structures]
    else:
        debug_log("No metadata.npz found, scanning LMDB for natoms...")
        env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)
        natoms_list = []
        with env.begin() as txn:
            for idx in tqdm(range(total_structures), desc="Scanning natoms"):
                atoms = pickle.loads(txn.get(f"{idx}".encode("ascii")))
                natoms_list.append(len(atoms))
        env.close()
        natoms_array = np.array(natoms_list, dtype=np.int32)

    # Greedy atom-budget batches: each batch is dataset indices whose total natoms <= max_atoms.
    indices = np.arange(total_structures, dtype=np.int64)
    batches: list[list[int]]
    atom_counts: list[int]
    filtered: int
    batches, atom_counts, filtered = get_batches(
        natoms_array.astype(np.int64), indices, max_atoms=max_atoms, min_atoms=0
    )
    debug_log(
        f"Created {len(batches)} batches, "
        f"mean atoms/batch: {np.mean(atom_counts):.0f}, "
        f"filtered: {filtered}"
    )

    # DataLoader with balanced batch sampler
    sampler = IndexTrackingBatchSampler(batches)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=partial(data_list_collater, otf_graph=True),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    # Output paths
    features_path = output_dir / f"{stem}_features.npy"
    progress_path = output_dir / f"{stem}_progress.json"

    # Resume: load completed batch indices and open existing mmap
    completed_batches: set[int] = set()
    features_mmap = None
    feature_dim = None

    if resume and features_path.exists() and progress_path.exists():
        completed_batches = set(json.loads(progress_path.read_text()))
        features_mmap = np.lib.format.open_memmap(str(features_path), mode="r+")
        feature_dim = features_mmap.shape[1]
        debug_log(f"Resuming: {len(completed_batches)}/{len(batches)} batches already done")

    # Hook state
    last_pre_activation = None

    def hook_fn(_mod, inp, _out):
        nonlocal last_pre_activation
        if isinstance(inp, (tuple, list)) and len(inp) > 0 and torch.is_tensor(inp[0]):
            last_pre_activation = inp[0].detach()
        elif torch.is_tensor(inp):
            last_pre_activation = inp.detach()
        else:
            last_pre_activation = None

    # Inference loop
    remaining = len(batches) - len(completed_batches)
    debug_log(f"Starting inference: {remaining} batches to process...")
    t0 = time.time()
    processed = 0
    skipped = 0
    handle = hook_layer.register_forward_hook(hook_fn)

    for batch_idx, batch_data in enumerate(tqdm(loader, desc="Inference", total=len(batches))):
        if batch_idx in completed_batches:
            skipped += 1
            continue

        batch_indices = batches[batch_idx]

        with torch.no_grad():
            predictor.predict(batch_data)

        if last_pre_activation is None:
            debug_log(f"WARNING: batch {batch_idx} produced no pre-activation")
            continue

        x = last_pre_activation.cpu().numpy()

        # Initialize mmap on first real batch
        if features_mmap is None:
            feature_dim = x.shape[1]
            debug_log(f"Feature dim: {feature_dim}")
            features_mmap = np.lib.format.open_memmap(
                str(features_path), mode="w+",
                shape=(total_structures, feature_dim), dtype=np.float32,
            )

        # Write features to correct output positions (vectorized scatter)
        natoms_batch = batch_data["natoms"].cpu().numpy()
        first_atom_offsets = np.zeros(len(natoms_batch), dtype=np.int64)
        np.cumsum(natoms_batch[:-1], out=first_atom_offsets[1:])
        features_mmap[batch_indices] = x[first_atom_offsets]

        processed += len(batch_indices)
        completed_batches.add(batch_idx)

        if (batch_idx + 1) % 500 == 0:
            features_mmap.flush()
            progress_path.write_text(json.dumps(sorted(completed_batches)))

    handle.remove()

    if features_mmap is not None:
        features_mmap.flush()

    # Save final progress (or clean up if fully done)
    if len(completed_batches) == len(batches):
        progress_path.unlink(missing_ok=True)
    else:
        progress_path.write_text(json.dumps(sorted(completed_batches)))

    elapsed = time.time() - t0
    debug_log(f"Inference done: {processed} structures in {elapsed:.1f}s ({processed/elapsed:.0f} struct/s)")

    # Extract and write metadata
    debug_log("Extracting metadata...")
    metadata_df = extract_metadata(lmdb_path, total_structures)
    parquet_path = output_dir / f"{stem}_metadata.parquet"
    metadata_df.to_parquet(str(parquet_path), index=False)
    debug_log(f"Written: {features_path} ({features_mmap.shape}), {parquet_path}")

    dataset.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract pre-activation features from LMDB structures via fairchem model."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to input .lmdb file.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="features_output",
        help="Directory for output NPY + Parquet files.",
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
        default=4096,
        help="Maximum atoms per batch for balanced sampler (default: 4096).",
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
        "--limit",
        type=int,
        default=None,
        help="Process only first N structures (for testing).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous progress (skip completed batches).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        lmdb_path=args.input_path,
        output_dir=args.output_dir,
        model_path=args.model_path,
        max_atoms=args.max_atoms,
        num_workers=args.num_workers,
        device=args.device,
        limit=args.limit,
        resume=args.resume,
    )
