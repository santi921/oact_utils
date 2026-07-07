"""Label LMDB-stored structures with energy + forces from the v1 fairchem model.

Reads ASE Atoms from an ``optimized_structures.lmdb`` (keyed "0".."N-1" by selection
rank, plus a "length" key), runs batched inference with the fine-tuned v1 checkpoint,
and saves per-structure energy and per-atom force labels.

Unlike ``run_lmdb_inference.py`` (which extracts no-grad features), this keeps forces
enabled and does NOT wrap ``predict`` in ``torch.no_grad`` -- the conserving model needs
autograd through positions to produce forces.

Distributed: run one task per GPU. Each task (rank ``SLURM_PROCID`` of ``SLURM_NTASKS``)
processes a strided shard of the dataset and writes ``{stem}_labels_rank{R}.npz``. A final
``--merge`` pass concatenates the shards (sorted by structure index) into
``{stem}_labels.npz``.

Usage (per rank, launched via srun):
    python -m oact_utilities.scripts.entropy_downselect.run_lmdb_labels \
        <input.lmdb> -o <output_dir>

Merge (once, after all ranks finish):
    python -m oact_utilities.scripts.entropy_downselect.run_lmdb_labels \
        <input.lmdb> -o <output_dir> --merge
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from functools import partial
from pathlib import Path

import lmdb
import numpy as np
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.collaters.simple_collater import data_list_collater
from fairchem.core.datasets.samplers.max_atom_distributed_sampler import get_batches
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
from torch.utils.data import DataLoader
from tqdm import tqdm

from oact_utilities.scripts.entropy_downselect.run_lmdb_inference import (
    MODEL_PATH,
    IndexTrackingBatchSampler,
    LmdbAtomicDataset,
    debug_log,
)


def create_predictor(
    model_path: str, device: str = "cuda", compile_model: bool = False
):
    """Load the v1 predict unit with forces ENABLED (no no_grad, no force disabling)."""
    inference_settings = InferenceSettings(
        tf32=False,
        activation_checkpointing=False,
        merge_mole=False,
        compile=compile_model,
        external_graph_gen=False,
    )
    return load_predict_unit(
        path=model_path, device=device, inference_settings=inference_settings
    )


def _stem_of(lmdb_path: str) -> str:
    p = Path(lmdb_path)
    return p.parent.name if p.stem == "data" else p.stem


def _shard_path(output_dir: Path, stem: str, rank: int) -> Path:
    return output_dir / f"{stem}_labels_rank{rank}.npz"


def _final_path(output_dir: Path, stem: str) -> Path:
    return output_dir / f"{stem}_labels.npz"


def run_labels(
    lmdb_path: str,
    output_dir: str,
    model_path: str = MODEL_PATH,
    max_atoms: int = 1024,
    num_workers: int = 4,
    device: str = "cuda",
    rank: int = 0,
    world_size: int = 1,
    compile_model: bool = False,
    flush_every: int = 200,
    resume: bool = False,
    limit: int | None = None,
) -> None:
    """Label this rank's shard of the dataset and write a per-rank NPZ."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _stem_of(lmdb_path)
    shard_path = _shard_path(output_dir, stem, rank)

    debug_log(f"[rank {rank}/{world_size}] Loading model from {model_path}")
    predictor = create_predictor(model_path, device=device, compile_model=compile_model)
    task_name = list(predictor.dataset_to_tasks.keys())[0]
    debug_log(f"[rank {rank}] task_name={task_name}")

    a2g_fn = partial(
        AtomicData.from_ase,
        task_name=task_name,
        r_edges=False,
        r_data_keys=["spin", "charge"],
        radius=6.0,
        target_dtype=predictor.inference_settings.base_precision_dtype,
    )

    dataset = LmdbAtomicDataset(lmdb_path, a2g_fn)
    total = len(dataset) if limit is None else min(limit, len(dataset))

    # natoms for balanced batching
    lmdb_dir = Path(lmdb_path).parent
    metadata_npz = lmdb_dir / "metadata.npz"
    if metadata_npz.exists():
        natoms_all = np.load(str(metadata_npz))["natoms"][:total].astype(np.int64)
    else:
        debug_log(f"[rank {rank}] No metadata.npz, scanning natoms...")
        env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)
        with env.begin() as txn:
            natoms_all = np.array(
                [
                    len(pickle.loads(txn.get(f"{i}".encode("ascii"))))
                    for i in range(total)
                ],
                dtype=np.int64,
            )
        env.close()

    # This rank's strided shard of structure indices.
    assigned = np.arange(rank, total, world_size, dtype=np.int64)

    # Resume: drop indices already present in an existing shard.
    done_energy: dict[int, float] = {}
    done_forces: dict[int, np.ndarray] = {}
    if resume and shard_path.exists():
        prev = np.load(str(shard_path), allow_pickle=False)
        prev_idx = prev["indices"]
        splits = np.split(prev["forces"], prev["offsets"][1:-1])
        for k, gi in enumerate(prev_idx):
            done_energy[int(gi)] = float(prev["energies"][k])
            done_forces[int(gi)] = splits[k]
        assigned = assigned[~np.isin(assigned, prev_idx)]
        debug_log(
            f"[rank {rank}] Resume: {len(done_energy)} done, {len(assigned)} remaining"
        )

    if len(assigned) == 0:
        debug_log(f"[rank {rank}] Nothing to do; writing shard.")
        _write_shard(shard_path, done_energy, done_forces)
        dataset.close()
        return

    batches, atom_counts, filtered = get_batches(
        natoms_all[assigned], assigned, max_atoms=max_atoms, min_atoms=0
    )
    debug_log(
        f"[rank {rank}] {len(assigned)} structs -> {len(batches)} batches, "
        f"mean atoms/batch {np.mean(atom_counts):.0f}, filtered {filtered}"
    )

    loader = DataLoader(
        dataset,
        batch_sampler=IndexTrackingBatchSampler(batches),
        collate_fn=partial(data_list_collater, otf_graph=True),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    t0 = time.time()
    done_batches = 0
    for batch_idx, batch_data in enumerate(
        tqdm(loader, desc=f"rank{rank} label", total=len(batches))
    ):
        batch_indices = batches[batch_idx]
        # Forces require grad; do NOT use torch.no_grad. predict() enables grad internally.
        preds = predictor.predict(batch_data)
        energies = preds["energy"].detach().float().cpu().numpy()
        forces = preds["forces"].detach().float().cpu().numpy()

        natoms_batch = batch_data["natoms"].cpu().numpy()
        offsets = np.zeros(len(natoms_batch) + 1, dtype=np.int64)
        np.cumsum(natoms_batch, out=offsets[1:])
        for k, gi in enumerate(batch_indices):
            done_energy[int(gi)] = float(energies[k])
            done_forces[int(gi)] = forces[offsets[k] : offsets[k + 1]]

        done_batches += 1
        if done_batches % flush_every == 0:
            _write_shard(shard_path, done_energy, done_forces)

    _write_shard(shard_path, done_energy, done_forces)
    dataset.close()
    elapsed = time.time() - t0
    n = len(done_energy)
    debug_log(
        f"[rank {rank}] Done: {n} structures in {elapsed:.1f}s "
        f"({n / max(elapsed, 0.01):.0f} struct/s) -> {shard_path}"
    )


def _write_shard(
    path: Path, energy: dict[int, float], forces: dict[int, np.ndarray]
) -> None:
    """Atomically write a per-rank shard NPZ (indices sorted ascending)."""
    idx = np.array(sorted(energy.keys()), dtype=np.int64)
    energies = np.array([energy[int(i)] for i in idx], dtype=np.float64)
    natoms = np.array([len(forces[int(i)]) for i in idx], dtype=np.int32)
    offsets = np.zeros(len(idx) + 1, dtype=np.int64)
    np.cumsum(natoms, out=offsets[1:])
    forces_flat = (
        np.concatenate([forces[int(i)] for i in idx], axis=0).astype(np.float32)
        if len(idx)
        else np.zeros((0, 3), dtype=np.float32)
    )
    tmp = path.with_suffix(".tmp.npz")
    np.savez(
        str(tmp),
        indices=idx,
        energies=energies,
        forces=forces_flat,
        natoms=natoms,
        offsets=offsets,
    )
    tmp.rename(path)


def merge_shards(lmdb_path: str, output_dir: str, world_size: int) -> None:
    """Concatenate all rank shards into a single {stem}_labels.npz sorted by index."""
    output_dir = Path(output_dir)
    stem = _stem_of(lmdb_path)
    shards = sorted(output_dir.glob(f"{stem}_labels_rank*.npz"))
    if not shards:
        raise FileNotFoundError(f"No shards matching {stem}_labels_rank*.npz")
    debug_log(f"Merging {len(shards)} shards for {stem}")

    all_idx, all_e, all_nat, all_f = [], [], [], []
    for sp in tqdm(shards, desc="Reading shards"):
        d = np.load(str(sp), allow_pickle=False)
        all_idx.append(d["indices"])
        all_e.append(d["energies"])
        all_nat.append(d["natoms"])
        all_f.append(d["forces"])

    idx = np.concatenate(all_idx)
    energies = np.concatenate(all_e)
    natoms = np.concatenate(all_nat)
    forces_by_shard = all_f

    # Rebuild per-structure force arrays, then reorder everything by structure index.
    forces_list: list[np.ndarray] = []
    for d_forces, d_nat in zip(forces_by_shard, all_nat):
        off = np.zeros(len(d_nat) + 1, dtype=np.int64)
        np.cumsum(d_nat, out=off[1:])
        forces_list.extend(d_forces[off[k] : off[k + 1]] for k in range(len(d_nat)))

    order = np.argsort(idx, kind="stable")
    idx = idx[order]
    if len(np.unique(idx)) != len(idx):
        raise ValueError("Duplicate structure indices across shards; check world_size.")
    energies = energies[order]
    natoms = natoms[order]
    forces_list = [forces_list[i] for i in order]

    offsets = np.zeros(len(idx) + 1, dtype=np.int64)
    np.cumsum(natoms, out=offsets[1:])
    forces_flat = np.concatenate(forces_list, axis=0).astype(np.float32)

    out_path = _final_path(output_dir, stem)
    np.savez(
        str(out_path),
        indices=idx,
        energies=energies,
        forces=forces_flat,
        natoms=natoms.astype(np.int32),
        offsets=offsets,
    )
    debug_log(
        f"Merged {len(idx)} structures ({forces_flat.shape[0]} atoms) -> {out_path}"
    )
    # forces for structure i (row order = structure index): forces_flat[offsets[i]:offsets[i+1]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label LMDB structures with v1-model energy + forces."
    )
    parser.add_argument(
        "input_path", type=str, help="Path to optimized_structures.lmdb"
    )
    parser.add_argument("-o", "--output-dir", type=str, required=True)
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--max-atoms", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile.")
    parser.add_argument("--flush-every", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge existing rank shards into the final labels NPZ (no inference).",
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.merge:
        merge_shards(args.input_path, args.output_dir, args.world_size)
    else:
        run_labels(
            lmdb_path=args.input_path,
            output_dir=args.output_dir,
            model_path=args.model_path,
            max_atoms=args.max_atoms,
            num_workers=args.num_workers,
            device=args.device,
            rank=args.rank,
            world_size=args.world_size,
            compile_model=args.compile,
            flush_every=args.flush_every,
            resume=args.resume,
            limit=args.limit,
        )
