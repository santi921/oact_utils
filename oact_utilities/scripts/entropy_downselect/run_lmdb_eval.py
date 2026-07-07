"""Evaluate a fairchem checkpoint on AseDBDataset val sets: energy + force MAE.

Loads each ``val`` directory as a fairchem ``AseDBDataset`` (``*.aselmdb`` + ``metadata.npz``),
runs batched inference with the fine-tuned checkpoint, and compares predictions to the DFT
ground-truth energy/forces carried on each structure. Reports two metrics per dataset,
matching fairchem's own eval conventions (``_metrics.py``):

    - Energy MAE per-atom (eV/atom): mean over structures of ``|E_pred - E_gt| / natoms``
    - Force MAE (eV/A): mean over all force components of ``|F_pred - F_gt|``

``predictor.predict`` returns TOTAL physical energy (element references added back +
denormalized) by default, directly comparable to the DFT label stored in the db. Forces
require autograd, so predict is NOT wrapped in ``torch.no_grad``.

Distributed: run one task per GPU. Each task (rank ``SLURM_PROCID`` of ``SLURM_NTASKS``)
processes a strided shard of every dataset and writes ``{name}_evalstats_rank{R}.npz``
(running sums, not per-structure output). A ``--merge`` pass reduces the shards and prints
the final MAE table. Single-process runs merge + print automatically.

Usage (per rank, launched via srun):
    python -m oact_utilities.scripts.entropy_downselect.run_lmdb_eval \
        <act/val> <nonact/val> -o <output_dir> --model-path <ckpt>

Merge (once, after all ranks finish):
    python -m oact_utilities.scripts.entropy_downselect.run_lmdb_eval \
        <act/val> <nonact/val> -o <output_dir> --merge --world-size <N>
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
from fairchem.core.datasets.ase_datasets import AseDBDataset
from fairchem.core.datasets.collaters.simple_collater import data_list_collater
from fairchem.core.datasets.samplers.max_atom_distributed_sampler import get_batches
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
from torch.utils.data import DataLoader
from tqdm import tqdm


def debug_log(message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [pid={os.getpid()}] {message}", flush=True)


class IndexTrackingBatchSampler:
    """Batch sampler that yields pre-computed index lists from get_batches."""

    def __init__(self, batches: list[list[int]]):
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


DEFAULT_MODEL_PATH = (
    "/global/homes/i/ishan_a/oact_utils/data/runs/"
    "202605-0213-3947-5676/checkpoints/final/inference_ckpt.pt"
)


def create_predictor(model_path: str, device: str = "cuda", compile_model: bool = False):
    """Load the predict unit with forces ENABLED (conserving model needs autograd)."""
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


def _dataset_name(val_dir: str) -> str:
    """Name a val set by its parent (``.../act/val`` -> ``act``) to avoid ``val`` collisions."""
    p = Path(val_dir)
    return p.parent.name if p.name == "val" else p.name


def _build_dataset(val_dir: str, task_name: str, target_dtype) -> AseDBDataset:
    """Construct an AseDBDataset matching the training a2g config (energy + forces ground truth)."""
    config = {
        "src": str(val_dir),
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


def _stats_path(output_dir: Path, name: str, rank: int) -> Path:
    return output_dir / f"{name}_evalstats_rank{rank}.npz"


def eval_dataset(
    val_dir: str,
    predictor,
    task_name: str,
    output_dir: Path,
    max_atoms: int,
    num_workers: int,
    rank: int,
    world_size: int,
    limit: int | None = None,
) -> None:
    """Evaluate this rank's strided shard of one val set; write running-sum stats NPZ."""
    name = _dataset_name(val_dir)
    dataset = _build_dataset(
        val_dir, task_name, predictor.inference_settings.base_precision_dtype
    )
    total = len(dataset) if limit is None else min(limit, len(dataset))

    # natoms for balanced batching (does NOT affect correctness, only batch grouping).
    metadata_npz = Path(val_dir) / "metadata.npz"
    if metadata_npz.exists():
        natoms_all = np.load(str(metadata_npz))["natoms"][:total].astype(np.int64)
    else:
        debug_log(f"[rank {rank}] No metadata.npz in {val_dir}; scanning natoms...")
        natoms_all = np.array(
            [int(dataset.get_metadata("natoms", i)) for i in range(total)],
            dtype=np.int64,
        )

    assigned = np.arange(rank, total, world_size, dtype=np.int64)
    batches, atom_counts, filtered = get_batches(
        natoms_all[assigned], assigned, max_atoms=max_atoms, min_atoms=0
    )
    debug_log(
        f"[rank {rank}/{world_size}] {name}: {len(assigned)} structs -> "
        f"{len(batches)} batches, mean atoms/batch {np.mean(atom_counts):.0f}, "
        f"filtered {filtered}"
    )

    loader = DataLoader(
        dataset,
        batch_sampler=IndexTrackingBatchSampler(batches),
        collate_fn=partial(data_list_collater, otf_graph=True),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    sum_abs_e_per_atom = 0.0  # sum over structs of |dE| / natoms
    n_systems = 0
    sum_abs_f = 0.0  # sum over all force components of |dF|
    n_force_comp = 0

    for batch in tqdm(loader, desc=f"{name} rank{rank}", total=len(batches)):
        # Forces require grad; predict() enables autograd internally.
        preds = predictor.predict(batch)
        pred_e = preds["energy"].detach().float().cpu().numpy().reshape(-1)
        pred_f = preds["forces"].detach().float().cpu().numpy()
        gt_e = batch["energy"].detach().float().cpu().numpy().reshape(-1)
        gt_f = batch["forces"].detach().float().cpu().numpy()
        natoms = batch["natoms"].detach().cpu().numpy().astype(np.float64)

        sum_abs_e_per_atom += float(np.sum(np.abs(pred_e - gt_e) / natoms))
        n_systems += len(natoms)
        sum_abs_f += float(np.sum(np.abs(pred_f - gt_f)))
        n_force_comp += pred_f.size

    stats_path = _stats_path(output_dir, name, rank)
    tmp = stats_path.with_suffix(".tmp.npz")
    np.savez(
        str(tmp),
        sum_abs_e_per_atom=np.float64(sum_abs_e_per_atom),
        n_systems=np.int64(n_systems),
        sum_abs_f=np.float64(sum_abs_f),
        n_force_comp=np.int64(n_force_comp),
    )
    tmp.rename(stats_path)
    debug_log(
        f"[rank {rank}] {name}: {n_systems} structs, "
        f"energy MAE/atom {sum_abs_e_per_atom / max(n_systems, 1):.6f} eV/atom, "
        f"force MAE {sum_abs_f / max(n_force_comp, 1):.6f} eV/A -> {stats_path}"
    )


def merge_and_report(val_dirs: list[str], output_dir: Path, world_size: int) -> None:
    """Reduce per-rank stats for each dataset and print the final MAE table."""
    rows = []
    for val_dir in val_dirs:
        name = _dataset_name(val_dir)
        shards = sorted(output_dir.glob(f"{name}_evalstats_rank*.npz"))
        if not shards:
            raise FileNotFoundError(f"No stats shards matching {name}_evalstats_rank*.npz")
        se = ns = sf = nf = 0.0
        for sp in shards:
            d = np.load(str(sp))
            se += float(d["sum_abs_e_per_atom"])
            ns += float(d["n_systems"])
            sf += float(d["sum_abs_f"])
            nf += float(d["n_force_comp"])
        rows.append((name, se / ns, sf / nf, int(ns), int(nf // 3)))

    header = (
        f"{'Dataset':<12} | {'Energy MAE (eV/atom)':>20} | "
        f"{'Force MAE (eV/A)':>16} | {'#structs':>8} | {'#atoms':>9}"
    )
    print("\n" + header)
    print("-" * len(header))
    for name, e_mae, f_mae, ns, na in rows:
        print(
            f"{name:<12} | {e_mae:>20.6f} | {f_mae:>16.6f} | {ns:>8d} | {na:>9d}"
        )
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fairchem checkpoint on AseDBDataset val sets (energy + force MAE)."
    )
    parser.add_argument(
        "val_dirs",
        type=str,
        nargs="+",
        help="One or more val directories (each holding *.aselmdb + metadata.npz).",
    )
    parser.add_argument("-o", "--output-dir", type=str, required=True)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-atoms", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--limit", type=int, default=None, help="Cap structures per dataset (for testing)."
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile.")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Reduce existing rank stats and print the MAE table (no inference).",
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merge_and_report(args.val_dirs, output_dir, args.world_size)
    else:
        debug_log(f"[rank {args.rank}/{args.world_size}] Loading model {args.model_path}")
        predictor = create_predictor(
            args.model_path, device=args.device, compile_model=args.compile
        )
        avail_tasks = list(predictor.dataset_to_tasks.keys())
        debug_log(f"[rank {args.rank}] available tasks: {avail_tasks}")

        for val_dir in args.val_dirs:
            # Route each val set to its matching task so the correct energy element
            # references are applied; fall back to the first task if no name match.
            name = _dataset_name(val_dir)
            task_name = name if name in avail_tasks else avail_tasks[0]
            if task_name != name:
                debug_log(
                    f"[rank {args.rank}] {name}: no matching task in {avail_tasks}; "
                    f"falling back to '{task_name}'"
                )
            eval_dataset(
                val_dir=val_dir,
                predictor=predictor,
                task_name=task_name,
                output_dir=output_dir,
                max_atoms=args.max_atoms,
                num_workers=args.num_workers,
                rank=args.rank,
                world_size=args.world_size,
                limit=args.limit,
            )

        # Single-process: no other ranks to wait on, so report immediately.
        if args.world_size == 1:
            merge_and_report(args.val_dirs, output_dir, args.world_size)
