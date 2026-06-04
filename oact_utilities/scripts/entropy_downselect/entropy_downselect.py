"""Greedy entropy-maximizing subset selection (D-optimal experimental design).

Selects structures that maximize log det(covariance) using batch greedy
with filtered candidate pools and Sherman-Morrison inverse updates.

The covariance/greedy machinery lives in ``oact_utilities.utils.entropy_selection``
and is shared with the position-optimizing variant.

Usage:
    python -m oact_utilities.scripts.entropy_downselect.entropy_downselect \
        --features-dir /path/to/candidate_features/ \
        --seed-features /path/to/seed_features.npy \
        --output-dir /path/to/output/ \
        --n-select 250000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from oact_utilities.utils.entropy_selection import (
    build_selected_lmdb,
    debug_log,
    load_features,
    run_selection,
    whiten,
)


def save_outputs(
    output_dir: Path,
    results: dict,
    features_dir: str,
    file_boundaries: list[tuple[str, int, int]],
    lmdb_dir: str | None = None,
) -> None:
    """Write final output files: indices, metadata, history, and optionally LMDB."""
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

    checkpoint_path = output_dir / "checkpoint.npz"
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        debug_log("Removed checkpoint (selection complete)")

    if lmdb_dir is not None:
        debug_log("Building selected structures LMDB...")
        lmdb_path = output_dir / "selected_structures.lmdb"
        build_selected_lmdb(
            selected_indices=selected,
            file_boundaries=file_boundaries,
            lmdb_dir=lmdb_dir,
            output_path=lmdb_path,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Greedy entropy-maximizing subset selection (D-optimal design).",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        required=True,
        help="Directory containing *_features.npy and *_metadata.parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for output files.",
    )
    parser.add_argument(
        "--n-select",
        type=int,
        default=250_000,
        help="Number of structures to select (default: 250000).",
    )
    parser.add_argument(
        "--seed-features",
        type=str,
        required=True,
        help="Path to seed features .npy file (e.g. from DFT-computed structures). "
        "Initial covariance is computed from this dataset.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=1e-6,
        help="Covariance regularization (default: 1e-6).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Selections per full scan batch (default: 1000).",
    )
    parser.add_argument(
        "--pool-factor",
        type=int,
        default=5,
        help="Pool size = batch_size * pool_factor (default: 5).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10_000,
        help="Checkpoint every N iterations (default: 10000).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only first N structures (for testing).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint.",
    )
    parser.add_argument(
        "--lmdb-dir",
        type=str,
        default=None,
        help="Directory containing source <stem>/data.lmdb files. "
        "If provided, writes selected structures to an LMDB in rank order.",
    )
    return parser.parse_args()


if __name__ == "__main__":
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

    debug_log(f"Starting selection: {args.n_select:,} from {X.shape[0]:,}")
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
    )

    debug_log("Saving outputs...")
    save_outputs(
        output_dir, results, args.features_dir, file_boundaries, lmdb_dir=args.lmdb_dir
    )

    elapsed = time.time() - t_start
    n_sel = len(results["selected_indices"])
    debug_log(
        f"Done. Selected {n_sel:,} structures in {elapsed:.1f}s. "
        f"Final log-det: {results['log_dets'][-1]:.4f}"
    )
