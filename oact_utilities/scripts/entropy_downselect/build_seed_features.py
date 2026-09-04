"""Concatenate per-chunk seed *_features.npy files into a single combined seed array.

Reuses load_features() (already used to load the entropy-downselect candidate pool) unmodified;
seed feature row order has no downstream meaning, so simple filename-sorted concatenation is
sufficient.

Usage:
    python -m oact_utilities.scripts.entropy_downselect.build_seed_features \
        /path/to/seed_features_v3 \
        -o /path/to/seed_features_v3_combined.npy
"""

from __future__ import annotations

import argparse

import numpy as np

from oact_utilities.utils.entropy_selection import debug_log, load_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine per-chunk seed feature NPY files into one array."
    )
    parser.add_argument(
        "seed_chunks_dir",
        type=str,
        help="Directory containing per-chunk *_features.npy files.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Path to write the combined seed features NPY.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    X, file_boundaries = load_features(args.seed_chunks_dir)
    debug_log(f"Combined {len(file_boundaries)} chunk files into seed features: {X.shape}")
    np.save(args.output, X.astype(np.float32))
    debug_log(f"Written: {args.output}")
