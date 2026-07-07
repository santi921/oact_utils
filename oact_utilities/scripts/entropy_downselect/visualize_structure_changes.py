"""Visualize how in-loop optimization changed selected structures vs their originals.

For a sample of selected structures, pairs the optimized geometry (from the run's
``optimized_structures.lmdb``) with its original geometry (from the source LMDBs, resolved
via ``selected_metadata.parquet``) and renders the per-atom displacement.

The optimizer only moves atoms in place (no global rotation/translation), so the two
geometries already share a frame and displacements are physical -- no alignment needed.

Outputs (in ``--output-dir``):
  - ``structure_changes_overview.png`` : grid of 2D ball-and-stick molecule panels, each
    auto-oriented by PCA. The ORIGINAL geometry is drawn solid (CPK colors); the OPTIMIZED
    destination is overlaid as a translucent "ghost" at ``--magnify`` x the true
    displacement, with crimson lines connecting each moved atom to where it went.
  - ``structure_changes_summary.png``  : global max-displacement histogram over all
    optimized structures (from ``optimization_report.npz``) plus the pooled per-atom
    displacement distribution of the sampled structures.

Usage:
    python -m oact_utilities.scripts.entropy_downselect.visualize_structure_changes \
        --run-dir   .../v2_seed_downselect_optimized \
        --source-lmdb-dir .../lmdb_inference \
        --n 20 --mode random --magnify 5.0
"""

from __future__ import annotations

import argparse
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
from ase import Atoms
from ase.data import covalent_radii
from ase.data.colors import jmol_colors
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

DEFAULT_RUN_DIR = (
    "/global/homes/i/ishan_a/oact_utils/data/entropy_downselect/"
    "v2_seed_downselect_optimized"
)
DEFAULT_SOURCE_LMDB_DIR = (
    "/pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/lmdb_inference"
)


def _read_atoms(env: lmdb.Environment, key: str) -> Atoms:
    with env.begin() as txn:
        return pickle.loads(txn.get(key.encode("ascii")))


def load_pairs(
    run_dir: Path,
    source_lmdb_dir: Path,
    ranks: list[int],
) -> list[tuple[int, Atoms, Atoms]]:
    """Load (rank, original, optimized) triples for the requested selection ranks."""
    sel = np.load(str(run_dir / "selected_indices.npy"))
    meta = pd.read_parquet(str(run_dir / "selected_metadata.parquet"))
    # global_index -> (source_file stem, local_index)
    resolver = {
        int(g): (s, int(l))
        for g, s, l in zip(
            meta["global_index"], meta["source_file"], meta["local_index"]
        )
    }

    opt_env = lmdb.open(
        str(run_dir / "optimized_structures.lmdb"),
        readonly=True,
        lock=False,
        subdir=False,
    )
    src_envs: dict[str, lmdb.Environment] = {}

    def _src_env(stem: str) -> lmdb.Environment:
        env = src_envs.get(stem)
        if env is None:
            env = lmdb.open(
                str(source_lmdb_dir / stem / "data.lmdb"),
                readonly=True,
                lock=False,
                subdir=False,
            )
            src_envs[stem] = env
        return env

    def _one(rank: int) -> tuple[int, Atoms, Atoms]:
        gidx = int(sel[rank])
        stem, loc = resolver[gidx]
        orig = _read_atoms(_src_env(stem), str(loc))
        opt = _read_atoms(opt_env, str(rank))
        if not np.array_equal(orig.numbers, opt.numbers):
            raise ValueError(f"atom-order mismatch at rank {rank}")
        return rank, orig, opt

    with ThreadPoolExecutor(max_workers=min(16, len(ranks))) as ex:
        out = list(tqdm(ex.map(_one, ranks), total=len(ranks), desc="Loading pairs"))
    opt_env.close()
    for env in src_envs.values():
        env.close()
    return out


def _bonds(atoms: Atoms, scale: float = 1.2) -> list[tuple[int, int]]:
    """Bond pairs by covalent-radius cutoff (isolated molecule, no PBC)."""
    pos = atoms.get_positions()
    r = covalent_radii[atoms.numbers]
    n = len(atoms)
    pairs = []
    for i in range(n):
        d = np.linalg.norm(pos[i + 1 :] - pos[i], axis=1)
        cut = scale * (r[i] + r[i + 1 :])
        for j_off in np.where(d < cut)[0]:
            pairs.append((i, int(i + 1 + j_off)))
    return pairs


def _draw_balls(ax, xy: np.ndarray, depth: np.ndarray, z: np.ndarray,
                radii: np.ndarray, ghost: bool, base_z: float) -> None:
    """Draw depth-sorted CPK circles (painter's algorithm); ghost = translucent."""
    colors = jmol_colors[z]
    for k, idx in enumerate(np.argsort(depth)):
        zo = base_z + k * 0.01
        if ghost:
            ax.add_patch(Circle(xy[idx], radii[idx], facecolor=colors[idx],
                                edgecolor="none", alpha=0.30, zorder=zo))
        else:
            ax.add_patch(Circle(xy[idx], radii[idx], facecolor=colors[idx],
                                edgecolor="k", linewidth=0.5, zorder=zo))
            # specular highlight for a ball look
            ax.add_patch(Circle(xy[idx] + radii[idx] * np.array([-0.3, 0.3]),
                                radii[idx] * 0.3, facecolor="white",
                                edgecolor="none", alpha=0.5, zorder=zo + 0.005))


def _draw_panel(ax, rank: int, orig: Atoms, opt: Atoms, magnify: float) -> None:
    p_o = orig.get_positions()
    p_p = opt.get_positions()
    disp = p_p - p_o
    dnorm = np.linalg.norm(disp, axis=1)
    z = opt.numbers
    radii = covalent_radii[z] * 0.5

    # PCA on the optimized geometry -> 2D viewing plane (auto-orient each molecule).
    centroid = p_p.mean(0)
    c = p_p - centroid
    _u, _s, vt = np.linalg.svd(c, full_matrices=False)
    rot = vt.T
    solid = c @ rot                                    # optimized (new), projected
    ghost = (p_p - magnify * disp - centroid) @ rot    # original (before), magnified back

    bonds = _bonds(opt)
    moved = dnorm > 0.02

    # ghost (original "before") behind: faint bonds + translucent atoms
    for i, j in bonds:
        ax.plot([ghost[i, 0], ghost[j, 0]], [ghost[i, 1], ghost[j, 1]],
                color="0.7", lw=1.0, alpha=0.35, zorder=1, solid_capstyle="round")
    _draw_balls(ax, ghost[:, :2], ghost[:, 2], z, radii, ghost=True, base_z=2)

    # crimson connectors from each moved atom to where it went
    for idx in np.where(moved)[0]:
        ax.plot([solid[idx, 0], ghost[idx, 0]], [solid[idx, 1], ghost[idx, 1]],
                color="crimson", lw=0.9, alpha=0.8, zorder=20)

    # solid (optimized "new") on top: gray sticks + CPK balls
    for i, j in bonds:
        ax.plot([solid[i, 0], solid[j, 0]], [solid[i, 1], solid[j, 1]],
                color="0.45", lw=2.2, zorder=25, solid_capstyle="round")
    _draw_balls(ax, solid[:, :2], solid[:, 2], z, radii, ghost=False, base_z=30)

    # label the metal atom (largest Z)
    mi = int(np.argmax(z))
    ax.text(solid[mi, 0], solid[mi, 1], f" {opt[mi].symbol}",
            color="k", fontsize=8, fontweight="bold", zorder=100)

    rmsd = float(np.sqrt((dnorm**2).mean()))
    ax.set_title(
        f"rank {rank} | {opt.get_chemical_formula()}\n"
        f"max {dnorm.max():.3f} A | rmsd {rmsd:.3f} A | {len(opt)} atoms",
        fontsize=8,
    )
    allpts = np.vstack([solid[:, :2], ghost[:, :2]])
    pad = float(radii.max()) + 0.4
    ax.set_xlim(allpts[:, 0].min() - pad, allpts[:, 0].max() + pad)
    ax.set_ylim(allpts[:, 1].min() - pad, allpts[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_overview(
    pairs: list[tuple[int, Atoms, Atoms]], magnify: float, out_path: Path
) -> None:
    n = len(pairs)
    ncol = 5
    nrow = (n + ncol - 1) // ncol
    fig = plt.figure(figsize=(4.2 * ncol, 4.0 * nrow))
    for k, (rank, orig, opt) in enumerate(
        tqdm(pairs, desc="Rendering panels")
    ):
        ax = fig.add_subplot(nrow, ncol, k + 1)
        _draw_panel(ax, rank, orig, opt, magnify)
    scale = "true scale (1x)" if magnify == 1 else f"displacement magnified {magnify:g}x"
    fig.suptitle(
        f"Solid CPK = optimized (new)   |   translucent ghost = original (before)   |   "
        f"crimson = atom motion   |   {scale}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_summary(
    run_dir: Path, pairs: list[tuple[int, Atoms, Atoms]], out_path: Path
) -> None:
    rep = np.load(str(run_dir / "optimization_report.npz"))
    md_all = rep["max_disp"]
    opt_mask = rep["optimized"] & np.isfinite(md_all)
    md_all = md_all[opt_mask]

    sample_disp = np.concatenate(
        [np.linalg.norm(opt.get_positions() - orig.get_positions(), axis=1)
         for _r, orig, opt in pairs]
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    ax.hist(md_all, bins=80, color="#2563eb", alpha=0.85)
    ax.axvline(md_all.mean(), color="k", linestyle="--", linewidth=1,
               label=f"mean {md_all.mean():.3f} A")
    ax.set_xlabel("max per-atom displacement per structure (A)")
    ax.set_ylabel("count")
    ax.set_title(f"All optimized structures (n={len(md_all):,})")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.hist(sample_disp, bins=40, color="#16a34a", alpha=0.85)
    ax.axvline(sample_disp.mean(), color="k", linestyle="--", linewidth=1,
               label=f"mean {sample_disp.mean():.3f} A")
    ax.set_xlabel("per-atom displacement (A)")
    ax.set_ylabel("count")
    ax.set_title(f"Pooled atoms of the {len(pairs)} sampled structures")
    ax.legend(fontsize=9)

    fig.suptitle("How much did optimization move atoms?", fontsize=13)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def choose_ranks(run_dir: Path, n: int, mode: str, seed: int) -> list[int]:
    n_total = int(np.load(str(run_dir / "selected_indices.npy"), mmap_mode="r").shape[0])
    if mode == "top":
        return list(range(min(n, n_total)))
    if mode == "spread":
        return [int(r) for r in np.linspace(0, n_total - 1, n, dtype=int)]
    if mode == "max-disp":
        rep = np.load(str(run_dir / "optimization_report.npz"))
        md = rep["max_disp"].copy()
        md[~(rep["optimized"] & np.isfinite(md))] = -1.0
        order = np.argsort(-md)[:n]
        # report rows are rank-ordered, so row index == rank
        return sorted(int(r) for r in order)
    if mode == "random":
        rng = np.random.default_rng(seed)
        return sorted(int(r) for r in rng.choice(n_total, size=n, replace=False))
    raise ValueError(f"unknown mode {mode!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    p.add_argument("--source-lmdb-dir", default=DEFAULT_SOURCE_LMDB_DIR)
    p.add_argument("--output-dir", default=None, help="Default: <run-dir>.")
    p.add_argument("--n", type=int, default=20, help="Number of structures.")
    p.add_argument(
        "--mode",
        choices=["random", "top", "spread", "max-disp"],
        default="random",
        help="How to pick structures (default: random).",
    )
    p.add_argument("--seed", type=int, default=0, help="Seed for --mode random.")
    p.add_argument(
        "--magnify",
        type=float,
        default=1.0,
        help="Ghost displacement magnification (default: 1x = true scale).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    source_lmdb_dir = Path(args.source_lmdb_dir)
    out_dir = Path(args.output_dir) if args.output_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ranks = choose_ranks(run_dir, args.n, args.mode, args.seed)
    print(f"Mode={args.mode}, {len(ranks)} ranks: {ranks[:10]}{'...' if len(ranks) > 10 else ''}")

    pairs = load_pairs(run_dir, source_lmdb_dir, ranks)

    plot_overview(pairs, args.magnify, out_dir / "structure_changes_overview.png")
    plot_summary(run_dir, pairs, out_dir / "structure_changes_summary.png")


if __name__ == "__main__":
    main()
