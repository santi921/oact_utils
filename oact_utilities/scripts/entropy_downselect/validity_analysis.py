"""Geometric validity analysis for in-loop optimized entropy-downselect runs.

Compares one or more runs on the metrics that decide whether the optimized geometries
are usable as DFT training data:

  Report-level (all structures, from optimization_report.npz):
    - min interatomic distance AFTER optimization (clash distribution)
    - max per-atom displacement per structure
    - fallback rate, mean delta-logdet gain

  Structure-level (sampled, loaded from LMDB):
    - bond-length change vs original (relative), using the original covalent bonds
    - bonds broken / formed

Usage:
    python -m oact_utilities.scripts.entropy_downselect.validity_analysis \
        --run aggressive=.../v2_seed_downselect_optimized_aggressive \
        --run baseline=.../v2_seed_downselect_optimized \
        --source-lmdb-dir .../lmdb_inference \
        --out-dir .../v2_seed_downselect_optimized_aggressive \
        --sample 2000
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
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_SOURCE_LMDB_DIR = (
    "/pscratch/sd/i/ishan_a/open_actinides/entropy_downselect/lmdb_inference"
)


def _read_atoms(env: lmdb.Environment, key: str) -> Atoms:
    with env.begin() as txn:
        return pickle.loads(txn.get(key.encode("ascii")))


def _orig_bonds(atoms: Atoms, scale: float = 1.2) -> np.ndarray:
    """Bonded (i, j) pairs by covalent-radius cutoff on the ORIGINAL geometry."""
    pos = atoms.get_positions()
    r = covalent_radii[atoms.numbers]
    n = len(atoms)
    pairs = []
    for i in range(n):
        d = np.linalg.norm(pos[i + 1 :] - pos[i], axis=1)
        cut = scale * (r[i] + r[i + 1 :])
        for j_off in np.where(d < cut)[0]:
            pairs.append((i, int(i + 1 + j_off)))
    return np.array(pairs, dtype=int).reshape(-1, 2)


def bond_change_stats(
    run_dir: Path, source_lmdb_dir: Path, ranks: list[int]
) -> dict[str, np.ndarray]:
    """Per-bond relative length change and per-structure bonds broken/formed."""
    sel = np.load(str(run_dir / "selected_indices.npy"))
    meta = pd.read_parquet(str(run_dir / "selected_metadata.parquet"))
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

    def _one(rank: int):
        gidx = int(sel[rank])
        stem, loc = resolver[gidx]
        orig = _read_atoms(_src_env(stem), str(loc))
        opt = _read_atoms(opt_env, str(rank))
        po, pp = orig.get_positions(), opt.get_positions()
        r = covalent_radii[orig.numbers]

        bonds = _orig_bonds(orig)
        if len(bonds) == 0:
            rel = np.empty(0)
        else:
            i, j = bonds[:, 0], bonds[:, 1]
            d0 = np.linalg.norm(po[i] - po[j], axis=1)
            d1 = np.linalg.norm(pp[i] - pp[j], axis=1)
            rel = np.abs(d1 - d0) / d0

        # bonds broken (was bonded, now beyond cutoff) / formed (reverse)
        n = len(orig)
        broken = formed = 0
        for a in range(n):
            dd0 = np.linalg.norm(po[a + 1 :] - po[a], axis=1)
            dd1 = np.linalg.norm(pp[a + 1 :] - pp[a], axis=1)
            cut = 1.2 * (r[a] + r[a + 1 :])
            was = dd0 < cut
            now = dd1 < cut
            broken += int(np.sum(was & ~now))
            formed += int(np.sum(~was & now))
        return rel, broken, formed

    with ThreadPoolExecutor(max_workers=min(16, len(ranks))) as ex:
        results = list(
            tqdm(ex.map(_one, ranks), total=len(ranks), desc=f"bonds:{run_dir.name}")
        )
    opt_env.close()
    for env in src_envs.values():
        env.close()

    rel_all = np.concatenate([r for r, _, _ in results]) if results else np.empty(0)
    rel_max_per_struct = np.array(
        [float(r.max()) if r.size else 0.0 for r, _, _ in results]
    )
    broken = np.array([b for _, b, _ in results])
    formed = np.array([f for _, _, f in results])
    return {
        "rel_all": rel_all,
        "rel_max_per_struct": rel_max_per_struct,
        "broken": broken,
        "formed": formed,
    }


def summarize_report(name: str, run_dir: Path) -> dict:
    z = np.load(str(run_dir / "optimization_report.npz"))
    opt = z["optimized"]
    fb = z["fallback"]
    mda = z["min_dist_after"]
    mdb = z["min_dist_before"]
    disp = z["max_disp"]
    gain = z["delta_logdet_opt"] - z["delta_logdet_orig"]
    finite = np.isfinite(mda)
    return {
        "name": name,
        "n": int(len(opt)),
        "fallback_rate": float(fb.mean()),
        "mean_max_disp": float(np.nanmean(disp[opt])),
        "min_dist_after": mda[finite],
        "min_dist_before": mdb[np.isfinite(mdb)],
        "max_disp": disp[opt & np.isfinite(disp)],
        "mean_gain": float(np.nanmean(gain[opt])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", required=True, help="NAME=/path/to/run_dir")
    ap.add_argument("--source-lmdb-dir", default=DEFAULT_SOURCE_LMDB_DIR)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sample", type=int, default=2000, help="Structures for bond stats.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(args.source_lmdb_dir)

    runs = []
    for spec in args.run:
        name, _, path = spec.partition("=")
        runs.append((name, Path(path)))

    reps = [summarize_report(name, rd) for name, rd in runs]

    # sampled bond-length analysis
    bond = {}
    rng = np.random.default_rng(args.seed)
    for name, rd in runs:
        n_total = int(np.load(str(rd / "selected_indices.npy"), mmap_mode="r").shape[0])
        k = min(args.sample, n_total)
        ranks = sorted(int(r) for r in rng.choice(n_total, size=k, replace=False))
        bond[name] = bond_change_stats(rd, src, ranks)

    # ---- text report ----
    CLASH_THRESHOLDS = [0.5, 0.7, 0.9, 1.0]
    print("\n================ VALIDITY SUMMARY ================")
    for r in reps:
        mda = r["min_dist_after"]
        print(f"\n[{r['name']}]  n={r['n']:,}")
        print(f"  fallback rate         : {r['fallback_rate']*100:.2f}%")
        print(f"  mean max-disp         : {r['mean_max_disp']:.3f} A")
        print(f"  mean delta-logdet gain: {r['mean_gain']:.5f}")
        print(f"  min(min_dist_after)   : {mda.min():.3f} A")
        print(f"  median min_dist_after : {np.median(mda):.3f} A")
        for t in CLASH_THRESHOLDS:
            frac = float((mda < t).mean()) * 100
            print(f"    structures w/ min_dist < {t:.1f} A : {frac:.2f}%")
        b = bond[r["name"]]
        rel = b["rel_all"]
        print(f"  bond-length |rel change| (sampled {len(b['broken']):,} structs,"
              f" {len(rel):,} bonds):")
        print(f"    median {np.median(rel)*100:.2f}% | p90 {np.percentile(rel,90)*100:.2f}%"
              f" | p99 {np.percentile(rel,99)*100:.2f}% | max {rel.max()*100:.2f}%")
        print(f"    bonds broken/struct: mean {b['broken'].mean():.3f}"
              f" (max {b['broken'].max()}) | formed: mean {b['formed'].mean():.3f}"
              f" (max {b['formed'].max()})")
    print("\n=================================================\n")

    # ---- figure ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.tab10(np.arange(len(reps)))

    ax = axes[0, 0]
    for r, c in zip(reps, colors):
        ax.hist(r["min_dist_after"], bins=120, range=(0, 3.0), histtype="step",
                lw=1.8, color=c, label=r["name"], density=True)
    ax.axvline(0.7, color="k", ls="--", lw=1, alpha=0.6, label="min-dist target 0.7 A")
    ax.set_xlabel("min interatomic distance AFTER opt (A)")
    ax.set_ylabel("density")
    ax.set_title("Clash distribution")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for r, c in zip(reps, colors):
        ax.hist(r["max_disp"], bins=80, histtype="step", lw=1.8, color=c,
                label=r["name"], density=True)
    ax.set_xlabel("max per-atom displacement per structure (A)")
    ax.set_ylabel("density")
    ax.set_title("Displacement distribution")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for (name, b), c in zip(bond.items(), colors):
        ax.hist(b["rel_all"] * 100, bins=120, range=(0, 60), histtype="step",
                lw=1.8, color=c, label=name, density=True)
    ax.set_xlabel("per-bond |length change| (%)")
    ax.set_ylabel("density")
    ax.set_title("Bond-length change (sampled)")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for (name, b), c in zip(bond.items(), colors):
        ax.hist(b["rel_max_per_struct"] * 100, bins=80, range=(0, 80),
                histtype="step", lw=1.8, color=c, label=name, density=True)
    ax.set_xlabel("max per-bond |length change| per structure (%)")
    ax.set_ylabel("density")
    ax.set_title("Worst bond change per structure (sampled)")
    ax.legend(fontsize=8)

    fig.suptitle("Geometric validity: optimized vs baseline", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = out_dir / "validity_analysis.png"
    fig.savefig(str(out), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
