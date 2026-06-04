"""Quick diversity analysis of entropy-selected structures.

Generates a text diversity report and a 4-panel selection history plot.

Usage:
    python analyze_diversity.py /path/to/output_dir
"""

from __future__ import annotations

import argparse
import pickle
from collections import Counter
from pathlib import Path

import lmdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


def plot_selection_history(history_path: Path, output_path: Path) -> None:
    """Create 4-panel selection history plot from selection_history.npz."""
    data = np.load(str(history_path))
    dld = data["delta_log_dets"]
    ld = data["log_dets"]
    steps = np.arange(len(dld))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top left: cumulative log-det
    ax = axes[0, 0]
    ax.plot(np.arange(len(ld)), ld, linewidth=0.8, color="#2563eb")
    peak_idx = np.argmax(ld)
    ax.axvline(peak_idx, color="red", linestyle="--", linewidth=1,
               label=f"Peak at step {peak_idx:,} ({ld[peak_idx]:.2f})")
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Selection step")
    ax.set_ylabel("log det(C)")
    ax.set_title("Cumulative log-determinant")
    ax.legend(fontsize=9)

    # Top right: delta log-det, log y-scale, positive vs negative
    ax = axes[0, 1]
    pos_mask = dld > 0
    neg_mask = dld < 0
    ax.scatter(steps[pos_mask], dld[pos_mask], s=0.05, alpha=0.3,
               color="#2563eb", rasterized=True, label="Positive")
    ax.scatter(steps[neg_mask], -dld[neg_mask], s=0.05, alpha=0.3,
               color="#dc2626", rasterized=True, label="Negative (abs)")
    ax.set_yscale("log")
    ax.set_xlabel("Selection step")
    ax.set_ylabel("|delta log det(C)|")
    ax.set_title("Per-step |delta| (log scale)")
    ax.legend(fontsize=9, markerscale=20)

    # Bottom left: early phase zoom (steps 0-10k)
    ax = axes[1, 0]
    n_zoom = min(10_000, len(dld))
    ax.plot(steps[:n_zoom], dld[:n_zoom], linewidth=0.5, color="#2563eb",
            alpha=0.5, rasterized=True)
    window = 100
    if n_zoom > window:
        smoothed = np.convolve(dld[:n_zoom], np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window // 2, window // 2 + len(smoothed)), smoothed,
                linewidth=1.5, color="#dc2626", label=f"Rolling mean (w={window})")
    ax.set_xlabel("Selection step")
    ax.set_ylabel("delta log det(C)")
    ax.set_title(f"Early phase (steps 0-{n_zoom:,})")
    ax.legend(fontsize=9)

    # Bottom right: zero-crossing region
    ax = axes[1, 1]
    lo = max(0, peak_idx - 40_000)
    hi = min(len(dld), peak_idx + 60_000)
    ax.plot(steps[lo:hi], dld[lo:hi], linewidth=0.3, color="#2563eb",
            alpha=0.4, rasterized=True)
    window = 1000
    if hi - lo > window:
        smoothed = np.convolve(dld[lo:hi], np.ones(window) / window, mode="valid")
        ax.plot(np.arange(lo + window // 2, lo + window // 2 + len(smoothed)),
                smoothed, linewidth=1.5, color="#dc2626",
                label=f"Rolling mean (w={window})")
    ax.axhline(0, color="gray", linestyle="-", linewidth=1)
    ax.axvline(peak_idx, color="green", linestyle="--", linewidth=1,
               label=f"Zero crossing ~{peak_idx:,}")
    ax.set_xlabel("Selection step")
    ax.set_ylabel("delta log det(C)")
    ax.set_title(f"Zero-crossing region (steps {lo:,}-{hi:,})")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def generate_diversity_report(output_dir: Path) -> None:
    """Generate text diversity report from selected structures LMDB."""
    lmdb_path = output_dir / "selected_structures.lmdb"
    meta_path = output_dir / "selected_metadata.parquet"
    report_path = output_dir / "diversity_report.txt"

    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, subdir=False)
    with env.begin() as txn:
        length = pickle.loads(txn.get(b"length"))

    metals, charges, spins, natoms_list = [], [], [], []
    formulas, element_sets, n_unique_elements = [], [], []
    all_element_counter: Counter = Counter()

    with env.begin() as txn:
        for i in tqdm(range(length), desc="Scanning structures"):
            atoms = pickle.loads(txn.get(f"{i}".encode("ascii")))
            info = atoms.info
            metals.append(info.get("metal", "?"))
            charges.append(info.get("charge", 0))
            spins.append(info.get("spin", 0))
            natoms_list.append(len(atoms))
            syms = atoms.get_chemical_symbols()
            sym_set = frozenset(syms)
            element_sets.append(sym_set)
            n_unique_elements.append(len(set(syms)))
            formulas.append(atoms.get_chemical_formula("hill"))
            for e in sym_set:
                all_element_counter[e] += 1
    env.close()

    lines: list[str] = []

    def p(s: str = "") -> None:
        lines.append(s)

    p("=" * 60)
    p(f"DIVERSITY ANALYSIS: {length:,} entropy-selected structures")
    p("=" * 60)

    mc = Counter(metals)
    p("\n--- Metal distribution ---")
    for m, c in mc.most_common():
        p(f"  {m:4s}: {c:>7,} ({100 * c / length:.1f}%)")
    p(f"  Unique metals: {len(mc)}")

    cc = Counter(charges)
    p("\n--- Charge distribution ---")
    for ch, c in sorted(cc.items()):
        p(f"  {ch:+d}: {c:>7,} ({100 * c / length:.1f}%)")

    sc = Counter(spins)
    p("\n--- Spin multiplicity distribution ---")
    for s, c in sorted(sc.items()):
        p(f"  {s:2d}: {c:>7,} ({100 * c / length:.1f}%)")

    na = np.array(natoms_list)
    p("\n--- Structure size (natoms) ---")
    p(f"  min={na.min()}, max={na.max()}, mean={na.mean():.1f}, median={np.median(na):.0f}")
    bins = [0, 10, 20, 30, 50, 75, 100, 150, 200, 500]
    hist, _ = np.histogram(na, bins=bins)
    for i in range(len(hist)):
        p(f"  {bins[i]:>3d}-{bins[i + 1]:>3d}: {hist[i]:>7,} ({100 * hist[i] / length:.1f}%)")

    unique_formulas = len(set(formulas))
    unique_element_combos = len(set(element_sets))
    nue = np.array(n_unique_elements)
    p("\n--- Compositional diversity ---")
    p(f"  Unique formulas: {unique_formulas:,}")
    p(f"  Unique element combinations: {unique_element_combos:,}")
    p(f"  Elements per structure: min={nue.min()}, max={nue.max()}, mean={nue.mean():.1f}")
    p(f"  Total unique elements: {len(all_element_counter)}")
    p("  Top 20 elements:")
    for e, c in all_element_counter.most_common(20):
        p(f"    {e:3s}: {c:>7,} ({100 * c / length:.1f}%)")

    cs_pairs = Counter(zip(charges, spins))
    p("\n--- Charge x Spin combinations ---")
    p(f"  Unique (charge, spin) pairs: {len(cs_pairs)}")
    p("  Top 10:")
    for (ch, sp), c in cs_pairs.most_common(10):
        p(f"    charge={ch:+d}, spin={sp:2d}: {c:>7,} ({100 * c / length:.1f}%)")

    mcs = Counter(zip(metals, charges, spins))
    p("\n--- Metal x Charge x Spin ---")
    p(f"  Unique (metal, charge, spin) triples: {len(mcs)}")

    if meta_path.exists():
        meta = pd.read_parquet(str(meta_path))
        if "is_distorted" in meta.columns:
            dc = meta["is_distorted"].value_counts()
            p("\n--- Distorted vs equilibrium ---")
            for k, v in dc.items():
                label = "distorted" if k else "equilibrium"
                p(f"  {label}: {v:>7,} ({100 * v / length:.1f}%)")

    report = "\n".join(lines)
    report_path.write_text(report)
    print(report)
    print(f"\nSaved report: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze diversity of entropy-selected structures."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory containing selection outputs (selected_structures.lmdb, "
        "selection_history.npz, selected_metadata.parquet).",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    history_path = output_dir / "selection_history.npz"
    if history_path.exists():
        plot_path = output_dir / "selection_history_plot.png"
        plot_selection_history(history_path, plot_path)
    else:
        print(f"No selection_history.npz found in {output_dir}, skipping plot.")

    lmdb_path = output_dir / "selected_structures.lmdb"
    if lmdb_path.exists():
        generate_diversity_report(output_dir)
    else:
        print(f"No selected_structures.lmdb found in {output_dir}, skipping report.")


if __name__ == "__main__":
    main()
