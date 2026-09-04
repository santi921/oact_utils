"""Per-step delta-logdet saturation for in-loop optimized entropy-downselect runs.

Reads ``delta_logdet_traj`` from ``optimization_report.npz`` (shape (n, max_steps+1),
column 0 = original geometry, column k = geometry after k gradient steps, NaN-padded
once a structure stops early) and plots how the acquisition gain accumulates per step.

The batched optimizer steps every structure for the full ``max_steps`` and commits the
best geometry seen, so two curves matter:
  - committed  : running max of the trajectory, i.e. what the run actually banks by step k.
  - instantaneous: the raw trajectory value at step k. It sits below the committed curve
                 wherever gradient ascent overshoots the local maximum.

Usage:
    python -m oact_utilities.scripts.entropy_downselect.plot_opt_saturation \
        --run run2_75k=.../v3_seed_downselect_75k_run2 \
        --out-dir .../v3_seed_downselect_75k_run2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_traj(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (instantaneous gain, committed gain, best step) per optimized structure.

    Args:
        run_dir: Run directory containing ``optimization_report.npz``.

    Returns:
        inst: (n, S) delta_logdet at step k minus the original value.
        gain: (n, S) running max of ``inst``, i.e. the gain committed by step k.
        best_step: (n,) step at which each structure's committed best occurred.
    """
    z = np.load(str(run_dir / "optimization_report.npz"))
    if "delta_logdet_traj" not in z.files:
        raise KeyError(f"{run_dir.name} has no delta_logdet_traj (run predates it)")
    traj = z["delta_logdet_traj"].astype(np.float64)
    keep = z["optimized"] & ~z["fallback"] & np.isfinite(traj[:, 0])
    raw = traj[keep]

    inst = raw - raw[:, [0]]
    # a frozen (clashing) structure records NaN; hold its last valid value
    filled = np.where(np.isfinite(inst), inst, -np.inf)
    gain = np.maximum.accumulate(filled, axis=1)
    return inst, gain, z["n_steps"][keep]


def summarize(name: str, inst: np.ndarray, gain: np.ndarray, best_step: np.ndarray) -> None:
    """Print the step at which the mean committed gain reaches fractions of its final."""
    mean_gain = gain.mean(0)
    final = mean_gain[-1]
    n_max = inst.shape[1] - 1

    print(f"\n[{name}]  n={len(inst):,}  steps run={n_max}")
    print(f"  mean committed gain at final step  : {final:.6f}")
    print(f"  mean/median best step              : {best_step.mean():.2f} / {np.median(best_step):.0f}")
    print(f"  fraction peaking at the last step  : {(best_step >= n_max).mean() * 100:.2f}%")
    print(f"  fraction of steps that lower score : {(np.diff(inst, axis=1) < 0).mean() * 100:.2f}%")
    for frac in (0.5, 0.9, 0.95, 0.99):
        hit = int(np.argmax(mean_gain >= frac * final))
        print(f"  step reaching {frac * 100:4.0f}% of final gain : {hit}")
    for k in (1, 5, 10, 15, 20, 25, 30):
        if k <= n_max:
            print(f"    step {k:2d}: {mean_gain[k]:.6f} "
                  f"({100 * mean_gain[k] / final:5.1f}% of final, "
                  f"{100 * (best_step >= k).mean():5.1f}% still improving)")


def plot(runs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = plt.cm.tab10(np.arange(len(runs)))

    ax = axes[0]
    for (name, inst, gain, _bs), c in zip(runs, colors):
        steps = np.arange(gain.shape[1])
        lo, hi = np.percentile(gain, [25, 75], axis=0)
        ax.plot(steps, gain.mean(0), color=c, lw=2, label=f"{name} mean (committed)")
        ax.plot(steps, np.nanmean(np.where(np.isfinite(inst), inst, np.nan), axis=0),
                color=c, lw=1.4, ls="--", label=f"{name} mean (instantaneous)")
        ax.plot(steps, np.median(gain, 0), color=c, lw=1.2, ls=":", label=f"{name} median")
        ax.fill_between(steps, lo, hi, color=c, alpha=0.15, label=f"{name} IQR")
    ax.set_xlabel("optimization step")
    ax.set_ylabel("delta_logdet gain over original")
    ax.set_title("Acquisition gain vs step")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for (name, inst, _gain, bs), c in zip(runs, colors):
        ax.hist(bs, bins=np.arange(inst.shape[1] + 1) - 0.5, histtype="step",
                lw=1.8, color=c, label=name, density=True)
    ax.set_xlabel("step of the committed best geometry")
    ax.set_ylabel("density")
    ax.set_title("Where each structure peaked")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.suptitle("In-loop optimization: does the delta-logdet gain cap out?", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(str(out_path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", required=True, help="NAME=/path/to/run_dir")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for spec in args.run:
        name, _, path = spec.partition("=")
        inst, gain, best_step = load_traj(Path(path))
        summarize(name, inst, gain, best_step)
        runs.append((name, inst, gain, best_step))

    plot(runs, out_dir / "opt_saturation.png")


if __name__ == "__main__":
    main()
