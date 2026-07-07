"""Compare selection_history.npz across entropy-downselect runs.

Produces a two-panel figure: cumulative log-det vs step, and the rolling-mean
per-step delta-log-det. Runs may have different lengths; each is plotted over its
own step range.

Usage:
    python compare_selection_histories.py \
        --run NAME=/path/to/run_dir [--run NAME=/path/to/other_dir ...] \
        --out /path/to/comparison.png [--window 1000]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Sliding-window mean (valid mode); returns array of len(x)-w+1."""
    if w <= 1:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run",
        action="append",
        required=True,
        help="NAME=/path/to/run_dir (repeatable). Dir must hold selection_history.npz.",
    )
    ap.add_argument("--out", required=True, help="Output PNG path.")
    ap.add_argument("--window", type=int, default=1000, help="Rolling-mean window.")
    ap.add_argument(
        "--linthresh",
        type=float,
        default=1e-5,
        help="symlog linear-region half-width for panel 2.",
    )
    args = ap.parse_args()

    runs = []
    for spec in args.run:
        name, _, path = spec.partition("=")
        z = np.load(Path(path) / "selection_history.npz")
        runs.append((name, z["log_dets"], z["delta_log_dets"]))

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    for name, log_dets, deltas in runs:
        steps = np.arange(len(log_dets))
        (line,) = ax_top.plot(steps, log_dets, lw=1.3, label=name)

        # mark the turnaround (peak) only if it is interior (not still climbing).
        peak = int(np.argmax(log_dets))
        if peak < len(log_dets) - 1:
            ax_top.plot(peak, log_dets[peak], "o", color=line.get_color(),
                        ms=7, zorder=5)
            ax_top.annotate(
                f"peak {log_dets[peak]:.1f} @ {peak:,}",
                xy=(peak, log_dets[peak]),
                xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=8, color=line.get_color(),
            )

        w = args.window
        rm = rolling_mean(deltas, w)
        x = np.arange(w - 1, w - 1 + len(rm)) if w > 1 else np.arange(len(rm))
        ax_bot.plot(x, rm, lw=1.0, label=name)

    ax_top.set_ylabel("cumulative log-det(C)")
    ax_top.set_title("Cumulative log-det vs selection step")
    ax_top.legend()
    ax_top.grid(alpha=0.3)

    ax_bot.axhline(0, color="k", lw=0.8, alpha=0.5)
    # symlog: log-scaled on both sides, linear within +/- linthresh so the sign
    # change (delta crossing zero late in selection) is visible.
    ax_bot.set_yscale("symlog", linthresh=args.linthresh)
    ax_bot.set_ylabel(f"delta log-det (rolling mean, w={args.window})")
    ax_bot.set_xlabel("selection step")
    ax_bot.set_title("Per-step marginal gain (smoothed, symlog)")
    ax_bot.legend()
    ax_bot.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")

    for name, log_dets, deltas in runs:
        i = int(np.argmax(log_dets))
        print(
            f"{name}: n={len(deltas)}, peak log-det={log_dets[i]:.2f} @ step {i}, "
            f"final={log_dets[-1]:.2f}"
        )


if __name__ == "__main__":
    main()
