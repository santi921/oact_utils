"""Talk figures for the omol-vs-wB97M-V/X2C benchmark, read from ``bench_pairs.parquet``.

Uses the primary ladders only (one per complex; see ``build_bench_pairs.py``). Those are Maria's
runs except crown_cryptands, which only Santi's runs cover and which is drawn as its own tier.
The basis-sensitivity panel uses Maria's runs only.

Figures (PNG) and the tables behind them (CSV) land in ``--outdir``:
    gap_parity          omol vs X2C spin gaps, by omol basis tier
    gap_error_family    signed gap error per family
    an_nn_parity        actinide-to-first-shell distances, one panel per donor element
    an_nn_error_donor   distance error per donor element
    svp_vs_tzvpd        gap error at def2-SVP vs def2-TZVPD on the complexes run at both
    omol_svp_vs_tzvpd   omol def2-SVP vs def2-TZVPD directly: geometry RMSD and gap shift
    spin_ordering.csv   ground-state agreement and spin-order rank correlation per family

Gaps are referenced to the X2C ground state of each ladder (both methods share that zero), as in
``openac_bench.score_spin_gaps``. Distances use the heavy atoms within ``--shell-cutoff`` of the
actinide in the X2C geometry, measured at the same atom indices in the omol geometry (atom order
is identical within a pair). Ladders whose omol and X2C geometries sit in different minima
(heavy-atom RMSD > ``--rmsd-flag``) are drawn hollow in the gap plots and excluded from the
distance plots.

Usage:
    python plot_bench_pairs.py --outdir figs_bench
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from build_bench_pairs import geom_rmsd, load_maria  # noqa: E402
from openac_bench import EH_TO_KCAL, parse_xyz, shell_indices  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "data" / "openac_nersc"
# Reference categorical slots 1-3 (light mode, all-pairs safe); tier also carries a marker shape.
TIER_STYLE = {
    "tzvpd_maria": {"color": "#2a78d6", "marker": "o", "label": "omol def2-TZVPD"},
    "svp_maria": {"color": "#eb6834", "marker": "^", "label": "omol def2-SVP"},
    "tzvpd_santi": {
        "color": "#1baf7a",
        "marker": "s",
        "label": "omol def2-TZVPD, Santi runs",
    },
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style() -> None:
    """Recessive axes and grid, text in ink colors."""
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "legend.frameon": False,
        }
    )


def gap_table(df: pd.DataFrame, rmsd_flag: float) -> pd.DataFrame:
    """One row per (ladder, non-ground spin): omol and X2C gaps and their difference (kcal/mol)."""
    w = df.pivot_table(
        index=["pair_id", "complex_id", "family", "tier", "mult"],
        columns="side",
        values="E_eh",
    ).reset_index()
    rms = df[df.side == "omol"].groupby("pair_id").rmsd_omol_x2c.max()
    rows = []
    for pid, g in w.groupby("pair_id"):
        if len(g) < 2:
            continue
        gs = g.loc[g.x2c.idxmin()]
        for _, r in g[g.mult != gs.mult].iterrows():
            rows.append(
                {
                    "pair_id": pid,
                    "complex_id": r.complex_id,
                    "family": r.family,
                    "tier": r.tier,
                    "mult": int(r.mult),
                    "gs_mult_x2c": int(gs.mult),
                    "gap_x2c_kcal": (r.x2c - gs.x2c) * EH_TO_KCAL,
                    "gap_omol_kcal": (r.omol - gs.omol) * EH_TO_KCAL,
                    "max_rmsd_ladder": rms[pid],
                }
            )
    t = pd.DataFrame(rows)
    t["err_kcal"] = t.gap_omol_kcal - t.gap_x2c_kcal
    t["diff_minimum"] = t.max_rmsd_ladder > rmsd_flag
    return t


def ordering_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per ladder: ground-state agreement and Kendall tau of the spin energy order."""
    w = df.pivot_table(
        index=["pair_id", "family", "tier", "mult"], columns="side", values="E_eh"
    ).reset_index()
    rows = []
    for (pid, fam, tier), g in w.groupby(["pair_id", "family", "tier"]):
        if len(g) < 2:
            continue
        tau = kendalltau(g.omol, g.x2c).statistic if len(g) >= 3 else np.nan
        rows.append(
            {
                "pair_id": pid,
                "family": fam,
                "tier": tier,
                "n_spins": len(g),
                "gs_agree": g.omol.idxmin() == g.x2c.idxmin(),
                "order_identical": list(g.sort_values("omol").mult)
                == list(g.sort_values("x2c").mult),
                "kendall_tau": tau,
            }
        )
    return pd.DataFrame(rows)


def distance_table(df: pd.DataFrame, cutoff: float, rmsd_flag: float) -> pd.DataFrame:
    """Actinide-to-first-shell distances (A) in matching omol / X2C geometries."""
    w = df.pivot_table(
        index=["pair_id", "complex_id", "family", "tier", "mult"],
        columns="side",
        values="final_xyz",
        aggfunc="first",
    ).reset_index()
    meta = df[df.side == "omol"].set_index(["pair_id", "mult"])
    rows = []
    for _, r in w.iterrows():
        m = meta.loc[(r.pair_id, r.mult)]
        if m.rmsd_omol_x2c > rmsd_flag:
            continue
        sx, xx = parse_xyz(r.x2c)
        so, xo = parse_xyz(r.omol)
        an, idx = shell_indices(sx, xx, cutoff)
        fallback = set(filter(None, str(m.default_basis_elements).split(",")))
        for i in idx:
            rows.append(
                {
                    "pair_id": r.pair_id,
                    "complex_id": r.complex_id,
                    "family": r.family,
                    "tier": r.tier,
                    "mult": int(r.mult),
                    "actinide": sx[an],
                    "donor": sx[i],
                    "d_x2c_A": float(np.linalg.norm(xx[i] - xx[an])),
                    "d_omol_A": float(np.linalg.norm(xo[i] - xo[an])),
                    "donor_basis_fallback": r.tier == "tzvpd_maria"
                    and sx[i] in fallback,
                }
            )
    t = pd.DataFrame(rows)
    t["err_A"] = t.d_omol_A - t.d_x2c_A
    return t


def stats(err: pd.Series) -> str:
    """Compact MAE / median-abs label."""
    a = err.abs()
    return f"MAE {a.mean():.1f}, median {a.median():.1f} kcal/mol, n={len(a)}"


def plot_gap_parity(g: pd.DataFrame, out: Path) -> None:
    """omol gap vs X2C gap, one marker style per tier, hollow for different-minimum ladders."""
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    lim = [min(g.gap_x2c_kcal.min(), g.gap_omol_kcal.min()) - 5]
    lim.append(max(g.gap_x2c_kcal.max(), g.gap_omol_kcal.max()) + 5)
    ax.plot(lim, lim, color=INK2, lw=1, zorder=1)
    for tier, s in TIER_STYLE.items():
        t = g[g.tier == tier]
        for hollow, sub in t.groupby("diff_minimum"):
            ax.scatter(
                sub.gap_x2c_kcal,
                sub.gap_omol_kcal,
                s=36,
                marker=s["marker"],
                facecolor="none" if hollow else s["color"],
                edgecolor=s["color"] if hollow else "white",
                linewidth=1.2 if hollow else 0.8,
                zorder=3,
                label=f"{s['label']} ({stats(t.err_kcal)})" if not hollow else None,
            )
    ax.scatter(
        [],
        [],
        s=36,
        marker="o",
        facecolor="none",
        edgecolor=INK2,
        label="omol and X2C in different minima",
    )
    for _, r in g.loc[g.err_kcal.abs().nlargest(3).index].iterrows():
        ax.annotate(
            f"{r.complex_id.split('/')[1]} m{r.mult}",
            (r.gap_x2c_kcal, r.gap_omol_kcal),
            xytext=(6, -10),
            textcoords="offset points",
            fontsize=7,
            color=INK2,
        )
    ax.set(
        xlim=lim,
        ylim=lim,
        aspect="equal",
        xlabel="wB97M-V/X2C spin gap (kcal/mol)",
        ylabel="omol spin gap (kcal/mol)",
        title="Spin gaps relative to the X2C ground state",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=7.5)
    fig.savefig(out)
    plt.close(fig)


def plot_gap_error_family(g: pd.DataFrame, out: Path) -> None:
    """Signed gap error per family as a jittered dot plot with a median tick."""
    fams = g.groupby("family").err_kcal.apply(lambda e: e.abs().median()).sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(fams) + 1.2))
    rng = np.random.default_rng(0)
    for y, fam in enumerate(fams.index):
        f = g[g.family == fam]
        for tier, s in TIER_STYLE.items():
            t = f[f.tier == tier]
            ax.scatter(
                t.err_kcal,
                y + rng.uniform(-0.18, 0.18, len(t)),
                s=26,
                marker=s["marker"],
                color=s["color"],
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
                label=s["label"] if y == 0 else None,
            )
        ax.plot([f.err_kcal.median()] * 2, [y - 0.3, y + 0.3], color=INK, lw=2)
    for tier, s in TIER_STYLE.items():
        if tier not in ax.get_legend_handles_labels()[1]:
            ax.scatter([], [], marker=s["marker"], color=s["color"], label=s["label"])
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(
        range(len(fams)), [f"{f} (n={(g.family == f).sum()})" for f in fams.index]
    )
    ax.grid(axis="y", visible=False)
    ax.set(
        xlabel="omol - X2C spin gap (kcal/mol); bar = median",
        title="Spin-gap error by family",
    )
    handles, labels = ax.get_legend_handles_labels()
    keep = dict(zip(labels, handles))
    ax.legend(keep.values(), keep.keys(), loc="lower right", fontsize=8)
    fig.savefig(out)
    plt.close(fig)


def plot_nn_parity(d: pd.DataFrame, out: Path) -> None:
    """Actinide-donor distances, one panel per donor element with at least 10 contacts."""
    counts = d.donor.value_counts()
    donors = counts[counts >= 10].index.tolist()
    ncol = 4
    nrow = int(np.ceil(len(donors) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.6 * nrow))
    for ax, el in zip(axes.flat, donors):
        e = d[d.donor == el]
        lo, hi = (
            e[["d_x2c_A", "d_omol_A"]].min().min() - 0.05,
            e[["d_x2c_A", "d_omol_A"]].max().max() + 0.05,
        )
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=1)
        for tier, s in TIER_STYLE.items():
            for fb, sub in e[e.tier == tier].groupby("donor_basis_fallback"):
                ax.scatter(
                    sub.d_x2c_A,
                    sub.d_omol_A,
                    s=14,
                    marker=s["marker"],
                    facecolor="none" if fb else s["color"],
                    edgecolor=s["color"] if fb else "white",
                    linewidth=0.8 if fb else 0.4,
                )
        ax.set(xlim=(lo, hi), ylim=(lo, hi), aspect="equal")
        ax.set_title(
            f"An-{el}: MAE {e.err_A.abs().mean():.3f} A, n={len(e)}", fontsize=8.5
        )
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(donors) :]:
        ax.axis("off")
    handles = [
        plt.Line2D(
            [], [], ls="", marker=s["marker"], color=s["color"], label=s["label"]
        )
        for s in TIER_STYLE.values()
    ]
    handles.append(
        plt.Line2D(
            [],
            [],
            ls="",
            marker="o",
            mfc="none",
            mec=INK2,
            label="donor on def2-SVP fallback in a def2-TZVPD job",
        )
    )
    fig.supxlabel("wB97M-V/X2C distance (A)")
    fig.supylabel("omol distance (A)")
    fig.suptitle("Actinide to first-shell distances", y=1.0)
    fig.tight_layout()
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        fontsize=8,
    )
    fig.savefig(out)
    plt.close(fig)


def plot_nn_error(d: pd.DataFrame, out: Path) -> None:
    """Distance error per donor element, box over all contacts."""
    order = d.groupby("donor").err_A.median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(order) + 1.2))
    data = [d[d.donor == el].err_A.values for el in order]
    ax.boxplot(
        data,
        vert=False,
        widths=0.5,
        showfliers=True,
        medianprops={"color": INK, "lw": 2},
        boxprops={"color": INK2},
        whiskerprops={"color": INK2},
        capprops={"color": INK2},
        flierprops={"marker": "o", "markersize": 3, "markeredgecolor": INK2},
    )
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(
        range(1, len(order) + 1),
        [f"An-{el} (n={(d.donor == el).sum()})" for el in order],
    )
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="omol - X2C distance (A)", title="Actinide-donor distance error")
    fig.savefig(out)
    plt.close(fig)


def plot_svp_vs_tzvpd(g_all: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Gap error at def2-SVP vs def2-TZVPD, same complex and spin, each against its own X2C."""
    w = g_all.pivot_table(
        index=["complex_id", "mult"], columns="tier", values="err_kcal"
    ).dropna(subset=["svp_maria", "tzvpd_maria"])
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    lim = [w.min().min() - 2, w.max().max() + 2]
    ax.plot(lim, lim, color=INK2, lw=1)
    ax.scatter(w.tzvpd_maria, w.svp_maria, s=32, color="#2a78d6", edgecolor="white")
    diff = (w.svp_maria - w.tzvpd_maria).abs()
    ax.set(
        xlim=lim,
        ylim=lim,
        aspect="equal",
        xlabel="gap error, omol def2-TZVPD (kcal/mol)",
        ylabel="gap error, omol def2-SVP (kcal/mol)",
        title=f"Basis sensitivity: {w.index.get_level_values(0).nunique()} complexes, "
        f"{len(w)} gaps\nmedian |SVP - TZVPD| {diff.median():.1f} kcal/mol",
    )
    fig.savefig(out)
    plt.close(fig)
    return w.reset_index()


def omol_basis_table(maria_db: Path) -> pd.DataFrame:
    """Maria's omol runs at both def2-SVP and def2-TZVPD: per spin, geometry RMSD and gap shift.

    Read straight from the DB rather than ``bench_pairs`` so spins without an X2C partner count.
    Gaps are referenced to the def2-TZVPD ground state of each complex.
    """
    m = load_maria(maria_db)
    w = m[m.lot == "omol"].pivot_table(
        index=["complex_id", "category", "mult"],
        columns="group_dir",
        values=["final_xyz", "final_energy_eh"],
        aggfunc="first",
    )
    w = w.dropna().reset_index()
    rows = []
    for (cx, fam), g in w.groupby(["complex_id", "category"]):
        e_tz, e_svp = (
            g[("final_energy_eh", "OMOL_TZVPD")],
            g[("final_energy_eh", "OMOL_SVP")],
        )
        gs = e_tz.idxmin()
        for i, r in g.iterrows():
            rows.append(
                {
                    "complex_id": cx,
                    "family": fam,
                    "mult": int(r[("mult", "")]),
                    "gs_mult_tzvpd": int(g.at[gs, ("mult", "")]),
                    "rmsd_svp_tzvpd_A": geom_rmsd(
                        r[("final_xyz", "OMOL_SVP")], r[("final_xyz", "OMOL_TZVPD")]
                    ),
                    "gap_tzvpd_kcal": (e_tz[i] - e_tz[gs]) * EH_TO_KCAL,
                    "gap_svp_kcal": (e_svp[i] - e_svp[gs]) * EH_TO_KCAL,
                }
            )
    t = pd.DataFrame(rows)
    t["gap_diff_kcal"] = t.gap_svp_kcal - t.gap_tzvpd_kcal
    return t


def plot_omol_basis(t: pd.DataFrame, out: Path) -> None:
    """Left: SVP-vs-TZVPD geometry RMSD per complex. Right: SVP vs TZVPD spin gaps."""
    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(10, 0.32 * t.complex_id.nunique() + 2.4)
    )
    order = t.groupby("complex_id").rmsd_svp_tzvpd_A.max().sort_values().index
    ypos = {c: i for i, c in enumerate(order)}
    a1.scatter(
        t.rmsd_svp_tzvpd_A,
        t.complex_id.map(ypos),
        s=28,
        color="#2a78d6",
        edgecolor="white",
    )
    a1.set_yticks(
        range(len(order)),
        [f"{c.split('/')[1]} ({c.split('/')[0]})" for c in order],
        fontsize=7.5,
    )
    a1.grid(axis="y", visible=False)
    a1.set(
        xlabel="heavy-atom RMSD, def2-SVP vs def2-TZVPD (A)",
        title=f"Geometry: median {t.rmsd_svp_tzvpd_A.median():.3f} A,\n"
        f"max {t.rmsd_svp_tzvpd_A.max():.3f} A ({len(t)} spin states)",
    )
    gp = t[t.mult != t.gs_mult_tzvpd]
    lim = [min(gp.gap_tzvpd_kcal.min(), gp.gap_svp_kcal.min()) - 3]
    lim.append(max(gp.gap_tzvpd_kcal.max(), gp.gap_svp_kcal.max()) + 3)
    a2.plot(lim, lim, color=INK2, lw=1)
    a2.scatter(
        gp.gap_tzvpd_kcal, gp.gap_svp_kcal, s=30, color="#2a78d6", edgecolor="white"
    )
    ad = gp.gap_diff_kcal.abs()
    a2.set(
        xlim=lim,
        ylim=lim,
        aspect="equal",
        xlabel="omol def2-TZVPD spin gap (kcal/mol)",
        ylabel="omol def2-SVP spin gap (kcal/mol)",
        title=f"Spin gaps: median |SVP - TZVPD| {ad.median():.2f} kcal/mol,\n"
        f"max {ad.max():.2f} kcal/mol ({len(gp)} gaps)",
    )
    fig.suptitle(
        f"omol basis sensitivity, wB97M-V: {t.complex_id.nunique()} complexes run at both bases",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs", type=Path, default=DATA / "bench_pairs.parquet")
    ap.add_argument("--maria-db", type=Path, default=DATA / "openac_dft.db")
    ap.add_argument("--outdir", type=Path, default=Path("figs_bench"))
    ap.add_argument("--shell-cutoff", type=float, default=3.4, help="A")
    ap.add_argument(
        "--rmsd-flag", type=float, default=0.5, help="A, different-minimum flag"
    )
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    style()

    allp = pd.read_parquet(args.pairs)
    prim = allp[allp.primary]
    allp = allp[allp.tier != "tzvpd_santi"]

    gaps = gap_table(prim, args.rmsd_flag)
    gaps.to_csv(args.outdir / "gaps.csv", index=False)
    plot_gap_parity(gaps, args.outdir / "gap_parity.png")
    plot_gap_error_family(gaps, args.outdir / "gap_error_family.png")

    order = ordering_table(prim)
    fam = order.groupby("family").agg(
        ladders=("pair_id", "size"),
        gs_agree_pct=("gs_agree", lambda x: 100 * x.mean()),
        ladders_3plus=("n_spins", lambda x: int((x >= 3).sum())),
        order_identical_pct=("order_identical", lambda x: 100 * x.mean()),
        median_kendall_tau=("kendall_tau", "median"),
    )
    fam.loc["all"] = [
        len(order),
        100 * order.gs_agree.mean(),
        int((order.n_spins >= 3).sum()),
        100 * order.order_identical.mean(),
        order.kendall_tau.median(),
    ]
    fam.round(2).to_csv(args.outdir / "spin_ordering.csv")

    dist = distance_table(prim, args.shell_cutoff, args.rmsd_flag)
    dist.to_csv(args.outdir / "an_nn_distances.csv", index=False)
    plot_nn_parity(dist, args.outdir / "an_nn_parity.png")
    plot_nn_error(dist, args.outdir / "an_nn_error_donor.png")

    both = gap_table(allp, args.rmsd_flag)
    sv = plot_svp_vs_tzvpd(both, args.outdir / "svp_vs_tzvpd.png")
    sv.to_csv(args.outdir / "svp_vs_tzvpd.csv", index=False)

    ob = omol_basis_table(args.maria_db)
    ob.to_csv(args.outdir / "omol_svp_vs_tzvpd.csv", index=False)
    plot_omol_basis(ob, args.outdir / "omol_svp_vs_tzvpd.png")

    print("gap stats by tier (kcal/mol):")
    print(
        gaps.groupby("tier")
        .err_kcal.agg(
            n="size",
            mae=lambda e: e.abs().mean(),
            median_abs=lambda e: e.abs().median(),
            p90_abs=lambda e: e.abs().quantile(0.9),
            max_abs=lambda e: e.abs().max(),
        )
        .round(2)
        .to_string()
    )
    print(f"gaps in different-minimum ladders: {int(gaps.diff_minimum.sum())}")
    print("\nspin ordering:\n" + fam.round(2).to_string())
    print("\nAn-donor distance error by donor (A):")
    print(
        dist.groupby("donor")
        .err_A.agg(n="size", mae=lambda e: e.abs().mean(), mean_signed="mean")
        .round(3)
        .sort_values("n", ascending=False)
        .to_string()
    )
    print(f"\nwrote figures and CSVs to {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
