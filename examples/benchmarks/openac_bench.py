"""Shared helpers for benchmarking relaxations against the NERSC OpenAc DFT optimizations.

The DFT reference is ``data/openac_nersc/openac_dft.db`` (table ``dft_opt``, built by
``oact_utilities/scripts/extract_openac_dft.py``). Each row is one converged ORCA optimization
at one spin multiplicity, so the reference supplies both energies (spin gaps) and a relaxed
geometry for every (complex, spin), in the same atom order as the starting structures.

Method results are passed around as a long table ``runs`` with one row per
(method, formula, spin) and columns ``method``, ``formula``, ``spin``, ``E_eV`` (relaxed
energy) and ``opt_struct`` (relaxed geometry as an XYZ or mol2 string).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from ase import units

EV_TO_KCAL = 1.0 / (units.kcal / units.mol)  # 23.0605 kcal/mol per eV
EH_TO_KCAL = units.Hartree * EV_TO_KCAL  # 627.5095 kcal/mol per Eh
ACTINIDES = {"Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es"}


def parse_xyz(s: str) -> tuple[list[str], np.ndarray]:
    """Symbols and coordinates (A) of a single-frame XYZ string."""
    lines = s.strip("\n").splitlines()
    n = int(lines[0].split()[0])
    rows = [ln.split() for ln in lines[2 : 2 + n]]
    return [r[0] for r in rows], np.array([[float(v) for v in r[1:4]] for r in rows])


def parse_mol2(s: str) -> tuple[list[str], np.ndarray]:
    """Symbols and coordinates (A) from the ATOM block of a mol2 string."""
    lines = s.splitlines()
    start = lines.index("@<TRIPOS>ATOM") + 1
    syms, xyz = [], []
    for ln in lines[start:]:
        if ln.startswith("@<TRIPOS>"):
            break
        p = ln.split()
        if len(p) >= 6:
            syms.append(p[5].split(".")[0])
            xyz.append([float(v) for v in p[2:5]])
    return syms, np.array(xyz)


def parse_struct(s: str) -> tuple[list[str], np.ndarray]:
    """Parse a geometry string that is either mol2 or XYZ."""
    return parse_mol2(s) if "@<TRIPOS>ATOM" in s else parse_xyz(s)


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """RMSD (A) after optimal superposition; rows of ``P`` and ``Q`` are the same atoms."""
    P, Q = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return float(np.sqrt(((P @ R.T - Q) ** 2).sum(1).mean()))


def shell_indices(
    syms: list[str], xyz: np.ndarray, cutoff: float
) -> tuple[int, np.ndarray]:
    """Actinide index and the heavy atoms within ``cutoff`` A of it (its first shell)."""
    an = next(i for i, s in enumerate(syms) if s in ACTINIDES)
    d = np.linalg.norm(xyz - xyz[an], axis=1)
    idx = np.array(
        [i for i, s in enumerate(syms) if i != an and s != "H" and d[i] < cutoff]
    )
    return an, idx


def load_dft_reference(
    db_path: Path, groups: tuple[str, ...] = ("OMOL_SVP", "OMOL_TZVPD")
) -> pd.DataFrame:
    """Converged DFT optimizations, one row per (formula, spin).

    Each complex takes all of its spins from the first group in ``groups`` that has it, so a
    complex never mixes basis sets across its spin ladder (gaps and geometries stay within one
    level of theory); ``group_dir`` records which one. ``E_dft_eh`` is the final energy from
    ``orca.engrad``. Rows whose name carries an oxidation-state tag are dropped: their formula
    alone does not identify the structure.
    """
    con = sqlite3.connect(db_path)
    have = {r[1] for r in con.execute("PRAGMA table_info(dft_opt)")}
    s2_cols = (
        ", hf_type, s2_final, s2_dev"
        if {"hf_type", "s2_final", "s2_dev"} <= have
        else ""
    )
    ref = pd.read_sql(
        "SELECT rel_path, group_dir, category, formula, spin, charge, natoms, elements, "
        f"basis_map, init_xyz, final_xyz, final_energy_eh AS E_dft_eh, trj_n_frames{s2_cols} "
        f"FROM dft_opt WHERE group_dir IN ({', '.join('?' for _ in groups)}) "
        "AND converged_banner = 1 AND terminated = 1 AND os_tag IS NULL",
        con,
        params=groups,
    )
    con.close()
    rank = {g: i for i, g in enumerate(groups)}
    best = ref.groupby("formula")["group_dir"].agg(lambda s: min(s, key=rank.get))
    ref = ref[ref.group_dir == ref.formula.map(best)].reset_index(drop=True)
    for col in ("hf_type", "s2_final", "s2_dev"):
        if col not in ref:
            ref[col] = None
    ref["s2_label"] = [spin_label(h, d) for h, d in zip(ref.hf_type, ref.s2_dev)]
    assert not ref.duplicated(
        ["formula", "spin"]
    ).any(), "duplicate (formula, spin) in reference"
    ref["basis_map"] = ref["basis_map"].map(json.loads)
    covered = ref.apply(
        lambda r: {k.split(":")[1] for k in r.basis_map if k.startswith("GTO")}, axis=1
    )
    # Elements without a NewGTO line fell back to ORCA's default, which is def2-SVP (+ def2-ECP
    # beyond Kr), checked on Si and I outputs; in a TZVPD job that leaves them a size smaller.
    ref["default_basis_elements"] = [
        ",".join(sorted(set(e.split(";")) - c)) for e, c in zip(ref.elements, covered)
    ]
    return ref


def same_start(
    df: pd.DataFrame, ref: pd.DataFrame, xyz_col: str = "init_xyz", tol: float = 0.5
) -> pd.Series:
    """True where a row's start is the DFT job's molecule: same atoms, same order, start RMSD < tol A.

    Formula labels alone are not unique across sources (NERSC "UC24H8" is a 57-atom complex,
    the benchmark's is 33 atoms), so rows are only scored against DFT when this holds. The
    tolerance is loose on purpose: some NERSC jobs (the TZVPD nitrates) started from an
    already-relaxed geometry 0.06-0.16 A from the crystal, which is the same molecule.
    """
    refi = ref.set_index(["formula", "spin"])["init_xyz"]
    out = []
    for f, s, x in zip(df.formula, df.spin, df[xyz_col]):
        if (f, s) not in refi.index:
            out.append(False)
            continue
        s0, x0 = parse_xyz(x)
        s1, x1 = parse_xyz(refi[(f, s)])
        out.append(s0 == s1 and kabsch_rmsd(x0, x1) < tol)
    return pd.Series(out, index=df.index)


def score_spin_gaps(
    runs: pd.DataFrame, ref: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spin-gap errors of every method against DFT on the (formula, spin) rows all methods share.

    The DFT ground state of a complex is the lowest-energy spin among the scored rows, and each
    method's ladder is re-referenced to that same spin, so method and DFT gaps share a zero.
    Rows at the DFT GS spin are 0 on both axes and carry ``is_gap_row = False``.

    Returns ``(scored, gs)``: one row per (method, formula, spin), and one row per
    (method, formula) with more than one spin holding the predicted vs DFT ground state.
    """
    methods = runs["method"].unique()
    ok = runs.dropna(subset=["E_eV"])
    keys = set(map(tuple, ref[["formula", "spin"]].values))
    for m in methods:
        keys &= set(map(tuple, ok.loc[ok.method == m, ["formula", "spin"]].values))
    keep = pd.DataFrame(sorted(keys), columns=["formula", "spin"])
    sc = ok.merge(keep, on=["formula", "spin"]).merge(
        ref[["formula", "spin", "E_dft_eh"]], on=["formula", "spin"]
    )
    gs_spin = sc.loc[sc.groupby(["method", "formula"])["E_dft_eh"].idxmin()].set_index(
        ["method", "formula"]
    )["spin"]
    sc["dft_gs_spin"] = [gs_spin[(m, f)] for m, f in zip(sc.method, sc.formula)]
    grp = sc.groupby(["method", "formula"])
    sc["dE_dft_kcal"] = (sc.E_dft_eh - grp["E_dft_eh"].transform("min")) * EH_TO_KCAL
    e_at_gs = sc[sc.spin == sc.dft_gs_spin].set_index(["method", "formula"])["E_eV"]
    sc["dE_method_kcal"] = (
        sc.E_eV - [e_at_gs[(m, f)] for m, f in zip(sc.method, sc.formula)]
    ) * EV_TO_KCAL
    sc["gap_err_kcal"] = sc.dE_method_kcal - sc.dE_dft_kcal
    sc["is_gap_row"] = sc.spin != sc.dft_gs_spin
    sc["n_spins"] = grp["spin"].transform("size")

    rows = []
    for (m, f), sub in sc[sc.n_spins > 1].groupby(["method", "formula"]):
        pred = sub.loc[sub.E_eV.idxmin()]
        rows.append(
            {
                "method": m,
                "formula": f,
                "n_spins": len(sub),
                "dft_gs_spin": int(sub.dft_gs_spin.iloc[0]),
                "pred_gs_spin": int(pred.spin),
                "dft_penalty_kcal": float(
                    pred.dE_dft_kcal
                ),  # DFT cost of the predicted GS
            }
        )
    gs = pd.DataFrame(rows)
    gs["match"] = gs.dft_gs_spin == gs.pred_gs_spin
    gs["spin_diff"] = gs.pred_gs_spin - gs.dft_gs_spin
    return sc, gs


def r2_vs_dft(rows: pd.DataFrame) -> float:
    """Coefficient of determination of method gaps against DFT, about y = x (no fitted line)."""
    ref, pred = rows.dE_dft_kcal, rows.dE_method_kcal
    return float(1 - ((pred - ref) ** 2).sum() / ((ref - ref.mean()) ** 2).sum())


def summarize_gaps(
    sc: pd.DataFrame, gs: pd.DataFrame, methods: list[str]
) -> pd.DataFrame:
    """Per-method gap error statistics (kcal/mol) over gap rows, plus GS-spin accuracy."""
    out = []
    for m in methods:
        sub = sc[(sc.method == m) & sc.is_gap_row]
        e = sub.gap_err_kcal
        g = gs[gs.method == m]
        out.append(
            {
                "method": m,
                "gap_rows": len(e),
                "MAE_kcal": e.abs().mean(),
                "RMSE_kcal": float(np.sqrt((e**2).mean())),
                "MaxAE_kcal": e.abs().max(),
                "MSE_kcal": e.mean(),
                "R2": r2_vs_dft(sub),
                "multi_spin_complexes": len(g),
                "GS_acc": g.match.mean(),
            }
        )
    return pd.DataFrame(out).set_index("method")


def geometry_vs_dft(
    runs: pd.DataFrame, ref: pd.DataFrame, shell_cutoff: float = 3.4
) -> pd.DataFrame:
    """Each relaxed structure against the DFT geometry at the same (formula, spin).

    Heavy-atom and all-atom Kabsch RMSD (A), and the signed / absolute mean change of the
    actinide first-shell distances (method - DFT, A), with the shell taken from the DFT
    geometry. ``start_vs_dft_rmsd_heavy`` is the crystal-to-DFT distance, for scale. Rows whose
    atom order differs from the DFT job are reported with ``order_ok = False`` and no metrics.
    """
    refi = ref.set_index(["formula", "spin"])
    rows = []
    for _, r in runs.dropna(subset=["opt_struct"]).iterrows():
        if (r.formula, r.spin) not in refi.index:
            continue
        rr = refi.loc[(r.formula, r.spin)]
        s2, x2 = parse_xyz(rr.final_xyz)
        s0, x0 = parse_xyz(rr.init_xyz)
        s1, x1 = parse_struct(r.opt_struct)
        rec = {
            "method": r.method,
            "formula": r.formula,
            "spin": r.spin,
            "order_ok": s1 == s2,
        }
        if s1 == s2:
            heavy = np.array([s != "H" for s in s2])
            an, idx = shell_indices(s2, x2, shell_cutoff)
            d1 = np.linalg.norm(x1[idx] - x1[an], axis=1)
            d2 = np.linalg.norm(x2[idx] - x2[an], axis=1)
            rec.update(
                rmsd_heavy=kabsch_rmsd(x1[heavy], x2[heavy]),
                rmsd_all=kabsch_rmsd(x1, x2),
                n_shell=len(idx),
                shell_dd_mean=float((d1 - d2).mean()),
                shell_dd_mae=float(np.abs(d1 - d2).mean()),
                start_vs_dft_rmsd_heavy=kabsch_rmsd(x0[heavy], x2[heavy]),
            )
        rows.append(rec)
    return pd.DataFrame(rows)


CHECKPOINTS = {
    "v1": Path("oact_v1_inference_ckpt.pt"),
    "v3": Path("oact_v3_inference.pt"),
    "v4": Path("oact_v4_27_epoch_inference_ckpt.pt"),
    "v4final": Path("v4_main_inference.pt"),
    "v4sf05": Path("v4_0.5_act_spinFilter_inference.pt"),
}


def relax_csv(tag: str) -> Path:
    """Cache file of one model's relaxations of the curated rows with NERSC DFT."""
    return Path(f"openan_opt_{tag}_nersc_dft.csv")


def select_jobs(curated_csv: Path, ref: pd.DataFrame) -> pd.DataFrame:
    """Curated (complex, spin) rows that have a DFT job starting from the same molecule."""
    cur = pd.read_csv(curated_csv).rename(
        columns={"Formula": "formula", "Multiplicity": "spin"}
    )
    jobs = cur.merge(
        ref[["formula", "spin", "group_dir", "category", "natoms", "charge"]],
        on=["formula", "spin"],
    )
    assert (
        jobs.Charge == jobs.charge
    ).all(), "charge differs between curated CSV and DFT job"
    jobs["same_start"] = same_start(jobs, ref)
    return jobs


def to_xyz(atoms) -> str:  # noqa: ANN001 -- ase.Atoms
    """Plain XYZ string of an ASE Atoms object."""
    body = "\n".join(
        f"{a.symbol:<3} {x:16.10f} {y:16.10f} {z:16.10f}"
        for a, (x, y, z) in zip(atoms, atoms.positions)
    )
    return f"{len(atoms)}\n\n{body}\n"


def relax_sweep(
    jobs: pd.DataFrame,
    ckpt: Path,
    out_csv: Path,
    fmax: float = 0.05,
    task_name: str = "omol",
    resume: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """SP + LBFGS relaxation of every job from ``init_xyz`` with one eSEN checkpoint, cached to CSV.

    Uses Architector's ``CalcExecutor`` with charge and unpaired electrons from the curated CSV,
    as in ``esen_opt_v4final.ipynb``. With ``resume`` only rows missing from ``out_csv`` run;
    the cache is rewritten every 10 rows so an interrupted sweep loses little. Failures are
    recorded in ``error`` rather than raised.
    """
    import time

    from architector import CalcExecutor, convert_io_molecule
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit
    from tqdm.auto import tqdm

    done = (
        pd.read_csv(out_csv)
        if resume and out_csv.exists()
        else pd.DataFrame(columns=["formula", "spin"])
    )
    todo = jobs.merge(done[["formula", "spin"]], how="left", indicator=True).query(
        '_merge == "left_only"'
    )
    print(f"[{ckpt.name}] {len(done)} cached, {len(todo)} to relax")
    if todo.empty:
        return done
    calc = FAIRChemCalculator(
        load_predict_unit(path=str(ckpt.resolve()), device="cpu"), task_name=task_name
    )
    records = []
    it = (
        tqdm(todo.iterrows(), total=len(todo), desc=ckpt.stem)
        if progress
        else todo.iterrows()
    )
    for _, row in it:
        mol = convert_io_molecule(row["init_xyz"])
        mol.charge = mol.xtb_charge = int(row["Charge"])
        mol.uhf = mol.xtb_uhf = int(row["spin"]) - 1
        t0 = time.time()
        try:
            sp = CalcExecutor(mol, relax=False, calculator=calc, method="custom")
            opt = CalcExecutor(
                mol, relax=True, calculator=calc, method="custom", fmax=fmax
            )
            rec = {
                "E_init_eV": sp.energy,
                "E_eV": opt.energy,
                "rmsd_crystal": opt.rmsd,
                "opt_struct": to_xyz(opt.mol.ase_atoms),
                "time_s": time.time() - t0,
                "error": "",
            }
        except (
            Exception
        ) as exc:  # noqa: BLE001 -- one bad structure must not end the sweep
            rec = {
                "E_init_eV": np.nan,
                "E_eV": np.nan,
                "rmsd_crystal": np.nan,
                "opt_struct": None,
                "time_s": np.nan,
                "error": f"{type(exc).__name__}: {exc}",
            }
        records.append({"formula": row["formula"], "spin": int(row["spin"]), **rec})
        if len(records) % 10 == 0 or len(records) == len(todo):
            pd.concat([done, pd.DataFrame(records)], ignore_index=True).to_csv(
                out_csv, index=False
            )
    return pd.concat([done, pd.DataFrame(records)], ignore_index=True)


# Spin-contamination cutoffs on |<S^2> - S(S+1)|, matching the training-set filters: 1.1 is the
# default rule for actinide structures (data/v4_model_dev/build_dataset.py), 0.5 the stricter
# rule behind the v4 "actinide spin filter 0.5" checkpoint.
S2_MILD, S2_BAD = 0.5, 1.1
S2_LABELS = ["clean", "restricted", "mild", "contaminated", "unknown"]


def spin_label(hf_type: str | None, s2_dev: float | None) -> str:
    """Contamination class of one DFT calculation.

    ``restricted``: RHF/RKS (or ROHF) reference, so <S^2> is exact by construction, but the
    state could not relax to a broken-symmetry solution; kept apart from ``clean``.
    ``unknown``: no <S^2> extracted (DB built without ``--s2``).
    """
    if hf_type in ("RHF", "ROHF"):
        return "restricted"
    if s2_dev is None or pd.isna(s2_dev):
        return "unknown"
    d = abs(float(s2_dev))
    return "clean" if d < S2_MILD else ("mild" if d < S2_BAD else "contaminated")


def label_gaps(sc: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Add per-state and per-gap contamination labels to a ``score_spin_gaps`` table.

    A gap involves two DFT states, the scored spin and the DFT ground-state spin it is measured
    from; ``gap_s2_label`` is the worse of the two (contaminated > mild > unknown > restricted >
    clean) and ``gap_s2_dev`` the larger |deviation| (restricted counts as 0).
    """
    lab = ref.set_index(["formula", "spin"])[["s2_label", "s2_dev"]]
    out = sc.copy()
    out["s2_label"] = [
        lab.s2_label.get((f, s), "unknown") for f, s in zip(out.formula, out.spin)
    ]
    out["gs_s2_label"] = [
        lab.s2_label.get((f, s), "unknown")
        for f, s in zip(out.formula, out.dft_gs_spin)
    ]
    rank = {"clean": 0, "restricted": 1, "unknown": 2, "mild": 3, "contaminated": 4}
    out["gap_s2_label"] = [
        max(a, b, key=rank.get) for a, b in zip(out.s2_label, out.gs_s2_label)
    ]

    def absdev(f: str, s: int) -> float:
        if lab.s2_label.get((f, s)) == "restricted":
            return 0.0
        d = lab.s2_dev.get((f, s))
        return abs(float(d)) if d is not None and not pd.isna(d) else np.nan

    out["gap_s2_dev"] = [
        (
            np.nanmax([absdev(f, s), absdev(f, g)])
            if not (np.isnan(absdev(f, s)) and np.isnan(absdev(f, g)))
            else np.nan
        )
        for f, s, g in zip(out.formula, out.spin, out.dft_gs_spin)
    ]
    return out


def stratified_gaps(sc: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """Gap error per method and gap contamination label (gap rows only, kcal/mol)."""
    gap = sc[sc.is_gap_row]
    tab = gap.groupby(["gap_s2_label", "method"]).gap_err_kcal.agg(
        rows="size",
        MAE=lambda e: e.abs().mean(),
        MSE="mean",
        MaxAE=lambda e: e.abs().max(),
    )
    order = [x for x in S2_LABELS if x in gap.gap_s2_label.unique()]
    return tab.reindex(
        pd.MultiIndex.from_product([order, methods], names=tab.index.names)
    ).dropna(how="all")
