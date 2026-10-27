"""Build the omol-vs-X2C benchmark table from Maria's OpenAc DFT DB and Santi's multi_spin runs.

Every comparison is wB97M-V: omol (def2-TZVPD or def2-SVP, ECP on the actinide) against
wB97M-V/DLU-X2C (all-electron). Each complex contributes one or more candidate ladders, one per
omol source, and each ladder stays inside a single omol basis and a single X2C basis so no spin
gap mixes basis sets:

    tzvpd_maria  Maria OMOL_TZVPD   vs Maria wB97M-V/X2C
    tzvpd_santi  Santi omol runs    vs Maria wB97M-V/X2C
    svp_maria    Maria OMOL_SVP     vs Maria wB97M-V/X2C

The candidate matching the most spin states is ``primary`` (ties go to the order above); the
others are kept with ``primary = False`` for basis-sensitivity panels. Santi ladders are
primary only in the families named by ``--santi-primary-families`` (default crown_cryptands,
which no Maria omol run covers): where they overlap Maria's def2-TZVPD runs, some spins sit
10-25 kcal/mol apart at the same geometry (different SCF solutions), so elsewhere they stay out
of headline numbers.

Complex identity:
    - Maria complexes are (category, formula label, charge). Labels are names, not compositions
      (906 of 4,235 rows do not match their own atoms), so they are never compared across sources.
    - A Santi run matches a Maria complex on the actual atom multiset plus charge, and is kept only
      when its final geometry is within ``--rmsd-tol`` (heavy-atom RMSD, median over shared
      spins) of Maria's X2C geometry (same atom order required). Between two Santi starting structures for one complex, the one
      closer to Maria's geometry wins, else the lower energy.

Ladders whose energies span more than ``--max-spread-kcal`` are dropped as broken SCF solutions
(Maria's AmC27N2H42I TZVPP and AmC72H60N3P6Se6 TZVP X2C ladders hold one spin 2-6 Eh off).

Status for Santi's runs comes from ``status_by_fluxid.csv`` (flux-ID ordered restarts, Sella
graded by sella.log), never from the status column of ``multi_spin_jobs.sqlite3``; energies and
final geometries come from that sqlite's /pscratch rows, with ``--santi-extra-geoms`` filling the
Sella jobs it never extracted.

Output: one row per (pair_id, mult, side) with side in {omol, x2c}.

Usage:
    python build_bench_pairs.py --maria-db "../../data/openac_nersc/openac_dft (1).db" \
        -o ../../data/openac_nersc/bench_pairs.parquet
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from ase.data import chemical_symbols
from openac_bench import EH_TO_KCAL, kabsch_rmsd, parse_xyz

DATA = Path(__file__).resolve().parents[2] / "data" / "openac_nersc"
TIERS = ("tzvpd_maria", "tzvpd_santi", "svp_maria")
X2C_ORDER = ("TZVPP", "TZVP", "SVP")
SANTI_FAMILY = {
    "COT": "COT",
    "ChalcogenEthers": "chalcogenether",
    "Chalcogenides": "chalcogenides",
    "Crown-Cryptands": "crown_cryptands",
    "Dithiocarbamates-dithiophosphates-dithiolates": "dithiolates",
    "Nitrates": "nitrates",
    "carbenes": "carbene",
    "tris-Cp": "cp",
}


def comp_key(symbols: list[str], charge: int) -> tuple:
    """Hashable (atom multiset, charge) identity of a molecule."""
    return (tuple(sorted(Counter(symbols).items())), int(charge))


def to_xyz(symbols: list[str], coords: np.ndarray) -> str:
    """Single-frame XYZ string."""
    body = "\n".join(
        f"{s:<3}{x:16.10f}{y:16.10f}{z:16.10f}" for s, (x, y, z) in zip(symbols, coords)
    )
    return f"{len(symbols)}\n\n{body}\n"


def geom_rmsd(xyz_a: str, xyz_b: str) -> float:
    """Heavy-atom Kabsch RMSD (A) between two XYZ strings, NaN when atom order differs.

    Hydrogens are excluded so free methyl rotations do not read as a different structure.
    """
    sa, xa = parse_xyz(xyz_a)
    sb, xb = parse_xyz(xyz_b)
    if sa != sb:
        return float("nan")
    heavy = np.array([s != "H" for s in sa])
    return kabsch_rmsd(xa[heavy], xb[heavy])


def spread_kcal(energies: pd.Series) -> float:
    """Energy range of a ladder in kcal/mol."""
    return float((energies.max() - energies.min()) * EH_TO_KCAL)


def load_maria(db_path: Path) -> pd.DataFrame:
    """Converged wB97M-V optimizations from Maria's DB, one row per job."""
    con = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT rel_path, category, group_dir, lot, formula, os_tag, charge, mult, natoms, "
        "elements, basis_map, final_xyz, final_energy_eh, s2_final, s2_dev, hf_type "
        "FROM dft_opt WHERE terminated = 1 AND converged_banner = 1 AND functional = 'WB97M-V'",
        con,
    )
    con.close()
    df["complex_id"] = [
        f"{c}/{f}/q{q}" for c, f, q in zip(df.category, df.formula, df.charge)
    ]
    df["ckey"] = [comp_key(e.split(";"), q) for e, q in zip(df.elements, df.charge)]
    covered = [
        {k.split(":")[1] for k in json.loads(b) if k.startswith("GTO")}
        for b in df.basis_map
    ]
    # Elements without a NewGTO line fall back to ORCA's default def2-SVP.
    df["default_basis_elements"] = [
        ",".join(sorted(set(e.split(";")) - c)) for e, c in zip(df.elements, covered)
    ]
    return df


def load_santi(
    status_csv: Path, inv_csv: Path, jobs_db: Path, extra_geoms: Path | None
) -> tuple[pd.DataFrame, list[str]]:
    """Santi's converged omol runs with charge, multiplicity, energy and final geometry.

    Geometry and energy come from the sqlite; jobs it never extracted (Sella runs it graded as
    failed) fall back to ``extra_geoms``, a JSON of ``{rel: {symbols, coords, E_eh}}`` read
    from each job's ``orca.engrad``. Returns the jobs plus the converged ones still lacking a
    geometry.
    """
    st = pd.read_csv(status_csv)
    inv = pd.read_csv(inv_csv)[["rel_path", "inp_charge", "inp_mult"]]
    st = st.merge(inv.rename(columns={"rel_path": "rel"}), on="rel")
    con = sqlite3.connect(jobs_db)
    db = pd.read_sql(
        "SELECT path, final_energy, final_coords, final_elements FROM jobs "
        "WHERE path LIKE '/pscratch%'",
        con,
    )
    con.close()
    db["rel"] = db.path.str.extract(r"/multi_spin/multi_spin/(.*)$")[0]
    df = st.merge(db.drop(columns="path"), on="rel", how="left")
    parts = df.rel.str.split("/")
    df["lot"], df["santi_category"], df["mol"] = (
        parts.str[0],
        parts.str[1],
        parts.str[2],
    )
    df = df[(df.lot == "omol") & df.done].copy()
    if extra_geoms is not None and extra_geoms.exists():
        extra = json.loads(extra_geoms.read_text())
        for i in df.index[df.final_coords.isna() & df.rel.isin(extra)]:
            e = extra[df.at[i, "rel"]]
            df.at[i, "final_coords"] = json.dumps(e["coords"])
            df.at[i, "final_elements"] = json.dumps(e["symbols"])
            df.at[i, "final_energy"] = e["E_eh"]
    missing = df.rel[df.final_coords.isna()].tolist()
    df = df[df.final_coords.notna()].copy()
    syms = [
        [chemical_symbols[e] if isinstance(e, int) else e for e in json.loads(s)]
        for s in df.final_elements
    ]
    df["final_xyz"] = [
        to_xyz(s, np.array(json.loads(c))) for s, c in zip(syms, df.final_coords)
    ]
    df["ckey"] = [comp_key(s, q) for s, q in zip(syms, df.inp_charge)]
    df["family"] = df.santi_category.map(SANTI_FAMILY)
    df = df.rename(
        columns={"inp_charge": "charge", "inp_mult": "mult", "final_energy": "E"}
    )
    return df, missing


def build(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    """Assemble the long benchmark table and a dict of diagnostics."""
    maria = load_maria(args.maria_db)
    santi, no_geom = load_santi(
        args.santi_status, args.santi_inv, args.santi_db, args.santi_extra_geoms
    )
    diag: dict = {
        "broken_ladders": [],
        "santi_no_geom": no_geom,
        "santi_unmatched": [],
        "santi_rejected": [],
    }

    def ladders(df: pd.DataFrame, by: list[str], label: str) -> dict:
        out = {}
        for k, g in df.groupby(by):
            g = g.drop_duplicates("mult")
            if len(g) > 1 and spread_kcal(g.final_energy_eh) > args.max_spread_kcal:
                diag["broken_ladders"].append((label, *k))
                continue
            out[k] = g.set_index("mult")
        return out

    omol_m = ladders(maria[maria.lot == "omol"], ["complex_id", "group_dir"], "omol")
    x2c_m = ladders(maria[maria.lot == "x2c"], ["complex_id", "group_dir"], "x2c")
    x2c_by_cx: dict[str, list[tuple[str, pd.DataFrame]]] = {}
    for (cx, grp), lad in x2c_m.items():
        x2c_by_cx.setdefault(cx, []).append((grp, lad))
    cx_meta = maria.drop_duplicates("complex_id").set_index("complex_id")

    # Santi ladders, one per (category, starting structure), matched onto Maria complexes.
    santi_lad = {}
    for k, g in santi.groupby(["santi_category", "mol"]):
        g = g.drop_duplicates("mult").rename(columns={"E": "final_energy_eh"})
        if len(g) > 1 and spread_kcal(g.final_energy_eh) > args.max_spread_kcal:
            diag["broken_ladders"].append(("santi_omol", *k))
            continue
        santi_lad[k] = g.set_index("mult")
    by_ckey: dict[tuple, list[str]] = {}
    for cx, row in cx_meta.iterrows():
        if cx in x2c_by_cx:
            by_ckey.setdefault(row.ckey, []).append(cx)
    santi_best: dict[str, tuple[float, float, tuple, pd.DataFrame]] = {}
    for k, lad in santi_lad.items():
        cands = by_ckey.get(lad.ckey.iloc[0], [])
        if not cands:
            diag["santi_unmatched"].append(k)
            continue
        for cx in cands:
            rms = [
                geom_rmsd(lad.at[mu, "final_xyz"], x.at[mu, "final_xyz"])
                for _, x in x2c_by_cx[cx]
                for mu in set(lad.index) & set(x.index)
            ]
            rms = [r for r in rms if np.isfinite(r)]
            med = float(np.median(rms)) if rms else float("nan")
            if not (med < args.rmsd_tol):
                diag["santi_rejected"].append((k, cx, med))
                continue
            score = (med, float(lad.final_energy_eh.min()))
            if cx not in santi_best or score < santi_best[cx][:2]:
                santi_best[cx] = (*score, k, lad)

    rows = []
    for cx, x2c_list in x2c_by_cx.items():
        cands = []
        for grp, tier in (("OMOL_TZVPD", "tzvpd_maria"), ("OMOL_SVP", "svp_maria")):
            if (cx, grp) in omol_m:
                cands.append((tier, grp, omol_m[(cx, grp)], None))
        if cx in santi_best:
            _, _, k, lad = santi_best[cx]
            cands.append(("tzvpd_santi", "def2-TZVPD", lad, k))
        scored = []
        for tier, grp, olad, k in cands:
            best = max(
                x2c_list,
                key=lambda gx: (
                    len(set(olad.index) & set(gx[1].index)),
                    -X2C_ORDER.index(gx[0]),
                ),
            )
            shared = sorted(set(olad.index) & set(best[1].index))
            if shared:
                scored.append((tier, grp, olad, k, best, shared))
        if not scored:
            continue
        meta = cx_meta.loc[cx]
        santi_ok = (
            "all" in args.santi_primary_families
            or meta.category in args.santi_primary_families
        )
        eligible = [s for s in scored if santi_ok or s[0] != "tzvpd_santi"]
        primary = (
            max(eligible, key=lambda s: (len(s[5]), -TIERS.index(s[0])))[0]
            if eligible
            else None
        )
        for tier, grp, olad, k, (xgrp, xlad), shared in scored:
            pair_id = f"{cx}|{tier}"
            for mu in shared:
                o, x = olad.loc[mu], xlad.loc[mu]
                rmsd = geom_rmsd(o.final_xyz, x.final_xyz)
                common = {
                    "pair_id": pair_id,
                    "complex_id": cx,
                    "family": meta.category,
                    "formula_label": meta.formula,
                    "os_tag": meta.os_tag,
                    "charge": int(meta.charge),
                    "mult": int(mu),
                    "natoms": int(meta.natoms),
                    "tier": tier,
                    "primary": tier == primary,
                    "n_shared_spins": len(shared),
                    "rmsd_omol_x2c": rmsd,
                }
                santi_side = k is not None
                rows.append(
                    {
                        **common,
                        "side": "omol",
                        "source": "santi" if santi_side else "maria",
                        "basis_group": grp,
                        "E_eh": float(o.final_energy_eh),
                        "final_xyz": o.final_xyz,
                        "s2_final": None if santi_side else o.s2_final,
                        "s2_dev": None if santi_side else o.s2_dev,
                        "default_basis_elements": (
                            "" if santi_side else o.default_basis_elements
                        ),
                        "job_path": (
                            f"multi_spin/{o.rel}"
                            if santi_side
                            else f"openac/{o.rel_path}"
                        ),
                    }
                )
                rows.append(
                    {
                        **common,
                        "side": "x2c",
                        "source": "maria",
                        "basis_group": xgrp,
                        "E_eh": float(x.final_energy_eh),
                        "final_xyz": x.final_xyz,
                        "s2_final": x.s2_final,
                        "s2_dev": x.s2_dev,
                        "default_basis_elements": x.default_basis_elements,
                        "job_path": f"openac/{x.rel_path}",
                    }
                )
    return pd.DataFrame(rows), diag


def summarize(df: pd.DataFrame, diag: dict) -> None:
    """Print tier and family counts for the primary ladders plus build diagnostics."""
    om = df[df.side == "omol"]
    lad = (
        om.groupby(["pair_id", "tier", "family", "primary"])
        .size()
        .rename("n")
        .reset_index()
    )
    lad["gaps"] = lad.n - 1
    p = lad[lad.primary]
    t = p.groupby("tier").agg(
        complexes=("n", "size"),
        with_gaps=("gaps", lambda g: int((g > 0).sum())),
        spin_pairs=("n", "sum"),
        gaps=("gaps", "sum"),
    )
    t.loc["total"] = t.sum()
    print("Primary ladders:\n" + t.to_string())
    f = pd.crosstab(p.family, p.tier)
    f["with_gaps"] = p.groupby("family").gaps.apply(lambda g: int((g > 0).sum()))
    f["total"] = p.groupby("family").size()
    print("\nPer family (primary):\n" + f.to_string())
    print(f"\nnon-primary ladders kept: {int((~lad.primary).sum())}")
    r = om.rmsd_omol_x2c
    print(
        f"omol vs X2C final RMSD (A): median {r.median():.3f}, p90 {r.quantile(0.9):.3f}, "
        f"max {r.max():.3f}, NaN (atom order differs) {int(r.isna().sum())}"
    )
    print(f"\nbroken ladders dropped ({len(diag['broken_ladders'])}):")
    for b in diag["broken_ladders"]:
        print(f"    {b}")
    print(f"santi converged omol jobs with no geometry: {len(diag['santi_no_geom'])}")
    for g in diag["santi_no_geom"]:
        print(f"    {g}")
    print(
        f"santi starts with no Maria complex of the same atoms+charge: {len(diag['santi_unmatched'])}"
    )
    for u in diag["santi_unmatched"]:
        print(f"    {u}")
    print(
        f"santi starts rejected by geometry (median RMSD >= tol or unordered): {len(diag['santi_rejected'])}"
    )
    for k, cx, med in diag["santi_rejected"]:
        print(f"    {k} -> {cx}: {med:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--maria-db", type=Path, default=DATA / "openac_dft.db")
    ap.add_argument("--santi-status", type=Path, default=DATA / "status_by_fluxid.csv")
    ap.add_argument("--santi-inv", type=Path, default=DATA / "inv_multi_spin_v2.csv")
    ap.add_argument("--santi-db", type=Path, default=DATA / "multi_spin_jobs.sqlite3")
    ap.add_argument(
        "--santi-extra-geoms",
        type=Path,
        default=DATA / "sella_missing_geoms.json",
        help="fallback geometries from orca.engrad for jobs the sqlite lacks",
    )
    ap.add_argument("-o", "--output", type=Path, default=DATA / "bench_pairs.parquet")
    ap.add_argument("--max-spread-kcal", type=float, default=200.0)
    ap.add_argument(
        "--rmsd-tol", type=float, default=0.5, help="A, Santi-vs-Maria identity check"
    )
    ap.add_argument(
        "--santi-primary-families",
        nargs="*",
        default=["crown_cryptands"],
        help="families where Santi ladders may be primary ('all', or none to disable)",
    )
    args = ap.parse_args()
    df, diag = build(args)
    df.to_parquet(args.output, index=False)
    print(f"wrote {args.output} ({len(df)} rows, {df.pair_id.nunique()} ladders)\n")
    summarize(df, diag)


if __name__ == "__main__":
    main()
