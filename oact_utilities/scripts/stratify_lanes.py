#!/usr/bin/env python3
"""Stratify a workflow DB by atom count x metal class and draft DRAC lane plans.

Dependency-free (stdlib only) so it runs on a login node with the CVMFS base
python or the venv. Opens the DB READ-ONLY, so it is safe to run against a live
DB that running jobs are still writing to.

Two modes:
  * --cluster <name>  : size lanes for that cluster's node (cores/mem) and note
                        its least-contended walltime tier. --ntasks gives a
                        partial-node (shared) allocation; omit for whole node.
  * (no --cluster)    : cluster-agnostic. Assume a whole node of --node-cores
                        (default 192), full-pack, lanes sized purely by the
                        atom-band -> cores-per-worker ratio. No partial, no
                        contention tier preference.

With --template, emit one filled submission script per populated lane. The
generator only substitutes per-lane VALUES into {{TOKEN}} placeholders, so it is
decoupled from whatever CLI flags submit_jobs supports on your branch.

Usage:
    python stratify_lanes.py <db> --cluster fir
    python stratify_lanes.py <db> --cluster fir --ntasks 64 \
        --template launch/run_parsl_drac.template.sh --outdir launch/generated/
    python stratify_lanes.py <db> --node-cores 192 \
        --template launch/run_parsl_drac.template.sh   # cluster-agnostic
"""

from __future__ import annotations

import argparse
import math
import os
import sqlite3
import statistics
import sys

ACTINIDES = {
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
}

# Per-cluster node spec: usable cores and memory per node, plus the
# least-contended by-core walltime tier observed on 2026-06-25 (see
# docs/plans/2026-06-25-drac-fairshare-throughput-diagnosis.md).
NODE_SPECS = {
    "fir": {
        "cores": 192,
        "mem_mb": 750000,
        "pref_tiers": "3h small / 12h bigger (contended everywhere)",
    },
    "narval": {
        "cores": 64,
        "mem_mb": 256000,
        "pref_tiers": "3h or 12h (both wide open)",
    },
    "rorqual": {
        "cores": 192,
        "mem_mb": 768000,
        "pref_tiers": "12h (3h is the most crowded tier)",
    },
    "nibi": {"cores": 192, "mem_mb": 766000, "pref_tiers": "flowing; 168h empty"},
    "trillium": {
        "cores": 192,
        "mem_mb": 768000,
        "pref_tiers": "whole-node only, 24h max",
    },
}

# Lane bands: (lo_atoms, hi_atoms inclusive, cores_per_worker, name).
# cores/worker IS the memory knob at --mem-per-cpu=3900M (8->31GB, 16->62GB,
# 32->125GB, 48->187GB per job on a fully-packed node) and the per-job speed
# knob (more cores -> shorter wall, so smaller molecules can hit shorter tiers).
LANE_BANDS = [
    (1, 40, 8, "small"),
    (41, 60, 16, "medium"),
    (61, 80, 32, "big"),
    (81, 100, 48, "huge"),
]

# Cold-start per-band wall estimate (hours), used ONLY when a band has no
# completed jobs. Deliberately conservative: measured actinide p90 runs well
# above naive guesses, and under-requesting walltime gets jobs KILLED for a
# wasted partial run. Prefer calibrating off a completed sibling chunk; these
# only protect a genuinely cold first launch.
DEFAULT_WALL_H = {(1, 40): 6.0, (41, 60): 10.0, (61, 80): 16.0, (81, 100): 20.0}

STD_TIERS_H = [3, 12, 24, 72, 168]

TIMEOUT_MARGIN_S = 600  # keep per-job timeout this far under the SBATCH walltime

# BATCH_SIZE = molecules pulled per allocation. Sized so workers stay fed for the
# whole walltime (workers * walltime/per_job_time), times a fluff factor. Per-job
# time comes from the calibration median when available, else these cold-start
# references at REF_CORES (your rule of thumb), scaled linearly by the lane cores.
BATCH_FLUFF = 1.3
ACT_REF_H = 4.0
NONACT_REF_H = 2.5
REF_CORES = 12


def is_actinide(metal: str | None) -> bool:
    return metal is not None and metal.strip() in ACTINIDES


def bucket_10(n: int) -> str:
    """10-atom-wide bucket label, capping at 100+."""
    if n <= 0:
        return "?"
    if n > 100:
        return "100+"
    lo = ((n - 1) // 10) * 10 + 1
    return f"{lo:>3}-{lo + 9:<3}"


def pick_tier(required_h: float) -> int:
    for t in STD_TIERS_H:
        if t >= required_h:
            return t
    return STD_TIERS_H[-1]


def strip_template_doc(text: str) -> str:
    """Drop the generator-only doc block (between the DOC sentinels) so emitted
    lane scripts don't carry the 'TEMPLATE -- do not run' header."""
    out, skip = [], False
    for ln in text.splitlines(keepends=True):
        s = ln.strip()
        if s.startswith("# ===TEMPLATE-DOC-START"):
            skip = True
            continue
        if s.startswith("# ===TEMPLATE-DOC-END"):
            skip = False
            continue
        if not skip:
            out.append(ln)
    return "".join(out)


def connect_ro(path: str) -> sqlite3.Connection:
    try:
        return sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.OperationalError as exc:
        sys.exit(f"Cannot open DB read-only: {path}\n{exc}")


def compute_lane(
    lo, hi, cpw, name, rows, band_walls, node_cores, ntasks_target, class_tag
):
    """Return a dict of per-lane values, or None if the band has no jobs."""
    act = sum(
        1
        for natoms, metal in rows
        if natoms and lo <= natoms <= hi and is_actinide(metal)
    )
    non = sum(
        1
        for natoms, metal in rows
        if natoms and lo <= natoms <= hi and not is_actinide(metal)
    )
    to_run = act + non
    if to_run == 0:
        return None

    workers = max(1, ntasks_target // cpw)
    ntasks = workers * cpw  # exact, avoids the oversubscription guard

    # Calibrate the tier from the wall times of the SAME metal class as this lane.
    bw = band_walls.get((lo, hi), {})
    if class_tag == "non_actinides":
        pairs = bw.get("non", [])
    elif class_tag == "actinides":
        pairs = bw.get("act", [])
    else:  # mixed
        pairs = bw.get("act", []) + bw.get("non", [])
    if pairs:
        hrs = sorted(p[0] for p in pairs)
        cores = [p[1] for p in pairs if p[1]]
        p90 = hrs[min(len(hrs) - 1, int(0.9 * len(hrs)))]
        med_cores = statistics.median(cores) if cores else cpw
        # Conservative core scaling: if this lane gives FEWER cores than the
        # completed jobs used, wall grows (scale up). Never assume a speedup.
        factor = max(1.0, med_cores / cpw)
        required_h = p90 * factor * 1.5
        typ_h = statistics.median(hrs) * factor  # typical per-job time, for batching
        basis = f"p90={p90:.1f}h@{int(med_cores)}c x{factor:.1f} x1.5 (n={len(pairs)})"
    else:
        required_h = DEFAULT_WALL_H[(lo, hi)]
        ref_h = NONACT_REF_H if class_tag == "non_actinides" else ACT_REF_H
        typ_h = (
            ref_h * REF_CORES / cpw
        )  # rule-of-thumb per-job time at this lane's cores
        basis = "default est (no completed data)"

    tier_h = pick_tier(required_h)
    job_timeout_s = min(int(required_h * 3600), tier_h * 3600 - TIMEOUT_MARGIN_S)
    # Pull enough molecules to keep every worker fed for the whole allocation + fluff.
    jobs_per_worker = max(1.0, tier_h / typ_h)
    batch_size = max(workers, math.ceil(workers * jobs_per_worker * BATCH_FLUFF))
    return {
        "name": name,
        "lo": lo,
        "hi": hi,
        "cpw": cpw,
        "workers": workers,
        "ntasks": ntasks,
        "tier_h": tier_h,
        "time": f"{tier_h:02d}:00:00",
        "job_timeout": job_timeout_s,
        "batch_size": batch_size,
        "to_run": to_run,
        "act": act,
        "non": non,
        "basis": basis,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "db",
        help="workflow .db to CALIBRATE from: its completed jobs set the per-band "
        "walltime tiers and its atom-count distribution shapes the lanes. This is "
        "NOT necessarily the DB you run -- point it at a finished sibling chunk for "
        "real tiers, and set --run-db to the chunk you actually submit. Opened "
        "read-only.",
    )
    ap.add_argument(
        "--cluster",
        choices=sorted(NODE_SPECS),
        help="size for this cluster; omit for cluster-agnostic whole-node lanes",
    )
    ap.add_argument(
        "--node-cores",
        type=int,
        help="node core count (default: cluster spec, or 192 if agnostic)",
    )
    ap.add_argument(
        "--ntasks",
        type=int,
        help="partial-node cores to reserve (cluster mode); omit = whole node",
    )
    ap.add_argument(
        "--status",
        default="to_run",
        help="status to stratify for the lane plan (default: to_run)",
    )
    ap.add_argument(
        "--mol-class",
        choices=["actinides", "non_actinides"],
        help="force the metal-class label in script names; "
        "default: auto-detect from the metal column",
    )
    ap.add_argument(
        "--run-db",
        help="DB the generated scripts actually submit against (baked into "
        "DB_PATH). Defaults to the calibration <db>; set it when you calibrate "
        "off a finished chunk but run a fresh one.",
    )
    ap.add_argument(
        "--venv-path",
        default="${HOME}/oact-env",
        help="per-cluster venv path baked into the scripts (differs on every "
        "cluster, e.g. Fir's ${HOME}/projects/def-yqw/${USER}/oact-env). Use "
        "${HOME}/${USER} forms -- a literal ~ does not expand inside the quotes. "
        "Default: ${HOME}/oact-env",
    )
    ap.add_argument(
        "--root-dir",
        default="${HOME}/scratch/oact_jobs/jobs_parsl/",
        help="per-cluster job-tree root baked into the scripts (scratch differs: "
        "Fir ${HOME}/scratch, Trillium/Rorqual $SCRATCH). Default: "
        "${HOME}/scratch/oact_jobs/jobs_parsl/",
    )
    ap.add_argument(
        "--template", help="tokenized .sh template; if set, emit one script per lane"
    )
    ap.add_argument(
        "--outdir",
        default="launch/generated",
        help="where to write generated lane scripts (default: launch/generated)",
    )
    args = ap.parse_args()

    if args.node_cores:
        node_cores = args.node_cores
    elif args.cluster:
        node_cores = NODE_SPECS[args.cluster]["cores"]
    else:
        node_cores = 192
    ntasks_target = args.ntasks if args.ntasks else node_cores
    tag = args.cluster or "generic"

    conn = connect_ro(args.db)
    cur = conn.cursor()

    print(
        f"=== DB: {args.db}   target: {tag} ({node_cores}c node, "
        f"reserving {ntasks_target}c) ===\n"
    )
    print("Status totals:")
    for status, n in cur.execute(
        "SELECT status, COUNT(*) FROM structures GROUP BY status ORDER BY COUNT(*) DESC"
    ):
        print(f"  {status:<12} {n:>8}")
    print()

    rows = cur.execute(
        "SELECT natoms, metal FROM structures WHERE status = ?", (args.status,)
    ).fetchall()
    if not rows:
        print(f"No rows with status='{args.status}'.")
        return

    # Metal-class label for script names: derive from the data so it can't drift.
    n_act = sum(1 for _, metal in rows if is_actinide(metal))
    n_non = len(rows) - n_act
    if args.mol_class:
        class_tag = args.mol_class
    elif n_non == 0 and n_act > 0:
        class_tag = "actinides"
    elif n_act == 0 and n_non > 0:
        class_tag = "non_actinides"
    else:
        class_tag = "mixed"
    print(
        f"Metal class: {class_tag}  ({n_act} actinide / {n_non} non-actinide rows "
        f"in '{args.status}')"
    )
    if class_tag == "mixed":
        print(
            "  WARNING: this DB mixes actinide and non-actinide molecules. Lanes run\n"
            "  BOTH (submit_jobs filters by atom count, not metal). Use single-class DBs\n"
            "  or pass --mol-class to force a label."
        )
    print()

    # Cross-tab: 10-atom bucket x metal class.
    buckets: dict[str, list[int]] = {}
    for natoms, metal in rows:
        if natoms is None:
            continue
        cell = buckets.setdefault(bucket_10(natoms), [0, 0])
        cell[0 if is_actinide(metal) else 1] += 1

    def bsort(label: str) -> int:
        return 10**9 if label in ("?", "100+") else int(label.split("-")[0])

    print(f"Stratification ('{args.status}') by atom count x metal class:")
    print(f"  {'bucket':<9} | {'actinide':>9} | {'non-act':>9} | {'total':>8}")
    print(f"  {'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*8}")
    ta = tn = 0
    for b in sorted(buckets, key=bsort):
        a, n = buckets[b]
        ta += a
        tn += n
        print(f"  {b:<9} | {a:>9} | {n:>9} | {a + n:>8}")
    print(f"  {'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*8}")
    print(f"  {'TOTAL':<9} | {ta:>9} | {tn:>9} | {ta + tn:>8}\n")

    # Completed wall-time per band x class (calibration data).
    comp = cur.execute(
        "SELECT natoms, metal, wall_time, n_cores FROM structures "
        "WHERE status = 'completed' AND wall_time IS NOT NULL AND wall_time > 0"
    ).fetchall()
    band_walls: dict[tuple, dict[str, list[tuple]]] = {}
    for natoms, metal, wt, ncw in comp:
        if natoms is None:
            continue
        for lo, hi, _, _ in LANE_BANDS:
            if lo <= natoms <= hi:
                cls = "act" if is_actinide(metal) else "non"
                band_walls.setdefault((lo, hi), {"act": [], "non": []})[cls].append(
                    (wt / 3600.0, ncw)
                )
                break

    print(
        "Completed wall-time by lane band (hours; cores = median n_cores those jobs used):"
    )
    if not comp:
        print(
            "  (no completed jobs with wall_time yet -- lane tiers below use default estimates)\n"
        )
    else:
        print(
            f"  {'band':<10} | {'class':<4} | {'n':>5} | {'cores':>5} | "
            f"{'median':>7} | {'p90':>6} | {'max':>6}"
        )
        print(f"  {'-'*10}-+-{'-'*4}-+-{'-'*5}-+-{'-'*5}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}")
        for lo, hi, _, _ in LANE_BANDS:
            for cls in ("act", "non"):
                pairs = band_walls.get((lo, hi), {}).get(cls, [])
                if not pairs:
                    continue
                hrs = sorted(p[0] for p in pairs)
                cores = [p[1] for p in pairs if p[1]]
                med_cores = int(statistics.median(cores)) if cores else 0
                p90 = hrs[min(len(hrs) - 1, int(0.9 * len(hrs)))]
                print(
                    f"  {f'{lo}-{hi}':<10} | {cls:<4} | {len(hrs):>5} | {med_cores:>5} | "
                    f"{statistics.median(hrs):>7.1f} | {p90:>6.1f} | {max(hrs):>6.1f}"
                )
        print()

    # Build lanes.
    lanes = [
        lane
        for lo, hi, cpw, name in LANE_BANDS
        if (
            lane := compute_lane(
                lo,
                hi,
                cpw,
                name,
                rows,
                band_walls,
                node_cores,
                ntasks_target,
                class_tag,
            )
        )
    ]

    pref = (
        NODE_SPECS[args.cluster]["pref_tiers"]
        if args.cluster
        else "agnostic (tier from p90 only)"
    )
    print(
        f"Draft lane plan for {tag} (node={node_cores}c, reserving {ntasks_target}c, "
        f"--mem-per-cpu=3900M):"
    )
    print(f"  preferred tier: {pref}")
    print(
        f"  {'lane':<7} | {'band':<8} | {'to_run':>7} | {'cores/wkr':>9} | {'workers':>7} | "
        f"{'ntasks':>6} | {'tier':>5} | {'batch':>5} | {'job_timeout':>11} | basis"
    )
    print(
        f"  {'-'*7}-+-{'-'*8}-+-{'-'*7}-+-{'-'*9}-+-{'-'*7}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}-+-{'-'*11}-+-{'-'*30}"
    )
    for L in lanes:
        band = f"{L['lo']}-{L['hi']}"
        jt = f"{L['job_timeout']}s"
        print(
            f"  {L['name']:<7} | {band:<8} | {L['to_run']:>7} | {L['cpw']:>9} | "
            f"{L['workers']:>7} | {L['ntasks']:>6} | {L['time']:>5} | {L['batch_size']:>5} | "
            f"{jt:>11} | {L['basis']}"
        )
    print("\n  DRAFT: validate each tier against measured p90 before locking. The")
    print("  ~13.9h SCF-nonconvergence tail (actinides) is NOT a sizing target --")
    print("  fix those with omol_base/KDIIS, not more walltime.\n")

    # Emit per-lane scripts.
    if not args.template:
        print("(no --template: plan only. Pass --template to emit lane scripts.)")
        return

    if not os.path.isfile(args.template):
        sys.exit(f"Template not found: {args.template}")
    with open(args.template) as fh:
        tpl = strip_template_doc(fh.read())
    os.makedirs(args.outdir, exist_ok=True)
    db_abs = os.path.abspath(args.run_db or args.db)
    if args.run_db:
        print(f"Generated scripts run against --run-db: {db_abs}")
    else:
        print(
            f"Generated scripts run against the calibration DB: {db_abs}\n"
            "  (pass --run-db to submit a different chunk than you calibrated from)"
        )

    print(
        f"Writing {len(lanes)} lane script(s) to {args.outdir}/ from {args.template}:"
    )
    for L in lanes:
        text = tpl
        subs = {
            "{{CLASS}}": class_tag,
            "{{VENV_PATH}}": args.venv_path,
            "{{ROOT_DIR}}": args.root_dir,
            "{{LANE}}": L["name"],
            "{{MIN_ATOMS}}": str(L["lo"]),
            "{{MAX_ATOMS}}": str(L["hi"]),
            "{{CORES_PER_WORKER}}": str(L["cpw"]),
            "{{MAX_WORKERS}}": str(L["workers"]),
            "{{NTASKS}}": str(L["ntasks"]),
            "{{TIME}}": L["time"],
            "{{BATCH_SIZE}}": str(L["batch_size"]),
            "{{JOB_TIMEOUT}}": str(L["job_timeout"]),
            "{{DB_PATH}}": db_abs,
        }
        for token, val in subs.items():
            text = text.replace(token, val)
        if "{{" in text:
            snippet = text[text.index("{{") : text.index("{{") + 40].splitlines()[0]
            print(
                f"  WARNING: unknown/unfilled token near '{snippet}' in lane {L['name']}"
            )
        out = os.path.join(args.outdir, f"run_drac_{tag}_{class_tag}_{L['name']}.sh")
        with open(out, "w") as fh:
            fh.write(text)
        os.chmod(out, 0o755)
        print(f"  {out}  ({L['to_run']} mols, {L['workers']}x{L['cpw']}c, {L['time']})")


if __name__ == "__main__":
    main()
