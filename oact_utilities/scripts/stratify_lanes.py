#!/usr/bin/env python3
"""Stratify a workflow DB by atom count x metal class and fan out DRAC lane scripts.

Dependency-free (stdlib only) so it runs on a login node with the CVMFS base
python or the venv. Opens the DB READ-ONLY, so it is safe against a live DB that
running jobs are still writing to.

Lean by design: it does NOT calibrate walltime/batch from completed jobs. The
durable value is the stratification (how many molecules sit in each atom-band x
actinide/non-actinide bucket -> which lanes to run, how many allocations). Lane
geometry (cores/worker, tier, waves) comes from fixed per-band defaults you tune
in BAND_SCHEMES (a default DRAC scheme + per-cluster overrides, e.g. sandia).

ONE template for every cluster (launch/run_parsl_drac.template.sh). Cluster
differences are CLI args (--cluster node size, --venv-path, --root-dir, --ntasks);
the actinide/non-actinide difference is --simple-input (omol vs omol_base),
auto-defaulted from the detected metal class.

Usage:
    python stratify_lanes.py <db> --cluster rorqual          # stratify + plan only
    python stratify_lanes.py <db> --cluster rorqual \
        --venv-path '${HOME}/oact-env' --root-dir '<scratch>/jobs_parsl/' \
        --template launch/run_parsl_drac.template.sh --outdir launch/generated/
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys

ACTINIDES = {
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
}  # fmt: skip

# Per-cluster node spec: usable cores per node + a backfill note (the
# least-contended tier observed 2026-06-25; see
# docs/plans/2026-06-25-drac-fairshare-throughput-diagnosis.md).
NODE_SPECS = {
    "fir": {"cores": 192, "note": "3h small / 12h bigger (contended everywhere)"},
    "narval": {"cores": 64, "note": "3h or 12h (both wide open)"},
    "rorqual": {"cores": 192, "note": "12h (3h is the most crowded tier)"},
    "nibi": {"cores": 192, "note": "flowing; 168h empty"},
    "trillium": {"cores": 192, "note": "whole-node only, 24h max"},
    "sandia": {"cores": 36, "note": "dedicated DoD alloc; longpri 48h; not packed"},
}

# Per-cluster band schemes: (lo_atoms, hi_atoms inclusive, cores_per_worker, name,
# tier_h, waves). cores/worker is the memory knob; tier_h is a fixed walltime;
# waves = sequential jobs per worker per allocation -> BATCH_SIZE = workers*waves.
# Tune to taste; the generator does NOT derive these from the DB.
BAND_SCHEMES = {
    # DRAC 192c clusters (Narval's 64c node packs fewer workers per band).
    "default": [
        (1, 40, 8, "small", 12, 4),
        (41, 60, 16, "medium", 12, 3),
        (61, 80, 32, "big", 24, 3),
        (81, 100, 48, "huge", 24, 2),
    ],
    # Sandia: 36c node, conda, longpri 48h wall. Two lanes by core count, both
    # whole-node: fast 6 workers x 6c (<=50 atoms), slow 3 workers x 12c (>50).
    # slow waves=2 keeps the 81-100 tail (~19h/job, clips the 48h wall) from
    # orphaning -- bump to 3 for 51-60-heavy chunks. 81-100 actinides are
    # SCF-nonconvergence prone: route them to omol_base/KDIIS, not a longer wall.
    "sandia": [
        (1, 50, 6, "fast", 48, 6),
        (51, 100, 12, "slow", 48, 2),
    ],
}

JOB_TIMEOUT_MARGIN_S = 1800  # Parsl kills stragglers this far before the SBATCH wall


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


def compute_lane(lo, hi, cpw, name, tier_h, waves, rows, ntasks_target):
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
    return {
        "name": name,
        "lo": lo,
        "hi": hi,
        "cpw": cpw,
        "workers": workers,
        "ntasks": workers * cpw,  # exact, avoids the oversubscription guard
        "tier_h": tier_h,
        "time": f"{tier_h:02d}:00:00",
        "job_timeout": tier_h * 3600 - JOB_TIMEOUT_MARGIN_S,
        "waves": waves,
        "batch_size": workers * waves,
        "to_run": to_run,
        "act": act,
        "non": non,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "db",
        help="workflow .db to stratify (and run, unless --run-db is set). Opened read-only.",
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
        help="partial-node cores to reserve; omit = whole node",
    )
    ap.add_argument(
        "--status", default="to_run", help="status to stratify (default: to_run)"
    )
    ap.add_argument(
        "--mol-class",
        choices=["actinides", "non_actinides"],
        help="force the metal-class label in script names; default: auto-detect",
    )
    ap.add_argument(
        "--simple-input",
        choices=["omol", "omol_base", "x2c", "dk3", "pm3"],
        help="ORCA template baked in; default: omol for actinide DBs, omol_base "
        "for non-actinide (from the detected metal class)",
    )
    ap.add_argument(
        "--run-db",
        help="DB the generated scripts submit against (DB_PATH). Defaults to <db>.",
    )
    ap.add_argument(
        "--venv-path",
        default="${HOME}/oact-env",
        help="per-cluster venv path baked in (use ${HOME}/${USER} forms; ~ won't expand)",
    )
    ap.add_argument(
        "--root-dir",
        default="${HOME}/scratch/oact_jobs/jobs_parsl/",
        help="per-cluster job-tree root baked in (use ${HOME}/$SCRATCH forms)",
    )
    ap.add_argument(
        "--template", help="per-cluster tokenized .sh; if set, emit one script per lane"
    )
    ap.add_argument(
        "--outdir", default="launch/generated", help="where to write lane scripts"
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
    # ORCA template: omol for actinides, omol_base for non-actinides, unless forced.
    if args.simple_input:
        simple_input = args.simple_input
        si_src = "from --simple-input"
    else:
        simple_input = "omol_base" if class_tag == "non_actinides" else "omol"
        si_src = f"auto from {class_tag}"
    print(f"ORCA template (SIMPLE_INPUT): {simple_input}  ({si_src})")
    print()

    # Stratification cross-tab: 10-atom bucket x metal class (the main output).
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
    print(f"  {'-' * 9}-+-{'-' * 9}-+-{'-' * 9}-+-{'-' * 8}")
    ta = tn = 0
    for b in sorted(buckets, key=bsort):
        a, n = buckets[b]
        ta += a
        tn += n
        print(f"  {b:<9} | {a:>9} | {n:>9} | {a + n:>8}")
    print(f"  {'-' * 9}-+-{'-' * 9}-+-{'-' * 9}-+-{'-' * 8}")
    print(f"  {'TOTAL':<9} | {ta:>9} | {tn:>9} | {ta + tn:>8}\n")

    # Build lanes (fixed per-band geometry; no DB calibration).
    bands = BAND_SCHEMES.get(args.cluster, BAND_SCHEMES["default"])
    lanes = [
        lane
        for lo, hi, cpw, name, tier_h, waves in bands
        if (lane := compute_lane(lo, hi, cpw, name, tier_h, waves, rows, ntasks_target))
    ]

    note = NODE_SPECS[args.cluster]["note"] if args.cluster else "cluster-agnostic"
    print(
        f"Lane plan for {tag} (node={node_cores}c, reserving {ntasks_target}c, "
        f"--mem-per-cpu=3900M):"
    )
    print(f"  backfill note: {note}")
    print(
        f"  {'lane':<7} | {'band':<8} | {'to_run':>7} | {'cores/wkr':>9} | {'workers':>7} | "
        f"{'ntasks':>6} | {'tier':>5} | {'waves':>5} | {'batch':>5}"
    )
    print(
        f"  {'-' * 7}-+-{'-' * 8}-+-{'-' * 7}-+-{'-' * 9}-+-{'-' * 7}-+-{'-' * 6}-+-{'-' * 5}-+-{'-' * 5}-+-{'-' * 5}"
    )
    for L in lanes:
        band = f"{L['lo']}-{L['hi']}"
        print(
            f"  {L['name']:<7} | {band:<8} | {L['to_run']:>7} | {L['cpw']:>9} | "
            f"{L['workers']:>7} | {L['ntasks']:>6} | {L['time']:>5} | {L['waves']:>5} | "
            f"{L['batch_size']:>5}"
        )
    print(
        "\n  Geometry/tier/batch are FIXED defaults from BAND_SCHEMES (tune in code), not\n"
        "  derived from the DB. batch = workers * waves; overshoot strands the tail as\n"
        "  RUNNING (clear with dashboard --recover-orphans), undershoot idles cores.\n"
    )

    # Emit per-lane scripts from the per-cluster template.
    if not args.template:
        print(
            "(no --template: plan only. Pass the cluster template to emit lane scripts.)"
        )
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
            f"Generated scripts run against the stratified DB: {db_abs}\n"
            "  (pass --run-db to submit a different chunk than you stratified)"
        )

    print(
        f"Writing {len(lanes)} lane script(s) to {args.outdir}/ from {args.template}:"
    )
    for L in lanes:
        text = tpl
        subs = {
            "{{CLASS}}": class_tag,
            "{{LANE}}": L["name"],
            "{{MIN_ATOMS}}": str(L["lo"]),
            "{{MAX_ATOMS}}": str(L["hi"]),
            "{{CORES_PER_WORKER}}": str(L["cpw"]),
            "{{MAX_WORKERS}}": str(L["workers"]),
            "{{NTASKS}}": str(L["ntasks"]),
            "{{TIME}}": L["time"],
            "{{N_HOURS}}": str(L["tier_h"]),
            "{{BATCH_SIZE}}": str(L["batch_size"]),
            "{{JOB_TIMEOUT}}": str(L["job_timeout"]),
            "{{DB_PATH}}": db_abs,
            "{{ROOT_DIR}}": args.root_dir,
            "{{VENV_PATH}}": args.venv_path,
            "{{SIMPLE_INPUT}}": simple_input,
        }
        for token, val in subs.items():
            text = text.replace(token, val)
        if "{{" in text:
            snippet = text[text.index("{{") : text.index("{{") + 40].splitlines()[0]
            print(f"  WARNING: unfilled token near '{snippet}' in lane {L['name']}")
        out = os.path.join(args.outdir, f"run_drac_{tag}_{class_tag}_{L['name']}.sh")
        with open(out, "w") as fh:
            fh.write(text)
        os.chmod(out, 0o755)
        print(f"  {out}  ({L['to_run']} mols, {L['workers']}x{L['cpw']}c, {L['time']})")


if __name__ == "__main__":
    main()
