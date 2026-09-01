#!/usr/bin/env python3
"""Validate census root directories before running a corpus-wide census.

Census treats the IMMEDIATE subdirectories of each root as job directories. If a
root instead holds one more level of nesting (a ``jobs/`` or ``jobs_parsl/``
container), pointing census at it yields one junk row per intermediate directory
and finds no jobs at all. This script says, per root, whether the job dirs sit
directly inside it or one level down, and prints a corrected root list.

Dependency-free (stdlib only) so it runs on a login node with the system python,
and read-only: it opens nothing and writes only the file named by --write-fixed.

Usage:
    python check_census_roots.py census_roots.json
    python check_census_roots.py census_roots.json --write-fixed census_roots_fixed.json
    python check_census_roots.py /path/to/one_root --sample 200
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

# A directory is "job-like" if it holds any of these. Mirrors what census reads:
# orca.inp for composition, orca.out/.logs for status, orca.engrad for forces.
_JOB_MARKERS = ("orca.inp", "orca.inp.gz", "orca.out", "orca.out.gz", "orca.engrad")
_JOB_SUFFIXES = (".inp", ".inp.gz", ".out", ".out.gz", ".logs")
# orca_atom<N>.out is ORCA's atomic SCF initial-guess output, not job output.
# census excludes it (utils.status._ORCA_ATOM_RE) and so must this, or a
# directory holding nothing but abandoned guess scratch reads as a real job.
_ORCA_ATOM_RE = re.compile(r"^orca_atom\d+\.")
# Conventional container names checked first when a root looks like it nests.
_CONTAINER_HINTS = ("jobs", "jobs_parsl", "jobs_scratch", "job_dirs", "runs")


def is_job_dir(path: str) -> bool:
    """Return True if the directory looks like an ORCA job directory."""
    try:
        with os.scandir(path) as it:
            for entry in it:
                if not entry.is_file(follow_symlinks=False):
                    continue
                name = entry.name
                if _ORCA_ATOM_RE.match(name):
                    continue
                if name in _JOB_MARKERS or name.endswith(_JOB_SUFFIXES):
                    return True
                if name == "run_sella.py" or name == "sella_status.txt":
                    return True
    except OSError:
        return False
    return False


def subdirs(path: str, limit: int | None = None) -> list[str]:
    """Immediate subdirectory paths, sorted; symlinked dirs are not followed."""
    try:
        with os.scandir(path) as it:
            out = sorted(e.path for e in it if e.is_dir(follow_symlinks=False))
    except OSError:
        return []
    return out[:limit] if limit is not None else out


def probe(root: str, sample: int) -> dict:
    """Classify one root: direct job dirs, nested container, or neither."""
    result: dict = {
        "root": root,
        "exists": os.path.isdir(root),
        "n_subdirs": 0,
        "n_sampled": 0,
        "n_job_like": 0,
        "verdict": "",
        "suggested": None,
        "containers": [],
    }
    if not result["exists"]:
        result["verdict"] = "MISSING"
        return result

    kids = subdirs(root)
    result["n_subdirs"] = len(kids)
    if not kids:
        result["verdict"] = "EMPTY"
        return result

    probed = kids[:sample]
    result["n_sampled"] = len(probed)
    result["n_job_like"] = sum(1 for d in probed if is_job_dir(d))

    if result["n_job_like"]:
        frac = result["n_job_like"] / result["n_sampled"]
        result["verdict"] = "OK" if frac >= 0.5 else "MIXED"
        return result

    # No job dirs directly inside. Look one level down for a container.
    ordered = [
        os.path.join(root, h)
        for h in _CONTAINER_HINTS
        if os.path.isdir(os.path.join(root, h))
    ]
    ordered += [d for d in kids if d not in ordered]
    for cand in ordered[: sample + len(_CONTAINER_HINTS)]:
        inner = subdirs(cand, limit=min(sample, 25))
        if inner and sum(1 for d in inner if is_job_dir(d)) >= max(1, len(inner) // 2):
            result["containers"].append(cand)

    if result["containers"]:
        result["verdict"] = "NESTED"
        # One container -> that is the root. Many -> the root's children are the roots.
        result["suggested"] = (
            result["containers"][0]
            if len(result["containers"]) == 1
            else sorted(result["containers"])
        )
    else:
        result["verdict"] = "NO_JOBS"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check that census roots directly contain ORCA job directories."
    )
    parser.add_argument(
        "target",
        help="JSON array of roots, a newline-delimited list, or a single directory",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=60,
        help="Subdirectories to probe per root (default: 60)",
    )
    parser.add_argument(
        "--write-fixed",
        metavar="PATH",
        default=None,
        help="Write a corrected JSON root list (nested roots replaced by their "
        "container, MISSING/NO_JOBS roots dropped)",
    )
    args = parser.parse_args()

    if os.path.isdir(args.target):
        roots = [args.target.rstrip("/")]
    else:
        text = open(args.target).read()
        if text.lstrip().startswith("["):
            roots = [str(p).rstrip("/") for p in json.loads(text)]
        else:
            roots = [
                ln.strip().rstrip("/")
                for ln in text.splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            ]

    # An entry that contains another entry is a container, never a job root.
    containers = {
        p for p in roots if any(q != p and q.startswith(p + "/") for q in roots)
    }

    print(f"Probing {len(roots)} root(s), sampling up to {args.sample} subdirs each\n")
    print(f"{'verdict':<9} {'subdirs':>8} {'job-like':>9}  root")
    print("-" * 100)

    results = []
    for root in roots:
        if root in containers:
            print(f"{'CONTAINER':<9} {'-':>8} {'-':>9}  {root}")
            print(f"{'':<9} {'':>8} {'':>9}  ^ contains other listed roots; remove it")
            results.append({"root": root, "verdict": "CONTAINER", "suggested": None})
            continue
        r = probe(root, args.sample)
        counts = f"{r['n_job_like']}/{r['n_sampled']}" if r["n_sampled"] else "-"
        print(f"{r['verdict']:<9} {r['n_subdirs']:>8,} {counts:>9}  {root}")
        if r["verdict"] == "NESTED":
            if isinstance(r["suggested"], str):
                print(f"{'':<9} {'':>8} {'':>9}  -> use {r['suggested']}")
            else:
                for c in r["suggested"]:
                    print(f"{'':<9} {'':>8} {'':>9}  -> use {c}")
        elif r["verdict"] == "MIXED":
            print(
                f"{'':<9} {'':>8} {'':>9}  ^ under half the sampled subdirs look like "
                f"job dirs; check what else is in here"
            )
        results.append(r)

    by_verdict: dict[str, int] = {}
    for r in results:
        by_verdict[r["verdict"]] = by_verdict.get(r["verdict"], 0) + 1
    print("\nSummary: " + ", ".join(f"{v}={n}" for v, n in sorted(by_verdict.items())))

    total = sum(
        r.get("n_subdirs", 0) for r in results if r["verdict"] in ("OK", "MIXED")
    )
    if total:
        print(f"Job directories under OK/MIXED roots: {total:,}")

    if args.write_fixed:
        fixed: list[str] = []
        for r in results:
            if r["verdict"] in ("OK", "MIXED"):
                fixed.append(r["root"])
            elif r["verdict"] == "NESTED":
                s = r["suggested"]
                fixed.extend([s] if isinstance(s, str) else s)
        fixed = sorted(set(fixed))
        with open(args.write_fixed, "w") as f:
            f.write(json.dumps(fixed, indent=2) + "\n")
        print(f"\nWrote {len(fixed)} usable root(s) to {args.write_fixed}")

    bad = [
        r["root"]
        for r in results
        if r["verdict"] in ("MISSING", "NO_JOBS", "EMPTY", "CONTAINER")
    ]
    if bad:
        print(f"\n{len(bad)} root(s) need attention before running census:")
        for b in bad:
            print(f"  {b}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
