#!/usr/bin/env python3
"""Launch a grouped, load-balanced census over many job roots, then merge.

Job roots differ in size by an order of magnitude, so grouping them by parent
directory leaves one process running long after the rest have finished. This
counts each root's job directories first (one scandir per root, cheap), greedily
bin-packs the roots into N groups of roughly equal job count, launches one
census process per group, and merges the shards once every group succeeds.

Concurrency is (groups x workers) I/O streams. On Lustre the metadata server,
not the CPU, is the limit -- 8 groups x 8 workers is a reasonable starting point;
back off with --groups/--workers if the filesystem gets unhappy.

Stdlib only, so it runs with the login-node python. It shells out to
``python -m oact_utilities.workflows.census``, which does need the package.

Usage:
    python run_census_all.py census_roots_fixed.json --dry-run
    python run_census_all.py census_roots_fixed.json --outdir $HOME/census_out
    python run_census_all.py census_roots_fixed.json --groups 4 --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

CENSUS = [sys.executable, "-m", "oact_utilities.workflows.census"]


def count_jobs(root: str) -> int:
    """Number of immediate subdirectories (job dirs) under a root."""
    try:
        with os.scandir(root) as it:
            return sum(1 for e in it if e.is_dir(follow_symlinks=False))
    except OSError:
        return 0


def bin_pack(
    sized: list[tuple[str, int]], n_groups: int
) -> list[list[tuple[str, int]]]:
    """Greedy longest-processing-time-first packing into n_groups buckets."""
    groups: list[list[tuple[str, int]]] = [[] for _ in range(n_groups)]
    loads = [0] * n_groups
    for root, size in sorted(sized, key=lambda t: -t[1]):
        i = loads.index(min(loads))
        groups[i].append((root, size))
        loads[i] += size
    return [g for g in groups if g]


def group_label(group: list[tuple[str, int]]) -> str:
    """Short, filesystem-safe label naming a group after its largest root."""
    biggest = group[0][0].rstrip("/")
    parts = [p for p in biggest.split("/") if p][-2:]
    label = "_".join(parts) or "group"
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in label)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("roots_file", help="JSON array of job roots")
    parser.add_argument(
        "--outdir",
        default=os.path.expanduser("~/census_out"),
        help="Where shards and logs go (default: ~/census_out)",
    )
    parser.add_argument(
        "--groups", type=int, default=8, help="Concurrent processes (default: 8)"
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Threads per process (default: 8)"
    )
    parser.add_argument(
        "--debug", type=int, default=None, help="Cap job dirs per root (smoke test)"
    )
    parser.add_argument("--no-merge", action="store_true", help="Skip the merge step")
    parser.add_argument("--dry-run", action="store_true", help="Show the plan and exit")
    args = parser.parse_args()

    roots = [str(p).rstrip("/") for p in json.load(open(args.roots_file))]
    for a in roots:
        for b in roots:
            if a != b and b.startswith(a + "/"):
                print(f"Error: {a} contains {b}; remove the parent", file=sys.stderr)
                return 1

    # A root that does not exist is a typo or an unmounted filesystem, and
    # silently skipping it is the exact failure this whole exercise is meant to
    # catch -- abort. A root that exists but is empty only earns a warning.
    missing = [r for r in roots if not os.path.isdir(r)]
    if missing:
        print(f"Error: {len(missing)} root(s) do not exist:", file=sys.stderr)
        for r in missing:
            print(f"  {r}", file=sys.stderr)
        print(
            "Fix the root list (or re-run the probe) before launching.", file=sys.stderr
        )
        return 1

    print(f"Counting job dirs under {len(roots)} root(s)...")
    sized = [(r, count_jobs(r)) for r in roots]
    total = sum(n for _, n in sized)
    for r, n in sized:
        if n == 0:
            print(f"  Warning: {r} exists but holds no subdirectories -- skipping")
    sized = [(r, n) for r, n in sized if n > 0]
    if not sized:
        print("Error: no root holds any job directories", file=sys.stderr)
        return 1
    print(f"  {total:,} job directories across {len(sized)} non-empty root(s)\n")

    groups = bin_pack(sized, args.groups)
    print(f"{'group':<34}{'roots':>6}{'jobs':>12}")
    print("-" * 52)
    for g in groups:
        print(f"{group_label(g):<34}{len(g):>6}{sum(n for _, n in g):>12,}")
    spread = [sum(n for _, n in g) for g in groups]
    print(f"\nbalance: largest {max(spread):,} vs smallest {min(spread):,} jobs")
    print(
        f"concurrency: {len(groups)} processes x {args.workers} workers = "
        f"{len(groups) * args.workers} I/O streams"
    )

    if args.dry_run:
        print("\n--dry-run: nothing launched")
        for g in groups:
            print(f"\n# {group_label(g)}")
            for r, n in g:
                print(f"#   {n:>8,}  {r}")
        return 0

    shard_dir = os.path.join(args.outdir, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    print(f"\nShards -> {shard_dir}\nLogs   -> {args.outdir}/<group>.log\n")

    started = time.time()
    procs = []
    for g in groups:
        label = group_label(g)
        cmd = CENSUS + [r for r, _ in g]
        cmd += [
            "-o",
            os.path.join(shard_dir, f"{label}.parquet"),
            "--workers",
            str(args.workers),
        ]
        if args.debug is not None:
            cmd += ["--debug", str(args.debug)]
        log = open(os.path.join(args.outdir, f"{label}.log"), "w")
        procs.append(
            (label, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log)
        )
        print(f"  launched {label} ({sum(n for _, n in g):,} jobs)")

    print(f"\nWaiting on {len(procs)} process(es)...")
    failed = []
    for label, proc, log in procs:
        rc = proc.wait()
        log.close()
        status = "ok" if rc == 0 else f"FAILED rc={rc}"
        print(f"  {label:<34} {status}")
        if rc != 0:
            failed.append(label)

    print(f"\nScan wall time: {(time.time() - started) / 60:.1f} min")

    for label, _, _ in procs:
        path = os.path.join(args.outdir, f"{label}.log")
        with open(path, errors="replace") as f:
            for line in f:
                if line.startswith(("Wrote", "Scanned in")) or "WARNING" in line:
                    print(f"  [{label}] {line.rstrip()}")

    if failed:
        print(f"\n{len(failed)} group(s) failed; NOT merging. Check the logs:")
        for label in failed:
            print(f"  {args.outdir}/{label}.log")
        return 1

    if args.no_merge:
        print(
            f"\nSkipping merge. To merge:\n  {' '.join(CENSUS)} --merge {shard_dir} "
            f"-o {args.outdir}/census_combined.parquet"
        )
        return 0

    # Merged output deliberately sits OUTSIDE shards/, so re-merging the shard
    # directory can never re-ingest a previous merge output.
    combined = os.path.join(args.outdir, "census_combined.parquet")
    print(f"\nMerging shards -> {combined}")
    rc = subprocess.call(CENSUS + ["--merge", shard_dir, "-o", combined])
    if rc != 0:
        print("Merge failed", file=sys.stderr)
        return 1
    print(f"\nCombined table: {combined}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
