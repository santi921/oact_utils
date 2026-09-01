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


def _sanitize(text: str) -> str:
    """Reduce a string to characters safe in a filename."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in text) or "group"


def _label_at_depth(group: list[tuple[str, int]], depth: int) -> str:
    """Label a group after the trailing ``depth`` components of its largest root."""
    parts = [p for p in group[0][0].rstrip("/").split("/") if p][-depth:]
    return _sanitize("_".join(parts))


def group_labels(groups: list[list[tuple[str, int]]]) -> list[str]:
    """Unique, filesystem-safe label per group.

    Labels name the shard and log file, so a collision means two processes open
    the same paths -- and because the SQLite sink unlinks an existing file on
    open, one silently destroys the other's output mid-write. Two roots can
    easily share their last two components (``oact/nonact_4_06/jobs_parsl`` and
    ``BLASTNet/nonact_4_06/jobs_parsl``), so deepen the label until every one is
    distinct, then fall back to a numeric suffix.
    """
    for depth in (2, 3, 4, 5, 6):
        labels = [_label_at_depth(g, depth) for g in groups]
        if len(set(labels)) == len(labels):
            return labels
    labels = [_label_at_depth(g, 3) for g in groups]
    return [f"{lbl}_{i}" for i, lbl in enumerate(labels)]


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
    parser.add_argument(
        "--format",
        choices=("auto", "parquet", "sqlite", "csv"),
        default="auto",
        help="Shard format. 'auto' (default) uses parquet when pyarrow is "
        "importable and sqlite otherwise -- census falls back on its own, but "
        "then the shard name would not match what this script expects.",
    )
    parser.add_argument(
        "--clean-shards",
        action="store_true",
        help="Delete existing shards in the output directory before launching. "
        "Without this, a non-empty shard directory aborts: stale shards from an "
        "earlier or aborted run get picked up by the merge, and one written by "
        "an older census version fails it outright on a column mismatch.",
    )
    parser.add_argument("--no-merge", action="store_true", help="Skip the merge step")
    parser.add_argument("--dry-run", action="store_true", help="Show the plan and exit")
    args = parser.parse_args()

    if args.format == "auto":
        try:
            import pyarrow  # noqa: F401

            args.format = "parquet"
        except ImportError:
            args.format = "sqlite"
            print(
                "pyarrow not importable: writing sqlite shards. Parquet is much "
                "smaller and faster to load at this scale -- `pip install pyarrow` "
                "and re-run if you can.\n"
            )
    args.suffix = {"parquet": ".parquet", "sqlite": ".db", "csv": ".csv"}[args.format]

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
    labels = group_labels(groups)
    assert len(set(labels)) == len(labels), "group labels must be unique"
    width = max(34, max(len(lbl) for lbl in labels) + 2)
    print(f"{'group':<{width}}{'roots':>6}{'jobs':>12}")
    print("-" * (width + 18))
    for lbl, g in zip(labels, groups):
        print(f"{lbl:<{width}}{len(g):>6}{sum(n for _, n in g):>12,}")
    spread = [sum(n for _, n in g) for g in groups]
    print(f"\nbalance: largest {max(spread):,} vs smallest {min(spread):,} jobs")
    print(
        f"concurrency: {len(groups)} processes x {args.workers} workers = "
        f"{len(groups) * args.workers} I/O streams"
    )

    if args.dry_run:
        print("\n--dry-run: nothing launched")
        for lbl, g in zip(labels, groups):
            print(f"\n# {lbl}")
            for r, n in g:
                print(f"#   {n:>8,}  {r}")
        return 0

    shard_dir = os.path.join(args.outdir, "shards")
    os.makedirs(shard_dir, exist_ok=True)

    # A stale shard is silently merged into the new run's output, and one from an
    # older census version fails the merge on a column mismatch -- after the
    # whole scan has already been paid for. Refuse up front instead.
    stale = sorted(
        os.path.join(shard_dir, f)
        for f in os.listdir(shard_dir)
        if f.endswith((".parquet", ".db", ".csv"))
    )
    if stale:
        if not args.clean_shards:
            print(
                f"Error: {len(stale)} existing shard(s) in {shard_dir}:",
                file=sys.stderr,
            )
            for f in stale[:10]:
                print(f"  {os.path.basename(f)}", file=sys.stderr)
            if len(stale) > 10:
                print(f"  ... (+{len(stale) - 10} more)", file=sys.stderr)
            print(
                "\nThese would be merged into this run's output, and any written by "
                "an\nolder census version will fail the merge on a column mismatch. "
                "Re-run\nwith --clean-shards to delete them, or point --outdir "
                "somewhere new.",
                file=sys.stderr,
            )
            return 1
        for f in stale:
            os.unlink(f)
        print(f"Deleted {len(stale)} stale shard(s) from {shard_dir}")

    print(f"\nShards -> {shard_dir}\nLogs   -> {args.outdir}/<group>.log\n")

    started = time.time()
    procs = []
    for label, g in zip(labels, groups):
        cmd = CENSUS + [r for r, _ in g]
        cmd += [
            "-o",
            os.path.join(shard_dir, f"{label}{args.suffix}"),
            "--format",
            args.format,
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
            f"-o {args.outdir}/census_combined{args.suffix} --format {args.format}"
        )
        return 0

    # Merged output deliberately sits OUTSIDE shards/, so re-merging the shard
    # directory can never re-ingest a previous merge output.
    combined = os.path.join(args.outdir, f"census_combined{args.suffix}")
    print(f"\nMerging shards -> {combined}")
    rc = subprocess.call(
        CENSUS + ["--merge", shard_dir, "-o", combined, "--format", args.format]
    )
    if rc != 0:
        print("Merge failed", file=sys.stderr)
        return 1
    print(f"\nCombined table: {combined}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
