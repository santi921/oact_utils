#!/usr/bin/env python3
"""
Simple helper to traverse a folder tree up to a given depth and launch any
`flux_job.flux` files it finds using `flux batch`.

Usage:
    python run_multi_spin.py /path/to/root

The script accepts a single positional argument (root folder). Optional flags:
    --max-depth N    : maximum directory depth to traverse (default: 5)
    --dry-run        : do not actually execute flux, just print what would run
    --verbose        : print extra diagnostic info
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from time import time
from typing import Iterator
from oact_utilities.utils.status import (
    done_geo_opt_ase,
    check_file_termination,
    check_job_termination,
    check_geometry_steps,
    check_sella_complete,
)


def iter_dirs_limited(root: str, max_depth: int) -> Iterator[str]:
    """Yield directory paths under `root` up to `max_depth` levels deep.

    Depth 0 yields `root` itself. Depth 1 yields immediate subdirectories, etc.
    """
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        return

    for current_dir, dirs, files in os.walk(root):
        # compute relative depth
        rel = os.path.relpath(current_dir, root)
        if rel == ".":
            depth = 0
        else:
            depth = len(rel.split(os.sep))
        if depth > max_depth:
            # tell os.walk not to recurse deeper
            dirs[:] = []
            continue
        yield current_dir


def find_and_launch_flux(
    root: str,
    max_depth: int = 5,
    dry_run: bool = False,
    verbose: bool = False,
    skip_done: bool = False,
    skip_running: bool = False,
) -> int:
    """Traverse `root` up to `max_depth` and launch flux jobs when `flux_job.flux` is found.

    Returns the number of jobs launched (or that would be launched in dry-run).
    """
    launched = 0
    for d in iter_dirs_limited(root, max_depth=max_depth):
        flux_file = os.path.join(d, "flux_job.flux")
        if os.path.exists(flux_file):
            if skip_done:
                if check_sella_complete(d):
                    if verbose:
                        print(f"Skipping {d} because it has a completed job")
                    continue

                if check_job_termination(d) == -1:
                    print(f"Skipping {d} - failed job found.")
                    continue

                if check_job_termination(d):
                    print(f"Skipping {d} as it has a completed job.")
                    continue

            print(f"Found flux job: {flux_file}")
            cmd = f"cd {shlex.quote(d)} && flux batch flux_job.flux"
            if dry_run:
                print(f"DRY-RUN: {cmd}")
                launched += 1
            else:
                if verbose:
                    print(f"Executing: {cmd}")
                try:
                    # check the most recently edited file in the directory to see if it is within the last hour
                    # skip if it's recently modified and skip_running is True
                    if skip_running:
                        latest_file = max(
                            (os.path.join(d, f) for f in os.listdir(d)),
                            key=os.path.getmtime,
                        )
                        latest_mtime = os.path.getmtime(latest_file)
                        if (time.time() - latest_mtime) < 3600:  #
                            print(
                                f"Skipping {d} because it was last modified within the last hour"
                            )
                            continue
                    ret = subprocess.run(cmd, shell=True)
                    if ret.returncode == 0:
                        print(f"Launched job in {d}")
                    else:
                        print(
                            f"flux returned non-zero exit code ({ret.returncode}) for {d}"
                        )
                    launched += 1
                except Exception as e:
                    print(f"Failed to launch job in {d}: {e}")
    return launched


def main() -> None:
    
    parser = argparse.ArgumentParser(
        description="Launch flux_job.flux files under a folder (depth-limited)"
    )
    parser.add_argument("root", help="Root folder to traverse")
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Max subdirectory depth to traverse (default: 5)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Don't actually run flux, only print"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--skip-done",
        action="store_true",
        help="Skip directories that already have completed",
    )
    parser.add_argument(
        "--skip-running",
        action="store_true",
        help="Skip directories that already have a flux job launched",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        parser.error(f"Root path does not exist or is not a directory: {args.root}")

    n = find_and_launch_flux(
        args.root,
        max_depth=args.max_depth,
        dry_run=args.dry_run,
        verbose=args.verbose,
        skip_done=args.skip_done,
        skip_running=args.skip_running,
    )
    print(f"Total flux jobs launched/found: {n}")


if __name__ == "__main__":
    main()
