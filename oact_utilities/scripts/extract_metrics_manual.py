"""
Walk a root directory of ORCA job subdirectories and write
``orca_metrics.json`` and ``generator_metrics.json`` next to each ORCA output.

Useful when you have a folder of completed jobs without the corresponding
workflow SQLite DB and want the same per-job caches that the Parsl pipeline
in ``submit_jobs.py`` produces on the fly.

Usage:
    python -m oact_utilities.scripts.extract_metrics_manual /path/to/root
    python -m oact_utilities.scripts.extract_metrics_manual /path/to/root \
        --workers 8 --recompute --no-generator
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from oact_utilities.utils.analysis import (
    GENERATOR_AVAILABLE,
    find_timings_and_cores,
    parse_generator_data,
    parse_job_metrics,
)
from oact_utilities.utils.status import pull_log_file


def _has_orca_output(job_dir: Path) -> bool:
    try:
        pull_log_file(str(job_dir))
        return True
    except FileNotFoundError:
        return False


def _find_job_dirs(root: Path, recursive: bool) -> list[Path]:
    if not recursive:
        return sorted(d for d in root.iterdir() if d.is_dir() and _has_orca_output(d))

    results: list[Path] = []
    for dirpath, _dirnames, _filenames in os.walk(root):
        d = Path(dirpath)
        if _has_orca_output(d):
            results.append(d)
    results.sort()
    return results


def process_job(
    job_dir: Path,
    recompute: bool,
    do_generator: bool,
    unzip: bool,
    hours_cutoff: float,
) -> dict:
    result: dict = {
        "job_dir": str(job_dir),
        "metrics_ok": False,
        "generator_ok": False,
        "success": False,
        "error": None,
    }

    try:
        metrics = parse_job_metrics(
            job_dir,
            unzip=unzip,
            hours_cutoff=hours_cutoff,
            recompute=recompute,
        )
        result["metrics_ok"] = True
        result["success"] = bool(metrics.get("success"))

        # Attach wall_time / n_cores into the cache so downstream tooling
        # has the same fields the DB-driven path records.
        try:
            log_file = pull_log_file(str(job_dir))
            n_cores, time_dict = find_timings_and_cores(log_file)
            if time_dict and "Total" in time_dict:
                metrics["wall_time"] = time_dict["Total"]
            if n_cores is not None:
                metrics["n_cores"] = n_cores
        except Exception as exc:
            result["error"] = f"timing: {exc}"

    except Exception as exc:
        result["error"] = f"metrics: {exc}"
        return result

    if do_generator and GENERATOR_AVAILABLE and result["success"]:
        try:
            gen = parse_generator_data(job_dir, recompute=recompute)
            result["generator_ok"] = gen is not None
        except Exception as exc:
            prev = result["error"]
            msg = f"generator: {exc}"
            result["error"] = f"{prev}; {msg}" if prev else msg

    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path, help="Root directory of job subdirectories")
    p.add_argument("--workers", type=int, default=4, help="Parallel workers")
    p.add_argument(
        "--recompute",
        action="store_true",
        help="Ignore existing orca_metrics.json / generator_metrics.json caches",
    )
    p.add_argument(
        "--no-generator",
        action="store_true",
        help="Skip parse_generator_data (qtaim_gen) even if available",
    )
    p.add_argument(
        "--unzip",
        action="store_true",
        help="Look for gzipped ORCA outputs (.out.gz from quacc)",
    )
    p.add_argument(
        "--hours-cutoff",
        type=float,
        default=6.0,
        help="Timeout threshold (h) passed to parse_job_metrics",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into nested subdirectories (default: only immediate children)",
    )
    args = p.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    if args.no_generator:
        do_generator = False
    else:
        do_generator = GENERATOR_AVAILABLE
        if not GENERATOR_AVAILABLE:
            print("Note: qtaim_gen not installed; skipping generator_metrics.json")

    job_dirs = _find_job_dirs(root, recursive=args.recursive)
    print(f"Found {len(job_dirs)} job directories under {root}")
    if not job_dirs:
        return

    n_metrics = 0
    n_generator = 0
    n_success = 0
    n_errors = 0
    errors: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                process_job,
                jd,
                args.recompute,
                do_generator,
                args.unzip,
                args.hours_cutoff,
            ): jd
            for jd in job_dirs
        }
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r["metrics_ok"]:
                n_metrics += 1
            if r["generator_ok"]:
                n_generator += 1
            if r["success"]:
                n_success += 1
            if r["error"]:
                n_errors += 1
                errors.append((r["job_dir"], r["error"]))
            if i % 50 == 0 or i == len(futures):
                print(
                    f"  [{i}/{len(futures)}] metrics={n_metrics} "
                    f"generator={n_generator} success={n_success} errors={n_errors}"
                )

    print()
    print(f"Job directories scanned : {len(job_dirs)}")
    print(f"orca_metrics.json written: {n_metrics}")
    print(f"generator_metrics.json   : {n_generator}")
    print(f"Successful ORCA jobs     : {n_success}")
    print(f"Errors                   : {n_errors}")

    if errors:
        print("\nFirst 20 errors:")
        for jd, msg in errors[:20]:
            print(f"  {jd}: {msg}")


if __name__ == "__main__":
    main()
