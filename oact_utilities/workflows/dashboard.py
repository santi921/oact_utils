"""Dashboard for monitoring architector workflow status.

This script provides a command-line dashboard to monitor the status of
high-throughput architector calculations. It can be run on HPC systems
to check job progress.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, TypedDict

from ..utils.analysis import read_sella_log_tail
from ..utils.status import check_job_termination, parse_failure_reason
from .architector_workflow import ArchitectorWorkflow, JobStatus, StatusGroupUpdate
from .clean import MARKER_ERROR_MESSAGE, is_marker_blocked
from .job_dir_patterns import (
    DEFAULT_JOB_DIR_PATTERN,
    apply_job_dir_prefix,
    render_job_dir_pattern,
)
from .wandb_logger import (
    WANDB_AVAILABLE,
    add_wandb_args,
    compute_metrics_stats,
    finish_wandb_run,
    init_wandb_run,
    log_campaign_snapshot,
)


def print_header(text: str, width: int = 80):
    """Print a formatted header."""
    print("\n" + "=" * width)
    print(text.center(width))
    print("=" * width)


def print_summary(workflow: ArchitectorWorkflow):
    """Print summary statistics table."""
    print_header("Workflow Status Summary")

    counts = workflow.count_by_status()
    total = sum(counts.values())

    # Create formatted table
    print(f"\n{'Status':<15} {'Count':>10} {'Percent':>10}")
    print("-" * 40)

    for status in JobStatus:
        count = counts.get(status, 0)
        pct = 100 * count / total if total > 0 else 0
        print(f"{status.value:<15} {count:>10} {pct:>9.1f}%")

    print("-" * 40)
    print(f"{'TOTAL':<15} {total:>10} {100.0:>9.1f}%")

    # Metrics availability breakdown
    metric_cols = [
        ("job_dir", "job_dir IS NOT NULL AND job_dir != ''"),
        ("max_forces", "max_forces IS NOT NULL"),
        ("final_energy", "final_energy IS NOT NULL"),
        ("scf_steps", "scf_steps IS NOT NULL"),
        ("wall_time", "wall_time IS NOT NULL"),
        ("n_cores", "n_cores IS NOT NULL"),
    ]

    print(f"\n{'Metric':<15} {'Completed':>12} {'All Jobs':>12}")
    print("-" * 40)

    completed_count = counts.get(JobStatus.COMPLETED, 0)

    # Single query to get all metric counts (avoids 12 round-trips on Lustre)
    cols_sql = ", ".join(
        f"SUM(CASE WHEN {cond} THEN 1 ELSE 0 END), "
        f"SUM(CASE WHEN status = 'completed' AND {cond} THEN 1 ELSE 0 END)"
        for _, cond in metric_cols
    )
    cur = workflow._execute_with_retry(f"SELECT {cols_sql} FROM structures")
    row = cur.fetchone()

    for i, (col_label, _condition) in enumerate(metric_cols):
        n_all = row[i * 2] or 0
        n_comp = row[i * 2 + 1] or 0
        pct_comp = 100 * n_comp / completed_count if completed_count > 0 else 0
        pct_all = 100 * n_all / total if total > 0 else 0
        print(
            f"{col_label:<15} {n_comp:>6}/{completed_count:<5} ({pct_comp:>5.1f}%)"
            f" {n_all:>6}/{total:<5} ({pct_all:>5.1f}%)"
        )

    # Chronic reset indicator -- always shown when jobs exceed the lower threshold
    count_5, count_25 = workflow.get_chronic_reset_counts()
    if count_5 > 0:
        print(
            f"\nChronic resets (to_run):  "
            f"{count_5} failed 5+ times,  {count_25} failed 25+ times"
        )

    print()


def print_metrics_summary(workflow: ArchitectorWorkflow):
    """Print summary of computational metrics (forces, SCF steps, timing)."""
    import pandas as pd

    # Query completed jobs with metrics
    cur = workflow._execute_with_retry(
        """
        SELECT max_forces, scf_steps, wall_time, n_cores, final_energy
        FROM structures
        WHERE status = 'completed' AND max_forces IS NOT NULL
        """
    )
    rows = cur.fetchall()

    if not rows:
        print("\nNo metrics data available yet.")
        return

    df = pd.DataFrame(
        rows,
        columns=["max_forces", "scf_steps", "wall_time", "n_cores", "final_energy"],
    )

    print_header("Computational Metrics (Completed Jobs)")

    # Energy statistics
    energy_data = df["final_energy"].dropna()
    if len(energy_data) > 0:
        print("\nFinal Energy (Eh):")
        print(f"  Mean:   {energy_data.mean():.6f}")
        print(f"  Median: {energy_data.median():.6f}")
        print(f"  Min:    {energy_data.min():.6f}")
        print(f"  Max:    {energy_data.max():.6f}")

    # Forces statistics
    print("\nMax Forces (Eh/Bohr):")
    print(f"  Mean:   {df['max_forces'].mean():.6f}")
    print(f"  Median: {df['max_forces'].median():.6f}")
    print(f"  Min:    {df['max_forces'].min():.6f}")
    print(f"  Max:    {df['max_forces'].max():.6f}")

    # SCF steps statistics
    scf_data = df["scf_steps"].dropna()
    if len(scf_data) > 0:
        print("\nSCF Steps:")
        print(f"  Mean:   {scf_data.mean():.1f}")
        print(f"  Median: {scf_data.median():.0f}")
        print(f"  Min:    {scf_data.min():.0f}")
        print(f"  Max:    {scf_data.max():.0f}")

    # Timing statistics
    time_data = df["wall_time"].dropna()
    if len(time_data) > 0:
        print("\nWall Time (seconds):")
        print(f"  Mean:   {time_data.mean():.1f}")
        print(f"  Median: {time_data.median():.1f}")
        print(f"  Min:    {time_data.min():.1f}")
        print(f"  Max:    {time_data.max():.1f}")
        print(f"  Total:  {time_data.sum():.1f} ({time_data.sum() / 3600:.2f} hours)")

    # Cores statistics
    cores_data = df["n_cores"].dropna()
    if len(cores_data) > 0:
        print("\nCores Used:")
        print(f"  Mean:   {cores_data.mean():.1f}")
        print(f"  Min:    {cores_data.min():.0f}")
        print(f"  Max:    {cores_data.max():.0f}")

        # Calculate total core-hours
        if len(time_data) > 0:
            # Only compute for jobs that have both timing and cores data
            valid_rows = df.dropna(subset=["wall_time", "n_cores"])
            if len(valid_rows) > 0:
                core_seconds = (valid_rows["wall_time"] * valid_rows["n_cores"]).sum()
                print(f"  Total core-hours: {core_seconds / 3600:.2f}")

    print(f"\nTotal jobs with metrics: {len(df)}")
    if len(time_data) > 0:
        print(f"Jobs with timing data: {len(time_data)}\n")
    else:
        print()


def has_sella_jobs(workflow: ArchitectorWorkflow) -> bool:
    """Return True if any row in the DB has optimizer='sella'.

    Uses ``SELECT 1 ... LIMIT 1`` instead of COUNT(*) so the check is
    cheap even on a 100k-row DB -- scans at most one matching row.
    """
    cur = workflow._execute_with_retry(
        "SELECT 1 FROM structures WHERE optimizer = 'sella' LIMIT 1"
    )
    return cur.fetchone() is not None


def print_sella_summary(workflow: ArchitectorWorkflow, limit: int = 20) -> None:
    """Print sella-specific convergence counts + step statistics.

    Sections:
      - Convergence tri-state counts (CONVERGED / NOT_CONVERGED /
        ERROR / RUNNING) for rows where optimizer='sella'
      - Step statistics (mean, max, count) over completed sella jobs,
        computed in-SQL so we never materialize the full step column
      - Non-converged job list (up to `limit`), so users can spot
        tunable-parameter problems without a separate view
    """
    print_header("Sella Optimization Summary")

    if not has_sella_jobs(workflow):
        print("\nNo sella jobs in this database.")
        return

    # One round-trip for tri-state counts + step aggregates. Pushing
    # AVG/MAX/COUNT into SQL avoids pulling every completed row's
    # sella_steps into Python, which on a 100k-row campaign would be a
    # ~400KB list just to compute 3 scalars.
    #
    # sella_converged tri-state:
    #   completed + sella_converged=1  -> CONVERGED
    #   completed + sella_converged=0  -> NOT_CONVERGED (hit max_steps)
    #   completed + sella_converged IS NULL -> ERROR
    #   running                        -> RUNNING
    cur = workflow._execute_with_retry(
        """
        SELECT
          SUM(CASE WHEN status = 'completed' AND sella_converged = 1 THEN 1 ELSE 0 END),
          SUM(CASE WHEN status = 'completed' AND sella_converged = 0 THEN 1 ELSE 0 END),
          SUM(CASE WHEN status IN ('failed','timeout') THEN 1 ELSE 0 END),
          SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END),
          COUNT(*),
          AVG(CASE WHEN status = 'completed' AND sella_steps IS NOT NULL THEN sella_steps END),
          MAX(CASE WHEN status = 'completed' AND sella_steps IS NOT NULL THEN sella_steps END),
          SUM(CASE WHEN status = 'completed' AND sella_steps IS NOT NULL THEN 1 ELSE 0 END)
        FROM structures
        WHERE optimizer = 'sella'
        """
    )
    row = cur.fetchone()
    n_conv = row[0] or 0
    n_not_conv = row[1] or 0
    n_error = row[2] or 0
    n_running = row[3] or 0
    total = row[4] or 0
    step_mean = row[5]  # None when there are no completed sella jobs
    step_max = row[6]
    step_count = row[7] or 0

    print(f"\n{'State':<18} {'Count':>8} {'Percent':>10}")
    print("-" * 40)
    for label, count in [
        ("CONVERGED", n_conv),
        ("NOT_CONVERGED", n_not_conv),
        ("ERROR", n_error),
        ("RUNNING", n_running),
    ]:
        pct = 100 * count / total if total > 0 else 0
        print(f"{label:<18} {count:>8} {pct:>9.1f}%")
    print("-" * 40)
    print(f"{'TOTAL':<18} {total:>8} {100.0:>9.1f}%")

    if step_count > 0 and step_mean is not None:
        print("\nSella steps (completed jobs):")
        print(f"  Mean:   {step_mean:.1f}")
        print(f"  Max:    {step_max}")
        print(f"  Jobs:   {step_count}")

    # Non-converged detail list, inline.
    if n_not_conv > 0:
        cur = workflow._execute_with_retry(
            f"""
            SELECT id, orig_index, sella_steps FROM structures
            WHERE optimizer = 'sella'
              AND status = 'completed'
              AND sella_converged = 0
            ORDER BY sella_steps DESC NULLS LAST
            LIMIT {int(limit)}
            """
        )
        nc_rows = cur.fetchall()
        print_header(f"Non-converged sella jobs (showing up to {limit})")
        print(f"\n{'ID':<8} {'Orig Index':<12} {'Steps':>6}")
        print("-" * 30)
        for job_id, orig_idx, n_steps in nc_rows:
            steps_str = str(n_steps) if n_steps is not None else "-"
            print(f"{job_id:<8} {orig_idx:<12} {steps_str:>6}")
        if n_not_conv > len(nc_rows):
            print(f"\n... and {n_not_conv - len(nc_rows)} more non-converged jobs")


class SellaProgressRow(TypedDict):
    """Snapshot of a running sella job's current state.

    Used by the running-progress dashboard view. Declared as a TypedDict
    so callers catch key typos at static-analysis time and the return
    shape of ``_probe_sella_current_step`` stays in sync with consumers.
    """

    job_id: int
    orig_index: int
    step: int
    fmax: float
    energy: float
    mtime: float | None


def _probe_sella_current_step(
    job_id: int,
    orig_index: int,
    job_dir: Path,
) -> SellaProgressRow | None:
    """Thread-pool worker: tail-read sella.log for one running job.

    Pure I/O -- no DB access. Does NOT open opt.traj (unsafe while
    the optimizer is appending).
    """
    sella_log = job_dir / "sella.log"
    if not sella_log.exists():
        return None
    row = read_sella_log_tail(sella_log)
    if row is None:
        return None
    try:
        mtime: float | None = os.path.getmtime(sella_log)
    except OSError:
        mtime = None
    return SellaProgressRow(
        job_id=job_id,
        orig_index=orig_index,
        step=row["step"],
        fmax=row["fmax"],
        energy=row["energy"],
        mtime=mtime,
    )


def show_sella_running_progress(
    workflow: ArchitectorWorkflow,
    root_dir: Path,
    job_dir_pattern: str = DEFAULT_JOB_DIR_PATTERN,
    limit: int = 20,
    workers: int = 8,
) -> None:
    """Show running sella jobs with their current step from sella.log.

    Columns: ID | Orig Index | Step | fmax | Energy | Last update

    Tail-reads each running job's sella.log in a thread pool (same
    pattern as _parallel_status_check). Never opens opt.traj.

    `job_dir_pattern` supports the existing {hostname}, {orig_index},
    {id} placeholders via render_job_dir_pattern().

    `workers` defaults to 8 -- this path does three metadata ops per
    job (exists, getmtime, read-tail) rather than one, so the shared
    dashboard default of 4 undersubscribes on Lustre at 1000+ jobs.
    Override with the existing ``--workers`` CLI flag.
    """
    import datetime
    from concurrent.futures import ThreadPoolExecutor, as_completed

    running = [
        j
        for j in workflow.get_jobs_by_status(JobStatus.RUNNING, include_geometry=False)
        if j.optimizer == "sella"
    ]
    if not running:
        print_header("Running Sella Jobs")
        print("\nNo running sella jobs.")
        return

    try:
        from tqdm import tqdm

        use_tqdm = True
    except ImportError:
        use_tqdm = False

    probed: list[SellaProgressRow] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for j in running:
            job_dir_name = render_job_dir_pattern(
                job_dir_pattern, orig_index=j.orig_index, job_id=j.id
            )
            job_dir = Path(root_dir) / job_dir_name
            futures[
                executor.submit(_probe_sella_current_step, j.id, j.orig_index, job_dir)
            ] = j

        iterator = as_completed(futures)
        if use_tqdm:
            iterator = tqdm(
                iterator,
                total=len(futures),
                desc="Reading sella.log tails",
                unit="job",
            )
        for fut in iterator:
            result = fut.result()
            if result is not None:
                probed.append(result)

    print_header(f"Running Sella Jobs (showing up to {limit})")
    if not probed:
        print("\nRunning sella jobs found, but none have parseable sella.log yet.")
        return

    # Sort by step descending -- high-step jobs are closest to max_steps
    probed.sort(key=lambda r: (r["step"], r["fmax"]), reverse=True)

    print(
        f"\n{'ID':<8} {'Orig':>6} {'Step':>5} {'fmax':>10} {'Energy':>16} {'Updated'}"
    )
    print("-" * 70)
    for r in probed[:limit]:
        if r["mtime"] is not None:
            upd = datetime.datetime.fromtimestamp(r["mtime"]).strftime("%Y-%m-%d %H:%M")
        else:
            upd = "-"
        print(
            f"{r['job_id']:<8} {r['orig_index']:>6} {r['step']:>5} "
            f"{r['fmax']:>10.4f} {r['energy']:>16.6f} {upd}"
        )
    if len(probed) > limit:
        print(f"\n... and {len(probed) - limit} more running jobs")


def print_progress_bar(
    completed: int, total: int, width: int = 50, label: str = "Progress"
):
    """Print a text-based progress bar."""
    if total == 0:
        fraction = 0.0
    else:
        fraction = completed / total

    filled = int(width * fraction)
    bar = "█" * filled + "░" * (width - filled)
    pct = 100 * fraction
    print(f"{label}: [{bar}] {pct:.1f}% ({completed}/{total})")


def _display_metrics_profile(
    profile_data: list[tuple[int, dict]], t_extract: float, t_write: float
):
    """Display profiling results for metrics extraction bottleneck analysis.

    Args:
        profile_data: List of (job_id, timing_dict) tuples.
        t_extract: Total extraction phase time (seconds).
        t_write: Total database write time (seconds).
    """
    if not profile_data:
        return

    # Aggregate timing data
    parse_times = [t["parse_sec"] for _, t in profile_data]
    timing_times = [t["timing_sec"] for _, t in profile_data]

    total_parse = sum(parse_times)
    total_timing = sum(timing_times)

    # Find slowest 10 jobs
    slowest = sorted(profile_data, key=lambda x: x[1]["total_sec"], reverse=True)[:10]

    # Print results
    print("\n" + "=" * 80)
    print("📊 Metrics Extraction Performance Profile".center(80))
    print("=" * 80)

    print(f"\n✓ Extracted metrics for {len(profile_data)} jobs")
    print(f"  Total extraction time: {t_extract:.1f}s")
    print(f"  Total DB write time: {t_write:.1f}s")
    print(f"  Jobs/sec rate: {len(profile_data) / t_extract:.1f}")

    # Bottleneck analysis
    total_worker_time = total_parse + total_timing
    if total_worker_time > 0:
        parse_pct = 100 * total_parse / total_worker_time
        timing_pct = 100 * total_timing / total_worker_time

        print("\n⏱️  Bottleneck breakdown (per-job average):")
        print(
            f"  Parsing:        {parse_pct:>5.1f}% ({total_parse/len(profile_data):>6.3f}s/job)"
        )
        print(
            f"  Timing extract: {timing_pct:>5.1f}% ({total_timing/len(profile_data):>6.3f}s/job)"
        )

    # Slowest jobs
    if slowest:
        print("\n🐢 Slowest 10 jobs:")
        for job_id, timing in slowest:
            total = timing["total_sec"]
            parse = timing["parse_sec"]
            timing_sec = timing["timing_sec"]
            print(
                f"  Job {job_id:>6}: {total:.3f}s "
                f"(parse: {parse:.3f}s, timing: {timing_sec:.3f}s)"
            )

    print("\n" + "=" * 80)


def _extract_metrics_from_dir(
    job_dir: Path,
    unzip: bool = False,
    profile: bool = False,
    recompute: bool = False,
) -> dict:
    """Extract metrics from a job directory (pure I/O, no DB writes).

    This function is safe to call from worker threads. Uses the single-pass
    parser to read each ORCA output file only once (instead of 6 separate reads).

    Args:
        job_dir: Path to job directory containing ORCA output.
        unzip: If True, handle gzipped output files (quacc).
        profile: If True, collect timing data for bottleneck analysis.
        recompute: If True, skip cache and regenerate orca_metrics.json.

    Returns:
        Dictionary with extracted metrics and an 'error' key (None on success).
        If profile=True, includes '_profile' key with timing breakdown.
    """
    import time

    from ..utils.analysis import (
        GENERATOR_AVAILABLE,
        parse_generator_data,
        parse_job_metrics,
    )

    t0 = time.perf_counter()

    try:
        # Single call now extracts everything: metrics + timing + nprocs
        # (uses single-pass file reader internally)
        t_parse = time.perf_counter()
        metrics = parse_job_metrics(job_dir, unzip=unzip, recompute=recompute)
        t_parse = time.perf_counter() - t_parse

        cache_hit = metrics.pop("_cache_hit", False)

        result = {
            "max_forces": metrics.get("max_forces"),
            "scf_steps": metrics.get("scf_steps"),
            "final_energy": metrics.get("final_energy"),
            "wall_time": metrics.get("wall_time"),
            "n_cores": metrics.get("nprocs"),
            "error": None,
            "timing_warning": None,
            "_cache_hit": cache_hit,
            "generator_data": None,
            "sella_steps": metrics.get("sella_steps"),
            "sella_converged": metrics.get("sella_converged"),
        }

        if GENERATOR_AVAILABLE:
            result["generator_data"] = parse_generator_data(
                job_dir, recompute=recompute
            )

        if profile:
            result["_profile"] = {
                "parse_sec": t_parse,
                "timing_sec": 0.0,  # timing now included in parse phase
                "total_sec": time.perf_counter() - t0,
                "cache_hit": cache_hit,
            }

        return result

    except Exception as e:
        return {"error": str(e)}


def _parallel_extract_metrics(
    workflow: ArchitectorWorkflow,
    work_items: list[tuple[int, Path]],
    unzip: bool = False,
    verbose: bool = False,
    workers: int = 4,
    mark_failed_on_error: bool = True,
    profile: bool = False,
    recompute: bool = False,
) -> tuple[int, int]:
    """Extract metrics in parallel, then batch-write results to DB.

    File I/O happens in a thread pool; DB writes happen in a single
    transaction after all extractions complete.

    Args:
        workflow: ArchitectorWorkflow instance.
        work_items: List of (job_id, job_dir) tuples.
        unzip: If True, handle gzipped output files (quacc).
        verbose: Print detailed progress messages.
        workers: Number of parallel worker threads.
        mark_failed_on_error: If True, mark jobs as FAILED when parsing fails.
        profile: If True, collect and display performance profiling data.
        recompute: If True, skip cache and regenerate orca_metrics.json.

    Returns:
        Tuple of (extracted_count, failed_count).
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    try:
        from tqdm import tqdm

        use_tqdm = True
    except ImportError:
        use_tqdm = False

    if not work_items:
        return 0, 0

    # Phase 1: extract metrics in parallel (pure I/O, no DB)
    success_metrics: list[dict] = []
    failed_jobs: list[tuple[int, str]] = []
    profile_data: list[tuple[int, dict]] = []  # (job_id, timing_dict)
    cache_hits = 0

    t_extract_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_job = {
            executor.submit(
                _extract_metrics_from_dir, job_dir, unzip, profile, recompute
            ): (
                job_id,
                job_dir,
            )
            for job_id, job_dir in work_items
        }

        # Create progress bar if tqdm available
        futures = as_completed(future_to_job)
        if use_tqdm:
            futures = tqdm(
                futures,
                total=len(work_items),
                desc="Extracting metrics",
                unit="job",
                disable=verbose,  # Disable tqdm if verbose (conflicts with detailed output)
            )

        for future in futures:
            job_id, job_dir = future_to_job[future]
            result = future.result()

            if result.get("error") is None:
                if result.get("_cache_hit"):
                    cache_hits += 1
                metrics_entry = {
                    "job_id": job_id,
                    "job_dir": str(job_dir),
                    "max_forces": result["max_forces"],
                    "scf_steps": result["scf_steps"],
                    "final_energy": result["final_energy"],
                    "wall_time": result["wall_time"],
                    "n_cores": result["n_cores"],
                }
                if result.get("generator_data") is not None:
                    metrics_entry["generator_data"] = result["generator_data"]
                if result.get("sella_steps") is not None:
                    metrics_entry["sella_steps"] = result["sella_steps"]
                if result.get("sella_converged") is not None:
                    metrics_entry["sella_converged"] = result["sella_converged"]
                success_metrics.append(metrics_entry)
                if profile and "_profile" in result:
                    profile_data.append((job_id, result["_profile"]))
                if verbose:
                    print(
                        f"    Job {job_id}: forces={result['max_forces']}, "
                        f"scf={result['scf_steps']}, "
                        f"energy={result['final_energy']}"
                    )
                    if result.get("timing_warning"):
                        print(f"      Warning: {result['timing_warning']}")
            else:
                failed_jobs.append((job_id, result["error"]))
                if verbose:
                    print(
                        f"    Job {job_id}: failed to extract metrics: "
                        f"{result['error']}"
                    )

    t_extract = time.perf_counter() - t_extract_start

    # Print cache stats if any cache hits occurred
    if cache_hits > 0:
        total = len(success_metrics) + len(failed_jobs)
        pct = (cache_hits / total * 100) if total > 0 else 0
        print(
            f"Cache: {cache_hits} hits, {total - cache_hits} misses "
            f"({pct:.0f}% hit rate)"
        )

    # Phase 2: batch-write all successful metrics in one transaction
    if success_metrics:
        print(
            f"Writing metrics to database ({len(success_metrics)} jobs)...",
            flush=True,
        )
    t_write_start = time.perf_counter()
    if success_metrics:
        workflow.update_job_metrics_bulk(success_metrics)
    t_write = time.perf_counter() - t_write_start
    if success_metrics:
        print(f"Database write completed in {t_write:.1f}s")

    # Phase 3: batch-mark failed jobs in a single transaction
    if mark_failed_on_error and failed_jobs:
        print(f"Marking {len(failed_jobs)} failed jobs...", flush=True)
        cur = workflow.conn.cursor()
        for job_id, error_msg in failed_jobs:
            cur.execute(
                "UPDATE structures SET status = ?, error_message = ?, "
                "fail_count = COALESCE(fail_count, 0) + 1, "
                "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (JobStatus.FAILED.value, f"Metrics parse error: {error_msg}", job_id),
            )
        workflow._commit_with_retry()

    # Phase 4: display profiling results if requested
    if profile and profile_data:
        _display_metrics_profile(profile_data, t_extract, t_write)

    return len(success_metrics), len(failed_jobs)


def _check_single_job_status(
    job_id: int,
    orig_index: int,
    current_status: JobStatus,
    job_dir: Path,
    check_func: Callable,
) -> dict | None:
    """Check status of a single job (pure I/O, no DB writes).

    Returns:
        Dictionary with job_id, old_status, new_status, job_dir, marker_blocked,
        or None if job_dir does not exist.
    """
    if not job_dir.exists():
        return None

    # .do_not_rerun.json (written by clean.py --purge-failed) overrides all
    # other checks. Without this, a purged directory (which has no orca.out)
    # would be classified as RUNNING or TIMEOUT and could be cycled by submit.
    marker_blocked = is_marker_blocked(job_dir)
    if marker_blocked:
        new_status = JobStatus.FAILED
    else:
        result = check_func(str(job_dir))

        if result == 1:
            new_status = JobStatus.COMPLETED
        elif result == -2:
            new_status = JobStatus.TIMEOUT
        elif result == -1:
            new_status = JobStatus.FAILED
        else:
            new_status = JobStatus.RUNNING

    return {
        "job_id": job_id,
        "orig_index": orig_index,
        "old_status": current_status,
        "new_status": new_status,
        "job_dir": job_dir,
        "marker_blocked": marker_blocked,
    }


def _commit_status_changes(
    workflow: ArchitectorWorkflow,
    status_groups: dict[JobStatus, list[int]],
    marker_blocked_by_old_status: dict[JobStatus, list[int]],
) -> None:
    """Persist status transitions + marker-blocked transitions in one commit.

    Marker-blocked rows are written with `only_if_status=old_status` (CAS)
    so concurrent writers (submit_jobs, a second dashboard) that already
    flipped the row to FAILED do not cause a second fail_count increment.
    """
    additional: list[StatusGroupUpdate] = [
        {
            "job_ids": ids,
            "new_status": JobStatus.FAILED,
            "error_message": MARKER_ERROR_MESSAGE,
            "increment_fail_count": True,
            "only_if_status": old_status,
        }
        for old_status, ids in marker_blocked_by_old_status.items()
        if ids
    ]
    workflow.update_status_bulk_multi(status_groups, additional=additional or None)

    total = sum(len(ids) for ids in marker_blocked_by_old_status.values())
    if total:
        print(f"  Marker-blocked: {total} job(s) reverted to FAILED")


def _parallel_status_check(
    workflow: ArchitectorWorkflow,
    jobs: list,
    root_dir: Path,
    job_dir_pattern: str,
    check_func: Callable,
    verbose: bool,
    workers: int,
    extract_metrics: bool,
) -> tuple[dict, list]:
    """Check job statuses in parallel, then batch-update DB.

    Args:
        workflow: ArchitectorWorkflow instance.
        jobs: List of job objects to check.
        root_dir: Root directory containing job subdirectories.
        job_dir_pattern: Pattern for job directory names. Supports
            {hostname}, {orig_index}, and {id}.
        check_func: Status checking function.
        verbose: Print detailed progress messages.
        workers: Number of parallel worker threads.
        extract_metrics: If True, collect completed jobs for metrics extraction.

    Returns:
        Tuple of (updated_counts dict, completed_for_metrics list).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    try:
        from tqdm import tqdm

        use_tqdm = True
    except ImportError:
        use_tqdm = False

    # Phase 1: Check statuses in parallel (pure I/O, no DB)
    status_updates = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all jobs
        future_to_job = {}
        for job in jobs:
            job_dir_name = render_job_dir_pattern(
                job_dir_pattern,
                orig_index=job.orig_index,
                job_id=job.id,
            )
            job_dir = root_dir / job_dir_name

            future = executor.submit(
                _check_single_job_status,
                job.id,
                job.orig_index,
                job.status,
                job_dir,
                check_func,
            )
            future_to_job[future] = job

        # Collect results with progress bar
        futures = as_completed(future_to_job)
        if use_tqdm:
            futures = tqdm(
                futures,
                total=len(jobs),
                desc="Checking statuses",
                unit="job",
                disable=verbose,
            )

        for future in futures:
            result = future.result()
            if result is not None:
                status_updates.append(result)

    # Phase 2: Batch-update database — group by status for fewer commits
    updated_counts = {
        JobStatus.COMPLETED: 0,
        JobStatus.FAILED: 0,
        JobStatus.TIMEOUT: 0,
        JobStatus.RUNNING: 0,
    }
    completed_for_metrics = []

    # Group changed jobs by their new status. Marker-blocked jobs are
    # grouped by snapshot old_status so _commit_status_changes can use
    # only_if_status as a CAS guard against concurrent writers.
    status_groups: dict[JobStatus, list[int]] = {}
    marker_blocked_by_old_status: dict[JobStatus, list[int]] = {}
    for update in status_updates:
        if update["new_status"] != update["old_status"]:
            if update["marker_blocked"]:
                marker_blocked_by_old_status.setdefault(
                    update["old_status"], []
                ).append(update["job_id"])
            else:
                status_groups.setdefault(update["new_status"], []).append(
                    update["job_id"]
                )
            updated_counts[update["new_status"]] += 1

            if verbose:
                tag = " [marker-blocked]" if update["marker_blocked"] else ""
                print(
                    f"  Job {update['job_id']}: {update['old_status'].value} -> {update['new_status'].value}{tag}"
                )

            if extract_metrics and update["new_status"] == JobStatus.COMPLETED:
                completed_for_metrics.append((update["job_id"], update["job_dir"]))

    _commit_status_changes(workflow, status_groups, marker_blocked_by_old_status)

    return updated_counts, completed_for_metrics


def _sequential_status_check(
    workflow: ArchitectorWorkflow,
    jobs: list,
    root_dir: Path,
    job_dir_pattern: str,
    check_func: Callable,
    verbose: bool,
    extract_metrics: bool,
) -> tuple[dict, list]:
    """Check job statuses sequentially (fallback for debugging).

    Args:
        workflow: ArchitectorWorkflow instance.
        jobs: List of job objects to check.
        root_dir: Root directory containing job subdirectories.
        job_dir_pattern: Pattern for job directory names. Supports
            {hostname}, {orig_index}, and {id}.
        check_func: Status checking function.
        verbose: Print detailed progress messages.
        extract_metrics: If True, collect completed jobs for metrics extraction.

    Returns:
        Tuple of (updated_counts dict, completed_for_metrics list).
    """
    try:
        from tqdm import tqdm

        use_tqdm = True
    except ImportError:
        use_tqdm = False

    updated_counts = {
        JobStatus.COMPLETED: 0,
        JobStatus.FAILED: 0,
        JobStatus.TIMEOUT: 0,
        JobStatus.RUNNING: 0,
    }
    completed_for_metrics = []

    # Wrap jobs iterator with tqdm if available
    jobs_iter = jobs
    if use_tqdm:
        jobs_iter = tqdm(
            jobs,
            desc="Updating statuses",
            unit="job",
            disable=verbose,
        )

    # Collect all status changes, then batch-write at the end.
    # Marker-blocked rows are grouped by snapshot old_status so the
    # commit helper can issue CAS-guarded updates.
    status_groups: dict[JobStatus, list[int]] = {}
    marker_blocked_by_old_status: dict[JobStatus, list[int]] = {}

    for i, job in enumerate(jobs_iter):
        # Format job directory name
        job_dir_name = render_job_dir_pattern(
            job_dir_pattern,
            orig_index=job.orig_index,
            job_id=job.id,
        )
        job_dir = root_dir / job_dir_name

        if not job_dir.exists():
            continue

        # .do_not_rerun.json marker overrides other checks (see
        # _check_single_job_status for rationale).
        marker_blocked = is_marker_blocked(job_dir)
        if marker_blocked:
            new_status = JobStatus.FAILED
        else:
            result = check_func(str(job_dir))

            if result == 1:
                new_status = JobStatus.COMPLETED
            elif result == -2:
                new_status = JobStatus.TIMEOUT
            elif result == -1:
                new_status = JobStatus.FAILED
            else:
                new_status = JobStatus.RUNNING

        # Track if changed
        if new_status != job.status:
            if marker_blocked:
                marker_blocked_by_old_status.setdefault(job.status, []).append(job.id)
            else:
                status_groups.setdefault(new_status, []).append(job.id)
            updated_counts[new_status] += 1

            if verbose:
                tag = " [marker-blocked]" if marker_blocked else ""
                print(
                    f"  [{i+1}/{len(jobs)}] Job {job.id}: {job.status.value} -> {new_status.value}{tag}"
                )

            if extract_metrics and new_status == JobStatus.COMPLETED:
                completed_for_metrics.append((job.id, job_dir))

    _commit_status_changes(workflow, status_groups, marker_blocked_by_old_status)

    return updated_counts, completed_for_metrics


def backfill_metrics(
    workflow: ArchitectorWorkflow,
    root_dir: str | Path,
    job_dir_pattern: str = DEFAULT_JOB_DIR_PATTERN,
    unzip: bool = False,
    verbose: bool = False,
    workers: int = 4,
    recompute: bool = False,
    max_jobs: int | None = None,
    profile: bool = False,
):
    """Extract metrics for completed jobs that don't have them yet.

    Args:
        workflow: ArchitectorWorkflow instance.
        root_dir: Root directory containing job subdirectories.
        job_dir_pattern: Pattern for job directory names. Supports
            {hostname}, {orig_index}, and {id}.
        unzip: If True, handle gzipped output files (quacc).
        verbose: Print detailed progress messages.
        workers: Number of parallel worker threads for extraction.
        recompute: If True, recompute metrics for all completed jobs, even those
            that already have metrics.
        max_jobs: If set, limit to this many jobs (useful for debugging).
        profile: If True, collect and display performance profiling data.
    """
    root_dir = Path(root_dir)

    limit_clause = f" LIMIT {int(max_jobs)}" if max_jobs is not None else ""

    from ..utils.analysis import GENERATOR_AVAILABLE

    if recompute:
        cur = workflow._execute_with_retry(
            f"SELECT id, orig_index FROM structures WHERE status = 'completed'{limit_clause}"
        )
    else:
        # Include jobs missing standard metrics OR (when available) missing qtaim data.
        if GENERATOR_AVAILABLE:
            missing_clause = "max_forces IS NULL OR generator_data IS NULL"
        else:
            missing_clause = "max_forces IS NULL"
        cur = workflow._execute_with_retry(
            f"""
            SELECT id, orig_index
            FROM structures
            WHERE status = 'completed' AND ({missing_clause}){limit_clause}
            """
        )
    rows = cur.fetchall()

    if not rows:
        print("\nAll completed jobs already have metrics.")
        return

    if max_jobs is not None:
        print(
            f"\n[debug] Limiting metrics extraction to {len(rows)} jobs (--debug {max_jobs})"
        )

    # Build work items, filtering out missing directories
    work_items = []
    skipped = 0
    for job_id, orig_index in rows:
        job_dir_name = render_job_dir_pattern(
            job_dir_pattern,
            orig_index=orig_index,
            job_id=job_id,
        )
        job_dir = root_dir / job_dir_name

        if not job_dir.exists():
            skipped += 1
            if verbose:
                print(f"  Job {job_id}: directory {job_dir} not found, skipping")
            continue

        work_items.append((job_id, job_dir))

    print(
        f"\nBackfilling metrics for {len(work_items)} completed jobs "
        f"({workers} workers)..."
    )

    extracted, failed = _parallel_extract_metrics(
        workflow,
        work_items,
        unzip=unzip,
        verbose=verbose,
        workers=workers,
        profile=profile,
        recompute=recompute,
    )

    print(
        f"Backfill metrics: {extracted} extracted, "
        f"{failed} failed, {skipped} skipped (no directory)"
    )


def reset_missing_jobs(
    workflow: ArchitectorWorkflow,
    root_dir: str | Path,
    job_dir_pattern: str = DEFAULT_JOB_DIR_PATTERN,
    statuses: list[JobStatus] | None = None,
) -> int:
    """Find jobs whose directories don't exist and reset them to TO_RUN.

    Scans jobs in the given statuses, checks if the expected job directory
    exists on disk, and resets any missing ones. Increments fail_count.

    Args:
        workflow: ArchitectorWorkflow instance.
        root_dir: Root directory containing job subdirectories.
        job_dir_pattern: Pattern for job directory names. Supports
            {hostname}, {orig_index}, and {id}.
        statuses: Job statuses to check. Defaults to RUNNING, FAILED, TIMEOUT.

    Returns:
        Number of jobs reset.
    """
    if statuses is None:
        statuses = [JobStatus.RUNNING, JobStatus.FAILED, JobStatus.TIMEOUT]

    root_dir = Path(root_dir)
    jobs = workflow.get_jobs_by_status(statuses, include_geometry=False)

    reset_count = 0
    for job in jobs:
        job_dir_name = render_job_dir_pattern(
            job_dir_pattern,
            orig_index=job.orig_index,
            job_id=job.id,
        )
        job_dir = root_dir / job_dir_name

        if not job_dir.exists():
            workflow.update_status(
                job.id,
                JobStatus.TO_RUN,
                error_message="Directory missing — reset by --reset-missing",
                increment_fail_count=True,
            )
            reset_count += 1

    return reset_count


# Status code -> JobStatus mapping for check_job_termination results
_DISK_STATUS_MAP: dict[int, JobStatus] = {
    1: JobStatus.COMPLETED,
    -1: JobStatus.FAILED,
    -2: JobStatus.TIMEOUT,
    0: JobStatus.TO_RUN,  # Ambiguous -> safe default to keep things moving
}


def _probe_unlinked_job(
    job_dir: Path,
    hours_cutoff: float,
) -> tuple[bool, int, str | None]:
    """Check if a job directory exists and revalidate status from disk.

    Pure I/O -- no DB access. Safe to call from a thread pool.

    Args:
        job_dir: Expected job directory path.
        hours_cutoff: Hours before considering a job timed out.

    Returns:
        (dir_exists, disk_status_code, error_msg)
    """
    if not job_dir.is_dir():
        return False, 0, None

    disk_status_code = check_job_termination(str(job_dir), hours_cutoff=hours_cutoff)

    error_msg = None
    if disk_status_code == -1:
        # Find the .out file for parse_failure_reason
        out_files = [
            f
            for f in job_dir.iterdir()
            if f.suffix == ".out" or f.name.endswith(".out.gz")
        ]
        if out_files:
            error_msg = parse_failure_reason(str(out_files[0]))

    return True, disk_status_code, error_msg


def fix_unlinked_jobs(
    workflow: ArchitectorWorkflow,
    root_dir: str | Path,
    job_dir_pattern: str = DEFAULT_JOB_DIR_PATTERN,
    hours_cutoff: float = 24.0,
    verbose: bool = False,
    max_jobs: int | None = None,
    max_retries: int | None = None,
    workers: int = 4,
) -> dict[str, int]:
    """Repair NULL job_dir entries by auto-linking or resetting.

    For each job where job_dir is NULL (excluding running jobs):
    1. Try to find a matching directory using job_dir_pattern
    2. If found: link it and revalidate status from disk
    3. If not found: reset to TO_RUN

    Disk is the source of truth -- DB status is always updated to match.
    Directory probing runs in parallel with ThreadPoolExecutor for
    performance on network filesystems (Lustre/GPFS).

    Args:
        workflow: ArchitectorWorkflow instance.
        root_dir: Root directory containing job subdirectories.
        job_dir_pattern: Pattern for job directory names. Supports
            {hostname}, {orig_index}, and {id}.
        hours_cutoff: Hours before considering a job timed out.
        verbose: Print per-job details.
        max_jobs: Limit to N jobs (for --debug).
        max_retries: Only reset jobs with fail_count < this value.
        workers: Number of parallel threads for directory probing.

    Returns:
        Summary dict with counts: linked, reset, skipped.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    try:
        from tqdm import tqdm

        use_tqdm = True
    except ImportError:
        use_tqdm = False

    root = Path(root_dir)
    counts = {"linked": 0, "reset": 0, "skipped": 0}

    # Query all non-running jobs with NULL job_dir
    all_statuses = [
        JobStatus.TO_RUN,
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.TIMEOUT,
    ]
    jobs = workflow.get_jobs_by_status(all_statuses, include_geometry=False)
    unlinked = [j for j in jobs if j.job_dir is None]

    if max_jobs is not None:
        unlinked = unlinked[:max_jobs]

    if not unlinked:
        print("No unlinked jobs found (all jobs have job_dir set).")
        return counts

    print(f"Found {len(unlinked)} jobs with NULL job_dir")

    # Phase 1: Probe directories in parallel (pure I/O, no DB)
    # Build job_id -> (job, job_dir) mapping for the probe
    probe_items: list[tuple[int, Path, int, int | None]] = []
    for job in unlinked:
        job_dir_name = render_job_dir_pattern(
            job_dir_pattern,
            orig_index=job.orig_index,
            job_id=job.id,
        )
        probe_items.append(
            (job.id, root / job_dir_name, job.orig_index, job.fail_count)
        )

    # Run probes in parallel
    probe_results: dict[int, tuple[bool, int, str | None, Path]] = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_job = {
            executor.submit(_probe_unlinked_job, job_dir, hours_cutoff): (
                job_id,
                job_dir,
                orig_index,
                fail_count,
            )
            for job_id, job_dir, orig_index, fail_count in probe_items
        }

        futures = as_completed(future_to_job)
        if use_tqdm:
            futures = tqdm(
                futures,
                total=len(probe_items),
                desc="Probing directories",
                unit="job",
                disable=verbose,
            )

        for future in futures:
            job_id, job_dir, orig_index, fail_count = future_to_job[future]
            dir_exists, disk_status_code, error_msg = future.result()
            probe_results[job_id] = (dir_exists, disk_status_code, error_msg, job_dir)

    # Phase 2: Build updates from probe results (single-threaded)
    updates: list[tuple[int, str | None, JobStatus, str | None, bool]] = []

    for job_id, job_dir, orig_index, fail_count in probe_items:
        dir_exists, disk_status_code, error_msg, _ = probe_results[job_id]

        if dir_exists:
            new_status = _DISK_STATUS_MAP.get(disk_status_code, JobStatus.TO_RUN)
            updates.append((job_id, str(job_dir), new_status, error_msg, False))
            counts["linked"] += 1

            if verbose:
                print(
                    f"  Link job {job_id} (orig_index={orig_index}) "
                    f"-> {job_dir.name} [{new_status.value}]"
                )
        else:
            # No directory found -- check max_retries before resetting
            if max_retries is not None and (fail_count or 0) >= max_retries:
                counts["skipped"] += 1
                if verbose:
                    print(
                        f"  Skip job {job_id} (orig_index={orig_index}) "
                        f"-- fail_count {fail_count} >= max_retries {max_retries}"
                    )
                continue

            updates.append(
                (
                    job_id,
                    None,
                    JobStatus.TO_RUN,
                    "No directory found -- reset by --fix-unlinked",
                    True,  # increment_fail_count
                )
            )
            counts["reset"] += 1

            if verbose:
                print(
                    f"  Reset job {job_id} (orig_index={orig_index}) "
                    f"-> to_run (no directory)"
                )

    # Phase 3: Batch commit all updates (single commit for Lustre).
    # Uses _execute_with_retry for lock safety on network filesystems.
    for job_id, job_dir_str, new_status, error_msg, inc_fail in updates:
        if job_dir_str is not None:
            set_clauses = ["job_dir = ?"]
            values: list[str | int] = [job_dir_str]
            set_clauses.append("status = ?")
            values.append(new_status.value)
            if error_msg is not None:
                set_clauses.append("error_message = ?")
                values.append(error_msg)
            query = f"UPDATE structures SET {', '.join(set_clauses)} WHERE id = ?"
            values.append(job_id)
            workflow._execute_with_retry(query, tuple(values))
        else:
            set_clauses = ["status = ?", "error_message = ?"]
            values = [new_status.value, error_msg or ""]
            if inc_fail:
                set_clauses.append("fail_count = COALESCE(fail_count, 0) + 1")
            query = f"UPDATE structures SET {', '.join(set_clauses)} WHERE id = ?"
            values.append(job_id)
            workflow._execute_with_retry(query, tuple(values))

    workflow._commit_with_retry()

    return counts


def update_all_statuses(
    workflow: ArchitectorWorkflow,
    root_dir: str | Path,
    job_dir_pattern: str = DEFAULT_JOB_DIR_PATTERN,
    check_func: Callable | None = None,
    verbose: bool = False,
    extract_metrics: bool = False,
    unzip: bool = False,
    workers: int = 4,
    recheck_completed: bool = False,
    hours_cutoff: float = 24,
    parallel_status_check: bool = True,
    max_jobs: int | None = None,
    profile: bool = False,
):
    """Scan job directories and update statuses in bulk.

    Args:
        workflow: ArchitectorWorkflow instance.
        root_dir: Root directory containing job subdirectories.
        job_dir_pattern: Pattern for job directory names. Supports
            {hostname}, {orig_index}, and {id}.
        check_func: Optional custom status checking function.
        verbose: Print detailed progress messages.
        extract_metrics: If True, extract computational metrics for newly completed jobs.
        unzip: If True, handle gzipped output files (quacc).
        workers: Number of parallel worker threads for status checking and metrics extraction.
        recheck_completed: If True, also re-verify jobs marked as completed.
        hours_cutoff: Hours of inactivity before a job is considered timed out.
        parallel_status_check: If True, parallelize status checking (default: True for scalability).
        max_jobs: If set, limit operations to this many jobs (useful for debugging).
        profile: If True, collect and display performance profiling data for metrics extraction.
    """
    from functools import partial

    if check_func is None:
        check_func = partial(check_job_termination, hours_cutoff=hours_cutoff)

    root_dir = Path(root_dir)
    if not root_dir.exists():
        print(f"Error: root directory {root_dir} does not exist")
        return

    # Get jobs to check — optionally include completed for re-verification
    statuses_to_check = [
        JobStatus.RUNNING,
        JobStatus.FAILED,
        JobStatus.TIMEOUT,
        JobStatus.TO_RUN,
    ]
    if recheck_completed:
        statuses_to_check.append(JobStatus.COMPLETED)

    jobs = workflow.get_jobs_by_status(statuses_to_check, limit=max_jobs)

    if max_jobs is not None:
        print(f"\n[debug] Limiting to {len(jobs)} jobs (--debug {max_jobs})")

    if verbose:
        print(f"\nScanning {len(jobs)} jobs for status updates...")

    if parallel_status_check:
        # Parallel implementation (scalable for large job counts)
        updated_counts, completed_for_metrics = _parallel_status_check(
            workflow,
            jobs,
            root_dir,
            job_dir_pattern,
            check_func,
            verbose,
            workers,
            extract_metrics,
        )
    else:
        # Sequential implementation (fallback for debugging)
        updated_counts, completed_for_metrics = _sequential_status_check(
            workflow,
            jobs,
            root_dir,
            job_dir_pattern,
            check_func,
            verbose,
            extract_metrics,
        )

    # Print update summary
    print(
        f"\nStatus updates: {updated_counts[JobStatus.COMPLETED]} completed, "
        f"{updated_counts[JobStatus.FAILED]} failed, "
        f"{updated_counts[JobStatus.TIMEOUT]} timeout, "
        f"{updated_counts[JobStatus.RUNNING]} running"
    )

    # Extract metrics in parallel for newly completed jobs
    if completed_for_metrics:
        print(
            f"Extracting metrics for {len(completed_for_metrics)} newly completed "
            f"jobs ({workers} workers)..."
        )
        metrics_extracted, metrics_failed = _parallel_extract_metrics(
            workflow,
            completed_for_metrics,
            unzip=unzip,
            verbose=verbose,
            workers=workers,
            profile=profile,
        )
        print(
            f"Metrics extraction: {metrics_extracted} succeeded, "
            f"{metrics_failed} failed"
        )


def show_failed_jobs(workflow: ArchitectorWorkflow, limit: int = 20):
    """Display details of failed jobs."""
    failed = workflow.get_jobs_by_status(JobStatus.FAILED)

    if not failed:
        print("\nNo failed jobs found.")
        return

    print_header(f"Failed Jobs (showing up to {limit})")
    print(f"\n{'ID':<8} {'Orig Index':<12} {'Fails':<6} {'Job Dir':<25} {'Error':<25}")
    print("-" * 90)

    for job in failed[:limit]:
        job_dir = (
            (job.job_dir[:22] + "...")
            if job.job_dir and len(job.job_dir) > 25
            else (job.job_dir or "N/A")
        )
        error = (
            (job.error_message[:22] + "...")
            if job.error_message and len(job.error_message) > 25
            else (job.error_message or "N/A")
        )
        print(
            f"{job.id:<8} {job.orig_index:<12} {job.fail_count:<6} {job_dir:<25} {error:<25}"
        )

    if len(failed) > limit:
        print(f"\n... and {len(failed) - limit} more failed jobs")


def show_timeout_jobs(workflow: ArchitectorWorkflow, limit: int = 20):
    """Display details of timed out jobs."""
    timeout = workflow.get_jobs_by_status(JobStatus.TIMEOUT)

    if not timeout:
        print("\nNo timeout jobs found.")
        return

    print_header(f"Timeout Jobs (showing up to {limit})")
    print(f"\n{'ID':<8} {'Orig Index':<12} {'Fails':<6} {'Job Dir':<25} {'Error':<25}")
    print("-" * 90)

    for job in timeout[:limit]:
        job_dir = (
            (job.job_dir[:22] + "...")
            if job.job_dir and len(job.job_dir) > 25
            else (job.job_dir or "N/A")
        )
        error = (
            (job.error_message[:22] + "...")
            if job.error_message and len(job.error_message) > 25
            else (job.error_message or "N/A")
        )
        print(
            f"{job.id:<8} {job.orig_index:<12} {job.fail_count:<6} {job_dir:<25} {error:<25}"
        )

    if len(timeout) > limit:
        print(f"\n... and {len(timeout) - limit} more timeout jobs")


def show_ready_jobs(workflow: ArchitectorWorkflow, limit: int = 20):
    """Display jobs that are ready to run."""
    ready = workflow.get_jobs_by_status(JobStatus.TO_RUN)

    if not ready:
        print("\nNo jobs ready to run.")
        return

    print_header(f"Ready Jobs (showing up to {limit})")
    print(f"\n{'ID':<8} {'Orig Index':<12} {'N Atoms':<10}")
    print("-" * 60)

    for job in ready[:limit]:
        print(f"{job.id:<8} {job.orig_index:<12} {job.natoms:<10}")

    if len(ready) > limit:
        print(f"\n... and {len(ready) - limit} more ready jobs")


def show_running_jobs(workflow: ArchitectorWorkflow, limit: int = 20):
    """Display jobs that are currently running."""
    running = workflow.get_jobs_by_status(JobStatus.RUNNING)

    if not running:
        print("\nNo jobs currently running.")
        return

    print_header(f"Running Jobs (showing up to {limit})")
    print(f"\n{'ID':<8} {'Orig Index':<12} {'N Atoms':<10} {'Worker ID':<20}")
    print("-" * 60)

    for job in running[:limit]:
        wid = job.worker_id or "N/A"
        print(f"{job.id:<8} {job.orig_index:<12} {job.natoms:<10} {wid:<20}")

    if len(running) > limit:
        print(f"\n... and {len(running) - limit} more running jobs")


def recover_orphaned_jobs(
    workflow: ArchitectorWorkflow,
    scheduler: str,
    hours_cutoff: float = 24.0,
    verbose: bool = False,
) -> dict[str, int]:
    """Detect and recover jobs orphaned by dead scheduler allocations.

    Queries the scheduler for active jobs, then checks RUNNING jobs whose
    worker_id is no longer in the active set. For each orphan, checks the
    output file on disk (via job_dir from the DB) to determine the correct
    status:
    - Completed on disk -> mark COMPLETED
    - Failed on disk -> mark FAILED
    - Inconclusive (no output or partial) -> reset to TO_RUN

    Content-based checks always take priority (a completed job is never reset).

    Args:
        workflow: ArchitectorWorkflow instance.
        scheduler: Scheduler type ("slurm" or "flux").
        hours_cutoff: Hours threshold for timeout detection.
        verbose: Print per-job details.

    Returns:
        Dict with counts: {"recovered": N, "completed": N, "failed": N,
        "reset": N, "dead_jobs": N, "skipped": N}.
    """
    from ..utils.scheduler import get_active_scheduler_jobs

    # Step 1: Get all RUNNING jobs with a worker_id
    running = workflow.get_jobs_by_status(JobStatus.RUNNING)
    tracked = [j for j in running if j.worker_id is not None]

    if not tracked:
        print("No RUNNING jobs with worker_id found -- nothing to recover.")
        return {
            "recovered": 0,
            "completed": 0,
            "failed": 0,
            "reset": 0,
            "dead_jobs": 0,
            "skipped": 0,
        }

    unique_workers = {j.worker_id for j in tracked}
    print(
        f"Found {len(tracked)} RUNNING jobs across {len(unique_workers)} "
        f"scheduler job(s)"
    )

    # Step 2: Query scheduler for active jobs (single call)
    active_jobs = get_active_scheduler_jobs(scheduler)
    if active_jobs is None:
        print(
            f"Cannot reach {scheduler} scheduler -- skipping orphan recovery "
            "(conservative: no jobs modified)"
        )
        return {
            "recovered": 0,
            "completed": 0,
            "failed": 0,
            "reset": 0,
            "dead_jobs": 0,
            "skipped": len(tracked),
        }

    # Step 3: Find dead scheduler jobs
    dead_workers = unique_workers - active_jobs
    if not dead_workers:
        print("All scheduler jobs are still active -- no orphans detected.")
        return {
            "recovered": 0,
            "completed": 0,
            "failed": 0,
            "reset": 0,
            "dead_jobs": 0,
            "skipped": 0,
        }

    orphans = [j for j in tracked if j.worker_id in dead_workers]
    print(
        f"Detected {len(dead_workers)} dead scheduler job(s) with "
        f"{len(orphans)} orphaned molecule(s)"
    )

    # Step 4: Check each orphan on disk, classify into batches
    completed_ids: list[int] = []
    failed_updates: list[tuple[int, str]] = []  # (job_id, error_msg)
    reset_ids: list[int] = []

    for job in orphans:
        job_dir = job.job_dir
        if not job_dir:
            reset_ids.append(job.id)
            if verbose:
                print(f"  Job {job.id}: no job_dir -- reset to TO_RUN")
            continue

        # Check output files on disk (content-based, preserves content > age rule)
        disk_status = check_job_termination(job_dir, hours_cutoff=hours_cutoff)

        if disk_status == 1:
            completed_ids.append(job.id)
            if verbose:
                print(f"  Job {job.id}: completed on disk -- marked COMPLETED")
        elif disk_status == -1:
            # Find the output file for error extraction
            error_msg = None
            try:
                from ..utils.status import pull_log_file

                log_file = pull_log_file(job_dir)
                error_msg = parse_failure_reason(log_file)
            except (FileNotFoundError, Exception):
                pass
            failed_updates.append((job.id, error_msg or "Orphaned job failed on disk"))
            if verbose:
                print(f"  Job {job.id}: failed on disk -- marked FAILED")
        else:
            reset_ids.append(job.id)
            if verbose:
                print(f"  Job {job.id}: inconclusive on disk -- reset to TO_RUN")

    # Step 5: Batch DB writes (minimizes commits on Lustre)
    if completed_ids:
        workflow.update_status_bulk(completed_ids, JobStatus.COMPLETED, worker_id=None)
    if reset_ids:
        workflow.update_status_bulk(
            reset_ids,
            JobStatus.TO_RUN,
            worker_id=None,
            increment_fail_count=True,
        )
    # Failed jobs have per-job error messages so they can't use update_status_bulk.
    # Execute all UPDATEs first, then commit once to minimize Lustre fsync cost.
    for job_id, error_msg in failed_updates:
        workflow._execute_with_retry(
            "UPDATE structures SET status = ?, error_message = ?, "
            "fail_count = COALESCE(fail_count, 0) + 1, worker_id = NULL, "
            "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (JobStatus.FAILED.value, error_msg, job_id),
        )
    if failed_updates:
        workflow._commit_with_retry()

    completed_count = len(completed_ids)
    failed_count = len(failed_updates)
    reset_count = len(reset_ids)

    total = completed_count + failed_count + reset_count
    print(
        f"\nRecovered {total} orphaned jobs from {len(dead_workers)} "
        f"dead scheduler job(s):"
    )
    print(f"  Completed on disk: {completed_count}")
    print(f"  Failed on disk: {failed_count}")
    print(f"  Reset to TO_RUN: {reset_count}")

    return {
        "recovered": total,
        "completed": completed_count,
        "failed": failed_count,
        "reset": reset_count,
        "dead_jobs": len(dead_workers),
        "skipped": 0,
    }


def main():
    """Main dashboard entry point."""
    parser = argparse.ArgumentParser(
        description="Dashboard for monitoring architector workflow status"
    )
    parser.add_argument("db_path", help="Path to workflow SQLite database")
    parser.add_argument(
        "--update",
        metavar="ROOT_DIR",
        help="Scan job directories and update statuses",
    )
    parser.add_argument(
        "--job-dir-pattern",
        default=DEFAULT_JOB_DIR_PATTERN,
        help=(
            "Pattern for job directory names. Supports {hostname}, "
            "{orig_index}, and {id} (default: "
            f"{DEFAULT_JOB_DIR_PATTERN})"
        ),
    )
    parser.add_argument(
        "--job-prefix",
        default=None,
        help=(
            "Optional stable prefix to prepend to job directories, for example "
            "'campaignA' -> campaignA_job_{orig_index}. Use the same prefix across "
            "coordinator requeues to keep scanning the same job directories."
        ),
    )
    parser.add_argument(
        "--show-failed",
        action="store_true",
        help="Show details of failed jobs",
    )
    parser.add_argument(
        "--show-timeout",
        action="store_true",
        help="Show details of timeout jobs",
    )
    parser.add_argument(
        "--show-ready",
        action="store_true",
        help="Show jobs ready to run",
    )
    parser.add_argument(
        "--show-running",
        action="store_true",
        help="Show jobs currently running",
    )
    parser.add_argument(
        "--reset-failed",
        action="store_true",
        help="Reset all failed jobs to ready status",
    )
    parser.add_argument(
        "--reset-timeout",
        action="store_true",
        help="Reset all timeout jobs to ready status",
    )
    parser.add_argument(
        "--reset-missing",
        metavar="ROOT_DIR",
        default=None,
        help="Reset jobs whose directories no longer exist back to TO_RUN. Requires root directory path.",
    )
    parser.add_argument(
        "--fix-unlinked",
        metavar="ROOT_DIR",
        default=None,
        help="Repair jobs with NULL job_dir: auto-link to directories or reset to TO_RUN. Requires root directory path.",
    )
    parser.add_argument(
        "--include-timeout-in-reset",
        action="store_true",
        help="When using --reset-failed, also reset timeout jobs",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="When resetting failed jobs, only reset those with fail_count < this value",
    )
    parser.add_argument(
        "--show-chronic-failures",
        type=int,
        metavar="N",
        help="Show jobs that have failed at least N times",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Limit number of jobs to display (default: 20)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output during updates",
    )
    parser.add_argument(
        "--show-metrics",
        action="store_true",
        help="Show computational metrics (forces, SCF steps)",
    )
    parser.add_argument(
        "--extract-metrics",
        action="store_true",
        help="Extract metrics (forces, SCF steps, energy, timing) for completed jobs during --update",
    )
    parser.add_argument(
        "--unzip",
        action="store_true",
        help="Handle gzipped output files (e.g., from quacc)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for metrics extraction (default: 4)",
    )
    parser.add_argument(
        "--recheck-completed",
        action="store_true",
        help="Re-verify completed jobs during --update (catches tampered outputs or status checker changes)",
    )
    parser.add_argument(
        "--hours-cutoff",
        type=float,
        default=24,
        help="Hours of inactivity before a job is considered timed out (default: 24)",
    )
    parser.add_argument(
        "--recompute-metrics",
        action="store_true",
        help="Recompute metrics for all completed jobs, even those that already have them",
    )
    parser.add_argument(
        "--recover-orphans",
        action="store_true",
        help="Detect and recover jobs orphaned by dead scheduler allocations. "
        "Checks if worker_id scheduler jobs are still active; resets orphans "
        "to TO_RUN (or marks COMPLETED/FAILED based on disk output).",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["slurm", "flux"],
        default=None,
        help="Scheduler type for --recover-orphans (required with --recover-orphans)",
    )
    parser.add_argument(
        "--debug",
        type=int,
        metavar="N",
        default=None,
        help="Limit status checks and metrics extraction to N jobs (for testing changes)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile metrics extraction to identify bottlenecks (I/O, parsing, DB)",
    )
    parser.add_argument(
        "--sella-view",
        choices=["summary", "running"],
        default=None,
        help=(
            "Sella-specific view. 'summary' prints convergence tri-state "
            "counts + step statistics + non-converged list. 'running' "
            "tail-reads sella.log for each running job to show the current "
            "step. Requires --update <root_dir> for 'running'."
        ),
    )

    add_wandb_args(parser)

    args = parser.parse_args()

    try:
        effective_job_dir_pattern = apply_job_dir_prefix(
            args.job_dir_pattern, args.job_prefix
        )
    except ValueError as exc:
        parser.error(str(exc))

    # Open workflow database
    try:
        workflow = ArchitectorWorkflow(args.db_path)
    except FileNotFoundError:
        print(f"Error: Database not found at {args.db_path}")
        sys.exit(1)

    # Initialize W&B run if requested
    wandb_run = None
    if args.wandb_project:
        if not WANDB_AVAILABLE:
            print(
                "Warning: wandb not installed; --wandb-project ignored. pip install wandb"
            )
        else:
            wandb_run = init_wandb_run(
                project=args.wandb_project,
                run_name=args.wandb_run_name or Path(args.db_path).stem,
                run_id=args.wandb_run_id,
            )

    # Update statuses if requested
    if args.update:
        update_all_statuses(
            workflow,
            args.update,
            job_dir_pattern=effective_job_dir_pattern,
            verbose=args.verbose,
            extract_metrics=args.extract_metrics,
            unzip=args.unzip,
            workers=args.workers,
            recheck_completed=args.recheck_completed,
            hours_cutoff=args.hours_cutoff,
            max_jobs=args.debug,
            profile=args.profile,
        )

        # Backfill metrics for previously completed jobs missing them
        if args.extract_metrics or args.recompute_metrics:
            backfill_metrics(
                workflow,
                args.update,
                job_dir_pattern=effective_job_dir_pattern,
                unzip=args.unzip,
                verbose=args.verbose,
                workers=args.workers,
                recompute=args.recompute_metrics,
                max_jobs=args.debug,
                profile=args.profile,
            )

    # Log campaign snapshot to W&B if a run is active
    if wandb_run is not None:
        _counts = workflow.count_by_status()
        _total = sum(_counts.values())
        _stats = compute_metrics_stats(workflow)
        log_campaign_snapshot(wandb_run, _counts, _total, _stats)

    # Reset failed jobs if requested
    if args.reset_failed:
        failed_jobs = workflow.get_jobs_by_status(JobStatus.FAILED)
        if args.max_retries is not None:
            eligible = [j for j in failed_jobs if j.fail_count < args.max_retries]
            skipped = len(failed_jobs) - len(eligible)
            workflow.reset_failed_jobs(
                max_fail_count=args.max_retries,
                include_timeout=args.include_timeout_in_reset,
            )
            status_msg = "failed/timeout" if args.include_timeout_in_reset else "failed"
            print(
                f"\nReset {len(eligible)} {status_msg} jobs to ready status "
                f"(skipped {skipped} jobs that have already failed {args.max_retries}+ times)"
            )
        else:
            workflow.reset_failed_jobs(include_timeout=args.include_timeout_in_reset)
            status_msg = "failed/timeout" if args.include_timeout_in_reset else "failed"
            print(f"\nReset {len(failed_jobs)} {status_msg} jobs to ready status")

    # Reset timeout jobs if requested
    if args.reset_timeout:
        timeout_jobs = workflow.get_jobs_by_status(JobStatus.TIMEOUT)
        if args.max_retries is not None:
            eligible = [j for j in timeout_jobs if j.fail_count < args.max_retries]
            skipped = len(timeout_jobs) - len(eligible)
            workflow.reset_timeout_jobs(max_fail_count=args.max_retries)
            print(
                f"\nReset {len(eligible)} timeout jobs to ready status "
                f"(skipped {skipped} jobs that have already timed out {args.max_retries}+ times)"
            )
        else:
            workflow.reset_timeout_jobs()
            print(f"\nReset {len(timeout_jobs)} timeout jobs to ready status")

    # Reset jobs with missing directories if requested
    if args.reset_missing:
        count = reset_missing_jobs(
            workflow,
            args.reset_missing,
            job_dir_pattern=effective_job_dir_pattern,
        )
        print(f"\nReset {count} jobs with missing directories to TO_RUN")

    # Fix unlinked jobs (NULL job_dir) if requested
    if args.fix_unlinked:
        result = fix_unlinked_jobs(
            workflow,
            args.fix_unlinked,
            job_dir_pattern=effective_job_dir_pattern,
            hours_cutoff=args.hours_cutoff,
            verbose=getattr(args, "verbose", False),
            max_jobs=args.debug if hasattr(args, "debug") else None,
            max_retries=args.max_retries,
            workers=args.workers,
        )
        print(
            f"\nFix-unlinked: {result['linked']} linked, "
            f"{result['reset']} reset to TO_RUN, "
            f"{result['skipped']} skipped"
        )

    # Recover orphaned jobs if requested
    if args.recover_orphans:
        if not args.scheduler:
            print("Error: --scheduler is required with --recover-orphans")
            sys.exit(1)
        recover_orphaned_jobs(
            workflow,
            scheduler=args.scheduler,
            hours_cutoff=args.hours_cutoff,
            verbose=getattr(args, "verbose", False),
        )

    # Always show summary
    print_summary(workflow)

    # Show progress bar
    counts = workflow.count_by_status()
    total = sum(counts.values())
    completed = counts.get(JobStatus.COMPLETED, 0)
    print_progress_bar(completed, total, width=60, label="Completion")

    # Show failed jobs if requested
    if args.show_failed:
        show_failed_jobs(workflow, limit=args.limit)

    # Show timeout jobs if requested
    if args.show_timeout:
        show_timeout_jobs(workflow, limit=args.limit)

    if args.show_running:
        show_running_jobs(workflow, limit=args.limit)

    # Show ready jobs if requested
    if args.show_ready:
        show_ready_jobs(workflow, limit=args.limit)

    # Show metrics if requested
    if args.show_metrics:
        print_metrics_summary(workflow)

    # Show chronic failures if requested
    if args.show_chronic_failures:
        chronic = workflow.get_jobs_by_fail_count(args.show_chronic_failures)
        if chronic:
            print_header(f"Jobs Failed {args.show_chronic_failures}+ Times")
            print(
                f"\n{'ID':<8} {'Orig Index':<12} {'Fails':<6} {'Status':<12} {'Error':<30}"
            )
            print("-" * 80)
            for job in chronic[: args.limit]:
                error = (
                    (job.error_message[:27] + "...")
                    if job.error_message and len(job.error_message) > 30
                    else (job.error_message or "N/A")
                )
                print(
                    f"{job.id:<8} {job.orig_index:<12} {job.fail_count:<6} "
                    f"{job.status.value:<12} {error:<30}"
                )
            if len(chronic) > args.limit:
                print(f"\n... and {len(chronic) - args.limit} more chronic failures")
        else:
            print(f"\nNo jobs have failed {args.show_chronic_failures}+ times.")

    # Sella-specific view, or auto-detect hint when the DB has sella rows.
    if args.sella_view == "summary":
        print_sella_summary(workflow, limit=args.limit)
    elif args.sella_view == "running":
        if not args.update:
            print(
                "\nError: --sella-view running requires --update <root_dir> "
                "to locate job directories."
            )
        else:
            show_sella_running_progress(
                workflow,
                Path(args.update),
                job_dir_pattern=effective_job_dir_pattern,
                limit=args.limit,
                workers=args.workers,
            )
    elif args.sella_view is None and has_sella_jobs(workflow):
        print(
            "\nTip: use --sella-view summary for opt-specific metrics "
            "on this database."
        )

    finish_wandb_run(wandb_run)
    workflow.close()


if __name__ == "__main__":
    main()
