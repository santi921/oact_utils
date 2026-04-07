"""Optional W&B logging utilities for oact_utilities workflows.

All functions are no-ops when wandb is not installed or when ``run`` is None,
so callers never need to guard with ``if WANDB_AVAILABLE``.

Usage (V1 -- Parsl submission loop)::

    from oact_utilities.workflows.wandb_logger import (
        WANDB_AVAILABLE,
        finish_wandb_run,
        init_wandb_run,
        log_job_result,
    )

    run = init_wandb_run(project="actinide-campaign", run_name="wave_two")
    try:
        # ... submit jobs ...
        log_job_result(run, job_id=42, status="completed", metrics=metrics_dict)
        log_job_result(run, job_id=43, status="failed")
    finally:
        finish_wandb_run(run)

Usage (V2 -- dashboard --update scan)::

    from oact_utilities.workflows.wandb_logger import (
        add_wandb_args,
        compute_metrics_stats,
        finish_wandb_run,
        init_wandb_run,
        log_campaign_snapshot,
    )

    # In argparse setup:
    add_wandb_args(parser)

    # In main(), after update_all_statuses():
    run = init_wandb_run(project=args.wandb_project, run_name=args.wandb_run_name)
    counts = workflow.count_by_status()
    stats = compute_metrics_stats(workflow)
    log_campaign_snapshot(run, counts, total=sum(counts.values()), metrics_stats=stats)
    finish_wandb_run(run)
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .architector_workflow import ArchitectorWorkflow

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None  # type: ignore[assignment]


def init_wandb_run(
    project: str,
    run_name: str | None = None,
    run_id: str | None = None,
) -> Any:
    """Initialize a W&B run and return the run object.

    Args:
        project: W&B project name.
        run_name: Display name for the run (shown in the W&B UI).
        run_id: Existing run ID to resume. Pass this across batches of the
            same campaign to keep all jobs in one W&B run.

    Returns:
        A ``wandb.Run`` object, or ``None`` if wandb is unavailable or init
        fails.
    """
    if not WANDB_AVAILABLE:
        return None
    try:
        return wandb.init(
            project=project,
            name=run_name,
            id=run_id,
            resume="allow",
        )
    except Exception as e:
        print(f"Warning: W&B init failed: {e}. Continuing without logging.")
        return None


def log_job_result(
    run: Any,
    job_id: int,
    status: str,
    metrics: dict[str, Any] | None = None,
) -> None:
    """Log a single job outcome to W&B.

    Logs ``progress/{status}: 1`` plus any available metrics under
    ``metrics/{key}``. Safe to call with ``run=None`` or when individual
    metric values are ``None``.

    Args:
        run: W&B run object returned by ``init_wandb_run``, or ``None``.
        job_id: Workflow database job ID (used in warning messages only).
        status: Job outcome string -- ``"completed"``, ``"failed"``, or
            ``"timeout"``.
        metrics: Optional dict of metric values (max_forces, final_energy,
            scf_steps, wall_time, n_cores). Keys with ``None`` values are
            skipped.
    """
    if run is None:
        return
    try:
        payload: dict[str, Any] = {f"progress/{status}": 1}
        if metrics:
            for k, v in metrics.items():
                if v is not None:
                    payload[f"metrics/{k}"] = v
        run.log(payload)
    except Exception as e:
        print(f"Warning: W&B log failed for job {job_id}: {e}")


def finish_wandb_run(run: Any) -> None:
    """Finish a W&B run. No-op if ``run`` is ``None``.

    Args:
        run: W&B run object returned by ``init_wandb_run``, or ``None``.
    """
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        pass


def compute_metrics_stats(workflow: ArchitectorWorkflow) -> dict[str, Any] | None:
    """Query aggregate metrics stats for completed jobs with force data.

    Shared by ``print_metrics_summary`` and ``log_campaign_snapshot`` so both
    use the same SQL query without duplication. Only includes jobs where
    ``max_forces IS NOT NULL``; jobs completed without gradient data are
    excluded.

    Args:
        workflow: Open ``ArchitectorWorkflow`` instance.

    Returns:
        Dict with aggregate stats for forces, energy, SCF steps, wall time,
        and cores; or ``None`` if no qualifying completed jobs exist yet.
        The ``n_completed_with_forces`` key reports the row count.
    """
    cur = workflow._execute_with_retry(
        """
        SELECT max_forces, scf_steps, wall_time, n_cores, final_energy
        FROM structures
        WHERE status = 'completed' AND max_forces IS NOT NULL
        """
    )
    rows = cur.fetchall()
    if not rows:
        return None

    import statistics

    forces = [r[0] for r in rows]
    scf = [r[1] for r in rows if r[1] is not None]
    wall = [r[2] for r in rows if r[2] is not None]
    cores = [r[3] for r in rows if r[3] is not None]
    energy = [r[4] for r in rows if r[4] is not None]

    result: dict[str, Any] = {}
    if forces:
        result["max_forces_mean"] = statistics.mean(forces)
        result["max_forces_median"] = statistics.median(forces)
    if scf:
        result["scf_steps_mean"] = statistics.mean(scf)
    if wall:
        result["wall_time_mean"] = statistics.mean(wall)
        result["wall_time_total_hours"] = sum(wall) / 3600
    if cores and wall:
        valid_pairs = [
            (r[2], r[3]) for r in rows if r[2] is not None and r[3] is not None
        ]
        result["core_hours_total"] = sum(w * c for w, c in valid_pairs) / 3600
    if energy:
        result["final_energy_mean"] = statistics.mean(energy)
        result["final_energy_min"] = min(energy)
        result["final_energy_max"] = max(energy)
    result["n_completed_with_forces"] = len(forces)
    return result


def log_campaign_snapshot(
    run: Any,
    counts: dict,
    total: int,
    metrics_stats: dict[str, Any] | None = None,
) -> None:
    """Log aggregate campaign status and optional metrics summary to W&B.

    Safe to call with ``run=None`` or when individual metric values are ``None``.

    Args:
        run: W&B run object returned by ``init_wandb_run``, or ``None``.
        counts: Dict mapping ``JobStatus`` -> count (from ``workflow.count_by_status()``).
        total: Total job count (``sum(counts.values())``).
        metrics_stats: Optional dict from ``compute_metrics_stats()``.
    """
    if run is None:
        return
    try:
        from .architector_workflow import JobStatus

        completed = counts.get(JobStatus.COMPLETED, 0)
        payload: dict[str, Any] = {
            "campaign/completed": completed,
            "campaign/failed": counts.get(JobStatus.FAILED, 0),
            "campaign/to_run": counts.get(JobStatus.TO_RUN, 0),
            "campaign/running": counts.get(JobStatus.RUNNING, 0),
            "campaign/timeout": counts.get(JobStatus.TIMEOUT, 0),
            "campaign/progress_pct": 100 * completed / total if total > 0 else 0,
        }
        if metrics_stats:
            for k, v in metrics_stats.items():
                if v is not None:
                    payload[f"metrics/{k}"] = v
        run.log(payload)
    except Exception as e:
        print(f"Warning: W&B log failed: {e}")


def add_wandb_args(parser: argparse.ArgumentParser) -> None:
    """Add W&B CLI arguments to an argparse parser.

    Shared by ``submit_jobs.main()`` and ``dashboard.main()`` to keep the
    argument names and help text consistent.

    Args:
        parser: The argparse parser (or argument group parent) to add args to.
    """
    group = parser.add_argument_group("W&B Options")
    group.add_argument(
        "--wandb-project",
        default=None,
        metavar="PROJECT",
        help="W&B project name (enables W&B logging when provided)",
    )
    group.add_argument(
        "--wandb-run-name",
        default=None,
        metavar="NAME",
        help="W&B run display name (default: database filename stem)",
    )
    group.add_argument(
        "--wandb-run-id",
        default=None,
        metavar="ID",
        help="W&B run ID to resume an existing run across multiple invocations",
    )
