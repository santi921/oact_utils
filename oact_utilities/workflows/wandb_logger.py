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
"""

from __future__ import annotations

from typing import Any

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
