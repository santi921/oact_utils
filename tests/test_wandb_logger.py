"""Tests for oact_utilities/workflows/wandb_logger.py.

All tests use unittest.mock to avoid real W&B network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

from oact_utilities.workflows.wandb_logger import (
    finish_wandb_run,
    init_wandb_run,
    log_campaign_snapshot,
    log_job_result,
)


class TestInitWandbRun:
    """Tests for init_wandb_run."""

    def test_calls_wandb_init_with_correct_args(self):
        """init_wandb_run forwards project, name, id, and resume='allow' to wandb.init."""
        mock_run = MagicMock()
        with patch("oact_utilities.workflows.wandb_logger.WANDB_AVAILABLE", True):
            with patch("oact_utilities.workflows.wandb_logger.wandb") as mock_wandb:
                mock_wandb.init.return_value = mock_run

                result = init_wandb_run(
                    project="test-project",
                    run_name="wave_two",
                    run_id="abc123",
                )

        mock_wandb.init.assert_called_once_with(
            project="test-project",
            name="wave_two",
            id="abc123",
            resume="allow",
        )
        assert result is mock_run

    def test_registers_timestamp_as_step_metric(self):
        """init_wandb_run calls define_metric to set _timestamp as x-axis for charts."""
        mock_run = MagicMock()
        with patch("oact_utilities.workflows.wandb_logger.WANDB_AVAILABLE", True):
            with patch("oact_utilities.workflows.wandb_logger.wandb") as mock_wandb:
                mock_wandb.init.return_value = mock_run

                init_wandb_run(project="test-project")

        assert call("_timestamp") in mock_run.define_metric.call_args_list
        assert (
            call("metrics/*", step_metric="_timestamp")
            in mock_run.define_metric.call_args_list
        )
        assert (
            call("progress/*", step_metric="_timestamp")
            in mock_run.define_metric.call_args_list
        )

    def test_returns_none_when_wandb_unavailable(self):
        """init_wandb_run returns None when WANDB_AVAILABLE is False."""
        with patch("oact_utilities.workflows.wandb_logger.WANDB_AVAILABLE", False):
            result = init_wandb_run(project="test-project")

        assert result is None

    def test_returns_none_and_prints_warning_when_wandb_init_raises(self, capsys):
        """init_wandb_run returns None (not raises) if wandb.init throws."""
        with patch("oact_utilities.workflows.wandb_logger.WANDB_AVAILABLE", True):
            with patch("oact_utilities.workflows.wandb_logger.wandb") as mock_wandb:
                mock_wandb.init.side_effect = RuntimeError("network error")

                result = init_wandb_run(project="test-project")

        assert result is None
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "W&B init failed" in captured.out


class TestLogJobResult:
    """Tests for log_job_result."""

    def test_logs_completed_job_with_metrics(self):
        """Completed job: logs _timestamp, progress/completed=1, and all non-None metric keys."""
        mock_run = MagicMock()

        log_job_result(
            mock_run,
            job_id=42,
            status="completed",
            metrics={
                "max_forces": 0.0023,
                "final_energy": -1204.3,
                "scf_steps": 42,
                "wall_time": 312.4,
                "n_cores": 16,
            },
        )

        mock_run.log.assert_called_once()
        payload = mock_run.log.call_args[0][0]
        assert "_timestamp" in payload
        assert payload["progress/completed"] == 1
        assert payload["metrics/max_forces"] == 0.0023
        assert payload["metrics/final_energy"] == -1204.3
        assert payload["metrics/scf_steps"] == 42
        assert payload["metrics/wall_time"] == 312.4
        assert payload["metrics/n_cores"] == 16

    def test_logs_only_progress_and_timestamp_when_metrics_is_none(self):
        """metrics=None: only _timestamp and progress/{status} are logged."""
        mock_run = MagicMock()

        log_job_result(mock_run, job_id=43, status="failed", metrics=None)

        mock_run.log.assert_called_once()
        payload = mock_run.log.call_args[0][0]
        assert payload["progress/failed"] == 1
        assert "_timestamp" in payload
        assert len(payload) == 2

    def test_skips_none_metric_values(self):
        """Metric values that are None are excluded from the payload."""
        mock_run = MagicMock()

        log_job_result(
            mock_run,
            job_id=44,
            status="completed",
            metrics={"max_forces": 0.001, "wall_time": None, "n_cores": None},
        )

        payload = mock_run.log.call_args[0][0]
        assert "metrics/max_forces" in payload
        assert "metrics/wall_time" not in payload
        assert "metrics/n_cores" not in payload

    def test_noop_when_run_is_none(self):
        """log_job_result is a no-op when run=None."""
        log_job_result(None, job_id=1, status="completed", metrics={"max_forces": 0.1})

    def test_swallows_run_log_exception(self, capsys):
        """If run.log raises, the exception is caught and a warning is printed."""
        mock_run = MagicMock()
        mock_run.log.side_effect = ConnectionError("timeout")

        log_job_result(mock_run, job_id=99, status="completed")

        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "W&B log failed" in captured.out


class TestFinishWandbRun:
    """Tests for finish_wandb_run."""

    def test_calls_run_finish(self):
        """finish_wandb_run calls run.finish()."""
        mock_run = MagicMock()

        finish_wandb_run(mock_run)

        mock_run.finish.assert_called_once_with()

    def test_noop_when_run_is_none(self):
        """finish_wandb_run is a no-op when run=None."""
        finish_wandb_run(None)


class TestLogCampaignSnapshot:
    """Tests for log_campaign_snapshot."""

    def _make_counts(self):
        from oact_utilities.workflows.architector_workflow import JobStatus

        return {
            JobStatus.COMPLETED: 100,
            JobStatus.FAILED: 5,
            JobStatus.TO_RUN: 50,
            JobStatus.RUNNING: 10,
            JobStatus.TIMEOUT: 2,
        }

    def test_campaign_keys_go_to_summary(self):
        """Campaign status counts are written to run.summary, not run.log."""
        mock_run = MagicMock()
        counts = self._make_counts()

        log_campaign_snapshot(mock_run, counts, total=167)

        mock_run.summary.update.assert_called_once()
        summary = mock_run.summary.update.call_args[0][0]
        assert summary["campaign/completed"] == 100
        assert summary["campaign/failed"] == 5
        assert summary["campaign/to_run"] == 50
        assert summary["campaign/running"] == 10
        assert summary["campaign/timeout"] == 2
        assert abs(summary["campaign/progress_pct"] - 59.88) < 0.1

    def test_run_log_not_called_when_no_metrics_stats(self):
        """run.log is not called when metrics_stats is None (avoids empty chart points)."""
        mock_run = MagicMock()
        counts = self._make_counts()

        log_campaign_snapshot(mock_run, counts, total=167, metrics_stats=None)

        mock_run.log.assert_not_called()

    def test_metrics_stats_logged_as_time_series_with_timestamp(self):
        """When metrics_stats provided, run.log is called with _timestamp and metrics/* keys."""
        mock_run = MagicMock()
        counts = self._make_counts()
        stats = {
            "max_forces_mean": 0.002,
            "wall_time_total_hours": 42.5,
            "n_completed_with_forces": 100,
        }

        log_campaign_snapshot(mock_run, counts, total=167, metrics_stats=stats)

        mock_run.log.assert_called_once()
        payload = mock_run.log.call_args[0][0]
        assert "_timestamp" in payload
        assert payload["metrics/max_forces_mean"] == 0.002
        assert payload["metrics/wall_time_total_hours"] == 42.5
        assert payload["metrics/n_completed_with_forces"] == 100

    def test_none_metric_values_skipped(self):
        """None values in metrics_stats are excluded from the log payload."""
        mock_run = MagicMock()
        counts = self._make_counts()
        stats = {"max_forces_mean": 0.002, "wall_time_total_hours": None}

        log_campaign_snapshot(mock_run, counts, total=167, metrics_stats=stats)

        payload = mock_run.log.call_args[0][0]
        assert "metrics/max_forces_mean" in payload
        assert "metrics/wall_time_total_hours" not in payload

    def test_noop_when_run_is_none(self):
        """log_campaign_snapshot is a no-op when run=None."""
        counts = self._make_counts()
        log_campaign_snapshot(None, counts, total=167)

    def test_progress_pct_zero_when_total_is_zero(self):
        """progress_pct is 0 when total=0 (no divide-by-zero)."""
        mock_run = MagicMock()
        log_campaign_snapshot(mock_run, {}, total=0)
        summary = mock_run.summary.update.call_args[0][0]
        assert summary["campaign/progress_pct"] == 0
