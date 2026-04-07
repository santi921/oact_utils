"""Tests for oact_utilities/workflows/wandb_logger.py.

All tests use unittest.mock to avoid real W&B network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestInitWandbRun:
    """Tests for init_wandb_run."""

    def test_calls_wandb_init_with_correct_args(self):
        """init_wandb_run forwards project, name, id, and resume='allow' to wandb.init."""
        mock_run = MagicMock()
        with patch("oact_utilities.workflows.wandb_logger.WANDB_AVAILABLE", True):
            with patch("oact_utilities.workflows.wandb_logger.wandb") as mock_wandb:
                mock_wandb.init.return_value = mock_run
                from oact_utilities.workflows.wandb_logger import init_wandb_run

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

    def test_returns_none_when_wandb_unavailable(self):
        """init_wandb_run returns None when WANDB_AVAILABLE is False."""
        with patch("oact_utilities.workflows.wandb_logger.WANDB_AVAILABLE", False):
            from oact_utilities.workflows.wandb_logger import init_wandb_run

            result = init_wandb_run(project="test-project")

        assert result is None

    def test_returns_none_and_prints_warning_when_wandb_init_raises(self, capsys):
        """init_wandb_run returns None (not raises) if wandb.init throws."""
        with patch("oact_utilities.workflows.wandb_logger.WANDB_AVAILABLE", True):
            with patch("oact_utilities.workflows.wandb_logger.wandb") as mock_wandb:
                mock_wandb.init.side_effect = RuntimeError("network error")
                from oact_utilities.workflows.wandb_logger import init_wandb_run

                result = init_wandb_run(project="test-project")

        assert result is None
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "W&B init failed" in captured.out


class TestLogJobResult:
    """Tests for log_job_result."""

    def test_logs_completed_job_with_metrics(self):
        """Completed job: logs progress/completed=1 and all non-None metric keys."""
        mock_run = MagicMock()
        from oact_utilities.workflows.wandb_logger import log_job_result

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
        assert payload["progress/completed"] == 1
        assert payload["metrics/max_forces"] == 0.0023
        assert payload["metrics/final_energy"] == -1204.3
        assert payload["metrics/scf_steps"] == 42
        assert payload["metrics/wall_time"] == 312.4
        assert payload["metrics/n_cores"] == 16

    def test_logs_only_progress_key_when_metrics_is_none(self):
        """metrics=None: only progress/{status} is logged, no metrics/ keys."""
        mock_run = MagicMock()
        from oact_utilities.workflows.wandb_logger import log_job_result

        log_job_result(mock_run, job_id=43, status="failed", metrics=None)

        mock_run.log.assert_called_once()
        payload = mock_run.log.call_args[0][0]
        assert payload == {"progress/failed": 1}

    def test_skips_none_metric_values(self):
        """Metric values that are None are excluded from the payload."""
        mock_run = MagicMock()
        from oact_utilities.workflows.wandb_logger import log_job_result

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
        from oact_utilities.workflows.wandb_logger import log_job_result

        # Should not raise
        log_job_result(None, job_id=1, status="completed", metrics={"max_forces": 0.1})

    def test_swallows_run_log_exception(self, capsys):
        """If run.log raises, the exception is caught and a warning is printed."""
        mock_run = MagicMock()
        mock_run.log.side_effect = ConnectionError("timeout")
        from oact_utilities.workflows.wandb_logger import log_job_result

        # Should not raise
        log_job_result(mock_run, job_id=99, status="completed")

        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "W&B log failed" in captured.out


class TestFinishWandbRun:
    """Tests for finish_wandb_run."""

    def test_calls_run_finish(self):
        """finish_wandb_run calls run.finish()."""
        mock_run = MagicMock()
        from oact_utilities.workflows.wandb_logger import finish_wandb_run

        finish_wandb_run(mock_run)

        mock_run.finish.assert_called_once_with()

    def test_noop_when_run_is_none(self):
        """finish_wandb_run is a no-op when run=None."""
        from oact_utilities.workflows.wandb_logger import finish_wandb_run

        # Should not raise
        finish_wandb_run(None)
