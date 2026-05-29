"""Tests for oact_utilities/workflows/wandb_logger.py.

All tests use unittest.mock to avoid real W&B network calls.
"""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock, call, patch

import pytest

from oact_utilities.workflows.wandb_logger import (
    backfill_terminal_cdfs,
    finish_wandb_run,
    init_wandb_run,
    log_campaign_snapshot,
    log_job_result,
    log_progress_snapshot,
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
        assert (
            call("cdf/*", step_metric="_timestamp")
            in mock_run.define_metric.call_args_list
        )
        assert (
            call("gauge/*", step_metric="_timestamp")
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


# ------------------------------------------------------------------------
# CDF backfill + live snapshot tests (V2 plan)
# ------------------------------------------------------------------------


@pytest.fixture
def populated_workflow(tmp_path):
    """Build a real ArchitectorWorkflow DB with a mix of terminal/non-terminal rows.

    Layout:
      - 3 completed rows at t=10, 20, 30
      - 2 failed rows at t=15, 25
      - 1 timeout row at t=28
      - 1 running row at t=5  (excluded from terminal history)
      - 1 to_run row at t=1   (excluded from terminal history)
    Timestamps are stored as the SQLite TIMESTAMP string format the schema
    uses (`%Y-%m-%d %H:%M:%S`, treated as UTC).
    """
    from oact_utilities.utils.architector import _init_db, _insert_row
    from oact_utilities.workflows import ArchitectorWorkflow

    db_path = tmp_path / "cdf.db"
    conn = _init_db(db_path)

    base = datetime.datetime(2026, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)

    rows = [
        ("completed", 10),
        ("completed", 20),
        ("completed", 30),
        ("failed", 15),
        ("failed", 25),
        ("timeout", 28),
        ("running", 5),
        ("to_run", 1),
    ]
    for i, (status, _offset) in enumerate(rows):
        _insert_row(
            conn,
            orig_index=i,
            elements="H;H",
            natoms=2,
            geometry="H 0 0 0\nH 0 0 0.74",
            status=status,
        )

    # Overwrite updated_at to deterministic values (CURRENT_TIMESTAMP default
    # gives "now" which would vary across runs).
    for row_id, (_status, offset) in enumerate(rows, start=1):
        ts = (base + datetime.timedelta(seconds=offset)).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("UPDATE structures SET updated_at = ? WHERE id = ?", (ts, row_id))
    conn.commit()
    conn.close()

    wf = ArchitectorWorkflow(db_path)
    yield wf, base
    wf.close()


class TestIterTerminalHistory:
    """Tests for ArchitectorWorkflow.iter_terminal_history."""

    def test_returns_only_terminal_rows_sorted_by_time(self, populated_workflow):
        wf, base = populated_workflow
        history = wf.iter_terminal_history()

        # 3 completed + 2 failed + 1 timeout = 6 rows. running/to_run excluded.
        assert len(history) == 6
        statuses = [s for s, _t in history]
        assert sorted(statuses) == sorted(
            ["completed", "completed", "completed", "failed", "failed", "timeout"]
        )

        # Returns raw SQL strings (callers parse), sorted ascending.
        times = [t for _s, t in history]
        assert times == sorted(times)
        assert all(isinstance(t, str) for t in times)

        # Spot-check: first terminal event is at offset 10s from base.
        expected_first = (base + datetime.timedelta(seconds=10)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        assert times[0] == expected_first

    def test_skips_rows_with_null_updated_at(self, populated_workflow):
        wf, _base = populated_workflow
        # Null out one row's updated_at directly. Use a subquery rather than
        # UPDATE ... LIMIT (which depends on a SQLite compile flag).
        conn = wf.conn
        conn.execute(
            "UPDATE structures SET updated_at = NULL "
            "WHERE id = (SELECT id FROM structures WHERE status = 'completed' LIMIT 1)"
        )
        conn.commit()

        history = wf.iter_terminal_history()
        # 6 - 1 = 5 surviving terminal rows.
        assert len(history) == 5


class TestBackfillTerminalCdfs:
    """Tests for backfill_terminal_cdfs."""

    def test_emits_one_log_per_terminal_row_with_cumulative_counts(
        self, populated_workflow
    ):
        wf, _base = populated_workflow
        mock_run = MagicMock()

        backfill_terminal_cdfs(mock_run, wf)

        # 6 terminal rows -> 6 run.log calls.
        assert mock_run.log.call_count == 6

        # Each payload has cumulative counts that monotonically grow.
        payloads = [c.args[0] for c in mock_run.log.call_args_list]
        completed_series = [p["cdf/completed"] for p in payloads]
        failed_series = [p["cdf/failed"] for p in payloads]
        timeout_series = [p["cdf/timeout"] for p in payloads]

        # Final values match the fixture: 3 completed, 2 failed, 1 timeout.
        assert completed_series[-1] == 3
        assert failed_series[-1] == 2
        assert timeout_series[-1] == 1

        # All three series are monotonically non-decreasing.
        assert completed_series == sorted(completed_series)
        assert failed_series == sorted(failed_series)
        assert timeout_series == sorted(timeout_series)

        # Every payload has a _timestamp.
        for p in payloads:
            assert "_timestamp" in p

    def test_noop_when_run_is_none(self, populated_workflow):
        wf, _base = populated_workflow
        # Should not raise; nothing to assert beyond non-explosion.
        backfill_terminal_cdfs(None, wf)

    def test_swallows_workflow_exception(self, capsys):
        mock_run = MagicMock()
        broken_wf = MagicMock()
        broken_wf.iter_terminal_history.side_effect = RuntimeError("db gone")

        backfill_terminal_cdfs(mock_run, broken_wf)

        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "backfill_terminal_cdfs failed" in captured.out

    def test_skips_rows_with_unparseable_timestamps(self, populated_workflow):
        """A malformed updated_at string is skipped, not raised."""
        wf, _base = populated_workflow
        wf.conn.execute(
            "UPDATE structures SET updated_at = 'not-a-timestamp' "
            "WHERE id = (SELECT id FROM structures WHERE status = 'completed' LIMIT 1)"
        )
        wf.conn.commit()

        mock_run = MagicMock()
        backfill_terminal_cdfs(mock_run, wf)

        # 6 terminal rows -> 5 surviving after the malformed one is skipped.
        assert mock_run.log.call_count == 5

    def test_handles_fractional_second_timestamps(self, tmp_path):
        """Backfill parses '2026-... HH:MM:SS.fff' as well as '...:SS'."""
        from oact_utilities.utils.architector import _init_db, _insert_row
        from oact_utilities.workflows import ArchitectorWorkflow

        db_path = tmp_path / "frac.db"
        conn = _init_db(db_path)
        _insert_row(
            conn,
            orig_index=0,
            elements="H;H",
            natoms=2,
            geometry="H 0 0 0\nH 0 0 0.74",
            status="completed",
        )
        conn.execute(
            "UPDATE structures SET updated_at = '2026-05-13 14:23:01.500' WHERE id = 1"
        )
        conn.commit()
        conn.close()

        wf = ArchitectorWorkflow(db_path)
        try:
            mock_run = MagicMock()
            backfill_terminal_cdfs(mock_run, wf)
            assert mock_run.log.call_count == 1
            payload = mock_run.log.call_args[0][0]
            # 2026-05-13 14:23:01.500 UTC -> epoch 1778336581.5
            expected = datetime.datetime(
                2026, 5, 13, 14, 23, 1, 500_000, tzinfo=datetime.timezone.utc
            ).timestamp()
            assert abs(payload["_timestamp"] - expected) < 0.01
        finally:
            wf.close()


class TestLogProgressSnapshot:
    """Tests for log_progress_snapshot."""

    def test_logs_cdf_and_gauge_keys_with_current_counts(self, populated_workflow):
        wf, _base = populated_workflow
        mock_run = MagicMock()

        log_progress_snapshot(mock_run, wf)

        mock_run.log.assert_called_once()
        payload = mock_run.log.call_args[0][0]
        assert "_timestamp" in payload
        # Fixture has 3 completed, 2 failed, 1 timeout, 1 running, 1 to_run.
        assert payload["cdf/completed"] == 3
        assert payload["cdf/failed"] == 2
        assert payload["cdf/timeout"] == 1
        assert payload["gauge/running"] == 1
        assert payload["gauge/to_run"] == 1

    def test_does_not_touch_run_summary(self, populated_workflow):
        """run.summary is owned by log_campaign_snapshot; this function only logs."""
        wf, _base = populated_workflow
        mock_run = MagicMock()

        log_progress_snapshot(mock_run, wf)

        mock_run.summary.update.assert_not_called()

    def test_noop_when_run_is_none(self, populated_workflow):
        wf, _base = populated_workflow
        log_progress_snapshot(None, wf)

    def test_swallows_exception(self, capsys):
        mock_run = MagicMock()
        broken_wf = MagicMock()
        broken_wf.count_by_status.side_effect = RuntimeError("db gone")

        log_progress_snapshot(mock_run, broken_wf)

        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "log_progress_snapshot failed" in captured.out
