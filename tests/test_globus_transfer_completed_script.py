"""Tests for the completed-job Globus transfer shell script."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import time
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "oact_utilities"
    / "launch"
    / "globus_transfer_completed.sh"
)

CARPENTER_SOURCE_ENDPOINT_ID = "b808a48a-4b2d-11f1-a9a0-02535127e3d7"
BARFOOT_SOURCE_ENDPOINT_ID = "1ea1ecb5-4d77-11f1-848e-0ea3589134b3"
DESTINATION_ENDPOINT_ID = "05d2c76a-e867-4f67-aa57-76edeb0beda0"


def _create_db(db_path: Path, rows: list[tuple[str, str | None]]) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE structures (status TEXT, job_dir TEXT)")
    conn.executemany("INSERT INTO structures(status, job_dir) VALUES (?, ?)", rows)
    conn.commit()
    conn.close()


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(0o755)


def _install_fake_commands(bin_dir: Path) -> tuple[Path, Path, Path]:
    globus_args_file = bin_dir / "globus_args.txt"
    globus_batch_copy = bin_dir / "globus_batch.txt"
    gcp_args_file = bin_dir / "gcp_args.txt"

    _write_executable(
        bin_dir / "hostname",
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "${FAKE_HOSTNAME:-carpenter-login}"
""",
    )
    _write_executable(
        bin_dir / "pgrep",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${FAKE_PGREP_RUNNING:-0}" == "1" ]]; then
    printf '12345\\n'
    exit 0
fi
exit 1
""",
    )
    _write_executable(
        bin_dir / "globusconnectpersonal",
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$@" > "${FAKE_GCP_ARGS_FILE}"
""",
    )
    _write_executable(
        bin_dir / "globus",
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$@" > "${FAKE_GLOBUS_ARGS_FILE}"
batch_file=""
prev=""
for arg in "$@"; do
    if [[ "$prev" == "--batch" ]]; then
        batch_file="$arg"
        break
    fi
    prev="$arg"
done
if [[ -n "$batch_file" ]]; then
    cp "$batch_file" "${FAKE_GLOBUS_BATCH_COPY}"
fi
printf '%s\\n' "${FAKE_GLOBUS_TASK_ID:-task-123}"
""",
    )

    return globus_args_file, globus_batch_copy, gcp_args_file


def _build_env(
    fake_bin: Path,
    globus_args_file: Path,
    globus_batch_copy: Path,
    gcp_args_file: Path,
    hostname: str,
    gcp_running: bool,
) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["FAKE_HOSTNAME"] = hostname
    env["FAKE_PGREP_RUNNING"] = "1" if gcp_running else "0"
    env["FAKE_GLOBUS_ARGS_FILE"] = str(globus_args_file)
    env["FAKE_GLOBUS_BATCH_COPY"] = str(globus_batch_copy)
    env["FAKE_GCP_ARGS_FILE"] = str(gcp_args_file)
    env["FAKE_GLOBUS_TASK_ID"] = "task-123"
    env["GLOBUS_CONNECT_STARTUP_WAIT"] = "0"
    env["GLOBUS_TRANSFER_MIN_FILE_AGE_MINUTES"] = "0"
    return env


class TestGlobusTransferCompletedScript:
    """CLI tests for the completed-job Globus transfer helper."""

    def test_requires_db_and_dest_root(self):
        result = subprocess.run(
            ["bash", str(SCRIPT_PATH)],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "Usage:" in result.stderr

    def test_submits_completed_job_dirs_with_carpenter_endpoint(self, tmp_path):
        db_path = tmp_path / "workflow.db"
        job_dir_1 = tmp_path / "chunk_0" / "job_123"
        job_dir_2 = tmp_path / "chunk_1" / "job_456"
        missing_job_dir = tmp_path / "chunk_2" / "job_789"
        job_dir_1.mkdir(parents=True)
        job_dir_2.mkdir(parents=True)

        _create_db(
            db_path,
            [
                ("completed", str(job_dir_1)),
                ("completed", str(job_dir_1)),
                ("completed", str(job_dir_2)),
                ("completed", str(missing_job_dir)),
                ("running", str(tmp_path / "chunk_3" / "job_999")),
                ("completed", None),
            ],
        )

        fake_bin = tmp_path / "fake_bin"
        fake_bin.mkdir()
        globus_args_file, globus_batch_copy, gcp_args_file = _install_fake_commands(
            fake_bin
        )
        env = _build_env(
            fake_bin,
            globus_args_file,
            globus_batch_copy,
            gcp_args_file,
            hostname="carpenter-login-01",
            gcp_running=False,
        )

        result = subprocess.run(
            ["bash", str(SCRIPT_PATH), str(db_path), "/BLASTNet/carpenter"],
            capture_output=True,
            text=True,
            env=env,
        )

        assert result.returncode == 0, result.stderr
        assert "Submitted Globus task: task-123" in result.stdout
        assert "Skipped missing job directories: 1" in result.stderr

        globus_args = globus_args_file.read_text().splitlines()
        assert globus_args[0] == "transfer"
        assert globus_args[1] == CARPENTER_SOURCE_ENDPOINT_ID
        assert globus_args[2] == DESTINATION_ENDPOINT_ID
        assert "--batch" in globus_args
        assert "--jmespath" in globus_args
        assert "task_id" in globus_args

        batch_lines = globus_batch_copy.read_text().splitlines()
        assert batch_lines == [
            f"{job_dir_1} /BLASTNet/carpenter/job_123 --recursive",
            f"{job_dir_2} /BLASTNet/carpenter/job_456 --recursive",
        ]

        assert gcp_args_file.read_text().splitlines() == ["-start"]

    def test_reuses_running_gcp_and_selects_barfoot_endpoint(self, tmp_path):
        db_path = tmp_path / "workflow.db"
        job_dir = tmp_path / "jobs" / "job_123"
        job_dir.mkdir(parents=True)
        _create_db(db_path, [("completed", str(job_dir))])

        fake_bin = tmp_path / "fake_bin"
        fake_bin.mkdir()
        globus_args_file, globus_batch_copy, gcp_args_file = _install_fake_commands(
            fake_bin
        )
        env = _build_env(
            fake_bin,
            globus_args_file,
            globus_batch_copy,
            gcp_args_file,
            hostname="barfoot-node-02",
            gcp_running=True,
        )

        result = subprocess.run(
            ["bash", str(SCRIPT_PATH), str(db_path), "/BLASTNet/carpenter"],
            capture_output=True,
            text=True,
            env=env,
        )

        assert result.returncode == 0, result.stderr
        assert "Globus Connect Personal is already running." in result.stdout
        assert not gcp_args_file.exists()

        globus_args = globus_args_file.read_text().splitlines()
        assert globus_args[0] == "transfer"
        assert globus_args[1] == BARFOOT_SOURCE_ENDPOINT_ID
        assert globus_batch_copy.read_text().strip() == (
            f"{job_dir} /BLASTNet/carpenter/job_123 --recursive"
        )

    def test_unknown_hostname_fails_fast(self, tmp_path):
        db_path = tmp_path / "workflow.db"
        job_dir = tmp_path / "jobs" / "job_123"
        job_dir.mkdir(parents=True)
        _create_db(db_path, [("completed", str(job_dir))])

        fake_bin = tmp_path / "fake_bin"
        fake_bin.mkdir()
        globus_args_file, globus_batch_copy, gcp_args_file = _install_fake_commands(
            fake_bin
        )
        env = _build_env(
            fake_bin,
            globus_args_file,
            globus_batch_copy,
            gcp_args_file,
            hostname="unknown-host",
            gcp_running=False,
        )

        result = subprocess.run(
            ["bash", str(SCRIPT_PATH), str(db_path), "/BLASTNet/carpenter"],
            capture_output=True,
            text=True,
            env=env,
        )

        assert result.returncode != 0
        assert "unsupported hostname" in result.stderr
        assert not globus_args_file.exists()

    def test_skips_recently_modified_job_dirs(self, tmp_path):
        db_path = tmp_path / "workflow.db"
        old_job_dir = tmp_path / "jobs" / "job_123"
        recent_job_dir = tmp_path / "jobs" / "job_456"
        old_job_dir.mkdir(parents=True)
        recent_job_dir.mkdir(parents=True)

        old_file = old_job_dir / "orca.out"
        recent_file = recent_job_dir / "orca.out"
        old_file.write_text("done\n")
        recent_file.write_text("done\n")
        old_mtime = time.time() - 600
        os.utime(old_file, (old_mtime, old_mtime))

        _create_db(
            db_path,
            [
                ("completed", str(old_job_dir)),
                ("completed", str(recent_job_dir)),
            ],
        )

        fake_bin = tmp_path / "fake_bin"
        fake_bin.mkdir()
        globus_args_file, globus_batch_copy, gcp_args_file = _install_fake_commands(
            fake_bin
        )
        env = _build_env(
            fake_bin,
            globus_args_file,
            globus_batch_copy,
            gcp_args_file,
            hostname="carpenter-login-01",
            gcp_running=True,
        )
        env["GLOBUS_TRANSFER_MIN_FILE_AGE_MINUTES"] = "5"

        result = subprocess.run(
            ["bash", str(SCRIPT_PATH), str(db_path), "/BLASTNet/carpenter"],
            capture_output=True,
            text=True,
            env=env,
        )

        assert result.returncode == 0, result.stderr
        assert "Skipped recently modified job directories: 1" in result.stderr
        assert globus_batch_copy.read_text().strip() == (
            f"{old_job_dir} /BLASTNet/carpenter/job_123 --recursive"
        )
