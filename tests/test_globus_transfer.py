"""Tests for Globus archive and transfer helpers."""

from __future__ import annotations

import sys
import tarfile
from types import SimpleNamespace

import pytest

from oact_utilities.workflows.globus_transfer import (
    GlobusTransferConfig,
    build_destination_path,
    create_job_archive,
    submit_archive_transfer,
)


def test_create_job_archive_creates_adjacent_tarball(tmp_path):
    """A job directory is archived as <job_dir>.tar.gz."""
    root_dir = tmp_path / "jobs"
    job_dir = root_dir / "job_1"
    job_dir.mkdir(parents=True)
    (job_dir / "orca.out").write_text("****ORCA TERMINATED NORMALLY****\n")

    archive_path = create_job_archive(job_dir)

    assert archive_path == (root_dir / "job_1.tar.gz").resolve()
    assert archive_path.exists()

    with tarfile.open(archive_path, "r:gz") as tar:
        assert "job_1/orca.out" in tar.getnames()


def test_build_destination_path_uses_archive_relative_to_root(tmp_path):
    """Destination path preserves the archive path below the workflow root."""
    root_dir = tmp_path / "jobs"
    archive_path = root_dir / "chunk_0" / "job_1.tar.gz"
    archive_path.parent.mkdir(parents=True)
    archive_path.write_text("placeholder")

    destination = build_destination_path(
        archive_path=archive_path,
        root_dir=root_dir,
        dest_root="/globus/backups",
    )

    assert destination == "/globus/backups/chunk_0/job_1.tar.gz"


def test_build_destination_path_rejects_archive_outside_root(tmp_path):
    """Archives outside the workflow root cannot be mapped."""
    with pytest.raises(ValueError, match="not under workflow root"):
        build_destination_path(
            archive_path=tmp_path / "outside.tar.gz",
            root_dir=tmp_path / "jobs",
            dest_root="/globus/backups",
        )


def test_submit_archive_transfer_builds_expected_transfer(monkeypatch, tmp_path):
    """TransferData receives the configured endpoints and mapped paths."""
    root_dir = tmp_path / "jobs"
    archive_path = root_dir / "job_1.tar.gz"
    archive_path.parent.mkdir(parents=True)
    archive_path.write_text("archive")

    submitted = {}

    class FakeAuthorizer:
        def __init__(self, token):
            self.token = token

    class FakeTransferData:
        def __init__(self, client, source_endpoint, destination_endpoint, label):
            self.client = client
            self.source_endpoint = source_endpoint
            self.destination_endpoint = destination_endpoint
            self.label = label
            self.items = []

        def add_item(self, source_path, destination_path):
            self.items.append((source_path, destination_path))

    class FakeTransferClient:
        def __init__(self, authorizer=None):
            self.authorizer = authorizer

        def submit_transfer(self, transfer_data):
            submitted["transfer_data"] = transfer_data
            submitted["authorizer"] = self.authorizer
            return {"task_id": "task-123"}

    fake_sdk = SimpleNamespace(
        AccessTokenAuthorizer=FakeAuthorizer,
        TransferClient=FakeTransferClient,
        TransferData=FakeTransferData,
    )
    monkeypatch.setitem(sys.modules, "globus_sdk", fake_sdk)

    config = GlobusTransferConfig(
        source_endpoint_id="source-ep",
        destination_endpoint_id="dest-ep",
        dest_root="/globus/backups",
        access_token="secret-token",
    )

    result = submit_archive_transfer(
        archive_path=archive_path,
        root_dir=root_dir,
        config=config,
    )

    transfer_data = submitted["transfer_data"]
    assert result.task_id == "task-123"
    assert result.archive_path == archive_path.resolve()
    assert result.destination_path == "/globus/backups/job_1.tar.gz"
    assert submitted["authorizer"].token == "secret-token"
    assert transfer_data.source_endpoint == "source-ep"
    assert transfer_data.destination_endpoint == "dest-ep"
    assert transfer_data.items == [
        (str(archive_path.resolve()), "/globus/backups/job_1.tar.gz")
    ]
