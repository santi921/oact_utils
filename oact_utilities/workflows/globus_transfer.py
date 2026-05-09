"""Globus transfer helpers for verified workflow job backups."""

from __future__ import annotations

import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


@dataclass(frozen=True)
class GlobusTransferConfig:
    """Configuration needed to submit a Globus transfer."""

    source_endpoint_id: str
    destination_endpoint_id: str
    dest_root: str
    client_id: str
    transfer_refresh_token: str
    client_secret: str | None = None


@dataclass(frozen=True)
class GlobusTransferResult:
    """Metadata for a submitted Globus archive transfer."""

    archive_path: Path
    destination_path: str
    task_id: str


def create_job_archive(
    job_dir: str | Path,
    archive_path: str | Path | None = None,
) -> Path:
    """Create a ``.tar.gz`` archive adjacent to a job directory.

    Args:
        job_dir: Directory containing ORCA/Sella job outputs.
        archive_path: Optional destination archive path. Defaults to
            ``<job_dir>.tar.gz``.

    Returns:
        Path to the created archive.

    """
    job_dir_path = Path(job_dir).resolve()
    if archive_path is None:
        archive = job_dir_path.with_name(f"{job_dir_path.name}.tar.gz")
    else:
        archive = Path(archive_path).resolve()

    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(job_dir_path, arcname=job_dir_path.name)

    return archive


def build_destination_path(
    archive_path: str | Path,
    root_dir: str | Path,
    dest_root: str,
) -> str:
    """Map an archive path under ``root_dir`` to the destination endpoint path.

    Args:
        archive_path: Path to the local archive.
        root_dir: Root directory used for workflow job outputs.
        dest_root: Destination directory on the Globus destination endpoint.

    Returns:
        POSIX-style destination path for the archive.

    Raises:
        ValueError: If the archive path is not under ``root_dir``.
    """
    archive = Path(archive_path).resolve()
    root = Path(root_dir).resolve()
    try:
        relative_archive = archive.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Archive path {archive} is not under workflow root {root}"
        ) from exc

    return str(PurePosixPath(dest_root) / PurePosixPath(relative_archive.as_posix()))


def build_transfer_client(config: GlobusTransferConfig) -> Any:
    """Build an authenticated ``globus_sdk.TransferClient``.

    Args:
        config: Globus transfer configuration.

    Returns:
        Authenticated Globus TransferClient.
    """
    import globus_sdk

    if config.client_secret:
        auth_client = globus_sdk.ConfidentialAppAuthClient(
            config.client_id,
            config.client_secret,
        )
    else:
        auth_client = globus_sdk.NativeAppAuthClient(config.client_id)

    authorizer = globus_sdk.RefreshTokenAuthorizer(
        config.transfer_refresh_token,
        auth_client,
    )
    return globus_sdk.TransferClient(authorizer=authorizer)


def submit_archive_transfer(
    archive_path: str | Path,
    root_dir: str | Path,
    config: GlobusTransferConfig,
    transfer_client: Any | None = None,
) -> GlobusTransferResult:
    """Submit one Globus transfer for an existing job archive.

    Args:
        archive_path: Path to ``<job_dir>.tar.gz`` on the source endpoint.
        root_dir: Workflow root used to derive the destination relative path.
        config: Globus transfer configuration.
        transfer_client: Optional prebuilt TransferClient, primarily for tests.

    Returns:
        Metadata for the submitted transfer task.

    """
    import globus_sdk

    archive = Path(archive_path).resolve()
    client = (
        transfer_client
        if transfer_client is not None
        else build_transfer_client(config)
    )
    destination_path = build_destination_path(archive, root_dir, config.dest_root)

    transfer_data = globus_sdk.TransferData(
        config.source_endpoint_id,
        config.destination_endpoint_id,
        label=f"oact backup {archive.name}",
    )
    transfer_data.add_item(str(archive), destination_path)

    response = client.submit_transfer(transfer_data)
    task_id = _extract_task_id(response)
    return GlobusTransferResult(
        archive_path=archive,
        destination_path=destination_path,
        task_id=task_id,
    )


def archive_and_submit_transfer(
    job_dir: str | Path,
    root_dir: str | Path,
    config: GlobusTransferConfig,
    transfer_client: Any | None = None,
) -> GlobusTransferResult:
    """Archive one job directory and submit the archive to Globus.

    Args:
        job_dir: Directory containing a verified successful job.
        root_dir: Workflow root used to map destination paths.
        config: Globus transfer configuration.
        transfer_client: Optional prebuilt TransferClient, primarily for tests.

    Returns:
        Metadata for the submitted transfer task.
    """
    client = (
        transfer_client
        if transfer_client is not None
        else build_transfer_client(config)
    )
    archive_path = create_job_archive(job_dir)
    return submit_archive_transfer(
        archive_path=archive_path,
        root_dir=root_dir,
        config=config,
        transfer_client=client,
    )


def _extract_task_id(response: Any) -> str:
    """Extract a Globus task ID from common SDK response shapes."""
    return str(response["task_id"])
