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
    access_token: str


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

    Raises:
        FileNotFoundError: If ``job_dir`` does not exist or is not a directory.
        ValueError: If ``archive_path`` is inside ``job_dir``.
    """
    job_dir_path = Path(job_dir).resolve()
    if not job_dir_path.is_dir():
        raise FileNotFoundError(f"Job directory not found: {job_dir_path}")

    if archive_path is None:
        archive = job_dir_path.with_name(f"{job_dir_path.name}.tar.gz")
    else:
        archive = Path(archive_path).resolve()

    try:
        archive.relative_to(job_dir_path)
    except ValueError:
        pass
    else:
        raise ValueError("archive_path must not be inside job_dir")

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


def build_transfer_client(access_token: str) -> Any:
    """Build an authenticated ``globus_sdk.TransferClient``.

    Args:
        access_token: Globus transfer access token.

    Returns:
        Authenticated Globus TransferClient.

    Raises:
        RuntimeError: If ``globus_sdk`` is not installed.
    """
    try:
        import globus_sdk
    except ImportError as exc:
        raise RuntimeError(
            "globus_sdk is required for --globus-transfer. "
            "Install with: pip install globus-sdk"
        ) from exc

    authorizer = globus_sdk.AccessTokenAuthorizer(access_token)
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

    Raises:
        RuntimeError: If ``globus_sdk`` is not installed.
    """
    try:
        import globus_sdk
    except ImportError as exc:
        raise RuntimeError(
            "globus_sdk is required for --globus-transfer. "
            "Install with: pip install globus-sdk"
        ) from exc

    archive = Path(archive_path).resolve()
    client = (
        transfer_client
        if transfer_client is not None
        else build_transfer_client(config.access_token)
    )
    destination_path = build_destination_path(archive, root_dir, config.dest_root)

    transfer_data = globus_sdk.TransferData(
        client,
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
        else build_transfer_client(config.access_token)
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
    try:
        task_id = response["task_id"]
    except (KeyError, TypeError):
        data = getattr(response, "data", {})
        task_id = data.get("task_id") if isinstance(data, dict) else None

    return "" if task_id is None else str(task_id)
