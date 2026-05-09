"""Shared helpers for workflow job directory naming."""

from __future__ import annotations

import re
import socket
from collections import OrderedDict

DEFAULT_JOB_DIR_PATTERN = "job_{orig_index}"
_JOB_PREFIX_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def get_job_dir_hostname() -> str:
    """Return the local coordinator hostname used in job directory names."""
    return socket.gethostname()


def apply_job_dir_prefix(job_dir_pattern: str, job_prefix: str | None = None) -> str:
    """Apply an optional stable run prefix to a job directory pattern.

    Args:
        job_dir_pattern: Base job directory pattern.
        job_prefix: Optional stable prefix to prepend. May contain only
            alphanumerics, ``.``, ``_``, and ``-``.

    Returns:
        Effective job directory pattern.

    Raises:
        ValueError: If ``job_prefix`` contains unsupported characters.
    """
    if not job_prefix:
        return job_dir_pattern
    if not _JOB_PREFIX_RE.fullmatch(job_prefix):
        raise ValueError(
            "job_prefix may contain only letters, numbers, dots, underscores, and dashes"
        )
    return f"{job_prefix}_{job_dir_pattern}"


def render_job_dir_pattern(
    job_dir_pattern: str,
    *,
    orig_index: int,
    job_id: int,
    hostname: str | None = None,
    elements: str | None = None,
    charge: int | None = None,
    spin: int | None = None,
) -> str:
    """Render a supported job directory pattern safely.

    Supported placeholders are ``{hostname}``, ``{orig_index}``, ``{id}``,
    ``{formula}``, ``{charge}``, and ``{spin}``.

    Args:
        job_dir_pattern: Pattern template to render.
        orig_index: Original structure index.
        job_id: Workflow database job ID.
        hostname: Optional hostname override. Defaults to the local hostname.
        elements: Optional semicolon-separated element string from the DB.
        charge: Optional molecular charge.
        spin: Optional spin multiplicity.

    Returns:
        Rendered directory name.

    Raises:
        ValueError: If unsupported placeholders or stray braces are present.
    """
    allowed_placeholders = (
        "{hostname}",
        "{orig_index}",
        "{id}",
        "{formula}",
        "{charge}",
        "{spin}",
    )
    temp_pattern = job_dir_pattern
    for placeholder in allowed_placeholders:
        temp_pattern = temp_pattern.replace(placeholder, "")
    if "{" in temp_pattern or "}" in temp_pattern:
        raise ValueError(
            f"Unsupported placeholder or stray brace in job_dir_pattern: {job_dir_pattern!r}"
        )

    resolved_hostname = hostname or get_job_dir_hostname()
    resolved_formula = _format_formula(elements)
    return (
        job_dir_pattern.replace("{hostname}", resolved_hostname)
        .replace("{orig_index}", str(orig_index))
        .replace("{id}", str(job_id))
        .replace("{formula}", resolved_formula)
        .replace("{charge}", str(charge if charge is not None else 0))
        .replace("{spin}", str(spin if spin is not None else 1))
    )


def _format_formula(elements: str | None) -> str:
    """Return a compact formula token from a semicolon-separated element list."""
    if not elements:
        return "unknown"

    counts: OrderedDict[str, int] = OrderedDict()
    for token in elements.split(";"):
        symbol = token.strip()
        if not symbol:
            continue
        counts[symbol] = counts.get(symbol, 0) + 1

    if not counts:
        return "unknown"

    parts = []
    for symbol, count in counts.items():
        suffix = "" if count == 1 else str(count)
        parts.append(f"{symbol}{suffix}")
    return "".join(parts)
