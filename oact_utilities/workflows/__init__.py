"""High-throughput workflow management for architector jobs.

Keep package imports lightweight so worker processes can import
``oact_utilities.workflows.submit_jobs`` without eagerly importing optional
dependencies from unrelated modules.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ArchitectorWorkflow",
    "JobStatus",
    "create_split_workflows",
    "create_workflow",
    "update_job_status",
]


def __getattr__(name: str) -> Any:
    """Lazy-export workflow symbols from ``architector_workflow``."""
    if name in __all__:
        module = import_module(".architector_workflow", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the package attribute list for interactive inspection."""
    return sorted(list(globals()) + __all__)
