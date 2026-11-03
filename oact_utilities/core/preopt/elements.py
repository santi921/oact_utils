"""Element-coverage checks for pre-optimizers.

An MLIP does not raise on an element it was never trained on: any Z below the embedding size
produces a number. Coverage therefore has to be checked explicitly, from the per-element energy
references the model was fitted with (a zero or noise-level reference means the element was absent from the fit).

Three checkpoint layouts are handled:

- OAct inference ``.pt`` files: ``tasks_config`` energy task, ``element_references`` list indexed
  by Z (length ``max_num_elements``).
- Registry names (``uma-s-1p1``, ``esen-sm-conserving-all-omol``, ...): ``atom_refs["omol"]``
  of the predict unit, a dict keyed by Z.
- g-xTB: fixed range Z = 1-103.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from ase.data import atomic_numbers, chemical_symbols

GXTB_MAX_Z = 103
# Absent elements have a reference of exactly 0 or fit noise (~1e-10 Eh in the v3/v4 OAct
# checkpoints for Rn); the smallest real reference (H) is about -0.5 Eh.
REFERENCE_TOL_EH = 1e-6
OMOL_TASK = "omol"


class UnsupportedElementsError(ValueError):
    """Raised when a structure contains elements the pre-optimizer does not cover."""

    def __init__(
        self,
        elements: Iterable[str],
        method: str,
        job_id: str | int | None = None,
    ) -> None:
        self.elements = sorted(set(elements), key=lambda s: atomic_numbers[s])
        self.method = method
        self.job_id = job_id
        where = f" (job {job_id})" if job_id is not None else ""
        super().__init__(
            f"{method} does not support element(s) {', '.join(self.elements)}{where}"
        )


def parse_elements(elements: str | Iterable[str]) -> list[str]:
    """Element symbols from the DB ``elements`` column (semicolon separated) or an iterable."""
    if isinstance(elements, str):
        return [e.strip() for e in elements.split(";") if e.strip()]
    return [e.strip() for e in elements if e.strip()]


def gxtb_supported_elements() -> frozenset[str]:
    """Elements g-xTB covers (Z = 1-103)."""
    return frozenset(chemical_symbols[1 : GXTB_MAX_Z + 1])


def supported_from_reference_list(references: Sequence[float]) -> frozenset[str]:
    """Elements with a non-negligible reference in a list indexed by atomic number (index 0 unused)."""
    return frozenset(
        chemical_symbols[z]
        for z, ref in enumerate(references)
        if 1 <= z < len(chemical_symbols) and abs(ref) > REFERENCE_TOL_EH
    )


def supported_from_reference_keys(keys: Iterable[int | str]) -> frozenset[str]:
    """Elements present as keys (atomic numbers) of a per-element reference dict."""
    return frozenset(
        chemical_symbols[int(k)] for k in keys if 1 <= int(k) < len(chemical_symbols)
    )


def _is_checkpoint_file(checkpoint: str | Path) -> bool:
    return Path(checkpoint).suffix == ".pt" or Path(checkpoint).is_file()


def _supported_from_pt(path: Path) -> frozenset[str]:
    import torch

    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    for task in ckpt.tasks_config:
        if task["name"] == "energy":
            refs = task["element_references"]["element_references"]["_args_"][0]
            return supported_from_reference_list(list(refs))
    raise ValueError(f"no energy task with element references in {path}")


def supported_from_predict_unit(unit: Any) -> frozenset[str]:
    """Elements in the omol atom references of a loaded fairchem predict unit."""
    atom_refs = unit.atom_refs
    if OMOL_TASK not in atom_refs:
        raise ValueError(f"model has no {OMOL_TASK!r} atom references")
    return supported_from_reference_keys(atom_refs[OMOL_TASK].keys())


def _supported_from_registry(name: str) -> frozenset[str]:
    from fairchem.core.calculate.pretrained_mlip import get_predict_unit

    return supported_from_predict_unit(get_predict_unit(name, device="cpu"))


@lru_cache(maxsize=16)
def get_supported_elements(checkpoint: str) -> frozenset[str]:
    """Elements an MLIP checkpoint was fitted on, read from its metadata (cached per string).

    Args:
        checkpoint: Path to an OAct inference ``.pt`` file, or a fairchem registry name.

    Returns:
        Element symbols with a nonzero energy reference. Necessary, not sufficient: a nonzero
        reference does not prove the element is well represented in the training data.
    """
    if _is_checkpoint_file(checkpoint):
        return _supported_from_pt(Path(checkpoint))
    return _supported_from_registry(checkpoint)


def unsupported_elements(
    elements: str | Iterable[str], supported: frozenset[str]
) -> list[str]:
    """Elements in ``elements`` that are not in ``supported``, ordered by atomic number."""
    missing = set(parse_elements(elements)) - supported
    return sorted(missing, key=lambda s: atomic_numbers.get(s, 999))


def check_elements(
    elements: str | Iterable[str],
    supported: frozenset[str],
    method: str,
    job_id: str | int | None = None,
) -> None:
    """Raise ``UnsupportedElementsError`` when any element is outside ``supported``."""
    missing = unsupported_elements(elements, supported)
    if missing:
        raise UnsupportedElementsError(missing, method, job_id)
