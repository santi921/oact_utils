"""Differentiable fairchem feature extractor for position optimization.

Unlike ``run_lmdb_inference.py`` (which runs under ``torch.no_grad()`` and detaches
the hooked activation for fast static extraction), this module keeps the graph from
atomic positions to the 128-dim metal-atom feature so the log-det score can be
back-propagated to positions.

Key differences from the inference path:
- ``compile=False`` (torch.compile complicates hooks / autograd).
- The forward hook does NOT detach.
- The model is called directly under ``torch.enable_grad()`` with
  ``pos.requires_grad_(True)``, bypassing ``predict()``'s no-grad inference context.

The 128-dim feature is the pre-activation input to the model's last linear layer,
read at atom index 0 (the metal center) -- identical to the inference extractor.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import torch
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.collaters.simple_collater import data_list_collater
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

from oact_utilities.scripts.entropy_downselect.run_lmdb_inference import (
    _disable_forces_and_stress,
    _find_last_linear,
    _get_torch_model_from_predictor,
)


def _create_grad_predictor(model_path: str, device: str = "cuda"):
    """Load a fairchem predict unit configured for gradient-enabled featurization."""
    inference_settings = InferenceSettings(
        tf32=False,
        activation_checkpointing=False,
        merge_mole=False,
        compile=False,
        external_graph_gen=False,
    )
    predictor = load_predict_unit(
        path=model_path, device=device, inference_settings=inference_settings
    )
    _disable_forces_and_stress(predictor)
    return predictor


class DifferentiableFeaturizer:
    """Map atomic positions to the metal-atom feature with autograd intact.

    Usage:
        feat = DifferentiableFeaturizer(model_path, device="cuda")
        data = feat.build_data(atoms)            # one AtomicData batch (batch of 1)
        x = feat.featurize(data, pos)            # (D,) torch tensor, grad to pos
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        radius: float = 6.0,
    ) -> None:
        self.predictor = _create_grad_predictor(model_path, device=device)
        self.device = torch.device(device)
        self.dtype = self.predictor.inference_settings.base_precision_dtype

        torch_model = _get_torch_model_from_predictor(self.predictor)
        _, hook_layer = _find_last_linear(torch_model)
        self._pre_activation: torch.Tensor | None = None
        hook_layer.register_forward_hook(self._hook_fn)

        task_name = list(self.predictor.dataset_to_tasks.keys())[0]
        self._a2g = partial(
            AtomicData.from_ase,
            task_name=task_name,
            r_edges=False,
            r_data_keys=["spin", "charge"],
            radius=radius,
            target_dtype=self.dtype,
        )
        self._initialized = False

    def _hook_fn(self, _mod, inp, _out) -> None:
        # Capture the pre-activation input WITHOUT detaching (preserve the graph).
        if isinstance(inp, (tuple, list)) and len(inp) > 0 and torch.is_tensor(inp[0]):
            self._pre_activation = inp[0]
        elif torch.is_tensor(inp):
            self._pre_activation = inp
        else:
            self._pre_activation = None

    def _ensure_init(self, data: AtomicData) -> None:
        """Trigger fairchem's lazy init (prepare_for_inference, device move) once."""
        if not self._initialized:
            with torch.no_grad():
                self.predictor.predict(data.clone())
            self._initialized = True

    def build_data(self, atoms) -> AtomicData:
        """Convert an ASE Atoms to a device-resident AtomicData batch of one."""
        data = data_list_collater([self._a2g(atoms)], otf_graph=True)
        self._ensure_init(data)
        data = data.to(self.device)
        for key, val in data:
            if torch.is_tensor(val) and val.is_floating_point():
                data[key] = val.to(self.dtype)
        self.predictor.model.module.on_predict_check(data)
        return data

    def initial_pos(self, data: AtomicData) -> torch.Tensor:
        """Return a detached copy of the batch positions (the optimization origin)."""
        return data.pos.detach().clone()

    def featurize(self, data: AtomicData, pos: torch.Tensor) -> torch.Tensor:
        """Run the model with the given positions; return the atom-0 feature (D,).

        ``pos`` must be a leaf tensor with ``requires_grad=True`` on the model device.
        Edges are regenerated from ``pos`` on the fly, so the returned feature carries
        a gradient back to ``pos``.
        """
        data.pos = pos
        self._pre_activation = None
        with torch.enable_grad():
            self.predictor.model(data)
        if self._pre_activation is None:
            raise RuntimeError("Forward hook captured no pre-activation.")
        return self._pre_activation[0]

    @torch.no_grad()
    def featurize_atoms_raw(self, atoms) -> np.ndarray:
        """No-grad raw (unwhitened) feature for an ASE Atoms, for parity checks."""
        data = self.build_data(atoms)
        self._pre_activation = None
        self.predictor.model(data)
        if self._pre_activation is None:
            raise RuntimeError("Forward hook captured no pre-activation.")
        return self._pre_activation[0].float().cpu().numpy()
