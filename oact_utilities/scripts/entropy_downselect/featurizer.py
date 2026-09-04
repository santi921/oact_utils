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
        # Keys present in a freshly built batch, before any forward pass. The
        # model injects graph-derived state back into the data object during
        # forward and caches some of it behind a ``not in data_dict`` guard
        # (e.g. ``scatter_target`` in escn_md._generate_graph). Because we reuse
        # one data object across optimization steps with otf_graph=True, that
        # cached state goes stale as soon as moving atoms change the edge count,
        # causing an index_add size mismatch. We snapshot the pristine key set
        # at build time and strip everything else before each forward.
        self._input_keys: set[str] = set()

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

    def _snapshot_input_keys(self, data: AtomicData) -> None:
        """Record the pristine key set of a freshly built batch (pre-forward)."""
        self._input_keys = {key for key, _ in data}

    def _strip_injected_keys(self, data: AtomicData) -> None:
        """Delete any model-injected key so the next forward regenerates it.

        Must run before every forward on a reused data object: it clears stale
        graph-derived state (notably ``scatter_target``) left by the previous
        step, which is otherwise kept behind the model's ``not in data_dict``
        guard and would mismatch the freshly regenerated edge count.
        """
        stale = [key for key, _ in data if key not in self._input_keys]
        for key in stale:
            del data[key]

    def build_data(self, atoms) -> AtomicData:
        """Convert an ASE Atoms to a device-resident AtomicData batch of one."""
        data = data_list_collater([self._a2g(atoms)], otf_graph=True)
        self._ensure_init(data)
        data = data.to(self.device)
        for key, val in data:
            if torch.is_tensor(val) and val.is_floating_point():
                data[key] = val.to(self.dtype)
        self.predictor.model.module.on_predict_check(data)
        self._snapshot_input_keys(data)
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
        self._strip_injected_keys(data)
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

    def build_batch(self, atoms_list: list):
        """Collate many ASE Atoms into one device-resident AtomicData batch.

        Returns:
            data: the collated AtomicData (positions still the originals).
            offsets: (M,) long -- first-atom index of each structure (metal atom).
            natoms: (M,) long -- atom count per structure.
            batch_idx: (A,) long -- structure index for each atom.
        """
        data = data_list_collater([self._a2g(a) for a in atoms_list], otf_graph=True)
        self._ensure_init(data)
        data = data.to(self.device)
        for key, val in data:
            if torch.is_tensor(val) and val.is_floating_point():
                data[key] = val.to(self.dtype)
        self.predictor.model.module.on_predict_check(data)
        self._snapshot_input_keys(data)
        natoms = data["natoms"].to(torch.long)
        offsets = torch.zeros(len(natoms), dtype=torch.long, device=self.device)
        if len(natoms) > 1:
            offsets[1:] = torch.cumsum(natoms, 0)[:-1]
        batch_idx = data["batch"].to(torch.long)
        return data, offsets, natoms, batch_idx

    def featurize_batch(
        self, data: AtomicData, pos: torch.Tensor, offsets: torch.Tensor
    ) -> torch.Tensor:
        """Run the model on a batch; return per-structure metal features (M, D).

        ``pos`` is the leaf (A, 3) position tensor (requires_grad for backprop). Edges
        are regenerated on the fly, so the returned features carry gradients to ``pos``.
        """
        data.pos = pos
        self._strip_injected_keys(data)
        self._pre_activation = None
        with torch.enable_grad():
            self.predictor.model(data)
        if self._pre_activation is None:
            raise RuntimeError("Forward hook captured no pre-activation.")
        return self._pre_activation[offsets]
