"""
steering/malleable_model.py
============================
Model-agnostic MalleableModel.

Wraps any HuggingFace VLM and adds conditional activation steering.
All model-specific logic (layer path, input formatting, output parsing)
is delegated to a ModelAdapter instance.

Identical steering algorithm to malleable_model_qwen3.py; Qwen3-VL
references have been removed and replaced by adapter calls.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..adapters.base import ModelAdapter
from .leash_layer import LeashLayer


# ─────────────────────────────────────────────────────────────────────────
# Steering vector container
# ─────────────────────────────────────────────────────────────────────────

class SteeringVectorPT:
    """
    Thin wrapper around the .pt dict produced by vectors/extract.py.

    Attributes
    ----------
    directions : dict[int, np.ndarray]
        layer_id → direction vector (float32, shape [hidden_dim]).
    explained_variances : dict[int, float]
        layer_id → PCA explained variance ratio.
    best_layer : int
        Layer with highest explained variance.
    metadata : dict
        Extra fields from the saved payload (adapter_name, model_path, …).
    """

    def __init__(self, data: dict):
        self.directions: Dict[int, np.ndarray] = {
            int(k): np.asarray(v, dtype=np.float32)
            for k, v in data["directions"].items()
        }
        self.explained_variances: Dict[int, float] = {
            int(k): float(v) for k, v in data["explained_variances"].items()
        }
        self.best_layer: int = int(data["best_layer"])
        self.metadata: dict = {
            k: v for k, v in data.items()
            if k not in ("directions", "explained_variances", "best_layer",
                         "best_direction")
        }

    @classmethod
    def load(cls, path: str) -> "SteeringVectorPT":
        data = torch.load(path, map_location="cpu", weights_only=False)
        print(f"[SteeringVectorPT] Loaded {path}  best_layer={data['best_layer']}")
        return cls(data)


# ─────────────────────────────────────────────────────────────────────────
# MalleableModel
# ─────────────────────────────────────────────────────────────────────────

class MalleableModel(nn.Module):
    """
    Adapter-driven MalleableModel.

    Wraps any VLM whose decoder layers are accessible via
    ``adapter.get_decoder_layers(model)`` and adds conditional activation
    steering through LeashLayer wrappers.

    Parameters
    ----------
    model
        The loaded HuggingFace model (already .eval()).
    processor
        The loaded processor / tokenizer.
    adapter
        A ModelAdapter instance that handles model-specific operations.
    """

    def __init__(self, model: Any, processor: Any, adapter: ModelAdapter) -> None:
        super().__init__()
        self.model = model
        self.processor = processor
        self.adapter = adapter

        layers = adapter.get_decoder_layers(model)
        for i in range(len(layers)):
            if not isinstance(layers[i], LeashLayer):
                layers[i] = LeashLayer(layers[i], i)

        self._num_layers = len(layers)
        print(
            f"[MalleableModel] Adapter={adapter.name!r}  "
            f"Wrapped {self._num_layers} decoder layers."
        )

    # ── properties ─────────────────────────────────────────────────────

    @property
    def config(self):
        return self.model.config

    @property
    def device(self) -> torch.device:
        return self.model.device

    @property
    def num_layers(self) -> int:
        return self._num_layers

    # ── multisteer ─────────────────────────────────────────────────────

    def multisteer(
        self,
        behavior_vectors: List[Optional[SteeringVectorPT]],
        behavior_layer_ids: List[List[int]],
        behavior_vector_strengths: List[float],
        condition_vectors: List[Optional[SteeringVectorPT]],
        condition_layer_ids: List[List[int]],
        condition_vector_thresholds: List[float],
        condition_comparator_threshold_is: List[str],
        rules: List[str],
        condition_threshold_comparison_modes: Optional[List[str]] = None,
        use_ooi_preventive_normalization: bool = False,
        **kwargs,
    ) -> None:
        """Configure C1/C2 condition gates and behavior-vector injection."""
        layers = self.adapter.get_decoder_layers(self.model)
        n = len(layers)
        n_conds = len(condition_vectors)
        n_behavs = len(behavior_vectors)

        if condition_threshold_comparison_modes is None:
            condition_threshold_comparison_modes = ["last"] * n_conds

        effective_cond_layer_ids = [list(lids) for lids in condition_layer_ids]

        # ── Build per-layer flag dicts ────────────────────────────────
        cond_layer_flags = [{i: False for i in range(n)} for _ in range(n_conds)]
        behv_layer_flags = [{i: False for i in range(n)} for _ in range(n_behavs)]

        for ci, lids in enumerate(effective_cond_layer_ids):
            for lid in lids:
                cond_layer_flags[ci][lid] = True
        for bi, lids in enumerate(behavior_layer_ids):
            for lid in lids:
                behv_layer_flags[bi][lid] = True

        LeashLayer.condition_layers = cond_layer_flags
        LeashLayer.behavior_layers = behv_layer_flags

        for lid in range(n):
            layer: LeashLayer = layers[lid]

            bvecs = []
            for bi in range(n_behavs):
                bv = None
                if behavior_vectors[bi] is not None and lid in behavior_layer_ids[bi]:
                    direction = behavior_vectors[bi].directions[lid]
                    bv = torch.tensor(
                        behavior_vector_strengths[bi] * direction,
                        dtype=self.model.dtype,
                    ).to(self.device)
                bvecs.append(bv)

            cprojs = []
            for ci in range(n_conds):
                cp = None
                if lid in effective_cond_layer_ids[ci]:
                    cv = condition_vectors[ci]
                    if cv is not None:
                        cdir = torch.tensor(
                            cv.directions[lid],
                            dtype=self.model.dtype,
                        ).to(self.device)
                        cp = cdir  # 1-D cosine branch in _process_multi_conditions
                cprojs.append(cp)

            layer.multisteer(
                behavior_vectors=bvecs,
                condition_projectors=cprojs,
                thresholds=condition_vector_thresholds,
                use_ooi_preventive_normalization=use_ooi_preventive_normalization,
                condition_comparator_threshold_is=condition_comparator_threshold_is,
                condition_threshold_comparison_modes=condition_threshold_comparison_modes,
                rules=rules,
                **kwargs,
            )

    # ── reset ──────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Remove all steering from the model."""
        layers = self.adapter.get_decoder_layers(self.model)
        for layer in layers:
            if isinstance(layer, LeashLayer):
                layer.reset_instance()
        LeashLayer.reset_class()

    def reset_per_sample(self) -> None:
        """Reset per-sample condition detection state between samples."""
        LeashLayer.reset_per_sample()

    # ── unwrap ─────────────────────────────────────────────────────────

    def unwrap(self) -> Any:
        """Restore the original layers and return the bare model."""
        layers = self.adapter.get_decoder_layers(self.model)
        for i in range(len(layers)):
            if isinstance(layers[i], LeashLayer):
                layers[i] = layers[i].layer
        return self.model

    # ── respond ────────────────────────────────────────────────────────

    def respond(
        self,
        messages: List[dict],
        image_paths: List[str],
        max_new_tokens: int = 256,
        reset_after_response: bool = False,
    ) -> str:
        """
        Generate a response for one sample.

        Parameters
        ----------
        messages
            Raw dataset messages (role/content).
        image_paths
            Absolute paths to screenshot images.
        max_new_tokens
            Maximum tokens to generate.
        reset_after_response
            If True, reset per-sample state after generation.
        """
        model_messages = self.adapter.build_messages(
            messages, image_paths, mode="generate"
        )
        text = self.adapter.apply_chat_template(
            self.processor, model_messages, add_generation_prompt=True
        )
        inputs = self.adapter.build_inputs(
            self.processor, model_messages, text, self.device
        )
        outputs = self.adapter.generate(self.model, inputs, max_new_tokens)
        response_text = self.adapter.decode_response(
            self.processor, outputs, inputs["input_ids"].shape[1]
        )

        if reset_after_response:
            self.reset_per_sample()

        return response_text

    # ── passthrough ────────────────────────────────────────────────────

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
