"""
steering/leash_layer.py
========================
Model-agnostic LeashLayer.

Wraps any transformer decoder layer and optionally applies conditional
activation steering (condition checking + behaviour vector injection) on
every forward pass.

Identical steering algorithm to leash_layer_qwen3.py; all Qwen3-VL
references have been removed.  The wrapper assumes:
  - The decoder layer receives hidden_states as its first positional arg.
  - hidden_states has shape [batch, seq_len, hidden_dim].
  - The layer can be replaced in an nn.ModuleList transparently.

Injection position
------------------
Vectors are injected *before* the wrapped layer (pre-layer injection).
During prefill (seq_len > 1) only the last token position is modified,
avoiding contamination of image/history token representations.
During decode (seq_len == 1) the single token is modified globally.

Class-level shared state
------------------------
Because many LeashLayer instances (one per decoder layer) must coordinate
condition detection across the sequence, key state is stored at the class
level and reset between samples via reset_class() / reset_per_sample().
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import Callable, List, Optional

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────
# LayerControlParams
# ─────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class LayerControlParams:
    """Per-layer steering operator (default: additive)."""
    control: Optional[torch.Tensor] = None
    operator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
        lambda current, control: current + control
    )

    @classmethod
    def default(cls) -> "LayerControlParams":
        return cls()


# ─────────────────────────────────────────────────────────────────────────
# LeashLayer
# ─────────────────────────────────────────────────────────────────────────

class LeashLayer(nn.Module):
    """
    Wraps a single decoder layer and optionally applies conditional
    activation steering on each forward pass.

    Parameters
    ----------
    layer : nn.Module
        The original decoder layer being wrapped.
    layer_id : int
        Zero-based index of this layer in the model's decoder stack.
    """

    # ── class-level shared state (reset between samples) ──────────────
    condition_met: defaultdict = defaultdict(bool)
    condition_similarities: defaultdict = defaultdict(lambda: defaultdict(float))
    condition_layers = None   # list[dict[int, bool]]
    behavior_layers = None    # list[dict[int, bool]]

    def __init__(self, layer: nn.Module, layer_id: int) -> None:
        super().__init__()
        self.layer = layer
        self.layer_id = layer_id

        self.use_ooi_preventive_normalization: bool = False

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            # 避免在 layer 初始化完成前发生死循环
            if "layer" in self._modules and hasattr(self.layer, name):
                return getattr(self.layer, name)
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
            
    # ── multi-condition API ────────────────────────────────────────────

    def multisteer(
        self,
        behavior_vectors: List[Optional[torch.Tensor]],
        condition_projectors: List[Optional[torch.Tensor]],
        thresholds: List[float],
        use_ooi_preventive_normalization: bool = False,
        condition_comparator_threshold_is: List[str] = None,
        condition_threshold_comparison_modes: List[str] = None,
        rules: List[str] = None,
        **kwargs,
    ) -> None:
        """Configure multi-condition steering for this layer."""
        self.behavior_vectors = behavior_vectors
        self.condition_projectors = condition_projectors
        self.thresholds = thresholds
        self.params_list = [LayerControlParams(**kwargs) for _ in behavior_vectors]
        self.use_ooi_preventive_normalization = use_ooi_preventive_normalization
        self.condition_comparator_threshold_is = condition_comparator_threshold_is or ["larger"]
        self.condition_threshold_comparison_modes = (
            condition_threshold_comparison_modes or ["last"]
        )
        self.rules = rules or []

    # ── forward ────────────────────────────────────────────────────────

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        """
        Pre-injection → wrapped layer forward.
        hidden_states shape: [batch, seq_len, hidden_dim]
        """
        if LeashLayer.condition_layers is None:
            is_condition = is_behavior = False
        else:
            is_condition = any(
                layers.get(self.layer_id, False) for layers in LeashLayer.condition_layers
            )
            is_behavior = any(
                layers.get(self.layer_id, False) for layers in LeashLayer.behavior_layers
            )

        original_norm = hidden_states.norm(dim=-1, keepdim=True)

        if is_condition:
            self._process_multi_conditions(hidden_states)

        if is_behavior:
            hidden_states = self._apply_multi_behaviors(hidden_states)

        if self.use_ooi_preventive_normalization and is_behavior:
            hidden_states = self._apply_ooi_normalization(hidden_states, original_norm)

        return self.layer(hidden_states, *args, **kwargs)

    # ── condition processing ───────────────────────────────────────────

    def _process_multi_conditions(self, hidden_state: torch.Tensor) -> None:
        for ci in range(len(self.condition_projectors)):
            if LeashLayer.condition_met[ci]:
                continue
            if not LeashLayer.condition_layers[ci].get(self.layer_id, False):
                continue

            c_vec = self.condition_projectors[ci]
            if c_vec is None:
                continue

            mode = (
                self.condition_threshold_comparison_modes[ci]
                if ci < len(self.condition_threshold_comparison_modes)
                else "last"
            )
            if mode == "mean":
                hs = hidden_state.mean(dim=(0, 1)).to(torch.float32)   # [D]
            else:
                hs = hidden_state[:, -1, :].mean(dim=0).to(torch.float32)  # [D]

            c = c_vec.to(dtype=torch.float32, device=hs.device)

            if c.ndim == 1:
                sim = torch.nn.functional.cosine_similarity(
                    hs.unsqueeze(0), c.unsqueeze(0), dim=-1
                ).squeeze(0).item()
            elif c.ndim == 2:
                proj = torch.tanh(c @ hs)  # [D]
                sim = torch.nn.functional.cosine_similarity(
                    hs.unsqueeze(0), proj.unsqueeze(0), dim=-1
                ).squeeze(0).item()
            else:
                raise ValueError(
                    f"Unsupported condition tensor ndim for C{ci+1}: {tuple(c.shape)}"
                )

            LeashLayer.condition_similarities[ci][self.layer_id] = sim

            threshold = self.thresholds[ci]
            comparator = (
                self.condition_comparator_threshold_is[ci]
                if ci < len(self.condition_comparator_threshold_is)
                else "larger"
            )
            if comparator == "smaller":
                LeashLayer.condition_met[ci] = sim < threshold
            else:
                LeashLayer.condition_met[ci] = sim > threshold

            print(
                f"[MULTI CONDITION] C{ci+1} | layer={self.layer_id} "
                f"| sim={sim:.4f} | thr={threshold:.4f} "
                f"| met={LeashLayer.condition_met[ci]}"
            )

    # ── behaviour injection ────────────────────────────────────────────

    def _apply_multi_behaviors(self, hidden_states: torch.Tensor) -> torch.Tensor:
        seq_len = hidden_states.shape[1]
        modified_hs = hidden_states.clone()

        for rule in self.rules:
            bi = int(rule.split("B")[1]) - 1
            if not self._evaluate_rule(rule):
                continue
            if not LeashLayer.behavior_layers[bi].get(self.layer_id, False):
                continue
            bv = self.behavior_vectors[bi]
            if bv is None:
                continue

            control = bv.to(dtype=hidden_states.dtype, device=hidden_states.device)

            params = self.params_list[bi]

            if seq_len > 1:
                modified_hs[:, -1:, :] = params.operator(
                    modified_hs[:, -1:, :], control.view(1, 1, -1)
                )
            else:
                modified_hs = params.operator(modified_hs, control.view(1, 1, -1))

        return modified_hs

    # ── rule evaluation ────────────────────────────────────────────────

    def _evaluate_rule(self, rule: str) -> bool:
        parts = rule.split("then")
        if len(parts) != 2:
            return False
        cond_part = parts[0].strip().lower().replace("if", "").strip().split()
        if "or" in cond_part:
            return any(
                self._check_cond(c) for c in cond_part if c not in ("or", "and")
            )
        if "and" in cond_part:
            return all(
                self._check_cond(c) for c in cond_part if c not in ("or", "and")
            )
        return self._check_cond(cond_part[0]) if cond_part else False

    @staticmethod
    def _check_cond(c: str) -> bool:
        if c.startswith("c"):
            try:
                return LeashLayer.condition_met[int(c[1:]) - 1]
            except (ValueError, IndexError):
                return False
        return False

    # ── helpers ────────────────────────────────────────────────────────

    def _apply_ooi_normalization(
        self, hidden_states: torch.Tensor, original_norm: torch.Tensor
    ) -> torch.Tensor:
        new_norm = hidden_states.norm(dim=-1, keepdim=True)
        ratio = (new_norm / (original_norm + 1e-8)).max().item()
        if ratio > 1 or torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
            hidden_states = hidden_states * (original_norm / (new_norm + 1e-8))
        return hidden_states

    # ── reset ──────────────────────────────────────────────────────────

    def reset_instance(self) -> None:
        """Reset per-instance steering config."""
        self.behavior_vectors = []
        self.condition_projectors = []
        self.thresholds = []
        self.params_list = []
        self.rules = []

    @classmethod
    def reset_class(cls) -> None:
        """Reset all class-level shared state (call between full eval runs)."""
        cls.condition_met = defaultdict(bool)
        cls.condition_similarities = defaultdict(lambda: defaultdict(float))
        cls.condition_layers = None
        cls.behavior_layers = None

    @classmethod
    def reset_per_sample(cls) -> None:
        """Reset per-sample condition detection state (call between samples)."""
        cls.condition_met = defaultdict(bool)
        cls.condition_similarities = defaultdict(lambda: defaultdict(float))
