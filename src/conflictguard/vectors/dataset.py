"""
vectors/dataset.py
===================
Model-agnostic ConflictDataset.

Builds contrastive pairs from the CONFLICT SFT dataset
(sft_set_qwen3.json or any compatible JSON).  Returns raw pairs that
carry the original dataset messages + image paths; the adapter is
responsible for converting them to model-specific message format.

Dataset modes
-------------
behavior
    Qwen3-VL uses adaptive suffixes from paired conflict1/clean assistant
    responses. Other model families use the adapter's fixed
    positive/negative suffixes.

conditional_c1 / conditional_c2
    No suffixes.  Positive = conflict1/2 prompt, negative = clean prompt.
    Used to extract condition vectors that detect conflict context.

Data format expected
--------------------
Each JSON entry must have at least:
  {
    "messages": [
        {"role": "user",      "content": "<image> ...instruction..."},
        {"role": "assistant", "content": "..."}
    ],
    "images":    ["/abs/or/relative/path/to/screen.png"],
    "task_id":   2338,
    "dataset":   "AMEX",
    "data_type": "clean" | "conflict1" | "conflict2"
  }
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional


# ---------------------------------------------------------------------------
# Raw contrastive pair
# ---------------------------------------------------------------------------

@dataclass
class RawContrastivePair:
    """
    A contrastive pair in model-agnostic form.

    The adapter's ``build_messages()`` method converts these into
    model-specific message lists before calling the processor.

    positive_messages / negative_messages
        Original dataset message lists (role + content strings).
    positive_images / negative_images
        Absolute image paths from the dataset entry.
    positive_suffix / negative_suffix
        Raw assistant-turn suffix strings for behaviour vector extraction.
        None for conditional-vector mode.
    task_id / dataset
        Metadata for traceability.
    """
    positive_messages: List[dict]
    negative_messages: List[dict]
    positive_images: List[str]
    negative_images: List[str]
    positive_suffix: Optional[str] = None
    negative_suffix: Optional[str] = None
    task_id: Optional[int] = None
    dataset: Optional[str] = None


# ---------------------------------------------------------------------------
# ConflictDataset
# ---------------------------------------------------------------------------

class ConflictDataset:
    """
    Builds a list of RawContrastivePair objects from a CONFLICT SFT JSON.

    Parameters
    ----------
    data_path : str
        Path to sft_set_qwen3.json (or any compatible JSON).
    mode : str
        One of: "behavior", "conditional_c1", "conditional_c2".
    adapter
        A ModelAdapter instance. Determines the behavior-pair strategy and
        supplies fallback positive/negative suffixes.
    n_samples : int
        Maximum number of pairs to sample.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        data_path: str,
        mode: Literal[
            "behavior",
            "conditional_c1",
            "conditional_c2",
        ],
        adapter=None,
        n_samples: int = 200,
        seed: int = 42,
    ):
        self.mode = mode
        self.behavior_strategy: Optional[str] = None
        self.pairs: List[RawContrastivePair] = []

        print(f"[ConflictDataset] Loading '{data_path}' …")
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data: List[dict] = json.load(f)

        # Group by (task_id, dataset)
        groups: Dict[tuple, Dict[str, dict]] = defaultdict(dict)
        for item in raw_data:
            key = (item["task_id"], item["dataset"])
            groups[key][item["data_type"]] = item

        # Qwen3-VL uses adaptive responses; other supported models use fixed
        # adapter-native termination/execution suffixes.
        pos_sfx_default = neg_sfx_default = None
        if mode == "behavior":
            if adapter is None:
                raise ValueError(
                    "An adapter must be provided for mode='behavior'."
                )
            pos_sfx_default = adapter.default_positive_suffix
            neg_sfx_default = adapter.default_negative_suffix
            self.behavior_strategy = (
                "adaptive" if adapter.name == "qwen3vl" else "fixed"
            )

        random.seed(seed)
        self._build_pairs(
            groups,
            mode,
            n_samples,
            pos_sfx_default,
            neg_sfx_default,
            self.behavior_strategy,
        )

        strategy = f" strategy={self.behavior_strategy!r}" if self.behavior_strategy else ""
        print(f"[ConflictDataset] mode={mode!r}{strategy} -> {len(self.pairs)} pairs ready.")

    # ------------------------------------------------------------------
    # Internal pair builder
    # ------------------------------------------------------------------

    def _build_pairs(
        self,
        groups: Dict[tuple, Dict[str, dict]],
        mode: str,
        n_samples: int,
        pos_sfx_default: Optional[str],
        neg_sfx_default: Optional[str],
        behavior_strategy: Optional[str],
    ) -> None:
        if mode == "behavior" and behavior_strategy == "fixed":
            valid_keys = [key for key, value in groups.items() if "clean" in value]
            sampled = random.sample(valid_keys, min(n_samples, len(valid_keys)))
            for key in sampled:
                item = groups[key]["clean"]
                self.pairs.append(RawContrastivePair(
                    positive_messages=item["messages"],
                    negative_messages=item["messages"],
                    positive_images=item.get("images", []),
                    negative_images=item.get("images", []),
                    positive_suffix=pos_sfx_default,
                    negative_suffix=neg_sfx_default,
                    task_id=item.get("task_id"),
                    dataset=item.get("dataset"),
                ))

        elif mode == "behavior" and behavior_strategy == "adaptive":
            valid_keys = [
                key for key, value in groups.items()
                if "clean" in value and "conflict1" in value
            ]
            sampled = random.sample(valid_keys, min(n_samples, len(valid_keys)))
            for key in sampled:
                clean = groups[key]["clean"]
                conflict = groups[key]["conflict1"]
                self.pairs.append(RawContrastivePair(
                    positive_messages=clean["messages"],
                    negative_messages=clean["messages"],
                    positive_images=clean.get("images", []),
                    negative_images=clean.get("images", []),
                    positive_suffix=(
                        self._assistant_text(conflict) or pos_sfx_default or ""
                    ),
                    negative_suffix=(
                        self._assistant_text(clean) or neg_sfx_default or ""
                    ),
                    task_id=clean.get("task_id"),
                    dataset=clean.get("dataset"),
                ))

        elif mode in ("conditional_c1", "conditional_c2"):
            conflict_key = "conflict1" if mode == "conditional_c1" else "conflict2"
            valid_keys = [
                k for k, v in groups.items() if "clean" in v and conflict_key in v
            ]
            sampled = random.sample(valid_keys, min(n_samples, len(valid_keys)))
            for key in sampled:
                clean = groups[key]["clean"]
                conflict = groups[key][conflict_key]
                self.pairs.append(RawContrastivePair(
                    positive_messages=conflict["messages"],
                    negative_messages=clean["messages"],
                    positive_images=conflict.get("images", []),
                    negative_images=clean.get("images", []),
                    positive_suffix=None,
                    negative_suffix=None,
                    task_id=clean.get("task_id"),
                    dataset=clean.get("dataset"),
                ))

        else:
            raise ValueError(f"Unknown dataset mode: {mode!r}")

    @staticmethod
    def _assistant_text(item: dict) -> str:
        for message in item.get("messages", []):
            if message.get("role") == "assistant":
                return str(message.get("content", "")).strip()
        return ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
