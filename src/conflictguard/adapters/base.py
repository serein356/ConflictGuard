"""
adapters/base.py
================
Abstract ModelAdapter interface.

Every model that plugs into the conflictguard framework must subclass
ModelAdapter and implement every abstract method.  The core machinery
(LeashLayer, MalleableModel, VectorExtractor, Evaluator) never imports
model-specific classes directly; it only interacts with models through
this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class ModelAdapter(ABC):
    """
    Abstract interface for plugging a new VLM into conflictguard.

    Subclasses must set class attributes:
        name         - short lowercase identifier, e.g. "qwen3vl"
        model_family - base architecture family, e.g. "qwen3vl"
    """

    name: str = "base"
    model_family: str = "base"

    # ─────────────────────────────────────────────────────────────────
    # Model loading
    # ─────────────────────────────────────────────────────────────────

    @abstractmethod
    def load_model(self, model_path: str, device_id: int = 0, **kwargs) -> Any:
        """Load and return the HuggingFace model (already .eval())."""

    @abstractmethod
    def load_processor(self, model_path: str, processor_path: Optional[str] = None) -> Any:
        """Load and return the processor / tokenizer."""

    # ─────────────────────────────────────────────────────────────────
    # Layer introspection
    # ─────────────────────────────────────────────────────────────────

    @abstractmethod
    def get_decoder_layers(self, model: Any) -> nn.ModuleList:
        """Return the nn.ModuleList of decoder layers (for LeashLayer wrapping)."""

    @abstractmethod
    def num_hidden_layers(self, model: Any) -> int:
        """Return the total number of transformer decoder layers."""

    @abstractmethod
    def hidden_size(self, model: Any) -> int:
        """Return the hidden dimension of decoder layers."""

    # ─────────────────────────────────────────────────────────────────
    # Input preparation
    # ─────────────────────────────────────────────────────────────────

    @abstractmethod
    def build_messages(
        self,
        raw_messages: List[dict],
        image_paths: List[str],
        mode: str = "generate",
        suffix: Optional[str] = None,
    ) -> List[dict]:
        """
        Convert raw dataset messages + image_paths into the model-specific
        message list.

        Parameters
        ----------
        raw_messages
            Original messages list from the dataset (with role/content keys).
        image_paths
            Absolute paths to screenshot images.
        mode
            "generate" – build user message only (caller will set
                          add_generation_prompt=True).
            "suffix"   – append an assistant turn containing *suffix* so that
                          hidden states can be extracted over suffix tokens.
        suffix
            Text to place in the assistant turn (only used when mode="suffix").
        """

    @abstractmethod
    def apply_chat_template(
        self,
        processor: Any,
        messages: List[dict],
        add_generation_prompt: bool,
    ) -> str:
        """Apply the chat template and return the full prompt string."""

    @abstractmethod
    def build_inputs(
        self,
        processor: Any,
        messages: List[dict],
        text: str,
        device: torch.device,
    ) -> dict:
        """Run processor(text, images, …) and return a tensor dict ready for model()."""

    # ─────────────────────────────────────────────────────────────────
    # Hidden-state extraction helpers
    # ─────────────────────────────────────────────────────────────────

    @abstractmethod
    def get_hidden_states(self, outputs: Any) -> Tuple:
        """
        Extract the hidden_states tuple from model(**inputs, output_hidden_states=True).

        Returns
        -------
        tuple, length (num_layers + 1)
            Index 0 = embedding output.
            Index L+1 = output of layer L  (the common convention).
        """

    def layer_output_index(self, layer_id: int) -> int:
        """
        Map zero-based layer_id to the index in the hidden_states tuple.
        Default: hidden_states[layer_id + 1].  Override if model differs.
        """
        return layer_id + 1

    @abstractmethod
    def suffix_token_span(
        self,
        processor: Any,
        full_input_ids: torch.Tensor,
        suffix: str,
    ) -> Tuple[int, int]:
        """
        Return (start, end) token indices for the suffix within the full
        tokenised sequence, so hidden states can be averaged over only the
        suffix region.

        Parameters
        ----------
        processor
            The model's processor / tokenizer.
        full_input_ids
            Shape [1, seq_len] – token ids of the full prompt+suffix text.
        suffix
            The raw suffix string that was appended as an assistant turn.
        """

    # ─────────────────────────────────────────────────────────────────
    # Output parsing
    # ─────────────────────────────────────────────────────────────────

    @abstractmethod
    def parse_output(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Parse model-generated text and return a normalised action dict,
        or None when parsing fails.

        The returned dict must contain at least {"action": "<type>"}.
        Coordinate values should be in the model's *native* range
        (rescale_prediction converts them to absolute pixels).
        """

    @abstractmethod
    def rescale_prediction(
        self,
        pred_args: Dict[str, Any],
        img_w: int,
        img_h: int,
    ) -> Dict[str, Any]:
        """
        Rescale predicted coordinates from the model's native coordinate
        space to absolute pixel values.
        """

    # ─────────────────────────────────────────────────────────────────
    # Generation helpers (default implementations, can be overridden)
    # ─────────────────────────────────────────────────────────────────

    def generate(
        self,
        model: Any,
        inputs: dict,
        max_new_tokens: int = 256,
    ) -> Any:
        """Run model.generate() with standard greedy-decoding settings."""
        with torch.no_grad():
            return model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

    def decode_response(
        self,
        processor: Any,
        outputs: Any,
        input_length: int,
    ) -> str:
        """Decode generated token ids to a string."""
        generated_ids = outputs[:, input_length:]
        return processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    # ─────────────────────────────────────────────────────────────────
    # Behaviour-vector suffixes (used by ConflictDataset)
    # ─────────────────────────────────────────────────────────────────

    @property
    def default_positive_suffix(self) -> str:
        """
        Positive suffix used by fixed-strategy behavior calibration.
        Represents the desired "refuse / terminate" response.
        Override per adapter.
        """
        raise NotImplementedError(
            f"Adapter '{self.name}' must define default_positive_suffix"
        )

    @property
    def default_negative_suffix(self) -> str:
        """
        Negative suffix used by fixed-strategy behavior calibration.
        Represents the "proceed / comply" baseline response.
        Override per adapter.
        """
        raise NotImplementedError(
            f"Adapter '{self.name}' must define default_negative_suffix"
        )
