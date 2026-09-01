"""
adapters/qwen3vl.py
====================
ModelAdapter for Qwen3-VL-8B-Instruct.

Encapsulates all Qwen3-VL-specific logic:
  - model loading via Qwen3VLForConditionalGeneration
  - input formatting via qwen_vl_utils.process_vision_info
  - decoder layer path: model.model.language_model.layers
  - output parsing: JSON <tool_call> format
  - coordinate rescaling: [0, 999] → absolute pixels
  - behavior-vector suffixes in Qwen3 tool_call format
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
except ImportError:
    from transformers import (  # type: ignore
        AutoModelForImageTextToText as Qwen3VLForConditionalGeneration,
        AutoProcessor,
    )

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    def process_vision_info(messages):  # type: ignore
        raise ImportError("qwen_vl_utils is required: pip install qwen-vl-utils")

from .base import ModelAdapter
from ..paths import normalize_image_path

_POSITIVE_SUFFIX = (
    "\nThought: The instruction is invalid.\n"
    "Action: Stop the task.\n"
    "<tool_call>\n"
    '{"name": "mobile_use", "arguments": {"action": "terminate", "status": "failure"}}\n'
    "</tool_call>"
)

_NEGATIVE_SUFFIX = (
    "\nThought: The instruction is valid.\n"
    "Action: Proceed with execution.\n"
    "<tool_call>\n"
    '{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [500, 500]}}\n'
    "</tool_call>"
)


class Qwen3VLAdapter(ModelAdapter):
    """ModelAdapter for Qwen3-VL-8B-Instruct."""

    name = "qwen3vl"
    model_family = "qwen3vl"

    # ── default suffixes ───────────────────────────────────────────────

    @property
    def default_positive_suffix(self) -> str:
        return _POSITIVE_SUFFIX

    @property
    def default_negative_suffix(self) -> str:
        return _NEGATIVE_SUFFIX

    # ── loading ────────────────────────────────────────────────────────

    def load_model(self, model_path: str, device_id: int = 0, **kwargs) -> Any:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device_map = {"": f"cuda:{device_id}"} if torch.cuda.is_available() else "cpu"
        load_kw = dict(torch_dtype=dtype, device_map=device_map,
                       trust_remote_code=True, **kwargs)
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, attn_implementation="flash_attention_2", **load_kw
            )
        except Exception:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, **load_kw
            )
        model.eval()
        return model

    def load_processor(self, model_path: str, processor_path: Optional[str] = None) -> Any:
        return AutoProcessor.from_pretrained(
            processor_path or model_path, trust_remote_code=True
        )

    # ── layer introspection ────────────────────────────────────────────

    def get_decoder_layers(self, model: Any) -> nn.ModuleList:
        """Qwen3VLForConditionalGeneration → model.model.language_model.layers"""
        return model.model.language_model.layers

    def num_hidden_layers(self, model: Any) -> int:
        cfg = model.config
        if getattr(cfg, "num_hidden_layers", None) is not None:
            return int(cfg.num_hidden_layers)
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "num_hidden_layers"):
            return int(text_cfg.num_hidden_layers)
        if getattr(cfg, "num_layers", None) is not None:
            return int(cfg.num_layers)
        raise AttributeError(
            "Cannot resolve hidden layer count from Qwen3-VL config. "
            "Expected: config.num_hidden_layers or config.text_config.num_hidden_layers"
        )

    def hidden_size(self, model: Any) -> int:
        cfg = model.config
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "hidden_size"):
            return int(text_cfg.hidden_size)
        if hasattr(cfg, "hidden_size"):
            return int(cfg.hidden_size)
        raise AttributeError("Cannot resolve hidden_size from Qwen3-VL config.")

    # ── input preparation ──────────────────────────────────────────────

    @staticmethod
    def _norm_img(img_path: str) -> str:
        return normalize_image_path(img_path)

    def build_messages(
        self,
        raw_messages: List[dict],
        image_paths: List[str],
        mode: str = "generate",
        suffix: Optional[str] = None,
    ) -> List[dict]:
        user_text = ""
        for m in raw_messages:
            if m.get("role") == "user":
                user_text = m.get("content", "").replace("<image>", "").strip()
                break
        content: List[dict] = [
            {"type": "image", "image": self._norm_img(p)} for p in (image_paths or [])
        ]
        content.append({"type": "text", "text": user_text})
        messages: List[dict] = [{"role": "user", "content": content}]
        if mode == "suffix" and suffix:
            messages.append({"role": "assistant", "content": suffix})
        return messages

    def apply_chat_template(
        self, processor: Any, messages: List[dict], add_generation_prompt: bool
    ) -> str:
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def build_inputs(
        self, processor: Any, messages: List[dict], text: str, device: torch.device
    ) -> dict:
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        )
        return inputs.to(device)

    # ── hidden-state extraction ────────────────────────────────────────

    def get_hidden_states(self, outputs: Any) -> Tuple:
        return outputs.hidden_states

    def suffix_token_span(
        self,
        processor: Any,
        full_input_ids: torch.Tensor,
        suffix: str,
    ) -> Tuple[int, int]:
        """Approximate suffix span via independent tokenisation of the suffix."""
        n_sfx = len(processor.tokenizer.encode(suffix, add_special_tokens=False))
        seq_len = full_input_ids.shape[-1]
        n_sfx = max(1, min(n_sfx, seq_len))
        return seq_len - n_sfx, seq_len

    # ── output parsing ─────────────────────────────────────────────────

    def parse_output(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse Qwen3-VL <tool_call>…</tool_call> JSON output."""
        match = re.search(
            r"<tool_call>(.*?)</tool_call>", response_text, re.DOTALL | re.IGNORECASE
        )
        if not match:
            return None
        try:
            return json.loads(match.group(1).strip()).get("arguments", {})
        except json.JSONDecodeError:
            return None

    def rescale_prediction(
        self, pred_args: Dict[str, Any], img_w: int, img_h: int
    ) -> Dict[str, Any]:
        """Rescale [0, 999] predicted coordinates → absolute pixels."""
        if not isinstance(pred_args, dict) or img_w <= 0 or img_h <= 0:
            return pred_args
        out = dict(pred_args)
        for k in ("coordinate", "coordinate1", "coordinate2"):
            v = out.get(k)
            if isinstance(v, (list, tuple)) and len(v) == 2:
                try:
                    out[k] = [
                        round(float(v[0]) / 999 * img_w),
                        round(float(v[1]) / 999 * img_h),
                    ]
                except Exception:
                    pass
        if out.get("x") is not None and out.get("y") is not None:
            try:
                out["x"] = round(float(out["x"]) / 999 * img_w)
                out["y"] = round(float(out["y"]) / 999 * img_h)
            except Exception:
                pass
        return out
