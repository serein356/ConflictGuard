"""
adapters/uitars.py
===================
ModelAdapter for UI-TARS models (Qwen2.5-VL base).

Key properties:
  - Model class: Qwen2_5_VLForConditionalGeneration (fallback to AutoModelForImageTextToText)
  - Output format: Thought/Action with UI-TARS function calls
  - Coordinate range: absolute pixel coordinates (no rescaling)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
except ImportError:
    try:
        from transformers import (  # type: ignore
            AutoModelForImageTextToText as Qwen2_5_VLForConditionalGeneration,
            AutoProcessor,
        )
    except Exception:  # pragma: no cover
        from transformers import (  # type: ignore
            AutoModelForImageTextToText,
            AutoProcessor,
        )
        Qwen2_5_VLForConditionalGeneration = AutoModelForImageTextToText

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    def process_vision_info(messages):  # type: ignore
        raise ImportError("qwen_vl_utils is required: pip install qwen-vl-utils")

from .base import ModelAdapter
from ..paths import normalize_image_path

_POSITIVE_SUFFIX = (
    "Thought: {language} The instruction is contradictory."
    "\nAction: finished(content='failure')"
)

_NEGATIVE_SUFFIX = (
    "Thought: {language} The instruction is valid."
    "\nAction: click(point='(500,500)')"
)


class UITARSAdapter(ModelAdapter):
    """ModelAdapter for UI-TARS models."""

    name = "uitars"
    model_family = "qwen2_5vl"

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
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, attn_implementation="flash_attention_2", **load_kw
            )
        except Exception:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            return model.model.language_model.layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
            return model.language_model.layers
        raise AttributeError("Could not locate decoder layers in UI-TARS model.")

    def num_hidden_layers(self, model: Any) -> int:
        cfg = model.config
        if getattr(cfg, "num_hidden_layers", None) is not None:
            return int(cfg.num_hidden_layers)
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "num_hidden_layers"):
            return int(text_cfg.num_hidden_layers)
        if getattr(cfg, "num_layers", None) is not None:
            return int(cfg.num_layers)
        raise AttributeError("Cannot resolve hidden layer count from UI-TARS config.")

    def hidden_size(self, model: Any) -> int:
        cfg = model.config
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "hidden_size"):
            return int(text_cfg.hidden_size)
        if hasattr(cfg, "hidden_size"):
            return int(cfg.hidden_size)
        raise AttributeError("Cannot resolve hidden_size from UI-TARS config.")

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
                user_text = str(m.get("content", "")).replace("<image>", "").strip()
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
        n_sfx = len(processor.tokenizer.encode(suffix, add_special_tokens=False))
        seq_len = full_input_ids.shape[-1]
        n_sfx = max(1, min(n_sfx, seq_len))
        return seq_len - n_sfx, seq_len

    # ── output parsing ─────────────────────────────────────────────────

    _FAILURE_KEYWORDS = frozenset({
        "failure", "task failed", "cannot complete", "cannot execute",
        "contradictory", "conflict", "not available", "unavailable",
        "not located", "not visible", "cannot find", "impossible",
        "not possible", "cannot", "fail", "not feasible", "infeasible","error","无法完成", "失败", "无法执行", "矛盾", "冲突", "不可用", "未找到", "不可能", "不可行", "无法", "不可完成"
    })

    @staticmethod
    def _extract_action_text(response_text: str) -> str:
        m = re.search(r"(?:^|\n)\s*Action\s*:\s*(.+)$",
                      response_text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip()
        return (response_text or "").strip()

    @staticmethod
    def _extract_two_numbers(text: str) -> Optional[List[float]]:
        if not text:
            return None
        nums = re.findall(r"[-+]?\d*\.?\d+", text)
        if len(nums) < 2:
            return None
        try:
            return [float(nums[0]), float(nums[1])]
        except Exception:
            return None

    @staticmethod
    def _extract_point(text: str) -> Optional[List[float]]:
        m = re.search(r"<point>\s*([\-\d.]+)\s+([\-\d.]+)\s*</point>", text)
        if m:
            try:
                return [float(m.group(1)), float(m.group(2))]
            except Exception:
                return None
        return UITARSAdapter._extract_two_numbers(text)

    @staticmethod
    def _extract_quoted_value(text: str, key: str) -> str:
        m = re.search(
            rf"{re.escape(key)}\s*=\s*(['\"])(.*?)\1",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return m.group(2).strip() if m else ""

    def parse_output(self, response_text: str) -> Optional[Dict[str, Any]]:
        action_text = self._extract_action_text(response_text)
        if not action_text:
            return None

        # click
        # 解析格式：click(point='(714,1967)') / click(start_box='(714,1404)')
        m = re.search(
            r"click\s*\(\s*(?:point|start_box)\s*=\s*(['\"])(.*?)\1\s*\)",
            action_text, re.IGNORECASE | re.DOTALL,
        )
        if m:
            pt = self._extract_point(m.group(2))
            if pt is not None:
                return {"action": "click", "coordinate": pt}
        # fallback: click with unquoted point
        m = re.search(r"click\s*\(\s*(?:point|start_box)\s*=\s*(.*?)\s*\)",
                      action_text, re.IGNORECASE | re.DOTALL)
        if m:
            pt = self._extract_point(m.group(1))
            if pt is not None:
                return {"action": "click", "coordinate": pt}

        # long_press
        m = re.search(r"long_press\s*\(\s*point\s*=\s*'([^']*)'\s*\)",
                      action_text, re.IGNORECASE)
        if m:
            pt = self._extract_point(m.group(1))
            if pt is not None:
                return {"action": "long_press", "coordinate": pt}

        # drag
        # 解析格式：drag(start_point='(x,y)', end_point='(x,y)') / drag(start_box='(x,y)', end_box='(x,y)')
        m = re.search(
            r"drag\s*\(\s*(?:start_point|start_box)\s*=\s*(['\"])(.*?)\1\s*,\s*"
            r"(?:end_point|end_box)\s*=\s*(['\"])(.*?)\3\s*\)",
            action_text, re.IGNORECASE | re.DOTALL,
        )
        if m:
            p1 = self._extract_point(m.group(2))
            p2 = self._extract_point(m.group(4))
            if p1 is not None and p2 is not None:
                return {"action": "swipe", "coordinate1": p1, "coordinate2": p2}

        # scroll
        # 解析格式：scroll(direction='left', start_box='(x,y)') / scroll(direction='down', point='(x,y)')
        m = re.search(
            r"scroll\s*\(.*?direction\s*=\s*(['\"])(.*?)\1",
            action_text, re.IGNORECASE | re.DOTALL,
        )
        if m:
            direction = m.group(2).strip().lower()
            return {"action": "swipe", "direction": direction}

        # type
        m = re.search(r"type\s*\(\s*content\s*=\s*(['\"])(.*?)\1\s*\)",
                      action_text, re.IGNORECASE | re.DOTALL)
        if m:
            return {"action": "type", "text": m.group(2)}

        # open_app
        m = re.search(r"open_app\s*\(\s*app_name\s*=\s*(['\"])(.*?)\1\s*\)",
                      action_text, re.IGNORECASE | re.DOTALL)
        if m:
            return {"action": "launch", "app": m.group(2)}

        # press_home / press_back
        if re.search(r"\bpress_home\s*\(\s*\)\b", action_text, re.IGNORECASE):
            return {"action": "system_button", "button": "home"}
        if re.search(r"\bpress_back\s*\(\s*\)\b", action_text, re.IGNORECASE):
            return {"action": "system_button", "button": "back"}

        # finished
        if re.search(r"\bfinished\s*\(", action_text, re.IGNORECASE):
            content = self._extract_quoted_value(action_text, "content").strip().lower()
            if any(k in content for k in self._FAILURE_KEYWORDS):
                return {"action": "terminate", "status": "failure"}
            if content in {"success", "done", "completed"}:
                return {"action": "terminate", "status": "success"}
            return {"action": "terminate", "status": "failure" if content else "success"}

        return None

    def rescale_prediction(
        self, pred_args: Dict[str, Any], img_w: int, img_h: int
    ) -> Dict[str, Any]:
        return pred_args
