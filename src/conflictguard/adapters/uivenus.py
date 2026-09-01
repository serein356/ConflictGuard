"""
adapters/uivenus.py
===================
ModelAdapter for UI-Venus models.

Key properties:
  - Model class: AutoModelForImageTextToText
  - Output format: <action>Click(box=(x,y))</action> etc.
  - Coordinate range: [0, 999] → absolute pixels (same as Qwen3-VL)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
except ImportError:  # pragma: no cover
    from transformers import (  # type: ignore
        AutoModelForImageTextToText,
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
    "<think>The task is contradictory.</think>\n"
    "<action>Finished(content='failure')</action>\n"
    "<conclusion>The task is contradictory.</conclusion>"
)

_NEGATIVE_SUFFIX = (
    "<think>The task is valid.</think>\n"
    "<action>Click(box=(500,500))</action>\n"
    "<conclusion>The task is valid.</conclusion>"
)


class UIVenusAdapter(ModelAdapter):
    """ModelAdapter for UI-Venus models."""

    name = "uivenus"
    model_family = "uivenus"

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
            model = AutoModelForImageTextToText.from_pretrained(
                model_path, attn_implementation="flash_attention_2", **load_kw
            )
        except Exception:
            model = AutoModelForImageTextToText.from_pretrained(
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
        raise AttributeError("Could not locate decoder layers in UI-Venus model.")

    def num_hidden_layers(self, model: Any) -> int:
        cfg = model.config
        if getattr(cfg, "num_hidden_layers", None) is not None:
            return int(cfg.num_hidden_layers)
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "num_hidden_layers"):
            return int(text_cfg.num_hidden_layers)
        if getattr(cfg, "num_layers", None) is not None:
            return int(cfg.num_layers)
        raise AttributeError("Cannot resolve hidden layer count from UI-Venus config.")

    def hidden_size(self, model: Any) -> int:
        cfg = model.config
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "hidden_size"):
            return int(text_cfg.hidden_size)
        if hasattr(cfg, "hidden_size"):
            return int(cfg.hidden_size)
        raise AttributeError("Cannot resolve hidden_size from UI-Venus config.")

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

    @staticmethod
    def _parse_number(val: str) -> Optional[float]:
        try:
            return float(val.strip())
        except Exception:
            return None

    @staticmethod
    def _extract_action_text(response_text: str) -> str:
        m = re.search(r"<action>\s*(.*?)\s*</action>", response_text,
                      re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
        m = re.search(r"\bAction\s*:\s*(.*)", response_text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        m = re.search(r"<action>\s*(.*?)\s*</conclusion>", response_text,
                      re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
        return response_text.strip()

    @staticmethod
    def _extract_quoted_value(text: str, key: str) -> str:
        m = re.search(
            rf"{re.escape(key)}\s*=\s*(['\"])(.*?)\1",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return m.group(2).strip() if m else ""
    
    @staticmethod
    def _extract_tag_text(text: str, tag: str) -> str:
        if not text:
            return ""
        m = re.search(rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>",
                      text, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else ""

    # ── failure keywords (shared across parse paths) ───────────────────
    _FAILURE_KEYWORDS = frozenset({
        "failure", "task failed", "cannot complete", "cannot execute",
        "contradictory", "conflict", "not available", "unavailable",
        "not located", "not visible", "cannot find", "impossible",
        "not possible", "cannot", "fail", "not feasible", "infeasible","error"
    })

    @staticmethod
    def _try_fix_parens(text: str) -> str:
        """Best-effort: append missing closing parentheses so regex can match."""
        open_count = text.count("(") - text.count(")")
        if open_count > 0:
            text = text + ")" * open_count
        return text

    @staticmethod
    def _extract_quoted_value_robust(text: str, key: str) -> str:
        """Extract key='...' / key="..." tolerating unclosed quotes at EOL."""
        # Standard closed-quote match
        m = re.search(
            rf"{re.escape(key)}\s*=\s*(['\"])(.*?)\1",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if m:
            return m.group(2).strip()
        # Fallback: quote opened but not closed (content runs to end / next paren)
        m = re.search(
            rf"{re.escape(key)}\s*=\s*['\"]([^'\")\n]*)",
            text,
            flags=re.IGNORECASE,
        )
        return m.group(1).strip() if m else ""

    def parse_output(self, response_text: str) -> Optional[Dict[str, Any]]:
        action_text = self._extract_action_text(response_text)
        if not action_text:
            return None

        # ── Handle <action>-1</action>: check <conclusion> for failure intent ──
        action_tag_text = self._extract_tag_text(response_text, "action") or action_text
        # Use a non-word-boundary pattern because '-' is not a \w char
        if re.search(r"(?<!\d)-1(?!\d)", action_tag_text):
            conclusion_text = self._extract_tag_text(response_text, "conclusion")
            status_text = (conclusion_text or response_text).strip().lower()
            if any(k in status_text for k in self._FAILURE_KEYWORDS):
                return {"action": "terminate", "status": "failure"}

        # Normalise the action text: attempt to fix unclosed parentheses so the
        # patterns below have a better chance of matching even truncated outputs.
        action_text_fixed = self._try_fix_parens(action_text)

        # Click
        m = re.search(
            r"Click\s*\(\s*box\s*=\s*\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)?\s*\)?",
            action_text_fixed, re.IGNORECASE,
        )
        if m:
            x = self._parse_number(m.group(1))
            y = self._parse_number(m.group(2))
            if x == -1 and y == -1:
                return {"action": "terminate", "status": "failure"}
            elif x is not None and y is not None:
                return {"action": "click", "coordinate": [x, y]}
            else:
                return None

        # LongPress
        m = re.search(
            r"LongPress\s*\(\s*box\s*=\s*\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)?\s*\)?",
            action_text_fixed, re.IGNORECASE,
        )
        if m:
            x = self._parse_number(m.group(1))
            y = self._parse_number(m.group(2))
            if x is not None and y is not None:
                return {"action": "long_press", "coordinate": [x, y]}

        # Drag / Scroll with coordinates
        m = re.search(
            r"(Drag|Scroll)\s*\(\s*start\s*=\s*\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)?\s*"
            r",\s*end\s*=\s*\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)?\s*\)?",
            action_text_fixed, re.IGNORECASE,
        )
        if m:
            x1 = self._parse_number(m.group(2))
            y1 = self._parse_number(m.group(3))
            x2 = self._parse_number(m.group(4))
            y2 = self._parse_number(m.group(5))
            if None not in (x1, y1, x2, y2):
                return {
                    "action": "swipe",
                    "coordinate1": [x1, y1],
                    "coordinate2": [x2, y2],
                }

        # Scroll direction (direction='...' form)
        m = re.search(
            r"Scroll\s*\(.*?direction\s*=\s*(['\"])(.*?)\1",
            action_text_fixed, re.IGNORECASE | re.DOTALL,
        )
        if m:
            return {"action": "swipe", "direction": m.group(2).strip().lower()}

        # Type
        if re.search(r"^Type\s*\(", action_text_fixed, re.IGNORECASE):
            text = self._extract_quoted_value_robust(action_text_fixed, "content")
            return {"action": "type", "text": text}

        # Launch
        if re.search(r"^Launch\s*\(", action_text_fixed, re.IGNORECASE):
            app = self._extract_quoted_value_robust(action_text_fixed, "app")
            return {"action": "launch", "app": app}

        # Wait — tolerate missing closing paren
        if re.match(r"^Wait\s*\(\s*\)?\s*$", action_text_fixed, re.IGNORECASE):
            return {"action": "wait"}

        # Finished — tolerate missing closing paren
        if re.search(r"^Finished\s*\(", action_text_fixed, re.IGNORECASE):
            status = self._extract_quoted_value_robust(action_text_fixed, "content")
            if not status:
                status = action_text_fixed.strip().lower()

            if any(k in status for k in self._FAILURE_KEYWORDS):
                status = "failure"
            else:
                status = "success"

            return {"action": "terminate", "status": status}

        # CallUser — tolerate missing closing paren
        if re.search(r"^CallUser\s*\(", action_text_fixed, re.IGNORECASE):
            text = self._extract_quoted_value_robust(action_text_fixed, "content")
            return {"action": "answer", "text": text}

        # System buttons — tolerate missing closing paren
        if re.match(r"^PressBack\s*\(\s*\)?\s*$", action_text_fixed, re.IGNORECASE):
            return {"action": "system_button", "button": "back"}
        if re.match(r"^PressHome\s*\(\s*\)?\s*$", action_text_fixed, re.IGNORECASE):
            return {"action": "system_button", "button": "home"}
        if re.match(r"^PressEnter\s*\(\s*\)?\s*$", action_text_fixed, re.IGNORECASE):
            return {"action": "system_button", "button": "enter"}
        if re.match(r"^PressRecent\s*\(\s*\)?\s*$", action_text_fixed, re.IGNORECASE):
            return {"action": "system_button", "button": "menu"}

        return None

    def rescale_prediction(
        self, pred_args: Dict[str, Any], img_w: int, img_h: int
    ) -> Dict[str, Any]:
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
