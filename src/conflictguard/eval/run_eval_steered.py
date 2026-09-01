"""Adapter-driven CONFLICTGUARD evaluation."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------
from ..adapters import get_adapter
from ..steering.leash_layer import LeashLayer
from ..steering.malleable_model import MalleableModel, SteeringVectorPT


# ---------------------------------------------------------------------------
# Shared evaluation helpers (model-agnostic)
# ---------------------------------------------------------------------------

def normalize_action_type(act_type: str) -> str:
    if not act_type:
        return ""
    act_type = act_type.strip().lower()
    if act_type in ("terminate", "stop"):
        return "terminate"
    if act_type in ("system_button", "press_button"):
        return "system_button"
    if act_type in ("swipe", "scroll"):
        return "swipe"
    return act_type


def infer_swipe_direction(args: dict) -> str:
    start = args.get("coordinate")
    end = args.get("coordinate2")
    if not (isinstance(start, list) and len(start) == 2
            and isinstance(end, list) and len(end) == 2):
        start = args.get("coordinate1")
        end = args.get("coordinate2")
    if not (isinstance(start, list) and len(start) == 2
            and isinstance(end, list) and len(end) == 2):
        return str(args.get("direction", "")).strip().lower()
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def calculate_f1_score(predicted_str: str, ground_truth_str: str) -> float:
    predicted_str = (predicted_str or "").replace("[", "").replace("]", "")
    ground_truth_str = (ground_truth_str or "").replace("[", "").replace("]", "")
    pt = set(predicted_str.lower().split())
    gt = set(ground_truth_str.lower().split())
    if len(pt) == 1 and len(gt) == 1:
        p, g = list(pt)[0], list(gt)[0]
        if p in g or g in p:
            return 1.0
    common = pt & gt
    prec = len(common) / len(pt) if pt else 0
    rec = len(common) / len(gt) if gt else 0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def is_answer_refusal_like(pred_args: dict) -> bool:
    act = normalize_action_type(pred_args.get("action", ""))
    if act not in ("answer", "calluser"):
        return False
    text = str(pred_args.get("text", pred_args.get("content", ""))).strip().lower()
    keywords = [
        "not available", "not located", "not visible", "cannot find","impossible", "not possible", "cannot", "fail", "failure","conflict", "contradictory","not feasible", "infeasible"
    ]
    return any(k in text for k in keywords)


def _normalize_pred_args_for_refusal(pred_args: Optional[dict]) -> Optional[dict]:
    if pred_args is None:
        return None
    if is_answer_refusal_like(pred_args):
        out = dict(pred_args)
        out["action"] = "terminate"
        out.setdefault("status", "failure")
        return out
    return pred_args


def evaluate_single_sample(
    pred_args: dict, gt_args: dict, img_w: int, img_h: int, instruction: str = ""
) -> Tuple[int, int]:
    pred_type = normalize_action_type(pred_args.get("action", ""))
    gt_type = normalize_action_type(gt_args.get("action_type", ""))

    if gt_type == "terminate" and is_answer_refusal_like(pred_args):
        if gt_args.get("status") == "failure":
            return 1, 1
        return 1, 0
    type_right = 1 if (pred_type == gt_type and gt_type != "") else 0
    ground_right = 0

    if type_right == 1:
        if gt_type == "click":
            px = py = gx = gy = None
            if isinstance(pred_args.get("coordinate"), (list, tuple)) and len(pred_args["coordinate"]) == 2:
                px, py = pred_args["coordinate"]
            elif pred_args.get("x") is not None and pred_args.get("y") is not None:
                px, py = pred_args["x"], pred_args["y"]
            if isinstance(gt_args.get("coordinate"), (list, tuple)) and len(gt_args["coordinate"]) == 2:
                gx, gy = gt_args["coordinate"]
            elif gt_args.get("x") is not None and gt_args.get("y") is not None:
                gx, gy = gt_args["x"], gt_args["y"]
            if None not in (px, py, gx, gy) and img_w > 0 and img_h > 0:
                dx = abs(float(px) - float(gx)) / img_w
                dy = abs(float(py) - float(gy)) / img_h
                if math.sqrt(dx ** 2 + dy ** 2) <= 0.14:
                    ground_right = 1
        elif gt_type == "swipe":
            if infer_swipe_direction(pred_args) == infer_swipe_direction(gt_args) != "":
                ground_right = 1
        elif gt_type == "system_button":
            if (str(pred_args.get("button", "")).strip().lower()
                    == str(gt_args.get("button", "")).strip().lower() != ""):
                ground_right = 1
        elif gt_type == "terminate":
            if (str(pred_args.get("status", "")).strip().lower()
                    == str(gt_args.get("status", "")).strip().lower() != ""):
                ground_right = 1
        elif gt_type == "type":
            if calculate_f1_score(str(pred_args.get("text", "")),
                                   str(gt_args.get("text", ""))) >= 0.5:
                ground_right = 1
        elif gt_type == "launch":
            pred_app = str(pred_args.get("app", "")).strip()
            inst = instruction or ""
            inst_lower = inst.lower()
            has_app_keyword = "app" in inst_lower
            if has_app_keyword:
                if pred_app and pred_app.lower() in inst_lower:
                    ground_right = 1
            else:
                gt_app = str(gt_args.get("app", "")).strip()
                if pred_app and gt_app:
                    pred_app_l = pred_app.lower()
                    gt_app_l = gt_app.lower()
                    if pred_app_l in gt_app_l or gt_app_l in pred_app_l:
                        ground_right = 1

    return type_right, ground_right


def parse_gt_action(raw_action: Any) -> Optional[Dict]:
    if raw_action is None:
        return None
    if isinstance(raw_action, list):
        if not raw_action:
            return None
        first = raw_action[0]
        if isinstance(first, dict):
            return first
        if isinstance(first, str):
            try:
                return json.loads(first)
            except Exception:
                return None
        return None
    if isinstance(raw_action, str):
        try:
            return json.loads(raw_action)
        except Exception:
            return None
    if isinstance(raw_action, dict):
        return raw_action
    return None


# ---------------------------------------------------------------------------
# Layer-selection helpers
# ---------------------------------------------------------------------------

def _parse_layer_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _top_k_variance_layers(vec: SteeringVectorPT, k: int = 5) -> List[int]:
    sorted_layers = sorted(vec.explained_variances.items(), key=lambda x: -x[1])
    return [lid for lid, _ in sorted_layers[:k]]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Steered VLM evaluation on the CONFLICT test set."
    )

    # Adapter + model
    p.add_argument("--model_adapter", type=str, required=True,
                   help="Adapter name: qwen3vl | uitars | uivenus")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--processor_path", type=str, default=None)
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=256)

    # Test data
    p.add_argument("--test_set", type=str, required=True)

    # Steering vectors — PCA (linear) .pt files
    p.add_argument("--behavior_vector", type=str, default=None,
                   help="Path to behaviour .pt file.")
    p.add_argument("--condition_c1", type=str, default=None,
                   help="Path to conditional_c1.pt  (PCA linear sentinel)")
    p.add_argument("--condition_c2", type=str, default=None,
                   help="Path to conditional_c2.pt  (PCA linear sentinel)")

    # Steering configuration
    p.add_argument("--steer_mode", type=str, default="none",
                   choices=["none", "multi"])
    p.add_argument("--behavior_strength", type=float, default=6.0)
    p.add_argument("--behavior_layers", type=str, default=None,
                   help="Comma-separated layer IDs for behaviour injection. "
                        "Defaults to top-5 variance layers from the vector.")
    p.add_argument("--condition_layers", type=str, default=None,
                   help="Legacy shared override for both condition layers.")
    p.add_argument("--condition_c1_layers", type=str, default=None,
                   help="C1 condition layers; defaults to the vector's best layer.")
    p.add_argument("--condition_c2_layers", type=str, default=None,
                   help="C2 condition layers; defaults to the vector's best layer.")
    p.add_argument("--condition_threshold", type=float, default=0.1,
                   help="Cosine-similarity threshold for PCA-based conditions.")
    p.add_argument("--condition_comparator", type=str, default="larger",
                   choices=["larger", "smaller"])
    p.add_argument("--condition_comparison_mode", type=str, default="last",
                   choices=["mean", "last"])
    p.add_argument("--use_ooi_norm", action="store_true")
    # Debug
    p.add_argument("--debug", type=int, default=0,
                   help="If > 0, only evaluate first N samples and print raw outputs.")
    # Output
    p.add_argument("--output", type=str, default=None,
                   help="Path to save per-sample results as JSON.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── Adapter ───────────────────────────────────────────────────────
    adapter = get_adapter(args.model_adapter)
    print(f"[main] Adapter: {adapter.name!r}  model_family: {adapter.model_family!r}")

    # ── Load model ────────────────────────────────────────────────────
    print(f"[main] Loading model: {args.model_path}")
    model = adapter.load_model(args.model_path, args.device_id)
    processor = adapter.load_processor(args.model_path, args.processor_path)

    # ── Wrap with MalleableModel ──────────────────────────────────────
    mm = MalleableModel(model, processor, adapter)

    # ── Load vectors ──────────────────────────────────────────────────
    bvec = SteeringVectorPT.load(args.behavior_vector) if args.behavior_vector else None
    cvec_c1 = SteeringVectorPT.load(args.condition_c1) if args.condition_c1 else None
    cvec_c2 = SteeringVectorPT.load(args.condition_c2) if args.condition_c2 else None

    # ── Determine layer IDs ───────────────────────────────────────────
    if args.behavior_layers:
        b_layers = _parse_layer_list(args.behavior_layers)
    elif bvec:
        b_layers = _top_k_variance_layers(bvec, k=5)
    else:
        b_layers = []

    def _cond_layers(
        cvec: Optional[SteeringVectorPT], specific_override: Optional[str]
    ) -> List[int]:
        override = specific_override or args.condition_layers
        if override:
            return _parse_layer_list(override)
        if cvec:
            return [cvec.best_layer]
        return []

    # ── Apply steering ────────────────────────────────────────────────
    if args.steer_mode == "none":
        print("[Steering] NONE - running without activation steering.")

    elif args.steer_mode == "multi":
        assert bvec and cvec_c1 and cvec_c2, (
            "--behavior_vector, --condition_c1 and --condition_c2 all required"
        )
        c1_layers = _cond_layers(cvec_c1, args.condition_c1_layers)
        c2_layers = _cond_layers(cvec_c2, args.condition_c2_layers)
        print(f"[Steering] multi (PCA)  b_layers={b_layers}  "
              f"c1_layers={c1_layers}  c2_layers={c2_layers}  "
              f"threshold={args.condition_threshold}")
        mm.multisteer(
            behavior_vectors=[bvec],
            behavior_layer_ids=[b_layers],
            behavior_vector_strengths=[args.behavior_strength],
            condition_vectors=[cvec_c1, cvec_c2],
            condition_layer_ids=[c1_layers, c2_layers],
            condition_vector_thresholds=[args.condition_threshold,
                                         args.condition_threshold],
            condition_comparator_threshold_is=[args.condition_comparator,
                                               args.condition_comparator],
            condition_threshold_comparison_modes=[args.condition_comparison_mode,
                                                  args.condition_comparison_mode],
            rules=["if c1 or c2 then B1"],
            use_ooi_preventive_normalization=args.use_ooi_norm,
        )

    # ── Load test data ────────────────────────────────────────────────
    print(f"[main] Loading test set: {args.test_set}")
    with open(args.test_set, "r", encoding="utf-8") as f:
        full_dataset = json.load(f)

    # Pre-scan: build (dataset, task_id) → clean GT lookup for False Execution.
    # Uses the FULL dataset so conflict samples always find their pair.
    clean_gt_lookup: Dict[Any, Dict] = {}
    for _item in full_dataset:
        if _item.get("data_type", "clean") == "clean":
            _raw = parse_gt_action(_item.get("action"))
            if _raw is not None:
                if "action_type" not in _raw and "action" in _raw:
                    _raw["action_type"] = _raw["action"]
                clean_gt_lookup[(_item.get("dataset"), _item.get("task_id"))] = _raw
    print(f"[main] Clean GT lookup built: {len(clean_gt_lookup)} entries.")

    dataset = full_dataset
    if args.debug > 0:
        dataset = full_dataset[: args.debug]
        print(f"[DEBUG] Running on first {args.debug} samples only.\n")

    # Whether this run uses any condition sentinel (for trigger stats)
    has_conditions = args.steer_mode == "multi"

    # ── Evaluation loop ───────────────────────────────────────────────
    def _sentinel_fields():
        return {"sentinel_total": 0, "sentinel_c1_triggered": 0,
                "sentinel_c2_triggered": 0, "sentinel_any_triggered": 0}

    def _false_exec_fields():
        return {"false_exec_total": 0,
                "false_exec_type_cnt": 0, "false_exec_ground_cnt": 0}

    metrics: Dict[str, dict] = {
        "clean": {
            "total": 0, "type_right_cnt": 0, "ground_right_cnt": 0,
            "non_terminate_gt_total": 0, "false_reject_cnt": 0,
            "token_count_total": 0,
            **_sentinel_fields(),
        },
        "conflict1": {
            "total": 0, "type_right_cnt": 0, "ground_right_cnt": 0,
            "token_count_total": 0,
            **_sentinel_fields(), **_false_exec_fields(),
        },
        "conflict2": {
            "total": 0, "type_right_cnt": 0, "ground_right_cnt": 0,
            "token_count_total": 0,
            **_sentinel_fields(), **_false_exec_fields(),
        },
    }
    # Determine tokenizer for response token counting
    _response_tokenizer = None
    if hasattr(processor, "tokenizer"):
        _response_tokenizer = processor.tokenizer
    elif hasattr(processor, "encode"):
        _response_tokenizer = processor

    records: List[dict] = []
    sample_times: List[float] = []
    response_token_counts: List[int] = []
    inference_start = time.perf_counter()

    for idx, item in enumerate(tqdm(dataset, desc="Evaluating",
                                    disable=args.debug > 0)):
        mm.reset_per_sample()
        data_type = item.get("data_type", "clean")
        img_w = int(item.get("imgwidth", 1080))
        img_h = int(item.get("imgheight", 2400))

        gt_args = parse_gt_action(item.get("action"))
        if not gt_args:
            continue
        if "action_type" not in gt_args and "action" in gt_args:
            gt_args["action_type"] = gt_args["action"]

        # Generate one action from the current screenshot and instruction.
        t0 = time.perf_counter()
        response_text = mm.respond(
            messages=item.get("messages", []),
            image_paths=item.get("images", []),
            max_new_tokens=args.max_new_tokens,
            reset_after_response=False,
        )
        sample_times.append(time.perf_counter() - t0)

        # Count response tokens
        if _response_tokenizer is not None:
            _num_tokens = len(_response_tokenizer.encode(response_text))
        else:
            _num_tokens = len(response_text.split())
        _total_response_tokens = _num_tokens
        response_token_counts.append(_total_response_tokens)

        # ── Capture sentinel state BEFORE reset ───────────────────
        c1_met = bool(LeashLayer.condition_met[0])
        c2_met = bool(LeashLayer.condition_met[1])
        any_met = c1_met or c2_met

        # Debug: show condition trigger state
        if args.debug > 0:
            print(f"\n{'='*60}")
            print(f"[DEBUG] data_type={data_type}  "
                  f"task_id={item.get('task_id')}  dataset={item.get('dataset')}")
            c1_sim = LeashLayer.condition_similarities[0]
            c2_sim = LeashLayer.condition_similarities[1]
            print(f"  C1 triggered: {c1_met}  similarities={dict(c1_sim)}")
            print(f"  C2 triggered: {c2_met}  similarities={dict(c2_sim)}")
            print(f"  --- RAW OUTPUT ---")
            print(f"  {response_text}")

        # Parse prediction via adapter
        pred_args = adapter.parse_output(response_text)
        if pred_args is not None:
            pred_args = adapter.rescale_prediction(pred_args, img_w, img_h)

        # Evaluate against conflict GT
        if pred_args is None:
            type_right, ground_right = 0, 0
        else:
            type_right, ground_right = evaluate_single_sample(
                pred_args, gt_args, img_w, img_h, item.get("instruction", "")
            )

        # ── False Execute: compare pred against the paired clean GT ─
        false_exec_type = false_exec_ground = 0
        has_clean_ref = False
        clean_gt_for_debug: Optional[dict] = None
        if data_type in ("conflict1", "conflict2") and pred_args is not None:
            clean_gt_for_debug = clean_gt_lookup.get(
                (item.get("dataset"), item.get("task_id"))
            )
            if clean_gt_for_debug is not None:
                has_clean_ref = True
                fe_t, fe_g = evaluate_single_sample(
                    pred_args,
                    clean_gt_for_debug,
                    img_w, img_h,
                    item.get("instruction", ""),
                )
                false_exec_type   = fe_t
                false_exec_ground = fe_g

        if args.debug > 0:
            print(f"  GT (conflict): {gt_args}")
            print(f"  Pred:          {pred_args}")
            print(f"  type_right={type_right}  ground_right={ground_right}")
            if data_type in ("conflict1", "conflict2"):
                print(f"  GT (clean):    {clean_gt_for_debug}")
                print(f"  false_exec_type={false_exec_type}  "
                      f"false_exec_ground={false_exec_ground}")
            print(f"{'='*60}")

        # ── Accumulate all metrics ────────────────────────────────
        if data_type in metrics:
            m = metrics[data_type]
            m["total"]             += 1
            m["type_right_cnt"]    += type_right
            m["ground_right_cnt"]  += ground_right
            m["token_count_total"] += _total_response_tokens

            # --- Sentinel trigger accuracy ---
            if has_conditions:
                m["sentinel_total"] += 1
                if c1_met:
                    m["sentinel_c1_triggered"] += 1
                if c2_met:
                    m["sentinel_c2_triggered"] += 1
                if any_met:
                    m["sentinel_any_triggered"] += 1

            # --- False Rejection (clean only) ---
            if data_type == "clean":
                gt_type = normalize_action_type(
                    gt_args.get("action_type", gt_args.get("action", ""))
                )
                pred_args_for_stats = _normalize_pred_args_for_refusal(pred_args)
                pred_type = normalize_action_type(
                    (pred_args_for_stats or {}).get("action", "")
                )
                if gt_type != "terminate":
                    m["non_terminate_gt_total"] += 1
                    if pred_type == "terminate":
                        m["false_reject_cnt"] += 1

            # --- False Execute (conflict1 / conflict2 only) ---
            elif data_type in ("conflict1", "conflict2") and has_clean_ref:
                m["false_exec_total"]      += 1
                m["false_exec_type_cnt"]   += false_exec_type
                m["false_exec_ground_cnt"] += false_exec_ground

        records.append({
            "data_type":           data_type,
            "task_id":             item.get("task_id"),
            "dataset":             item.get("dataset"),
            "type_right":          type_right,
            "ground_right":        ground_right,
            "pred_args":           pred_args,
            "gt_args":             gt_args,
            "raw_output":          response_text,
            "response_token_count": _num_tokens,
            "total_response_token_count": _total_response_tokens,
            "c1_triggered":        c1_met,
            "c2_triggered":        c2_met,
            "false_exec_type":     false_exec_type,
            "false_exec_ground":   false_exec_ground,
        })
        mm.reset_per_sample()

    total_inference_time = time.perf_counter() - inference_start
    avg_sample_time = total_inference_time / len(sample_times) if sample_times else 0.0

    # ── Save records ──────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"\nPer-sample results saved to {out_path}")

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("FINAL EVALUATION RESULTS")
    print(f"  adapter        = {adapter.name}")
    print(f"  steer_mode     = {args.steer_mode}")
    if bvec:
        print(f"  behavior_vec   = {args.behavior_vector}")
        print(f"  behavior_layers= {b_layers}")
        print(f"  strength       = {args.behavior_strength}")
    if args.steer_mode == "multi":
        print(f"  threshold (cos)= {args.condition_threshold}")
        print(f"  comparator     = {args.condition_comparator}")
    avg_response_tokens = (sum(response_token_counts) / len(response_token_counts)
                           if response_token_counts else 0.0)
    print(f"  samples        = {len(sample_times)}")
    print(f"  total_time     = {total_inference_time:.2f}s  "
          f"({total_inference_time/60:.2f} min)")
    print(f"  avg_per_sample = {avg_sample_time:.3f}s")
    print(f"  avg_resp_tokens= {avg_response_tokens:.1f}  "
          f"(total={sum(response_token_counts)})")
    print(f"{'='*50}")

    for d_type, stats in metrics.items():
        total = stats["total"]
        if total == 0:
            continue
        type_acc   = stats["type_right_cnt"]   / total * 100
        ground_acc = stats["ground_right_cnt"] / total * 100
        avg_tokens = stats["token_count_total"] / total
        print(f"[{d_type.upper()}] Total={total}")
        print(f"  Action Type Accuracy : {type_acc:.2f}%")
        print(f"  Action Param Accuracy: {ground_acc:.2f}%")
        print(f"  Avg Response Tokens  : {avg_tokens:.1f}")

        # ── False Rejection (clean) ───────────────────────────────
        if d_type == "clean":
            denom  = stats.get("non_terminate_gt_total", 0)
            fr_cnt = stats.get("false_reject_cnt", 0)
            fr_rate = (fr_cnt / denom * 100) if denom > 0 else 0.0
            print(f"  False Rejection Rate : {fr_rate:.2f}%  ({fr_cnt}/{denom})")

        # ── False Execute (conflict1 / conflict2) ─────────────────
        if d_type in ("conflict1", "conflict2"):
            fe_total = stats.get("false_exec_total", 0)
            if fe_total > 0:
                fe_type_rate   = stats["false_exec_type_cnt"]   / fe_total * 100
                fe_ground_rate = stats["false_exec_ground_cnt"] / fe_total * 100
                print(f"  False Execute Type   : {fe_type_rate:.2f}%"
                      f"  ({stats['false_exec_type_cnt']}/{fe_total})")
                print(f"  False Execute Ground : {fe_ground_rate:.2f}%"
                      f"  ({stats['false_exec_ground_cnt']}/{fe_total})")

        # ── Sentinel trigger accuracy ─────────────────────────────
        if has_conditions:
            s_total = stats.get("sentinel_total", 0)
            if s_total > 0:
                c1_rate  = stats["sentinel_c1_triggered"]  / s_total * 100
                c2_rate  = stats["sentinel_c2_triggered"]  / s_total * 100
                any_rate = stats["sentinel_any_triggered"] / s_total * 100
                if d_type == "clean":
                    # Correct behaviour = NO trigger (false alarm = any trigger)
                    correct_rate = 100.0 - any_rate
                    print(f"  Sentinel C1  Trigger : {c1_rate:.2f}%"
                          f"  ({stats['sentinel_c1_triggered']}/{s_total})")
                    print(f"  Sentinel C2  Trigger : {c2_rate:.2f}%"
                          f"  ({stats['sentinel_c2_triggered']}/{s_total})")
                    print(f"  Sentinel False Alarm : {any_rate:.2f}%"
                          f"  ({stats['sentinel_any_triggered']}/{s_total})  "
                          f"[correct={correct_rate:.2f}%]")
                else:
                    # Correct behaviour = any trigger (missed = no trigger)
                    missed     = s_total - stats["sentinel_any_triggered"]
                    miss_rate  = missed / s_total * 100
                    print(f"  Sentinel C1  Trigger : {c1_rate:.2f}%"
                          f"  ({stats['sentinel_c1_triggered']}/{s_total})")
                    print(f"  Sentinel C2  Trigger : {c2_rate:.2f}%"
                          f"  ({stats['sentinel_c2_triggered']}/{s_total})")
                    print(f"  Sentinel Any Trigger : {any_rate:.2f}%"
                          f"  ({stats['sentinel_any_triggered']}/{s_total})"
                          f"  [missed={miss_rate:.2f}%]")

        print("-" * 50)


if __name__ == "__main__":
    main()
