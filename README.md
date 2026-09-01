# CONFLICTGUARD

Core implementation of **Do GUI Agents Know When Not to Act? Enabling
Conflict-Aware Termination for Multimodal GUI Agents**.

This repository contains only the CONFLICTGUARD method and its evaluator:

1. extract C1/C2 conflict-condition directions from feasible-conflict pairs;
2. extract an anti-overcompliance behavior direction contrasting termination
   and execution responses;
3. add a feasibility-verification prompt to the model-native action prompt;
4. trigger activation steering when either conflict-direction cosine score
   exceeds its calibrated threshold;
5. inject the termination-oriented direction into selected decoder layers.

It supports the paper's intervened model families: Qwen3-VL, UI-TARS, and
UI-Venus.

## Installation

```bash
python -m pip install -e '.[qwen,dev]'
```

The paper environment used Python 3.10, CUDA 12.4, PyTorch 2.8.0, Transformers 5.2.0, and NVIDIA A800 80GB GPUs.

## Inputs

The code expects model-adapted JSON files containing paired `clean`,
`conflict1`, and `conflict2` rows. Each row must provide `messages`, `images`,
`action`, `task_id`, `dataset`, `data_type`, `imgwidth`, and `imgheight`.

Set `CONFLICTGUI_ROOT` when image paths are relative:

```bash
export CONFLICTGUI_ROOT=/path/to/prepared/conflictgui-workspace
```

## Extract vectors

Calibration uses 300 feasible-C1 pairs and 300 feasible-C2 pairs. PCA is fit
at every decoder layer and oriented so positive conflict/termination examples
have larger projections. Behavior calibration uses paired conflict/clean
assistant responses for Qwen3-VL and adapter-native fixed response suffixes
for UI-TARS and UI-Venus.

```bash
scripts/extract_vectors.sh qwen3vl \
  /path/to/Qwen3-VL-8B-Instruct \
  /path/to/qwen3vl_calibration_feasibility.json \
  outputs/qwen3vl-8b-vectors 0
```

This writes `behavior.pt`, `conditional_c1.pt`, and
`conditional_c2.pt`.

## Evaluate

Vanilla inference uses the original prompt data:

```bash
scripts/evaluate.sh vanilla qwen3vl-8b \
  /path/to/Qwen3-VL-8B-Instruct \
  /path/to/qwen3vl_test_vanilla.json \
  outputs/qwen3vl_8b_vanilla.json 0
```

Feasibility prompting uses the enhanced prompt data without steering:

```bash
scripts/evaluate.sh prompt qwen3vl-8b \
  /path/to/Qwen3-VL-8B-Instruct \
  /path/to/qwen3vl_test_feasibility.json \
  outputs/qwen3vl_8b_prompt.json 0
```

Full CONFLICTGUARD uses feasibility prompting and released steering vectors:

```bash
scripts/evaluate.sh conflictguard qwen3vl-8b \
  /path/to/Qwen3-VL-8B-Instruct \
  /path/to/qwen3vl_test_feasibility.json \
  outputs/qwen3vl_8b_conflictguard.json 0
```

Paper hyperparameters for Qwen3-VL-4B/8B/32B, UI-Venus-1.5-8B, and
UI-TARS-1.5-7B are stored in `configs/paper_hyperparameters.json`.

## Evaluation rules

- Click: normalized Euclidean distance at most `0.14`.
- Swipe: predicted and reference directions must match.
- System button: exact case-insensitive button match.
- Type: word-token F1 at least `0.5`.
- Conflict: `terminate` with `status=failure`.

Code is released under the Apache License 2.0. Dataset images and upstream
model artifacts remain subject to their original licenses.
