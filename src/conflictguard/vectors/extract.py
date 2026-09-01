"""Extract behavior and conditional PCA directions from calibration pairs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

from ..adapters import get_adapter
from ..adapters.base import ModelAdapter
from .dataset import ConflictDataset, RawContrastivePair


# ---------------------------------------------------------------------------
# Single-sample hidden-state extraction
# ---------------------------------------------------------------------------

def _extract_one(
    model,
    processor,
    adapter: ModelAdapter,
    raw_messages: List[dict],
    image_paths: List[str],
    suffix: Optional[str],
    layer_ids: List[int],
) -> Dict[int, np.ndarray]:
    """
    Run one forward pass and return per-layer hidden state vectors.

    Parameters
    ----------
    suffix
        If provided, appended as an assistant turn → hidden states are
        averaged over the suffix token positions.
        If None → last-token hidden state (generation-start) is returned.
    layer_ids
        Which layers to extract from (0-indexed).
    """
    if suffix is not None:
        messages = adapter.build_messages(
            raw_messages, image_paths, mode="suffix", suffix=suffix
        )
        text = adapter.apply_chat_template(processor, messages, add_generation_prompt=False)
    else:
        messages = adapter.build_messages(
            raw_messages, image_paths, mode="generate"
        )
        text = adapter.apply_chat_template(processor, messages, add_generation_prompt=True)

    inputs = adapter.build_inputs(processor, messages, text, model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = adapter.get_hidden_states(outputs)
    # hidden_states: tuple of (n_layers+1) tensors, each [1, seq_len, H]

    result: Dict[int, np.ndarray] = {}
    for layer_id in layer_ids:
        idx = adapter.layer_output_index(layer_id)
        layer_hidden = hidden_states[idx][0]  # [seq_len, H]

        if suffix is not None:
            start, end = adapter.suffix_token_span(
                processor, inputs["input_ids"], suffix
            )
            vec = layer_hidden[start:end].mean(dim=0)
        else:
            vec = layer_hidden[-1]

        result[layer_id] = vec.cpu().float().numpy()

    del outputs, inputs
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Batched hidden-state collection
# ---------------------------------------------------------------------------

def collect_hidden_states(
    model,
    processor,
    adapter: ModelAdapter,
    pairs: List[RawContrastivePair],
    layer_ids: List[int],
    extraction_mode: Literal["behavior", "conditional"],
) -> Dict[int, np.ndarray]:
    """
    Collect hidden states for all contrastive pairs.

    Returns
    -------
    dict[layer_id → np.ndarray of shape [2*N, hidden_dim]]
        Row 2i   = positive example i
        Row 2i+1 = negative example i
    """
    layer_hiddens: Dict[int, List[np.ndarray]] = {lid: [] for lid in layer_ids}

    for pair in tqdm(pairs, desc=f"[{extraction_mode}] hidden states"):
        if extraction_mode == "behavior":
            pos_h = _extract_one(
                model, processor, adapter,
                pair.positive_messages, pair.positive_images,
                pair.positive_suffix, layer_ids,
            )
            neg_h = _extract_one(
                model, processor, adapter,
                pair.negative_messages, pair.negative_images,
                pair.negative_suffix, layer_ids,
            )
        else:  # conditional
            pos_h = _extract_one(
                model, processor, adapter,
                pair.positive_messages, pair.positive_images,
                None, layer_ids,
            )
            neg_h = _extract_one(
                model, processor, adapter,
                pair.negative_messages, pair.negative_images,
                None, layer_ids,
            )

        for lid in layer_ids:
            layer_hiddens[lid].append(pos_h[lid])
            layer_hiddens[lid].append(neg_h[lid])

    return {lid: np.stack(v) for lid, v in layer_hiddens.items()}


# ---------------------------------------------------------------------------
# PCA-based direction computation
# ---------------------------------------------------------------------------

def compute_directions(
    layer_hiddens: Dict[int, np.ndarray],
    layer_ids: Iterable[int],
    method: Literal["pca_diff", "pca_center", "pca_pairwise"] = "pca_pairwise",
) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
    """
    Fit a 1-component PCA per layer to the contrastive hidden states.

    Convention: the returned direction points from negative → positive.

    Parameters
    ----------
    layer_hiddens
        Output of collect_hidden_states.
    layer_ids
        Layers to process.
    method
        pca_diff     : train on positive − negative differences.
        pca_center   : train on mean-centred hidden states.
        pca_pairwise : train on pairwise-centred hidden states (default).

    Returns
    -------
    directions : dict[layer_id → np.ndarray [hidden_dim]]
    variances  : dict[layer_id → float]
    """
    directions: Dict[int, np.ndarray] = {}
    variances: Dict[int, float] = {}

    for layer in tqdm(list(layer_ids), desc="Computing PCA directions"):
        h = layer_hiddens[layer]  # [2N, hidden_dim]

        if method == "pca_diff":
            train = h[::2] - h[1::2]
        elif method == "pca_center":
            train = h - h.mean(axis=0)
        elif method == "pca_pairwise":
            center = (h[::2] + h[1::2]) / 2
            train = h.copy()
            train[::2] -= center
            train[1::2] -= center
        else:
            raise ValueError(f"Unknown PCA method: {method!r}")

        pca = PCA(n_components=1, whiten=False).fit(train)
        direction = pca.components_.squeeze(axis=0).astype(np.float32)
        variance = float(pca.explained_variance_ratio_[0])

        # Orient so positive examples project higher
        projected = h @ direction
        pos_larger_frac = np.mean(
            [projected[2 * i] > projected[2 * i + 1] for i in range(len(h) // 2)]
        )
        if pos_larger_frac < 0.5:
            direction *= -1

        directions[layer] = direction
        variances[layer] = variance

    return directions, variances


# ---------------------------------------------------------------------------
# Best-layer selection
# ---------------------------------------------------------------------------

def select_best_layer(variances: Dict[int, float]) -> int:
    return max(variances, key=variances.__getitem__)


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_vector(
    path: str,
    mode: str,
    adapter: ModelAdapter,
    model_path: str,
    hidden_sz: int,
    directions: Dict[int, np.ndarray],
    variances: Dict[int, float],
    n_pairs: int,
    behavior_strategy: Optional[str] = None,
) -> None:
    """Save steering vector data to a .pt file with full metadata."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    best_layer = select_best_layer(variances)
    payload = {
        # Algorithm outputs
        "mode": mode,
        "best_layer": best_layer,
        "best_direction": directions[best_layer],
        "directions": directions,
        "explained_variances": variances,
        # Metadata for cross-model validation
        "adapter_name": adapter.name,
        "model_family": adapter.model_family,
        "model_path": model_path,
        "hidden_size": hidden_sz,
        "n_pairs": n_pairs,
        "layer_space": "post",        # vectors extracted at layer output
        "token_scope": (
            "suffix_avg" if mode == "behavior" else "generation_start"
        ),
    }
    if behavior_strategy is not None:
        payload["behavior_strategy"] = behavior_strategy
    torch.save(payload, path)
    print(
        f"[save_vector] → {path}\n"
        f"  best_layer={best_layer}  expl_var={variances[best_layer]:.4f}  "
        f"  direction shape={directions[best_layer].shape}"
    )


def load_vector(path: str) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    print(
        f"[load_vector] ← {path}\n"
        f"  mode={data['mode']}  best_layer={data['best_layer']}  "
        f"  shape={data['best_direction'].shape}"
    )
    return data


def validate_vector_metadata(
    data: dict,
    adapter: ModelAdapter,
    model_hidden_size: int,
) -> None:
    """Warn if the loaded vector was trained on a different adapter or hidden size."""
    saved_adapter = data.get("adapter_name", "unknown")
    saved_hs = data.get("hidden_size", -1)
    if saved_adapter != adapter.name:
        print(
            f"[WARN] Vector adapter '{saved_adapter}' != current adapter "
            f"'{adapter.name}'.  Results may be unreliable."
        )
    if saved_hs != model_hidden_size and saved_hs != -1:
        print(
            f"[WARN] Vector hidden_size={saved_hs} != model hidden_size="
            f"{model_hidden_size}.  This vector cannot be used with this model."
        )


# ---------------------------------------------------------------------------
# High-level training functions
# ---------------------------------------------------------------------------

def train_behavior_vector(
    model,
    processor,
    adapter: ModelAdapter,
    dataset: ConflictDataset,
    model_path: str,
    output_path: str,
    pca_method: str = "pca_pairwise",
    layer_ids: Optional[List[int]] = None,
) -> Dict[int, np.ndarray]:
    """Train and save a behaviour steering vector."""
    if layer_ids is None:
        layer_ids = list(range(adapter.num_hidden_layers(model)))

    layer_hiddens = collect_hidden_states(
        model, processor, adapter, dataset.pairs, layer_ids, "behavior"
    )
    directions, variances = compute_directions(layer_hiddens, layer_ids, pca_method)
    save_vector(
        output_path, dataset.mode, adapter, model_path,
        adapter.hidden_size(model),
        directions, variances, len(dataset.pairs), dataset.behavior_strategy,
    )
    return directions


def train_conditional_vector(
    model,
    processor,
    adapter: ModelAdapter,
    dataset: ConflictDataset,
    model_path: str,
    output_path: str,
    pca_method: str = "pca_pairwise",
    layer_ids: Optional[List[int]] = None,
) -> Dict[int, np.ndarray]:
    """Train and save a conditional (conflict-detection) vector."""
    if layer_ids is None:
        layer_ids = list(range(adapter.num_hidden_layers(model)))

    layer_hiddens = collect_hidden_states(
        model, processor, adapter, dataset.pairs, layer_ids, "conditional"
    )
    directions, variances = compute_directions(layer_hiddens, layer_ids, pca_method)
    save_vector(
        output_path, dataset.mode, adapter, model_path,
        adapter.hidden_size(model),
        directions, variances, len(dataset.pairs),
    )
    return directions


# ---------------------------------------------------------------------------
# Variance table helper
# ---------------------------------------------------------------------------

def print_layer_variance_table(variances: Dict[int, float]) -> None:
    best = select_best_layer(variances)
    print(f"\n{'Layer':>6}  {'Expl. Var':>10}  {'Best':>5}")
    print("-" * 28)
    for layer in sorted(variances):
        marker = "<-- best" if layer == best else ""
        print(f"{layer:>6}  {variances[layer]:>10.4f}  {marker}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract steering vectors from any supported VLM."
    )
    p.add_argument("--adapter", type=str, required=True,
                   help="Adapter name: qwen3vl | uitars | uivenus")
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to the model (local or HuggingFace ID).")
    p.add_argument("--processor_path", type=str, default=None,
                   help="Processor path (defaults to model_path).")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./vectors")
    p.add_argument("--mode", type=str, default="all",
                   choices=["all", "behavior",
                            "conditional_c1", "conditional_c2"])
    p.add_argument("--n_samples", type=int, default=300)
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--pca_method", type=str, default="pca_pairwise",
                   choices=["pca_diff", "pca_center", "pca_pairwise"])
    p.add_argument("--layers", type=str, default=None,
                   help="Comma-separated layer indices (default: all layers).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print_variance_table", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    adapter = get_adapter(args.adapter)
    print(f"[main] Using adapter: {adapter.name!r}")

    model = adapter.load_model(args.model_path, args.device_id)
    processor = adapter.load_processor(args.model_path, args.processor_path)

    layer_ids: Optional[List[int]] = None
    if args.layers is not None:
        layer_ids = [int(x.strip()) for x in args.layers.split(",")]

    modes = (
        ["behavior", "conditional_c1", "conditional_c2"]
        if args.mode == "all"
        else [args.mode]
    )

    for mode in modes:
        extraction = "behavior" if mode.startswith("behavior") else "conditional"

        dataset = ConflictDataset(
            data_path=args.data_path,
            mode=mode,
            adapter=adapter,
            n_samples=args.n_samples,
            seed=args.seed,
        )

        if extraction == "behavior":
            out_path = str(Path(args.output_dir) / f"{mode}.pt")
            train_behavior_vector(
                model, processor, adapter, dataset,
                args.model_path, out_path,
                args.pca_method, layer_ids,
            )
            if args.print_variance_table:
                data = load_vector(out_path)
                print_layer_variance_table(data["explained_variances"])
        else:
            out_path = str(Path(args.output_dir) / f"{mode}.pt")
            train_conditional_vector(
                model, processor, adapter, dataset,
                args.model_path, out_path,
                args.pca_method, layer_ids,
            )
            if args.print_variance_table:
                data = load_vector(out_path)
                print_layer_variance_table(data["explained_variances"])


if __name__ == "__main__":
    main()
