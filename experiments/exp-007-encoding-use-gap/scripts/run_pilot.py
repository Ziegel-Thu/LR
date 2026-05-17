#!/usr/bin/env python3
"""
Exp-007: Encoding ≠ Use — end-to-end pilot pipeline.

Runs the full probe + ablation pipeline for 5 easy-to-label features
on Pythia-1.4B. Outputs probe accuracies, ablation Δloss, ghost ratio,
and the encoding-use scatter plot.

Features (labeling from token text, no NLP toolkit needed):
  1. is_capitalized: first character is uppercase
  2. is_quoted: token is inside quotation marks
  3. is_high_freq: token rank < 1000 in corpus
  4. is_plural: ends with 's'/'es'/'ies' (heuristic)
  5. is_non_english: contains non-ASCII characters

Usage:
  CUDA_VISIBLE_DEVICES=3 python run_pilot.py \
    --cache-dir /nvmessd/lifanhong/LR-env/cache/hf \
    --out-dir /nvmessd/lifanhong/LR-env/exp007_pilot
"""

import argparse
import gc
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_ID = "EleutherAI/pythia-1.4b-deduped"
N_SENTENCES = 2000
SEQ_LEN = 128
BATCH_SIZE = 8


def with_retry(fn, label, attempts=5, sleep_s=20):
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            print(f"[{label}] attempt {attempt}/{attempts}: {e}", flush=True)
            if attempt < attempts:
                time.sleep(sleep_s)
    raise RuntimeError(f"{label} failed after {attempts} attempts")


# ── Feature labeling functions ───────────────────────────────────────────────

def label_is_capitalized(token_strs: list[str]) -> np.ndarray:
    """1 if first non-space char is uppercase."""
    labels = []
    for t in token_strs:
        t_stripped = t.lstrip()
        labels.append(1 if t_stripped and t_stripped[0].isupper() else 0)
    return np.array(labels)


def label_is_quoted(token_strs: list[str], input_ids: torch.Tensor, tokenizer) -> np.ndarray:
    """Heuristic: track quote state across sequence."""
    # Decode full sequence and track quotes
    full_text = tokenizer.decode(input_ids)
    labels = np.zeros(len(token_strs), dtype=int)
    in_quote = False
    pos = 0
    for i, t in enumerate(token_strs):
        for ch in t:
            if ch in '"\'':
                in_quote = not in_quote
        labels[i] = 1 if in_quote else 0
    return labels


def label_is_high_freq(token_ids: np.ndarray, freq_threshold: int = 1000,
                       token_counts: Counter = None) -> np.ndarray:
    """1 if token rank < threshold in the dataset."""
    if token_counts is None:
        return np.zeros(len(token_ids), dtype=int)
    ranked = [tid for tid, _ in token_counts.most_common()]
    rank_map = {tid: rank for rank, tid in enumerate(ranked)}
    return np.array([1 if rank_map.get(int(t), 999999) < freq_threshold else 0
                     for t in token_ids])


def label_is_plural(token_strs: list[str]) -> np.ndarray:
    """Heuristic: ends with s/es/ies and length > 2."""
    labels = []
    for t in token_strs:
        t = t.strip().lower()
        if len(t) > 2 and (t.endswith("ies") or t.endswith("es") or t.endswith("s")):
            labels.append(1)
        else:
            labels.append(0)
    return np.array(labels)


def label_is_non_english(token_strs: list[str]) -> np.ndarray:
    """1 if token contains non-ASCII characters."""
    labels = []
    for t in token_strs:
        has_non_ascii = any(ord(c) > 127 for c in t)
        labels.append(1 if has_non_ascii else 0)
    return np.array(labels)


# ── Main pipeline ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="/nvmessd/lifanhong/LR-env/cache/hf")
    parser.add_argument("--out-dir", default="/nvmessd/lifanhong/LR-env/exp007_pilot")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    # ── Load model ──
    print("Loading model and tokenizer...", flush=True)
    tokenizer = with_retry(
        lambda: AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=args.cache_dir),
        "load tokenizer",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = with_retry(
        lambda: AutoModelForCausalLM.from_pretrained(
            MODEL_ID, cache_dir=args.cache_dir,
            torch_dtype=torch.float16, output_hidden_states=True,
        ),
        "load model",
    )
    model = model.to(args.device).eval()
    n_layers = model.config.num_hidden_layers
    print(f"Model: {MODEL_ID}, {n_layers} layers, d={model.config.hidden_size}", flush=True)

    # ── Load data ──
    print("Loading data...", flush=True)
    ds = with_retry(
        lambda: load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                             split="validation",
                             cache_dir=str(Path(args.cache_dir) / "datasets")),
        "load wikitext",
    )
    texts = [t for t in ds["text"] if len(t.strip()) > 50][:N_SENTENCES]
    print(f"  {len(texts)} sentences", flush=True)

    # ── Phase 1: Cache hidden states + compute labels ──
    print("\nPhase 1: Caching hidden states and computing labels...", flush=True)

    all_hidden = {l: [] for l in range(n_layers)}  # layer → list of (n_tokens, d_model)
    all_token_ids = []
    all_token_strs = []
    token_counter = Counter()

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Caching hidden states"):
        batch_texts = texts[i:i + BATCH_SIZE]
        tokens = tokenizer(
            batch_texts, return_tensors="pt",
            max_length=SEQ_LEN, truncation=True, padding="max_length",
        ).to(args.device)

        with torch.no_grad():
            outputs = model(**tokens)

        mask = tokens["attention_mask"].cpu().bool()

        for layer_idx in range(n_layers):
            h = outputs.hidden_states[layer_idx + 1].cpu().float()  # skip embedding
            for b in range(h.shape[0]):
                valid = mask[b]
                all_hidden[layer_idx].append(h[b][valid].numpy())

        # Collect token info
        for b in range(tokens["input_ids"].shape[0]):
            valid = mask[b]
            ids = tokens["input_ids"][b][valid].cpu().numpy()
            strs = [tokenizer.decode([tid]) for tid in ids]
            all_token_ids.append(ids)
            all_token_strs.extend(strs)
            token_counter.update(ids.tolist())

    # Concatenate
    for layer_idx in range(n_layers):
        all_hidden[layer_idx] = np.concatenate(all_hidden[layer_idx], axis=0)
    all_token_ids_flat = np.concatenate(all_token_ids)
    n_tokens = len(all_token_strs)

    print(f"  Cached: {n_tokens} tokens × {n_layers} layers", flush=True)

    # Compute labels
    features = {
        "is_capitalized": label_is_capitalized(all_token_strs),
        "is_plural": label_is_plural(all_token_strs),
        "is_high_freq": label_is_high_freq(all_token_ids_flat, 1000, token_counter),
        "is_non_english": label_is_non_english(all_token_strs),
    }

    # Print label balance
    for fname, labels in features.items():
        pos_rate = labels.mean()
        print(f"  {fname}: {pos_rate:.3f} positive ({labels.sum()}/{len(labels)})")

    # Skip features with < 5% or > 95% positive (too imbalanced)
    valid_features = {k: v for k, v in features.items()
                      if 0.05 < v.mean() < 0.95}
    print(f"\n  Valid features (5-95% balance): {list(valid_features.keys())}")

    if not valid_features:
        print("No valid features! Exiting.", flush=True)
        return

    # ── Phase 2: Train probes ──
    print("\nPhase 2: Training probes...", flush=True)

    probe_results = {}  # {feature: {layer: accuracy}}
    probe_weights = {}  # {feature: {layer: W}}

    for fname, labels in valid_features.items():
        probe_results[fname] = {}
        probe_weights[fname] = {}

        for layer_idx in tqdm(range(n_layers), desc=f"  Probes: {fname}"):
            X = all_hidden[layer_idx]
            y = labels

            clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
            scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
            acc = scores.mean()

            # Fit on all data to get probe direction
            clf.fit(X, y)
            W = clf.coef_[0]  # (d_model,)

            probe_results[fname][layer_idx] = float(acc)
            probe_weights[fname][layer_idx] = W

        best_layer = max(probe_results[fname], key=probe_results[fname].get)
        print(f"    {fname}: best acc={probe_results[fname][best_layer]:.4f} @ layer {best_layer}")

    # Free model memory before ablation
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # ── Phase 3: Ablation ──
    print("\nPhase 3: Running ablations...", flush=True)

    # Reload model for ablation
    model = with_retry(
        lambda: AutoModelForCausalLM.from_pretrained(
            MODEL_ID, cache_dir=args.cache_dir, torch_dtype=torch.float16,
        ),
        "reload model",
    )
    model = model.to(args.device).eval()

    # Compute baseline loss
    print("  Computing baseline loss...", flush=True)
    baseline_losses = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="  Baseline"):
        batch_texts = texts[i:i + BATCH_SIZE]
        tokens = tokenizer(
            batch_texts, return_tensors="pt",
            max_length=SEQ_LEN, truncation=True, padding="max_length",
        ).to(args.device)

        with torch.no_grad():
            outputs = model(**tokens, labels=tokens["input_ids"])
        baseline_losses.append(outputs.loss.item())

    baseline_loss = np.mean(baseline_losses)
    print(f"  Baseline loss: {baseline_loss:.4f}", flush=True)

    # Ablation: for each (feature, layer), project out probe direction
    ablation_results = {}  # {feature: {layer: delta_loss}}

    for fname in valid_features:
        ablation_results[fname] = {}

        for layer_idx in tqdm(range(n_layers), desc=f"  Ablation: {fname}"):
            W = probe_weights[fname][layer_idx]
            W_tensor = torch.tensor(W, dtype=torch.float16, device=args.device)
            W_norm = W_tensor / (W_tensor.norm() + 1e-8)

            # Hook: project out probe direction
            def make_hook(w_norm):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        proj = (h @ w_norm).unsqueeze(-1) * w_norm
                        h_ablated = h - proj
                        return (h_ablated,) + output[1:]
                    else:
                        proj = (output @ w_norm).unsqueeze(-1) * w_norm
                        return output - proj
                return hook_fn

            # Register hook on the target layer
            hook = model.gpt_neox.layers[layer_idx].register_forward_hook(
                make_hook(W_norm)
            )

            ablated_losses = []
            for i in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[i:i + BATCH_SIZE]
                tokens = tokenizer(
                    batch_texts, return_tensors="pt",
                    max_length=SEQ_LEN, truncation=True, padding="max_length",
                ).to(args.device)

                with torch.no_grad():
                    outputs = model(**tokens, labels=tokens["input_ids"])
                ablated_losses.append(outputs.loss.item())

            hook.remove()
            delta_loss = np.mean(ablated_losses) - baseline_loss
            ablation_results[fname][layer_idx] = float(delta_loss)

        best_layer = max(ablation_results[fname], key=lambda l: abs(ablation_results[fname][l]))
        print(f"    {fname}: max |Δloss|={abs(ablation_results[fname][best_layer]):.4f} "
              f"@ layer {best_layer}")

    # ── Phase 4: Analysis ──
    print("\nPhase 4: Analysis...", flush=True)

    # Scatter plot: probe accuracy vs Δloss
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(valid_features)))

    ghost_count = 0
    encoded_count = 0
    ghost_threshold = 0.01  # Δloss threshold for "not used"
    probe_threshold = 0.7  # probe acc threshold for "encoded"

    for ci, fname in enumerate(valid_features):
        accs = [probe_results[fname][l] for l in range(n_layers)]
        deltas = [ablation_results[fname][l] for l in range(n_layers)]

        ax.scatter(accs, deltas, c=[colors[ci]], label=fname, alpha=0.7, s=30)

        # Count ghost information
        for acc, delta in zip(accs, deltas):
            if acc > probe_threshold:
                encoded_count += 1
                if abs(delta) < ghost_threshold:
                    ghost_count += 1

    ghost_ratio = ghost_count / max(encoded_count, 1)

    ax.set_xlabel("Probe Accuracy (Encoding Strength)")
    ax.set_ylabel("Δ Loss from Ablation (Use Strength)")
    ax.set_title(f"Encoding vs Use — Ghost Ratio = {ghost_ratio:.1%} "
                 f"(threshold: acc>{probe_threshold}, |Δloss|<{ghost_threshold})")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=ghost_threshold, color="red", linestyle=":", alpha=0.5, label=f"Δloss={ghost_threshold}")
    ax.axhline(y=-ghost_threshold, color="red", linestyle=":", alpha=0.5)
    ax.axvline(x=probe_threshold, color="blue", linestyle=":", alpha=0.5, label=f"acc={probe_threshold}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "encoding_vs_use.png", dpi=150)
    print(f"  Scatter plot: {out_dir / 'encoding_vs_use.png'}")

    # Layer-wise plot
    fig2, axes2 = plt.subplots(len(valid_features), 2, figsize=(14, 4 * len(valid_features)),
                                squeeze=False)
    for ci, fname in enumerate(valid_features):
        layers = list(range(n_layers))
        accs = [probe_results[fname][l] for l in layers]
        deltas = [ablation_results[fname][l] for l in layers]

        axes2[ci, 0].plot(layers, accs, "o-", markersize=4)
        axes2[ci, 0].set_ylabel("Probe Accuracy")
        axes2[ci, 0].set_title(f"{fname} — Encoding")
        axes2[ci, 0].set_xlabel("Layer")

        axes2[ci, 1].plot(layers, deltas, "o-", markersize=4, color="orange")
        axes2[ci, 1].set_ylabel("Δ Loss")
        axes2[ci, 1].set_title(f"{fname} — Use (Ablation)")
        axes2[ci, 1].set_xlabel("Layer")
        axes2[ci, 1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    fig2.suptitle("Exp-007: Layer-wise Encoding vs Use", fontsize=14)
    fig2.tight_layout()
    fig2.savefig(out_dir / "layer_profiles.png", dpi=150)
    print(f"  Layer profiles: {out_dir / 'layer_profiles.png'}")

    # Save results
    results = {
        "model": MODEL_ID,
        "n_layers": n_layers,
        "n_tokens": n_tokens,
        "baseline_loss": baseline_loss,
        "ghost_ratio": ghost_ratio,
        "ghost_threshold": ghost_threshold,
        "probe_threshold": probe_threshold,
        "ghost_count": ghost_count,
        "encoded_count": encoded_count,
        "features": {},
    }

    for fname in valid_features:
        results["features"][fname] = {
            "label_pos_rate": float(valid_features[fname].mean()),
            "probe_accuracy": {str(l): probe_results[fname][l] for l in range(n_layers)},
            "ablation_delta_loss": {str(l): ablation_results[fname][l] for l in range(n_layers)},
        }

    with open(out_dir / "pilot_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results: {out_dir / 'pilot_results.json'}")

    print(f"\n{'='*60}")
    print("PILOT SUMMARY")
    print(f"{'='*60}")
    print(f"  Ghost ratio:  {ghost_ratio:.1%} ({ghost_count}/{encoded_count})")
    print(f"  Baseline loss: {baseline_loss:.4f}")
    for fname in valid_features:
        best_acc = max(probe_results[fname].values())
        max_delta = max(abs(v) for v in ablation_results[fname].values())
        print(f"  {fname}: best_acc={best_acc:.3f}, max_|Δloss|={max_delta:.4f}")


if __name__ == "__main__":
    main()
