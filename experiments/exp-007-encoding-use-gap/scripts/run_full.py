#!/usr/bin/env python3
"""
Exp-007: Encoding ≠ Use — full parallel pipeline.

Step 1: Cache hidden states to disk (one-time, single GPU)
Step 2: Train probes per feature (parallelizable, CPU or GPU)
Step 3: Run ablation per feature (parallelizable, one GPU each)
Step 4: Analyze and generate scatter plot

Design: each step is a separate subcommand so features can run on different GPUs.

Usage:
  # Step 1: cache hidden states (GPU 0, ~10min)
  CUDA_VISIBLE_DEVICES=0 python run_full.py cache \
    --cache-dir /nvmessd/lifanhong/LR-env/cache/hf \
    --work-dir /nvmessd/lifanhong/LR-env/exp007_full

  # Step 2+3: run one feature (can launch 8 in parallel on 8 GPUs)
  CUDA_VISIBLE_DEVICES=0 python run_full.py feature --feature is_capitalized \
    --work-dir /nvmessd/lifanhong/LR-env/exp007_full
  CUDA_VISIBLE_DEVICES=1 python run_full.py feature --feature is_plural \
    --work-dir /nvmessd/lifanhong/LR-env/exp007_full
  # ... etc

  # Step 4: analyze (CPU, after all features done)
  python run_full.py analyze --work-dir /nvmessd/lifanhong/LR-env/exp007_full
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
import torch.nn as nn
import torch.nn.functional as F
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

ALL_FEATURES = [
    "is_capitalized", "is_plural", "is_high_freq", "is_non_english",
    "is_numeric", "is_punctuation", "is_short", "is_stopword",
    "has_prefix", "is_rare", "is_title_case", "starts_with_space",
    "is_subword", "ends_with_ing", "ends_with_tion",
]


def with_retry(fn, label, attempts=5, sleep_s=20):
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            print(f"[{label}] attempt {attempt}/{attempts}: {e}", flush=True)
            if attempt < attempts:
                time.sleep(sleep_s)
    raise RuntimeError(f"{label} failed after {attempts} attempts")


# ── Feature labelers ─────────────────────────────────────────────────────────

STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
             "have", "has", "had", "do", "does", "did", "will", "would", "could",
             "should", "may", "might", "shall", "can", "to", "of", "in", "for",
             "on", "with", "at", "by", "from", "as", "into", "about", "that",
             "this", "it", "its", "and", "but", "or", "not", "no", "so", "if"}

def label_feature(feature_name: str, token_strs: list[str], token_ids: np.ndarray,
                  token_counts: Counter) -> np.ndarray:
    """Return binary labels for a feature."""
    if feature_name == "is_capitalized":
        return np.array([1 if t.lstrip() and t.lstrip()[0].isupper() else 0 for t in token_strs])
    elif feature_name == "is_plural":
        return np.array([1 if len(t.strip()) > 2 and t.strip().lower().endswith(("s", "es", "ies"))
                         else 0 for t in token_strs])
    elif feature_name == "is_high_freq":
        ranked = {tid: r for r, (tid, _) in enumerate(token_counts.most_common())}
        return np.array([1 if ranked.get(int(t), 999999) < 1000 else 0 for t in token_ids])
    elif feature_name == "is_non_english":
        return np.array([1 if any(ord(c) > 127 for c in t) else 0 for t in token_strs])
    elif feature_name == "is_numeric":
        return np.array([1 if any(c.isdigit() for c in t) else 0 for t in token_strs])
    elif feature_name == "is_punctuation":
        return np.array([1 if t.strip() and all(not c.isalnum() for c in t.strip()) else 0
                         for t in token_strs])
    elif feature_name == "is_short":
        return np.array([1 if len(t.strip()) <= 2 else 0 for t in token_strs])
    elif feature_name == "is_stopword":
        return np.array([1 if t.strip().lower() in STOPWORDS else 0 for t in token_strs])
    elif feature_name == "has_prefix":
        prefixes = ("un", "re", "pre", "dis", "mis", "non", "over", "under")
        return np.array([1 if t.strip().lower().startswith(prefixes) and len(t.strip()) > 4
                         else 0 for t in token_strs])
    elif feature_name == "is_rare":
        ranked = {tid: r for r, (tid, _) in enumerate(token_counts.most_common())}
        return np.array([1 if ranked.get(int(t), 999999) > 10000 else 0 for t in token_ids])
    elif feature_name == "is_title_case":
        return np.array([1 if t.strip() and t.strip()[0].isupper() and t.strip()[1:].islower()
                         and len(t.strip()) > 1 else 0 for t in token_strs])
    elif feature_name == "starts_with_space":
        return np.array([1 if t.startswith(" ") or t.startswith("Ġ") else 0 for t in token_strs])
    elif feature_name == "is_subword":
        return np.array([1 if not (t.startswith(" ") or t.startswith("Ġ")) else 0
                         for t in token_strs])
    elif feature_name == "ends_with_ing":
        return np.array([1 if t.strip().lower().endswith("ing") and len(t.strip()) > 4
                         else 0 for t in token_strs])
    elif feature_name == "ends_with_tion":
        return np.array([1 if t.strip().lower().endswith(("tion", "sion")) and len(t.strip()) > 5
                         else 0 for t in token_strs])
    else:
        raise ValueError(f"Unknown feature: {feature_name}")


# ── GPU Linear Probe ─────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def train_probe_gpu(X: torch.Tensor, y: torch.Tensor, device: str,
                    lr=1e-3, epochs=10, batch_size=4096) -> tuple[float, np.ndarray]:
    """Train a linear probe on GPU. Returns (accuracy, weight_vector)."""
    d_model = X.shape[1]
    probe = LinearProbe(d_model).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    X_dev = X.to(device)
    y_dev = y.float().to(device)

    # Train
    probe.train()
    n = len(X)
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits = probe(X_dev[idx])
            loss = criterion(logits, y_dev[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    probe.eval()
    with torch.no_grad():
        logits = probe(X_dev)
        preds = (logits > 0).long()
        acc = (preds == y_dev.long()).float().mean().item()

    W = probe.linear.weight.data.cpu().numpy().flatten()
    return acc, W


# ── Subcommands ──────────────────────────────────────────────────────────────

def cmd_cache(args):
    """Cache hidden states + token info to disk."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...", flush=True)
    model_id = args.model
    tokenizer = with_retry(
        lambda: AutoTokenizer.from_pretrained(model_id, cache_dir=args.cache_dir), "tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = with_retry(
        lambda: AutoModelForCausalLM.from_pretrained(
            model_id, cache_dir=args.cache_dir,
            torch_dtype=torch.float16, output_hidden_states=True), "model")
    model = model.to(args.device).eval()
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size

    # Architecture-agnostic layer accessor
    def get_model_layers(m):
        if hasattr(m, 'gpt_neox'):
            return m.gpt_neox.layers
        elif hasattr(m, 'model') and hasattr(m.model, 'layers'):
            return m.model.layers
        elif hasattr(m, 'transformer') and hasattr(m.transformer, 'h'):
            return m.transformer.h
        else:
            raise ValueError(f"Unknown architecture: {type(m).__name__}")
    model_layers = get_model_layers(model)
    print(f"  {model_id}: {n_layers} layers, d={d_model}", flush=True)

    ds = with_retry(
        lambda: load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                             split="validation",
                             cache_dir=str(Path(args.cache_dir) / "datasets")), "dataset")
    texts = [t for t in ds["text"] if len(t.strip()) > 50][:N_SENTENCES]
    print(f"  {len(texts)} sentences", flush=True)

    # Cache per-layer hidden states to separate files
    all_token_ids = []
    all_token_strs = []
    token_counter = Counter()
    layer_files = {}
    layer_buffers = {l: [] for l in range(n_layers)}

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Caching hidden states"):
        batch_texts = texts[i:i + BATCH_SIZE]
        tokens = tokenizer(batch_texts, return_tensors="pt", max_length=SEQ_LEN,
                           truncation=True, padding="max_length").to(args.device)

        with torch.no_grad():
            outputs = model(**tokens)

        mask = tokens["attention_mask"].cpu().bool()

        for layer_idx in range(n_layers):
            h = outputs.hidden_states[layer_idx + 1].cpu().float()
            for b in range(h.shape[0]):
                layer_buffers[layer_idx].append(h[b][mask[b]])

        for b in range(tokens["input_ids"].shape[0]):
            valid = mask[b]
            ids = tokens["input_ids"][b][valid].cpu().numpy()
            strs = [tokenizer.decode([tid]) for tid in ids]
            all_token_ids.append(ids)
            all_token_strs.extend(strs)
            token_counter.update(ids.tolist())

    # Save per-layer
    print("Saving hidden states...", flush=True)
    hiddens_dir = work_dir / "hiddens"
    hiddens_dir.mkdir(exist_ok=True)
    for layer_idx in tqdm(range(n_layers), desc="Saving layers"):
        h = torch.cat(layer_buffers[layer_idx], dim=0)
        torch.save(h, hiddens_dir / f"layer_{layer_idx}.pt")

    # Save token info
    all_token_ids_flat = np.concatenate(all_token_ids)
    torch.save({
        "token_ids": all_token_ids_flat,
        "token_strs": all_token_strs,
        "token_counts": dict(token_counter),
    }, work_dir / "token_info.pt")

    meta = {"model": model_id, "n_layers": n_layers, "d_model": d_model,
            "n_tokens": len(all_token_strs), "n_sentences": len(texts)}
    with open(work_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done: {len(all_token_strs)} tokens × {n_layers} layers → {work_dir}", flush=True)


def cmd_feature(args):
    """Run probe + ablation for one feature. Can run in parallel per feature."""
    work_dir = Path(args.work_dir)
    feature = args.feature

    with open(work_dir / "meta.json") as f:
        meta = json.load(f)
    n_layers = meta["n_layers"]

    # Load token info
    token_info = torch.load(work_dir / "token_info.pt", map_location="cpu", weights_only=False)
    token_ids = token_info["token_ids"]
    token_strs = token_info["token_strs"]
    token_counts = Counter(token_info["token_counts"])

    # Compute labels
    labels = label_feature(feature, token_strs, token_ids, token_counts)
    pos_rate = labels.mean()
    print(f"Feature '{feature}': pos_rate={pos_rate:.3f} ({labels.sum()}/{len(labels)})", flush=True)

    if pos_rate < 0.02 or pos_rate > 0.98:
        print(f"  SKIP: too imbalanced", flush=True)
        result = {"feature": feature, "skipped": True, "reason": "imbalanced", "pos_rate": float(pos_rate)}
        out_dir = work_dir / "features"
        out_dir.mkdir(exist_ok=True)
        with open(out_dir / f"{feature}.json", "w") as f:
            json.dump(result, f, indent=2)
        return

    y = torch.tensor(labels, dtype=torch.long)

    # Phase 1: Probe training (GPU, fast)
    print(f"Phase 1: Training probes across {n_layers} layers...", flush=True)
    probe_accs = {}
    probe_weights = {}

    for layer_idx in tqdm(range(n_layers), desc=f"  Probing {feature}"):
        X = torch.load(work_dir / "hiddens" / f"layer_{layer_idx}.pt",
                        map_location="cpu", weights_only=True)
        acc, W = train_probe_gpu(X, y, args.device)
        probe_accs[layer_idx] = acc
        probe_weights[layer_idx] = W

    best_layer = max(probe_accs, key=probe_accs.get)
    print(f"  Best probe: acc={probe_accs[best_layer]:.4f} @ layer {best_layer}", flush=True)

    # Phase 2: Ablation (needs model)
    print(f"Phase 2: Ablation...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    model_id = args.model
    tokenizer = with_retry(
        lambda: AutoTokenizer.from_pretrained(model_id, cache_dir=args.cache_dir), "tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = with_retry(
        lambda: AutoModelForCausalLM.from_pretrained(
            model_id, cache_dir=args.cache_dir, torch_dtype=torch.float16), "model")
    model = model.to(args.device).eval()

    ds = with_retry(
        lambda: load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation",
                             cache_dir=str(Path(args.cache_dir) / "datasets")), "dataset")
    texts = [t for t in ds["text"] if len(t.strip()) > 50][:N_SENTENCES]

    # Baseline loss
    baseline_losses = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        tokens = tokenizer(batch, return_tensors="pt", max_length=SEQ_LEN,
                           truncation=True, padding="max_length").to(args.device)
        with torch.no_grad():
            out = model(**tokens, labels=tokens["input_ids"])
        baseline_losses.append(out.loss.item())
    baseline_loss = np.mean(baseline_losses)
    print(f"  Baseline loss: {baseline_loss:.4f}", flush=True)

    # Ablation per layer
    ablation_deltas = {}
    for layer_idx in tqdm(range(n_layers), desc=f"  Ablating {feature}"):
        W = probe_weights[layer_idx]
        W_tensor = torch.tensor(W, dtype=torch.float16, device=args.device)
        W_norm = W_tensor / (W_tensor.norm() + 1e-8)

        def make_hook(w_norm):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                    proj = (h @ w_norm).unsqueeze(-1) * w_norm
                    return (h - proj,) + output[1:]
                else:
                    proj = (output @ w_norm).unsqueeze(-1) * w_norm
                    return output - proj
            return hook_fn

        hook = model_layers[layer_idx].register_forward_hook(make_hook(W_norm))

        abl_losses = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            tokens = tokenizer(batch, return_tensors="pt", max_length=SEQ_LEN,
                               truncation=True, padding="max_length").to(args.device)
            with torch.no_grad():
                out = model(**tokens, labels=tokens["input_ids"])
            abl_losses.append(out.loss.item())

        hook.remove()
        ablation_deltas[layer_idx] = float(np.mean(abl_losses) - baseline_loss)

    max_delta_layer = max(ablation_deltas, key=lambda l: abs(ablation_deltas[l]))
    print(f"  Max |Δloss|={abs(ablation_deltas[max_delta_layer]):.4f} @ layer {max_delta_layer}")

    # Save
    out_dir = work_dir / "features"
    out_dir.mkdir(exist_ok=True)
    result = {
        "feature": feature,
        "pos_rate": float(pos_rate),
        "baseline_loss": baseline_loss,
        "probe_accuracy": {str(l): probe_accs[l] for l in range(n_layers)},
        "ablation_delta_loss": {str(l): ablation_deltas[l] for l in range(n_layers)},
        "best_probe_layer": best_layer,
        "best_probe_acc": probe_accs[best_layer],
        "max_ablation_layer": max_delta_layer,
        "max_ablation_delta": ablation_deltas[max_delta_layer],
    }
    with open(out_dir / f"{feature}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_dir / f'{feature}.json'}", flush=True)

    del model
    torch.cuda.empty_cache()
    gc.collect()


def cmd_analyze(args):
    """Aggregate all feature results and generate plots."""
    work_dir = Path(args.work_dir)
    features_dir = work_dir / "features"

    with open(work_dir / "meta.json") as f:
        meta = json.load(f)
    n_layers = meta["n_layers"]

    # Load all feature results
    results = {}
    for fp in sorted(features_dir.glob("*.json")):
        with open(fp) as f:
            r = json.load(f)
        if r.get("skipped"):
            print(f"  {r['feature']}: SKIPPED ({r['reason']})")
            continue
        results[r["feature"]] = r
        print(f"  {r['feature']}: best_acc={r['best_probe_acc']:.3f}, "
              f"max_|Δloss|={abs(r['max_ablation_delta']):.4f}")

    if not results:
        print("No valid features found!")
        return

    # Ghost ratio
    ghost_threshold = 0.01
    probe_threshold = 0.7
    ghost_count = 0
    encoded_count = 0

    for fname, r in results.items():
        for l in range(n_layers):
            acc = r["probe_accuracy"][str(l)]
            delta = r["ablation_delta_loss"][str(l)]
            if acc > probe_threshold:
                encoded_count += 1
                if abs(delta) < ghost_threshold:
                    ghost_count += 1

    ghost_ratio = ghost_count / max(encoded_count, 1)

    # Scatter plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))

    for ci, (fname, r) in enumerate(results.items()):
        accs = [r["probe_accuracy"][str(l)] for l in range(n_layers)]
        deltas = [r["ablation_delta_loss"][str(l)] for l in range(n_layers)]
        ax.scatter(accs, deltas, c=[colors[ci]], label=fname, alpha=0.7, s=30)

    ax.set_xlabel("Probe Accuracy (Encoding Strength)", fontsize=12)
    ax.set_ylabel("Δ Loss from Ablation (Use Strength)", fontsize=12)
    ax.set_title(f"Encoding vs Use — Ghost Ratio = {ghost_ratio:.1%}\n"
                 f"(acc>{probe_threshold}, |Δloss|<{ghost_threshold})", fontsize=13)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=ghost_threshold, color="red", linestyle=":", alpha=0.4)
    ax.axhline(y=-ghost_threshold, color="red", linestyle=":", alpha=0.4)
    ax.axvline(x=probe_threshold, color="blue", linestyle=":", alpha=0.4)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(work_dir / "encoding_vs_use.png", dpi=150)
    print(f"\nScatter: {work_dir / 'encoding_vs_use.png'}")

    # Layer profiles
    n_features = len(results)
    fig2, axes2 = plt.subplots(n_features, 2, figsize=(14, 3.5 * n_features), squeeze=False)
    for ci, (fname, r) in enumerate(results.items()):
        layers = list(range(n_layers))
        accs = [r["probe_accuracy"][str(l)] for l in layers]
        deltas = [r["ablation_delta_loss"][str(l)] for l in layers]
        axes2[ci, 0].plot(layers, accs, "o-", markersize=3)
        axes2[ci, 0].set_ylabel("Probe Acc")
        axes2[ci, 0].set_title(f"{fname} — Encoding")
        axes2[ci, 1].plot(layers, deltas, "o-", markersize=3, color="orange")
        axes2[ci, 1].set_ylabel("Δ Loss")
        axes2[ci, 1].set_title(f"{fname} — Use")
        axes2[ci, 1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    fig2.tight_layout()
    fig2.savefig(work_dir / "layer_profiles.png", dpi=150)
    print(f"Profiles: {work_dir / 'layer_profiles.png'}")

    # Summary
    summary = {
        "model": meta["model"],
        "n_layers": n_layers,
        "n_features": len(results),
        "ghost_ratio": ghost_ratio,
        "ghost_threshold": ghost_threshold,
        "probe_threshold": probe_threshold,
        "ghost_count": ghost_count,
        "encoded_count": encoded_count,
        "features": {fname: {
            "pos_rate": r["pos_rate"],
            "best_probe_acc": r["best_probe_acc"],
            "best_probe_layer": r["best_probe_layer"],
            "max_ablation_delta": r["max_ablation_delta"],
            "max_ablation_layer": r["max_ablation_layer"],
        } for fname, r in results.items()},
    }
    with open(work_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {work_dir / 'summary.json'}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Ghost ratio: {ghost_ratio:.1%} ({ghost_count}/{encoded_count})")
    for fname, r in results.items():
        print(f"  {fname}: best_acc={r['best_probe_acc']:.3f}, max_|Δloss|={abs(r['max_ablation_delta']):.4f}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cache = sub.add_parser("cache", help="Cache hidden states to disk")
    p_cache.add_argument("--cache-dir", default="/nvmessd/lifanhong/LR-env/cache/hf")
    p_cache.add_argument("--work-dir", required=True)
    p_cache.add_argument("--device", default="cuda")
    p_cache.add_argument("--model", default=MODEL_ID, help="HuggingFace model ID")

    p_feat = sub.add_parser("feature", help="Run probe + ablation for one feature")
    p_feat.add_argument("--feature", required=True, choices=ALL_FEATURES)
    p_feat.add_argument("--work-dir", required=True)
    p_feat.add_argument("--cache-dir", default="/nvmessd/lifanhong/LR-env/cache/hf")
    p_feat.add_argument("--device", default="cuda")
    p_feat.add_argument("--model", default=MODEL_ID, help="HuggingFace model ID")

    p_analyze = sub.add_parser("analyze", help="Aggregate results and plot")
    p_analyze.add_argument("--work-dir", required=True)

    args = parser.parse_args()

    if args.cmd == "cache":
        cmd_cache(args)
    elif args.cmd == "feature":
        cmd_feature(args)
    elif args.cmd == "analyze":
        cmd_analyze(args)


if __name__ == "__main__":
    main()
