#!/usr/bin/env python3
"""
Exp-020: Steering Vector OOD Boundary Detection.

For each feature direction from exp-007, sweep steering magnitude α
and measure loss / KL divergence to find the OOD boundary.

Usage:
  CUDA_VISIBLE_DEVICES=2 python steering_sweep.py \
    --model EleutherAI/pythia-2.8b-deduped \
    --cache-dir /nvmessd/lifanhong/LR-env/cache/hf \
    --work-dir /nvmessd/lifanhong/LR-env/exp007_28b \
    --output-dir /nvmessd/lifanhong/LR-env/exp020-steering
"""

import argparse
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

# ── Feature labelers (copied from exp-007 for consistency) ───────────────────

STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
             "have", "has", "had", "do", "does", "did", "will", "would", "could",
             "should", "may", "might", "shall", "can", "to", "of", "in", "for",
             "on", "with", "at", "by", "from", "as", "into", "about", "that",
             "this", "it", "its", "and", "but", "or", "not", "no", "so", "if"}

ALL_FEATURES = [
    "is_capitalized", "is_plural", "is_high_freq", "is_non_english",
    "is_numeric", "is_punctuation", "is_short", "is_stopword",
    "has_prefix", "is_rare", "is_title_case", "starts_with_space",
    "is_subword", "ends_with_ing", "ends_with_tion",
]

def label_feature(feature_name: str, token_strs: list[str], token_ids: np.ndarray,
                  token_counts: Counter) -> np.ndarray:
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


# ── Linear probe ─────────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def train_probe_gpu(X: torch.Tensor, y: torch.Tensor, device: str,
                    lr=1e-3, epochs=10, batch_size=4096):
    """Train linear probe. Returns (accuracy, weight_vector, bias)."""
    d_model = X.shape[1]
    probe = LinearProbe(d_model).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    X_dev = X.to(device)
    y_dev = y.float().to(device)

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

    probe.eval()
    with torch.no_grad():
        logits = probe(X_dev)
        preds = (logits > 0).long()
        acc = (preds == y_dev.long()).float().mean().item()

    W = probe.linear.weight.data.cpu().numpy().flatten()
    b = probe.linear.bias.data.cpu().item()
    return acc, W, b


# ── Steering sweep ───────────────────────────────────────────────────────────

def get_model_layers(m):
    if hasattr(m, 'gpt_neox'):
        return m.gpt_neox.layers
    elif hasattr(m, 'model') and hasattr(m.model, 'layers'):
        return m.model.layers
    elif hasattr(m, 'transformer') and hasattr(m.transformer, 'h'):
        return m.transformer.h
    else:
        raise ValueError(f"Unknown architecture: {type(m).__name__}")


def compute_baseline(model, tokenizer, texts, device, batch_size=4, seq_len=128):
    """Compute baseline loss and collect logits (log-probs) for KL reference."""
    all_losses = []
    all_log_probs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(batch, return_tensors="pt", max_length=seq_len,
                           truncation=True, padding="max_length").to(device)
        with torch.no_grad():
            out = model(**tokens)
            logits = out.logits  # (B, T, V)
            # shift for LM loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = tokens["input_ids"][:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            all_losses.append(loss.item())
            all_log_probs.append(F.log_softmax(shift_logits, dim=-1).cpu())

    return float(np.mean(all_losses)), all_log_probs


def compute_steered(model, tokenizer, texts, model_layers, layer_idx,
                    steering_vec, alpha, device, batch_size=4, seq_len=128):
    """Run model with steering vector added at layer_idx. Return loss + log_probs."""
    W_tensor = torch.tensor(steering_vec, dtype=torch.float16, device=device)
    W_norm = W_tensor / (W_tensor.norm() + 1e-8)

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
            h = h + alpha * W_norm.to(h.dtype)
            return (h,) + output[1:]
        else:
            return output + alpha * W_norm.to(output.dtype)

    hook = model_layers[layer_idx].register_forward_hook(hook_fn)

    all_losses = []
    all_log_probs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(batch, return_tensors="pt", max_length=seq_len,
                           truncation=True, padding="max_length").to(device)
        with torch.no_grad():
            out = model(**tokens)
            shift_logits = out.logits[:, :-1, :].contiguous()
            shift_labels = tokens["input_ids"][:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            all_losses.append(loss.item())
            all_log_probs.append(F.log_softmax(shift_logits, dim=-1).cpu())

    hook.remove()
    return float(np.mean(all_losses)), all_log_probs


def compute_kl(baseline_log_probs, steered_log_probs):
    """Mean KL(baseline || steered) across all tokens."""
    kl_values = []
    for bl, st in zip(baseline_log_probs, steered_log_probs):
        # bl, st: (B, T, V) log-probs
        p = bl.exp()
        kl = (p * (bl - st)).sum(dim=-1)  # (B, T)
        kl_values.append(kl.mean().item())
    return float(np.mean(kl_values))


def find_ood_boundary(alphas, kl_values, threshold_factor=5.0):
    """Find α where KL divergence crosses threshold.
    Threshold = threshold_factor × KL at smallest |α| > 0."""
    # find KL at smallest positive α
    pos_mask = [i for i, a in enumerate(alphas) if a > 0]
    neg_mask = [i for i, a in enumerate(alphas) if a < 0]

    if not pos_mask or not neg_mask:
        return None, None

    # baseline KL at small α (first positive and last negative)
    small_kl = min(kl_values[pos_mask[0]], kl_values[neg_mask[-1]])
    threshold = max(threshold_factor * small_kl, 0.5)  # at least 0.5 nats

    # find positive boundary
    pos_boundary = None
    for i in pos_mask:
        if kl_values[i] > threshold:
            pos_boundary = alphas[i]
            break

    # find negative boundary
    neg_boundary = None
    for i in reversed(neg_mask):
        if kl_values[i] > threshold:
            neg_boundary = alphas[i]
            break

    return neg_boundary, pos_boundary


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_results(all_results, output_dir, ghost_features):
    """Generate loss-vs-α and KL-vs-α plots."""
    output_dir = Path(output_dir)

    # 1. KL vs α for all features
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    for name, res in sorted(all_results.items()):
        if name.startswith("random_"):
            continue
        style = "--" if name in ghost_features else "-"
        color = "gray" if name in ghost_features else None
        ax.plot(res["alphas"], res["kl_values"], style, label=name,
                color=color, alpha=0.7)
    ax.set_xlabel("Steering magnitude α")
    ax.set_ylabel("KL divergence (nats)")
    ax.set_title("KL divergence vs steering magnitude")
    ax.legend(fontsize=7, ncol=2)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for name, res in sorted(all_results.items()):
        if name.startswith("random_"):
            continue
        style = "--" if name in ghost_features else "-"
        color = "gray" if name in ghost_features else None
        ax.plot(res["alphas"], res["losses"], style, label=name,
                color=color, alpha=0.7)
    ax.set_xlabel("Steering magnitude α")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs steering magnitude")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "steering_sweep_all.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. OOD boundary comparison: ghost vs used
    fig, ax = plt.subplots(figsize=(10, 6))
    ghost_bounds = []
    used_bounds = []
    random_bounds = []
    labels_ghost = []
    labels_used = []
    labels_random = []

    for name, res in sorted(all_results.items()):
        neg_b, pos_b = res.get("ood_boundary_neg"), res.get("ood_boundary_pos")
        # use the minimum absolute boundary
        if pos_b is not None and neg_b is not None:
            bound = min(abs(pos_b), abs(neg_b))
        elif pos_b is not None:
            bound = abs(pos_b)
        elif neg_b is not None:
            bound = abs(neg_b)
        else:
            bound = 5.0  # no boundary found within range

        if name.startswith("random_"):
            random_bounds.append(bound)
            labels_random.append(name)
        elif name in ghost_features:
            ghost_bounds.append(bound)
            labels_ghost.append(name)
        else:
            used_bounds.append(bound)
            labels_used.append(name)

    categories = []
    boundaries = []
    colors = []
    tick_labels = []

    for l, b in zip(labels_used, used_bounds):
        categories.append("Used")
        boundaries.append(b)
        colors.append("tab:blue")
        tick_labels.append(l)
    for l, b in zip(labels_ghost, ghost_bounds):
        categories.append("Ghost")
        boundaries.append(b)
        colors.append("tab:gray")
        tick_labels.append(l)
    for l, b in zip(labels_random, random_bounds):
        categories.append("Random")
        boundaries.append(b)
        colors.append("tab:red")
        tick_labels.append(l)

    x = np.arange(len(boundaries))
    ax.bar(x, boundaries, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("OOD boundary |α|")
    ax.set_title("OOD boundary: Ghost vs Used vs Random directions")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="tab:blue", label="Used feature"),
                       Patch(facecolor="tab:gray", label="Ghost feature"),
                       Patch(facecolor="tab:red", label="Random direction")]
    ax.legend(handles=legend_elements)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "ood_boundary_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Individual feature plots
    indiv_dir = output_dir / "individual"
    indiv_dir.mkdir(exist_ok=True)
    for name, res in all_results.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(res["alphas"], res["losses"], "b-o", markersize=3)
        ax1.axhline(res["baseline_loss"], color="r", linestyle="--", alpha=0.5,
                     label=f"baseline={res['baseline_loss']:.3f}")
        ax1.set_xlabel("α")
        ax1.set_ylabel("Loss")
        ax1.set_title(f"{name}: Loss vs α")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(res["alphas"], res["kl_values"], "r-o", markersize=3)
        if res.get("ood_boundary_pos") is not None:
            ax2.axvline(res["ood_boundary_pos"], color="g", linestyle="--",
                         alpha=0.7, label=f"+boundary={res['ood_boundary_pos']:.1f}")
        if res.get("ood_boundary_neg") is not None:
            ax2.axvline(res["ood_boundary_neg"], color="g", linestyle="--",
                         alpha=0.7, label=f"-boundary={res['ood_boundary_neg']:.1f}")
        ax2.set_xlabel("α")
        ax2.set_ylabel("KL divergence (nats)")
        ax2.set_title(f"{name}: KL vs α")
        ax2.set_yscale("log")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(indiv_dir / f"{name}.png", dpi=100, bbox_inches="tight")
        plt.close()

    print(f"Plots saved to {output_dir}", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Steering Vector OOD Boundary Detection")
    parser.add_argument("--model", default="EleutherAI/pythia-2.8b-deduped")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--work-dir", required=True, help="exp007 work dir with cached hiddens")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-sentences", type=int, default=100,
                        help="Number of sentences for steering evaluation")
    parser.add_argument("--alpha-min", type=float, default=-5.0)
    parser.add_argument("--alpha-max", type=float, default=5.0)
    parser.add_argument("--alpha-steps", type=int, default=21)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    alphas = np.linspace(args.alpha_min, args.alpha_max, args.alpha_steps).tolist()
    print(f"α range: {alphas[0]:.1f} to {alphas[-1]:.1f}, {len(alphas)} steps", flush=True)

    # ── Step 1: Load exp-007 feature results to get best layers ──────────────
    print("\n=== Step 1: Loading exp-007 feature results ===", flush=True)
    features_dir = work_dir / "features"
    feature_info = {}  # name -> {best_layer, probe_acc, max_delta, ghost}
    for fp in sorted(features_dir.glob("*.json")):
        with open(fp) as f:
            r = json.load(f)
        if r.get("skipped"):
            continue
        name = r["feature"]
        # Classify as ghost: probe acc > 0.7 and max |Δloss| < 0.01 at majority of layers
        probe_accs = r["probe_accuracy"]
        deltas = r["ablation_delta_loss"]
        n_layers = len(probe_accs)
        ghost_count = sum(1 for l in range(n_layers)
                          if float(probe_accs[str(l)]) > 0.7
                          and abs(float(deltas[str(l)])) < 0.01)
        encoded_count = sum(1 for l in range(n_layers)
                            if float(probe_accs[str(l)]) > 0.7)
        ghost_ratio = ghost_count / encoded_count if encoded_count > 0 else 0

        feature_info[name] = {
            "best_layer": r["best_probe_layer"],
            "probe_acc": r["best_probe_acc"],
            "max_delta": abs(r["max_ablation_delta"]),
            "ghost_ratio": ghost_ratio,
            "is_ghost": ghost_ratio > 0.7,  # >70% of encoded layers are ghost
        }
        tag = "GHOST" if feature_info[name]["is_ghost"] else "USED"
        print(f"  {name}: best_layer=L{r['best_probe_layer']}, "
              f"acc={r['best_probe_acc']:.4f}, max|Δ|={abs(r['max_ablation_delta']):.4f}, "
              f"ghost_ratio={ghost_ratio:.2f} [{tag}]", flush=True)

    ghost_features = {n for n, info in feature_info.items() if info["is_ghost"]}
    used_features = {n for n, info in feature_info.items() if not info["is_ghost"]}
    print(f"\nGhost features ({len(ghost_features)}): {sorted(ghost_features)}", flush=True)
    print(f"Used features ({len(used_features)}): {sorted(used_features)}", flush=True)

    # ── Step 2: Retrain probes to get weight vectors ─────────────────────────
    print("\n=== Step 2: Retraining probes to extract direction vectors ===", flush=True)
    with open(work_dir / "meta.json") as f:
        meta = json.load(f)
    d_model = meta["d_model"]

    token_info = torch.load(work_dir / "token_info.pt", map_location="cpu", weights_only=False)
    token_ids = token_info["token_ids"]
    token_strs = token_info["token_strs"]
    token_counts = Counter(token_info["token_counts"])

    steering_vectors = {}  # name -> (layer_idx, W_normalized)
    for name, info in feature_info.items():
        layer_idx = info["best_layer"]
        labels = label_feature(name, token_strs, token_ids, token_counts)
        y = torch.tensor(labels, dtype=torch.long)

        X = torch.load(work_dir / "hiddens" / f"layer_{layer_idx}.pt",
                        map_location="cpu", weights_only=True)
        acc, W, bias = train_probe_gpu(X, y, args.device)
        # Normalize
        W_norm = W / (np.linalg.norm(W) + 1e-8)
        steering_vectors[name] = (layer_idx, W_norm)
        print(f"  {name} @ L{layer_idx}: retrained acc={acc:.4f}", flush=True)

    # Add random directions as controls
    rng = np.random.RandomState(42)
    for i in range(3):
        name = f"random_{i}"
        vec = rng.randn(d_model).astype(np.float32)
        vec /= np.linalg.norm(vec)
        # use middle layer
        mid_layer = meta["n_layers"] // 2
        steering_vectors[name] = (mid_layer, vec)
        feature_info[name] = {"best_layer": mid_layer, "is_ghost": False,
                              "probe_acc": 0.0, "max_delta": 0.0, "ghost_ratio": 0.0}
        print(f"  {name} @ L{mid_layer}: random unit vector", flush=True)

    # ── Step 3: Load model and eval data ─────────────────────────────────────
    print("\n=== Step 3: Loading model ===", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=args.cache_dir, torch_dtype=torch.float16)
    model = model.to(args.device).eval()
    model_layers = get_model_layers(model)

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                      split="validation",
                      cache_dir=str(Path(args.cache_dir) / "datasets"))
    texts = [t for t in ds["text"] if len(t.strip()) > 50][:args.n_sentences]
    print(f"  Eval data: {len(texts)} sentences", flush=True)

    # ── Step 4: Baseline ─────────────────────────────────────────────────────
    print("\n=== Step 4: Computing baseline ===", flush=True)
    baseline_loss, baseline_log_probs = compute_baseline(
        model, tokenizer, texts, args.device, args.batch_size, args.seq_len)
    print(f"  Baseline loss: {baseline_loss:.4f}", flush=True)

    # ── Step 5: Steering sweep ───────────────────────────────────────────────
    print("\n=== Step 5: Steering sweep ===", flush=True)
    all_results = {}

    for feat_name in sorted(steering_vectors.keys()):
        layer_idx, W = steering_vectors[feat_name]
        info = feature_info[feat_name]
        tag = "GHOST" if info.get("is_ghost") else "USED"
        if feat_name.startswith("random_"):
            tag = "RANDOM"
        print(f"\n--- {feat_name} [{tag}] @ L{layer_idx} ---", flush=True)

        losses = []
        kl_values = []

        for alpha in tqdm(alphas, desc=f"  {feat_name}"):
            if alpha == 0.0:
                losses.append(baseline_loss)
                kl_values.append(0.0)
                continue

            steered_loss, steered_log_probs = compute_steered(
                model, tokenizer, texts, model_layers, layer_idx,
                W, alpha, args.device, args.batch_size, args.seq_len)
            kl = compute_kl(baseline_log_probs, steered_log_probs)

            losses.append(steered_loss)
            kl_values.append(kl)

        neg_bound, pos_bound = find_ood_boundary(alphas, kl_values)

        result = {
            "feature": feat_name,
            "layer": layer_idx,
            "is_ghost": info.get("is_ghost", False),
            "ghost_ratio": info.get("ghost_ratio", 0.0),
            "probe_acc": info.get("probe_acc", 0.0),
            "max_ablation_delta": info.get("max_delta", 0.0),
            "baseline_loss": baseline_loss,
            "alphas": alphas,
            "losses": losses,
            "kl_values": kl_values,
            "ood_boundary_neg": neg_bound,
            "ood_boundary_pos": pos_bound,
        }
        all_results[feat_name] = result

        bound_str = f"[{neg_bound}, {pos_bound}]" if neg_bound or pos_bound else "not found"
        print(f"  Loss range: {min(losses):.4f} – {max(losses):.4f}", flush=True)
        print(f"  KL range: {min(kl_values):.4f} – {max(kl_values):.4f}", flush=True)
        print(f"  OOD boundary: {bound_str}", flush=True)

    # ── Step 6: Save results ─────────────────────────────────────────────────
    print("\n=== Step 6: Saving results ===", flush=True)
    with open(output_dir / "steering_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary table
    summary = []
    for name, res in sorted(all_results.items()):
        neg_b, pos_b = res["ood_boundary_neg"], res["ood_boundary_pos"]
        if pos_b is not None and neg_b is not None:
            boundary = min(abs(pos_b), abs(neg_b))
        elif pos_b is not None:
            boundary = abs(pos_b)
        elif neg_b is not None:
            boundary = abs(neg_b)
        else:
            boundary = None

        summary.append({
            "feature": name,
            "layer": res["layer"],
            "is_ghost": res["is_ghost"],
            "ghost_ratio": res.get("ghost_ratio", 0),
            "probe_acc": res.get("probe_acc", 0),
            "max_ablation_delta": res.get("max_ablation_delta", 0),
            "baseline_loss": baseline_loss,
            "max_loss": max(res["losses"]),
            "max_kl": max(res["kl_values"]),
            "ood_boundary": boundary,
            "ood_boundary_neg": neg_b,
            "ood_boundary_pos": pos_b,
        })
    with open(output_dir / "steering_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===", flush=True)
    print(f"{'Feature':<22} {'Type':<7} {'Layer':>5} {'Acc':>6} "
          f"{'MaxΔ':>7} {'MaxKL':>8} {'OOD |α|':>8}", flush=True)
    print("-" * 75, flush=True)
    for s in summary:
        tag = "GHOST" if s["is_ghost"] else "USED"
        if s["feature"].startswith("random_"):
            tag = "RANDOM"
        bound_str = f"{s['ood_boundary']:.2f}" if s["ood_boundary"] is not None else ">5.0"
        print(f"{s['feature']:<22} {tag:<7} L{s['layer']:>3} {s['probe_acc']:>6.3f} "
              f"{s['max_ablation_delta']:>7.4f} {s['max_kl']:>8.3f} {bound_str:>8}", flush=True)

    # ── Step 7: Plot ─────────────────────────────────────────────────────────
    print("\n=== Step 7: Plotting ===", flush=True)
    plot_results(all_results, output_dir, ghost_features)

    print("\n=== Done! ===", flush=True)
    print(f"Results: {output_dir / 'steering_results.json'}", flush=True)
    print(f"Summary: {output_dir / 'steering_summary.json'}", flush=True)
    print(f"Plots: {output_dir}/*.png", flush=True)


if __name__ == "__main__":
    main()
