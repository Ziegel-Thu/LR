#!/usr/bin/env python3
"""
Exp-009 Phase 1+2: LRH systematic test — concept direction consistency.

Extracts concept directions via difference-in-means at each layer of Pythia-1.4B,
computes cross-layer cosine similarity matrices, and concept-concept angle matrices.

Uses heuristic labeling (same approach as exp-007) for binary concepts.

Usage:
  CUDA_VISIBLE_DEVICES=0 python run_lrh_test.py \
    --cache-dir /nvmessd/lifanhong/LR-env/cache/hf \
    --out-dir /nvmessd/lifanhong/LR-env/exp009
"""

import argparse
import gc
import json
import os
import time
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


# ── Concept labelers (heuristic, token-level) ────────────────────────────────

def label_concepts(token_strs: list[str]) -> dict[str, np.ndarray]:
    """Label tokens for 10 binary concepts."""
    n = len(token_strs)

    concepts = {}

    # 1. Capitalized (proxy for proper nouns / sentence start)
    concepts["capitalized"] = np.array(
        [1 if t.lstrip() and t.lstrip()[0].isupper() else 0 for t in token_strs])

    # 2. Numeric
    concepts["numeric"] = np.array(
        [1 if any(c.isdigit() for c in t) else 0 for t in token_strs])

    # 3. Punctuation
    concepts["punctuation"] = np.array(
        [1 if t.strip() and all(not c.isalnum() for c in t.strip()) else 0 for t in token_strs])

    # 4. Short token (≤2 chars, proxy for function words)
    concepts["short"] = np.array(
        [1 if len(t.strip()) <= 2 else 0 for t in token_strs])

    # 5. Plural (heuristic)
    concepts["plural"] = np.array(
        [1 if len(t.strip()) > 2 and t.strip().lower().endswith(("s", "es", "ies"))
         else 0 for t in token_strs])

    # 6. Past tense (heuristic: ends with -ed)
    concepts["past_tense"] = np.array(
        [1 if len(t.strip()) > 3 and t.strip().lower().endswith("ed") else 0
         for t in token_strs])

    # 7. Subword (doesn't start with space/Ġ)
    concepts["subword"] = np.array(
        [1 if not (t.startswith(" ") or t.startswith("Ġ")) else 0 for t in token_strs])

    # 8. Title case
    concepts["title_case"] = np.array(
        [1 if t.strip() and len(t.strip()) > 1 and t.strip()[0].isupper()
         and t.strip()[1:].islower() else 0 for t in token_strs])

    # 9. Contains hyphen
    concepts["hyphenated"] = np.array(
        [1 if "-" in t else 0 for t in token_strs])

    # 10. Starts with vowel
    concepts["starts_vowel"] = np.array(
        [1 if t.strip().lower()[:1] in "aeiou" else 0 for t in token_strs])

    return concepts


def difference_in_means(hidden: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute the concept direction via difference-in-means."""
    pos_mask = labels == 1
    neg_mask = labels == 0
    if pos_mask.sum() < 10 or neg_mask.sum() < 10:
        return None
    mean_pos = hidden[pos_mask].mean(axis=0)
    mean_neg = hidden[neg_mask].mean(axis=0)
    direction = mean_pos - mean_neg
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        return None
    return direction / norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="/nvmessd/lifanhong/LR-env/cache/hf")
    parser.add_argument("--out-dir", default="/nvmessd/lifanhong/LR-env/exp009")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    # Load model
    print("Loading model...", flush=True)
    tokenizer = with_retry(
        lambda: AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=args.cache_dir), "tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = with_retry(
        lambda: AutoModelForCausalLM.from_pretrained(
            MODEL_ID, cache_dir=args.cache_dir, torch_dtype=torch.float16,
            output_hidden_states=True), "model")
    model = model.to(args.device).eval()
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    print(f"  {n_layers} layers, d={d_model}", flush=True)

    # Load data
    ds = with_retry(
        lambda: load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                             split="validation",
                             cache_dir=str(Path(args.cache_dir) / "datasets")), "dataset")
    texts = [t for t in ds["text"] if len(t.strip()) > 50][:N_SENTENCES]
    print(f"  {len(texts)} sentences", flush=True)

    # Cache per-layer hidden states and collect token info
    print("\nPhase 0: Caching hidden states...", flush=True)
    layer_hiddens = {l: [] for l in range(n_layers)}
    all_token_strs = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Caching"):
        batch = texts[i:i + BATCH_SIZE]
        tokens = tokenizer(batch, return_tensors="pt", max_length=SEQ_LEN,
                           truncation=True, padding="max_length").to(args.device)
        with torch.no_grad():
            out = model(**tokens)
        mask = tokens["attention_mask"].cpu().bool()

        for layer_idx in range(n_layers):
            h = out.hidden_states[layer_idx + 1].cpu().float()
            for b in range(h.shape[0]):
                layer_hiddens[layer_idx].append(h[b][mask[b]].numpy())

        for b in range(tokens["input_ids"].shape[0]):
            valid = mask[b]
            ids = tokens["input_ids"][b][valid].cpu()
            strs = [tokenizer.decode([tid]) for tid in ids]
            all_token_strs.extend(strs)

    # Concatenate
    for l in range(n_layers):
        layer_hiddens[l] = np.concatenate(layer_hiddens[l], axis=0)
    n_tokens = len(all_token_strs)
    print(f"  {n_tokens} tokens cached", flush=True)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    # Compute concept labels
    concepts = label_concepts(all_token_strs)
    valid_concepts = {}
    for name, labels in concepts.items():
        pos_rate = labels.mean()
        if 0.02 < pos_rate < 0.98:
            valid_concepts[name] = labels
            print(f"  {name}: pos_rate={pos_rate:.3f}")
        else:
            print(f"  {name}: SKIP (pos_rate={pos_rate:.3f})")

    # ── Phase 1: Cross-layer consistency ──
    print(f"\n{'='*60}")
    print("Phase 1: Cross-layer concept direction consistency")
    print(f"{'='*60}")

    directions = {}  # {concept: {layer: direction_vector}}
    cross_layer_sim = {}  # {concept: (n_layers, n_layers) matrix}

    for cname, labels in valid_concepts.items():
        directions[cname] = {}
        for l in range(n_layers):
            d = difference_in_means(layer_hiddens[l], labels)
            directions[cname][l] = d

        # Cross-layer cosine similarity
        sim_matrix = np.zeros((n_layers, n_layers))
        for li in range(n_layers):
            for lj in range(n_layers):
                di = directions[cname][li]
                dj = directions[cname][lj]
                if di is not None and dj is not None:
                    sim_matrix[li, lj] = np.dot(di, dj)
                else:
                    sim_matrix[li, lj] = 0
        cross_layer_sim[cname] = sim_matrix

        # Adjacent layer consistency
        adj_sims = [sim_matrix[l, l+1] for l in range(n_layers - 1)
                    if directions[cname][l] is not None and directions[cname][l+1] is not None]
        mean_adj = np.mean(adj_sims) if adj_sims else 0
        print(f"  {cname}: mean adjacent cosine = {mean_adj:.4f}")

    # ── Phase 2: Concept-concept angle matrix ──
    print(f"\n{'='*60}")
    print("Phase 2: Concept-concept angle matrix")
    print(f"{'='*60}")

    concept_names = list(valid_concepts.keys())
    n_concepts = len(concept_names)

    # At middle layer
    mid_layer = n_layers // 2
    concept_angles_mid = np.zeros((n_concepts, n_concepts))
    for i, ci in enumerate(concept_names):
        for j, cj in enumerate(concept_names):
            di = directions[ci].get(mid_layer)
            dj = directions[cj].get(mid_layer)
            if di is not None and dj is not None:
                concept_angles_mid[i, j] = np.dot(di, dj)

    print(f"  Concept-concept angles at layer {mid_layer}:")
    print(f"  {'':>15}", end="")
    for c in concept_names:
        print(f"  {c[:8]:>8}", end="")
    print()
    for i, ci in enumerate(concept_names):
        print(f"  {ci:>15}", end="")
        for j in range(n_concepts):
            print(f"  {concept_angles_mid[i,j]:>8.3f}", end="")
        print()

    # Check near-orthogonality
    off_diag = concept_angles_mid[np.triu_indices(n_concepts, k=1)]
    print(f"\n  Off-diagonal cosines: mean={np.mean(np.abs(off_diag)):.4f}, "
          f"max={np.max(np.abs(off_diag)):.4f}")
    print(f"  Near-orthogonal (<0.1): {(np.abs(off_diag) < 0.1).sum()}/{len(off_diag)}")

    # ── Figures ──
    # 1. Cross-layer heatmaps
    n_valid = len(valid_concepts)
    cols = min(5, n_valid)
    rows = (n_valid + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)

    for idx, cname in enumerate(concept_names):
        ax = axes[idx // cols, idx % cols]
        im = ax.imshow(cross_layer_sim[cname], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_title(cname, fontsize=10)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Layer")

    # Hide unused subplots
    for idx in range(n_valid, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    fig.suptitle("Exp-009: Cross-Layer Concept Direction Consistency", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "cross_layer_heatmaps.png", dpi=150)
    print(f"\nFig 1: {out_dir / 'cross_layer_heatmaps.png'}")

    # 2. Concept-concept angle matrix
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 7))
    im2 = ax2.imshow(concept_angles_mid, cmap="RdBu_r", vmin=-1, vmax=1)
    ax2.set_xticks(range(n_concepts))
    ax2.set_yticks(range(n_concepts))
    ax2.set_xticklabels(concept_names, rotation=45, ha="right", fontsize=8)
    ax2.set_yticklabels(concept_names, fontsize=8)
    plt.colorbar(im2, ax=ax2, label="Cosine similarity")
    ax2.set_title(f"Concept-Concept Angles (Layer {mid_layer})")
    fig2.tight_layout()
    fig2.savefig(out_dir / "concept_angles.png", dpi=150)
    print(f"Fig 2: {out_dir / 'concept_angles.png'}")

    # 3. Adjacent-layer consistency profile
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 5))
    for cname in concept_names:
        sims = []
        for l in range(n_layers - 1):
            di = directions[cname][l]
            dj = directions[cname][l + 1]
            if di is not None and dj is not None:
                sims.append(np.dot(di, dj))
            else:
                sims.append(0)
        ax3.plot(range(n_layers - 1), sims, "o-", label=cname, markersize=3, alpha=0.7)
    ax3.set_xlabel("Layer")
    ax3.set_ylabel("Cosine similarity with next layer")
    ax3.set_title("Adjacent-Layer Direction Consistency")
    ax3.legend(fontsize=7, ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    fig3.tight_layout()
    fig3.savefig(out_dir / "adjacent_consistency.png", dpi=150)
    print(f"Fig 3: {out_dir / 'adjacent_consistency.png'}")

    # Save results
    results = {
        "model": MODEL_ID,
        "n_layers": n_layers,
        "n_tokens": n_tokens,
        "concepts": {c: {"pos_rate": float(valid_concepts[c].mean())} for c in concept_names},
        "cross_layer_consistency": {
            c: {
                "mean_adjacent_cosine": float(np.mean([
                    cross_layer_sim[c][l, l+1] for l in range(n_layers-1)])),
                "mean_all_cosine": float(np.mean(cross_layer_sim[c])),
            } for c in concept_names
        },
        "concept_angles_mid_layer": mid_layer,
        "concept_angles_mean_abs_offdiag": float(np.mean(np.abs(off_diag))),
        "concept_angles_near_orthogonal_pct": float((np.abs(off_diag) < 0.1).mean()),
    }
    with open(out_dir / "results_phase1_2.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {out_dir / 'results_phase1_2.json'}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for c in concept_names:
        adj = results["cross_layer_consistency"][c]["mean_adjacent_cosine"]
        print(f"  {c:>15}: adj_cosine={adj:.4f}")
    print(f"\n  Concept angles (off-diag): mean_|cos|={results['concept_angles_mean_abs_offdiag']:.4f}")
    print(f"  Near-orthogonal (<0.1): {results['concept_angles_near_orthogonal_pct']:.1%}")


if __name__ == "__main__":
    main()
