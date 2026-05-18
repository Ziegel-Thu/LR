#!/usr/bin/env python3
"""
Exp-013: CLIP ViT-B/32 ID profile — hunchback test on a cross-modal model.

Usage:
  CUDA_VISIBLE_DEVICES=4 python clip_profile.py \
    --cache-dir /nvmessd/lifanhong/LR-env/cache/hf \
    --out-dir /nvmessd/lifanhong/LR-env/exp013_clip
"""

import argparse, json, os
from pathlib import Path
import numpy as np, torch
from tqdm import tqdm
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def twonn_id(X):
    from sklearn.neighbors import NearestNeighbors
    if len(X) > 2000: X = X[np.random.choice(len(X), 2000, replace=False)]
    nn = NearestNeighbors(n_neighbors=3, metric="euclidean").fit(X)
    d, _ = nn.kneighbors(X); r1, r2 = d[:,1], d[:,2]
    mask = r1 > 1e-10; mu = r2[mask]/r1[mask]
    s = np.sum(np.log(mu)); return float(len(mu)/s) if s > 1e-10 else 0

def stable_rank(X):
    if len(X) > 2000: X = X[np.random.choice(len(X), 2000, replace=False)]
    sv = np.linalg.svd(X - X.mean(0), compute_uv=False)
    s2 = sv**2; return float(s2.sum()/(s2[0]+1e-10))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="/nvmessd/lifanhong/LR-env/cache/hf")
    parser.add_argument("--out-dir", default="/nvmessd/lifanhong/LR-env/exp013_clip")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import CLIPModel, CLIPTokenizer
    from datasets import load_dataset

    print("Loading CLIP ViT-B/32...", flush=True)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=args.cache_dir)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", cache_dir=args.cache_dir)
    model = model.to(args.device).eval()

    # Use text encoder for ID profile (comparable to LLMs)
    text_model = model.text_model
    n_layers = text_model.config.num_hidden_layers
    d_model = text_model.config.hidden_size
    print(f"  Text encoder: {n_layers} layers, d={d_model}", flush=True)

    # Load text data
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation",
                      cache_dir=str(Path(args.cache_dir) / "datasets"))
    texts = [t for t in ds["text"] if len(t.strip()) > 30][:2000]
    print(f"  {len(texts)} texts", flush=True)

    # Cache hidden states per layer
    layer_h = {l: [] for l in range(n_layers)}
    for i in tqdm(range(0, len(texts), 32), desc="Cache"):
        batch = texts[i:i+32]
        tokens = tokenizer(batch, return_tensors="pt", max_length=77, truncation=True,
                           padding="max_length").to(args.device)
        with torch.no_grad():
            out = text_model(**tokens, output_hidden_states=True)
        # Use CLS token (first position) representation
        for l in range(n_layers):
            h = out.hidden_states[l+1][:, 0].cpu().float().numpy()  # CLS token
            layer_h[l].append(h)

    for l in range(n_layers):
        layer_h[l] = np.concatenate(layer_h[l], axis=0)

    # Compute ID profile
    print(f"\n{'='*60}\nCLIP Text Encoder ID Profile\n{'='*60}", flush=True)
    id_profile = []
    for l in range(n_layers):
        tid = twonn_id(layer_h[l]); sr = stable_rank(layer_h[l])
        id_profile.append({"layer": l, "twonn_id": tid, "stable_rank": sr})
        print(f"  L{l}: ID={tid:.1f}, srank={sr:.1f}")

    # Hunchback detection
    ids = [p["twonn_id"] for p in id_profile]
    has_hunch = any(ids[l] > ids[0] and ids[l] > ids[-1] for l in range(1, n_layers-1))
    peak_layer = np.argmax(ids)

    # Figure
    fig, ax = plt.subplots(figsize=(10, 5))
    layers = list(range(n_layers))
    ax.plot(layers, ids, "o-", label="TwoNN ID", color="blue", markersize=6)
    ax.plot(layers, [p["stable_rank"] for p in id_profile], "s-", label="Stable Rank",
            color="red", markersize=6)
    ax.set_xlabel("Layer"); ax.set_ylabel("Metric")
    ax.set_title(f"CLIP ViT-B/32 Text Encoder — Hunchback: {'YES' if has_hunch else 'NO'}")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "clip_profile.png", dpi=150)

    results = {"model": "openai/clip-vit-base-patch32", "n_layers": n_layers, "d_model": d_model,
               "id_profile": id_profile, "has_hunchback": has_hunch, "peak_layer": int(peak_layer)}
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"  Hunchback: {'YES' if has_hunch else 'NO'} (peak @ L{peak_layer}, ID={ids[peak_layer]:.1f})")
    print(f"  ID range: {min(ids):.1f} - {max(ids):.1f}")

if __name__ == "__main__":
    main()
