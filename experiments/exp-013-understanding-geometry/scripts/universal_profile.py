#!/usr/bin/env python3
"""
Exp-013: Universal model ID profile — works with any HF causal LM.

Usage:
  CUDA_VISIBLE_DEVICES=4 python universal_profile.py \
    --model tomg-group-umd/huginn-0125 --name Huginn \
    --cache-dir /nvmessd/lifanhong/LR-env/cache/hf \
    --out-dir /nvmessd/lifanhong/LR-env/exp013_profiles
"""

import argparse, json, os
from pathlib import Path
import numpy as np, torch
from tqdm import tqdm
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
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
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--name", required=True, help="Short name for output")
    parser.add_argument("--cache-dir", default="/nvmessd/lifanhong/LR-env/cache/hf")
    parser.add_argument("--out-dir", default="/nvmessd/lifanhong/LR-env/exp013_profiles")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-sentences", type=int, default=500)
    args = parser.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.name}.json"
    if out_path.exists():
        print(f"SKIP: {out_path} exists"); return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"Loading {args.name} ({args.model})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir,
                                               trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=args.cache_dir, torch_dtype=torch.float16,
        output_hidden_states=True, trust_remote_code=True)
    model = model.to(args.device).eval()

    # Get n_layers
    if hasattr(model.config, "num_hidden_layers"):
        n_layers = model.config.num_hidden_layers
    elif hasattr(model.config, "n_layer"):
        n_layers = model.config.n_layer
    else:
        n_layers = len(model(torch.zeros(1,1, dtype=torch.long, device=args.device),
                             output_hidden_states=True).hidden_states) - 1

    d_model = model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.n_embd
    print(f"  {n_layers} layers, d={d_model}", flush=True)

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation",
                      cache_dir=str(Path(args.cache_dir) / "datasets"))
    texts = [t for t in ds["text"] if len(t.strip()) > 30][:args.n_sentences]

    # Cache hidden states (mean-pooled per sentence)
    layer_h = {l: [] for l in range(n_layers)}
    for i in tqdm(range(0, len(texts), 16), desc=f"Cache {args.name}"):
        batch = texts[i:i+16]
        tokens = tokenizer(batch, return_tensors="pt", max_length=128,
                           truncation=True, padding="max_length").to(args.device)
        with torch.no_grad():
            out = model(**tokens)
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        for l in range(n_layers):
            h = out.hidden_states[l+1].cpu().float()
            pooled = (h * mask.cpu()).sum(dim=1) / mask.cpu().sum(dim=1).clamp(min=1)
            layer_h[l].append(pooled.numpy())

    for l in range(n_layers):
        layer_h[l] = np.concatenate(layer_h[l], axis=0)

    # ID profile
    print(f"\nID Profile:", flush=True)
    id_profile = []
    for l in tqdm(range(n_layers), desc="ID"):
        tid = twonn_id(layer_h[l]); sr = stable_rank(layer_h[l])
        id_profile.append({"layer": l, "twonn_id": tid, "stable_rank": sr})
        if l % max(1, n_layers//8) == 0 or l == n_layers-1:
            print(f"  L{l}: ID={tid:.1f}, srank={sr:.1f}")

    ids = [p["twonn_id"] for p in id_profile]
    has_hunch = any(ids[l] > ids[0] and ids[l] > ids[-1] for l in range(1, n_layers-1))
    peak = int(np.argmax(ids))

    # Figure
    fig, ax = plt.subplots(figsize=(10, 5))
    layers = list(range(n_layers))
    ax.plot(layers, ids, "o-", label="TwoNN ID", markersize=4)
    ax.plot(layers, [p["stable_rank"] for p in id_profile], "s-", label="Stable Rank", markersize=4)
    ax.set_xlabel("Layer"); ax.set_ylabel("Metric")
    ax.set_title(f"{args.name} — Hunchback: {'YES' if has_hunch else 'NO'}")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{args.name}_profile.png", dpi=150)

    results = {"model": args.model, "name": args.name, "n_layers": n_layers, "d_model": d_model,
               "id_profile": id_profile, "has_hunchback": has_hunch, "peak_layer": peak,
               "id_range": [float(min(ids)), float(max(ids))]}
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}\nSUMMARY: {args.name}\n{'='*60}")
    print(f"  Hunchback: {'YES' if has_hunch else 'NO'} (peak L{peak}, ID={ids[peak]:.1f})")
    print(f"  ID range: {min(ids):.1f} - {max(ids):.1f}")

if __name__ == "__main__":
    main()
