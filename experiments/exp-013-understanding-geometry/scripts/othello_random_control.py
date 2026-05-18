#!/usr/bin/env python3
"""
Exp-013: Random OthelloGPT control — does hunchback come from training or architecture?

Usage:
  CUDA_VISIBLE_DEVICES=4 python othello_random_control.py \
    --out-dir /nvmessd/lifanhong/LR-env/exp013_othello
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
    parser.add_argument("--out-dir", default="/nvmessd/lifanhong/LR-env/exp013_othello")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    import transformer_lens as tl

    # Load trained OthelloGPT
    print("Loading trained OthelloGPT...", flush=True)
    trained = tl.HookedTransformer.from_pretrained("othello-gpt", device=args.device)
    n_layers = trained.cfg.n_layers; d_model = trained.cfg.d_model

    # Create random OthelloGPT (same architecture, random weights)
    print("Creating random OthelloGPT...", flush=True)
    random_model = tl.HookedTransformer.from_pretrained("othello-gpt", device=args.device)
    for p in random_model.parameters():
        torch.nn.init.normal_(p, mean=0, std=0.02)

    # Generate random token sequences (valid range 0-60)
    print("Generating sequences...", flush=True)
    n_seqs = 3000; seq_len = 30
    seqs = [torch.randint(0, 60, (seq_len,)) for _ in range(n_seqs)]

    # Cache hidden states for both
    results = {}
    for name, model in [("trained", trained), ("random", random_model)]:
        print(f"\n{'='*40} {name} {'='*40}", flush=True)
        layer_h = {l: [] for l in range(n_layers)}

        for i in tqdm(range(n_seqs), desc=f"Cache {name}"):
            seq = seqs[i].unsqueeze(0).to(args.device)
            _, cache = model.run_with_cache(seq)
            for l in range(n_layers):
                h = cache[f"blocks.{l}.hook_resid_post"][0, -1].cpu().float().numpy()
                layer_h[l].append(h)

        for l in range(n_layers):
            layer_h[l] = np.stack(layer_h[l])

        id_profile = []
        for l in range(n_layers):
            tid = twonn_id(layer_h[l]); sr = stable_rank(layer_h[l])
            id_profile.append({"layer": l, "twonn_id": tid, "stable_rank": sr})
            print(f"  L{l}: ID={tid:.1f}, srank={sr:.1f}")

        ids = [p["twonn_id"] for p in id_profile]
        has_hunch = any(ids[l] > ids[0] and ids[l] > ids[-1] for l in range(1, n_layers-1))
        results[name] = {"id_profile": id_profile, "has_hunchback": has_hunch}
        print(f"  Hunchback: {'YES' if has_hunch else 'NO'}")

    # Figure: trained vs random
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    layers = list(range(n_layers))
    for mi, metric in enumerate(["twonn_id", "stable_rank"]):
        for name, color in [("trained", "blue"), ("random", "red")]:
            vals = [p[metric] for p in results[name]["id_profile"]]
            axes[mi].plot(layers, vals, "o-", label=name, color=color, markersize=6)
        axes[mi].set_xlabel("Layer"); axes[mi].set_ylabel(metric)
        axes[mi].set_title(f"OthelloGPT {metric}: Trained vs Random")
        axes[mi].legend(); axes[mi].grid(True, alpha=0.3)

    fig.suptitle("Exp-013: Hunchback — Training Effect or Architecture Effect?", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "trained_vs_random.png", dpi=150)

    with open(out_dir / "random_control.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"  Trained: hunchback={'YES' if results['trained']['has_hunchback'] else 'NO'}")
    print(f"  Random:  hunchback={'YES' if results['random']['has_hunchback'] else 'NO'}")
    if results['trained']['has_hunchback'] and not results['random']['has_hunchback']:
        print("  → Hunchback is a TRAINING effect (understanding signature)")
    elif results['trained']['has_hunchback'] and results['random']['has_hunchback']:
        print("  → Hunchback is an ARCHITECTURE effect")
    else:
        print("  → Unexpected pattern")

if __name__ == "__main__":
    main()
