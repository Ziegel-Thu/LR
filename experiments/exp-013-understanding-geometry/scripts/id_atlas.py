#!/usr/bin/env python3
"""
Exp-013 Phase 2 (partial): ID-profile atlas for cached models.

Computes TwoNN ID + MLE ID + stable rank depth profile for models
where we already have reps (Pythia, Mamba, RWKV at multiple scales).

Usage:
  python id_atlas.py --data-dir /nvmessd/lifanhong/LR-env/exp003_reps \
    --out-dir /nvmessd/lifanhong/LR-env/exp013
"""

import argparse, json
from pathlib import Path
import numpy as np, torch
from tqdm import tqdm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

MODELS = [
    ("Pythia-70M", "Pythia-70M_reps.pt"),
    ("Pythia-410M", "Pythia-410M_reps.pt"),
    ("Pythia-1.4B", "Pythia-1.4B_reps.pt"),
    ("Pythia-6.9B", "Pythia-6.9B_reps.pt"),
    ("Mamba-370M", "Mamba-370M_reps.pt"),
    ("Mamba-1.4B", "Mamba-1.4B_reps.pt"),
    ("Mamba-2.8B", "Mamba-2.8B_reps.pt"),
    ("RWKV-430M", "RWKV-430M_reps.pt"),
    ("RWKV-1.5B", "RWKV-1.5B_reps.pt"),
    ("RWKV-4-3B", "RWKV-4-3B_reps.pt"),
]

def twonn_id(X):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=3, metric="euclidean").fit(X)
    d, _ = nn.kneighbors(X)
    r1, r2 = d[:,1], d[:,2]
    mask = r1 > 1e-10; mu = r2[mask]/r1[mask]
    s = np.sum(np.log(mu))
    return float(len(mu)/s) if s > 1e-10 else 0

def mle_id(X, k=100):
    from sklearn.neighbors import NearestNeighbors
    k = min(k, len(X)-1)
    nn = NearestNeighbors(n_neighbors=k+1, metric="euclidean").fit(X)
    d, _ = nn.kneighbors(X)
    ids = []
    for i in range(len(X)):
        di = d[i,1:]; di = di[di>1e-10]
        if len(di)<2: continue
        ids.append((len(di)-1)/np.sum(np.log(di[-1]/di[:-1])))
    return float(np.mean(ids)) if ids else 0

def stable_rank(X):
    sv = np.linalg.svd(X-X.mean(0), compute_uv=False)
    s2 = sv**2; return float(s2.sum()/(s2[0]+1e-10))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", default="/nvmessd/lifanhong/LR-env/exp013")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, fname in MODELS:
        path = data_dir / fname
        if not path.exists():
            print(f"SKIP {name}"); continue
        reps = torch.load(path, map_location="cpu", weights_only=True).numpy()
        n_layers = reps.shape[0]
        print(f"\n{name}: {reps.shape}", flush=True)

        profiles = []
        for l in tqdm(range(n_layers), desc=f"  {name}"):
            X = reps[l]
            profiles.append({
                "layer": l, "depth": l/(n_layers-1),
                "twonn_id": twonn_id(X),
                "mle_id": mle_id(X),
                "stable_rank": stable_rank(X),
            })
        results[name] = {"n_layers": n_layers, "d_model": int(reps.shape[2]), "profiles": profiles}

    # Figure: ID profiles for all models
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))
    for ci, (name, r) in enumerate(results.items()):
        depths = [p["depth"] for p in r["profiles"]]
        for mi, metric in enumerate(["twonn_id", "mle_id", "stable_rank"]):
            vals = [p[metric] for p in r["profiles"]]
            axes[mi].plot(depths, vals, "-", label=name, color=colors[ci], linewidth=1.5, alpha=0.8)

    for mi, metric in enumerate(["TwoNN ID", "MLE ID", "Stable Rank"]):
        axes[mi].set_xlabel("Normalized depth")
        axes[mi].set_ylabel(metric)
        axes[mi].set_title(f"{metric} Depth Profile")
        axes[mi].legend(fontsize=6, ncol=2)
        axes[mi].grid(True, alpha=0.3)

    fig.suptitle("Exp-013: Cross-Model ID-Profile Atlas", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "id_atlas.png", dpi=150)
    print(f"\nFigure: {out_dir / 'id_atlas.png'}")

    # Hunchback detection
    print(f"\n{'='*60}\nHunchback Analysis\n{'='*60}")
    for name, r in results.items():
        ids = [p["twonn_id"] for p in r["profiles"]]
        mid = len(ids)//2
        early = np.mean(ids[:max(1,mid//2)])
        middle = np.mean(ids[mid-2:mid+2])
        late = np.mean(ids[-max(1,mid//2):])
        has_hunch = middle < early and middle < late
        print(f"  {name:>15}: early={early:.1f}, mid={middle:.1f}, late={late:.1f} "
              f"{'← HUNCHBACK' if has_hunch else ''}")

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults: {out_dir / 'results.json'}")

if __name__ == "__main__":
    main()
