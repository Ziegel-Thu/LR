#!/usr/bin/env python3
"""
Exp-013: V-JEPA 2 ID profile — vision model hunchback test.

V-JEPA 2 is a video encoder (ViT). We feed random video frames and
extract per-layer hidden states for ID analysis.

Usage:
  CUDA_VISIBLE_DEVICES=7 python vjepa2_profile.py \
    --cache-dir /nvmessd/lifanhong/LR-env/cache/hf \
    --out-dir /nvmessd/lifanhong/LR-env/exp013_profiles
"""

import argparse, json, os
from pathlib import Path
import numpy as np, torch
from tqdm import tqdm
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"

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
    parser.add_argument("--out-dir", default="/nvmessd/lifanhong/LR-env/exp013_profiles")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-samples", type=int, default=500)
    args = parser.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModel, AutoConfig

    print("Loading V-JEPA 2 ViT-L...", flush=True)
    config = AutoConfig.from_pretrained(MODEL_ID, cache_dir=args.cache_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, cache_dir=args.cache_dir,
                                       trust_remote_code=True, torch_dtype=torch.float16)
    model = model.to(args.device).eval()

    n_layers = config.num_hidden_layers
    d_model = config.hidden_size
    # V-JEPA 2 expects pixel_values: (B, C, T, H, W) for video
    # or (B, C, H, W) for single frame
    patch_size = getattr(config, "patch_size", 16)
    img_size = getattr(config, "image_size", 256)
    n_channels = getattr(config, "num_channels", 3)
    n_frames = getattr(config, "num_frames", 16)
    print(f"  {n_layers} layers, d={d_model}, img={img_size}, frames={n_frames}", flush=True)

    # Generate random "video" inputs and extract hidden states
    print(f"Generating {args.n_samples} random inputs...", flush=True)
    layer_h = {l: [] for l in range(n_layers)}
    batch_size = 4

    for i in tqdm(range(0, args.n_samples, batch_size), desc="Cache"):
        bs = min(batch_size, args.n_samples - i)
        # Random pixel values as input (normalized)
        pixel_values = torch.randn(bs, n_channels, n_frames, img_size, img_size,
                                    dtype=torch.float16, device=args.device)
        try:
            with torch.no_grad():
                out = model(pixel_values, output_hidden_states=True)
            # hidden_states: tuple of (B, n_patches, d_model) per layer
            for l in range(n_layers):
                h = out.hidden_states[l+1]  # skip embedding
                # Mean pool over patches
                pooled = h.mean(dim=1).cpu().float().numpy()  # (B, d_model)
                layer_h[l].append(pooled)
        except Exception as e:
            # Try single-frame input
            if i == 0:
                print(f"  Video input failed: {e}", flush=True)
                print("  Trying single-frame...", flush=True)
                pixel_values = torch.randn(bs, n_channels, img_size, img_size,
                                            dtype=torch.float16, device=args.device)
                try:
                    with torch.no_grad():
                        out = model(pixel_values, output_hidden_states=True)
                    for l in range(n_layers):
                        h = out.hidden_states[l+1]
                        pooled = h.mean(dim=1).cpu().float().numpy()
                        layer_h[l].append(pooled)
                except Exception as e2:
                    print(f"  Single-frame also failed: {e2}", flush=True)
                    return

    for l in range(n_layers):
        layer_h[l] = np.concatenate(layer_h[l], axis=0)

    n_samples = layer_h[0].shape[0]
    print(f"  Cached: {n_samples} samples × {n_layers} layers", flush=True)

    # ID profile
    print(f"\nID Profile:", flush=True)
    id_profile = []
    for l in tqdm(range(n_layers), desc="ID"):
        tid = twonn_id(layer_h[l]); sr = stable_rank(layer_h[l])
        id_profile.append({"layer": l, "twonn_id": tid, "stable_rank": sr})
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
    hstr = "YES" if has_hunch else "NO"
    ax.set_title(f"V-JEPA 2 ViT-L — Hunchback: {hstr}")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "VJEPA2_profile.png", dpi=150)

    results = {"model": MODEL_ID, "name": "V-JEPA2-ViT-L", "n_layers": n_layers,
               "d_model": d_model, "n_samples": n_samples,
               "id_profile": id_profile, "has_hunchback": has_hunch, "peak_layer": peak,
               "id_range": [float(min(ids)), float(max(ids))],
               "note": "random pixel input (no real video data)"}
    with open(out_dir / "VJEPA2.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"SUMMARY: V-JEPA 2 ViT-L")
    print(f"  Hunchback: {hstr} (peak L{peak}, ID={ids[peak]:.1f})")
    print(f"  ID range: {min(ids):.1f} - {max(ids):.1f}")

if __name__ == "__main__":
    main()
