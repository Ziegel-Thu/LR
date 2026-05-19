#!/usr/bin/env python3
"""
Exp-013: DINOv2 + Whisper ID profiles — more understanding models.

DINOv2: self-supervised vision (should have hunchback?)
Whisper: speech understanding (should have hunchback?)

Usage:
  CUDA_VISIBLE_DEVICES=2 python vision_audio_profiles.py --model dinov2 ...
  CUDA_VISIBLE_DEVICES=3 python vision_audio_profiles.py --model whisper ...
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

def run_dinov2(cache_dir, out_dir, device, n_samples=500):
    from transformers import AutoModel
    print("Loading DINOv2-base...", flush=True)
    model = AutoModel.from_pretrained("facebook/dinov2-base", cache_dir=cache_dir,
                                       torch_dtype=torch.float16).to(device).eval()
    n_layers = model.config.num_hidden_layers; d = model.config.hidden_size
    print(f"  {n_layers} layers, d={d}", flush=True)

    layer_h = {l: [] for l in range(n_layers)}
    for i in tqdm(range(0, n_samples, 8), desc="DINOv2"):
        bs = min(8, n_samples - i)
        x = torch.randn(bs, 3, 224, 224, dtype=torch.float16, device=device)
        with torch.no_grad():
            out = model(x, output_hidden_states=True)
        for l in range(n_layers):
            h = out.hidden_states[l+1][:, 0].cpu().float().numpy()  # CLS token
            layer_h[l].append(h)

    for l in range(n_layers):
        layer_h[l] = np.concatenate(layer_h[l])

    profile = []
    for l in range(n_layers):
        tid = twonn_id(layer_h[l]); sr = stable_rank(layer_h[l])
        profile.append({"layer": l, "twonn_id": tid, "stable_rank": sr})
        print(f"  L{l}: ID={tid:.1f}, srank={sr:.1f}")

    ids = [p["twonn_id"] for p in profile]
    has_hunch = any(ids[l] > ids[0] and ids[l] > ids[-1] for l in range(1, n_layers-1))
    return {"name": "DINOv2-base", "model": "facebook/dinov2-base", "n_layers": n_layers,
            "d_model": d, "id_profile": profile, "has_hunchback": has_hunch,
            "peak_layer": int(np.argmax(ids)), "id_range": [float(min(ids)), float(max(ids))]}

def run_whisper(cache_dir, out_dir, device, n_samples=500):
    from transformers import WhisperModel
    print("Loading Whisper-small...", flush=True)
    model = WhisperModel.from_pretrained("openai/whisper-small", cache_dir=cache_dir,
                                          torch_dtype=torch.float16).to(device).eval()
    encoder = model.encoder
    n_layers = model.config.encoder_layers; d = model.config.d_model
    print(f"  Encoder: {n_layers} layers, d={d}", flush=True)

    layer_h = {l: [] for l in range(n_layers)}
    # Whisper encoder expects log-mel spectrogram: (B, 80, 3000)
    for i in tqdm(range(0, n_samples, 16), desc="Whisper"):
        bs = min(16, n_samples - i)
        x = torch.randn(bs, 80, 3000, dtype=torch.float16, device=device)
        with torch.no_grad():
            out = encoder(x, output_hidden_states=True)
        for l in range(n_layers):
            h = out.hidden_states[l+1].mean(dim=1).cpu().float().numpy()  # mean pool
            layer_h[l].append(h)

    for l in range(n_layers):
        layer_h[l] = np.concatenate(layer_h[l])

    profile = []
    for l in range(n_layers):
        tid = twonn_id(layer_h[l]); sr = stable_rank(layer_h[l])
        profile.append({"layer": l, "twonn_id": tid, "stable_rank": sr})
        print(f"  L{l}: ID={tid:.1f}, srank={sr:.1f}")

    ids = [p["twonn_id"] for p in profile]
    has_hunch = any(ids[l] > ids[0] and ids[l] > ids[-1] for l in range(1, n_layers-1))
    return {"name": "Whisper-small-encoder", "model": "openai/whisper-small", "n_layers": n_layers,
            "d_model": d, "id_profile": profile, "has_hunchback": has_hunch,
            "peak_layer": int(np.argmax(ids)), "id_range": [float(min(ids)), float(max(ids))]}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["dinov2", "whisper"])
    parser.add_argument("--cache-dir", default="/nvmessd/lifanhong/LR-env/cache/hf")
    parser.add_argument("--out-dir", default="/nvmessd/lifanhong/LR-env/exp013_profiles")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "dinov2":
        result = run_dinov2(args.cache_dir, out_dir, args.device)
    else:
        result = run_whisper(args.cache_dir, out_dir, args.device)

    with open(out_dir / f"{result['name']}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    hstr = "YES" if result["has_hunchback"] else "NO"
    peak = result["peak_layer"]
    ids = [p["twonn_id"] for p in result["id_profile"]]
    print(f"\nSUMMARY: {result['name']}")
    print(f"  Hunchback: {hstr} (peak L{peak}, ID={ids[peak]:.1f})")

if __name__ == "__main__":
    main()
