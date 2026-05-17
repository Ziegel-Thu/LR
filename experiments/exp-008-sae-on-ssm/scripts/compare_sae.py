#!/usr/bin/env python3
"""
Exp-008 Phase 2: Train SAE on Pythia-160M and compare features with Mamba SAE.

1. Cache Pythia-160M activations at middle layer
2. Train TopK SAE with same hyperparameters as Mamba SAE
3. Compute MMCS (max mean cosine similarity) between the two SAE dictionaries
4. Analyze feature overlap and differences

Usage:
  CUDA_VISIBLE_DEVICES=5 python compare_sae.py \
    --mamba-sae /nvmessd/lifanhong/LR-env/exp008_sae/sae_final.pt \
    --cache-dir /nvmessd/lifanhong/LR-env/cache/hf \
    --out-dir /nvmessd/lifanhong/LR-env/exp008_phase2
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PYTHIA_MODEL = "EleutherAI/pythia-160m-deduped"
SEQ_LEN = 256
BATCH_SIZE = 16


class TopKSAE(nn.Module):
    def __init__(self, d_model, d_sae, k):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k
        self.encoder = nn.Linear(d_model, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_model, bias=True)
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.kaiming_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x):
        pre_acts = F.relu(self.encoder(x))
        topk_vals, topk_idx = pre_acts.topk(self.k, dim=-1)
        sparse_acts = torch.zeros_like(pre_acts)
        sparse_acts.scatter_(1, topk_idx, topk_vals)
        return sparse_acts, pre_acts

    def forward(self, x):
        sparse_acts, pre_acts = self.encode(x)
        x_hat = self.decoder(sparse_acts)
        mse = F.mse_loss(x_hat, x)
        l1 = pre_acts.abs().mean()
        return {"x_hat": x_hat, "sparse_acts": sparse_acts, "mse": mse, "l1": l1}


def with_retry(fn, label, attempts=5, sleep_s=20):
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            print(f"[{label}] attempt {attempt}/{attempts}: {e}", flush=True)
            if attempt < attempts:
                time.sleep(sleep_s)
    raise RuntimeError(f"{label} failed after {attempts} attempts")


def cache_pythia_activations(cache_dir, device, out_dir):
    """Cache Pythia-160M middle layer activations."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print("Loading Pythia-160M...", flush=True)
    tokenizer = with_retry(
        lambda: AutoTokenizer.from_pretrained(PYTHIA_MODEL, cache_dir=cache_dir), "tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = with_retry(
        lambda: AutoModelForCausalLM.from_pretrained(
            PYTHIA_MODEL, cache_dir=cache_dir, torch_dtype=torch.float16,
            output_hidden_states=True), "model")
    model = model.to(device).eval()

    n_layers = model.config.num_hidden_layers
    target_layer = n_layers // 2
    d_model = model.config.hidden_size
    print(f"  {n_layers} layers, d_model={d_model}, hooking layer {target_layer}", flush=True)

    ds = with_retry(
        lambda: load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation",
                             cache_dir=str(Path(cache_dir) / "datasets")), "dataset")
    texts = [t for t in ds["text"] if len(t.strip()) > 50]
    print(f"  {len(texts)} texts", flush=True)

    all_acts = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Caching Pythia"):
        batch = texts[i:i + BATCH_SIZE]
        tokens = tokenizer(batch, return_tensors="pt", max_length=SEQ_LEN,
                           truncation=True, padding="max_length").to(device)
        with torch.no_grad():
            out = model(**tokens)
        h = out.hidden_states[target_layer + 1].cpu().float()  # skip embedding
        mask = tokens["attention_mask"].cpu().bool()
        for b in range(h.shape[0]):
            all_acts.append(h[b][mask[b]])

    acts = torch.cat(all_acts, dim=0)
    act_path = out_dir / "pythia_acts.pt"
    torch.save(acts, act_path)
    print(f"  Cached: {acts.shape} → {act_path}", flush=True)

    meta = {"model": PYTHIA_MODEL, "layer": target_layer, "n_layers": n_layers,
            "d_model": d_model, "n_tokens": acts.shape[0]}
    with open(out_dir / "pythia_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return acts, d_model


def train_sae(acts, d_model, d_sae, k, device, out_dir, n_steps=30000, lr=3e-4, l1_coeff=1e-3):
    """Train TopK SAE on activations."""
    acts_mean = acts.mean(dim=0)
    acts_std = acts.std(dim=0) + 1e-8
    acts_norm = (acts - acts_mean) / acts_std

    sae = TopKSAE(d_model, d_sae, k).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    dataloader = DataLoader(TensorDataset(acts_norm), batch_size=4096,
                            shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    print(f"Training Pythia SAE: d={d_model}→{d_sae}, K={k}, {n_steps} steps", flush=True)
    step = 0
    while step < n_steps:
        for (batch,) in dataloader:
            if step >= n_steps:
                break
            batch = batch.to(device)
            out = sae(batch)
            loss = out["mse"] + l1_coeff * out["l1"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step % 5000 == 0:
                print(f"  Step {step}: MSE={out['mse'].item():.6f}, L1={out['l1'].item():.4f}", flush=True)

    ckpt = {"state_dict": sae.state_dict(), "config": {"d_model": d_model, "d_sae": d_sae, "k": k},
            "acts_mean": acts_mean, "acts_std": acts_std}
    torch.save(ckpt, out_dir / "pythia_sae_final.pt")
    print(f"  Saved: {out_dir / 'pythia_sae_final.pt'}", flush=True)
    return sae


def compute_mmcs(W_a, W_b):
    """Max Mean Cosine Similarity between two SAE decoder matrices.
    W_a: (d_sae_a, d_model), W_b: (d_sae_b, d_model)
    Returns: mean of max cosine similarity for each feature in A to best match in B, and vice versa.
    """
    W_a_norm = F.normalize(W_a, dim=1)
    W_b_norm = F.normalize(W_b, dim=1)
    cos_sim = W_a_norm @ W_b_norm.T  # (d_sae_a, d_sae_b)

    max_a_to_b = cos_sim.max(dim=1).values  # best match in B for each A feature
    max_b_to_a = cos_sim.max(dim=0).values  # best match in A for each B feature

    return {
        "mmcs_a_to_b": float(max_a_to_b.mean()),
        "mmcs_b_to_a": float(max_b_to_a.mean()),
        "mmcs_symmetric": float((max_a_to_b.mean() + max_b_to_a.mean()) / 2),
        "max_a_to_b_median": float(max_a_to_b.median()),
        "max_b_to_a_median": float(max_b_to_a.median()),
        "high_overlap_count": int((max_a_to_b > 0.9).sum()),
        "high_overlap_pct": float((max_a_to_b > 0.9).float().mean() * 100),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mamba-sae", required=True, help="Path to Mamba SAE checkpoint")
    parser.add_argument("--cache-dir", default="/nvmessd/lifanhong/LR-env/cache/hf")
    parser.add_argument("--out-dir", default="/nvmessd/lifanhong/LR-env/exp008_phase2")
    parser.add_argument("--d-sae", type=int, default=4096)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--n-steps", type=int, default=30000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Cache Pythia activations
    pythia_acts_path = out_dir / "pythia_acts.pt"
    if pythia_acts_path.exists():
        print("Loading cached Pythia activations...", flush=True)
        acts = torch.load(pythia_acts_path, map_location="cpu", weights_only=True)
        with open(out_dir / "pythia_meta.json") as f:
            meta = json.load(f)
        d_model = meta["d_model"]
    else:
        acts, d_model = cache_pythia_activations(args.cache_dir, args.device, out_dir)

    # Step 2: Train Pythia SAE
    pythia_sae_path = out_dir / "pythia_sae_final.pt"
    if pythia_sae_path.exists():
        print("Loading cached Pythia SAE...", flush=True)
        pythia_ckpt = torch.load(pythia_sae_path, map_location="cpu", weights_only=True)
        pythia_sae = TopKSAE(d_model, args.d_sae, args.k)
        pythia_sae.load_state_dict(pythia_ckpt["state_dict"])
    else:
        pythia_sae = train_sae(acts, d_model, args.d_sae, args.k, args.device, out_dir,
                               n_steps=args.n_steps)

    # Step 3: Load Mamba SAE
    print("Loading Mamba SAE...", flush=True)
    mamba_ckpt = torch.load(args.mamba_sae, map_location="cpu", weights_only=True)
    mamba_config = mamba_ckpt["config"]
    mamba_sae = TopKSAE(mamba_config["d_model"], mamba_config["d_sae"], mamba_config["k"])
    mamba_sae.load_state_dict(mamba_ckpt["state_dict"])
    print(f"  Mamba SAE: d={mamba_config['d_model']}→{mamba_config['d_sae']}, K={mamba_config['k']}")

    # Step 4: MMCS comparison
    print("\nComputing MMCS...", flush=True)
    # Use decoder weights as feature directions
    W_pythia = pythia_sae.decoder.weight.data.T  # (d_sae, d_model)
    W_mamba = mamba_sae.decoder.weight.data.T     # (d_sae, d_model)

    # Both have d_model=768, so direct comparison is possible
    if W_pythia.shape[1] != W_mamba.shape[1]:
        print(f"WARNING: d_model mismatch: Pythia={W_pythia.shape[1]}, Mamba={W_mamba.shape[1]}")
        print("Cannot compute MMCS directly.")
        mmcs = {"error": "d_model mismatch"}
    else:
        mmcs = compute_mmcs(W_pythia, W_mamba)
        print(f"  MMCS (Pythia→Mamba): {mmcs['mmcs_a_to_b']:.4f}")
        print(f"  MMCS (Mamba→Pythia): {mmcs['mmcs_b_to_a']:.4f}")
        print(f"  MMCS (symmetric):    {mmcs['mmcs_symmetric']:.4f}")
        print(f"  High overlap (>0.9): {mmcs['high_overlap_count']}/{args.d_sae} ({mmcs['high_overlap_pct']:.1f}%)")

    # Step 5: Figures
    if "error" not in mmcs:
        W_pn = F.normalize(W_pythia, dim=1)
        W_mn = F.normalize(W_mamba, dim=1)
        cos_sim = (W_pn @ W_mn.T).numpy()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Cosine similarity heatmap (subsample for visibility)
        step = max(1, args.d_sae // 200)
        axes[0].imshow(cos_sim[::step, ::step], aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        axes[0].set_xlabel("Mamba features")
        axes[0].set_ylabel("Pythia features")
        axes[0].set_title("Feature Cosine Similarity")

        # Max cosine similarity histograms
        max_p2m = cos_sim.max(axis=1)
        max_m2p = cos_sim.max(axis=0)
        axes[1].hist(max_p2m, bins=50, alpha=0.7, label="Pythia→Mamba", color="blue")
        axes[1].hist(max_m2p, bins=50, alpha=0.7, label="Mamba→Pythia", color="red")
        axes[1].set_xlabel("Max Cosine Similarity")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"Feature Matching (MMCS={mmcs['mmcs_symmetric']:.3f})")
        axes[1].legend()
        axes[1].axvline(0.9, color="black", linestyle="--", alpha=0.5, label="threshold=0.9")

        # Sorted max similarity
        axes[2].plot(np.sort(max_p2m)[::-1], label="Pythia→Mamba", color="blue")
        axes[2].plot(np.sort(max_m2p)[::-1], label="Mamba→Pythia", color="red")
        axes[2].set_xlabel("Feature rank")
        axes[2].set_ylabel("Best match cosine similarity")
        axes[2].set_title("Sorted Feature Match Quality")
        axes[2].legend()
        axes[2].axhline(0.9, color="black", linestyle="--", alpha=0.3)
        axes[2].grid(True, alpha=0.3)

        fig.suptitle("Exp-008 Phase 2: Mamba vs Pythia SAE Feature Comparison", fontsize=14)
        fig.tight_layout()
        fig.savefig(out_dir / "sae_comparison.png", dpi=150)
        print(f"\nFigure: {out_dir / 'sae_comparison.png'}")

    # Save results
    results = {
        "pythia_model": PYTHIA_MODEL,
        "mamba_sae_path": str(args.mamba_sae),
        "d_sae": args.d_sae,
        "k": args.k,
        "mmcs": mmcs,
    }
    with open(out_dir / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results: {out_dir / 'comparison_results.json'}")

    print(f"\n{'='*60}")
    print("PHASE 2 SUMMARY")
    print(f"{'='*60}")
    if "error" not in mmcs:
        print(f"  MMCS symmetric: {mmcs['mmcs_symmetric']:.4f}")
        print(f"  High overlap features (>0.9): {mmcs['high_overlap_pct']:.1f}%")
        if mmcs['mmcs_symmetric'] > 0.5:
            print("  → Substantial feature overlap: Mamba and Transformer share many concepts")
        elif mmcs['mmcs_symmetric'] > 0.3:
            print("  → Moderate overlap: some shared, some architecture-specific features")
        else:
            print("  → Low overlap: largely different feature sets")


if __name__ == "__main__":
    main()
