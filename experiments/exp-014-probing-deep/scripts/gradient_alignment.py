#!/usr/bin/env python3
"""
Exp-014 Phase 1: Gradient alignment test.

For each concept × layer: cos(probe_direction, loss_gradient).
Tests if probe directions align with gradients — a free causal proxy.

Usage:
  CUDA_VISIBLE_DEVICES=5 python gradient_alignment.py \
    --cache-dir /nvmessd/lifanhong/LR-env/cache/hf \
    --out-dir /nvmessd/lifanhong/LR-env/exp014
"""

import argparse, json, os, gc
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from tqdm import tqdm
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

MODEL_ID = "openai-community/gpt2"
N_SENTENCES = 500; SEQ_LEN = 64; BATCH_SIZE = 16

CONCEPTS = {
    "capitalized": lambda strs: np.array([1 if t.lstrip() and t.lstrip()[0].isupper() else 0 for t in strs]),
    "numeric": lambda strs: np.array([1 if any(c.isdigit() for c in t) else 0 for t in strs]),
    "punctuation": lambda strs: np.array([1 if t.strip() and all(not c.isalnum() for c in t.strip()) else 0 for t in strs]),
    "short": lambda strs: np.array([1 if len(t.strip()) <= 2 else 0 for t in strs]),
    "plural": lambda strs: np.array([1 if len(t.strip())>2 and t.strip().lower().endswith(("s","es","ies")) else 0 for t in strs]),
}

def with_retry(fn, label, attempts=5, sleep_s=20):
    for a in range(1, attempts+1):
        try: return fn()
        except Exception as e:
            print(f"[{label}] {a}/{attempts}: {e}", flush=True)
            if a < attempts: import time; time.sleep(sleep_s)
    raise RuntimeError(f"{label} failed")

class LinearProbe(nn.Module):
    def __init__(self, d): super().__init__(); self.linear = nn.Linear(d, 1)
    def forward(self, x): return self.linear(x).squeeze(-1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="/nvmessd/lifanhong/LR-env/cache/hf")
    parser.add_argument("--out-dir", default="/nvmessd/lifanhong/LR-env/exp014")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tokenizer = with_retry(lambda: AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=args.cache_dir), "tok")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = with_retry(lambda: AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir=args.cache_dir, torch_dtype=torch.float32), "model")
    model = model.to(args.device).eval()
    n_layers = model.config.n_layer; d_model = model.config.n_embd

    ds = with_retry(lambda: load_dataset("Salesforce/wikitext","wikitext-103-raw-v1",split="validation",
                    cache_dir=str(Path(args.cache_dir)/"datasets")), "data")
    texts = [t for t in ds["text"] if len(t.strip())>30][:N_SENTENCES]

    # Cache hidden states + compute gradients
    print("Caching hidden states + gradients...", flush=True)
    layer_hiddens = {l:[] for l in range(n_layers)}
    layer_grads = {l:[] for l in range(n_layers)}
    all_strs = []

    for i in tqdm(range(0,len(texts),BATCH_SIZE), desc="Cache"):
        batch = texts[i:i+BATCH_SIZE]
        tokens = tokenizer(batch, return_tensors="pt", max_length=SEQ_LEN,
                           truncation=True, padding="max_length").to(args.device)
        # Forward with gradient tracking on hidden states
        model.zero_grad()
        out = model(**tokens, output_hidden_states=True, labels=tokens["input_ids"])
        loss = out.loss
        mask = tokens["attention_mask"].bool()

        # Get gradients
        hidden_list = [out.hidden_states[l+1] for l in range(n_layers)]
        grads = torch.autograd.grad(loss, hidden_list, retain_graph=False, allow_unused=True)

        for l in range(n_layers):
            h = hidden_list[l].detach().cpu().float()
            g = grads[l].detach().cpu().float() if grads[l] is not None else torch.zeros_like(h.cpu())
            for b in range(h.shape[0]):
                valid = mask[b].cpu()
                layer_hiddens[l].append(h[b][valid].numpy())
                layer_grads[l].append(g[b][valid].numpy())

        for b in range(tokens["input_ids"].shape[0]):
            valid = mask[b].cpu()
            ids = tokens["input_ids"][b][valid].cpu()
            all_strs.extend([tokenizer.decode([tid]) for tid in ids])

    for l in range(n_layers):
        layer_hiddens[l] = np.concatenate(layer_hiddens[l], axis=0)
        layer_grads[l] = np.concatenate(layer_grads[l], axis=0)

    # Compute concept labels
    concept_labels = {}
    for cname, labeler in CONCEPTS.items():
        labels = labeler(all_strs)
        if 0.02 < labels.mean() < 0.98:
            concept_labels[cname] = labels
            print(f"  {cname}: {labels.mean():.3f}")

    # For each concept × layer: train probe, compute cos(w, mean_gradient)
    print("\nComputing gradient alignment...", flush=True)
    results = {}

    for cname, labels in concept_labels.items():
        results[cname] = {}
        y = torch.tensor(labels, dtype=torch.long)

        for l in tqdm(range(n_layers), desc=f"  {cname}"):
            X = torch.tensor(layer_hiddens[l], dtype=torch.float32)
            # Train probe
            probe = LinearProbe(d_model).to(args.device)
            opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
            Xd, yd = X.to(args.device), y.float().to(args.device)
            probe.train()
            for _ in range(10):
                perm = torch.randperm(len(X), device=args.device)
                for j in range(0, len(X), 4096):
                    idx = perm[j:j+4096]
                    logits = probe(Xd[idx])
                    loss = nn.BCEWithLogitsLoss()(logits, yd[idx])
                    opt.zero_grad(); loss.backward(); opt.step()
            probe.eval()
            with torch.no_grad():
                acc = ((probe(Xd)>0).long() == y.to(args.device)).float().mean().item()

            w = probe.linear.weight.data.cpu().numpy().flatten()
            w_norm = w / (np.linalg.norm(w) + 1e-10)

            # Mean gradient direction
            g = layer_grads[l].mean(axis=0)
            g_norm = g / (np.linalg.norm(g) + 1e-10)

            cos_wg = float(np.dot(w_norm, g_norm))

            results[cname][l] = {"probe_acc": acc, "cos_w_g": cos_wg}

    # Summary
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for cname in concept_labels:
        accs = [results[cname][l]["probe_acc"] for l in range(n_layers)]
        coss = [results[cname][l]["cos_w_g"] for l in range(n_layers)]
        print(f"  {cname}: best_acc={max(accs):.3f}, mean_|cos(w,g)|={np.mean(np.abs(coss)):.4f}")

    # Figure
    fig, axes = plt.subplots(len(concept_labels), 2, figsize=(14, 3.5*len(concept_labels)), squeeze=False)
    for ci, cname in enumerate(concept_labels):
        layers = list(range(n_layers))
        accs = [results[cname][l]["probe_acc"] for l in layers]
        coss = [results[cname][l]["cos_w_g"] for l in layers]
        axes[ci,0].plot(layers, accs, "o-", markersize=4)
        axes[ci,0].set_ylabel("Probe Acc"); axes[ci,0].set_title(f"{cname} — Probe")
        axes[ci,1].plot(layers, coss, "o-", markersize=4, color="red")
        axes[ci,1].set_ylabel("cos(w, grad)"); axes[ci,1].set_title(f"{cname} — Gradient Alignment")
        axes[ci,1].axhline(0, color="gray", linestyle="--", alpha=0.5)
    fig.suptitle("Exp-014: Probe Direction vs Gradient Direction", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "gradient_alignment.png", dpi=150)
    print(f"\nFigure: {out_dir / 'gradient_alignment.png'}")

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results: {out_dir / 'results.json'}")

if __name__ == "__main__":
    main()
