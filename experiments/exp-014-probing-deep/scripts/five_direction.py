#!/usr/bin/env python3
"""
Exp-014 Phase 3: Five-direction head-to-head comparison.

For each concept × layer, extract directions via:
  1. Linear probe weight
  2. Difference-in-means
  3. Loss gradient (mean gradient direction)
Compare pairwise cosine similarities and ablation effects.

Usage:
  CUDA_VISIBLE_DEVICES=5 python five_direction.py \
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
    "plural": lambda strs: np.array([1 if len(t.strip())>2 and t.strip().lower().endswith(("s","es","ies")) else 0 for t in strs]),
}

def with_retry(fn, label, attempts=5, sleep_s=20):
    for a in range(1, attempts+1):
        try: return fn()
        except Exception as e:
            print(f"[{label}] {a}/{attempts}: {e}", flush=True)
            if a < attempts: import time; time.sleep(sleep_s)
    raise RuntimeError(f"{label} failed")

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

    ds = with_retry(lambda: load_dataset("Salesforce/wikitext","wikitext-103-raw-v1",split="validation",
                    cache_dir=str(Path(args.cache_dir)/"datasets")), "data")
    texts = [t for t in ds["text"] if len(t.strip())>30][:N_SENTENCES]

    # Load model with gradient tracking
    model = with_retry(lambda: AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir=args.cache_dir, torch_dtype=torch.float32, output_hidden_states=True), "model")
    model = model.to(args.device).eval()
    n_layers = model.config.n_layer; d_model = model.config.n_embd

    # Cache hidden states + gradients
    print("Caching hidden states + gradients...", flush=True)
    layer_h = {l: [] for l in range(n_layers)}
    layer_g = {l: [] for l in range(n_layers)}
    all_strs = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Cache"):
        batch = texts[i:i+BATCH_SIZE]
        tokens = tokenizer(batch, return_tensors="pt", max_length=SEQ_LEN,
                           truncation=True, padding="max_length").to(args.device)
        model.zero_grad()
        out = model(**tokens, output_hidden_states=True, labels=tokens["input_ids"])
        hidden_list = [out.hidden_states[l+1] for l in range(n_layers)]
        grads = torch.autograd.grad(out.loss, hidden_list, retain_graph=False, allow_unused=True)
        mask = tokens["attention_mask"].cpu().bool()

        for l in range(n_layers):
            h = hidden_list[l].detach().cpu().float()
            g = grads[l].detach().cpu().float() if grads[l] is not None else torch.zeros_like(h.cpu())
            for b in range(h.shape[0]):
                layer_h[l].append(h[b][mask[b]].numpy())
                layer_g[l].append(g[b][mask[b]].numpy())

        for b in range(tokens["input_ids"].shape[0]):
            valid = mask[b]
            all_strs.extend([tokenizer.decode([tid]) for tid in tokens["input_ids"][b][valid].cpu()])

    for l in range(n_layers):
        layer_h[l] = np.concatenate(layer_h[l], axis=0)
        layer_g[l] = np.concatenate(layer_g[l], axis=0)

    # Concept labels
    concept_labels = {k: v(all_strs) for k, v in CONCEPTS.items()}
    concept_labels = {k: v for k, v in concept_labels.items() if 0.05 < v.mean() < 0.95}
    print(f"Concepts: {list(concept_labels.keys())}", flush=True)

    # For each concept × layer: extract 3 directions
    print("\nExtracting directions...", flush=True)
    results = {}

    for cname, labels in concept_labels.items():
        results[cname] = {"layers": {}}
        y = torch.tensor(labels, dtype=torch.long)

        for l in tqdm(range(n_layers), desc=f"  {cname}"):
            X = layer_h[l]; G = layer_g[l]
            pos, neg = labels==1, labels==0

            # 1. Probe direction
            probe = nn.Linear(d_model, 1).to(args.device)
            opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
            Xd = torch.tensor(X, dtype=torch.float32, device=args.device)
            yd = y.float().to(args.device)
            probe.train()
            for _ in range(10):
                perm = torch.randperm(len(X), device=args.device)
                for j in range(0, len(X), 4096):
                    idx = perm[j:j+4096]
                    logits = probe(Xd[idx]).squeeze()
                    loss = nn.BCEWithLogitsLoss()(logits, yd[idx])
                    opt.zero_grad(); loss.backward(); opt.step()
            w_probe = probe.linear.weight.data.cpu().numpy().flatten()
            w_probe = w_probe / (np.linalg.norm(w_probe) + 1e-10)

            # 2. DiffMeans direction
            if pos.sum() > 10 and neg.sum() > 10:
                w_diff = X[pos].mean(0) - X[neg].mean(0)
                w_diff = w_diff / (np.linalg.norm(w_diff) + 1e-10)
            else:
                w_diff = np.zeros(d_model)

            # 3. Gradient direction
            w_grad = G.mean(0)
            w_grad = w_grad / (np.linalg.norm(w_grad) + 1e-10)

            # Pairwise cosines
            cos_probe_diff = float(np.dot(w_probe, w_diff))
            cos_probe_grad = float(np.dot(w_probe, w_grad))
            cos_diff_grad = float(np.dot(w_diff, w_grad))

            results[cname]["layers"][l] = {
                "cos_probe_diff": cos_probe_diff,
                "cos_probe_grad": cos_probe_grad,
                "cos_diff_grad": cos_diff_grad,
            }

    # Summary
    print(f"\n{'='*60}\nSUMMARY: Mean |cosine| across layers\n{'='*60}")
    for cname in concept_labels:
        pd = np.mean([abs(results[cname]["layers"][l]["cos_probe_diff"]) for l in range(n_layers)])
        pg = np.mean([abs(results[cname]["layers"][l]["cos_probe_grad"]) for l in range(n_layers)])
        dg = np.mean([abs(results[cname]["layers"][l]["cos_diff_grad"]) for l in range(n_layers)])
        print(f"  {cname}: probe↔diff={pd:.4f}, probe↔grad={pg:.4f}, diff↔grad={dg:.4f}")
        results[cname]["summary"] = {"probe_diff": pd, "probe_grad": pg, "diff_grad": dg}

    # Figure
    fig, axes = plt.subplots(len(concept_labels), 3, figsize=(15, 4*len(concept_labels)), squeeze=False)
    pairs = [("cos_probe_diff", "Probe ↔ DiffMeans"),
             ("cos_probe_grad", "Probe ↔ Gradient"),
             ("cos_diff_grad", "DiffMeans ↔ Gradient")]

    for ci, cname in enumerate(concept_labels):
        for pi, (key, label) in enumerate(pairs):
            vals = [results[cname]["layers"][l][key] for l in range(n_layers)]
            axes[ci, pi].plot(range(n_layers), vals, "o-", markersize=4)
            axes[ci, pi].axhline(0, color="gray", linestyle="--", alpha=0.5)
            axes[ci, pi].set_xlabel("Layer"); axes[ci, pi].set_ylabel("Cosine")
            axes[ci, pi].set_title(f"{cname}: {label}")
            axes[ci, pi].set_ylim(-1, 1); axes[ci, pi].grid(True, alpha=0.3)

    fig.suptitle("Exp-014 Phase 3: Direction Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "five_direction.png", dpi=150)
    print(f"\nFigure: {out_dir / 'five_direction.png'}")

    with open(out_dir / "five_direction.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results: {out_dir / 'five_direction.json'}")

if __name__ == "__main__":
    main()
