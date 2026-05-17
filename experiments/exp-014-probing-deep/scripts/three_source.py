#!/usr/bin/env python3
"""
Exp-014 Phase 4: Three-source decomposition.

Decomposes probe signal into:
  1. Input bleed-through (probe on embedding layer)
  2. Architecture contribution (probe on random network)
  3. Training contribution (trained - random - input)

Usage:
  CUDA_VISIBLE_DEVICES=1 python three_source.py \
    --cache-dir /nvmessd/lifanhong/LR-env/cache/hf \
    --out-dir /nvmessd/lifanhong/LR-env/exp014
"""

import argparse, json, os, gc, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from tqdm import tqdm
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

MODEL_ID = "EleutherAI/pythia-1.4b-deduped"
N_SENTENCES = 1000; SEQ_LEN = 64; BATCH_SIZE = 16

CONCEPTS = {
    "capitalized": lambda strs: np.array([1 if t.lstrip() and t.lstrip()[0].isupper() else 0 for t in strs]),
    "numeric": lambda strs: np.array([1 if any(c.isdigit() for c in t) else 0 for t in strs]),
    "plural": lambda strs: np.array([1 if len(t.strip())>2 and t.strip().lower().endswith(("s","es","ies")) else 0 for t in strs]),
    "short": lambda strs: np.array([1 if len(t.strip()) <= 2 else 0 for t in strs]),
    "subword": lambda strs: np.array([1 if not (t.startswith(" ") or t.startswith("Ġ")) else 0 for t in strs]),
}


def with_retry(fn, label, attempts=5, sleep_s=20):
    for a in range(1, attempts+1):
        try: return fn()
        except Exception as e:
            print(f"[{label}] {a}/{attempts}: {e}", flush=True)
            if a < attempts: time.sleep(sleep_s)
    raise RuntimeError(f"{label} failed")


def cache_hiddens(model, tokenizer, texts, device, n_layers):
    layer_h = {l: [] for l in range(-1, n_layers)}  # -1 = embedding
    all_strs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Cache"):
        batch = texts[i:i+BATCH_SIZE]
        tokens = tokenizer(batch, return_tensors="pt", max_length=SEQ_LEN,
                           truncation=True, padding="max_length").to(device)
        with torch.no_grad():
            out = model(**tokens, output_hidden_states=True)
        mask = tokens["attention_mask"].bool()
        # Embedding = hidden_states[0]
        for l in range(-1, n_layers):
            h = out.hidden_states[l+1].cpu().float()
            for b in range(h.shape[0]):
                layer_h[l].append(h[b][mask[b]].numpy())
        for b in range(tokens["input_ids"].shape[0]):
            valid = mask[b]
            ids = tokens["input_ids"][b][valid].cpu()
            all_strs.extend([tokenizer.decode([tid]) for tid in ids])
    for l in range(-1, n_layers):
        layer_h[l] = np.concatenate(layer_h[l], axis=0)
    return layer_h, all_strs


def probe_accuracy(X, y, device, epochs=10):
    d = X.shape[1]
    probe = nn.Linear(d, 1).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    Xd = torch.tensor(X, dtype=torch.float32, device=device)
    yd = torch.tensor(y, dtype=torch.float32, device=device)
    probe.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X), device=device)
        for j in range(0, len(X), 4096):
            idx = perm[j:j+4096]
            logits = probe(Xd[idx]).squeeze()
            loss = nn.BCEWithLogitsLoss()(logits, yd[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    with torch.no_grad():
        acc = ((probe(Xd).squeeze() > 0).long() == yd.long()).float().mean().item()
    return acc


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

    # 1. Trained model
    print("=== Trained model ===", flush=True)
    model = with_retry(lambda: AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir=args.cache_dir, torch_dtype=torch.float16, output_hidden_states=True), "model")
    model = model.to(args.device).eval()
    n_layers = model.config.num_hidden_layers
    trained_h, all_strs = cache_hiddens(model, tokenizer, texts, args.device, n_layers)
    del model; torch.cuda.empty_cache(); gc.collect()

    # 2. Random model (same architecture, random weights)
    print("\n=== Random model ===", flush=True)
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(MODEL_ID, cache_dir=args.cache_dir)
    model_rand = AutoModelForCausalLM.from_config(config).half().to(args.device).eval()
    random_h, _ = cache_hiddens(model_rand, tokenizer, texts, args.device, n_layers)
    del model_rand; torch.cuda.empty_cache(); gc.collect()

    # Compute concept labels
    concept_labels = {k: v(all_strs) for k, v in CONCEPTS.items()}
    concept_labels = {k: v for k, v in concept_labels.items() if 0.02 < v.mean() < 0.98}

    # 3. Probe at every layer for trained, random, and embedding
    print("\n=== Probing ===", flush=True)
    results = {}

    for cname, labels in concept_labels.items():
        print(f"\n  {cname} (pos_rate={labels.mean():.3f}):", flush=True)
        emb_acc = probe_accuracy(trained_h[-1], labels, args.device)
        results[cname] = {"embedding_acc": emb_acc, "trained": {}, "random": {}}

        for l in tqdm(range(n_layers), desc=f"    {cname}"):
            t_acc = probe_accuracy(trained_h[l], labels, args.device)
            r_acc = probe_accuracy(random_h[l], labels, args.device)
            results[cname]["trained"][l] = t_acc
            results[cname]["random"][l] = r_acc

    # Plot stacked area
    fig, axes = plt.subplots(len(concept_labels), 1, figsize=(12, 4*len(concept_labels)), squeeze=False)
    for ci, (cname, r) in enumerate(results.items()):
        ax = axes[ci, 0]
        layers = list(range(n_layers))
        emb = r["embedding_acc"]
        random_accs = [r["random"][l] for l in layers]
        trained_accs = [r["trained"][l] for l in layers]

        # Decompose: input = emb, arch = random - emb, training = trained - random
        input_contrib = [emb] * n_layers
        arch_contrib = [max(0, ra - emb) for ra in random_accs]
        train_contrib = [max(0, ta - ra) for ta, ra in zip(trained_accs, random_accs)]

        ax.fill_between(layers, 0, input_contrib, alpha=0.7, label="Input bleed-through", color="blue")
        ax.fill_between(layers, input_contrib,
                        [i+a for i,a in zip(input_contrib, arch_contrib)],
                        alpha=0.7, label="Architecture", color="orange")
        ax.fill_between(layers, [i+a for i,a in zip(input_contrib, arch_contrib)],
                        [i+a+t for i,a,t in zip(input_contrib, arch_contrib, train_contrib)],
                        alpha=0.7, label="Training", color="green")
        ax.plot(layers, trained_accs, "k-", linewidth=2, label="Total (trained)")
        ax.plot(layers, random_accs, "k--", linewidth=1, label="Random network")
        ax.axhline(emb, color="blue", linestyle=":", alpha=0.5)
        ax.set_xlabel("Layer"); ax.set_ylabel("Probe Accuracy")
        ax.set_title(f"{cname} — Source Decomposition")
        ax.legend(fontsize=7); ax.set_ylim(0.4, 1.05)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Exp-014 Phase 4: Three-Source Probe Signal Decomposition", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "three_source.png", dpi=150)
    print(f"\nFigure: {out_dir / 'three_source.png'}")

    with open(out_dir / "three_source.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for cname, r in results.items():
        emb = r["embedding_acc"]
        mid = n_layers // 2
        rand_mid = r["random"][mid]
        train_mid = r["trained"][mid]
        print(f"  {cname}: emb={emb:.3f}, random_mid={rand_mid:.3f}, trained_mid={train_mid:.3f}")
        print(f"    Input: {emb:.1%}, Arch: {max(0,rand_mid-emb):.1%}, Train: {max(0,train_mid-rand_mid):.1%}")

if __name__ == "__main__":
    main()
