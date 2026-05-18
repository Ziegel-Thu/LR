#!/usr/bin/env python3
"""
Training dynamics: extract concept directions + ID profiles across Pythia-1.4B checkpoints.

Combines exp-009 Phase 4 (concept direction emergence) and exp-012 Phase 3 (hunchback formation).
Each checkpoint runs on a separate GPU for max parallelism.

Usage (single checkpoint):
  CUDA_VISIBLE_DEVICES=0 python training_dynamics.py --step step1000 \
    --cache-dir /nvmessd/lifanhong/LR-env/cache/hf \
    --out-dir /nvmessd/lifanhong/LR-env/training_dynamics

  # After all checkpoints done:
  python training_dynamics.py --analyze --out-dir /nvmessd/lifanhong/LR-env/training_dynamics
"""

import argparse, json, os, gc
from pathlib import Path
import numpy as np, torch
from tqdm import tqdm
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

MODEL_ID = "EleutherAI/pythia-1.4b-deduped"
N_SENTENCES = 1000; SEQ_LEN = 128; BATCH_SIZE = 8

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
            if a < attempts: import time; time.sleep(sleep_s)
    raise RuntimeError(f"{label} failed")


def twonn_id(X):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=3, metric="euclidean").fit(X)
    d, _ = nn.kneighbors(X)
    r1, r2 = d[:,1], d[:,2]
    mask = r1 > 1e-10; mu = r2[mask]/r1[mask]
    s = np.sum(np.log(mu))
    return float(len(mu)/s) if s > 1e-10 else 0


def stable_rank(X):
    sv = np.linalg.svd(X - X.mean(0), compute_uv=False)
    s2 = sv**2; return float(s2.sum()/(s2[0]+1e-10))


def diff_in_means(hidden, labels):
    pos, neg = labels==1, labels==0
    if pos.sum()<10 or neg.sum()<10: return None
    d = hidden[pos].mean(0) - hidden[neg].mean(0)
    n = np.linalg.norm(d)
    return d/n if n>1e-10 else None


def process_checkpoint(step, cache_dir, out_dir, device):
    """Process one checkpoint: extract concept directions + ID profile."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{step}.json"
    if out_path.exists():
        print(f"SKIP {step}: already done", flush=True)
        return

    revision = step if step != "main" else None
    print(f"\n{'='*60}\nCheckpoint: {step}\n{'='*60}", flush=True)

    tokenizer = with_retry(lambda: AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=cache_dir, revision=revision), "tok")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = with_retry(lambda: AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir=cache_dir, revision=revision,
        torch_dtype=torch.float16, output_hidden_states=True), "model")
    model = model.to(device).eval()
    n_layers = model.config.num_hidden_layers

    ds = with_retry(lambda: load_dataset("Salesforce/wikitext","wikitext-103-raw-v1",
        split="validation", cache_dir=str(Path(cache_dir)/"datasets")), "data")
    texts = [t for t in ds["text"] if len(t.strip())>50][:N_SENTENCES]

    # Cache hidden states
    layer_h = {l: [] for l in range(n_layers)}
    all_strs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"Cache {step}"):
        batch = texts[i:i+BATCH_SIZE]
        tokens = tokenizer(batch, return_tensors="pt", max_length=SEQ_LEN,
                           truncation=True, padding="max_length").to(device)
        with torch.no_grad():
            out = model(**tokens)
        mask = tokens["attention_mask"].cpu().bool()
        for l in range(n_layers):
            h = out.hidden_states[l+1].cpu().float()
            for b in range(h.shape[0]):
                layer_h[l].append(h[b][mask[b]].numpy())
        for b in range(tokens["input_ids"].shape[0]):
            valid = mask[b]
            ids = tokens["input_ids"][b][valid].cpu()
            all_strs.extend([tokenizer.decode([tid]) for tid in ids])

    for l in range(n_layers):
        layer_h[l] = np.concatenate(layer_h[l], axis=0)

    # Compute loss
    losses = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        tokens = tokenizer(batch, return_tensors="pt", max_length=SEQ_LEN,
                           truncation=True, padding="max_length").to(device)
        with torch.no_grad():
            out = model(**tokens, labels=tokens["input_ids"])
        losses.append(out.loss.item())
    val_loss = float(np.mean(losses))

    # Concept labels
    concept_labels = {k: v(all_strs) for k, v in CONCEPTS.items()}
    concept_labels = {k: v for k, v in concept_labels.items() if 0.02 < v.mean() < 0.98}

    # Extract concept directions + ID profile
    result = {"step": step, "val_loss": val_loss, "n_layers": n_layers,
              "n_tokens": len(all_strs), "concepts": {}, "id_profile": []}

    for l in range(n_layers):
        X = layer_h[l]
        tid = twonn_id(X)
        sr = stable_rank(X)
        result["id_profile"].append({"layer": l, "twonn_id": tid, "stable_rank": sr})

    for cname, labels in concept_labels.items():
        directions = {}
        for l in range(n_layers):
            d = diff_in_means(layer_h[l], labels)
            if d is not None:
                directions[l] = d.tolist()
        result["concepts"][cname] = {"n_directions": len(directions)}
        # Store directions for later comparison
        for l, d in directions.items():
            result["concepts"][cname][f"dir_L{l}"] = d

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path} (loss={val_loss:.4f})", flush=True)

    del model; torch.cuda.empty_cache(); gc.collect()


def analyze(out_dir):
    """Aggregate all checkpoints and generate figures."""
    out_dir = Path(out_dir)

    # Load all checkpoints
    step_order = ["step1000", "step5000", "step10000", "step50000", "step100000", "step143000", "main"]
    step_nums = [1000, 5000, 10000, 50000, 100000, 143000, 143000]
    results = {}
    for step in step_order:
        path = out_dir / f"{step}.json"
        if path.exists():
            with open(path) as f:
                results[step] = json.load(f)
            print(f"Loaded {step}: loss={results[step]['val_loss']:.4f}")

    if len(results) < 2:
        print("Need ≥2 checkpoints"); return

    steps = [s for s in step_order if s in results]
    n_layers = results[steps[0]]["n_layers"]

    # Load final directions for comparison
    final_step = steps[-1]
    final_dirs = {}
    for cname in results[final_step]["concepts"]:
        final_dirs[cname] = {}
        for l in range(n_layers):
            key = f"dir_L{l}"
            if key in results[final_step]["concepts"][cname]:
                final_dirs[cname][l] = np.array(results[final_step]["concepts"][cname][key])

    # Figure 1: ID profile evolution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(steps)))
    for si, step in enumerate(steps):
        depths = [p["layer"]/(n_layers-1) for p in results[step]["id_profile"]]
        ids = [p["twonn_id"] for p in results[step]["id_profile"]]
        srs = [p["stable_rank"] for p in results[step]["id_profile"]]
        axes[0].plot(depths, ids, "-", color=colors[si], label=step, linewidth=1.5)
        axes[1].plot(depths, srs, "-", color=colors[si], label=step, linewidth=1.5)
    axes[0].set_xlabel("Depth"); axes[0].set_ylabel("TwoNN ID")
    axes[0].set_title("ID Profile Evolution"); axes[0].legend(fontsize=7)
    axes[1].set_xlabel("Depth"); axes[1].set_ylabel("Stable Rank")
    axes[1].set_title("Stable Rank Evolution"); axes[1].legend(fontsize=7)
    for ax in axes: ax.grid(True, alpha=0.3)
    fig.suptitle("Exp-012 Phase 3: Representation Geometry During Training", fontsize=14)
    fig.tight_layout(); fig.savefig(out_dir / "id_evolution.png", dpi=150)
    print(f"\nFig 1: {out_dir / 'id_evolution.png'}")

    # Figure 2: Concept direction emergence
    concepts = list(final_dirs.keys())
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    # Cosine similarity with final direction vs training step
    for cname in concepts:
        cos_vs_step = []
        for step in steps:
            sims = []
            for l in range(n_layers):
                key = f"dir_L{l}"
                if key in results[step]["concepts"].get(cname, {}) and l in final_dirs.get(cname, {}):
                    d = np.array(results[step]["concepts"][cname][key])
                    d_final = final_dirs[cname][l]
                    if len(d) == len(d_final):
                        sims.append(np.dot(d, d_final))
            cos_vs_step.append(np.mean(sims) if sims else 0)
        axes2[0].plot(range(len(steps)), cos_vs_step, "o-", label=cname, markersize=5)

    axes2[0].set_xticks(range(len(steps)))
    axes2[0].set_xticklabels(steps, rotation=45, fontsize=7)
    axes2[0].set_ylabel("cos(direction, final_direction)")
    axes2[0].set_title("Concept Direction Emergence")
    axes2[0].legend(fontsize=7); axes2[0].grid(True, alpha=0.3)

    # Loss curve
    losses = [results[s]["val_loss"] for s in steps]
    axes2[1].plot(range(len(steps)), losses, "o-", color="red", markersize=8)
    axes2[1].set_xticks(range(len(steps)))
    axes2[1].set_xticklabels(steps, rotation=45, fontsize=7)
    axes2[1].set_ylabel("Validation Loss")
    axes2[1].set_title("Loss Curve"); axes2[1].grid(True, alpha=0.3)

    fig2.suptitle("Exp-009 Phase 4: Concept Direction Training Dynamics", fontsize=14)
    fig2.tight_layout(); fig2.savefig(out_dir / "direction_emergence.png", dpi=150)
    print(f"Fig 2: {out_dir / 'direction_emergence.png'}")

    # Summary
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for step in steps:
        print(f"  {step}: loss={results[step]['val_loss']:.4f}")
    print(f"\n  Direction emergence (cos with final):")
    for cname in concepts:
        early_cos = 0
        for l in range(n_layers):
            key = f"dir_L{l}"
            if key in results[steps[0]]["concepts"].get(cname, {}) and l in final_dirs.get(cname, {}):
                d = np.array(results[steps[0]]["concepts"][cname][key])
                d_final = final_dirs[cname][l]
                if len(d) == len(d_final):
                    early_cos = np.dot(d, d_final)
                    break
        print(f"    {cname}: step1000_cos={early_cos:.4f}")

    json.dump({"steps": steps, "losses": losses},
              open(out_dir / "summary.json", "w"), indent=2, default=str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", help="Checkpoint step (e.g. step1000)")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--cache-dir", default="/nvmessd/lifanhong/LR-env/cache/hf")
    parser.add_argument("--out-dir", default="/nvmessd/lifanhong/LR-env/training_dynamics")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.analyze:
        analyze(args.out_dir)
    elif args.step:
        process_checkpoint(args.step, args.cache_dir, args.out_dir, args.device)
    else:
        print("Specify --step or --analyze")

if __name__ == "__main__":
    main()
