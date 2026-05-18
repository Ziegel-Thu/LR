#!/usr/bin/env python3
"""
Exp-009 Phase 3: LRH on Mamba — same concept directions, compare with Pythia.

Usage:
  CUDA_VISIBLE_DEVICES=2 python run_mamba_lrh.py \
    --cache-dir /nvmessd/lifanhong/LR-env/cache/hf \
    --out-dir /nvmessd/lifanhong/LR-env/exp009
"""

import argparse, gc, json, os, time
from pathlib import Path
import numpy as np, torch
from tqdm import tqdm
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

MODELS = {
    "Mamba-130M": "state-spaces/mamba-130m-hf",
    "Pythia-160M": "EleutherAI/pythia-160m-deduped",
}
N_SENTENCES = 2000; SEQ_LEN = 128; BATCH_SIZE = 8

def with_retry(fn, label, attempts=5, sleep_s=20):
    for a in range(1, attempts+1):
        try: return fn()
        except Exception as e:
            print(f"[{label}] {a}/{attempts}: {e}", flush=True)
            if a < attempts: time.sleep(sleep_s)
    raise RuntimeError(f"{label} failed")

def label_concepts(token_strs):
    return {
        "capitalized": np.array([1 if t.lstrip() and t.lstrip()[0].isupper() else 0 for t in token_strs]),
        "numeric": np.array([1 if any(c.isdigit() for c in t) else 0 for t in token_strs]),
        "punctuation": np.array([1 if t.strip() and all(not c.isalnum() for c in t.strip()) else 0 for t in token_strs]),
        "short": np.array([1 if len(t.strip()) <= 2 else 0 for t in token_strs]),
        "plural": np.array([1 if len(t.strip())>2 and t.strip().lower().endswith(("s","es","ies")) else 0 for t in token_strs]),
        "subword": np.array([1 if not (t.startswith(" ") or t.startswith("Ġ")) else 0 for t in token_strs]),
        "starts_vowel": np.array([1 if t.strip().lower()[:1] in "aeiou" else 0 for t in token_strs]),
    }

def diff_in_means(hidden, labels):
    pos, neg = labels==1, labels==0
    if pos.sum()<10 or neg.sum()<10: return None
    d = hidden[pos].mean(0) - hidden[neg].mean(0)
    n = np.linalg.norm(d)
    return d/n if n>1e-10 else None

def extract_directions(model_name, model_id, cache_dir, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tokenizer = with_retry(lambda: AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir), "tok")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = with_retry(lambda: AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=cache_dir, torch_dtype=torch.float16, output_hidden_states=True), "model")
    model = model.to(device).eval()

    if hasattr(model.config, 'num_hidden_layers'):
        n_layers = model.config.num_hidden_layers
    elif hasattr(model.config, 'n_layer'):
        n_layers = model.config.n_layer
    else:
        n_layers = len([m for m in model.modules() if 'Block' in type(m).__name__ or 'Layer' in type(m).__name__])

    ds = with_retry(lambda: load_dataset("Salesforce/wikitext","wikitext-103-raw-v1",split="validation",
                    cache_dir=str(Path(cache_dir)/"datasets")), "data")
    texts = [t for t in ds["text"] if len(t.strip())>50][:N_SENTENCES]

    layer_hiddens = {l:[] for l in range(n_layers)}
    all_strs = []
    for i in tqdm(range(0,len(texts),BATCH_SIZE), desc=f"Cache {model_name}"):
        batch = texts[i:i+BATCH_SIZE]
        tokens = tokenizer(batch, return_tensors="pt", max_length=SEQ_LEN, truncation=True, padding="max_length").to(device)
        with torch.no_grad():
            out = model(**tokens)
        mask = tokens["attention_mask"].cpu().bool()
        for l in range(n_layers):
            h = out.hidden_states[l+1].cpu().float()
            for b in range(h.shape[0]):
                layer_hiddens[l].append(h[b][mask[b]].numpy())
        for b in range(tokens["input_ids"].shape[0]):
            valid = mask[b]
            ids = tokens["input_ids"][b][valid].cpu()
            all_strs.extend([tokenizer.decode([tid]) for tid in ids])

    for l in range(n_layers):
        layer_hiddens[l] = np.concatenate(layer_hiddens[l], axis=0)

    concepts = label_concepts(all_strs)
    valid = {k:v for k,v in concepts.items() if 0.02<v.mean()<0.98}

    directions = {}
    for cname, labels in valid.items():
        directions[cname] = {}
        for l in range(n_layers):
            directions[cname][l] = diff_in_means(layer_hiddens[l], labels)

    del model; torch.cuda.empty_cache(); gc.collect()
    return directions, n_layers, valid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="/nvmessd/lifanhong/LR-env/cache/hf")
    parser.add_argument("--out-dir", default="/nvmessd/lifanhong/LR-env/exp009")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    all_dirs = {}
    all_nlayers = {}
    for mname, mid in MODELS.items():
        print(f"\n{'='*60}\n{mname}\n{'='*60}", flush=True)
        dirs, nl, _ = extract_directions(mname, mid, args.cache_dir, args.device)
        all_dirs[mname] = dirs
        all_nlayers[mname] = nl

    # Compare: same concept, Mamba vs Pythia at matched relative depth
    print(f"\n{'='*60}\nCross-architecture concept alignment\n{'='*60}", flush=True)
    concepts = list(set(all_dirs["Mamba-130M"].keys()) & set(all_dirs["Pythia-160M"].keys()))
    cross_results = {}

    for cname in concepts:
        nl_m = all_nlayers["Mamba-130M"]
        nl_p = all_nlayers["Pythia-160M"]
        n_compare = min(nl_m, nl_p)
        sims = []
        for ci in range(n_compare):
            lm = int(ci*(nl_m-1)/max(n_compare-1,1))
            lp = int(ci*(nl_p-1)/max(n_compare-1,1))
            dm = all_dirs["Mamba-130M"][cname].get(lm)
            dp = all_dirs["Pythia-160M"][cname].get(lp)
            if dm is not None and dp is not None and len(dm)==len(dp):
                sims.append(float(np.dot(dm, dp)))
            else:
                sims.append(0)
        cross_results[cname] = {"sims": sims, "mean": float(np.mean(sims)), "max": float(np.max(np.abs(sims)))}
        print(f"  {cname}: mean_cos={cross_results[cname]['mean']:.4f}, max_|cos|={cross_results[cname]['max']:.4f}")

    # Figure
    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    for cname in concepts:
        sims = cross_results[cname]["sims"]
        alphas = np.linspace(0, 1, len(sims))
        ax.plot(alphas, sims, "o-", label=cname, markersize=3, alpha=0.7)
    ax.set_xlabel("Normalized depth")
    ax.set_ylabel("Cosine similarity (Mamba ↔ Pythia)")
    ax.set_title("Concept Direction Alignment: Mamba-130M vs Pythia-160M")
    ax.legend(fontsize=7, ncol=2)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "cross_arch_lrh.png", dpi=150)
    print(f"\nFigure: {out_dir / 'cross_arch_lrh.png'}")

    results = {"models": list(MODELS.keys()), "cross_arch_alignment": cross_results}
    with open(out_dir / "results_phase3.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results: {out_dir / 'results_phase3.json'}")

if __name__ == "__main__":
    main()
