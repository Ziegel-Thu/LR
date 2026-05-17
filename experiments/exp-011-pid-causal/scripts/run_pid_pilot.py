#!/usr/bin/env python3
"""
Exp-011: Information vs Causation — MI(neuron, output) vs ablation effect.

For top-50 neurons at GPT-2 small layer 5:
  1. MI(neuron_activation, next_token_logprob) via binned estimation
  2. Causal effect via zero ablation → Δloss
  3. Spearman correlation between the two rankings

Usage:
  CUDA_VISIBLE_DEVICES=5 python run_pid_pilot.py \
    --cache-dir /nvmessd/lifanhong/LR-env/cache/hf \
    --out-dir /nvmessd/lifanhong/LR-env/exp011
"""

import argparse, json, os, gc, time
from pathlib import Path
import numpy as np, torch
from tqdm import tqdm
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats as sp_stats

MODEL_ID = "openai-community/gpt2"
N_SENTENCES = 500; SEQ_LEN = 64; BATCH_SIZE = 16
TARGET_LAYER = 5; N_NEURONS = 50


def with_retry(fn, label, attempts=5, sleep_s=20):
    for a in range(1, attempts+1):
        try: return fn()
        except Exception as e:
            print(f"[{label}] {a}/{attempts}: {e}", flush=True)
            if a < attempts: time.sleep(sleep_s)
    raise RuntimeError(f"{label} failed")


def binned_mi(x, y, n_bins=20):
    x_bins = np.digitize(x, np.linspace(x.min()-1e-10, x.max()+1e-10, n_bins+1)) - 1
    y_bins = np.digitize(y, np.linspace(y.min()-1e-10, y.max()+1e-10, n_bins+1)) - 1
    pxy = np.zeros((n_bins, n_bins))
    for xi, yi in zip(x_bins, y_bins):
        pxy[min(xi,n_bins-1), min(yi,n_bins-1)] += 1
    pxy = pxy / pxy.sum() + 1e-10
    px, py = pxy.sum(1), pxy.sum(0)
    mi = sum(pxy[i,j]*np.log(pxy[i,j]/(px[i]*py[j]+1e-10))
             for i in range(n_bins) for j in range(n_bins) if pxy[i,j]>1e-10)
    return max(0, mi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="/nvmessd/lifanhong/LR-env/cache/hf")
    parser.add_argument("--out-dir", default="/nvmessd/lifanhong/LR-env/exp011")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print("Loading GPT-2 small...", flush=True)
    tokenizer = with_retry(lambda: AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=args.cache_dir), "tok")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = with_retry(lambda: AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir=args.cache_dir, torch_dtype=torch.float32), "model")
    model = model.to(args.device).eval()
    d_model = model.config.n_embd
    print(f"  {model.config.n_layer} layers, d={d_model}", flush=True)

    ds = with_retry(lambda: load_dataset("Salesforce/wikitext","wikitext-103-raw-v1",split="validation",
                    cache_dir=str(Path(args.cache_dir)/"datasets")), "data")
    texts = [t for t in ds["text"] if len(t.strip())>30][:N_SENTENCES]

    # Cache activations
    print(f"\nCaching layer {TARGET_LAYER} activations...", flush=True)
    all_acts, all_logits = [], []
    for i in tqdm(range(0,len(texts),BATCH_SIZE), desc="Cache"):
        batch = texts[i:i+BATCH_SIZE]
        tokens = tokenizer(batch, return_tensors="pt", max_length=SEQ_LEN,
                           truncation=True, padding="max_length").to(args.device)
        with torch.no_grad():
            out = model(**tokens, output_hidden_states=True)
        h = out.hidden_states[TARGET_LAYER+1]
        mask = tokens["attention_mask"].bool()
        logits = out.logits
        for b in range(h.shape[0]):
            valid = mask[b].clone(); valid[-1] = False
            if valid.sum() < 2: continue
            acts = h[b][valid].cpu().float().numpy()
            next_ids = tokens["input_ids"][b,1:][valid[:-1]].cpu()
            log_p = torch.log_softmax(logits[b,:-1][valid[:-1]], dim=-1)
            nlp = log_p.gather(1, next_ids.unsqueeze(1).to(args.device)).squeeze().cpu().numpy()
            all_acts.append(acts[:len(nlp)]); all_logits.append(nlp)
    acts = np.concatenate(all_acts); logit_vals = np.concatenate(all_logits)
    print(f"  {len(acts)} tokens", flush=True)

    # Top neurons
    top_neurons = np.argsort(acts.var(0))[::-1][:N_NEURONS]

    # MI
    print(f"\nComputing MI for {N_NEURONS} neurons...", flush=True)
    mi_scores = {int(n): binned_mi(acts[:,n], logit_vals) for n in tqdm(top_neurons, desc="MI")}

    # Ablation
    print(f"\nZero ablation...", flush=True)
    baseline_losses = []
    for i in range(0,len(texts),BATCH_SIZE):
        tokens = tokenizer(texts[i:i+BATCH_SIZE], return_tensors="pt", max_length=SEQ_LEN,
                           truncation=True, padding="max_length").to(args.device)
        with torch.no_grad(): out = model(**tokens, labels=tokens["input_ids"])
        baseline_losses.append(out.loss.item())
    baseline = np.mean(baseline_losses)

    abl_effects = {}
    for ni in tqdm(top_neurons, desc="Ablation"):
        def make_hook(idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]; h[:,:,idx] = 0; return (h,)+output[1:]
                output[:,:,idx] = 0; return output
            return hook_fn
        hook = model.transformer.h[TARGET_LAYER].register_forward_hook(make_hook(int(ni)))
        losses = []
        for i in range(0,len(texts),BATCH_SIZE):
            tokens = tokenizer(texts[i:i+BATCH_SIZE], return_tensors="pt", max_length=SEQ_LEN,
                               truncation=True, padding="max_length").to(args.device)
            with torch.no_grad(): out = model(**tokens, labels=tokens["input_ids"])
            losses.append(out.loss.item())
        hook.remove()
        abl_effects[int(ni)] = np.mean(losses) - baseline

    # Correlation
    neurons = list(mi_scores.keys())
    mi_v = [mi_scores[n] for n in neurons]
    abl_v = [abs(abl_effects[n]) for n in neurons]
    rho, p = sp_stats.spearmanr(mi_v, abl_v)
    r, rp = sp_stats.pearsonr(mi_v, abl_v)

    print(f"\n{'='*60}\nRESULTS\n{'='*60}")
    print(f"  Spearman ρ(MI, |Δloss|) = {rho:.4f} (p={p:.4f})")
    print(f"  Pearson r = {r:.4f} (p={rp:.4f})")

    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(mi_v, abl_v, s=40, alpha=0.7)
    ax.set_xlabel("MI(neuron, next_token_logprob)")
    ax.set_ylabel("|Δloss| from zero ablation")
    ax.set_title(f"Exp-011: Info vs Causation (ρ={rho:.3f}, p={p:.3f})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "mi_vs_ablation.png", dpi=150)

    json.dump({"spearman_rho": rho, "spearman_p": p, "pearson_r": r, "n_neurons": len(neurons),
               "baseline_loss": baseline, "mi_scores": mi_scores, "ablation_effects": abl_effects},
              open(out_dir / "results.json", "w"), indent=2, default=str)
    print(f"Saved: {out_dir}")

if __name__ == "__main__":
    main()
