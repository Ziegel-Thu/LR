#!/usr/bin/env python3
"""
Exp-014 Phase 2: Information lifecycle on IOI task.

Probe accuracy + AMNESIC ablation Δloss for "repeated name identity"
at every layer of GPT-2 small on IOI prompts.

Usage:
  CUDA_VISIBLE_DEVICES=0 python ioi_lifecycle.py \
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
# IOI prompts: "When [A] and [B] went ..., [B] gave ... to" → answer = A
IOI_TEMPLATES = [
    ("When {A} and {B} went to the store, {B} gave a drink to", 1),
    ("When {A} and {B} played together, {B} passed the ball to", 1),
    ("When {A} and {B} ate dinner, {B} offered the dessert to", 1),
    ("When {A} and {B} walked home, {B} handed the keys to", 1),
    ("When {A} and {B} worked late, {B} sent the report to", 1),
]
NAME_PAIRS = [("Mary","John"),("Alice","Bob"),("Sarah","Tom"),("Emma","Mike"),
              ("Lisa","David"),("Kate","James"),("Amy","Chris"),("Jane","Mark"),
              ("Laura","Paul"),("Susan","Steve")]


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

    tokenizer = with_retry(lambda: AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=args.cache_dir), "tok")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = with_retry(lambda: AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir=args.cache_dir, torch_dtype=torch.float32,
        output_hidden_states=True), "model")
    model = model.to(args.device).eval()
    n_layers = model.config.n_layer; d_model = model.config.n_embd

    # Generate IOI prompts + labels
    prompts, labels = [], []  # label: 1=A is IO (correct answer is A token)
    correct_tokens, wrong_tokens = [], []
    for template, _ in IOI_TEMPLATES:
        for A, B in NAME_PAIRS:
            prompts.append(template.format(A=A, B=B))
            labels.append(1)  # IO = A
            correct_tokens.append(f" {A}")
            wrong_tokens.append(f" {B}")

    print(f"{len(prompts)} IOI prompts", flush=True)

    # Cache hidden states at last position
    print("Caching hidden states...", flush=True)
    all_hidden = {l: [] for l in range(n_layers)}
    all_correct = []

    for i, prompt in enumerate(tqdm(prompts, desc="Cache")):
        tokens = tokenizer(prompt, return_tensors="pt").to(args.device)
        with torch.no_grad():
            out = model(**tokens)
        for l in range(n_layers):
            h = out.hidden_states[l+1][0, -1].cpu().float().numpy()
            all_hidden[l].append(h)
        # Binary label: does model predict correct token?
        correct_id = tokenizer.encode(correct_tokens[i])[-1]
        wrong_id = tokenizer.encode(wrong_tokens[i])[-1]
        logit_diff = (out.logits[0, -1, correct_id] - out.logits[0, -1, wrong_id]).item()
        all_correct.append(1 if logit_diff > 0 else 0)

    for l in range(n_layers):
        all_hidden[l] = np.stack(all_hidden[l])
    y = np.array(all_correct)
    print(f"  Model accuracy: {y.mean():.3f}", flush=True)

    # For probing: use "is IO = A" as binary label (always 1 in our setup)
    # Better: use which name is IO — probe if model knows identity
    # Create binary label: A vs B as IO
    io_labels = np.ones(len(prompts))  # All are "A is IO"
    # Add flipped versions
    flipped_prompts = []
    for template, _ in IOI_TEMPLATES:
        for A, B in NAME_PAIRS:
            flipped_prompts.append(template.format(A=B, B=A))  # Swap A and B
    prompts_all = prompts + flipped_prompts
    io_labels_all = np.concatenate([np.ones(len(prompts)), np.zeros(len(flipped_prompts))])

    # Cache flipped hidden states
    for i, prompt in enumerate(tqdm(flipped_prompts, desc="Cache flipped")):
        tokens = tokenizer(prompt, return_tensors="pt").to(args.device)
        with torch.no_grad():
            out = model(**tokens)
        for l in range(n_layers):
            h = out.hidden_states[l+1][0, -1].cpu().float().numpy()
            all_hidden[l] = np.vstack([all_hidden[l], h.reshape(1, -1)])

    # Phase 1: Probe accuracy per layer
    print("\nProbing IO identity per layer...", flush=True)
    probe_accs = {}
    probe_weights = {}
    y_all = torch.tensor(io_labels_all, dtype=torch.long)

    for l in tqdm(range(n_layers), desc="Probing"):
        X = torch.tensor(all_hidden[l], dtype=torch.float32)
        probe = nn.Linear(d_model, 1).to(args.device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        Xd, yd = X.to(args.device), y_all.float().to(args.device)
        for _ in range(20):
            logits = probe(Xd).squeeze()
            loss = nn.BCEWithLogitsLoss()(logits, yd)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc = ((probe(Xd).squeeze() > 0).long() == y_all.to(args.device)).float().mean().item()
        probe_accs[l] = acc
        probe_weights[l] = probe.weight.data.cpu().numpy().flatten()

    # Phase 2: Ablation per layer
    print("\nAblating IO direction per layer...", flush=True)

    # Baseline logit diff
    baseline_diffs = []
    for i, prompt in enumerate(prompts[:len(NAME_PAIRS)*len(IOI_TEMPLATES)]):
        tokens = tokenizer(prompt, return_tensors="pt").to(args.device)
        with torch.no_grad():
            out = model(**tokens)
        correct_id = tokenizer.encode(correct_tokens[i])[-1]
        wrong_id = tokenizer.encode(wrong_tokens[i])[-1]
        baseline_diffs.append((out.logits[0, -1, correct_id] - out.logits[0, -1, wrong_id]).item())
    mean_baseline = np.mean(baseline_diffs)

    ablation_effects = {}
    for l in tqdm(range(n_layers), desc="Ablating"):
        W = probe_weights[l]
        W_t = torch.tensor(W, dtype=torch.float32, device=args.device)
        W_norm = W_t / (W_t.norm() + 1e-8)

        def make_hook(w_norm):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                    proj = (h @ w_norm).unsqueeze(-1) * w_norm
                    return (h - proj,) + output[1:]
                proj = (output @ w_norm).unsqueeze(-1) * w_norm
                return output - proj
            return hook_fn

        hook = model.transformer.h[l].register_forward_hook(make_hook(W_norm))
        abl_diffs = []
        for i, prompt in enumerate(prompts[:len(NAME_PAIRS)*len(IOI_TEMPLATES)]):
            tokens = tokenizer(prompt, return_tensors="pt").to(args.device)
            with torch.no_grad():
                out = model(**tokens)
            correct_id = tokenizer.encode(correct_tokens[i])[-1]
            wrong_id = tokenizer.encode(wrong_tokens[i])[-1]
            abl_diffs.append((out.logits[0, -1, correct_id] - out.logits[0, -1, wrong_id]).item())
        hook.remove()
        ablation_effects[l] = mean_baseline - np.mean(abl_diffs)

    # Plot dual curve
    fig, ax = plt.subplots(figsize=(10, 6))
    layers = list(range(n_layers))
    accs = [probe_accs[l] for l in layers]
    deltas = [ablation_effects[l] for l in layers]

    ax.plot(layers, accs, "o-", color="blue", label="Probe accuracy (encoding)", markersize=6)
    ax2 = ax.twinx()
    ax2.plot(layers, deltas, "s-", color="red", label="Δlogit_diff (use)", markersize=6)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe Accuracy", color="blue")
    ax2.set_ylabel("Ablation Δlogit_diff", color="red")
    ax.set_title("Exp-014 Phase 2: IOI Information Lifecycle")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "ioi_lifecycle.png", dpi=150)
    print(f"\nFigure: {out_dir / 'ioi_lifecycle.png'}")

    results = {
        "model": MODEL_ID, "n_layers": n_layers, "n_prompts": len(prompts),
        "mean_baseline_logit_diff": mean_baseline,
        "probe_accuracy": {str(l): probe_accs[l] for l in layers},
        "ablation_effect": {str(l): ablation_effects[l] for l in layers},
    }
    with open(out_dir / "ioi_lifecycle.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"  Baseline logit diff: {mean_baseline:.4f}")
    best_probe = max(probe_accs, key=probe_accs.get)
    best_abl = max(ablation_effects, key=lambda l: abs(ablation_effects[l]))
    print(f"  Best probe: layer {best_probe}, acc={probe_accs[best_probe]:.3f}")
    print(f"  Best ablation: layer {best_abl}, Δ={ablation_effects[best_abl]:.4f}")
    print(f"  Peak alignment: {'YES' if best_probe == best_abl else 'NO'} "
          f"(probe@L{best_probe} vs ablation@L{best_abl})")

if __name__ == "__main__":
    main()
