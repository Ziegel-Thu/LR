#!/usr/bin/env python3
"""
Exp-010 Phase 4: Circuit dark matter — add MLPs to circuit search.

For each MLP layer in GPT-2 small, measure ablation effect on IOI.
Then do cumulative ablation of circuit-external components.

Usage:
  CUDA_VISIBLE_DEVICES=5 python dark_matter.py \
    --cache-dir /nvmessd/lifanhong/LR-env/cache/hf \
    --out-dir /nvmessd/lifanhong/LR-env/exp010
"""

import argparse, json, os
from pathlib import Path
import numpy as np, torch
from tqdm import tqdm
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

IOI_PROMPTS = [
    ("When Mary and John went to the store, John gave a drink to", " Mary", " John"),
    ("When Alice and Bob played together, Bob passed the ball to", " Alice", " Bob"),
    ("When Sarah and Tom ate dinner, Tom offered the dessert to", " Sarah", " Tom"),
    ("When Emma and Mike walked home, Mike handed the keys to", " Emma", " Mike"),
    ("When Lisa and David worked late, David sent the report to", " Lisa", " David"),
    ("When Kate and James studied math, James explained the problem to", " Kate", " James"),
    ("When Amy and Chris went shopping, Chris bought a gift for", " Amy", " Chris"),
    ("When Jane and Mark cooked dinner, Mark served the food to", " Jane", " Mark"),
    ("When Laura and Paul ran together, Paul gave the water to", " Laura", " Paul"),
    ("When Susan and Steve cleaned up, Steve handed the mop to", " Susan", " Steve"),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="/nvmessd/lifanhong/LR-env/cache/hf")
    parser.add_argument("--out-dir", default="/nvmessd/lifanhong/LR-env/exp010")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    import transformer_lens as tl

    print("Loading GPT-2 small...", flush=True)
    model = tl.HookedTransformer.from_pretrained("gpt2", cache_dir=args.cache_dir, device=args.device)
    n_layers = model.cfg.n_layers; n_heads = model.cfg.n_heads

    # Baseline logit diff
    print("Computing baseline...", flush=True)
    baseline_diffs = []
    for prompt, correct, wrong in IOI_PROMPTS:
        tokens = model.to_tokens(prompt)
        logits = model(tokens)
        cid = model.to_single_token(correct); wid = model.to_single_token(wrong)
        baseline_diffs.append((logits[0, -1, cid] - logits[0, -1, wid]).item())
    mean_baseline = np.mean(baseline_diffs)
    print(f"  Baseline logit diff: {mean_baseline:.4f}")

    # Phase 4a: Per-MLP ablation
    print(f"\n{'='*60}\nPhase 4a: Per-MLP Layer Ablation\n{'='*60}", flush=True)
    mlp_effects = {}
    for l in tqdm(range(n_layers), desc="MLP ablation"):
        diffs = []
        for pi, (prompt, correct, wrong) in enumerate(IOI_PROMPTS):
            tokens = model.to_tokens(prompt)
            def mlp_hook(mlp_out, hook):
                mlp_out[:, -1, :] = 0  # zero the MLP output at last position
                return mlp_out
            logits = model.run_with_hooks(tokens,
                fwd_hooks=[(f"blocks.{l}.hook_mlp_out", mlp_hook)])
            cid = model.to_single_token(correct); wid = model.to_single_token(wrong)
            diffs.append(baseline_diffs[pi] - (logits[0, -1, cid] - logits[0, -1, wid]).item())
        mlp_effects[l] = float(np.mean(diffs))
        print(f"  MLP L{l}: Δ={mlp_effects[l]:.4f}")

    # Phase 4b: Per-head ablation (for comparison)
    print(f"\n{'='*60}\nPhase 4b: Per-Head Ablation\n{'='*60}", flush=True)
    head_effects = {}
    for l in tqdm(range(n_layers), desc="Head ablation"):
        for h in range(n_heads):
            diffs = []
            for pi, (prompt, correct, wrong) in enumerate(IOI_PROMPTS):
                tokens = model.to_tokens(prompt)
                def head_hook(z, hook, head_idx=h):
                    z[0, -1, head_idx] = 0
                    return z
                logits = model.run_with_hooks(tokens,
                    fwd_hooks=[(f"blocks.{l}.attn.hook_z", head_hook)])
                cid = model.to_single_token(correct); wid = model.to_single_token(wrong)
                diffs.append(baseline_diffs[pi] - (logits[0, -1, cid] - logits[0, -1, wid]).item())
            head_effects[(l, h)] = float(np.mean(diffs))

    # Phase 4c: Cumulative ablation of least-important components
    print(f"\n{'='*60}\nPhase 4c: Cumulative Dark Matter Ablation\n{'='*60}", flush=True)

    # Rank all components by importance
    all_components = []
    for l in range(n_layers):
        all_components.append(("mlp", l, abs(mlp_effects[l])))
    for (l, h), eff in head_effects.items():
        all_components.append(("head", (l, h), abs(eff)))

    all_components.sort(key=lambda x: x[2])  # ascending by importance

    # Cumulatively ablate from least important
    cumulative_results = []
    n_total = len(all_components)
    test_fracs = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

    for frac in test_fracs:
        n_ablate = int(frac * n_total)
        to_ablate = all_components[:n_ablate]

        hooks = []
        for comp_type, comp_id, _ in to_ablate:
            if comp_type == "mlp":
                l = comp_id
                def make_mlp_hook(layer=l):
                    def hook(mlp_out, hook): mlp_out[:, -1, :] = 0; return mlp_out
                    return hook
                hooks.append((f"blocks.{l}.hook_mlp_out", make_mlp_hook()))
            else:
                l, h = comp_id
                def make_head_hook(head_idx=h):
                    def hook(z, hook): z[0, -1, head_idx] = 0; return z
                    return hook
                hooks.append((f"blocks.{l}.attn.hook_z", make_head_hook()))

        diffs = []
        for pi, (prompt, correct, wrong) in enumerate(IOI_PROMPTS):
            tokens = model.to_tokens(prompt)
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            cid = model.to_single_token(correct); wid = model.to_single_token(wrong)
            diffs.append((logits[0, -1, cid] - logits[0, -1, wid]).item())

        remaining_diff = np.mean(diffs)
        frac_preserved = remaining_diff / mean_baseline
        cumulative_results.append({
            "frac_ablated": frac, "n_ablated": n_ablate,
            "remaining_logit_diff": float(remaining_diff),
            "frac_preserved": float(frac_preserved),
        })
        print(f"  Ablate {frac:.0%} ({n_ablate} components): "
              f"remaining={remaining_diff:.4f} ({frac_preserved:.1%} preserved)")

    # Summary
    total_head_importance = sum(abs(v) for v in head_effects.values())
    total_mlp_importance = sum(abs(v) for v in mlp_effects.values())
    mlp_frac = total_mlp_importance / (total_head_importance + total_mlp_importance)

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"  Total head importance: {total_head_importance:.4f}")
    print(f"  Total MLP importance:  {total_mlp_importance:.4f}")
    print(f"  MLP fraction: {mlp_frac:.1%}")
    print(f"  Top 10% components carry: {1-cumulative_results[-1]['frac_preserved']:.1%} of signal")

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # MLP importance
    axes[0].bar(range(n_layers), [mlp_effects[l] for l in range(n_layers)])
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Δlogit_diff")
    axes[0].set_title("MLP Layer Importance (IOI)"); axes[0].grid(True, alpha=0.3)

    # Head importance heatmap
    head_matrix = np.zeros((n_layers, n_heads))
    for (l,h), v in head_effects.items(): head_matrix[l,h] = v
    im = axes[1].imshow(head_matrix, aspect="auto", cmap="RdBu_r")
    axes[1].set_xlabel("Head"); axes[1].set_ylabel("Layer")
    axes[1].set_title("Head Importance"); plt.colorbar(im, ax=axes[1])

    # Cumulative ablation
    fracs = [r["frac_ablated"] for r in cumulative_results]
    preserved = [r["frac_preserved"] for r in cumulative_results]
    axes[2].plot(fracs, preserved, "o-", markersize=8)
    axes[2].set_xlabel("Fraction of components ablated")
    axes[2].set_ylabel("Fraction of logit diff preserved")
    axes[2].set_title("Cumulative Dark Matter Ablation")
    axes[2].axhline(0.87, color="red", linestyle="--", label="Wang 2022 (87%)")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    fig.suptitle("Exp-010 Phase 4: Circuit Dark Matter", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "dark_matter.png", dpi=150)

    results = {"baseline": mean_baseline, "mlp_effects": mlp_effects,
               "mlp_fraction": mlp_frac, "cumulative": cumulative_results}
    with open(out_dir / "dark_matter.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults: {out_dir / 'dark_matter.json'}")

if __name__ == "__main__":
    main()
