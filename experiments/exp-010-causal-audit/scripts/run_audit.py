#!/usr/bin/env python3
"""
Exp-010: Causal Abstraction Methodology Audit

Phase 1: Off-distribution quantification — Mahalanobis distance of patched activations
Phase 2: Ablation baseline comparison — mean/zero/resample on IOI
Phase 3: Trivialization baseline — random GPT-2 IOI circuit discovery

Uses TransformerLens for model access and activation patching.

Usage:
  CUDA_VISIBLE_DEVICES=0 python run_audit.py \
    --cache-dir /nvmessd/lifanhong/LR-env/cache/hf \
    --out-dir /nvmessd/lifanhong/LR-env/exp010
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="/nvmessd/lifanhong/LR-env/cache/hf")
    parser.add_argument("--out-dir", default="/nvmessd/lifanhong/LR-env/exp010")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import transformer_lens as tl

    # ── Load model ──
    print("Loading GPT-2 small...", flush=True)
    model = tl.HookedTransformer.from_pretrained(
        "gpt2", cache_dir=args.cache_dir, device=args.device)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    print(f"  {n_layers} layers, {n_heads} heads, d_head={d_head}", flush=True)

    # ── IOI setup ──
    # Simple IOI-style prompts: "When Mary and John went to the store, John gave a drink to"
    # Correct answer: " Mary", Wrong answer: " John"
    ioi_prompts = [
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

    # Get baseline logit diff
    print("\nComputing baseline logit differences...", flush=True)
    baseline_diffs = []
    for prompt, correct, wrong in ioi_prompts:
        tokens = model.to_tokens(prompt)
        logits = model(tokens)
        correct_id = model.to_single_token(correct)
        wrong_id = model.to_single_token(wrong)
        diff = (logits[0, -1, correct_id] - logits[0, -1, wrong_id]).item()
        baseline_diffs.append(diff)
    mean_baseline = np.mean(baseline_diffs)
    print(f"  Mean baseline logit diff: {mean_baseline:.4f}", flush=True)

    # ── Phase 1: Off-distribution quantification ──
    print(f"\n{'='*60}")
    print("Phase 1: Off-distribution activation patching")
    print(f"{'='*60}")

    # Collect clean activation statistics per head
    print("  Collecting clean activation stats...", flush=True)
    head_means = {}  # (layer, head) -> mean vector
    head_covs = {}   # (layer, head) -> covariance

    all_head_acts = {(l, h): [] for l in range(n_layers) for h in range(n_heads)}

    for prompt, _, _ in ioi_prompts:
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens)
        for l in range(n_layers):
            z = cache[f"blocks.{l}.attn.hook_z"]  # (1, seq, n_heads, d_head)
            for h in range(n_heads):
                all_head_acts[(l, h)].append(z[0, -1, h].cpu().numpy())

    for key in all_head_acts:
        acts = np.stack(all_head_acts[key])  # (n_prompts, d_head)
        head_means[key] = acts.mean(axis=0)
        cov = np.cov(acts.T) + np.eye(d_head) * 1e-6
        head_covs[key] = np.linalg.inv(cov)

    # Patch each head and measure Mahalanobis distance + logit diff change
    print("  Patching each head...", flush=True)
    patch_results = []

    # Use corrupted prompts (swap names)
    corrupt_prompts = [
        ("When John and Mary went to the store, Mary gave a drink to", " Mary", " John"),
        ("When Bob and Alice played together, Alice passed the ball to", " Alice", " Bob"),
        ("When Tom and Sarah ate dinner, Sarah offered the dessert to", " Sarah", " Tom"),
        ("When Mike and Emma walked home, Emma handed the keys to", " Emma", " Mike"),
        ("When David and Lisa worked late, Lisa sent the report to", " Lisa", " David"),
        ("When James and Kate studied math, Kate explained the problem to", " Kate", " James"),
        ("When Chris and Amy went shopping, Amy bought a gift for", " Amy", " Chris"),
        ("When Mark and Jane cooked dinner, Jane served the food to", " Jane", " Mark"),
        ("When Paul and Laura ran together, Laura gave the water to", " Laura", " Paul"),
        ("When Steve and Susan cleaned up, Susan handed the mop to", " Susan", " Steve"),
    ]

    for l in tqdm(range(n_layers), desc="  Layers"):
        for h in range(n_heads):
            dists = []
            diff_changes = []

            for pi, ((clean_p, correct, wrong), (corrupt_p, _, _)) in enumerate(
                    zip(ioi_prompts, corrupt_prompts)):

                clean_tokens = model.to_tokens(clean_p)
                corrupt_tokens = model.to_tokens(corrupt_p)

                # Get corrupt cache
                _, corrupt_cache = model.run_with_cache(corrupt_tokens)
                corrupt_z = corrupt_cache[f"blocks.{l}.attn.hook_z"][0, -1, h]

                # Patch: replace clean head with corrupt
                def patch_hook(z, hook, head_idx=h, corrupt_act=corrupt_z):
                    z[0, -1, head_idx] = corrupt_act
                    return z

                patched_logits = model.run_with_hooks(
                    clean_tokens,
                    fwd_hooks=[(f"blocks.{l}.attn.hook_z", patch_hook)]
                )

                correct_id = model.to_single_token(correct)
                wrong_id = model.to_single_token(wrong)
                patched_diff = (patched_logits[0, -1, correct_id] -
                                patched_logits[0, -1, wrong_id]).item()

                # Mahalanobis distance of corrupt activation from clean distribution
                delta = corrupt_z.cpu().numpy() - head_means[(l, h)]
                mahal = np.sqrt(delta @ head_covs[(l, h)] @ delta)
                dists.append(mahal)
                diff_changes.append(baseline_diffs[pi] - patched_diff)

            patch_results.append({
                "layer": l, "head": h,
                "mean_mahalanobis": float(np.mean(dists)),
                "mean_logit_change": float(np.mean(diff_changes)),
                "abs_logit_change": float(np.mean(np.abs(diff_changes))),
            })

    # ── Phase 2: Ablation baseline comparison ──
    print(f"\n{'='*60}")
    print("Phase 2: Ablation baseline comparison")
    print(f"{'='*60}")

    ablation_results = {"zero": {}, "mean": {}, "resample": {}}

    for baseline_type in ["zero", "mean", "resample"]:
        print(f"  Baseline: {baseline_type}", flush=True)
        for l in tqdm(range(n_layers), desc=f"    {baseline_type}"):
            for h in range(n_heads):
                diffs_after = []
                for pi, (prompt, correct, wrong) in enumerate(ioi_prompts):
                    tokens = model.to_tokens(prompt)

                    if baseline_type == "zero":
                        def abl_hook(z, hook, head_idx=h):
                            z[0, -1, head_idx] = 0
                            return z
                    elif baseline_type == "mean":
                        mean_act = torch.tensor(head_means[(l, h)],
                                                dtype=torch.float32, device=args.device)
                        def abl_hook(z, hook, head_idx=h, ma=mean_act):
                            z[0, -1, head_idx] = ma
                            return z
                    else:  # resample
                        other_idx = (pi + 1) % len(ioi_prompts)
                        other_tokens = model.to_tokens(ioi_prompts[other_idx][0])
                        _, other_cache = model.run_with_cache(other_tokens)
                        resample_act = other_cache[f"blocks.{l}.attn.hook_z"][0, -1, h]
                        def abl_hook(z, hook, head_idx=h, ra=resample_act):
                            z[0, -1, head_idx] = ra
                            return z

                    abl_logits = model.run_with_hooks(
                        tokens, fwd_hooks=[(f"blocks.{l}.attn.hook_z", abl_hook)])
                    correct_id = model.to_single_token(correct)
                    wrong_id = model.to_single_token(wrong)
                    d = (abl_logits[0, -1, correct_id] - abl_logits[0, -1, wrong_id]).item()
                    diffs_after.append(baseline_diffs[pi] - d)

                ablation_results[baseline_type][(l, h)] = float(np.mean(diffs_after))

    # Compare baselines: rank correlation
    from scipy import stats as sp_stats
    heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    zero_vals = [ablation_results["zero"][k] for k in heads]
    mean_vals = [ablation_results["mean"][k] for k in heads]
    resample_vals = [ablation_results["resample"][k] for k in heads]

    rho_zm, _ = sp_stats.spearmanr(zero_vals, mean_vals)
    rho_zr, _ = sp_stats.spearmanr(zero_vals, resample_vals)
    rho_mr, _ = sp_stats.spearmanr(mean_vals, resample_vals)
    print(f"  Baseline rank correlations:")
    print(f"    zero↔mean: ρ={rho_zm:.4f}")
    print(f"    zero↔resample: ρ={rho_zr:.4f}")
    print(f"    mean↔resample: ρ={rho_mr:.4f}")

    # ── Phase 3: Trivialization baseline ──
    print(f"\n{'='*60}")
    print("Phase 3: Random network trivialization")
    print(f"{'='*60}")

    random_model = tl.HookedTransformer.from_pretrained(
        "gpt2", cache_dir=args.cache_dir, device=args.device)
    # Randomize weights
    for p in random_model.parameters():
        torch.nn.init.normal_(p, mean=0, std=0.02)

    random_ablation = {}
    for l in tqdm(range(n_layers), desc="  Random model"):
        for h in range(n_heads):
            diffs_after = []
            for pi, (prompt, correct, wrong) in enumerate(ioi_prompts):
                tokens = random_model.to_tokens(prompt)

                def abl_hook(z, hook, head_idx=h):
                    z[0, -1, head_idx] = 0
                    return z

                try:
                    abl_logits = random_model.run_with_hooks(
                        tokens, fwd_hooks=[(f"blocks.{l}.attn.hook_z", abl_hook)])
                    correct_id = random_model.to_single_token(correct)
                    wrong_id = random_model.to_single_token(wrong)
                    d = (abl_logits[0, -1, correct_id] - abl_logits[0, -1, wrong_id]).item()
                    base = random_model(tokens)
                    base_d = (base[0, -1, correct_id] - base[0, -1, wrong_id]).item()
                    diffs_after.append(base_d - d)
                except Exception:
                    diffs_after.append(0)

            random_ablation[(l, h)] = float(np.mean(diffs_after))

    # Compare trained vs random
    trained_importance = [abs(ablation_results["zero"][k]) for k in heads]
    random_importance = [abs(random_ablation[k]) for k in heads]
    rho_tr, _ = sp_stats.spearmanr(trained_importance, random_importance)
    print(f"  Trained vs random importance rank correlation: ρ={rho_tr:.4f}")

    del random_model
    torch.cuda.empty_cache()

    # ── Figures ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Mahalanobis distance vs logit change
    ax = axes[0, 0]
    mahals = [r["mean_mahalanobis"] for r in patch_results]
    changes = [r["abs_logit_change"] for r in patch_results]
    ax.scatter(mahals, changes, s=15, alpha=0.5)
    ax.set_xlabel("Mahalanobis Distance (off-distribution)")
    ax.set_ylabel("|Δ Logit Diff| (causal importance)")
    ax.set_title("Phase 1: Off-Distribution vs Causal Effect")
    ax.grid(True, alpha=0.3)

    # 2. Baseline comparison
    ax = axes[0, 1]
    ax.scatter(zero_vals, mean_vals, s=15, alpha=0.5, label=f"zero↔mean ρ={rho_zm:.3f}")
    ax.scatter(zero_vals, resample_vals, s=15, alpha=0.5, label=f"zero↔resample ρ={rho_zr:.3f}")
    ax.set_xlabel("Zero ablation effect")
    ax.set_ylabel("Other baseline effect")
    ax.set_title("Phase 2: Ablation Baseline Comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Trained vs random importance
    ax = axes[1, 0]
    ax.scatter(trained_importance, random_importance, s=15, alpha=0.5)
    ax.set_xlabel("Trained model |ablation effect|")
    ax.set_ylabel("Random model |ablation effect|")
    ax.set_title(f"Phase 3: Trained vs Random (ρ={rho_tr:.3f})")
    ax.grid(True, alpha=0.3)

    # 4. Head importance heatmap (trained, zero ablation)
    ax = axes[1, 1]
    importance_matrix = np.zeros((n_layers, n_heads))
    for l in range(n_layers):
        for h in range(n_heads):
            importance_matrix[l, h] = abs(ablation_results["zero"][(l, h)])
    im = ax.imshow(importance_matrix, aspect="auto", cmap="Reds")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title("Head Importance (zero ablation)")
    plt.colorbar(im, ax=ax)

    fig.suptitle("Exp-010: Causal Abstraction Methodology Audit", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "audit_figures.png", dpi=150)
    print(f"\nFigures: {out_dir / 'audit_figures.png'}")

    # Save results
    results = {
        "n_layers": n_layers, "n_heads": n_heads,
        "mean_baseline_logit_diff": mean_baseline,
        "phase1_patch_results": patch_results,
        "phase2_baseline_correlations": {
            "zero_mean_rho": rho_zm, "zero_resample_rho": rho_zr, "mean_resample_rho": rho_mr,
        },
        "phase3_trained_vs_random_rho": rho_tr,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results: {out_dir / 'results.json'}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Phase 1: Mahalanobis-logit correlation (check figure)")
    print(f"  Phase 2: zero↔mean ρ={rho_zm:.4f}, zero↔resample ρ={rho_zr:.4f}")
    print(f"  Phase 3: trained↔random importance ρ={rho_tr:.4f}")


if __name__ == "__main__":
    main()
