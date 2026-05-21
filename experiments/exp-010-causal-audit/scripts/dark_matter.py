#!/usr/bin/env python3
"""
Exp-010 Phase 4: Circuit "dark matter" exploration (expanded)

Three analyses on GPT-2 small using WikiText-2 validation:
  1. Layer knockout — zero each layer's attn/mlp/both, measure Δperplexity
  2. Cumulative ablation — remove layers least→most important, find failure threshold
  3. Residual stream decomposition — direct logit attribution per component

Usage:
  CUDA_VISIBLE_DEVICES=1 python dark_matter.py \
    --output-dir /nvmessd/lifanhong/LR-env/exp010-dark-matter
"""

import argparse, json, os, time
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Helpers ──────────────────────────────────────────────────────

def load_wikitext(tokenizer, max_seqs=100, seq_len=256):
    """Tokenize WikiText-2 validation into fixed-length sequences."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer.encode(text)
    seqs = []
    for i in range(0, len(ids) - seq_len, seq_len):
        seqs.append(ids[i : i + seq_len])
        if len(seqs) >= max_seqs:
            break
    return torch.tensor(seqs)


@torch.no_grad()
def eval_loss(model, data, device, batch_size=8):
    """Average cross-entropy loss over data."""
    total_loss, total_tok = 0.0, 0
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size].to(device)
        out = model(batch, labels=batch)
        n = batch.shape[0] * (batch.shape[1] - 1)
        total_loss += out.loss.item() * n
        total_tok += n
    return total_loss / total_tok


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seqs", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--decomp-seqs", type=int, default=20,
                        help="Sequences for residual-stream decomposition")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print("Loading GPT-2 small …", flush=True)
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(dev)
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model.eval()

    n_layers = model.config.n_layer   # 12
    d_model  = model.config.n_embd    # 768
    print(f"  {n_layers} layers, d={d_model}, device={dev}")

    print("Loading WikiText-2 validation …", flush=True)
    data = load_wikitext(tok, max_seqs=args.max_seqs, seq_len=args.seq_len)
    print(f"  {data.shape[0]} seqs × {data.shape[1]} tokens")

    base_loss = eval_loss(model, data, dev, args.batch_size)
    base_ppl  = float(np.exp(base_loss))
    print(f"  Baseline: loss={base_loss:.4f}, PPL={base_ppl:.2f}\n")

    results = {
        "config": {"model": "gpt2", "n_layers": n_layers, "d_model": d_model,
                    "n_seqs": int(data.shape[0]), "seq_len": int(data.shape[1])},
        "baseline": {"loss": float(base_loss), "ppl": base_ppl},
    }

    # ================================================================
    # Analysis 1: Layer Knockout
    # ================================================================
    print("=" * 60)
    print("Analysis 1: Layer Knockout")
    print("=" * 60)

    knockout = []
    for layer in range(n_layers):
        for comp in ("attn", "mlp", "both"):
            hooks = []
            if comp in ("attn", "both"):
                def _ah(mod, inp, out, _l=layer):
                    return (torch.zeros_like(out[0]),) + out[1:]
                hooks.append(
                    model.transformer.h[layer].attn.register_forward_hook(_ah))
            if comp in ("mlp", "both"):
                def _mh(mod, inp, out, _l=layer):
                    return torch.zeros_like(out)
                hooks.append(
                    model.transformer.h[layer].mlp.register_forward_hook(_mh))

            loss = eval_loss(model, data, dev, args.batch_size)
            for h in hooks:
                h.remove()

            dl = loss - base_loss
            ppl = float(np.exp(loss))
            knockout.append({"layer": layer, "comp": comp,
                             "loss": float(loss), "ppl": ppl,
                             "delta_loss": float(dl)})
            print(f"  L{layer:2d} {comp:4s}: loss={loss:.4f} (Δ={dl:+.4f}), "
                  f"PPL={ppl:.1f}")

    results["knockout"] = knockout

    # ================================================================
    # Analysis 2: Cumulative Ablation
    # ================================================================
    print(f"\n{'=' * 60}")
    print("Analysis 2: Cumulative Ablation (least → most important)")
    print("=" * 60)

    layer_imp = sorted(
        [(r["layer"], r["delta_loss"]) for r in knockout if r["comp"] == "both"],
        key=lambda x: x[1],
    )
    print("Importance ranking (least → most):")
    for rank, (l, d) in enumerate(layer_imp):
        print(f"  #{rank + 1}: Layer {l} (Δloss={d:.4f})")

    cumul = []
    for n_abl in range(n_layers + 1):
        to_abl = [l for l, _ in layer_imp[:n_abl]]
        hooks = []
        for l in to_abl:
            def _a(mod, inp, out, _l=l):
                return (torch.zeros_like(out[0]),) + out[1:]
            def _m(mod, inp, out, _l=l):
                return torch.zeros_like(out)
            hooks.append(
                model.transformer.h[l].attn.register_forward_hook(_a))
            hooks.append(
                model.transformer.h[l].mlp.register_forward_hook(_m))

        loss = eval_loss(model, data, dev, args.batch_size)
        for h in hooks:
            h.remove()

        ppl = float(np.exp(loss))
        cumul.append({"n_ablated": n_abl, "layers": to_abl,
                       "loss": float(loss), "ppl": ppl,
                       "ppl_ratio": float(ppl / base_ppl)})
        tag = str(to_abl) if to_abl else "none"
        print(f"  Ablate {n_abl:2d}/{n_layers}: "
              f"PPL={ppl:.1f} ({ppl / base_ppl:.2f}× baseline)  [{tag}]")

    results["cumulative"] = cumul

    # Find catastrophic-failure threshold (PPL > 5× baseline)
    failure_threshold = None
    for c in cumul:
        if c["ppl_ratio"] > 5.0:
            failure_threshold = c["n_ablated"]
            break
    print(f"\n  Catastrophic failure (>5× PPL) at {failure_threshold} layers ablated"
          if failure_threshold else "\n  No catastrophic failure within range")

    # ================================================================
    # Analysis 3: Residual Stream Decomposition (Direct Logit Attribution)
    # ================================================================
    print(f"\n{'=' * 60}")
    print("Analysis 3: Residual Stream Decomposition")
    print("=" * 60)

    n_dec = min(args.decomp_seqs, len(data))
    comp_names = (["embed"]
                  + [f"L{l}_attn" for l in range(n_layers)]
                  + [f"L{l}_mlp"  for l in range(n_layers)])
    n_comp = len(comp_names)  # 25

    all_abs_contribs = []          # per-position absolute contributions
    top_k_buckets = {k: [] for k in [1, 3, 5, 10, 15, 24]}
    sanity_errors = []             # decomposition sanity-check errors

    ln_w = model.transformer.ln_f.weight.detach()    # (d,)
    ln_b = model.transformer.ln_f.bias.detach()      # (d,)
    W_U  = model.lm_head.weight.detach()             # (V, d)

    for si in tqdm(range(n_dec), desc="Decompose"):
        ids = data[si : si + 1].to(dev)
        seq_len = ids.shape[1]

        # Cache attn / mlp outputs per layer
        cached_attn, cached_mlp = {}, {}
        hooks = []
        for l in range(n_layers):
            def _ca(mod, inp, out, layer=l):
                cached_attn[layer] = out[0].detach()
                return out
            def _cm(mod, inp, out, layer=l):
                cached_mlp[layer] = out.detach()
                return out
            hooks.append(
                model.transformer.h[l].attn.register_forward_hook(_ca))
            hooks.append(
                model.transformer.h[l].mlp.register_forward_hook(_cm))

        logits = model(ids).logits  # (1, S, V)
        for h in hooks:
            h.remove()

        # Embedding contribution
        pos_ids = torch.arange(seq_len, device=dev)
        embed = (model.transformer.wte(ids[0])
                 + model.transformer.wpe(pos_ids))  # (S, d)

        # Full residual = embed + Σ_l (attn_l + mlp_l)
        residual = embed.clone()
        for l in range(n_layers):
            residual = residual + cached_attn[l][0] + cached_mlp[l][0]

        # Targets: position p predicts token at p+1
        target_ids = ids[0, 1:]  # (S-1,)
        # Use positions 0..S-2 (each predicts the next token)
        P = seq_len - 1

        # Per-position normalization stats from full residual
        r = residual[:P]                          # (P, d)
        mu  = r.mean(dim=-1, keepdim=True)        # (P, 1)
        var = r.var(dim=-1, correction=0, keepdim=True)
        sigma = torch.sqrt(var + 1e-5)            # (P, 1)

        # Target unembedding vectors
        w_u = W_U[target_ids]                     # (P, d)

        # Scaling factor: (1/σ) * γ * w_u  — same for all components
        scale = w_u * ln_w.unsqueeze(0) / sigma   # (P, d)

        # Build component list: embed, L0_attn, L0_mlp, …, L11_attn, L11_mlp
        comps = [embed[:P]]
        for l in range(n_layers):
            comps.append(cached_attn[l][0, :P])
            comps.append(cached_mlp[l][0, :P])

        # Per-component logit contribution: (scale * c_i).sum(dim=-1)  → (P,)
        contribs = torch.stack([(scale * c).sum(dim=-1) for c in comps])  # (n_comp, P)

        # Bias term: -(μ/σ) (w_u · γ) + w_u · β
        bias = (-(mu / sigma) * (w_u * ln_w.unsqueeze(0)).sum(dim=-1, keepdim=True)
                + (w_u * ln_b.unsqueeze(0)).sum(dim=-1, keepdim=True))  # (P, 1)

        # Sanity check: Σcontrib + bias ≈ actual logit?
        reconstructed = contribs.sum(dim=0) + bias.squeeze(-1)
        actual = logits[0, :P, :].gather(
            1, target_ids.unsqueeze(-1)).squeeze(-1)
        err = (reconstructed - actual).abs().mean().item()
        sanity_errors.append(err)

        # Per-position statistics
        abs_c = contribs.abs()                     # (n_comp, P)
        total_per_pos = abs_c.sum(dim=0)           # (P,)

        # Average absolute contribution per component (over positions)
        all_abs_contribs.append(abs_c.mean(dim=1).cpu().numpy())  # (n_comp,)

        # Top-K explained fraction per position, then average
        for k in top_k_buckets:
            # topk along component dimension for each position
            topk_vals, _ = abs_c.topk(min(k, n_comp), dim=0)
            frac = (topk_vals.sum(dim=0) / (total_per_pos + 1e-12)).mean().item()
            top_k_buckets[k].append(frac)

    # Aggregate across sequences
    avg_contribs = np.mean(all_abs_contribs, axis=0)
    avg_top_k = {str(k): float(np.mean(v)) for k, v in top_k_buckets.items()}
    mean_sanity = float(np.mean(sanity_errors))

    print(f"\n  Decomposition sanity check: mean |error| = {mean_sanity:.4f}")
    print(f"\n  Average |contribution| by component (top 10):")
    ranked = sorted(zip(comp_names, avg_contribs), key=lambda x: -x[1])
    for name, val in ranked[:10]:
        print(f"    {name:10s}: {val:.4f}")

    print(f"\n  Fraction explained by top-K components (avg over "
          f"{n_dec} seqs × {data.shape[1]-1} positions):")
    for k in sorted(top_k_buckets):
        print(f"    Top-{k:2d}: {avg_top_k[str(k)]:.1%}")

    results["decomposition"] = {
        "avg_abs_contributions": {n: float(v) for n, v in zip(comp_names, avg_contribs)},
        "ranked_components": [(n, float(v)) for n, v in ranked],
        "top_k_explained": avg_top_k,
        "sanity_check_mean_error": mean_sanity,
        "n_sequences": n_dec,
    }

    # ================================================================
    # Summary
    # ================================================================
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)

    # Most / least important layers
    most_imp = layer_imp[-1]
    least_imp = layer_imp[0]
    print(f"  Most important layer:  L{most_imp[0]} (Δloss={most_imp[1]:.4f})")
    print(f"  Least important layer: L{least_imp[0]} (Δloss={least_imp[1]:.4f})")

    # Attn vs MLP breakdown
    attn_total = sum(r["delta_loss"] for r in knockout if r["comp"] == "attn")
    mlp_total  = sum(r["delta_loss"] for r in knockout if r["comp"] == "mlp")
    both_total = attn_total + mlp_total
    print(f"  Attn total Δloss: {attn_total:.4f} ({attn_total/both_total:.1%})")
    print(f"  MLP total Δloss:  {mlp_total:.4f} ({mlp_total/both_total:.1%})")

    # Cumulative ablation threshold
    for c in cumul:
        if c["ppl_ratio"] > 2.0:
            print(f"  PPL doubles at {c['n_ablated']} layers ablated")
            break

    # Top-K
    print(f"  Top-5 components explain {avg_top_k['5']:.1%} of logit (avg)")
    print(f"  Top-10 components explain {avg_top_k['10']:.1%} of logit (avg)")
    print(f"  Elapsed: {elapsed:.0f}s")

    results["summary"] = {
        "most_important_layer": most_imp[0],
        "least_important_layer": least_imp[0],
        "attn_fraction": float(attn_total / both_total),
        "mlp_fraction": float(mlp_total / both_total),
        "failure_threshold_layers": failure_threshold,
        "elapsed_seconds": float(elapsed),
    }

    # ================================================================
    # Figures
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1a: Layer knockout grouped bar
    ax = axes[0, 0]
    attn_dl = [r["delta_loss"] for r in knockout if r["comp"] == "attn"]
    mlp_dl  = [r["delta_loss"] for r in knockout if r["comp"] == "mlp"]
    both_dl = [r["delta_loss"] for r in knockout if r["comp"] == "both"]
    x = np.arange(n_layers)
    w = 0.25
    ax.bar(x - w, attn_dl, w, label="Attn only", color="steelblue")
    ax.bar(x,     mlp_dl,  w, label="MLP only",  color="coral")
    ax.bar(x + w, both_dl, w, label="Both",       color="mediumpurple")
    ax.set_xlabel("Layer"); ax.set_ylabel("Δ loss")
    ax.set_title("Layer Knockout: Loss Increase per Layer")
    ax.set_xticks(x); ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    # 1b: Cumulative ablation curve
    ax = axes[0, 1]
    ns = [c["n_ablated"] for c in cumul]
    ppl_r = [c["ppl_ratio"] for c in cumul]
    ax.plot(ns, ppl_r, "o-", markersize=6, color="darkred")
    ax.axhline(2.0, color="gray", ls="--", alpha=0.5, label="2× baseline")
    ax.axhline(5.0, color="gray", ls=":",  alpha=0.5, label="5× baseline")
    ax.set_xlabel("# Layers Ablated (least → most important)")
    ax.set_ylabel("PPL / Baseline PPL")
    ax.set_title("Cumulative Layer Ablation")
    ax.set_xticks(range(0, n_layers + 1))
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # 2a: Top residual stream components
    ax = axes[1, 0]
    show_n = min(15, len(ranked))
    r_names = [n for n, _ in ranked[:show_n]]
    r_vals  = [v for _, v in ranked[:show_n]]
    colors = []
    for n in r_names:
        if "attn" in n:
            colors.append("steelblue")
        elif "mlp" in n:
            colors.append("coral")
        else:
            colors.append("seagreen")
    ax.barh(range(show_n), r_vals, color=colors)
    ax.set_yticks(range(show_n)); ax.set_yticklabels(r_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Average |logit contribution|")
    ax.set_title("Top Residual Stream Components (DLA)")
    ax.grid(True, alpha=0.3, axis="x")

    # 2b: Top-K explained curve
    ax = axes[1, 1]
    ks = sorted(top_k_buckets.keys())
    fracs = [avg_top_k[str(k)] for k in ks]
    ax.plot(ks, fracs, "s-", markersize=8, color="darkgreen")
    ax.axhline(0.7, color="red",    ls="--", alpha=0.6, label="Wang 2022 circuit (70%)")
    ax.axhline(0.9, color="orange", ls="--", alpha=0.6, label="90% threshold")
    ax.set_xlabel("K (# components)")
    ax.set_ylabel("Fraction of |logit| explained")
    ax.set_title("Cumulative Logit Attribution")
    ax.set_xticks(ks); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.suptitle("Exp-010 Phase 4: Circuit Dark Matter Exploration (WikiText-2)",
                 fontsize=14)
    fig.tight_layout()
    fig_path = out / "dark_matter_v2.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure: {fig_path}")

    # Save JSON
    json_path = out / "dark_matter_v2.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results: {json_path}")
    print("Done!")


if __name__ == "__main__":
    main()
