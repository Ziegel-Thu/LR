"""
Exp-002: 量化临界 bit-width Pilot
对 Pythia-160M 做不同 bit-width 量化，用已训好的 SAE 测特征恢复率

对应: experiments/exp-002-quantization-phase/plan.md
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "exp-001-sae-identifiability" / "scripts"))
from run_pilot import TopKSAE


def get_quantized_activations(
    model_name: str,
    layer_idx: int,
    n_tokens: int,
    nbits: int,
    group_size: int = 64,
    device: str = "mps",
) -> torch.Tensor:
    """Get MLP activations from a quantized model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"\n{'='*50}")
    print(f"Quantizing to {nbits} bits (group_size={group_size})")
    print(f"{'='*50}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if nbits >= 16:
        # FP32/FP16 baseline
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        ).to(device)
    else:
        from hqq.models.hf.base import AutoHQQHFModel
        from hqq.core.quantize import BaseQuantizeConfig

        quant_config = BaseQuantizeConfig(
            nbits=nbits, group_size=group_size, axis=1
        )
        model = AutoHQQHFModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
        model.quantize_model(quant_config=quant_config)
        model = model.to(device)

    model.eval()

    ds = load_dataset("EleutherAI/the_pile_deduplicated", split="train", streaming=True)

    activations = []
    total = 0
    hook_output = {}

    def hook_fn(module, input, output):
        hook_output["act"] = output.detach()

    layer = model.gpt_neox.layers[layer_idx].mlp
    handle = layer.register_forward_hook(hook_fn)

    texts = []
    for example in ds:
        texts.append(example["text"])
        if len(texts) >= 4:
            tokens = tokenizer(
                texts, return_tensors="pt", max_length=1024,
                truncation=True, padding="max_length"
            ).to(device)

            with torch.no_grad():
                model(**tokens)

            act = hook_output["act"].reshape(-1, hook_output["act"].shape[-1])
            activations.append(act.cpu().float())
            total += act.shape[0]
            texts = []

            if total >= n_tokens:
                break

    handle.remove()
    del model
    if device == "mps":
        torch.mps.empty_cache()

    result = torch.cat(activations, dim=0)[:n_tokens]
    print(f"  Collected {result.shape[0]} activations")
    return result


def analyze_feature_recovery(
    sae: TopKSAE,
    fp32_acts: torch.Tensor,
    quant_acts: torch.Tensor,
    nbits: int,
) -> dict:
    """Compare SAE features between FP32 and quantized activations."""
    sae.eval()
    n = min(len(fp32_acts), len(quant_acts), 50000)
    fp32_sample = fp32_acts[:n]
    quant_sample = quant_acts[:n]

    with torch.no_grad():
        # Get SAE activations for both
        _, z_fp32, _ = sae(fp32_sample)
        _, z_quant, _ = sae(quant_sample)

        # Reconstruction quality
        x_hat_fp32, _, _ = sae(fp32_sample)
        x_hat_quant, _, _ = sae(quant_sample)
        nmse_fp32 = F.mse_loss(x_hat_fp32, fp32_sample).item() / fp32_sample.var().item()
        nmse_quant = F.mse_loss(x_hat_quant, quant_sample).item() / quant_sample.var().item()

    # Feature-wise analysis
    z_fp32_np = z_fp32.numpy()
    z_quant_np = z_quant.numpy()

    # Per-feature activation correlation
    correlations = []
    fp32_firing_rates = []
    for j in range(sae.d_sae):
        f_fp32 = z_fp32_np[:, j]
        f_quant = z_quant_np[:, j]

        fp32_rate = (f_fp32 > 0).mean()
        fp32_firing_rates.append(fp32_rate)

        if f_fp32.std() > 1e-8 and f_quant.std() > 1e-8:
            r = np.corrcoef(f_fp32, f_quant)[0, 1]
            correlations.append(r if not np.isnan(r) else 0.0)
        else:
            correlations.append(0.0)

    correlations = np.array(correlations)
    fp32_firing_rates = np.array(fp32_firing_rates)

    # Alive features (firing rate > 0 in FP32)
    alive_mask = fp32_firing_rates > 1e-5
    n_alive = alive_mask.sum()

    # Feature recovery by threshold
    alive_corrs = correlations[alive_mask]

    # Stratify by firing rate
    if n_alive > 0:
        rates = fp32_firing_rates[alive_mask]
        corrs = correlations[alive_mask]

        # Split into terciles
        tercile_bounds = np.percentile(rates[rates > 0], [33, 67])
        low_mask = rates <= tercile_bounds[0]
        mid_mask = (rates > tercile_bounds[0]) & (rates <= tercile_bounds[1])
        high_mask = rates > tercile_bounds[1]

        strat = {
            "low_freq_mean_corr": float(corrs[low_mask].mean()) if low_mask.sum() > 0 else 0,
            "mid_freq_mean_corr": float(corrs[mid_mask].mean()) if mid_mask.sum() > 0 else 0,
            "high_freq_mean_corr": float(corrs[high_mask].mean()) if high_mask.sum() > 0 else 0,
            "low_freq_count": int(low_mask.sum()),
            "mid_freq_count": int(mid_mask.sum()),
            "high_freq_count": int(high_mask.sum()),
        }
    else:
        strat = {}

    result = {
        "nbits": nbits,
        "nmse_fp32": float(nmse_fp32),
        "nmse_quant": float(nmse_quant),
        "nmse_ratio": float(nmse_quant / nmse_fp32) if nmse_fp32 > 0 else float('inf'),
        "n_alive_fp32": int(n_alive),
        "mean_correlation": float(alive_corrs.mean()) if n_alive > 0 else 0,
        "median_correlation": float(np.median(alive_corrs)) if n_alive > 0 else 0,
        "frac_corr_above_0.8": float((alive_corrs > 0.8).mean()) if n_alive > 0 else 0,
        "frac_corr_above_0.5": float((alive_corrs > 0.5).mean()) if n_alive > 0 else 0,
        "frac_corr_above_0.3": float((alive_corrs > 0.3).mean()) if n_alive > 0 else 0,
        "frequency_stratified": strat,
    }

    print(f"\n  [{nbits}-bit] NMSE ratio: {result['nmse_ratio']:.3f}")
    print(f"  [{nbits}-bit] Mean correlation (alive): {result['mean_correlation']:.4f}")
    print(f"  [{nbits}-bit] Features with r>0.8: {result['frac_corr_above_0.8']:.1%}")
    print(f"  [{nbits}-bit] Features with r>0.5: {result['frac_corr_above_0.5']:.1%}")
    if strat:
        print(f"  [{nbits}-bit] By frequency — low: {strat['low_freq_mean_corr']:.3f}, "
              f"mid: {strat['mid_freq_mean_corr']:.3f}, high: {strat['high_freq_mean_corr']:.3f}")

    return result


def main():
    import os
    os.environ.pop("HF_ENDPOINT", None)

    exp_dir = Path(__file__).resolve().parent.parent
    sae_path = exp_dir.parent / "exp-001-sae-identifiability" / "checkpoints" / "d4096_k32" / "sae_seed0.pt"

    # Check if Run 1 SAE exists
    if not sae_path.exists():
        # Try old path
        sae_path = exp_dir.parent / "exp-001-sae-identifiability" / "checkpoints" / "sae_seed0.pt"
    if not sae_path.exists():
        print(f"ERROR: SAE not found at {sae_path}")
        print("Run exp-001 first to train a reference SAE.")
        return

    print(f"Loading SAE from {sae_path}")
    ckpt = torch.load(sae_path, map_location="cpu", weights_only=True)
    sae = TopKSAE(768, 4096, 32)
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()

    model_name = "EleutherAI/pythia-160m-deduped"
    layer_idx = 6
    n_tokens = 500_000

    # Get FP32 baseline activations
    print("Getting FP32 baseline activations...")
    fp32_acts = get_quantized_activations(model_name, layer_idx, n_tokens, nbits=32, device="mps")

    # Test each bit-width
    results = []
    for nbits in [8, 4, 3, 2]:
        try:
            quant_acts = get_quantized_activations(model_name, layer_idx, n_tokens, nbits=nbits, device="mps")
            result = analyze_feature_recovery(sae, fp32_acts, quant_acts, nbits)
            results.append(result)
            del quant_acts
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception as e:
            print(f"\n  [{nbits}-bit] FAILED: {e}")
            results.append({"nbits": nbits, "error": str(e)})

    # Add FP32 baseline
    fp32_result = analyze_feature_recovery(sae, fp32_acts, fp32_acts, nbits=32)
    results.insert(0, fp32_result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Feature Recovery by Bit-Width")
    print(f"{'='*60}")
    print(f"{'Bits':>6} {'NMSE ratio':>12} {'Mean r':>10} {'r>0.8':>8} {'r>0.5':>8} {'Low freq':>10} {'High freq':>10}")
    print("-" * 70)
    for r in results:
        if "error" in r:
            print(f"{r['nbits']:>6} ERROR: {r['error']}")
        else:
            strat = r.get("frequency_stratified", {})
            print(f"{r['nbits']:>6} {r['nmse_ratio']:>12.3f} {r['mean_correlation']:>10.4f} "
                  f"{r['frac_corr_above_0.8']:>8.1%} {r['frac_corr_above_0.5']:>8.1%} "
                  f"{strat.get('low_freq_mean_corr', 0):>10.3f} {strat.get('high_freq_mean_corr', 0):>10.3f}")

    # Save
    results_path = exp_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
