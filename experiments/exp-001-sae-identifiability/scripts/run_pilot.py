"""
Exp-001: SAE 可识别性 Pilot 实验
TopK Sparse Autoencoder 实现 + 训练 + 跨种子特征比较

对应实验记录: experiments/exp-001-sae-identifiability/plan.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import argparse
from tqdm import tqdm


# ============================================================
# TopK Sparse Autoencoder
# ============================================================

class TopKSAE(nn.Module):
    def __init__(self, d_model: int, d_sae: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k

        self.encoder = nn.Linear(d_model, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_model, bias=True)

        # Initialize decoder columns to unit norm
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def forward(self, x):
        # Encode
        z_pre = self.encoder(x)

        # TopK activation
        topk_vals, topk_idx = torch.topk(z_pre, self.k, dim=-1)
        z = torch.zeros_like(z_pre)
        z.scatter_(-1, topk_idx, F.relu(topk_vals))

        # Decode
        x_hat = self.decoder(z)

        return x_hat, z, z_pre

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def get_decoder_directions(self) -> torch.Tensor:
        """Return decoder column directions (d_model, d_sae), unit normalized."""
        return F.normalize(self.decoder.weight.data, dim=0)


# ============================================================
# Activation caching
# ============================================================

def cache_activations(
    model_name: str,
    layer_idx: int,
    n_tokens: int,
    batch_size: int,
    seq_len: int,
    save_path: Path,
    device: str = "mps",
):
    """Cache MLP output activations from a transformer model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    ).to(device)
    model.eval()

    print(f"Loading dataset...")
    ds = load_dataset("EleutherAI/the_pile_deduplicated", split="train", streaming=True)

    activations = []
    total_collected = 0
    hook_output = {}

    # Hook to capture MLP output
    def hook_fn(module, input, output):
        hook_output["act"] = output.detach()

    # Register hook on MLP of target layer
    layer = model.gpt_neox.layers[layer_idx].mlp
    handle = layer.register_forward_hook(hook_fn)

    print(f"Caching activations from layer {layer_idx} MLP...")
    texts = []
    for example in ds:
        texts.append(example["text"])
        if len(texts) >= batch_size:
            tokens = tokenizer(
                texts, return_tensors="pt", max_length=seq_len,
                truncation=True, padding="max_length"
            ).to(device)

            with torch.no_grad():
                model(**tokens)

            act = hook_output["act"].reshape(-1, hook_output["act"].shape[-1])
            activations.append(act.cpu())
            total_collected += act.shape[0]
            texts = []

            if total_collected >= n_tokens:
                break

    handle.remove()
    del model
    torch.mps.empty_cache() if device == "mps" else None

    all_acts = torch.cat(activations, dim=0)[:n_tokens]
    print(f"Cached {all_acts.shape[0]} activation vectors, shape: {all_acts.shape}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_acts, save_path)
    print(f"Saved to {save_path}")
    return all_acts


# ============================================================
# SAE Training
# ============================================================

def train_sae(
    activations: torch.Tensor,
    d_sae: int,
    k: int,
    seed: int,
    n_epochs: int = 1,
    batch_size: int = 4096,
    lr: float = 3e-4,
    device: str = "mps",
    save_path: Path = None,
):
    """Train a TopK SAE on cached activations."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    d_model = activations.shape[1]
    sae = TopKSAE(d_model, d_sae, k).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    n_samples = activations.shape[0]
    n_batches = n_samples // batch_size

    losses = []

    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0

        pbar = tqdm(range(n_batches), desc=f"Seed {seed}, Epoch {epoch+1}/{n_epochs}")
        for i in pbar:
            idx = perm[i * batch_size : (i + 1) * batch_size]
            x = activations[idx].to(device)

            x_hat, z, _ = sae(x)
            loss = F.mse_loss(x_hat, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Keep decoder columns normalized
            sae.normalize_decoder()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        print(f"  Epoch {epoch+1} avg loss: {avg_loss:.6f}")

    # Compute final stats
    with torch.no_grad():
        x_sample = activations[:min(50000, n_samples)].to(device)
        x_hat, z, _ = sae(x_sample)
        nmse = F.mse_loss(x_hat, x_sample).item() / x_sample.var().item()
        l0 = (z > 0).float().sum(dim=-1).mean().item()

    stats = {
        "seed": seed,
        "d_model": d_model,
        "d_sae": d_sae,
        "k": k,
        "nmse": nmse,
        "mean_l0": l0,
        "final_loss": losses[-1],
        "n_tokens": n_samples,
    }
    print(f"  NMSE: {nmse:.4f}, Mean L0: {l0:.1f}")

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": sae.state_dict(),
            "stats": stats,
        }, save_path)
        print(f"  Saved to {save_path}")

    return sae, stats


# ============================================================
# Feature comparison metrics
# ============================================================

def compute_mmcs(dict1: torch.Tensor, dict2: torch.Tensor) -> dict:
    """
    Compute Mean Max Cosine Similarity between two dictionaries.
    dict1, dict2: (d_model, d_sae) normalized decoder weights.
    """
    # Cosine similarity matrix: (d_sae1, d_sae2)
    cos_sim = dict1.T @ dict2  # both columns already unit-normalized

    # For each feature in dict1, find max similarity in dict2
    max_sim_1to2, _ = cos_sim.max(dim=1)
    # For each feature in dict2, find max similarity in dict1
    max_sim_2to1, _ = cos_sim.max(dim=0)

    return {
        "mmcs_1to2": max_sim_1to2.mean().item(),
        "mmcs_2to1": max_sim_2to1.mean().item(),
        "mmcs_mean": (max_sim_1to2.mean().item() + max_sim_2to1.mean().item()) / 2,
        "max_sim_1to2_dist": max_sim_1to2.cpu().numpy(),
        "max_sim_2to1_dist": max_sim_2to1.cpu().numpy(),
    }


def compute_hungarian_sharing(dict1: torch.Tensor, dict2: torch.Tensor, threshold: float = 0.7) -> dict:
    """
    Compute feature sharing rate via Hungarian optimal matching.
    """
    from scipy.optimize import linear_sum_assignment

    cos_sim = (dict1.T @ dict2).cpu().numpy()
    # Hungarian maximization = minimize negative
    row_ind, col_ind = linear_sum_assignment(-cos_sim)
    matched_sims = cos_sim[row_ind, col_ind]

    shared_fraction = (matched_sims > threshold).mean()
    mean_matched_sim = matched_sims.mean()

    return {
        "shared_fraction": float(shared_fraction),
        "mean_matched_sim": float(mean_matched_sim),
        "threshold": threshold,
        "matched_sims": matched_sims,
    }


def compute_mutual_nn(dict1: torch.Tensor, dict2: torch.Tensor, threshold: float = 0.7) -> dict:
    """
    Mutual nearest neighbors: pair counted only if i is j's best AND j is i's best.
    """
    cos_sim = dict1.T @ dict2
    nn_1to2 = cos_sim.argmax(dim=1)
    nn_2to1 = cos_sim.argmax(dim=0)

    mutual = 0
    mutual_above_threshold = 0
    d_sae = dict1.shape[1]

    for i in range(d_sae):
        j = nn_1to2[i].item()
        if nn_2to1[j].item() == i:
            mutual += 1
            if cos_sim[i, j].item() > threshold:
                mutual_above_threshold += 1

    return {
        "mutual_nn_count": mutual,
        "mutual_nn_fraction": mutual / d_sae,
        "mutual_nn_above_threshold": mutual_above_threshold / d_sae,
        "threshold": threshold,
    }


def compare_all_pairs(sae_paths: list[Path], device: str = "mps") -> list[dict]:
    """Load all SAEs and compute pairwise metrics."""
    # Load decoder directions
    decoders = {}
    stats_all = {}
    for path in sae_paths:
        ckpt = torch.load(path, map_location=device, weights_only=True)
        seed = ckpt["stats"]["seed"]
        state = ckpt["model_state_dict"]
        dec_weight = F.normalize(state["decoder.weight"], dim=0)
        decoders[seed] = dec_weight
        stats_all[seed] = ckpt["stats"]

    seeds = sorted(decoders.keys())
    results = []

    for i, s1 in enumerate(seeds):
        for s2 in seeds[i+1:]:
            print(f"\nComparing seed {s1} vs seed {s2}:")
            d1, d2 = decoders[s1], decoders[s2]

            mmcs = compute_mmcs(d1, d2)
            hungarian = compute_hungarian_sharing(d1, d2, threshold=0.7)
            mutual = compute_mutual_nn(d1, d2, threshold=0.7)

            result = {
                "seed_1": s1,
                "seed_2": s2,
                "mmcs_mean": mmcs["mmcs_mean"],
                "hungarian_shared_0.7": hungarian["shared_fraction"],
                "hungarian_mean_sim": hungarian["mean_matched_sim"],
                "mutual_nn_fraction": mutual["mutual_nn_fraction"],
                "mutual_nn_above_0.7": mutual["mutual_nn_above_threshold"],
            }
            results.append(result)

            print(f"  MMCS: {mmcs['mmcs_mean']:.4f}")
            print(f"  Hungarian shared (>0.7): {hungarian['shared_fraction']:.2%}")
            print(f"  Mutual NN fraction: {mutual['mutual_nn_fraction']:.2%}")

    return results, stats_all


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="SAE Identifiability Pilot")
    parser.add_argument("--step", choices=["cache", "train", "compare", "all"], default="all")
    parser.add_argument("--model", default="EleutherAI/pythia-160m-deduped")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--n-tokens", type=int, default=2_000_000)
    parser.add_argument("--d-sae", type=int, default=4096)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=0, help="Starting seed index")
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    exp_dir = Path("experiments/exp-001-sae-identifiability")
    cache_path = exp_dir / "data" / f"activations_layer{args.layer}.pt"
    sae_dir = exp_dir / "checkpoints"

    if args.step in ("cache", "all"):
        cache_activations(
            model_name=args.model,
            layer_idx=args.layer,
            n_tokens=args.n_tokens,
            batch_size=8,
            seq_len=1024,
            save_path=cache_path,
            device=args.device,
        )

    if args.step in ("train", "all"):
        activations = torch.load(cache_path, weights_only=True)
        print(f"Loaded activations: {activations.shape}")

        for seed in range(args.seed_offset, args.seed_offset + args.n_seeds):
            print(f"\n{'='*60}")
            print(f"Training SAE with seed {seed}")
            print(f"{'='*60}")
            train_sae(
                activations=activations,
                d_sae=args.d_sae,
                k=args.k,
                seed=seed,
                n_epochs=args.n_epochs,
                batch_size=args.batch_size,
                device=args.device,
                save_path=sae_dir / f"sae_seed{seed}.pt",
            )

    if args.step in ("compare", "all"):
        sae_paths = sorted(sae_dir.glob("sae_seed*.pt"))
        if len(sae_paths) < 2:
            print("Need at least 2 SAEs to compare.")
            return

        results, stats = compare_all_pairs(sae_paths, device=args.device)

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        mmcs_vals = [r["mmcs_mean"] for r in results]
        shared_vals = [r["hungarian_shared_0.7"] for r in results]
        mutual_vals = [r["mutual_nn_fraction"] for r in results]

        print(f"MMCS:              {np.mean(mmcs_vals):.4f} ± {np.std(mmcs_vals):.4f}")
        print(f"Hungarian (>0.7):  {np.mean(shared_vals):.2%} ± {np.std(shared_vals):.2%}")
        print(f"Mutual NN:         {np.mean(mutual_vals):.2%} ± {np.std(mutual_vals):.2%}")

        # Save results
        summary = {
            "config": vars(args),
            "pairwise_results": [{k: v for k, v in r.items() if not isinstance(v, np.ndarray)} for r in results],
            "summary": {
                "mmcs_mean": float(np.mean(mmcs_vals)),
                "mmcs_std": float(np.std(mmcs_vals)),
                "hungarian_shared_mean": float(np.mean(shared_vals)),
                "hungarian_shared_std": float(np.std(shared_vals)),
                "mutual_nn_mean": float(np.mean(mutual_vals)),
                "mutual_nn_std": float(np.std(mutual_vals)),
            },
            "per_seed_stats": {str(k): v for k, v in stats.items()},
        }

        results_path = exp_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
