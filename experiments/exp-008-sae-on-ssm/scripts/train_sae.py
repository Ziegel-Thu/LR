#!/usr/bin/env python3
"""
Exp-008 Phase 1: Train a TopK Sparse Autoencoder on Mamba-130M activations.

Custom TopK SAE implementation (sae-lens doesn't natively support Mamba).
Architecture: encoder (d_model → d_sae), TopK activation, decoder (d_sae → d_model).

Usage:
  python train_sae.py --acts-dir /nvmessd/lifanhong/LR-env/exp008_acts \
                      --out-dir /nvmessd/lifanhong/LR-env/exp008_sae
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TopKSAE(nn.Module):
    """Sparse Autoencoder with TopK activation."""

    def __init__(self, d_model: int, d_sae: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k

        self.encoder = nn.Linear(d_model, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_model, bias=True)

        # Kaiming init
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.kaiming_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode → TopK. Returns (sparse_acts, pre_topk_acts)."""
        pre_acts = self.encoder(x)  # (batch, d_sae)
        pre_acts_relu = F.relu(pre_acts)

        # TopK: keep only top-k activations
        topk_vals, topk_idx = pre_acts_relu.topk(self.k, dim=-1)
        sparse_acts = torch.zeros_like(pre_acts_relu)
        sparse_acts.scatter_(1, topk_idx, topk_vals)

        return sparse_acts, pre_acts_relu

    def decode(self, sparse_acts: torch.Tensor) -> torch.Tensor:
        return self.decoder(sparse_acts)

    def forward(self, x: torch.Tensor) -> dict:
        sparse_acts, pre_acts = self.encode(x)
        x_hat = self.decode(sparse_acts)

        mse = F.mse_loss(x_hat, x)
        # L1 on pre-TopK for auxiliary sparsity pressure
        l1 = pre_acts.abs().mean()

        return {
            "x_hat": x_hat,
            "sparse_acts": sparse_acts,
            "pre_acts": pre_acts,
            "mse": mse,
            "l1": l1,
        }


def load_activations(acts_dir: Path) -> torch.Tensor:
    """Load and concatenate all activation chunks."""
    chunks = sorted(acts_dir.glob("acts_*.pt"))
    if not chunks:
        raise FileNotFoundError(f"No activation chunks found in {acts_dir}")

    all_acts = []
    for cp in chunks:
        acts = torch.load(cp, map_location="cpu", weights_only=True)
        all_acts.append(acts)
        print(f"  Loaded {cp.name}: {acts.shape}")

    combined = torch.cat(all_acts, dim=0)
    print(f"  Total: {combined.shape}")
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acts-dir", required=True, help="Dir with cached activation chunks")
    parser.add_argument("--out-dir", required=True, help="Output dir for SAE checkpoint")
    parser.add_argument("--d-sae", type=int, default=4096)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--l1-coeff", type=float, default=1e-3)
    parser.add_argument("--n-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    acts_dir = Path(args.acts_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    meta_path = acts_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        d_model = meta["d_model"]
        print(f"d_model={d_model} (from metadata)")
    else:
        d_model = None

    # Load activations
    print("Loading activations...", flush=True)
    acts = load_activations(acts_dir)

    if d_model is None:
        d_model = acts.shape[1]
        print(f"d_model={d_model} (from data)")

    # Normalize: zero mean, unit variance per dimension
    acts_mean = acts.mean(dim=0)
    acts_std = acts.std(dim=0) + 1e-8
    acts = (acts - acts_mean) / acts_std

    # Create model
    sae = TopKSAE(d_model, args.d_sae, args.k).to(args.device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr)
    print(f"SAE: d_model={d_model} → d_sae={args.d_sae}, K={args.k}", flush=True)
    print(f"  Parameters: {sum(p.numel() for p in sae.parameters()):,}", flush=True)

    # Training
    dataset = TensorDataset(acts)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True, drop_last=True)

    losses_mse = []
    losses_l1 = []
    dead_feature_pcts = []
    step = 0
    t0 = time.time()

    feature_activation_counts = torch.zeros(args.d_sae, device=args.device)

    print(f"\nTraining for {args.n_steps} steps...", flush=True)
    while step < args.n_steps:
        for (batch,) in dataloader:
            if step >= args.n_steps:
                break

            batch = batch.to(args.device)
            out = sae(batch)

            loss = out["mse"] + args.l1_coeff * out["l1"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track feature activations
            with torch.no_grad():
                active = (out["sparse_acts"] > 0).any(dim=0).float()
                feature_activation_counts += active

            step += 1

            if step % args.log_every == 0:
                mse_val = out["mse"].item()
                l1_val = out["l1"].item()
                dead_pct = (feature_activation_counts == 0).float().mean().item() * 100
                losses_mse.append(mse_val)
                losses_l1.append(l1_val)
                dead_feature_pcts.append(dead_pct)

                elapsed = time.time() - t0
                steps_per_sec = step / elapsed
                print(f"  Step {step:>6d} | MSE={mse_val:.6f} | L1={l1_val:.4f} | "
                      f"Dead={dead_pct:.1f}% | {steps_per_sec:.1f} steps/s", flush=True)

            if step % args.save_every == 0:
                ckpt_path = out_dir / f"sae_step{step}.pt"
                torch.save({
                    "state_dict": sae.state_dict(),
                    "step": step,
                    "config": {"d_model": d_model, "d_sae": args.d_sae, "k": args.k},
                    "acts_mean": acts_mean,
                    "acts_std": acts_std,
                }, ckpt_path)
                print(f"  Checkpoint: {ckpt_path}", flush=True)

    # Final save
    final_path = out_dir / "sae_final.pt"
    torch.save({
        "state_dict": sae.state_dict(),
        "step": step,
        "config": {"d_model": d_model, "d_sae": args.d_sae, "k": args.k},
        "acts_mean": acts_mean,
        "acts_std": acts_std,
    }, final_path)
    print(f"\nFinal checkpoint: {final_path}", flush=True)

    # Save training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(losses_mse)
    axes[0].set_xlabel("Log step (×{})".format(args.log_every))
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Reconstruction Loss")
    axes[0].set_yscale("log")

    axes[1].plot(losses_l1)
    axes[1].set_xlabel("Log step (×{})".format(args.log_every))
    axes[1].set_ylabel("L1 Loss")
    axes[1].set_title("Sparsity Loss")

    axes[2].plot(dead_feature_pcts)
    axes[2].set_xlabel("Log step (×{})".format(args.log_every))
    axes[2].set_ylabel("Dead Features %")
    axes[2].set_title("Dead Feature Rate")

    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150)
    print(f"Training curves: {out_dir / 'training_curves.png'}")

    # Save training log
    log = {
        "config": {
            "d_model": d_model,
            "d_sae": args.d_sae,
            "k": args.k,
            "lr": args.lr,
            "l1_coeff": args.l1_coeff,
            "n_steps": step,
            "batch_size": args.batch_size,
        },
        "final_mse": losses_mse[-1] if losses_mse else None,
        "final_l1": losses_l1[-1] if losses_l1 else None,
        "final_dead_pct": dead_feature_pcts[-1] if dead_feature_pcts else None,
        "losses_mse": losses_mse,
        "dead_feature_pcts": dead_feature_pcts,
    }
    with open(out_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"Training log: {out_dir / 'training_log.json'}")


if __name__ == "__main__":
    main()
