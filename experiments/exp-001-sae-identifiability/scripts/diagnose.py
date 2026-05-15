"""
Exp-001 诊断脚本：分析为什么跨种子共享率这么低
1. Dead feature 比例
2. MCS 分布直方图（是否双峰）
3. 特征激活频率分布
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

exp_dir = Path(__file__).resolve().parent.parent
sae_dir = exp_dir / "checkpoints"
fig_dir = exp_dir / "figures"
fig_dir.mkdir(exist_ok=True)

# Load activations
act_path = exp_dir / "data" / "activations_layer6.pt"
activations = torch.load(act_path, weights_only=True)
print(f"Activations shape: {activations.shape}")

# Load all SAEs
saes = {}
for path in sorted(sae_dir.glob("sae_seed*.pt")):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    seed = ckpt["stats"]["seed"]
    saes[seed] = ckpt
    print(f"Loaded seed {seed}: NMSE={ckpt['stats']['nmse']:.4f}")

# ============================================================
# 1. Dead feature analysis
# ============================================================
print("\n" + "="*60)
print("1. DEAD FEATURE ANALYSIS")
print("="*60)

from run_pilot import TopKSAE

dead_rates = {}
for seed, ckpt in saes.items():
    sae = TopKSAE(768, 4096, 32)
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()
    
    # Run a batch through and check which features ever activate
    n_check = min(100000, activations.shape[0])
    batch_size = 4096
    ever_active = torch.zeros(4096, dtype=torch.bool)
    
    with torch.no_grad():
        for i in range(0, n_check, batch_size):
            x = activations[i:i+batch_size]
            _, z, _ = sae(x)
            ever_active |= (z > 0).any(dim=0)
    
    n_dead = (~ever_active).sum().item()
    dead_rate = n_dead / 4096
    dead_rates[seed] = dead_rate
    print(f"Seed {seed}: {n_dead}/4096 dead features ({dead_rate:.1%})")

mean_dead = np.mean(list(dead_rates.values()))
print(f"\nAverage dead rate: {mean_dead:.1%}")
print(f"Alive features per SAE: ~{4096 * (1 - mean_dead):.0f}")

# ============================================================
# 2. MCS distribution (take seed 0 vs seed 1 as example)
# ============================================================
print("\n" + "="*60)
print("2. MCS DISTRIBUTION")
print("="*60)

d0 = F.normalize(saes[0]["model_state_dict"]["decoder.weight"], dim=0)
d1 = F.normalize(saes[1]["model_state_dict"]["decoder.weight"], dim=0)

cos_sim = d0.T @ d1  # (4096, 4096)
max_cos_0to1, _ = cos_sim.max(dim=1)
max_cos_1to0, _ = cos_sim.max(dim=0)

mcs = max_cos_0to1.numpy()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(mcs, bins=100, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.3)
axes[0].set_xlabel("Max Cosine Similarity")
axes[0].set_ylabel("Count")
axes[0].set_title("MCS Distribution (Seed 0 → Seed 1)")
axes[0].axvline(x=0.7, color='red', linestyle='--', label='threshold=0.7')
axes[0].axvline(x=np.mean(mcs), color='orange', linestyle='--', label=f'mean={np.mean(mcs):.3f}')
axes[0].legend()

# Also show the distribution of ALL pairwise cosine sims (random sample)
rand_idx = np.random.choice(4096, 500, replace=False)
rand_cos = cos_sim[rand_idx][:, rand_idx].numpy().flatten()
axes[1].hist(rand_cos, bins=100, alpha=0.7, color='gray', edgecolor='black', linewidth=0.3)
axes[1].set_xlabel("Cosine Similarity")
axes[1].set_ylabel("Count")
axes[1].set_title("All Pairwise Cosine Sims (500×500 sample)")

plt.tight_layout()
plt.savefig(fig_dir / "mcs_distribution.png", dpi=150)
print(f"Saved figure to {fig_dir / 'mcs_distribution.png'}")

# Stats
print(f"MCS stats: mean={np.mean(mcs):.4f}, median={np.median(mcs):.4f}, "
      f"std={np.std(mcs):.4f}, max={np.max(mcs):.4f}")
print(f"Features with MCS > 0.7: {(mcs > 0.7).sum()}")
print(f"Features with MCS > 0.5: {(mcs > 0.5).sum()}")
print(f"Features with MCS > 0.3: {(mcs > 0.3).sum()}")

# ============================================================
# 3. Alive-only sharing rate
# ============================================================
print("\n" + "="*60)
print("3. ALIVE-ONLY SHARING RATE")
print("="*60)

# Recompute sharing rate excluding dead features
sae0 = TopKSAE(768, 4096, 32)
sae0.load_state_dict(saes[0]["model_state_dict"])
sae0.eval()
sae1 = TopKSAE(768, 4096, 32)
sae1.load_state_dict(saes[1]["model_state_dict"])
sae1.eval()

alive0 = torch.zeros(4096, dtype=torch.bool)
alive1 = torch.zeros(4096, dtype=torch.bool)

with torch.no_grad():
    n_check = min(100000, activations.shape[0])
    for i in range(0, n_check, 4096):
        x = activations[i:i+4096]
        _, z0, _ = sae0(x)
        _, z1, _ = sae1(x)
        alive0 |= (z0 > 0).any(dim=0)
        alive1 |= (z1 > 0).any(dim=0)

n_alive0 = alive0.sum().item()
n_alive1 = alive1.sum().item()

# Restrict to alive features
d0_alive = F.normalize(saes[0]["model_state_dict"]["decoder.weight"][:, alive0], dim=0)
d1_alive = F.normalize(saes[1]["model_state_dict"]["decoder.weight"][:, alive1], dim=0)

cos_alive = d0_alive.T @ d1_alive
max_cos_alive, _ = cos_alive.max(dim=1)

above_07 = (max_cos_alive > 0.7).sum().item()
above_05 = (max_cos_alive > 0.5).sum().item()

print(f"Alive features: seed0={n_alive0}, seed1={n_alive1}")
print(f"Alive MCS mean: {max_cos_alive.mean().item():.4f}")
print(f"Alive features with MCS > 0.7: {above_07}/{n_alive0} ({above_07/n_alive0:.1%})")
print(f"Alive features with MCS > 0.5: {above_05}/{n_alive0} ({above_05/n_alive0:.1%})")

# ============================================================
# Summary
# ============================================================
print("\n" + "="*60)
print("DIAGNOSIS SUMMARY")
print("="*60)

summary = {
    "dead_rates": dead_rates,
    "mean_dead_rate": float(mean_dead),
    "alive_per_sae": int(4096 * (1 - mean_dead)),
    "mcs_stats": {
        "mean": float(np.mean(mcs)),
        "median": float(np.median(mcs)),
        "std": float(np.std(mcs)),
        "max": float(np.max(mcs)),
        "above_0.7": int((mcs > 0.7).sum()),
        "above_0.5": int((mcs > 0.5).sum()),
        "above_0.3": int((mcs > 0.3).sum()),
    },
    "alive_only": {
        "alive_seed0": n_alive0,
        "alive_seed1": n_alive1,
        "alive_mcs_mean": float(max_cos_alive.mean()),
        "alive_above_0.7": above_07,
        "alive_above_0.5": above_05,
    }
}

with open(exp_dir / "diagnosis.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nDiagnosis saved to {exp_dir / 'diagnosis.json'}")
