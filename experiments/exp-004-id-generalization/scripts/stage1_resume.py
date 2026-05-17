"""
Exp-004 Stage 1 Resume: Run remaining weight decays (1e-3, 5e-3, 1e-2).
Loads previously completed results and appends new ones.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

from stage1_baseline import train_and_evaluate, twonn_id
import torch
import numpy as np
import json
import gc
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent.parent

# Previously completed results (12 runs: wd=0, 1e-5, 1e-4, 5e-4)
completed = [
    {"weight_decay": 0, "seed": 0, "test_acc": 0.9228, "twonn_id": 17.9, "param_norm": 72.4},
    {"weight_decay": 0, "seed": 1, "test_acc": 0.9228, "twonn_id": 18.8, "param_norm": 71.1},
    {"weight_decay": 0, "seed": 2, "test_acc": 0.9215, "twonn_id": 18.6, "param_norm": 70.8},
    {"weight_decay": 1e-5, "seed": 0, "test_acc": 0.9218, "twonn_id": 17.9, "param_norm": 70.3},
    {"weight_decay": 1e-5, "seed": 1, "test_acc": 0.9168, "twonn_id": 18.0, "param_norm": 70.2},
    {"weight_decay": 1e-5, "seed": 2, "test_acc": 0.9229, "twonn_id": 18.2, "param_norm": 70.1},
    {"weight_decay": 1e-4, "seed": 0, "test_acc": 0.9256, "twonn_id": 15.5, "param_norm": 68.9},
    {"weight_decay": 1e-4, "seed": 1, "test_acc": 0.9273, "twonn_id": 15.5, "param_norm": 68.9},
    {"weight_decay": 1e-4, "seed": 2, "test_acc": 0.9293, "twonn_id": 14.8, "param_norm": 68.6},
    {"weight_decay": 5e-4, "seed": 0, "test_acc": 0.9373, "twonn_id": 10.3, "param_norm": 31.0},
    {"weight_decay": 5e-4, "seed": 1, "test_acc": 0.9415, "twonn_id": 10.0, "param_norm": 31.0},
    {"weight_decay": 5e-4, "seed": 2, "test_acc": 0.0, "twonn_id": 0.0, "param_norm": 0.0},  # placeholder
]


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    remaining_wds = [5e-4, 1e-3, 5e-3, 1e-2]
    # 5e-4 seed=2 was not completed, so include it
    n_seeds = 3
    n_epochs = 50

    results = [r for r in completed if not (r["weight_decay"] == 5e-4 and r["seed"] == 2)]

    for wd in remaining_wds:
        for seed in range(n_seeds):
            # Skip already completed
            if any(r["weight_decay"] == wd and r["seed"] == seed for r in results):
                continue
            print(f"\n{'='*50}")
            print(f"Weight decay={wd}, seed={seed}")
            print(f"{'='*50}")
            r = train_and_evaluate(wd, seed, n_epochs=n_epochs, device=device)
            print(f"  Test acc: {r['test_acc']:.4f}")
            print(f"  TwoNN ID: {r['twonn_id']:.1f}")
            print(f"  Param norm: {r['param_norm']:.1f}")
            results.append(r)
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()

    # Summary
    all_wds = [0, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'WD':>8} {'Acc':>8} {'ID':>8} {'‖θ‖':>8}")
    print("-" * 36)

    for wd in all_wds:
        wd_results = [r for r in results if r["weight_decay"] == wd]
        if not wd_results:
            continue
        acc_mean = np.mean([r["test_acc"] for r in wd_results])
        id_mean = np.mean([r["twonn_id"] for r in wd_results])
        norm_mean = np.mean([r["param_norm"] for r in wd_results])
        print(f"{wd:>8.0e} {acc_mean:>8.4f} {id_mean:>8.1f} {norm_mean:>8.1f}")

    from scipy.stats import pearsonr
    accs = [r["test_acc"] for r in results]
    ids = [r["twonn_id"] for r in results]
    r_val, p_val = pearsonr(ids, accs)
    print(f"\nPearson r(ID, acc) = {r_val:.3f}, p = {p_val:.4f}")

    # Save
    with open(EXP_DIR / "results_stage1.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to results_stage1.json")


if __name__ == "__main__":
    main()
