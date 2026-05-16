"""
Exp-003 Phase 1: Pythia-2.8B / Mamba-2.8B / RWKV-4-3B 跨架构比较
服务器版本 (CUDA)

用法:
  python run_phase1.py                    # 全部跑
  python run_phase1.py --step extract     # 只提取
  python run_phase1.py --step compare     # 只比较（需要先提取）

对应: experiments/exp-003-cross-arch-platonic/phase1/plan.md
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse
import os
import gc

os.environ.pop("HF_ENDPOINT", None)

EXP_DIR = Path(__file__).resolve().parent.parent
PHASE_DIR = EXP_DIR / "phase1"
DATA_DIR = PHASE_DIR / "data"
FIG_DIR = PHASE_DIR / "figures"


def extract_representations(model_name, n_stimuli=5000, seq_len=256, device="cuda"):
    """Extract per-layer mean-pooled representations."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, output_hidden_states=True
    ).to(device)
    model.eval()

    print(f"Loading stimuli...")
    ds = load_dataset("wikitext", "wikitext-103-v1", split="validation")
    texts = [t for t in ds["text"] if len(t.strip()) > 50][:n_stimuli]
    print(f"Using {len(texts)} stimuli")

    all_hidden = None
    batch_size = 8

    for i in tqdm(range(0, len(texts), batch_size), desc=f"Extracting {model_name.split('/')[-1]}"):
        batch_texts = texts[i:i+batch_size]
        tokens = tokenizer(
            batch_texts, return_tensors="pt", max_length=seq_len,
            truncation=True, padding="max_length"
        ).to(device)

        with torch.no_grad():
            outputs = model(**tokens)

        hidden_states = outputs.hidden_states
        attention_mask = tokens["attention_mask"].unsqueeze(-1).float()

        batch_reps = []
        for h in hidden_states:
            pooled = (h.float() * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            batch_reps.append(pooled.cpu())

        batch_reps = torch.stack(batch_reps)  # (n_layers, batch, d)

        if all_hidden is None:
            all_hidden = batch_reps
        else:
            all_hidden = torch.cat([all_hidden, batch_reps], dim=1)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    print(f"Extracted: {all_hidden.shape} (layers, stimuli, d)")
    return all_hidden


def mutual_knn(X, Y, k=10):
    """Mutual k-NN overlap."""
    X_norm = F.normalize(torch.tensor(X, dtype=torch.float32), dim=1)
    Y_norm = F.normalize(torch.tensor(Y, dtype=torch.float32), dim=1)

    sim_XX = X_norm @ X_norm.T
    sim_YY = Y_norm @ Y_norm.T
    sim_XX.fill_diagonal_(-1)
    sim_YY.fill_diagonal_(-1)

    nn_X = sim_XX.topk(k, dim=1).indices
    nn_Y = sim_YY.topk(k, dim=1).indices

    overlaps = []
    for i in range(len(X)):
        set_x = set(nn_X[i].numpy())
        set_y = set(nn_Y[i].numpy())
        overlaps.append(len(set_x & set_y) / k)
    return np.mean(overlaps)


def linear_cka(X, Y):
    """Linear CKA."""
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    hsic_xy = (X @ X.T * (Y @ Y.T)).sum()
    hsic_xx = (X @ X.T * (X @ X.T)).sum()
    hsic_yy = (Y @ Y.T * (Y @ Y.T)).sum()
    return (hsic_xy / (hsic_xx.sqrt() * hsic_yy.sqrt())).item()


def compute_null(X, Y, metric_fn, n_perms=100, **kwargs):
    """Permutation null."""
    nulls = []
    for _ in range(n_perms):
        perm = np.random.permutation(len(Y))
        nulls.append(metric_fn(X, Y[perm], **kwargs))
    return np.array(nulls)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["extract", "compare", "all"], default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-stimuli", type=int, default=5000)
    parser.add_argument("--n-perms", type=int, default=100)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    models = {
        "Pythia-2.8B": "EleutherAI/pythia-2.8b-deduped",
        "Mamba-2.8B": "state-spaces/mamba-2.8b-hf",
        "RWKV-4-3B": "RWKV/rwkv-4-3b-pile",
    }

    # === Extract ===
    if args.step in ("extract", "all"):
        for name, model_id in models.items():
            cache_path = DATA_DIR / f"{name}_reps.pt"
            if cache_path.exists():
                print(f"Skipping {name} (cached)")
                continue
            reps = extract_representations(model_id, args.n_stimuli, device=args.device)
            torch.save(reps, cache_path)
            print(f"Saved {name} to {cache_path}")

    # === Compare ===
    if args.step in ("compare", "all"):
        reps = {}
        for name in models:
            cache_path = DATA_DIR / f"{name}_reps.pt"
            if not cache_path.exists():
                print(f"Missing {name}, run --step extract first")
                return
            reps[name] = torch.load(cache_path, weights_only=True)
            print(f"Loaded {name}: {reps[name].shape}")

        model_names = list(reps.keys())
        pairs = [(model_names[i], model_names[j])
                 for i in range(len(model_names))
                 for j in range(i+1, len(model_names))]

        results = {}
        for name_a, name_b in pairs:
            pair_key = f"{name_a} ↔ {name_b}"
            print(f"\n{'='*60}")
            print(f"Comparing {pair_key}")
            print(f"{'='*60}")

            reps_a = reps[name_a]
            reps_b = reps[name_b]
            n_layers_a = reps_a.shape[0]
            n_layers_b = reps_b.shape[0]
            n_points = min(n_layers_a, n_layers_b)
            alphas = np.linspace(0, 1, n_points)

            knn_scores = []
            cka_scores = []

            for alpha in tqdm(alphas, desc="Layer sweep"):
                la = min(int(alpha * (n_layers_a - 1)), n_layers_a - 1)
                lb = min(int(alpha * (n_layers_b - 1)), n_layers_b - 1)

                X = reps_a[la].numpy()
                Y = reps_b[lb].numpy()

                knn = mutual_knn(X, Y, k=10)
                knn_null = compute_null(X, Y, mutual_knn, n_perms=args.n_perms, k=10)
                knn_z = (knn - knn_null.mean()) / (knn_null.std() + 1e-8)

                cka = linear_cka(X, Y)
                cka_null = compute_null(X, Y, linear_cka, n_perms=args.n_perms)
                cka_z = (cka - cka_null.mean()) / (cka_null.std() + 1e-8)

                knn_scores.append({"alpha": float(alpha), "layer_a": la, "layer_b": lb,
                                   "raw": float(knn), "null_mean": float(knn_null.mean()),
                                   "null_std": float(knn_null.std()), "z": float(knn_z)})
                cka_scores.append({"alpha": float(alpha), "layer_a": la, "layer_b": lb,
                                   "raw": float(cka), "null_mean": float(cka_null.mean()),
                                   "null_std": float(cka_null.std()), "z": float(cka_z)})

            max_knn = max(knn_scores, key=lambda x: x["z"])
            max_cka = max(cka_scores, key=lambda x: x["z"])
            print(f"  Best kNN: z={max_knn['z']:.1f} (raw={max_knn['raw']:.4f}) at α={max_knn['alpha']:.2f}")
            print(f"  Best CKA: z={max_cka['z']:.1f} (raw={max_cka['raw']:.4f}) at α={max_cka['alpha']:.2f}")

            results[pair_key] = {
                "knn": knn_scores, "cka": cka_scores,
                "best_knn_z": max_knn["z"], "best_cka_z": max_cka["z"],
                "best_knn_raw": max_knn["raw"], "best_cka_raw": max_cka["raw"],
            }

        # Plot
        fig, axes = plt.subplots(len(pairs), 2, figsize=(12, 4 * len(pairs)))
        if len(pairs) == 1:
            axes = axes.reshape(1, -1)

        for idx, (pair_key, res) in enumerate(results.items()):
            alphas = [s["alpha"] for s in res["knn"]]

            axes[idx, 0].plot(alphas, [s["raw"] for s in res["knn"]], 'o-', label="Raw", color="blue")
            axes[idx, 0].plot(alphas, [s["null_mean"] for s in res["knn"]], 's--', label="Null", color="gray")
            axes[idx, 0].fill_between(alphas,
                [s["null_mean"] - 2*s["null_std"] for s in res["knn"]],
                [s["null_mean"] + 2*s["null_std"] for s in res["knn"]],
                alpha=0.2, color="gray")
            axes[idx, 0].set_ylabel("Mutual k-NN overlap")
            axes[idx, 0].set_title(f"{pair_key} — Mutual k-NN (k=10)")
            axes[idx, 0].legend()
            axes[idx, 0].set_xlabel("Normalized depth α")

            axes[idx, 1].plot(alphas, [s["raw"] for s in res["cka"]], 'o-', label="Raw", color="red")
            axes[idx, 1].plot(alphas, [s["null_mean"] for s in res["cka"]], 's--', label="Null", color="gray")
            axes[idx, 1].fill_between(alphas,
                [s["null_mean"] - 2*s["null_std"] for s in res["cka"]],
                [s["null_mean"] + 2*s["null_std"] for s in res["cka"]],
                alpha=0.2, color="gray")
            axes[idx, 1].set_ylabel("Linear CKA")
            axes[idx, 1].set_title(f"{pair_key} — CKA")
            axes[idx, 1].legend()
            axes[idx, 1].set_xlabel("Normalized depth α")

        plt.tight_layout()
        plt.savefig(FIG_DIR / "phase1_cross_arch.png", dpi=150)
        print(f"\nFigure saved to {FIG_DIR / 'phase1_cross_arch.png'}")

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY (2.8B scale)")
        print(f"{'='*60}")
        print(f"{'Pair':>30} {'Best kNN z':>12} {'kNN raw':>10} {'Best CKA z':>12} {'CKA raw':>10}")
        print("-" * 75)
        for pair_key, res in results.items():
            print(f"{pair_key:>30} {res['best_knn_z']:>12.1f} {res['best_knn_raw']:>10.4f} "
                  f"{res['best_cka_z']:>12.1f} {res['best_cka_raw']:>10.4f}")

        # Compare with pilot
        print(f"\n{'='*60}")
        print("COMPARISON: Pilot (160M) vs Phase 1 (2.8B)")
        print(f"{'='*60}")
        pilot_knn = {"Pythia ↔ Mamba": 379.4, "Pythia ↔ RWKV": 353.5, "Mamba ↔ RWKV": 436.0}
        for pair_key, res in results.items():
            short = pair_key.split("-")[0].strip() + " ↔ " + pair_key.split("↔")[1].split("-")[0].strip()
            pilot_z = pilot_knn.get(short, "N/A")
            print(f"  {pair_key}: pilot_z={pilot_z}, phase1_z={res['best_knn_z']:.1f}")

        with open(PHASE_DIR / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {PHASE_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
