"""
Exp-003: 跨架构柏拉图收敛 Pilot
Pythia-160M vs Mamba-130M vs RWKV-4-169M

对应: experiments/exp-003-cross-arch-platonic/plan.md
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import os

os.environ.pop("HF_ENDPOINT", None)

EXP_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# Extract representations
# ============================================================

def extract_representations(model_name, n_stimuli=5000, seq_len=256, device="cpu"):
    """Extract per-layer mean-pooled representations."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, output_hidden_states=True
    ).to(device)
    model.eval()

    print(f"Loading stimuli...")
    ds = load_dataset("wikitext", "wikitext-103-v1", split="validation")
    texts = [t for t in ds["text"] if len(t.strip()) > 50][:n_stimuli]
    print(f"Using {len(texts)} stimuli")

    all_hidden = None
    batch_size = 16
    n_batches = (len(texts) + batch_size - 1) // batch_size

    for i in tqdm(range(n_batches), desc=f"Extracting {model_name.split('/')[-1]}"):
        batch_texts = texts[i * batch_size : (i + 1) * batch_size]
        tokens = tokenizer(
            batch_texts, return_tensors="pt", max_length=seq_len,
            truncation=True, padding="max_length"
        ).to(device)

        with torch.no_grad():
            outputs = model(**tokens)

        hidden_states = outputs.hidden_states  # tuple of (batch, seq, d)
        attention_mask = tokens["attention_mask"].unsqueeze(-1).float()

        # Mean pool over non-padding tokens, per layer
        batch_reps = []
        for h in hidden_states:
            pooled = (h * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            batch_reps.append(pooled.cpu())

        # Stack: (n_layers, batch, d)
        batch_reps = torch.stack(batch_reps)

        if all_hidden is None:
            all_hidden = batch_reps
        else:
            all_hidden = torch.cat([all_hidden, batch_reps], dim=1)

    del model
    print(f"Extracted: {all_hidden.shape} (layers, stimuli, d)")
    return all_hidden  # (n_layers, n_stimuli, d)


# ============================================================
# Similarity metrics
# ============================================================

def mutual_knn(X, Y, k=10):
    """Mutual k-NN overlap between two (N, d) matrices."""
    # Compute distance matrices
    X_norm = F.normalize(torch.tensor(X, dtype=torch.float32), dim=1)
    Y_norm = F.normalize(torch.tensor(Y, dtype=torch.float32), dim=1)

    sim_XX = X_norm @ X_norm.T
    sim_YY = Y_norm @ Y_norm.T

    # Get k-NN indices (exclude self)
    sim_XX.fill_diagonal_(-1)
    sim_YY.fill_diagonal_(-1)
    nn_X = sim_XX.topk(k, dim=1).indices  # (N, k)
    nn_Y = sim_YY.topk(k, dim=1).indices  # (N, k)

    # Compute overlap
    overlaps = []
    for i in range(len(X)):
        set_x = set(nn_X[i].numpy())
        set_y = set(nn_Y[i].numpy())
        overlaps.append(len(set_x & set_y) / k)

    return np.mean(overlaps)


def linear_cka(X, Y):
    """Linear CKA between two (N, d) matrices."""
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    # Center
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    hsic_xy = (X @ X.T * (Y @ Y.T)).sum()
    hsic_xx = (X @ X.T * (X @ X.T)).sum()
    hsic_yy = (Y @ Y.T * (Y @ Y.T)).sum()

    return (hsic_xy / (hsic_xx.sqrt() * hsic_yy.sqrt())).item()


def compute_null(X, Y, metric_fn, n_perms=100, **kwargs):
    """Permutation null distribution."""
    nulls = []
    for _ in range(n_perms):
        perm = np.random.permutation(len(Y))
        nulls.append(metric_fn(X, Y[perm], **kwargs))
    return np.array(nulls)


# ============================================================
# Main
# ============================================================

def main():
    n_stimuli = 5000
    seq_len = 256
    k = 10
    n_perms = 100

    models = {
        "Pythia-160M": "EleutherAI/pythia-160m-deduped",
        "Mamba-130M": "state-spaces/mamba-130m-hf",
        "RWKV-4-169M": "RWKV/rwkv-4-169m-pile",
    }

    # Extract representations
    reps = {}
    for name, model_id in models.items():
        cache_path = EXP_DIR / "data" / f"{name}_reps.pt"
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            print(f"Loading cached {name}...")
            reps[name] = torch.load(cache_path, weights_only=True)
        else:
            try:
                reps[name] = extract_representations(model_id, n_stimuli, seq_len)
                torch.save(reps[name], cache_path)
            except Exception as e:
                print(f"FAILED to load {name}: {e}")
                continue

    if len(reps) < 2:
        print("Need at least 2 models. Exiting.")
        return

    # Compare all pairs
    model_names = list(reps.keys())
    pairs = [(model_names[i], model_names[j])
             for i in range(len(model_names))
             for j in range(i + 1, len(model_names))]

    results = {}

    for name_a, name_b in pairs:
        pair_key = f"{name_a} ↔ {name_b}"
        print(f"\n{'='*60}")
        print(f"Comparing {pair_key}")
        print(f"{'='*60}")

        reps_a = reps[name_a]  # (layers_a, N, d)
        reps_b = reps[name_b]  # (layers_b, N, d)
        n_layers_a = reps_a.shape[0]
        n_layers_b = reps_b.shape[0]

        # Use normalized depth matching
        n_points = min(n_layers_a, n_layers_b)
        alphas = np.linspace(0, 1, n_points)

        knn_scores = []
        cka_scores = []
        knn_nulls_mean = []
        cka_nulls_mean = []

        for alpha in tqdm(alphas, desc="Layer sweep"):
            la = min(int(alpha * (n_layers_a - 1)), n_layers_a - 1)
            lb = min(int(alpha * (n_layers_b - 1)), n_layers_b - 1)

            X = reps_a[la].numpy()
            Y = reps_b[lb].numpy()

            # Mutual k-NN
            knn = mutual_knn(X, Y, k=k)
            knn_null = compute_null(X, Y, mutual_knn, n_perms=n_perms, k=k)
            knn_z = (knn - knn_null.mean()) / (knn_null.std() + 1e-8)

            # Linear CKA
            cka = linear_cka(X, Y)
            cka_null = compute_null(X, Y, linear_cka, n_perms=n_perms)
            cka_z = (cka - cka_null.mean()) / (cka_null.std() + 1e-8)

            knn_scores.append({"alpha": float(alpha), "layer_a": la, "layer_b": lb,
                               "raw": float(knn), "null_mean": float(knn_null.mean()),
                               "null_std": float(knn_null.std()), "z": float(knn_z)})
            cka_scores.append({"alpha": float(alpha), "layer_a": la, "layer_b": lb,
                               "raw": float(cka), "null_mean": float(cka_null.mean()),
                               "null_std": float(cka_null.std()), "z": float(cka_z)})

            knn_nulls_mean.append(float(knn_null.mean()))
            cka_nulls_mean.append(float(cka_null.mean()))

        # Summary
        max_knn = max(knn_scores, key=lambda x: x["z"])
        max_cka = max(cka_scores, key=lambda x: x["z"])
        print(f"  Best mutual-kNN: z={max_knn['z']:.1f} (raw={max_knn['raw']:.4f}, null={max_knn['null_mean']:.4f}) at α={max_knn['alpha']:.2f}")
        print(f"  Best CKA:        z={max_cka['z']:.1f} (raw={max_cka['raw']:.4f}, null={max_cka['null_mean']:.4f}) at α={max_cka['alpha']:.2f}")

        results[pair_key] = {
            "knn": knn_scores,
            "cka": cka_scores,
            "best_knn_z": max_knn["z"],
            "best_cka_z": max_cka["z"],
        }

    # Plot
    fig, axes = plt.subplots(len(pairs), 2, figsize=(12, 4 * len(pairs)))
    if len(pairs) == 1:
        axes = axes.reshape(1, -1)

    for idx, (pair_key, res) in enumerate(results.items()):
        alphas = [s["alpha"] for s in res["knn"]]
        knn_raw = [s["raw"] for s in res["knn"]]
        knn_null = [s["null_mean"] for s in res["knn"]]
        knn_z = [s["z"] for s in res["knn"]]

        cka_raw = [s["raw"] for s in res["cka"]]
        cka_null = [s["null_mean"] for s in res["cka"]]
        cka_z = [s["z"] for s in res["cka"]]

        axes[idx, 0].plot(alphas, knn_raw, 'o-', label="Raw", color="blue")
        axes[idx, 0].plot(alphas, knn_null, 's--', label="Null", color="gray")
        axes[idx, 0].fill_between(alphas,
            [s["null_mean"] - 2*s["null_std"] for s in res["knn"]],
            [s["null_mean"] + 2*s["null_std"] for s in res["knn"]],
            alpha=0.2, color="gray")
        axes[idx, 0].set_ylabel("Mutual k-NN overlap")
        axes[idx, 0].set_title(f"{pair_key} — Mutual k-NN (k={k})")
        axes[idx, 0].legend()
        axes[idx, 0].set_xlabel("Normalized depth α")

        axes[idx, 1].plot(alphas, cka_raw, 'o-', label="Raw", color="red")
        axes[idx, 1].plot(alphas, cka_null, 's--', label="Null", color="gray")
        axes[idx, 1].fill_between(alphas,
            [s["null_mean"] - 2*s["null_std"] for s in res["cka"]],
            [s["null_mean"] + 2*s["null_std"] for s in res["cka"]],
            alpha=0.2, color="gray")
        axes[idx, 1].set_ylabel("Linear CKA")
        axes[idx, 1].set_title(f"{pair_key} — CKA")
        axes[idx, 1].legend()
        axes[idx, 1].set_xlabel("Normalized depth α")

    plt.tight_layout()
    fig_path = EXP_DIR / "figures" / "cross_arch_pilot.png"
    plt.savefig(fig_path, dpi=150)
    print(f"\nFigure saved to {fig_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Pair':>30} {'Best kNN z':>12} {'Best CKA z':>12}")
    print("-" * 55)
    for pair_key, res in results.items():
        print(f"{pair_key:>30} {res['best_knn_z']:>12.1f} {res['best_cka_z']:>12.1f}")

    # Save
    with open(EXP_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {EXP_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
