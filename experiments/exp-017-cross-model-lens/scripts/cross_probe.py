#!/usr/bin/env python3
"""
Exp-017: Cross-model lens transfer.

Tests whether a linear probe trained on one model's hidden states can decode
the same feature from another model's hidden states.

Focuses on three d=2560 models (Pythia-2.8B, Mamba-2.8B, RWKV-3B) that share
dimensionality, so probes transfer directly without alignment.

Data comes from exp-007 caches (hidden states + token_info.pt per model).

NOTE: The exp-007 caches live on node-local NVMe SSDs.  Before running, ensure
all three cache dirs are accessible from the execution node.  Current layout:
  - Pythia-2.8B → jiagpu4:/nvmessd/lifanhong/LR-env/exp007_28b
  - Mamba-2.8B  → jiagpu5:/nvmessd/lifanhong/LR-env/exp007-mamba28b
  - RWKV-3B     → jiagpu4:/nvmessd/lifanhong/LR-env/exp007-rwkv3b
You may need to copy the missing dir(s) to the execution node first.

Usage:
  source /nvmessd/lifanhong/LR-env/venv/bin/activate
  python cross_probe.py [--output cross_transfer.json]
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── Configuration ────────────────────────────────────────────────────────────

MODELS = {
    "Pythia-2.8B": {
        "cache_dir": "/nvmessd/lifanhong/LR-env/exp007_28b",
        "n_layers": 32,
        "d_model": 2560,
    },
    # Mamba-2.8B cache is on jiagpu5 (108GB), skip for now
    # "Mamba-2.8B": {
    #     "cache_dir": "/nvmessd/lifanhong/LR-env/exp007-mamba28b",
    #     "n_layers": 64,
    #     "d_model": 2560,
    # },
    "RWKV-3B": {
        "cache_dir": "/nvmessd/lifanhong/LR-env/exp007-rwkv3b",
        "n_layers": 32,
        "d_model": 2560,
    },
}

ALL_FEATURES = [
    "is_capitalized", "is_plural", "is_high_freq",
    "is_numeric", "is_punctuation", "is_short", "is_stopword",
    "is_rare", "is_title_case", "starts_with_space",
    "is_subword",
]

RANDOM_SEED = 42
TEST_RATIO = 0.2
MAX_ITER = 2000  # sklearn LogisticRegression

# ── Feature labelers (from exp-007 run_full.py) ─────────────────────────────

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "that",
    "this", "it", "its", "and", "but", "or", "not", "no", "so", "if",
}


def label_feature(feature_name: str, token_strs: list[str],
                  token_ids: np.ndarray, token_counts: Counter) -> np.ndarray:
    """Return binary labels for a feature (same logic as exp-007)."""
    if feature_name == "is_capitalized":
        return np.array([1 if t.lstrip() and t.lstrip()[0].isupper() else 0
                         for t in token_strs])
    elif feature_name == "is_plural":
        return np.array([1 if len(t.strip()) > 2
                         and t.strip().lower().endswith(("s", "es", "ies"))
                         else 0 for t in token_strs])
    elif feature_name == "is_high_freq":
        ranked = {tid: r for r, (tid, _) in enumerate(token_counts.most_common())}
        return np.array([1 if ranked.get(int(t), 999999) < 1000 else 0
                         for t in token_ids])
    elif feature_name == "is_numeric":
        return np.array([1 if any(c.isdigit() for c in t) else 0
                         for t in token_strs])
    elif feature_name == "is_punctuation":
        return np.array([1 if t.strip() and all(not c.isalnum() for c in t.strip())
                         else 0 for t in token_strs])
    elif feature_name == "is_short":
        return np.array([1 if len(t.strip()) <= 2 else 0 for t in token_strs])
    elif feature_name == "is_stopword":
        return np.array([1 if t.strip().lower() in STOPWORDS else 0
                         for t in token_strs])
    elif feature_name == "is_rare":
        ranked = {tid: r for r, (tid, _) in enumerate(token_counts.most_common())}
        return np.array([1 if ranked.get(int(t), 999999) > 10000 else 0
                         for t in token_ids])
    elif feature_name == "is_title_case":
        return np.array([1 if t.strip() and t.strip()[0].isupper()
                         and t.strip()[1:].islower() and len(t.strip()) > 1
                         else 0 for t in token_strs])
    elif feature_name == "starts_with_space":
        return np.array([1 if t.startswith(" ") or t.startswith("Ġ") else 0
                         for t in token_strs])
    elif feature_name == "is_subword":
        return np.array([1 if not (t.startswith(" ") or t.startswith("Ġ"))
                         else 0 for t in token_strs])
    else:
        raise ValueError(f"Unknown feature: {feature_name}")


# ── Data loading ─────────────────────────────────────────────────────────────

def load_token_info(cache_dir: Path) -> dict:
    """Load token_info.pt → {token_ids, token_strs, token_counts}."""
    path = cache_dir / "token_info.pt"
    info = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "token_ids": info["token_ids"],
        "token_strs": info["token_strs"],
        "token_counts": Counter(info["token_counts"]),
    }


def load_hidden_layer(cache_dir: Path, layer_idx: int) -> torch.Tensor:
    """Load a single layer's hidden states: [n_tokens, d_model]."""
    path = cache_dir / "hiddens" / f"layer_{layer_idx}.pt"
    return torch.load(path, map_location="cpu", weights_only=True)


def load_feature_results(cache_dir: Path, feature: str) -> dict | None:
    """Load the exp-007 feature probe results JSON (has best_probe_layer etc.)."""
    path = cache_dir / "features" / f"{feature}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_best_layer(cache_dir: Path, feature: str) -> int | None:
    """Get best probe layer from exp-007 results."""
    result = load_feature_results(cache_dir, feature)
    if result is None or result.get("skipped"):
        return None
    return result.get("best_probe_layer")


def proportional_layer(src_layer: int, src_n_layers: int,
                       tgt_n_layers: int) -> int:
    """Map a layer index proportionally between models of different depth."""
    frac = src_layer / (src_n_layers - 1) if src_n_layers > 1 else 0.0
    return round(frac * (tgt_n_layers - 1))


# ── Probe training & evaluation ─────────────────────────────────────────────

def train_probe(X: np.ndarray, y: np.ndarray, seed: int = RANDOM_SEED):
    """
    Train logistic regression probe.
    Returns (test_accuracy, model, X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=seed, stratify=y)
    clf = LogisticRegression(
        max_iter=MAX_ITER, solver="lbfgs", random_state=seed, n_jobs=-1)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    return acc, clf, X_train, X_test, y_train, y_test


def evaluate_probe(clf, X: np.ndarray, y: np.ndarray,
                   train_indices: np.ndarray | None = None,
                   test_indices: np.ndarray | None = None) -> float:
    """Evaluate a trained probe on new data. Uses same test split indices."""
    if test_indices is not None:
        X_eval = X[test_indices]
        y_eval = y[test_indices]
    else:
        _, X_eval, _, y_eval = train_test_split(
            X, y, test_size=TEST_RATIO, random_state=RANDOM_SEED, stratify=y)
    preds = clf.predict(X_eval)
    return accuracy_score(y_eval, preds)


# ── Main ─────────────────────────────────────────────────────────────────────

def validate_cache_dirs():
    """Check all cache directories are accessible."""
    missing = []
    for name, cfg in MODELS.items():
        d = Path(cfg["cache_dir"])
        if not d.exists():
            missing.append(f"  {name}: {d}")
        else:
            meta_path = d / "meta.json"
            if not meta_path.exists():
                missing.append(f"  {name}: {d} (no meta.json)")
    if missing:
        print("ERROR: Missing cache directories:", file=sys.stderr)
        print("\n".join(missing), file=sys.stderr)
        print("\nSee script header for data locations and how to copy them.",
              file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-model probe transfer experiment")
    parser.add_argument("--output", default="cross_transfer.json",
                        help="Output JSON path (default: cross_transfer.json)")
    parser.add_argument("--features", nargs="+", default=ALL_FEATURES,
                        help="Features to test (default: all)")
    args = parser.parse_args()

    validate_cache_dirs()

    model_names = list(MODELS.keys())
    n_models = len(model_names)

    # ── Load token info & pre-compute labels per model ───────────────────
    print("=" * 70)
    print("Loading token info for each model...")
    print("=" * 70)
    token_data = {}
    for name in model_names:
        cache_dir = Path(MODELS[name]["cache_dir"])
        info = load_token_info(cache_dir)
        n_tok = len(info["token_strs"])
        print(f"  {name}: {n_tok} tokens")
        token_data[name] = info

    # Pre-compute labels per (model, feature)
    print("\nComputing feature labels...")
    labels = {}  # (model_name, feature) → np.ndarray
    skipped_features = set()
    for feat in args.features:
        for name in model_names:
            td = token_data[name]
            y = label_feature(feat, td["token_strs"], td["token_ids"],
                              td["token_counts"])
            pos_rate = y.mean()
            if pos_rate < 0.02 or pos_rate > 0.98:
                print(f"  SKIP {feat} ({name}): pos_rate={pos_rate:.3f}")
                skipped_features.add(feat)
            labels[(name, feat)] = y

    active_features = [f for f in args.features if f not in skipped_features]
    print(f"\nActive features: {len(active_features)}/{len(args.features)}")
    for f in active_features:
        rates = [labels[(m, f)].mean() for m in model_names]
        print(f"  {f}: pos_rate = {', '.join(f'{r:.3f}' for r in rates)}")

    # ── Fixed train/test split indices ───────────────────────────────────
    # All models have the same n_tokens (176213), so we use a single
    # index-based split.  Even though token strings differ across
    # tokenizers, the split is by position index.
    n_tokens = len(token_data[model_names[0]]["token_strs"])
    all_indices = np.arange(n_tokens)
    train_idx, test_idx = train_test_split(
        all_indices, test_size=TEST_RATIO, random_state=RANDOM_SEED)

    # ── Run cross-model transfer ─────────────────────────────────────────
    results = {}  # feature → {src → {tgt → {acc, ...}}}
    summary_matrix = {}  # feature → 2D list (src × tgt) of transfer_ratio

    for feat_i, feat in enumerate(active_features):
        print(f"\n{'=' * 70}")
        print(f"Feature [{feat_i+1}/{len(active_features)}]: {feat}")
        print("=" * 70)

        feat_results = {}

        # Step 1: For each source model, find best layer & train probe
        probes = {}  # model_name → (best_layer, clf, same_model_acc)
        for src in model_names:
            cache_dir = Path(MODELS[src]["cache_dir"])

            # Get best layer from exp-007 results or find it ourselves
            best_layer = get_best_layer(cache_dir, feat)
            if best_layer is not None:
                print(f"  {src}: best layer = {best_layer} (from exp-007)")
            else:
                # Scan a few layers to find the best
                print(f"  {src}: scanning layers for best probe...")
                n_layers = MODELS[src]["n_layers"]
                # Sample ~8 evenly spaced layers
                sample_layers = np.linspace(
                    0, n_layers - 1, min(8, n_layers), dtype=int).tolist()
                best_acc, best_layer = 0.0, 0
                y_src = labels[(src, feat)]
                for li in sample_layers:
                    H = load_hidden_layer(cache_dir, li).numpy()
                    X_tr, X_te = H[train_idx], H[test_idx]
                    y_tr, y_te = y_src[train_idx], y_src[test_idx]
                    clf = LogisticRegression(
                        max_iter=MAX_ITER, solver="lbfgs",
                        random_state=RANDOM_SEED, n_jobs=-1)
                    clf.fit(X_tr, y_tr)
                    acc = accuracy_score(y_te, clf.predict(X_te))
                    if acc > best_acc:
                        best_acc, best_layer = acc, li
                    del H
                print(f"  {src}: best layer = {best_layer} "
                      f"(acc={best_acc:.4f}, scanned)")

            # Train probe on best layer
            print(f"  {src}: training probe on layer {best_layer}...", end=" ",
                  flush=True)
            t0 = time.time()
            H = load_hidden_layer(cache_dir, best_layer).numpy()
            y_src = labels[(src, feat)]
            X_tr, X_te = H[train_idx], H[test_idx]
            y_tr, y_te = y_src[train_idx], y_src[test_idx]

            clf = LogisticRegression(
                max_iter=MAX_ITER, solver="lbfgs",
                random_state=RANDOM_SEED, n_jobs=-1)
            clf.fit(X_tr, y_tr)
            same_acc = accuracy_score(y_te, clf.predict(X_te))
            print(f"acc={same_acc:.4f}  ({time.time()-t0:.1f}s)")

            probes[src] = (best_layer, clf, same_acc)
            del H

        # Step 2: Apply each probe to every target model
        for src in model_names:
            src_best_layer, clf, same_acc = probes[src]
            src_n_layers = MODELS[src]["n_layers"]
            feat_results[src] = {}

            for tgt in model_names:
                tgt_n_layers = MODELS[tgt]["n_layers"]
                tgt_cache = Path(MODELS[tgt]["cache_dir"])

                if tgt == src:
                    # Same-model: already have accuracy
                    feat_results[src][tgt] = {
                        "src_layer": src_best_layer,
                        "tgt_layer": src_best_layer,
                        "accuracy": same_acc,
                        "same_model_acc": same_acc,
                        "transfer_ratio": 1.0,
                    }
                    continue

                # Proportional layer mapping
                tgt_layer = proportional_layer(
                    src_best_layer, src_n_layers, tgt_n_layers)

                print(f"  Transfer {src} L{src_best_layer} → "
                      f"{tgt} L{tgt_layer}...", end=" ", flush=True)
                t0 = time.time()
                H_tgt = load_hidden_layer(tgt_cache, tgt_layer).numpy()
                y_tgt = labels[(tgt, feat)]
                X_te_tgt = H_tgt[test_idx]
                y_te_tgt = y_tgt[test_idx]

                transfer_acc = accuracy_score(
                    y_te_tgt, clf.predict(X_te_tgt))
                ratio = transfer_acc / same_acc if same_acc > 0 else 0.0
                print(f"acc={transfer_acc:.4f} "
                      f"(ratio={ratio:.3f}, {time.time()-t0:.1f}s)")

                feat_results[src][tgt] = {
                    "src_layer": src_best_layer,
                    "tgt_layer": tgt_layer,
                    "accuracy": transfer_acc,
                    "same_model_acc": same_acc,
                    "transfer_ratio": ratio,
                }
                del H_tgt

        results[feat] = feat_results

        # Print transfer matrix for this feature
        print(f"\n  Transfer matrix for '{feat}':")
        header_label = "src \\ tgt"
        header = f"  {header_label:>14s}" + "".join(
            f"{m:>14s}" for m in model_names)
        print(header)
        for src in model_names:
            row = f"  {src:>14s}"
            for tgt in model_names:
                acc = feat_results[src][tgt]["accuracy"]
                ratio = feat_results[src][tgt]["transfer_ratio"]
                row += f"  {acc:.4f}({ratio:.2f})"
            print(row)

    # ── Aggregate summary ────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("AGGREGATE TRANSFER MATRIX (avg transfer ratio across features)")
    print("=" * 70)

    avg_matrix = {src: {tgt: [] for tgt in model_names} for src in model_names}
    avg_acc_matrix = {src: {tgt: [] for tgt in model_names} for src in model_names}
    for feat in active_features:
        for src in model_names:
            for tgt in model_names:
                r = results[feat][src][tgt]
                avg_matrix[src][tgt].append(r["transfer_ratio"])
                avg_acc_matrix[src][tgt].append(r["accuracy"])

    header_label = "src \\ tgt"
    header = f"  {header_label:>14s}" + "".join(
        f"{m:>14s}" for m in model_names)
    print(header)
    for src in model_names:
        row = f"  {src:>14s}"
        for tgt in model_names:
            mean_ratio = np.mean(avg_matrix[src][tgt])
            row += f"       {mean_ratio:.3f}  "
        print(row)

    print("\n  (accuracy)")
    print(header)
    for src in model_names:
        row = f"  {src:>14s}"
        for tgt in model_names:
            mean_acc = np.mean(avg_acc_matrix[src][tgt])
            row += f"       {mean_acc:.4f} "
        print(row)

    # ── Per-feature transfer summary ─────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("PER-FEATURE SUMMARY (avg cross-model transfer ratio)")
    print("=" * 70)
    for feat in active_features:
        cross_ratios = []
        for src in model_names:
            for tgt in model_names:
                if src != tgt:
                    cross_ratios.append(
                        results[feat][src][tgt]["transfer_ratio"])
        mean_cross = np.mean(cross_ratios)
        min_cross = np.min(cross_ratios)
        max_cross = np.max(cross_ratios)
        same_accs = [results[feat][m][m]["accuracy"] for m in model_names]
        print(f"  {feat:>20s}:  avg_ratio={mean_cross:.3f}  "
              f"range=[{min_cross:.3f}, {max_cross:.3f}]  "
              f"same_acc={np.mean(same_accs):.4f}")

    # ── Save results ─────────────────────────────────────────────────────
    output = {
        "models": {name: {"n_layers": cfg["n_layers"], "d_model": cfg["d_model"]}
                   for name, cfg in MODELS.items()},
        "features": results,
        "aggregate": {
            "avg_transfer_ratio": {
                src: {tgt: float(np.mean(avg_matrix[src][tgt]))
                      for tgt in model_names}
                for src in model_names
            },
            "avg_accuracy": {
                src: {tgt: float(np.mean(avg_acc_matrix[src][tgt]))
                      for tgt in model_names}
                for src in model_names
            },
        },
    }

    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
