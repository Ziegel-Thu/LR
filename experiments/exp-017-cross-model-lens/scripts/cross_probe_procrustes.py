#!/usr/bin/env python3
"""
Exp-017 extension: Cross-model probe transfer with Procrustes alignment.

Raw cross-model probe transfer (exp-017) gave ~55% transfer ratio.
This could be due to a global rotation — the same linear subspace might
exist but rotated.  Orthogonal Procrustes alignment removes this rotation.

For each feature at its best probe layer:
  1. Train probe on source model (train split)
  2. Raw transfer: apply probe to target hidden states (test split)
  3. Procrustes: find R minimizing ||X_src_train @ R - X_tgt_train||
     then evaluate probe on X_tgt_test @ R^T ... wait, convention:
     We want to map TARGET into SOURCE space so we can reuse source probe.
     R = argmin ||X_tgt_train @ R - X_src_train||
     Then apply source probe to X_tgt_test @ R.

If Procrustes >> raw: same subspace, different rotation (supports PRH).
If Procrustes ≈ raw: genuinely different representations.

Usage:
  source /nvmessd/lifanhong/LR-env/venv/bin/activate
  python cross_probe_procrustes.py
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ── Configuration ────────────────────────────────────────────────────────────

MODELS = {
    "Pythia-2.8B": {
        "cache_dir": "/nvmessd/lifanhong/LR-env/exp007_28b",
        "n_layers": 32,
        "d_model": 2560,
    },
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
MAX_ITER = 2000
SUBSAMPLE_N = 30000  # Procrustes on 176K×2560 is slow; subsample

# ── Feature labelers (from exp-007) ─────────────────────────────────────────

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
    path = cache_dir / "token_info.pt"
    info = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "token_ids": info["token_ids"],
        "token_strs": info["token_strs"],
        "token_counts": Counter(info["token_counts"]),
    }


def load_hidden_layer(cache_dir: Path, layer_idx: int) -> np.ndarray:
    path = cache_dir / "hiddens" / f"layer_{layer_idx}.pt"
    return torch.load(path, map_location="cpu", weights_only=True).numpy()


def get_best_layer(cache_dir: Path, feature: str) -> int | None:
    path = cache_dir / "features" / f"{feature}.json"
    if not path.exists():
        return None
    with open(path) as f:
        result = json.load(f)
    if result.get("skipped"):
        return None
    return result.get("best_probe_layer")


def proportional_layer(src_layer: int, src_n_layers: int,
                       tgt_n_layers: int) -> int:
    frac = src_layer / (src_n_layers - 1) if src_n_layers > 1 else 0.0
    return round(frac * (tgt_n_layers - 1))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    model_names = list(MODELS.keys())

    # Validate cache dirs
    for name, cfg in MODELS.items():
        d = Path(cfg["cache_dir"])
        if not d.exists():
            print(f"ERROR: cache dir missing: {name} → {d}", file=sys.stderr)
            sys.exit(1)

    # Load token info
    print("=" * 70)
    print("Loading token info...")
    print("=" * 70)
    token_data = {}
    for name in model_names:
        info = load_token_info(Path(MODELS[name]["cache_dir"]))
        n_tok = len(info["token_strs"])
        print(f"  {name}: {n_tok} tokens")
        token_data[name] = info

    # Subsample indices (deterministic)
    n_total = len(token_data[model_names[0]]["token_strs"])
    rng = np.random.RandomState(RANDOM_SEED)
    if SUBSAMPLE_N < n_total:
        sub_idx = np.sort(rng.choice(n_total, SUBSAMPLE_N, replace=False))
        print(f"\nSubsampled {SUBSAMPLE_N} / {n_total} tokens for speed")
    else:
        sub_idx = np.arange(n_total)
    n_sub = len(sub_idx)

    # Train/test split on subsampled indices
    train_pos, test_pos = train_test_split(
        np.arange(n_sub), test_size=TEST_RATIO, random_state=RANDOM_SEED)

    # Compute labels (on subsampled tokens)
    print("\nComputing feature labels...")
    labels = {}
    skipped = set()
    for feat in ALL_FEATURES:
        for name in model_names:
            td = token_data[name]
            y_full = label_feature(feat, td["token_strs"], td["token_ids"],
                                   td["token_counts"])
            y = y_full[sub_idx]
            pos_rate = y.mean()
            if pos_rate < 0.02 or pos_rate > 0.98:
                print(f"  SKIP {feat} ({name}): pos_rate={pos_rate:.3f}")
                skipped.add(feat)
            labels[(name, feat)] = y

    active_features = [f for f in ALL_FEATURES if f not in skipped]
    print(f"\nActive features: {len(active_features)}/{len(ALL_FEATURES)}")

    # ── Run per-feature analysis ─────────────────────────────────────────
    results = {}

    for feat_i, feat in enumerate(active_features):
        print(f"\n{'=' * 70}")
        print(f"Feature [{feat_i+1}/{len(active_features)}]: {feat}")
        print("=" * 70)

        feat_results = {}

        for src_name in model_names:
            for tgt_name in model_names:
                if src_name == tgt_name:
                    continue

                src_cfg = MODELS[src_name]
                tgt_cfg = MODELS[tgt_name]
                src_cache = Path(src_cfg["cache_dir"])
                tgt_cache = Path(tgt_cfg["cache_dir"])

                # Get best layer for source model
                best_layer = get_best_layer(src_cache, feat)
                if best_layer is None:
                    # Scan layers
                    n_layers = src_cfg["n_layers"]
                    sample_layers = np.linspace(
                        0, n_layers - 1, min(8, n_layers), dtype=int).tolist()
                    best_acc, best_layer = 0.0, 0
                    y_src = labels[(src_name, feat)]
                    for li in sample_layers:
                        H = load_hidden_layer(src_cache, li)[sub_idx]
                        X_tr, X_te = H[train_pos], H[test_pos]
                        y_tr, y_te = y_src[train_pos], y_src[test_pos]
                        clf = LogisticRegression(
                            max_iter=MAX_ITER, solver="lbfgs",
                            random_state=RANDOM_SEED, n_jobs=-1)
                        clf.fit(X_tr, y_tr)
                        acc = accuracy_score(y_te, clf.predict(X_te))
                        if acc > best_acc:
                            best_acc, best_layer = acc, li
                        del H

                # Map to target layer
                tgt_layer = proportional_layer(
                    best_layer, src_cfg["n_layers"], tgt_cfg["n_layers"])

                direction = f"{src_name} L{best_layer} → {tgt_name} L{tgt_layer}"
                print(f"\n  {direction}")

                # Load hidden states (subsampled)
                t0 = time.time()
                H_src = load_hidden_layer(src_cache, best_layer)[sub_idx]
                H_tgt = load_hidden_layer(tgt_cache, tgt_layer)[sub_idx]
                y_src = labels[(src_name, feat)]
                y_tgt = labels[(tgt_name, feat)]
                load_time = time.time() - t0
                print(f"    Loaded hiddens: {H_src.shape} ({load_time:.1f}s)")

                # Split
                H_src_tr, H_src_te = H_src[train_pos], H_src[test_pos]
                H_tgt_tr, H_tgt_te = H_tgt[train_pos], H_tgt[test_pos]
                y_src_tr, y_src_te = y_src[train_pos], y_src[test_pos]
                y_tgt_tr, y_tgt_te = y_tgt[train_pos], y_tgt[test_pos]

                # 1) Train probe on source
                t0 = time.time()
                clf = LogisticRegression(
                    max_iter=MAX_ITER, solver="lbfgs",
                    random_state=RANDOM_SEED, n_jobs=-1)
                clf.fit(H_src_tr, y_src_tr)
                same_acc = accuracy_score(y_src_te, clf.predict(H_src_te))
                print(f"    Same-model acc:     {same_acc:.4f}  "
                      f"({time.time()-t0:.1f}s)")

                # 2) Raw transfer: apply source probe to target hidden states
                raw_acc = accuracy_score(y_tgt_te, clf.predict(H_tgt_te))
                raw_ratio = raw_acc / same_acc if same_acc > 0 else 0.0
                print(f"    Raw transfer acc:   {raw_acc:.4f}  "
                      f"(ratio={raw_ratio:.3f})")

                # 3) Procrustes: align target → source space
                # Center both (Procrustes works on centered data)
                t0 = time.time()
                mu_src = H_src_tr.mean(axis=0)
                mu_tgt = H_tgt_tr.mean(axis=0)
                H_src_tr_c = H_src_tr - mu_src
                H_tgt_tr_c = H_tgt_tr - mu_tgt

                # R = argmin ||H_tgt_tr_c @ R - H_src_tr_c||_F
                R, scale = orthogonal_procrustes(H_tgt_tr_c, H_src_tr_c)
                proc_time = time.time() - t0
                print(f"    Procrustes R: {R.shape}, scale={scale:.2f}  "
                      f"({proc_time:.1f}s)")

                # Apply rotation to target test data (centered)
                H_tgt_te_aligned = (H_tgt_te - mu_tgt) @ R + mu_src

                proc_acc = accuracy_score(
                    y_tgt_te, clf.predict(H_tgt_te_aligned))
                proc_ratio = proc_acc / same_acc if same_acc > 0 else 0.0
                improvement = proc_acc - raw_acc
                print(f"    Procrustes acc:     {proc_acc:.4f}  "
                      f"(ratio={proc_ratio:.3f}, "
                      f"Δ={improvement:+.4f})")

                key = f"{src_name}→{tgt_name}"
                feat_results[key] = {
                    "src_layer": best_layer,
                    "tgt_layer": tgt_layer,
                    "same_model_acc": float(same_acc),
                    "raw_transfer_acc": float(raw_acc),
                    "raw_transfer_ratio": float(raw_ratio),
                    "procrustes_acc": float(proc_acc),
                    "procrustes_ratio": float(proc_ratio),
                    "improvement": float(improvement),
                }

                del H_src, H_tgt

        results[feat] = feat_results

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SUMMARY: Raw vs Procrustes Transfer")
    print("=" * 70)
    print(f"{'Feature':>22s}  {'Direction':>22s}  {'Same':>6s}  "
          f"{'Raw':>6s}  {'Proc':>6s}  {'RawR':>5s}  {'ProcR':>5s}  "
          f"{'Δacc':>7s}")
    print("-" * 100)

    agg_raw_ratios = []
    agg_proc_ratios = []
    agg_improvements = []

    for feat in active_features:
        for key, r in results[feat].items():
            print(f"{feat:>22s}  {key:>22s}  "
                  f"{r['same_model_acc']:.4f}  "
                  f"{r['raw_transfer_acc']:.4f}  "
                  f"{r['procrustes_acc']:.4f}  "
                  f"{r['raw_transfer_ratio']:.3f}  "
                  f"{r['procrustes_ratio']:.3f}  "
                  f"{r['improvement']:+.4f}")
            agg_raw_ratios.append(r["raw_transfer_ratio"])
            agg_proc_ratios.append(r["procrustes_ratio"])
            agg_improvements.append(r["improvement"])

    print("-" * 100)
    print(f"{'MEAN':>22s}  {'':>22s}  {'':>6s}  {'':>6s}  {'':>6s}  "
          f"{np.mean(agg_raw_ratios):.3f}  "
          f"{np.mean(agg_proc_ratios):.3f}  "
          f"{np.mean(agg_improvements):+.4f}")

    # Per-feature averages
    print(f"\n{'=' * 70}")
    print("PER-FEATURE AVERAGE (across both directions)")
    print("=" * 70)
    print(f"{'Feature':>22s}  {'Raw Ratio':>10s}  {'Proc Ratio':>11s}  "
          f"{'Δacc':>7s}  {'Verdict':>12s}")
    print("-" * 70)

    for feat in active_features:
        raw_rs = [r["raw_transfer_ratio"] for r in results[feat].values()]
        proc_rs = [r["procrustes_ratio"] for r in results[feat].values()]
        imps = [r["improvement"] for r in results[feat].values()]
        mean_raw = np.mean(raw_rs)
        mean_proc = np.mean(proc_rs)
        mean_imp = np.mean(imps)
        # Verdict
        if mean_proc > mean_raw + 0.05:
            verdict = "ROTATED ✓"
        elif mean_proc < mean_raw - 0.02:
            verdict = "WORSE"
        else:
            verdict = "~SAME"
        print(f"{feat:>22s}  {mean_raw:>10.3f}  {mean_proc:>11.3f}  "
              f"{mean_imp:>+7.4f}  {verdict:>12s}")

    print("-" * 70)
    print(f"{'OVERALL':>22s}  {np.mean(agg_raw_ratios):>10.3f}  "
          f"{np.mean(agg_proc_ratios):>11.3f}  "
          f"{np.mean(agg_improvements):>+7.4f}")

    # ── Save JSON ────────────────────────────────────────────────────────
    output = {
        "config": {
            "subsample_n": SUBSAMPLE_N,
            "test_ratio": TEST_RATIO,
            "seed": RANDOM_SEED,
        },
        "features": results,
        "aggregate": {
            "mean_raw_ratio": float(np.mean(agg_raw_ratios)),
            "mean_procrustes_ratio": float(np.mean(agg_proc_ratios)),
            "mean_improvement": float(np.mean(agg_improvements)),
        },
    }

    out_path = Path("experiments/exp-017-cross-model-lens/procrustes_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
