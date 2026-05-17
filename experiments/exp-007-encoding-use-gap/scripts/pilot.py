"""
Exp-007 Pilot: Encoding ≠ Use 量化
本地 pilot: Pythia-410M, 5 特征, 5 层
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json
import os
import gc

MODEL = "EleutherAI/pythia-410m"
PILOT_LAYERS = [0, 6, 12, 18, 23]  # 5 layers spanning depth
N_SAMPLES = 1000
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "..", "results_pilot.json")


def get_features_from_tokens(tokens, tokenizer):
    """Extract binary features from token strings."""
    features = {}

    # 1. Capitalization: is the token capitalized?
    decoded = [tokenizer.decode([t]) for t in tokens]
    features["capitalized"] = [1 if len(s.strip()) > 0 and s.strip()[0].isupper() else 0 for s in decoded]

    # 2. Punctuation: is the token punctuation?
    features["punctuation"] = [1 if len(s.strip()) > 0 and not s.strip()[0].isalnum() else 0 for s in decoded]

    # 3. Numeric: does the token contain a digit?
    features["numeric"] = [1 if any(c.isdigit() for c in s) else 0 for s in decoded]

    # 4. Short token: token length <= 2 chars (after strip)
    features["short_token"] = [1 if len(s.strip()) <= 2 else 0 for s in decoded]

    # 5. Space-prefixed: token starts with space (common in GPT-NeoX tokenizer)
    features["space_prefix"] = [1 if len(s) > 0 and s[0] == " " else 0 for s in decoded]

    return features


def extract_representations(model, tokenizer, texts):
    """Extract per-token representations from specified layers."""
    all_reps = {l: [] for l in PILOT_LAYERS}
    all_tokens = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), 8):  # batch size 8
            batch_texts = texts[i:i+8]
            inputs = tokenizer(
                batch_texts, return_tensors="pt",
                max_length=SEQ_LEN, truncation=True, padding=True
            ).to(DEVICE)

            outputs = model(**inputs, output_hidden_states=True)

            attention_mask = inputs["attention_mask"]

            for layer_idx in PILOT_LAYERS:
                hidden = outputs.hidden_states[layer_idx]  # (B, L, D)
                # Collect all non-padding tokens
                for b in range(hidden.shape[0]):
                    mask = attention_mask[b].bool()
                    reps = hidden[b, mask].cpu().numpy()  # (L_actual, D)
                    all_reps[layer_idx].append(reps)

            # Collect token ids
            for b in range(inputs["input_ids"].shape[0]):
                mask = attention_mask[b].bool()
                toks = inputs["input_ids"][b, mask].cpu().tolist()
                all_tokens.extend(toks)

            if i % 100 == 0:
                print(f"  Extracted {i}/{len(texts)} texts")

    # Concatenate
    for l in PILOT_LAYERS:
        all_reps[l] = np.concatenate(all_reps[l], axis=0)

    return all_reps, all_tokens


def train_probe(X_train, y_train, X_test, y_test):
    """Train logistic regression probe, return accuracy."""
    # Skip if label is constant
    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        return None

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    return acc, clf.coef_  # return coef for ablation


def ablation_effect(model, tokenizer, texts, layer_idx, direction, device):
    """
    Measure loss change when projecting out 'direction' from layer layer_idx.
    direction: (D,) unit vector
    Returns: mean Δloss (positive = ablation hurts = model uses this info)
    """
    direction = torch.tensor(direction, dtype=torch.float32, device=device)
    direction = direction / direction.norm()

    losses_original = []
    losses_ablated = []

    # Hook to project out direction
    handle = None
    ablate_flag = [False]

    def hook_fn(module, input, output):
        if ablate_flag[0]:
            # output is (B, L, D) for transformer layers
            if isinstance(output, tuple):
                h = output[0]
                proj = torch.einsum("bld,d->bl", h, direction).unsqueeze(-1) * direction
                h_ablated = h - proj
                return (h_ablated,) + output[1:]
            else:
                proj = torch.einsum("bld,d->bl", output, direction).unsqueeze(-1) * direction
                return output - proj

    # Register hook on the target layer
    layer_module = model.gpt_neox.layers[layer_idx]
    handle = layer_module.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        for i in range(0, min(len(texts), 200), 8):
            batch_texts = texts[i:i+8]
            inputs = tokenizer(
                batch_texts, return_tensors="pt",
                max_length=SEQ_LEN, truncation=True, padding=True
            ).to(device)

            labels = inputs["input_ids"].clone()
            labels[inputs["attention_mask"] == 0] = -100

            # Original loss
            ablate_flag[0] = False
            out_orig = model(**inputs, labels=labels)
            losses_original.append(out_orig.loss.item())

            # Ablated loss
            ablate_flag[0] = True
            out_abl = model(**inputs, labels=labels)
            losses_ablated.append(out_abl.loss.item())

    handle.remove()

    mean_orig = np.mean(losses_original)
    mean_abl = np.mean(losses_ablated)
    delta_loss = mean_abl - mean_orig

    return delta_loss


def main():
    print("=" * 60)
    print("Exp-007 Pilot: Encoding ≠ Use")
    print("=" * 60)

    # Load model and tokenizer
    print(f"\nLoading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(DEVICE)
    print(f"  Model on {DEVICE}, {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

    # Load data
    print("\nLoading WikiText-103...")
    ds = load_dataset("wikitext", "wikitext-103-v1", split="validation")
    texts = [t for t in ds["text"] if len(t.split()) > 20][:N_SAMPLES]
    print(f"  {len(texts)} texts")

    # Extract representations
    print("\nExtracting representations...")
    reps, tokens = extract_representations(model, tokenizer, texts)
    n_tokens = len(tokens)
    print(f"  {n_tokens} tokens, rep dim = {reps[PILOT_LAYERS[0]].shape[1]}")

    # Get features
    print("\nExtracting token features...")
    features = get_features_from_tokens(tokens, tokenizer)
    for fname, fvals in features.items():
        pos_rate = sum(fvals) / len(fvals)
        print(f"  {fname}: positive rate = {pos_rate:.3f}")

    # Train/test split
    split = int(0.8 * n_tokens)
    results = []

    for fname, fvals in features.items():
        y = np.array(fvals)
        y_train, y_test = y[:split], y[split:]

        for layer_idx in PILOT_LAYERS:
            X = reps[layer_idx]
            X_train, X_test = X[:split], X[split:]

            # Train probe
            result = train_probe(X_train, y_train, X_test, y_test)
            if result is None:
                print(f"  {fname} @ layer {layer_idx}: skipped (constant label)")
                continue

            probe_acc, coef = result
            direction = coef[0] if coef.shape[0] == 1 else coef[0]  # for binary

            # Ablation
            print(f"  {fname} @ layer {layer_idx}: probe_acc={probe_acc:.3f}, computing ablation...")
            delta_loss = ablation_effect(model, tokenizer, texts, layer_idx, direction, DEVICE)

            results.append({
                "feature": fname,
                "layer": layer_idx,
                "probe_accuracy": round(probe_acc, 4),
                "delta_loss": round(delta_loss, 6),
            })
            print(f"    → Δloss = {delta_loss:.6f}")

    # Save results
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Encoding vs Use")
    print("=" * 60)
    print(f"{'Feature':<15} {'Layer':>5} {'Probe Acc':>10} {'ΔLoss':>10} {'Ghost?':>7}")
    print("-" * 52)
    for r in results:
        ghost = "YES" if r["probe_accuracy"] > 0.7 and abs(r["delta_loss"]) < 0.01 else ""
        print(f"{r['feature']:<15} {r['layer']:>5} {r['probe_accuracy']:>10.3f} {r['delta_loss']:>10.6f} {ghost:>7}")


if __name__ == "__main__":
    main()
