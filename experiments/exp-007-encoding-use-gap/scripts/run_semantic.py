#!/usr/bin/env python3
"""
Exp-007 Extended: Semantic features with spaCy NER/POS labels.

Adds 5 semantic features that require NLP annotation:
  1. is_noun (POS)
  2. is_verb (POS)  
  3. is_entity (NER)
  4. is_content_word (not function word)
  5. is_past_tense (morphological)

Uses the same parallel pipeline as run_full.py but with spaCy labels.

Usage:
  # Step 1: cache + label
  CUDA_VISIBLE_DEVICES=4 python run_semantic.py cache \
    --work-dir /nvmessd/lifanhong/LR-env/exp007_semantic

  # Step 2: per-feature (parallel)
  CUDA_VISIBLE_DEVICES=4 python run_semantic.py feature --feature is_noun \
    --work-dir /nvmessd/lifanhong/LR-env/exp007_semantic

  # Step 3: analyze
  python run_semantic.py analyze --work-dir /nvmessd/lifanhong/LR-env/exp007_semantic
"""

import argparse, json, os, gc, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from tqdm import tqdm
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

MODEL_ID = "EleutherAI/pythia-1.4b-deduped"
N_SENTENCES = 1000; SEQ_LEN = 128; BATCH_SIZE = 8
SEMANTIC_FEATURES = ["is_noun", "is_verb", "is_entity", "is_content_word", "is_past_tense"]

FUNCTION_WORDS = {"the","a","an","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","shall","can","to","of","in",
    "for","on","with","at","by","from","as","into","about","that","this","it","its","and","but",
    "or","not","no","so","if","then","than","when","where","which","who","whom","whose","what",
    "how","all","each","every","both","few","more","most","other","some","such","any","only"}

def with_retry(fn, label, attempts=5, sleep_s=20):
    for a in range(1, attempts+1):
        try: return fn()
        except Exception as e:
            print(f"[{label}] {a}/{attempts}: {e}", flush=True)
            if a < attempts: time.sleep(sleep_s)
    raise RuntimeError(f"{label} failed")


def cmd_cache(args):
    """Cache hidden states + spaCy labels."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    import spacy

    work_dir = Path(args.work_dir); work_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = with_retry(lambda: AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=args.cache_dir), "tok")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = with_retry(lambda: AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir=args.cache_dir, torch_dtype=torch.float16, output_hidden_states=True), "model")
    model = model.to(args.device).eval()
    n_layers = model.config.num_hidden_layers; d_model = model.config.hidden_size

    ds = with_retry(lambda: load_dataset("Salesforce/wikitext","wikitext-103-raw-v1",split="validation",
                    cache_dir=str(Path(args.cache_dir)/"datasets")), "data")
    texts = [t for t in ds["text"] if len(t.strip())>50][:N_SENTENCES]

    # spaCy for NER/POS
    nlp = spacy.load("en_core_web_sm")
    print(f"spaCy loaded, processing {len(texts)} sentences", flush=True)

    # Process texts with spaCy to get per-char annotations
    spacy_docs = list(nlp.pipe(texts, batch_size=64))

    # Cache hidden states + collect token-level spaCy labels
    layer_buffers = {l: [] for l in range(n_layers)}
    all_labels = {f: [] for f in SEMANTIC_FEATURES}

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Cache"):
        batch_texts = texts[i:i+BATCH_SIZE]
        tokens = tokenizer(batch_texts, return_tensors="pt", max_length=SEQ_LEN,
                           truncation=True, padding="max_length").to(args.device)

        with torch.no_grad():
            out = model(**tokens)
        mask = tokens["attention_mask"].cpu().bool()

        for l in range(n_layers):
            h = out.hidden_states[l+1].cpu().float()
            for b in range(h.shape[0]):
                layer_buffers[l].append(h[b][mask[b]])

        # Map token positions to spaCy annotations
        for b in range(tokens["input_ids"].shape[0]):
            valid = mask[b]
            doc_idx = i + b
            if doc_idx >= len(spacy_docs): continue
            doc = spacy_docs[doc_idx]

            for pos in range(valid.sum().item()):
                tid = tokens["input_ids"][b, pos].item()
                tok_str = tokenizer.decode([tid])

                # Find best matching spaCy token
                best_spacy = None
                tok_clean = tok_str.strip().lower()
                for sp_tok in doc:
                    if sp_tok.text.lower() == tok_clean or tok_clean in sp_tok.text.lower():
                        best_spacy = sp_tok; break

                if best_spacy is not None:
                    all_labels["is_noun"].append(1 if best_spacy.pos_ in ("NOUN","PROPN") else 0)
                    all_labels["is_verb"].append(1 if best_spacy.pos_ == "VERB" else 0)
                    all_labels["is_entity"].append(1 if best_spacy.ent_type_ != "" else 0)
                    all_labels["is_content_word"].append(0 if tok_clean in FUNCTION_WORDS else 1)
                    all_labels["is_past_tense"].append(
                        1 if "Tense=Past" in str(best_spacy.morph) else 0)
                else:
                    all_labels["is_noun"].append(0)
                    all_labels["is_verb"].append(0)
                    all_labels["is_entity"].append(0)
                    all_labels["is_content_word"].append(0 if tok_clean in FUNCTION_WORDS else 1)
                    all_labels["is_past_tense"].append(0)

    # Save per-layer
    hiddens_dir = work_dir / "hiddens"; hiddens_dir.mkdir(exist_ok=True)
    for l in tqdm(range(n_layers), desc="Saving"):
        h = torch.cat(layer_buffers[l], dim=0)
        torch.save(h, hiddens_dir / f"layer_{l}.pt")

    # Save labels
    n_tokens = len(all_labels["is_noun"])
    # Truncate to match hidden states
    min_len = min(n_tokens, layer_buffers[0][0].shape[0] if layer_buffers[0] else n_tokens)
    torch.save({k: np.array(v[:min_len]) for k, v in all_labels.items()}, work_dir / "labels.pt")

    meta = {"model": MODEL_ID, "n_layers": n_layers, "d_model": d_model, "n_tokens": min_len}
    with open(work_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nLabel balance:", flush=True)
    for f in SEMANTIC_FEATURES:
        arr = np.array(all_labels[f][:min_len])
        print(f"  {f}: {arr.mean():.3f} ({arr.sum()}/{len(arr)})")
    print(f"Done: {min_len} tokens × {n_layers} layers")

    del model; torch.cuda.empty_cache(); gc.collect()


def cmd_feature(args):
    """Probe + ablation for one semantic feature."""
    work_dir = Path(args.work_dir); feature = args.feature

    with open(work_dir / "meta.json") as f:
        meta = json.load(f)
    n_layers = meta["n_layers"]

    labels_all = torch.load(work_dir / "labels.pt", map_location="cpu", weights_only=False)
    labels = labels_all[feature]
    pos_rate = labels.mean()
    print(f"Feature '{feature}': pos_rate={pos_rate:.3f}", flush=True)

    if pos_rate < 0.02 or pos_rate > 0.98:
        result = {"feature": feature, "skipped": True, "pos_rate": float(pos_rate)}
        out_dir = work_dir / "features"; out_dir.mkdir(exist_ok=True)
        with open(out_dir / f"{feature}.json", "w") as f:
            json.dump(result, f, indent=2)
        print("SKIP: imbalanced"); return

    y = torch.tensor(labels, dtype=torch.long)

    # Probe
    print("Training probes...", flush=True)
    probe_accs, probe_weights = {}, {}
    for l in tqdm(range(n_layers), desc=f"Probe {feature}"):
        X = torch.load(work_dir / "hiddens" / f"layer_{l}.pt", map_location="cpu", weights_only=True)
        X = X[:len(y)]  # match length
        d = X.shape[1]
        probe = nn.Linear(d, 1).to(args.device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        Xd, yd = X.to(args.device), y.float().to(args.device)
        probe.train()
        for _ in range(10):
            perm = torch.randperm(len(X), device=args.device)
            for j in range(0, len(X), 4096):
                idx = perm[j:j+4096]
                logits = probe(Xd[idx]).squeeze()
                loss = nn.BCEWithLogitsLoss()(logits, yd[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            acc = ((probe(Xd).squeeze()>0).long() == y.to(args.device)).float().mean().item()
        probe_accs[l] = acc
        probe_weights[l] = probe.weight.data.cpu().numpy().flatten()

    best_l = max(probe_accs, key=probe_accs.get)
    print(f"Best probe: {probe_accs[best_l]:.3f} @ L{best_l}", flush=True)

    # Ablation
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    print("Running ablation...", flush=True)
    tokenizer = with_retry(lambda: AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=args.cache_dir), "tok")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = with_retry(lambda: AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir=args.cache_dir, torch_dtype=torch.float16), "model")
    model = model.to(args.device).eval()

    ds = with_retry(lambda: load_dataset("Salesforce/wikitext","wikitext-103-raw-v1",split="validation",
                    cache_dir=str(Path(args.cache_dir)/"datasets")), "data")
    texts = [t for t in ds["text"] if len(t.strip())>50][:N_SENTENCES]

    # Baseline
    bl = []
    for i in range(0, len(texts), BATCH_SIZE):
        tokens = tokenizer(texts[i:i+BATCH_SIZE], return_tensors="pt", max_length=SEQ_LEN,
                           truncation=True, padding="max_length").to(args.device)
        with torch.no_grad(): out = model(**tokens, labels=tokens["input_ids"])
        bl.append(out.loss.item())
    baseline = np.mean(bl)

    ablation_deltas = {}
    for l in tqdm(range(n_layers), desc=f"Ablate {feature}"):
        W = probe_weights[l]
        W_t = torch.tensor(W, dtype=torch.float16, device=args.device)
        W_norm = W_t / (W_t.norm() + 1e-8)
        def make_hook(w):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]; proj = (h @ w).unsqueeze(-1) * w
                    return (h - proj,) + output[1:]
                proj = (output @ w).unsqueeze(-1) * w; return output - proj
            return hook_fn
        hook = model.gpt_neox.layers[l].register_forward_hook(make_hook(W_norm))
        losses = []
        for i in range(0, len(texts), BATCH_SIZE):
            tokens = tokenizer(texts[i:i+BATCH_SIZE], return_tensors="pt", max_length=SEQ_LEN,
                               truncation=True, padding="max_length").to(args.device)
            with torch.no_grad(): out = model(**tokens, labels=tokens["input_ids"])
            losses.append(out.loss.item())
        hook.remove()
        ablation_deltas[l] = float(np.mean(losses) - baseline)

    out_dir = work_dir / "features"; out_dir.mkdir(exist_ok=True)
    result = {"feature": feature, "pos_rate": float(pos_rate), "baseline_loss": baseline,
              "probe_accuracy": {str(l): probe_accs[l] for l in range(n_layers)},
              "ablation_delta_loss": {str(l): ablation_deltas[l] for l in range(n_layers)},
              "best_probe_acc": probe_accs[best_l], "best_probe_layer": best_l}
    with open(out_dir / f"{feature}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_dir / f'{feature}.json'}")
    del model; torch.cuda.empty_cache(); gc.collect()


def cmd_analyze(args):
    """Aggregate and compare with token-level features."""
    work_dir = Path(args.work_dir)
    with open(work_dir / "meta.json") as f:
        meta = json.load(f)
    n_layers = meta["n_layers"]

    results = {}
    for fp in sorted((work_dir / "features").glob("*.json")):
        with open(fp) as f: r = json.load(f)
        if r.get("skipped"): print(f"  {r['feature']}: SKIPPED"); continue
        results[r["feature"]] = r

    # Ghost ratio
    ghost_count = encoded_count = 0
    for r in results.values():
        for l in range(n_layers):
            acc = r["probe_accuracy"][str(l)]
            delta = r["ablation_delta_loss"][str(l)]
            if acc > 0.7:
                encoded_count += 1
                if abs(delta) < 0.01: ghost_count += 1
    ghost_ratio = ghost_count / max(encoded_count, 1)

    print(f"\n{'='*60}\nSEMANTIC FEATURES SUMMARY\n{'='*60}")
    print(f"  Ghost ratio: {ghost_ratio:.1%} ({ghost_count}/{encoded_count})")
    for fname, r in results.items():
        print(f"  {fname}: best_acc={r['best_probe_acc']:.3f}, "
              f"max_|Δloss|={max(abs(r['ablation_delta_loss'][str(l)]) for l in range(n_layers)):.4f}")

    # Compare with exp-007 token-level ghost ratio (71%)
    print(f"\n  Token-level ghost ratio (exp-007): 70.8%")
    print(f"  Semantic ghost ratio: {ghost_ratio:.1%}")

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for ci, (fname, r) in enumerate(results.items()):
        accs = [r["probe_accuracy"][str(l)] for l in range(n_layers)]
        deltas = [r["ablation_delta_loss"][str(l)] for l in range(n_layers)]
        ax.scatter(accs, deltas, c=[colors[ci]], label=fname, alpha=0.7, s=30)
    ax.set_xlabel("Probe Accuracy"); ax.set_ylabel("Δ Loss")
    ax.set_title(f"Semantic Features: Ghost Ratio = {ghost_ratio:.1%}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(work_dir / "semantic_encoding_vs_use.png", dpi=150)

    summary = {"ghost_ratio": ghost_ratio, "n_features": len(results),
               "features": {k: {"best_acc": v["best_probe_acc"]} for k, v in results.items()}}
    with open(work_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cache"); p.add_argument("--cache-dir", default="/nvmessd/lifanhong/LR-env/cache/hf")
    p.add_argument("--work-dir", required=True); p.add_argument("--device", default="cuda")
    p = sub.add_parser("feature"); p.add_argument("--feature", required=True, choices=SEMANTIC_FEATURES)
    p.add_argument("--work-dir", required=True); p.add_argument("--cache-dir", default="/nvmessd/lifanhong/LR-env/cache/hf")
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("analyze"); p.add_argument("--work-dir", required=True)
    args = parser.parse_args()
    {"cache": cmd_cache, "feature": cmd_feature, "analyze": cmd_analyze}[args.cmd](args)

if __name__ == "__main__":
    main()
