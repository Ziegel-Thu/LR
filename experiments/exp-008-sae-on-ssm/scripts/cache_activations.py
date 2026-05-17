#!/usr/bin/env python3
"""
Exp-008 Phase 1: Cache Mamba-130M hidden state activations for SAE training.

Hooks into the middle layer of Mamba-130M (HF format) and caches output
hidden states from Pile validation data.

Usage:
  CUDA_VISIBLE_DEVICES=4 python cache_activations.py \
    --cache-dir /nvmessd/lifanhong/LR-env/cache/hf \
    --out-dir /nvmessd/lifanhong/LR-env/exp008_acts
"""

import argparse
import gc
import os
import time
from pathlib import Path

import torch
from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

MODEL_ID = "state-spaces/mamba-130m-hf"
N_TOKENS_TARGET = 2_000_000
SEQ_LEN = 256
BATCH_SIZE = 16


def with_retry(fn, label, attempts=5, sleep_s=20):
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            print(f"[{label}] attempt {attempt}/{attempts}: {e}", flush=True)
            if attempt < attempts:
                time.sleep(sleep_s)
    raise RuntimeError(f"{label} failed after {attempts} attempts")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="/nvmessd/lifanhong/LR-env/cache/hf")
    parser.add_argument("--out-dir", default="/nvmessd/lifanhong/LR-env/exp008_acts")
    parser.add_argument("--target-layer", type=int, default=None,
                        help="Layer index to hook. Default: middle layer.")
    parser.add_argument("--n-tokens", type=int, default=N_TOKENS_TARGET)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print("Loading tokenizer...", flush=True)
    tokenizer = with_retry(
        lambda: AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=args.cache_dir),
        "load tokenizer",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...", flush=True)
    model = with_retry(
        lambda: AutoModelForCausalLM.from_pretrained(
            MODEL_ID, cache_dir=args.cache_dir, torch_dtype=torch.float16,
        ),
        "load model",
    )
    model = model.to(args.device).eval()

    # Determine layer count and target layer
    # Mamba HF models use model.backbone.layers
    if hasattr(model, "backbone"):
        n_layers = len(model.backbone.layers)
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        n_layers = len(model.model.layers)
    else:
        # Fallback: count modules
        n_layers = sum(1 for _ in model.modules() if "MambaBlock" in type(_).__name__
                       or "MixerModel" in type(_).__name__)
        if n_layers == 0:
            n_layers = 24  # default guess

    target_layer = args.target_layer if args.target_layer is not None else n_layers // 2
    print(f"Model has {n_layers} layers. Hooking layer {target_layer}.", flush=True)
    print(f"d_model = {model.config.d_model}", flush=True)

    # Register hook
    activations_buffer = []

    def hook_fn(module, input, output):
        # Mamba layers output a tuple or tensor
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        # hidden shape: (batch, seq_len, d_model)
        activations_buffer.append(hidden.detach().cpu().float())

    # Find the target layer module
    if hasattr(model, "backbone"):
        target_module = model.backbone.layers[target_layer]
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        target_module = model.model.layers[target_layer]
    else:
        raise RuntimeError("Cannot find layers in model architecture")

    hook = target_module.register_forward_hook(hook_fn)

    # Load data
    print("Loading Pile validation data...", flush=True)

    def _load_data():
        return load_dataset(
            "Salesforce/wikitext", "wikitext-103-raw-v1",
            split="validation",
            cache_dir=str(Path(args.cache_dir) / "datasets"),
        )

    ds = with_retry(_load_data, "load wikitext")
    texts = [t for t in ds["text"] if len(t.strip()) > 50]
    print(f"  {len(texts)} texts available", flush=True)

    # Tokenize and process
    n_tokens_collected = 0
    chunk_idx = 0
    all_acts = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Caching"):
        if n_tokens_collected >= args.n_tokens:
            break

        batch_texts = texts[i:i + BATCH_SIZE]
        tokens = tokenizer(
            batch_texts,
            return_tensors="pt",
            max_length=SEQ_LEN,
            truncation=True,
            padding="max_length",
        ).to(args.device)

        activations_buffer.clear()
        with torch.no_grad():
            model(**tokens)

        if activations_buffer:
            # (batch, seq_len, d_model) → flatten to (batch*seq_len, d_model)
            act = activations_buffer[0]  # (B, L, D)
            mask = tokens["attention_mask"].cpu()  # (B, L)
            # Only keep non-padding tokens
            act_flat = act[mask.bool()]  # (n_real_tokens, d_model)
            all_acts.append(act_flat)
            n_tokens_collected += act_flat.shape[0]

        # Save in chunks of ~500K tokens
        if n_tokens_collected >= (chunk_idx + 1) * 500_000 and all_acts:
            chunk = torch.cat(all_acts, dim=0)
            chunk_path = out_dir / f"acts_layer{target_layer}_chunk{chunk_idx}.pt"
            torch.save(chunk, chunk_path)
            print(f"  Saved chunk {chunk_idx}: {chunk.shape} → {chunk_path}", flush=True)
            all_acts = []
            chunk_idx += 1

    # Save remaining
    if all_acts:
        chunk = torch.cat(all_acts, dim=0)
        chunk_path = out_dir / f"acts_layer{target_layer}_chunk{chunk_idx}.pt"
        torch.save(chunk, chunk_path)
        print(f"  Saved chunk {chunk_idx}: {chunk.shape} → {chunk_path}", flush=True)

    hook.remove()
    print(f"\nDone! Cached {n_tokens_collected} tokens from layer {target_layer}", flush=True)

    # Save metadata
    import json
    meta = {
        "model_id": MODEL_ID,
        "target_layer": target_layer,
        "n_layers": n_layers,
        "d_model": model.config.d_model,
        "n_tokens": n_tokens_collected,
        "seq_len": SEQ_LEN,
        "n_chunks": chunk_idx + 1,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved: {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
