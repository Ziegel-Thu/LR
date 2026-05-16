#!/usr/bin/env python3
"""
Extract mean-pooled residual representations for exp-003 scaling runs.

Usage:
  CUDA_VISIBLE_DEVICES=0 python extract_reps.py \
    --models Pythia-410M=EleutherAI/pythia-410m-deduped \
    --cache-dir /nvmessd/lifanhong/LR-env/cache/hf
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

import torch
from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def parse_model_specs(specs: list[str]) -> list[tuple[str, str]]:
    models = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Model spec must be NAME=HF_ID, got: {spec}")
        name, model_id = spec.split("=", 1)
        models.append((name.strip(), model_id.strip()))
    return models


def with_retry(fn, label: str, attempts: int = 5, sleep_s: int = 20):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # network downloads on the cluster can SSL EOF
            last_error = exc
            print(f"[{label}] attempt {attempt}/{attempts} failed: {exc}", flush=True)
            if attempt < attempts:
                time.sleep(sleep_s)
    raise RuntimeError(f"{label} failed after {attempts} attempts") from last_error


def load_texts(cache_dir: str, n_stimuli: int) -> list[str]:
    from datasets import load_dataset

    def _load():
        return load_dataset(
            "Salesforce/wikitext",
            "wikitext-103-raw-v1",
            split="validation",
            cache_dir=str(Path(cache_dir) / "datasets"),
        )

    ds = with_retry(_load, "load wikitext")
    texts = [t for t in ds["text"] if len(t.strip()) > 50][:n_stimuli]
    print(f"Using {len(texts)} stimuli", flush=True)
    return texts


def extract_model(
    name: str,
    model_id: str,
    texts: list[str],
    out_dir: Path,
    cache_dir: str,
    seq_len: int,
    batch_size: int,
    device: str,
) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_path = out_dir / f"{name}_reps.pt"
    if out_path.exists():
        print(f"Skipping {name}: {out_path} already exists", flush=True)
        return

    print(f"\n=== {name}: {model_id} ===", flush=True)

    tokenizer = with_retry(
        lambda: AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir),
        f"load tokenizer {name}",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = with_retry(
        lambda: AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16,
            output_hidden_states=True,
            cache_dir=cache_dir,
        ),
        f"load model {name}",
    )
    model = model.to(device).eval()

    all_hidden = None
    for i in tqdm(range(0, len(texts), batch_size), desc=name):
        batch_texts = texts[i : i + batch_size]
        tokens = tokenizer(
            batch_texts,
            return_tensors="pt",
            max_length=seq_len,
            truncation=True,
            padding="max_length",
        ).to(device)

        with torch.no_grad():
            hidden_states = model(**tokens).hidden_states

        attention_mask = tokens["attention_mask"].unsqueeze(-1).float()
        batch_reps = []
        for hidden in hidden_states:
            pooled = (hidden.float() * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            batch_reps.append(pooled.cpu())
        batch_reps = torch.stack(batch_reps)
        all_hidden = batch_reps if all_hidden is None else torch.cat([all_hidden, batch_reps], dim=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(all_hidden, out_path)
    print(f"Saved {name}: {tuple(all_hidden.shape)} -> {out_path}", flush=True)

    del model
    torch.cuda.empty_cache()
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="One or more NAME=HF_ID specs")
    parser.add_argument("--cache-dir", default="/nvmessd/lifanhong/LR-env/cache/hf")
    parser.add_argument("--out-dir", default="experiments/exp-003-cross-arch-platonic/phase1/data")
    parser.add_argument("--n-stimuli", type=int, default=5000)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    models = parse_model_specs(args.models)
    texts = load_texts(args.cache_dir, args.n_stimuli)
    out_dir = Path(args.out_dir)

    for name, model_id in models:
        extract_model(
            name=name,
            model_id=model_id,
            texts=texts,
            out_dir=out_dir,
            cache_dir=args.cache_dir,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            device=args.device,
        )


if __name__ == "__main__":
    main()
