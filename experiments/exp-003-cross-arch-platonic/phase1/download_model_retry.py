#!/usr/bin/env python3
"""Download a HuggingFace model with retries for unstable cluster networking."""

from __future__ import annotations

import argparse
import time

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--cache-dir", default="/nvmessd/lifanhong/LR-env/cache/hf")
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--sleep", type=int, default=20)
    args = parser.parse_args()

    last_error = None
    for attempt in range(1, args.attempts + 1):
        try:
            print(f"Downloading {args.model_id} attempt {attempt}/{args.attempts}", flush=True)
            snapshot_download(args.model_id, cache_dir=args.cache_dir)
            print("DOWNLOAD DONE", flush=True)
            return
        except Exception as exc:
            last_error = exc
            print(f"FAILED attempt {attempt}: {exc}", flush=True)
            if attempt < args.attempts:
                time.sleep(args.sleep)
    raise RuntimeError(f"Download failed after {args.attempts} attempts") from last_error


if __name__ == "__main__":
    main()
