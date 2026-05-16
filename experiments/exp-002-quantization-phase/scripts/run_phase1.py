#!/usr/bin/env python3
"""
Exp-002 Phase 1: Weight Quantization × SAE Feature Recovery
Model: Gemma-2 2B    SAE: Gemma Scope 16K (residual stream)

Architecture:
  GPU 0 — FP16 model + SAEs (encoders)
  GPU 1 — quantized model (swapped per bit-width)

For each bit-width b in {8,6,4,3,2}:
  For each batch of tokens (same tokens for both models):
    z_fp16  = SAE.encode(FP16_model(tokens)[layer])
    z_quant = SAE.encode(Quant_model(tokens)[layer])
    accumulate per-feature Pearson(z_fp16, z_quant)
"""

import os, gc, json, sys, time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# HF_ENDPOINT: use default huggingface.co (token-gated access)

# ── Config ──────────────────────────────────────────────────
MODEL      = "google/gemma-2-2b"
LAYERS     = [0, 5, 12, 17, 23]   # early/early-mid/mid/mid-late/late
BITS       = [8, 6, 4, 3, 2]
N_TOKENS   = 2_000_000
BS         = 8
SEQ        = 512
DEV0       = "cuda:0"          # FP16 + SAEs
DEV1       = "cuda:1"          # quantized
CACHE      = Path("/nvmessd/lifanhong/LR-env/cache")
EXP        = Path(__file__).resolve().parent.parent


# ── Online Pearson-r accumulator ────────────────────────────
class Acc:
    """Per-feature online Pearson-r between FP16 and quantized SAE features."""
    def __init__(self, d):
        z = lambda: np.zeros(d, np.float64)
        self.n = 0
        self.sx, self.sy = z(), z()
        self.sx2, self.sy2, self.sxy = z(), z(), z()
        self.fx, self.fy = z(), z()

    def add(self, x, y):
        """x, y: (n_tokens, d_sae) numpy float32."""
        self.n += x.shape[0]
        self.sx  += x.sum(0);  self.sy  += y.sum(0)
        self.sx2 += (x*x).sum(0); self.sy2 += (y*y).sum(0)
        self.sxy += (x*y).sum(0)
        self.fx  += (x>0).sum(0); self.fy  += (y>0).sum(0)

    def r(self):
        n = self.n
        c = self.sxy/n - (self.sx/n)*(self.sy/n)
        vx = np.maximum(self.sx2/n - (self.sx/n)**2, 0)
        vy = np.maximum(self.sy2/n - (self.sy/n)**2, 0)
        d = np.sqrt(vx*vy); d[d<1e-12] = 1e-12
        return c/d

    @property
    def fr_x(self): return self.fx / max(self.n, 1)
    @property
    def fr_y(self): return self.fy / max(self.n, 1)


# ── Data ────────────────────────────────────────────────────
def tokenize(n_tok):
    from transformers import AutoTokenizer
    from datasets import load_dataset

    p = CACHE / f"tok_gemma2_{n_tok}_{SEQ}.pt"
    if p.exists():
        d = torch.load(p, weights_only=True)
        print(f"Loaded {len(d['ids'])} seqs from cache")
        return d

    tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=str(CACHE/"hf"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    datasets_to_try = [
        ("Salesforce/wikitext", {"name": "wikitext-103-raw-v1", "split": "train"}),
        ("wikitext", {"name": "wikitext-103-raw-v1", "split": "train"}),
    ]

    ds = None
    for ds_name, kwargs in datasets_to_try:
        try:
            print(f"Trying dataset: {ds_name}")
            ds = load_dataset(ds_name, cache_dir=str(CACHE/"datasets"), **kwargs)
            print(f"  Loaded {ds_name}")
            break
        except Exception as e:
            print(f"  Failed: {e}")

    if ds is None:
        raise RuntimeError("Could not load any dataset")

    ids_list, mask_list = [], []
    total = 0

    for ex in tqdm(ds, desc="Tokenizing"):
        text = ex.get("text", "")
        if len(text) < 50:
            continue
        e = tok(text, truncation=True, max_length=SEQ,
                padding="max_length", return_tensors="pt")
        n_real = e["attention_mask"][0].sum().item()
        if n_real >= SEQ // 4:
            ids_list.append(e["input_ids"][0])
            mask_list.append(e["attention_mask"][0])
            total += n_real
            if total >= n_tok:
                break

    d = {"ids": torch.stack(ids_list), "mask": torch.stack(mask_list)}
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save(d, p)
    print(f"Cached {len(ids_list)} seqs ({total} tokens)")
    return d


# ── SAE ─────────────────────────────────────────────────────
def load_saes(layers, dev):
    from sae_lens import SAE

    saes = {}
    for l in layers:
        ok = False
        for l0 in [71, 77, 82, 63, 57, 46, 100, 130, 21, 164, 34, 11]:
            try:
                s, _, _ = SAE.from_pretrained(
                    release="gemma-scope-2b-pt-res",
                    sae_id=f"layer_{l}/width_16k/average_l0_{l0}",
                    device=dev)
                saes[l] = s
                print(f"  L{l}: OK (L0={l0}, d_sae={s.cfg.d_sae})")
                ok = True; break
            except Exception:
                continue
        if not ok:
            print(f"  L{l}: FAILED — no SAE found")
    return saes


# ── Models ──────────────────────────────────────────────────
def load_fp16(dev):
    from transformers import AutoModelForCausalLM
    print(f"Loading {MODEL} FP16 on {dev}...")
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16,
        cache_dir=str(CACHE/"hf"), device_map=dev)
    m.eval()
    print("  Done")
    return m


def load_quant(nbits, dev):
    from transformers import AutoModelForCausalLM
    from hqq.core.quantize import HQQLinear, BaseQuantizeConfig

    print(f"Loading {MODEL} → {nbits}-bit on {dev}...")
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, cache_dir=str(CACHE/"hf"))
    cfg = BaseQuantizeConfig(nbits=nbits, group_size=64, axis=1)
    cnt = 0
    for name, mod in list(m.named_modules()):
        if isinstance(mod, torch.nn.Linear):
            try:
                q = HQQLinear(mod, cfg, compute_dtype=torch.float16, device=dev)
                parts = name.split(".")
                par = m
                for p in parts[:-1]:
                    par = getattr(par, p)
                setattr(par, parts[-1], q)
                cnt += 1
            except Exception:
                pass
    m.to(dev).eval()
    print(f"  Quantized {cnt} linears → {nbits}-bit")
    return m


# ── Activation collection ──────────────────────────────────
def get_acts(model, ids, mask, layers, dev):
    """Forward pass → {layer: (n_valid_tokens, d_model)} on CPU."""
    buf = {}
    handles = []
    for l in layers:
        def hk(mod, inp, out, _l=l):
            o = out[0] if isinstance(out, tuple) else out
            buf[_l] = o.detach()
        handles.append(model.model.layers[l].register_forward_hook(hk))

    with torch.no_grad():
        model(input_ids=ids.to(dev), attention_mask=mask.to(dev))

    for h in handles:
        h.remove()

    m = mask.bool()
    return {l: buf[l].float()[m.to(buf[l].device)].cpu() for l in layers}


# ── Analysis ────────────────────────────────────────────────
def summarize(acc, fp16_fr, layer, nbits):
    """Compute summary stats from correlation accumulator."""
    r = acc.r()
    alive = fp16_fr > 1e-4
    n = int(alive.sum())
    ar = r[alive] if n else np.array([])

    res = dict(
        layer=layer, nbits=nbits, n_alive=n,
        mean_r=float(ar.mean()) if n else 0,
        median_r=float(np.median(ar)) if n else 0,
        above_08=float((ar > .8).mean()) if n else 0,
        above_05=float((ar > .5).mean()) if n else 0,
    )

    # Quintile stratification by FP16 firing rate
    if n > 50:
        rates = fp16_fr[alive]
        corrs = r[alive]
        ps = np.percentile(rates[rates > 0], [20, 40, 60, 80])
        bins = [
            ("Q1_rare",   rates <= ps[0]),
            ("Q2",        (rates > ps[0]) & (rates <= ps[1])),
            ("Q3",        (rates > ps[1]) & (rates <= ps[2])),
            ("Q4",        (rates > ps[2]) & (rates <= ps[3])),
            ("Q5_common", rates > ps[3]),
        ]
        quintiles = {}
        for name, m in bins:
            if m.sum() > 0:
                quintiles[name] = dict(
                    mean_r=float(corrs[m].mean()),
                    above_08=float((corrs[m] > .8).mean()),
                    n=int(m.sum()),
                )
        res["quintiles"] = quintiles

    return res


def print_summary(results, layers):
    print(f"\n{'='*80}")
    print("SUMMARY: Mean Feature Correlation (alive features)")
    print(f"{'='*80}")

    by_bits = {}
    for r in results:
        by_bits.setdefault(r["nbits"], {})[r["layer"]] = r

    header = f"{'Bits':>4} | " + " | ".join(f"{'L'+str(l):>8}" for l in layers)
    print(header)
    print("-" * len(header))
    for b in sorted(by_bits.keys(), reverse=True):
        vals = " | ".join(
            f"{by_bits[b].get(l, {}).get('mean_r', 0):>8.4f}" for l in layers)
        print(f"{b:>4} | {vals}")

    print(f"\n--- Frequency stratification @ 4-bit ---")
    if 4 in by_bits:
        for l in layers:
            r = by_bits[4].get(l, {})
            q = r.get("quintiles", {})
            if q:
                vals = "  ".join(f"{k}={v['mean_r']:.3f}" for k, v in q.items())
                print(f"  L{l}: {vals}")


# ── Main ────────────────────────────────────────────────────
def main():
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)

    # 1. Data
    print("=" * 60 + "\n STEP 1: Tokenize data\n" + "=" * 60)
    data = tokenize(N_TOKENS)
    ids, mask = data["ids"], data["mask"]
    n_seq = len(ids)
    n_valid = mask.sum().item()
    print(f"  {n_seq} sequences, {n_valid} valid tokens")

    # 2. SAEs
    print("\n" + "=" * 60 + "\n STEP 2: Load SAEs\n" + "=" * 60)
    saes = load_saes(LAYERS, DEV0)
    active = sorted(saes.keys())
    if not active:
        print("ERROR: No SAEs loaded"); return
    d_sae = list(saes.values())[0].cfg.d_sae
    print(f"  Layers: {active}, d_sae={d_sae}")

    # 3. FP16 model + firing rates
    print("\n" + "=" * 60 + "\n STEP 3: FP16 baseline\n" + "=" * 60)
    fp16 = load_fp16(DEV0)

    fire = {l: np.zeros(d_sae, np.int64) for l in active}
    n_tok = 0
    n_batches = (n_seq + BS - 1) // BS

    for i in tqdm(range(0, n_seq, BS), total=n_batches, desc="FP16 firing"):
        batch_ids = ids[i:i+BS]
        batch_mask = mask[i:i+BS]
        acts = get_acts(fp16, batch_ids, batch_mask, active, DEV0)
        for l in active:
            with torch.no_grad():
                z = saes[l].encode(acts[l].to(DEV0))
            fire[l] += (z.cpu().numpy() > 0).sum(0).astype(np.int64)
        n_tok += batch_mask.sum().item()

    fp16_fr = {l: fire[l] / n_tok for l in active}
    print(f"  {n_tok} tokens")
    for l in active:
        na = (fp16_fr[l] > 1e-4).sum()
        print(f"  L{l}: {na}/{d_sae} alive ({na/d_sae:.1%})")

    # 4. Quantized comparisons
    print("\n" + "=" * 60 + "\n STEP 4: Quantized comparisons\n" + "=" * 60)
    all_results = []

    for nbits in BITS:
        print(f"\n{'─'*40} {nbits}-bit {'─'*40}")
        qm = load_quant(nbits, DEV1)
        acc = {l: Acc(d_sae) for l in active}

        for i in tqdm(range(0, n_seq, BS), total=n_batches, desc=f"{nbits}-bit"):
            batch_ids = ids[i:i+BS]
            batch_mask = mask[i:i+BS]

            fp16_acts = get_acts(fp16, batch_ids, batch_mask, active, DEV0)
            quant_acts = get_acts(qm, batch_ids, batch_mask, active, DEV1)

            for l in active:
                with torch.no_grad():
                    zf = saes[l].encode(fp16_acts[l].to(DEV0)).cpu().numpy()
                    zq = saes[l].encode(quant_acts[l].to(DEV0)).cpu().numpy()
                acc[l].add(zf, zq)

        del qm
        torch.cuda.empty_cache()
        gc.collect()

        for l in active:
            res = summarize(acc[l], fp16_fr[l], l, nbits)
            all_results.append(res)
            q = res.get("quintiles", {})
            q1 = q.get("Q1_rare", {}).get("mean_r", 0)
            q5 = q.get("Q5_common", {}).get("mean_r", 0)
            print(f"  L{l}: mean_r={res['mean_r']:.4f}  r>0.8={res['above_08']:.1%}"
                  f"  rare={q1:.3f}  common={q5:.3f}")

    del fp16
    torch.cuda.empty_cache()

    # 5. Save
    elapsed = time.time() - t0
    out = {
        "config": dict(model=MODEL, layers=active, bit_widths=BITS,
                       n_tokens=n_tok, seq_len=SEQ, batch_size=BS),
        "results": all_results,
        "elapsed_sec": elapsed,
    }
    out_path = EXP / "results_phase1.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")
    print(f"Total time: {elapsed/60:.1f} min")

    print_summary(all_results, active)


if __name__ == "__main__":
    main()
