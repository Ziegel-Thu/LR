#!/usr/bin/env python3
"""
Exp-002 Phase 1: 单 bit-width 并行版
用法: python run_phase1_single.py --nbits 4 --dev0 cuda:2 --dev1 cuda:3
"""

import os, gc, json, sys, time, argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL      = "google/gemma-2-2b"
LAYERS     = [0, 5, 12, 17, 23]
BS         = 8
SEQ        = 512
CACHE      = Path("/nvmessd/lifanhong/LR-env/cache")
EXP        = Path(__file__).resolve().parent.parent


class Acc:
    def __init__(self, d):
        z = lambda: np.zeros(d, np.float64)
        self.n = 0
        self.sx, self.sy = z(), z()
        self.sx2, self.sy2, self.sxy = z(), z(), z()
        self.fx, self.fy = z(), z()

    def add(self, x, y):
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


def load_saes(layers, dev):
    from sae_lens import SAE
    saes = {}
    for l in layers:
        for l0 in [71, 77, 82, 63, 57, 46, 100, 130, 21, 164, 34, 11]:
            try:
                s = SAE.from_pretrained(
                    release="gemma-scope-2b-pt-res",
                    sae_id=f"layer_{l}/width_16k/average_l0_{l0}",
                    device=dev)
                if not isinstance(s, tuple):
                    saes[l] = s
                else:
                    saes[l] = s[0]
                print(f"  L{l}: OK (L0={l0})")
                break
            except Exception:
                continue
        else:
            print(f"  L{l}: FAILED")
    return saes


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


def get_acts(model, ids, mask, layers, dev):
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


def summarize(acc, fp16_fr, layer, nbits):
    r = acc.r()
    alive = fp16_fr > 1e-4
    n = int(alive.sum())
    ar = r[alive] if n else np.array([])
    res = dict(layer=layer, nbits=nbits, n_alive=n,
               mean_r=float(ar.mean()) if n else 0,
               median_r=float(np.median(ar)) if n else 0,
               above_08=float((ar > .8).mean()) if n else 0,
               above_05=float((ar > .5).mean()) if n else 0)
    if n > 50:
        rates = fp16_fr[alive]
        corrs = r[alive]
        ps = np.percentile(rates[rates > 0], [20, 40, 60, 80])
        bins = [("Q1_rare", rates <= ps[0]), ("Q2", (rates > ps[0]) & (rates <= ps[1])),
                ("Q3", (rates > ps[1]) & (rates <= ps[2])), ("Q4", (rates > ps[2]) & (rates <= ps[3])),
                ("Q5_common", rates > ps[3])]
        quintiles = {}
        for name, m in bins:
            if m.sum() > 0:
                quintiles[name] = dict(mean_r=float(corrs[m].mean()),
                                       above_08=float((corrs[m] > .8).mean()), n=int(m.sum()))
        res["quintiles"] = quintiles
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nbits", type=int, required=True)
    parser.add_argument("--dev0", default="cuda:0", help="FP16 model + SAEs")
    parser.add_argument("--dev1", default="cuda:1", help="Quantized model")
    args = parser.parse_args()

    t0 = time.time()

    # Data
    tok_path = CACHE / f"tok_gemma2_2000000_{SEQ}.pt"
    assert tok_path.exists(), f"Tokenized data not found: {tok_path}"
    data = torch.load(tok_path, weights_only=True)
    ids, mask = data["ids"], data["mask"]
    n_seq = len(ids)
    print(f"{n_seq} sequences loaded")

    # SAEs
    saes = load_saes(LAYERS, args.dev0)
    active = sorted(saes.keys())
    d_sae = list(saes.values())[0].cfg.d_sae
    print(f"Layers: {active}, d_sae={d_sae}")

    # Models
    fp16 = load_fp16(args.dev0)
    qm = load_quant(args.nbits, args.dev1)

    # FP16 firing rates + comparison
    n_batches = (n_seq + BS - 1) // BS
    fire = {l: np.zeros(d_sae, np.int64) for l in active}
    acc = {l: Acc(d_sae) for l in active}
    n_tok = 0

    for i in tqdm(range(0, n_seq, BS), total=n_batches, desc=f"{args.nbits}-bit"):
        batch_ids = ids[i:i+BS]
        batch_mask = mask[i:i+BS]

        fp16_acts = get_acts(fp16, batch_ids, batch_mask, active, args.dev0)
        quant_acts = get_acts(qm, batch_ids, batch_mask, active, args.dev1)

        for l in active:
            with torch.no_grad():
                zf = saes[l].encode(fp16_acts[l].to(args.dev0)).cpu().numpy()
                zq = saes[l].encode(quant_acts[l].to(args.dev0)).cpu().numpy()
            fire[l] += (zf > 0).sum(0).astype(np.int64)
            acc[l].add(zf, zq)
        n_tok += batch_mask.sum().item()

    fp16_fr = {l: fire[l] / n_tok for l in active}

    results = []
    for l in active:
        res = summarize(acc[l], fp16_fr[l], l, args.nbits)
        results.append(res)
        q = res.get("quintiles", {})
        q1 = q.get("Q1_rare", {}).get("mean_r", 0)
        q5 = q.get("Q5_common", {}).get("mean_r", 0)
        print(f"  L{l}: mean_r={res['mean_r']:.4f}  r>0.8={res['above_08']:.1%}"
              f"  rare={q1:.3f}  common={q5:.3f}")

    elapsed = time.time() - t0
    out = {"config": dict(model=MODEL, layers=active, nbits=args.nbits,
                          n_tokens=n_tok, dev0=args.dev0, dev1=args.dev1),
           "results": results, "elapsed_sec": elapsed}

    out_path = EXP / f"results_phase1_{args.nbits}bit.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}  ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
