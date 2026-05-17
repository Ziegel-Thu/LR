"""
Exp-005 Phase 5: GPT-2 Small 真实模型验证

在预训练 GPT-2 small 上验证 worst-case vs average-case gap。

方法：
  1. 选两类语义不同的 prompt（如 "science" vs "sports"）
  2. 提取某一层的 hidden state
  3. 用 token-level IG/SHAP attribution 区分两类
  4. 对比自然 prompt vs adversarial prompt（精心构造使 attribution 相似）

如果结论成立：自然 prompt 下 SHAP 能区分，adversarial 构造下不能。
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import gc

EXP_DIR = Path(__file__).resolve().parent.parent


def load_gpt2():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_hidden_state(model, tokenizer, text, layer=6):
    """Extract mean-pooled hidden state at specified layer."""
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        outputs = model(**tokens)
    hidden = outputs.hidden_states[layer]  # (1, seq_len, d_model)
    mask = tokens["attention_mask"].unsqueeze(-1).float()
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # (1, d_model)
    return pooled.squeeze(0)  # (d_model,)


def get_logit_diff(model, tokenizer, text, target_token_id):
    """Get logit for a specific next-token prediction."""
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        logits = model(**tokens).logits
    last_logit = logits[0, -1, :]  # (vocab_size,)
    return last_logit[target_token_id].item()


def integrated_gradients_lm(model, tokenizer, text, target_token_id,
                             embed_layer=None, steps=30):
    """
    IG for language model: attribute the logit of target_token_id
    to each input token's embedding.

    Returns per-token attribution scores (sum over embedding dim).
    """
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    # Get embeddings
    if embed_layer is None:
        embed_layer = model.transformer.wte

    embeddings = embed_layer(input_ids).detach()  # (1, seq_len, d_model)
    baseline = torch.zeros_like(embeddings)

    # Interpolation path
    alphas = torch.linspace(0, 1, steps + 1)
    grads_sum = torch.zeros_like(embeddings)

    for alpha in alphas:
        interp = baseline + alpha * (embeddings - baseline)
        interp.requires_grad_(True)

        # Forward with interpolated embeddings
        outputs = model(inputs_embeds=interp, attention_mask=attention_mask)
        logit = outputs.logits[0, -1, target_token_id]

        logit.backward()
        grads_sum += interp.grad.detach()
        interp.grad = None

    avg_grads = grads_sum / (steps + 1)
    ig = (embeddings - baseline) * avg_grads  # (1, seq_len, d_model)

    # Sum over embedding dim → per-token score
    token_scores = ig.sum(dim=-1).squeeze(0)  # (seq_len,)
    return token_scores.detach().numpy()


def main():
    print("=" * 60)
    print("Exp-005 Phase 5: GPT-2 Small Verification")
    print("=" * 60)

    model, tokenizer = load_gpt2()
    print(f"Model: GPT-2 small ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")

    # Two categories of prompts
    science_prompts = [
        "The experiment showed that the chemical reaction",
        "According to quantum mechanics, particles can",
        "The DNA sequence was analyzed using",
        "Researchers discovered a new species of",
        "The telescope captured images of distant",
        "Neural networks learn representations by",
        "The mathematical proof demonstrates that",
        "Climate models predict that temperatures will",
        "The protein structure was determined through",
        "Evolutionary theory suggests that species",
        "The physics experiment confirmed that light",
        "Genome sequencing revealed mutations in",
        "The algorithm converges when the loss function",
        "Statistical analysis showed significant correlation between",
        "The chemical compound reacts with oxygen to",
    ]

    sports_prompts = [
        "The quarterback threw the ball to",
        "The championship game was won by",
        "The athlete broke the world record in",
        "The coach decided to substitute the",
        "The referee called a foul on",
        "The basketball team scored three points with",
        "The soccer match ended with a score of",
        "The Olympic swimmer finished the race in",
        "The tennis player served an ace during",
        "The baseball pitcher struck out the",
        "The hockey team scored in overtime to",
        "The marathon runner crossed the finish line",
        "The boxing match went to the final",
        "The golf tournament was decided on the",
        "The football team rushed for over two hundred",
    ]

    # Target tokens for IG: "the" (common, neutral)
    target_id = tokenizer.encode(" the")[0]

    print(f"\nComputing IG attributions for {len(science_prompts)} science + {len(sports_prompts)} sports prompts...")
    print(f"Target token: '{tokenizer.decode([target_id])}' (id={target_id})")

    # 1. Hidden state analysis: can we distinguish categories?
    print("\n--- Hidden State Analysis (Layer 6) ---")
    sci_hiddens = []
    spo_hiddens = []
    for p in science_prompts:
        sci_hiddens.append(get_hidden_state(model, tokenizer, p, layer=6).numpy())
    for p in sports_prompts:
        spo_hiddens.append(get_hidden_state(model, tokenizer, p, layer=6).numpy())

    sci_hiddens = np.array(sci_hiddens)
    spo_hiddens = np.array(spo_hiddens)

    # Cosine similarity within vs between
    from numpy.linalg import norm
    def cos_sim(a, b):
        return np.dot(a, b) / (norm(a) * norm(b) + 1e-10)

    intra_sci = np.mean([cos_sim(sci_hiddens[i], sci_hiddens[j])
                         for i in range(len(sci_hiddens))
                         for j in range(i+1, len(sci_hiddens))])
    intra_spo = np.mean([cos_sim(spo_hiddens[i], spo_hiddens[j])
                         for i in range(len(spo_hiddens))
                         for j in range(i+1, len(spo_hiddens))])
    inter = np.mean([cos_sim(sci_hiddens[i], spo_hiddens[j])
                     for i in range(len(sci_hiddens))
                     for j in range(len(spo_hiddens))])

    print(f"  Intra-science cosine: {intra_sci:.4f}")
    print(f"  Intra-sports cosine:  {intra_spo:.4f}")
    print(f"  Inter-category cosine: {inter:.4f}")
    print(f"  Gap (intra - inter):  {(intra_sci + intra_spo)/2 - inter:.4f}")

    # 2. IG attribution analysis
    print("\n--- IG Attribution Analysis ---")
    sci_attrs = []
    spo_attrs = []

    for i, p in enumerate(science_prompts):
        attr = integrated_gradients_lm(model, tokenizer, p, target_id, steps=20)
        sci_attrs.append({
            'text': p,
            'attr_mean': float(attr.mean()),
            'attr_std': float(attr.std()),
            'attr_max': float(attr.max()),
            'attr_l2': float(np.sqrt((attr**2).sum())),
        })
        if i % 5 == 0:
            print(f"  Science {i+1}/{len(science_prompts)}: mean={attr.mean():.4f}, L2={np.sqrt((attr**2).sum()):.4f}")

    for i, p in enumerate(sports_prompts):
        attr = integrated_gradients_lm(model, tokenizer, p, target_id, steps=20)
        spo_attrs.append({
            'text': p,
            'attr_mean': float(attr.mean()),
            'attr_std': float(attr.std()),
            'attr_max': float(attr.max()),
            'attr_l2': float(np.sqrt((attr**2).sum())),
        })
        if i % 5 == 0:
            print(f"  Sports {i+1}/{len(sports_prompts)}: mean={attr.mean():.4f}, L2={np.sqrt((attr**2).sum()):.4f}")

    # 3. Can IG attributions distinguish categories?
    print("\n--- Distinguishability ---")
    sci_l2 = [a['attr_l2'] for a in sci_attrs]
    spo_l2 = [a['attr_l2'] for a in spo_attrs]
    sci_mean = [a['attr_mean'] for a in sci_attrs]
    spo_mean = [a['attr_mean'] for a in spo_attrs]

    # Simple threshold classifier on attr_mean
    all_means = sci_mean + spo_mean
    labels = [0] * len(sci_mean) + [1] * len(spo_mean)
    best_iga = 1.0
    for t in sorted(set(all_means)):
        pred = [1 if v > t else 0 for v in all_means]
        spec = sum(1 for l, p in zip(labels, pred) if l == 0 and p == 0) / labels.count(0)
        sens = sum(1 for l, p in zip(labels, pred) if l == 1 and p == 1) / labels.count(1)
        best_iga = max(best_iga, spec + sens)
        spec_r = sum(1 for l, p in zip(labels, pred) if l == 0 and p == 1) / labels.count(0)
        sens_r = sum(1 for l, p in zip(labels, pred) if l == 1 and p == 0) / labels.count(1)
        best_iga = max(best_iga, spec_r + sens_r)

    # Same for L2
    all_l2 = sci_l2 + spo_l2
    best_iga_l2 = 1.0
    for t in sorted(set(all_l2)):
        pred = [1 if v > t else 0 for v in all_l2]
        spec = sum(1 for l, p in zip(labels, pred) if l == 0 and p == 0) / labels.count(0)
        sens = sum(1 for l, p in zip(labels, pred) if l == 1 and p == 1) / labels.count(1)
        best_iga_l2 = max(best_iga_l2, spec + sens)
        spec_r = sum(1 for l, p in zip(labels, pred) if l == 0 and p == 1) / labels.count(0)
        sens_r = sum(1 for l, p in zip(labels, pred) if l == 1 and p == 0) / labels.count(1)
        best_iga_l2 = max(best_iga_l2, spec_r + sens_r)

    print(f"  IGA (attr_mean): {best_iga:.3f}")
    print(f"  IGA (attr_L2):   {best_iga_l2:.3f}")
    print(f"  Science attr_mean: {np.mean(sci_mean):.4f} ± {np.std(sci_mean):.4f}")
    print(f"  Sports  attr_mean: {np.mean(spo_mean):.4f} ± {np.std(spo_mean):.4f}")
    print(f"  Science attr_L2:   {np.mean(sci_l2):.4f} ± {np.std(sci_l2):.4f}")
    print(f"  Sports  attr_L2:   {np.mean(spo_l2):.4f} ± {np.std(spo_l2):.4f}")

    # Save
    results = {
        'hidden_state': {
            'intra_science': float(intra_sci),
            'intra_sports': float(intra_spo),
            'inter': float(inter),
        },
        'attribution': {
            'iga_mean': float(best_iga),
            'iga_l2': float(best_iga_l2),
            'science_attrs': sci_attrs,
            'sports_attrs': spo_attrs,
        },
    }
    out_path = EXP_DIR / "results_phase5.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
