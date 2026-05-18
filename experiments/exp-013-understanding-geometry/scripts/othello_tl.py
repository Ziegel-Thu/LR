#!/usr/bin/env python3
"""
Exp-013/014: OthelloGPT probing via TransformerLens.

1. ID profile (geometric portrait)
2. Board state probe (replication of Nanda 2023)
3. Strategy concept probing (mobility, corner control, winning)
4. ID vs probe accuracy correlation

Usage:
  CUDA_VISIBLE_DEVICES=4 python othello_tl.py \
    --out-dir /nvmessd/lifanhong/LR-env/exp013_othello
"""

import argparse, json, os
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from tqdm import tqdm
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt


def twonn_id(X):
    from sklearn.neighbors import NearestNeighbors
    if len(X) > 2000: X = X[np.random.choice(len(X), 2000, replace=False)]
    nn = NearestNeighbors(n_neighbors=3, metric="euclidean").fit(X)
    d, _ = nn.kneighbors(X); r1, r2 = d[:,1], d[:,2]
    mask = r1 > 1e-10; mu = r2[mask]/r1[mask]
    s = np.sum(np.log(mu)); return float(len(mu)/s) if s > 1e-10 else 0

def stable_rank(X):
    if len(X) > 2000: X = X[np.random.choice(len(X), 2000, replace=False)]
    sv = np.linalg.svd(X - X.mean(0), compute_uv=False)
    s2 = sv**2; return float(s2.sum()/(s2[0]+1e-10))


# ── Othello game logic ──

def get_valid_moves(board, player):
    valid = []
    for r in range(8):
        for c in range(8):
            if board[r,c] == 0 and _would_flip(board, r, c, player):
                valid.append((r,c))
    return valid

def _would_flip(board, r, c, player):
    for dr in [-1,0,1]:
        for dc in [-1,0,1]:
            if dr==0 and dc==0: continue
            nr, nc = r+dr, c+dc
            found = False
            while 0<=nr<8 and 0<=nc<8:
                if board[nr,nc] == -player: found = True
                elif board[nr,nc] == player and found: return True
                else: break
                nr, nc = nr+dr, nc+dc
    return False

def apply_move(board, move, player):
    board = board.copy(); r, c = move; board[r,c] = player
    for dr in [-1,0,1]:
        for dc in [-1,0,1]:
            if dr==0 and dc==0: continue
            to_flip = []; nr, nc = r+dr, c+dc
            while 0<=nr<8 and 0<=nc<8:
                if board[nr,nc] == -player: to_flip.append((nr,nc))
                elif board[nr,nc] == player and to_flip:
                    for fr,fc in to_flip: board[fr,fc] = player
                    break
                else: break
                nr, nc = nr+dr, nc+dc
    return board

def generate_games(n=500):
    # OthelloGPT token mapping: 64 squares minus 4 center = 60 playable, mapped to 0-59
    # Center squares (row,col): (3,3),(3,4),(4,3),(4,4) = flat 27,28,35,36
    CENTER = {27, 28, 35, 36}
    SQUARE_TO_TOKEN = {}
    token_idx = 0
    for sq in range(64):
        if sq not in CENTER:
            SQUARE_TO_TOKEN[sq] = token_idx
            token_idx += 1
    # token_idx should be 60

    games = []
    for _ in range(n):
        board = np.zeros((8,8), dtype=int)
        board[3,3] = board[4,4] = -1; board[3,4] = board[4,3] = 1
        cur = 1; moves = []; states = [board.copy()]
        for _ in range(58):  # max 58 moves (60 squares - 4 pre-filled + some passes)
            valid = get_valid_moves(board, cur)
            if not valid:
                cur = -cur; valid = get_valid_moves(board, cur)
                if not valid: break
            mv = valid[np.random.randint(len(valid))]
            board = apply_move(board, mv, cur)
            flat_sq = mv[0]*8 + mv[1]
            token = SQUARE_TO_TOKEN.get(flat_sq)
            if token is not None:
                moves.append(token)
            states.append(board.copy()); cur = -cur
        if len(moves) > 10: games.append({"moves": moves, "states": states})
    return games

def strategy_features(board, player):
    corners = [(0,0),(0,7),(7,0),(7,7)]
    my_p = int((board==player).sum()); opp_p = int((board==-player).sum())
    mob = len(get_valid_moves(board, player))
    opp_mob = len(get_valid_moves(board, -player))
    return {
        "is_winning": int(my_p > opp_p),
        "has_corner": int(any(board[r,c]==player for r,c in corners)),
        "high_mobility": int(mob > opp_mob),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/nvmessd/lifanhong/LR-env/exp013_othello")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-games", type=int, default=500)
    args = parser.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    import transformer_lens as tl

    # Load OthelloGPT
    print("Loading OthelloGPT via TransformerLens...", flush=True)
    model = tl.HookedTransformer.from_pretrained("othello-gpt", device=args.device)
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    print(f"  {n_layers} layers, d_model={d_model}", flush=True)

    # Generate games
    print(f"\nGenerating {args.n_games} games...", flush=True)
    games = generate_games(args.n_games)
    print(f"  {len(games)} games, avg {np.mean([len(g['moves']) for g in games]):.0f} moves")

    # Prepare data: sequences + labels
    max_seq = model.cfg.n_ctx  # 59 for OthelloGPT
    print(f"  Max context: {max_seq}", flush=True)
    all_seqs, all_boards, all_feats = [], [], []
    for g in games:
        for t in range(5, min(len(g["moves"]), max_seq)):
            seq = g["moves"][:t]
            # Verify all tokens in range
            if any(tok >= model.cfg.d_vocab for tok in seq):
                continue
            board = g["states"][t]
            player = 1 if t%2==0 else -1
            all_seqs.append(torch.tensor(seq, dtype=torch.long))
            all_boards.append(board.flatten())
            all_feats.append(strategy_features(board, player))

    # Subsample
    if len(all_seqs) > 5000:
        idx = np.random.choice(len(all_seqs), 5000, replace=False)
        all_seqs = [all_seqs[i] for i in idx]
        all_boards = [all_boards[i] for i in idx]
        all_feats = [all_feats[i] for i in idx]
    n_samples = len(all_seqs)
    print(f"  {n_samples} positions", flush=True)

    # Cache hidden states via run_with_cache
    print("\nCaching hidden states...", flush=True)
    layer_h = {l: [] for l in range(n_layers)}

    for i in tqdm(range(n_samples), desc="Cache"):
        seq = all_seqs[i].unsqueeze(0).to(args.device)
        _, cache = model.run_with_cache(seq)
        for l in range(n_layers):
            h = cache[f"blocks.{l}.hook_resid_post"][0, -1].cpu().float().numpy()
            layer_h[l].append(h)

    for l in range(n_layers):
        layer_h[l] = np.stack(layer_h[l])
    print(f"  Done: {n_samples} × {n_layers} layers", flush=True)

    # ── 1. ID Profile ──
    print(f"\n{'='*60}\n1. ID Profile\n{'='*60}", flush=True)
    id_profile = []
    for l in range(n_layers):
        tid = twonn_id(layer_h[l]); sr = stable_rank(layer_h[l])
        id_profile.append({"layer": l, "twonn_id": tid, "stable_rank": sr})
        print(f"  L{l}: ID={tid:.1f}, srank={sr:.1f}")

    # ── 2. Board State Probe ──
    print(f"\n{'='*60}\n2. Board State Probe\n{'='*60}", flush=True)
    board_array = np.stack(all_boards)  # (n, 64), values -1/0/1
    board_accs = {}
    for l in range(n_layers):
        X = torch.tensor(layer_h[l], dtype=torch.float32).to(args.device)
        Y = torch.tensor(board_array + 1, dtype=torch.long).to(args.device)  # 0,1,2
        probe = nn.Linear(d_model, 64*3).to(args.device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        for _ in range(20):
            logits = probe(X).view(-1, 64, 3)
            loss = nn.CrossEntropyLoss()(logits.reshape(-1,3), Y.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            preds = probe(X).view(-1,64,3).argmax(-1)
            acc = (preds == Y).float().mean().item()
        board_accs[l] = acc

    best_bl = max(board_accs, key=board_accs.get)
    print(f"  Best: {board_accs[best_bl]:.3f} @ L{best_bl} (random={1/3:.3f})")

    # ── 3. Strategy Probes ──
    print(f"\n{'='*60}\n3. Strategy Probes\n{'='*60}", flush=True)
    strat_results = {}
    for fname in ["is_winning", "has_corner", "high_mobility"]:
        labels = np.array([f[fname] for f in all_feats])
        if labels.mean() < 0.05 or labels.mean() > 0.95:
            print(f"  {fname}: SKIP ({labels.mean():.3f})"); continue
        y = torch.tensor(labels, dtype=torch.long)
        accs = {}
        for l in range(n_layers):
            X = torch.tensor(layer_h[l], dtype=torch.float32).to(args.device)
            probe = nn.Linear(d_model, 1).to(args.device)
            opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
            yd = y.float().to(args.device)
            for _ in range(15):
                logits = probe(X).squeeze()
                loss = nn.BCEWithLogitsLoss()(logits, yd)
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                acc = ((probe(X).squeeze()>0).long() == y.to(args.device)).float().mean().item()
            accs[l] = acc
        best_l = max(accs, key=accs.get)
        strat_results[fname] = {"accs": accs, "best_layer": best_l, "best_acc": accs[best_l],
                                "pos_rate": float(labels.mean())}
        print(f"  {fname}: {accs[best_l]:.3f} @ L{best_l} (pos={labels.mean():.3f})")

    # ── Figures ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    layers = list(range(n_layers))
    axes[0,0].plot(layers, [p["twonn_id"] for p in id_profile], "o-", label="TwoNN ID")
    axes[0,0].plot(layers, [p["stable_rank"] for p in id_profile], "s-", label="Stable Rank")
    axes[0,0].set_xlabel("Layer"); axes[0,0].set_ylabel("Metric")
    axes[0,0].set_title("OthelloGPT ID Profile"); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

    axes[0,1].plot(layers, [board_accs[l] for l in layers], "o-", color="red")
    axes[0,1].axhline(1/3, color="gray", linestyle="--", label="Random")
    axes[0,1].set_xlabel("Layer"); axes[0,1].set_ylabel("Accuracy")
    axes[0,1].set_title("Board State Probe"); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

    for fname, r in strat_results.items():
        axes[1,0].plot(layers, [r["accs"][l] for l in layers], "o-", label=fname, markersize=4)
    axes[1,0].axhline(0.5, color="gray", linestyle="--")
    axes[1,0].set_xlabel("Layer"); axes[1,0].set_ylabel("Accuracy")
    axes[1,0].set_title("Strategy Probes"); axes[1,0].legend(fontsize=8); axes[1,0].grid(True, alpha=0.3)

    ids = [id_profile[l]["twonn_id"] for l in layers]
    ba = [board_accs[l] for l in layers]
    axes[1,1].scatter(ids, ba, s=60)
    for l in layers: axes[1,1].annotate(f"L{l}", (ids[l], ba[l]), fontsize=7)
    axes[1,1].set_xlabel("TwoNN ID"); axes[1,1].set_ylabel("Board Probe Acc")
    axes[1,1].set_title("ID vs Probe Accuracy"); axes[1,1].grid(True, alpha=0.3)

    fig.suptitle("OthelloGPT: Geometric Portrait + Probing", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "othello_probing.png", dpi=150)
    print(f"\nFigure: {out_dir / 'othello_probing.png'}")

    results = {
        "model": "othello-gpt (TransformerLens)", "n_layers": n_layers, "d_model": d_model,
        "n_samples": n_samples,
        "id_profile": id_profile,
        "board_probe": {"accs": {str(l): board_accs[l] for l in layers},
                        "best_layer": best_bl, "best_acc": board_accs[best_bl]},
        "strategy_probes": {k: {"best_acc": v["best_acc"], "best_layer": v["best_layer"],
                                "pos_rate": v["pos_rate"]}
                           for k, v in strat_results.items()},
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"  Board state: {board_accs[best_bl]:.3f} @ L{best_bl}")
    for fn, r in strat_results.items():
        print(f"  {fn}: {r['best_acc']:.3f} @ L{r['best_layer']}")
    has_hunch = any(id_profile[l]["twonn_id"] > id_profile[0]["twonn_id"]
                    and id_profile[l]["twonn_id"] > id_profile[-1]["twonn_id"]
                    for l in range(1, n_layers-1))
    print(f"  Hunchback: {'YES' if has_hunch else 'NO'}")

if __name__ == "__main__":
    main()
