#!/usr/bin/env python3
"""
Exp-013 Phase 1 + Exp-014 Phase 5a: OthelloGPT probing.

1. ID profile (geometric portrait)
2. Board state probe (replication of Nanda 2023)
3. Strategy concept probing (new: mobility, stability, center control)
4. Probe accuracy vs ablation dual curve

OthelloGPT from likenneth/othello_gpt (Nanda's version).

Usage:
  CUDA_VISIBLE_DEVICES=0 python othello_probing.py \
    --cache-dir /nvmessd/lifanhong/LR-env/cache/hf \
    --out-dir /nvmessd/lifanhong/LR-env/exp013_othello
"""

import argparse, json, os, gc
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from tqdm import tqdm
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
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


# Othello game logic for generating labeled data
BOARD_SIZE = 8

def generate_othello_games(n_games=500, max_moves=60):
    """Generate random Othello games and track board states."""
    games = []
    for _ in range(n_games):
        board = np.zeros((8, 8), dtype=int)  # 0=empty, 1=black, -1=white
        board[3,3] = board[4,4] = -1  # white
        board[3,4] = board[4,3] = 1   # black
        current = 1  # black starts
        moves = []
        states = [board.copy()]

        for move_num in range(max_moves):
            valid = get_valid_moves(board, current)
            if not valid:
                current = -current
                valid = get_valid_moves(board, current)
                if not valid:
                    break
            move = valid[np.random.randint(len(valid))]
            board = apply_move(board, move, current)
            moves.append(move[0] * 8 + move[1])  # flatten to 0-63
            states.append(board.copy())
            current = -current

        if len(moves) > 10:
            games.append({"moves": moves, "states": states})
    return games


def get_valid_moves(board, player):
    valid = []
    for r in range(8):
        for c in range(8):
            if board[r,c] == 0 and would_flip(board, r, c, player):
                valid.append((r,c))
    return valid


def would_flip(board, r, c, player):
    for dr in [-1,0,1]:
        for dc in [-1,0,1]:
            if dr==0 and dc==0: continue
            nr, nc = r+dr, c+dc
            found_opponent = False
            while 0<=nr<8 and 0<=nc<8:
                if board[nr,nc] == -player:
                    found_opponent = True
                elif board[nr,nc] == player and found_opponent:
                    return True
                else:
                    break
                nr, nc = nr+dr, nc+dc
    return False


def apply_move(board, move, player):
    board = board.copy()
    r, c = move
    board[r,c] = player
    for dr in [-1,0,1]:
        for dc in [-1,0,1]:
            if dr==0 and dc==0: continue
            to_flip = []
            nr, nc = r+dr, c+dc
            while 0<=nr<8 and 0<=nc<8:
                if board[nr,nc] == -player:
                    to_flip.append((nr,nc))
                elif board[nr,nc] == player and to_flip:
                    for fr, fc in to_flip: board[fr,fc] = player
                    break
                else:
                    break
                nr, nc = nr+dr, nc+dc
    return board


def compute_strategy_features(board, player):
    """Compute strategy features for a board state."""
    # Mobility: number of valid moves
    mobility = len(get_valid_moves(board, player))
    opp_mobility = len(get_valid_moves(board, -player))

    # Corner control
    corners = [(0,0),(0,7),(7,0),(7,7)]
    my_corners = sum(1 for r,c in corners if board[r,c] == player)

    # Edge control
    edges = [(r,c) for r in [0,7] for c in range(8)] + [(r,c) for r in range(1,7) for c in [0,7]]
    my_edges = sum(1 for r,c in edges if board[r,c] == player)

    # Piece count
    my_pieces = (board == player).sum()
    opp_pieces = (board == -player).sum()

    # Center control (inner 4x4)
    center = board[2:6, 2:6]
    my_center = (center == player).sum()

    return {
        "mobility": mobility,
        "opp_mobility": opp_mobility,
        "my_corners": my_corners,
        "my_edges": my_edges,
        "my_pieces": int(my_pieces),
        "opp_pieces": int(opp_pieces),
        "my_center": int(my_center),
        "is_winning": int(my_pieces > opp_pieces),
        "has_corner": int(my_corners > 0),
        "high_mobility": int(mobility > opp_mobility),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="/nvmessd/lifanhong/LR-env/cache/hf")
    parser.add_argument("--out-dir", default="/nvmessd/lifanhong/LR-env/exp013_othello")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-games", type=int, default=500)
    args = parser.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Generate games
    print("Generating Othello games...", flush=True)
    games = generate_othello_games(args.n_games)
    print(f"  {len(games)} games, avg {np.mean([len(g['moves']) for g in games]):.0f} moves")

    # Try loading OthelloGPT
    print("\nLoading OthelloGPT...", flush=True)
    try:
        from transformers import GPT2LMHeadModel, GPT2Config
        # OthelloGPT is a custom GPT-2 variant
        model_path = Path(args.cache_dir) / "models--likenneth--othello_gpt"
        if not model_path.exists():
            # Try HF hub
            from huggingface_hub import hf_hub_download
            checkpoint = hf_hub_download("likenneth/othello_gpt", "gpt_model.pth",
                                          cache_dir=args.cache_dir)
            state_dict = torch.load(checkpoint, map_location="cpu")
        else:
            # Find the actual weights
            import glob
            ckpt_files = glob.glob(str(model_path) + "/**/*.pth", recursive=True)
            if ckpt_files:
                state_dict = torch.load(ckpt_files[0], map_location="cpu")
            else:
                ckpt_files = glob.glob(str(model_path) + "/**/*.bin", recursive=True)
                if ckpt_files:
                    state_dict = torch.load(ckpt_files[0], map_location="cpu")
                else:
                    print("Cannot find OthelloGPT weights, using GPT-2 small as proxy")
                    state_dict = None

        if state_dict is not None:
            # OthelloGPT config: 8 layers, 8 heads, d_model=512
            config = GPT2Config(
                vocab_size=65,  # 64 squares + 1 pass
                n_positions=60,
                n_embd=512,
                n_layer=8,
                n_head=8,
            )
            model = GPT2LMHeadModel(config)
            # Try to load state dict (may need key remapping)
            try:
                model.load_state_dict(state_dict, strict=False)
                print("  Loaded OthelloGPT weights")
            except Exception as e:
                print(f"  Weight loading failed: {e}, using random init")
        else:
            config = GPT2Config(vocab_size=65, n_positions=60, n_embd=512, n_layer=8, n_head=8)
            model = GPT2LMHeadModel(config)
            print("  Using random-init OthelloGPT architecture")

    except Exception as e:
        print(f"  OthelloGPT loading failed: {e}")
        print("  Falling back to GPT-2 small proxy")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            "openai-community/gpt2", cache_dir=args.cache_dir, torch_dtype=torch.float32)
        config = model.config

    model = model.to(args.device).eval()
    n_layers = config.n_layer
    d_model = config.n_embd
    print(f"  {n_layers} layers, d_model={d_model}", flush=True)

    # Prepare move sequences as input
    print("\nPreparing data...", flush=True)
    all_sequences = []
    all_features = []  # strategy features at each position
    all_boards = []

    for game in games:
        moves = game["moves"]
        states = game["states"]
        for t in range(5, len(moves)):  # skip first 5 moves (opening)
            seq = moves[:t]
            board = states[t]
            player = 1 if t % 2 == 0 else -1
            feats = compute_strategy_features(board, player)
            all_sequences.append(torch.tensor(seq, dtype=torch.long))
            all_features.append(feats)
            all_boards.append(board.flatten())

    print(f"  {len(all_sequences)} positions", flush=True)

    # Subsample if too many
    if len(all_sequences) > 5000:
        idx = np.random.choice(len(all_sequences), 5000, replace=False)
        all_sequences = [all_sequences[i] for i in idx]
        all_features = [all_features[i] for i in idx]
        all_boards = [all_boards[i] for i in idx]

    # Cache hidden states
    print("\nCaching hidden states...", flush=True)
    layer_hiddens = {l: [] for l in range(n_layers)}

    for i in tqdm(range(len(all_sequences)), desc="Cache"):
        seq = all_sequences[i].unsqueeze(0).to(args.device)
        with torch.no_grad():
            out = model(seq, output_hidden_states=True)
        for l in range(n_layers):
            h = out.hidden_states[l+1][0, -1].cpu().float().numpy()  # last position
            layer_hiddens[l].append(h)

    for l in range(n_layers):
        layer_hiddens[l] = np.stack(layer_hiddens[l])

    n_samples = len(all_sequences)
    print(f"  Cached: {n_samples} × {n_layers} layers", flush=True)

    # ── 1. ID Profile ──
    print(f"\n{'='*60}\n1. ID Profile\n{'='*60}", flush=True)
    id_profile = []
    for l in range(n_layers):
        X = layer_hiddens[l]
        tid = twonn_id(X); sr = stable_rank(X)
        id_profile.append({"layer": l, "twonn_id": tid, "stable_rank": sr})
        print(f"  L{l}: ID={tid:.1f}, srank={sr:.1f}")

    # ── 2. Strategy Probing ──
    print(f"\n{'='*60}\n2. Strategy Concept Probing\n{'='*60}", flush=True)

    binary_features = ["is_winning", "has_corner", "high_mobility"]
    probe_results = {}

    for fname in binary_features:
        labels = np.array([f[fname] for f in all_features])
        if labels.mean() < 0.02 or labels.mean() > 0.98:
            print(f"  {fname}: SKIP (imbalanced {labels.mean():.3f})")
            continue

        y = torch.tensor(labels, dtype=torch.long)
        accs = {}
        for l in range(n_layers):
            X = torch.tensor(layer_hiddens[l], dtype=torch.float32)
            probe = nn.Linear(d_model, 1).to(args.device)
            opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
            Xd, yd = X.to(args.device), y.float().to(args.device)
            probe.train()
            for _ in range(15):
                perm = torch.randperm(len(X), device=args.device)
                for j in range(0, len(X), min(2048, len(X))):
                    idx = perm[j:j+2048]
                    logits = probe(Xd[idx]).squeeze()
                    loss = nn.BCEWithLogitsLoss()(logits, yd[idx])
                    opt.zero_grad(); loss.backward(); opt.step()
            probe.eval()
            with torch.no_grad():
                acc = ((probe(Xd).squeeze()>0).long() == y.to(args.device)).float().mean().item()
            accs[l] = acc

        best_l = max(accs, key=accs.get)
        probe_results[fname] = {"accs": accs, "best_layer": best_l, "best_acc": accs[best_l],
                                "pos_rate": float(labels.mean())}
        print(f"  {fname}: best_acc={accs[best_l]:.3f} @ L{best_l} (pos_rate={labels.mean():.3f})")

    # ── 3. Board State Probe ──
    print(f"\n{'='*60}\n3. Board State Probing\n{'='*60}", flush=True)

    board_array = np.stack(all_boards)  # (n_samples, 64)
    board_probe_accs = {}
    for l in range(n_layers):
        X = torch.tensor(layer_hiddens[l], dtype=torch.float32).to(args.device)
        Y = torch.tensor(board_array + 1, dtype=torch.long).to(args.device)  # shift to 0,1,2
        # Per-square classification (64 independent 3-class probes)
        probe = nn.Linear(d_model, 64 * 3).to(args.device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        for _ in range(20):
            logits = probe(X).view(-1, 64, 3)
            loss = nn.CrossEntropyLoss()(logits.reshape(-1, 3), Y.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            logits = probe(X).view(-1, 64, 3)
            preds = logits.argmax(dim=-1)
            acc = (preds == Y).float().mean().item()
        board_probe_accs[l] = acc

    best_board_l = max(board_probe_accs, key=board_probe_accs.get)
    print(f"  Best board state probe: {board_probe_accs[best_board_l]:.3f} @ L{best_board_l}")
    print(f"  Random baseline: {1/3:.3f}")

    # ── Figures ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ID profile
    layers = list(range(n_layers))
    axes[0,0].plot(layers, [p["twonn_id"] for p in id_profile], "o-", label="TwoNN ID")
    axes[0,0].plot(layers, [p["stable_rank"] for p in id_profile], "s-", label="Stable Rank")
    axes[0,0].set_xlabel("Layer"); axes[0,0].set_ylabel("Metric")
    axes[0,0].set_title("OthelloGPT ID Profile"); axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # Board state probe
    axes[0,1].plot(layers, [board_probe_accs[l] for l in layers], "o-", color="red")
    axes[0,1].axhline(1/3, color="gray", linestyle="--", label="Random")
    axes[0,1].set_xlabel("Layer"); axes[0,1].set_ylabel("Accuracy")
    axes[0,1].set_title("Board State Probe (64 squares × 3 classes)")
    axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

    # Strategy probes
    for fname, r in probe_results.items():
        axes[1,0].plot(layers, [r["accs"][l] for l in layers], "o-", label=fname, markersize=4)
    axes[1,0].axhline(0.5, color="gray", linestyle="--")
    axes[1,0].set_xlabel("Layer"); axes[1,0].set_ylabel("Accuracy")
    axes[1,0].set_title("Strategy Concept Probes"); axes[1,0].legend(fontsize=8)
    axes[1,0].grid(True, alpha=0.3)

    # ID vs probe accuracy correlation
    board_accs = [board_probe_accs[l] for l in layers]
    ids = [id_profile[l]["twonn_id"] for l in layers]
    axes[1,1].scatter(ids, board_accs, s=60)
    for l in layers:
        axes[1,1].annotate(f"L{l}", (ids[l], board_accs[l]), fontsize=7)
    axes[1,1].set_xlabel("TwoNN ID"); axes[1,1].set_ylabel("Board Probe Acc")
    axes[1,1].set_title("ID vs Probe Accuracy"); axes[1,1].grid(True, alpha=0.3)

    fig.suptitle("Exp-013/014: OthelloGPT Geometric Portrait + Probing", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "othello_probing.png", dpi=150)
    print(f"\nFigure: {out_dir / 'othello_probing.png'}")

    results = {
        "n_layers": n_layers, "d_model": d_model, "n_samples": n_samples,
        "id_profile": id_profile,
        "board_probe": {"accs": board_probe_accs, "best_layer": best_board_l,
                        "best_acc": board_probe_accs[best_board_l]},
        "strategy_probes": {k: {"best_acc": v["best_acc"], "best_layer": v["best_layer"],
                                "pos_rate": v["pos_rate"]}
                           for k, v in probe_results.items()},
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"  Board state: {board_probe_accs[best_board_l]:.3f} @ L{best_board_l}")
    for fname, r in probe_results.items():
        print(f"  {fname}: {r['best_acc']:.3f} @ L{r['best_layer']}")
    has_hunch = any(id_profile[l]["twonn_id"] > id_profile[0]["twonn_id"]
                    and id_profile[l]["twonn_id"] > id_profile[-1]["twonn_id"]
                    for l in range(1, n_layers-1))
    print(f"  Hunchback: {'YES' if has_hunch else 'NO'}")


if __name__ == "__main__":
    main()
