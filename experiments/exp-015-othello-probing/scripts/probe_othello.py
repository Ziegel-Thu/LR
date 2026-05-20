#!/usr/bin/env python3
"""
Exp-015: OthelloGPT Strategy Probing

Probes OthelloGPT for board state AND strategic concepts:
- Board state (64 squares × 3 classes) — known baseline
- Tile stability (is each disc stable/unstable)
- Frontier discs (adjacent to empty squares)
- Mobility (number of legal moves)
- Corner/edge control

Uses TransformerLens for clean activation extraction.
"""

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


# ── Othello game logic ───────────────────────────────────────────────────────

BOARD_SIZE = 8
DIRECTIONS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

# OthelloGPT token encoding: 64 board positions → 60 tokens (skip 4 center squares)
CENTER_SQUARES = {27, 28, 35, 36}  # d4, e4, d5, e5

def board_pos_to_token(pos):
    """Convert board position (0-63) to OthelloGPT token (0-59)."""
    assert pos not in CENTER_SQUARES, f"Center square {pos} is never a valid move"
    return pos - sum(1 for c in CENTER_SQUARES if c < pos)

def token_to_board_pos(tok):
    """Convert OthelloGPT token (0-59) to board position (0-63)."""
    pos = tok
    for c in sorted(CENTER_SQUARES):
        if pos >= c:
            pos += 1
    return pos

class OthelloBoard:
    """Minimal Othello game engine for label generation."""
    
    def __init__(self):
        self.board = np.zeros((8, 8), dtype=int)  # 0=empty, 1=black, -1=white
        self.board[3,3] = self.board[4,4] = -1  # white
        self.board[3,4] = self.board[4,3] = 1   # black
        self.turn = 1  # black moves first
    
    def _valid_moves_for(self, player):
        moves = []
        for r in range(8):
            for c in range(8):
                if self.board[r,c] != 0:
                    continue
                for dr, dc in DIRECTIONS:
                    nr, nc = r+dr, c+dc
                    found_opp = False
                    while 0<=nr<8 and 0<=nc<8 and self.board[nr,nc] == -player:
                        nr += dr; nc += dc
                        found_opp = True
                    if found_opp and 0<=nr<8 and 0<=nc<8 and self.board[nr,nc] == player:
                        moves.append(r*8+c)
                        break
        return moves
    
    def play(self, move):
        """Play move (0-63 flattened index). Returns True if valid."""
        r, c = move // 8, move % 8
        if self.board[r,c] != 0:
            return False
        player = self.turn
        flipped = []
        for dr, dc in DIRECTIONS:
            nr, nc = r+dr, c+dc
            path = []
            while 0<=nr<8 and 0<=nc<8 and self.board[nr,nc] == -player:
                path.append((nr,nc))
                nr += dr; nc += dc
            if path and 0<=nr<8 and 0<=nc<8 and self.board[nr,nc] == player:
                flipped.extend(path)
        if not flipped:
            return False
        self.board[r,c] = player
        for fr, fc in flipped:
            self.board[fr,fc] = player
        self.turn = -self.turn
        # If opponent has no moves, turn passes back
        if not self._valid_moves_for(self.turn):
            self.turn = -self.turn
        return True
    
    def get_state(self):
        """Return board state: 0=white, 1=empty, 2=black (matches Li et al.)"""
        return (self.board + 1).flatten().tolist()
    
    def get_frontier(self):
        """Return frontier status for each square (1 if disc adjacent to empty)."""
        frontier = np.zeros(64, dtype=int)
        for r in range(8):
            for c in range(8):
                if self.board[r,c] == 0:
                    continue
                for dr, dc in DIRECTIONS:
                    nr, nc = r+dr, c+dc
                    if 0<=nr<8 and 0<=nc<8 and self.board[nr,nc] == 0:
                        frontier[r*8+c] = 1
                        break
        return frontier.tolist()
    
    def get_stability(self):
        """Simplified stability: corners are stable, edges with full lines are stable."""
        stable = np.zeros(64, dtype=int)
        corners = [(0,0),(0,7),(7,0),(7,7)]
        for r,c in corners:
            if self.board[r,c] != 0:
                stable[r*8+c] = 1
        return stable.tolist()
    
    def get_mobility(self):
        """Return number of legal moves for current player."""
        return len(self._valid_moves_for(self.turn))
    
    def get_disc_count(self):
        """Return (black_count, white_count)."""
        return int((self.board == 1).sum()), int((self.board == -1).sum())


def generate_games(n_games=10000, seed=42):
    """Generate random Othello games, return move sequences."""
    rng = np.random.RandomState(seed)
    games = []
    for _ in range(n_games):
        board = OthelloBoard()
        moves = []
        for _ in range(60):
            valid = board._valid_moves_for(board.turn)
            if not valid:
                break
            move = valid[rng.randint(len(valid))]
            board.play(move)
            moves.append(move)
        if len(moves) >= 10:
            games.append(moves)
    return games


def collect_labels(games, max_positions=50000):
    """Collect board states and strategic labels from games."""
    positions = []
    labels_state = []      # [n_pos, 64] board state
    labels_frontier = []   # [n_pos, 64] frontier discs
    labels_mobility = []   # [n_pos] mobility count
    move_seqs = []         # [n_pos] the move sequence up to that point
    
    for game in games:
        board = OthelloBoard()
        for t, move in enumerate(game):
            board.play(move)
            if t < 4:  # skip opening
                continue
            labels_state.append(board.get_state())
            labels_frontier.append(board.get_frontier())
            labels_mobility.append(board.get_mobility())
            move_seqs.append(game[:t+1])
            
            if len(labels_state) >= max_positions:
                break
        if len(labels_state) >= max_positions:
            break
    
    return {
        'state': np.array(labels_state),        # [N, 64]
        'frontier': np.array(labels_frontier),   # [N, 64]
        'mobility': np.array(labels_mobility),   # [N]
        'move_seqs': move_seqs,
    }


# ── Probing ──────────────────────────────────────────────────────────────────

def train_board_probe(hiddens, labels, n_classes=3):
    """Train 64 parallel linear probes for board state (one per square)."""
    from sklearn.linear_model import LogisticRegression
    
    n_samples, d = hiddens.shape
    n_squares = labels.shape[1]
    
    # Split
    split = int(0.8 * n_samples)
    X_tr, X_te = hiddens[:split], hiddens[split:]
    y_tr, y_te = labels[:split], labels[split:]
    
    accs = []
    for sq in range(n_squares):
        clf = LogisticRegression(max_iter=200, C=1.0, solver='lbfgs', 
                                  multi_class='multinomial')
        clf.fit(X_tr, y_tr[:, sq])
        acc = clf.score(X_te, y_te[:, sq])
        accs.append(acc)
    
    return np.mean(accs), accs


def train_binary_probe(hiddens, labels):
    """Train single linear probe for binary classification."""
    from sklearn.linear_model import LogisticRegression
    
    n_samples = hiddens.shape[0]
    split = int(0.8 * n_samples)
    X_tr, X_te = hiddens[:split], hiddens[split:]
    y_tr, y_te = labels[:split], labels[split:]
    
    clf = LogisticRegression(max_iter=200, C=1.0)
    clf.fit(X_tr, y_tr)
    return clf.score(X_te, y_te)


def train_regression_probe(hiddens, labels):
    """Train linear regression probe for continuous labels."""
    from sklearn.linear_model import Ridge
    
    n_samples = hiddens.shape[0]
    split = int(0.8 * n_samples)
    X_tr, X_te = hiddens[:split], hiddens[split:]
    y_tr, y_te = labels[:split].astype(float), labels[split:].astype(float)
    
    reg = Ridge(alpha=1.0)
    reg.fit(X_tr, y_tr)
    y_pred = reg.predict(X_te)
    r2 = 1 - np.sum((y_te - y_pred)**2) / np.sum((y_te - y_te.mean())**2)
    return float(r2)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OthelloGPT strategy probing")
    parser.add_argument("--n-games", type=int, default=5000)
    parser.add_argument("--max-positions", type=int, default=20000)
    parser.add_argument("--output-dir", type=str, 
                        default="/nvmessd/lifanhong/LR-env/exp015-othello")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model", type=str, default="synthetic",
                        choices=["synthetic", "championship"])
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load OthelloGPT
    print("Loading OthelloGPT...", flush=True)
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    
    cfg = HookedTransformerConfig(
        n_layers=8, d_model=512, d_head=64, n_heads=8, d_mlp=2048,
        d_vocab=61, n_ctx=59, act_fn="gelu", normalization_type="LNPre"
    )
    model = HookedTransformer(cfg)
    
    # Download weights
    from transformer_lens.utils import download_file_from_hf
    fname = "synthetic_model.pth" if args.model == "synthetic" else "championship_model.pth"
    sd = download_file_from_hf("NeelNanda/Othello-GPT-Transformer-Lens", fname)
    model.load_state_dict(sd)
    model = model.to(args.device).eval()
    print(f"  Loaded {args.model} model: 8 layers, d=512", flush=True)
    
    # 2. Generate games and labels
    print(f"Generating {args.n_games} games...", flush=True)
    games = generate_games(args.n_games)
    print(f"  Generated {len(games)} valid games", flush=True)
    
    print(f"Collecting labels (max {args.max_positions} positions)...", flush=True)
    data = collect_labels(games, args.max_positions)
    n_pos = len(data['state'])
    print(f"  Collected {n_pos} positions", flush=True)
    
    # 3. Extract hidden states per layer
    print("Extracting hidden states...", flush=True)
    
    # Pad move sequences to same length and batch (max 59 = OthelloGPT context)
    max_len = min(max(len(s) for s in data['move_seqs']), 59)
    all_hiddens = {l: [] for l in range(9)}  # 8 layers + embed
    
    batch_size = 64
    for i in tqdm(range(0, n_pos, batch_size), desc="  Extracting"):
        batch_seqs = data['move_seqs'][i:i+batch_size]
        # Each seq is a list of moves (ints 0-63)
        # OthelloGPT vocab: moves 0-59 map to tokens, need to check encoding
        # In Li et al., moves are encoded as-is (0-60 valid moves)
        
        # Pad sequences — convert board positions to OthelloGPT tokens
        batch_lens = [min(len(s), max_len) for s in batch_seqs]
        padded = torch.zeros(len(batch_seqs), max_len, dtype=torch.long)
        for j, seq in enumerate(batch_seqs):
            tokens = [board_pos_to_token(m) for m in seq[:max_len]]
            padded[j, :len(tokens)] = torch.tensor(tokens)
        
        padded = padded.to(args.device)
        
        with torch.no_grad():
            _, cache = model.run_with_cache(padded)
        
        # Extract last-position activations (the position we have labels for)
        for l in range(8):
            acts = cache[f"blocks.{l}.hook_resid_post"]  # [B, T, 512]
            # Get activation at the last real position for each sequence
            last_acts = torch.stack([acts[j, batch_lens[j]-1] for j in range(len(batch_seqs))])
            all_hiddens[l].append(last_acts.cpu().numpy())
        
        # Also get embedding layer
        acts_emb = cache["hook_embed"] + cache["hook_pos_embed"]
        last_emb = torch.stack([acts_emb[j, batch_lens[j]-1] for j in range(len(batch_seqs))])
        all_hiddens[8].append(last_emb.cpu().numpy())
    
    # Concatenate
    for l in all_hiddens:
        all_hiddens[l] = np.concatenate(all_hiddens[l], axis=0)
    print(f"  Hidden states shape: {all_hiddens[0].shape}", flush=True)
    
    # 4. Probe each layer
    results = {}
    
    for layer in range(9):
        layer_name = f"L{layer}" if layer < 8 else "embed"
        print(f"\nProbing {layer_name}...", flush=True)
        H = all_hiddens[layer]
        
        # Board state probe (64 squares × 3 classes)
        state_acc, per_square_accs = train_board_probe(H, data['state'])
        print(f"  Board state: {state_acc:.4f} (mean over 64 squares)", flush=True)
        
        # Frontier probe (64 binary labels)
        frontier_accs = []
        for sq in range(64):
            if data['frontier'][:, sq].mean() < 0.02 or data['frontier'][:, sq].mean() > 0.98:
                continue
            acc = train_binary_probe(H, data['frontier'][:, sq])
            frontier_accs.append(acc)
        frontier_acc = np.mean(frontier_accs) if frontier_accs else 0
        print(f"  Frontier discs: {frontier_acc:.4f} (mean over {len(frontier_accs)} squares)", flush=True)
        
        # Mobility probe (regression)
        mobility_r2 = train_regression_probe(H, data['mobility'])
        print(f"  Mobility: R²={mobility_r2:.4f}", flush=True)
        
        results[layer_name] = {
            "board_state_acc": float(state_acc),
            "board_state_per_square": [float(a) for a in per_square_accs],
            "frontier_acc": float(frontier_acc),
            "frontier_n_squares": len(frontier_accs),
            "mobility_r2": float(mobility_r2),
        }
    
    # 5. Save results
    summary = {
        "model": args.model,
        "n_games": len(games),
        "n_positions": n_pos,
        "results": results,
    }
    
    with open(out_dir / "probe_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Layer':<8} {'Board State':<14} {'Frontier':<12} {'Mobility R²':<12}")
    print("-"*46)
    for layer in range(9):
        ln = f"L{layer}" if layer < 8 else "embed"
        r = results[ln]
        print(f"{ln:<8} {r['board_state_acc']:.4f}         {r['frontier_acc']:.4f}       {r['mobility_r2']:.4f}")
    
    print(f"\nResults saved to {out_dir}/probe_results.json")


if __name__ == "__main__":
    main()
