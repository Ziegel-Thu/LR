#!/usr/bin/env python3
"""
Exp-015 Phase 3: Championship vs Synthetic OthelloGPT + Tile Stability Probing

Phase 1-2 found (on SYNTHETIC model):
  - Linear probe board state: 72%
  - MLP probe board state: 87.7%
  - Frontier discs: 98.3% @ L4
  - Mobility: ghost representation (encoded but not causally used)

This script:
  Part 1: Load championship model, run same linear probes, compare with synthetic
  Part 2: Tile stability probing (proper edge-propagation computation)
  
Hypothesis: Championship model (trained on real games) should have BETTER
strategic representations than synthetic (trained on random play).
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


# ── Othello game logic ───────────────────────────────────────────────────────

BOARD_SIZE = 8
DIRECTIONS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
CENTER_SQUARES = {27, 28, 35, 36}


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
        self.board[3, 3] = self.board[4, 4] = -1  # white
        self.board[3, 4] = self.board[4, 3] = 1   # black
        self.turn = 1  # black moves first

    def _valid_moves_for(self, player):
        moves = []
        for r in range(8):
            for c in range(8):
                if self.board[r, c] != 0:
                    continue
                for dr, dc in DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    found_opp = False
                    while 0 <= nr < 8 and 0 <= nc < 8 and self.board[nr, nc] == -player:
                        nr += dr; nc += dc
                        found_opp = True
                    if found_opp and 0 <= nr < 8 and 0 <= nc < 8 and self.board[nr, nc] == player:
                        moves.append(r * 8 + c)
                        break
        return moves

    def play(self, move):
        """Play move (0-63 flattened index). Returns True if valid."""
        r, c = move // 8, move % 8
        if self.board[r, c] != 0:
            return False
        player = self.turn
        flipped = []
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            path = []
            while 0 <= nr < 8 and 0 <= nc < 8 and self.board[nr, nc] == -player:
                path.append((nr, nc))
                nr += dr; nc += dc
            if path and 0 <= nr < 8 and 0 <= nc < 8 and self.board[nr, nc] == player:
                flipped.extend(path)
        if not flipped:
            return False
        self.board[r, c] = player
        for fr, fc in flipped:
            self.board[fr, fc] = player
        self.turn = -self.turn
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
                if self.board[r, c] == 0:
                    continue
                for dr, dc in DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 8 and 0 <= nc < 8 and self.board[nr, nc] == 0:
                        frontier[r * 8 + c] = 1
                        break
        return frontier.tolist()

    def get_mobility(self):
        """Return number of legal moves for current player."""
        return len(self._valid_moves_for(self.turn))


# ── Tile stability computation ───────────────────────────────────────────────

def compute_stability(board):
    """Compute stability for each occupied square.

    A disc is "stable" if it can never be flipped. Stability propagates:
      1. Corners are always stable once occupied.
      2. An occupied edge square is stable if it connects to a stable disc
         of the same color along that edge.
      3. Interior squares are stable if, for ALL four axes (horizontal,
         vertical, two diagonals), at least one of:
         - the entire line through the square in that axis direction is full, OR
         - the square is anchored to a stable same-color disc in that axis.

    This implements the full directional-anchor algorithm (simplified but
    much better than corners-only).

    Args:
        board: np.array of shape (8, 8), values in {-1, 0, 1}.

    Returns:
        stable: np.array of shape (64,), binary (1=stable, 0=not stable).
    """
    stable = np.zeros((8, 8), dtype=int)

    # The four axis directions (each pair covers one axis)
    AXES = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for _ in range(8):  # iterate until convergence
        changed = False
        for r in range(8):
            for c in range(8):
                if stable[r, c] or board[r, c] == 0:
                    continue
                color = board[r, c]

                # A square is stable if it is anchored in ALL 4 axes
                all_axes_stable = True
                for dr, dc in AXES:
                    # Check both directions along this axis
                    axis_stable = False

                    # Option A: entire line in this axis is full (no empty squares)
                    all_full = True
                    # Walk positive direction
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < 8 and 0 <= nc < 8:
                        if board[nr, nc] == 0:
                            all_full = False
                            break
                        nr += dr; nc += dc
                    # Walk negative direction
                    if all_full:
                        nr, nc = r - dr, c - dc
                        while 0 <= nr < 8 and 0 <= nc < 8:
                            if board[nr, nc] == 0:
                                all_full = False
                                break
                            nr -= dr; nc -= dc
                    if all_full:
                        axis_stable = True

                    # Option B: anchored to a stable same-color disc in one direction
                    if not axis_stable:
                        # Check positive direction: unbroken same-color chain to stable
                        nr, nc = r + dr, c + dc
                        while 0 <= nr < 8 and 0 <= nc < 8 and board[nr, nc] == color:
                            if stable[nr, nc]:
                                axis_stable = True
                                break
                            nr += dr; nc += dc

                    if not axis_stable:
                        # Check negative direction
                        nr, nc = r - dr, c - dc
                        while 0 <= nr < 8 and 0 <= nc < 8 and board[nr, nc] == color:
                            if stable[nr, nc]:
                                axis_stable = True
                                break
                            nr -= dr; nc -= dc

                    if not axis_stable:
                        all_axes_stable = False
                        break

                if all_axes_stable:
                    stable[r, c] = 1
                    changed = True

        if not changed:
            break

    return stable.flatten()


# ── Data generation ──────────────────────────────────────────────────────────

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
    """Collect board states and all strategic labels including stability."""
    labels = defaultdict(list)
    move_seqs = []

    for game in games:
        board = OthelloBoard()
        for t, move in enumerate(game):
            board.play(move)
            if t < 4:  # skip opening
                continue

            labels['state'].append(board.get_state())
            labels['frontier'].append(board.get_frontier())
            labels['mobility'].append(board.get_mobility())

            # Tile stability
            stab = compute_stability(board.board)
            labels['stability'].append(stab.tolist())

            move_seqs.append(game[:t + 1])

            if len(labels['state']) >= max_positions:
                break
        if len(labels['state']) >= max_positions:
            break

    return {
        'state': np.array(labels['state']),
        'frontier': np.array(labels['frontier']),
        'mobility': np.array(labels['mobility']),
        'stability': np.array(labels['stability']),
        'move_seqs': move_seqs,
    }


# ── Probing ──────────────────────────────────────────────────────────────────

def train_board_probe(hiddens, labels, n_classes=3):
    """Train 64 parallel linear probes for board state (one per square)."""
    from sklearn.linear_model import LogisticRegression

    n_samples, d = hiddens.shape
    split = int(0.8 * n_samples)
    X_tr, X_te = hiddens[:split], hiddens[split:]
    y_tr, y_te = labels[:split], labels[split:]

    accs = []
    for sq in range(64):
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
    ss_res = np.sum((y_te - y_pred) ** 2)
    ss_tot = np.sum((y_te - y_te.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(r2)


def train_stability_probes(hiddens, stability_labels):
    """Train 64 binary probes for tile stability (one per square).

    Only train probes on squares that have sufficient variance (at least 2%
    positive rate). Returns mean accuracy over valid squares plus per-square list.
    """
    from sklearn.linear_model import LogisticRegression

    n_samples, d = hiddens.shape
    split = int(0.8 * n_samples)
    X_tr, X_te = hiddens[:split], hiddens[split:]
    y_tr, y_te = stability_labels[:split], stability_labels[split:]

    accs = []
    valid_squares = []
    for sq in range(64):
        col = stability_labels[:, sq]
        pos_rate = col.mean()
        if pos_rate < 0.02 or pos_rate > 0.98:
            accs.append(None)
            continue

        clf = LogisticRegression(max_iter=200, C=1.0)
        clf.fit(X_tr, y_tr[:, sq])
        acc = clf.score(X_te, y_te[:, sq])
        accs.append(acc)
        valid_squares.append(sq)

    valid_accs = [a for a in accs if a is not None]
    mean_acc = np.mean(valid_accs) if valid_accs else 0.0
    return mean_acc, accs, valid_squares


# ── Model loading & activation extraction ────────────────────────────────────

def load_othello_model(model_type, device='cuda:0'):
    """Load OthelloGPT model (synthetic or championship)."""
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    from transformer_lens.utils import download_file_from_hf

    cfg = HookedTransformerConfig(
        n_layers=8, d_model=512, d_head=64, n_heads=8, d_mlp=2048,
        d_vocab=61, n_ctx=59, act_fn="gelu", normalization_type="LNPre",
    )
    model = HookedTransformer(cfg)

    fname = "synthetic_model.pth" if model_type == "synthetic" else "championship_model.pth"
    sd = download_file_from_hf("NeelNanda/Othello-GPT-Transformer-Lens", fname)
    model.load_state_dict(sd)
    model = model.to(device).eval()
    return model


def extract_hidden_states(model, move_seqs, device='cuda:0', batch_size=64):
    """Extract per-layer hidden states at the last real position."""
    n_pos = len(move_seqs)
    max_len = min(max(len(s) for s in move_seqs), 59)
    all_hiddens = {l: [] for l in range(9)}  # L0-L7 + embed

    for i in tqdm(range(0, n_pos, batch_size), desc="  Extracting"):
        batch_seqs = move_seqs[i:i + batch_size]
        batch_lens = [min(len(s), max_len) for s in batch_seqs]

        padded = torch.zeros(len(batch_seqs), max_len, dtype=torch.long)
        for j, seq in enumerate(batch_seqs):
            tokens = [board_pos_to_token(m) for m in seq[:max_len]]
            padded[j, :len(tokens)] = torch.tensor(tokens)
        padded = padded.to(device)

        with torch.no_grad():
            _, cache = model.run_with_cache(padded)

        for l in range(8):
            acts = cache[f"blocks.{l}.hook_resid_post"]
            last_acts = torch.stack(
                [acts[j, batch_lens[j] - 1] for j in range(len(batch_seqs))]
            )
            all_hiddens[l].append(last_acts.cpu().numpy())

        acts_emb = cache["hook_embed"] + cache["hook_pos_embed"]
        last_emb = torch.stack(
            [acts_emb[j, batch_lens[j] - 1] for j in range(len(batch_seqs))]
        )
        all_hiddens[8].append(last_emb.cpu().numpy())

    for l in all_hiddens:
        all_hiddens[l] = np.concatenate(all_hiddens[l], axis=0)

    return all_hiddens


# ── Probe one model ──────────────────────────────────────────────────────────

def probe_model(model_name, all_hiddens, data):
    """Run all linear probes on a model's hidden states. Returns results dict."""
    results = {}
    stability_stats = data['stability'].mean(axis=0)
    n_stable_total = int((data['stability'] > 0).any(axis=1).sum())
    total_stable_discs = int(data['stability'].sum())
    print(f"\n  Stability label stats: {n_stable_total}/{len(data['stability'])} "
          f"positions have ≥1 stable disc, {total_stable_discs} total stable discs",
          flush=True)

    for layer in range(9):
        layer_name = f"L{layer}" if layer < 8 else "embed"
        print(f"\n  [{model_name}] Probing {layer_name}...", flush=True)
        H = all_hiddens[layer]

        # Board state probe (64 squares × 3 classes)
        state_acc, per_sq_accs = train_board_probe(H, data['state'])
        print(f"    Board state: {state_acc:.4f}", flush=True)

        # Frontier probe (64 binary)
        frontier_accs = []
        for sq in range(64):
            col = data['frontier'][:, sq]
            if col.mean() < 0.02 or col.mean() > 0.98:
                continue
            acc = train_binary_probe(H, col)
            frontier_accs.append(acc)
        frontier_acc = np.mean(frontier_accs) if frontier_accs else 0
        print(f"    Frontier discs: {frontier_acc:.4f} "
              f"({len(frontier_accs)} squares)", flush=True)

        # Mobility probe (regression)
        mobility_r2 = train_regression_probe(H, data['mobility'])
        print(f"    Mobility: R²={mobility_r2:.4f}", flush=True)

        # Stability probe (64 binary)
        stab_acc, stab_per_sq, stab_valid_sqs = train_stability_probes(
            H, data['stability']
        )
        print(f"    Stability: {stab_acc:.4f} "
              f"({len(stab_valid_sqs)} valid squares)", flush=True)

        results[layer_name] = {
            'board_state_acc': float(state_acc),
            'frontier_acc': float(frontier_acc),
            'frontier_n_squares': len(frontier_accs),
            'mobility_r2': float(mobility_r2),
            'stability_acc': float(stab_acc),
            'stability_n_squares': len(stab_valid_sqs),
            'stability_valid_squares': stab_valid_sqs,
        }

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Exp-015 Phase 3: Championship vs Synthetic + Stability"
    )
    parser.add_argument("--n-games", type=int, default=5000)
    parser.add_argument("--max-positions", type=int, default=20000)
    parser.add_argument("--output-dir", type=str,
                        default="/nvmessd/lifanhong/LR-env/exp015-othello-phase3")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("Phase 3: Championship vs Synthetic OthelloGPT + Tile Stability", flush=True)
    print("=" * 70, flush=True)

    # ── 1. Generate games and labels ─────────────────────────────────────────
    print(f"\n[1/4] Generating {args.n_games} games...", flush=True)
    games = generate_games(args.n_games)
    print(f"  Generated {len(games)} valid games", flush=True)

    print(f"  Collecting labels (max {args.max_positions} positions)...", flush=True)
    data = collect_labels(games, args.max_positions)
    n_pos = len(data['state'])
    print(f"  Collected {n_pos} positions", flush=True)

    # Quick stability sanity check
    stab_sum = data['stability'].sum(axis=1)
    print(f"  Stability: mean {stab_sum.mean():.2f} stable discs/position, "
          f"max {stab_sum.max()}", flush=True)

    # ── 2. Probe SYNTHETIC model ─────────────────────────────────────────────
    print(f"\n[2/4] Loading & probing SYNTHETIC model...", flush=True)
    model_syn = load_othello_model("synthetic", args.device)
    print(f"  Model loaded. Extracting hidden states...", flush=True)
    hiddens_syn = extract_hidden_states(model_syn, data['move_seqs'],
                                        device=args.device)
    print(f"  Hidden states shape: {hiddens_syn[0].shape}", flush=True)
    results_syn = probe_model("synthetic", hiddens_syn, data)

    # Free memory
    del model_syn, hiddens_syn
    torch.cuda.empty_cache()

    # ── 3. Probe CHAMPIONSHIP model ──────────────────────────────────────────
    print(f"\n[3/4] Loading & probing CHAMPIONSHIP model...", flush=True)
    model_champ = load_othello_model("championship", args.device)
    print(f"  Model loaded. Extracting hidden states...", flush=True)
    hiddens_champ = extract_hidden_states(model_champ, data['move_seqs'],
                                          device=args.device)
    print(f"  Hidden states shape: {hiddens_champ[0].shape}", flush=True)
    results_champ = probe_model("championship", hiddens_champ, data)

    del model_champ, hiddens_champ
    torch.cuda.empty_cache()

    # ── 4. Compare & save ────────────────────────────────────────────────────
    print(f"\n[4/4] Comparing results...", flush=True)

    summary = {
        'config': {
            'n_games': len(games),
            'n_positions': n_pos,
            'probe_type': 'linear (LogisticRegression / Ridge)',
            'stability_algorithm': 'directional-anchor (4-axis, iterative)',
        },
        'synthetic': results_syn,
        'championship': results_champ,
    }

    out_path = out_dir / "phase3_results.json"
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # ── Print comparison table ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PHASE 3 RESULTS: SYNTHETIC vs CHAMPIONSHIP")
    print("=" * 80)

    print(f"\n{'Layer':<7} │ {'Board State':<23} │ {'Frontier':<23} │ "
          f"{'Mobility R²':<23} │ {'Stability':<23}")
    print(f"{'':7} │ {'Synth':>10} {'Champ':>10}   │ {'Synth':>10} {'Champ':>10}   │ "
          f"{'Synth':>10} {'Champ':>10}   │ {'Synth':>10} {'Champ':>10}")
    print("─" * 105)

    for layer in range(9):
        ln = f"L{layer}" if layer < 8 else "embed"
        s = results_syn[ln]
        c = results_champ[ln]
        print(f"{ln:<7} │ {s['board_state_acc']:>10.4f} {c['board_state_acc']:>10.4f}   │ "
              f"{s['frontier_acc']:>10.4f} {c['frontier_acc']:>10.4f}   │ "
              f"{s['mobility_r2']:>10.4f} {c['mobility_r2']:>10.4f}   │ "
              f"{s['stability_acc']:>10.4f} {c['stability_acc']:>10.4f}")

    # Best-layer summary
    print("\n── Best layer comparison ──")
    for metric_key, metric_name in [
        ('board_state_acc', 'Board State'),
        ('frontier_acc', 'Frontier'),
        ('mobility_r2', 'Mobility R²'),
        ('stability_acc', 'Stability'),
    ]:
        best_syn = max(results_syn.items(), key=lambda x: x[1][metric_key])
        best_champ = max(results_champ.items(), key=lambda x: x[1][metric_key])
        delta = best_champ[1][metric_key] - best_syn[1][metric_key]
        winner = "CHAMP" if delta > 0 else "SYNTH" if delta < 0 else "TIE"
        print(f"  {metric_name:<15}: Synth={best_syn[1][metric_key]:.4f} @ {best_syn[0]}, "
              f"Champ={best_champ[1][metric_key]:.4f} @ {best_champ[0]}  "
              f"(Δ={delta:+.4f}, {winner})")

    print(f"\nResults saved to {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
