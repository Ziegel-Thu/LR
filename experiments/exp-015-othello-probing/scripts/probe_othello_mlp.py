#!/usr/bin/env python3
"""
Exp-015 Phase 2: MLP Probes + Ablation for OthelloGPT

Phase 1 used linear probes and found:
  - Board state: 72% (linear) vs Li et al.'s 98% (MLP)
  - Frontier discs: 96.7% @ L4
  - Mobility: R²=0.73 @ L6

This script adds:
  1. MLP probes (2-layer) to replicate Li et al.'s ~98% board state
  2. Additional strategic features: corner occupancy, edge control, disc advantage
  3. Ablation (encoding ≠ use): project out probe direction, measure Δ next-move acc
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

# ── Othello game logic (from Phase 1) ────────────────────────────────────────

BOARD_SIZE = 8
DIRECTIONS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
CENTER_SQUARES = {27, 28, 35, 36}

def board_pos_to_token(pos):
    assert pos not in CENTER_SQUARES, f"Center square {pos} is never a valid move"
    return pos - sum(1 for c in CENTER_SQUARES if c < pos)

def token_to_board_pos(tok):
    pos = tok
    for c in sorted(CENTER_SQUARES):
        if pos >= c:
            pos += 1
    return pos

class OthelloBoard:
    def __init__(self):
        self.board = np.zeros((8, 8), dtype=int)
        self.board[3,3] = self.board[4,4] = -1
        self.board[3,4] = self.board[4,3] = 1
        self.turn = 1

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
        if not self._valid_moves_for(self.turn):
            self.turn = -self.turn
        return True

    def get_state(self):
        return (self.board + 1).flatten().tolist()

    def get_frontier(self):
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

    def get_mobility(self):
        return len(self._valid_moves_for(self.turn))

    def get_corner_occupancy(self):
        """4-dim binary: is each corner occupied by current player?"""
        corners = [(0,0), (0,7), (7,0), (7,7)]
        return [1 if self.board[r,c] == self.turn else 0 for r,c in corners]

    def get_edge_control(self):
        """Count of edge squares controlled by current player."""
        edges = []
        for i in range(8):
            edges.extend([(0,i), (7,i), (i,0), (i,7)])
        edges = list(set(edges))
        return sum(1 for r,c in edges if self.board[r,c] == self.turn)

    def get_disc_advantage(self):
        """current_player_discs - opponent_discs."""
        mine = int((self.board == self.turn).sum())
        theirs = int((self.board == -self.turn).sum())
        return mine - theirs


# ── Data generation ──────────────────────────────────────────────────────────

def generate_games(n_games=10000, seed=42):
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
    """Collect board states and all strategic labels from games."""
    labels = defaultdict(list)
    move_seqs = []
    next_moves = []  # for ablation: the next move token

    for game in games:
        board = OthelloBoard()
        for t, move in enumerate(game):
            board.play(move)
            if t < 4:
                continue
            labels['state'].append(board.get_state())
            labels['frontier'].append(board.get_frontier())
            labels['mobility'].append(board.get_mobility())
            labels['corner'].append(board.get_corner_occupancy())
            labels['edge_control'].append(board.get_edge_control())
            labels['disc_advantage'].append(board.get_disc_advantage())
            move_seqs.append(game[:t+1])
            # Next move for ablation (if available)
            if t + 1 < len(game):
                next_moves.append(board_pos_to_token(game[t+1]))
            else:
                next_moves.append(-1)  # no next move

            if len(labels['state']) >= max_positions:
                break
        if len(labels['state']) >= max_positions:
            break

    return {
        'state': np.array(labels['state']),
        'frontier': np.array(labels['frontier']),
        'mobility': np.array(labels['mobility']),
        'corner': np.array(labels['corner']),
        'edge_control': np.array(labels['edge_control']),
        'disc_advantage': np.array(labels['disc_advantage']),
        'move_seqs': move_seqs,
        'next_moves': np.array(next_moves),
    }


# ── MLP Probe ────────────────────────────────────────────────────────────────

class MLPProbe(nn.Module):
    """2-layer MLP probe: Linear → ReLU → Linear."""
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp_probe(X_tr, y_tr, X_te, y_te, d_in, n_classes,
                    d_hidden=256, lr=1e-3, epochs=20, batch_size=256,
                    device='cuda:0', task='classification'):
    """Train a 2-layer MLP probe. Returns (metric, trained_probe)."""
    probe = MLPProbe(d_in, d_hidden, n_classes if task == 'classification' else 1).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    X_te_t = torch.tensor(X_te, dtype=torch.float32)

    if task == 'classification':
        y_tr_t = torch.tensor(y_tr, dtype=torch.long)
        y_te_t = torch.tensor(y_te, dtype=torch.long)
        loss_fn = nn.CrossEntropyLoss()
    else:
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
        y_te_t = torch.tensor(y_te, dtype=torch.float32)
        loss_fn = nn.MSELoss()

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    probe.train()
    for epoch in range(epochs):
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = probe(xb)
            if task == 'regression':
                pred = pred.squeeze(-1)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    probe.eval()
    with torch.no_grad():
        pred = probe(X_te_t.to(device))
        if task == 'classification':
            acc = (pred.argmax(-1).cpu() == y_te_t).float().mean().item()
            return acc, probe
        else:
            pred = pred.squeeze(-1).cpu()
            ss_res = ((y_te_t - pred) ** 2).sum().item()
            ss_tot = ((y_te_t - y_te_t.mean()) ** 2).sum().item()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            return r2, probe


# ── Board state probing (64 parallel MLP probes) ────────────────────────────

def train_board_state_mlp(hiddens, labels, device='cuda:0',
                          d_hidden=256, lr=1e-3, epochs=20, batch_size=256):
    """Train 64 MLP probes for board state (one per square, 3 classes each)."""
    n_samples, d = hiddens.shape
    split = int(0.8 * n_samples)
    X_tr, X_te = hiddens[:split], hiddens[split:]
    y_tr, y_te = labels[:split], labels[split:]

    accs = []
    for sq in tqdm(range(64), desc="    Board squares", leave=False):
        acc, _ = train_mlp_probe(
            X_tr, y_tr[:, sq], X_te, y_te[:, sq],
            d_in=d, n_classes=3, d_hidden=d_hidden, lr=lr,
            epochs=epochs, batch_size=batch_size, device=device,
        )
        accs.append(acc)

    return np.mean(accs), accs


# ── Ablation: Encoding ≠ Use ─────────────────────────────────────────────────

def get_probe_direction(probe, device='cuda:0'):
    """Extract the effective probe direction from a trained MLP probe.
    
    For ablation, we use the input-layer weights to define the "probe subspace".
    W1 is [d_hidden, d_model] — each row is a direction the probe reads from.
    We project out this entire subspace from the hidden state.
    """
    W1 = probe.net[0].weight.detach()  # [d_hidden, d_model]
    # SVD to get the principal directions of the probe subspace
    U, S, Vh = torch.linalg.svd(W1, full_matrices=False)  # Vh: [min(d_h,d_m), d_model]
    return Vh.to(device)  # [k, d_model] — orthonormal basis of probe subspace


def ablation_hook_factory(probe_directions):
    """Create a hook that projects out the probe subspace from residual stream.
    
    probe_directions: [k, d_model] orthonormal basis vectors.
    """
    def hook_fn(value, hook):
        # value: [batch, seq, d_model]
        V = probe_directions  # [k, d_model]
        # Project out: x' = x - V^T V x
        proj = torch.einsum('ki,bsi->bsk', V, value)   # [batch, seq, k]
        removed = torch.einsum('bsk,ki->bsi', proj, V) # [batch, seq, d_model]
        return value - removed
    return hook_fn


def measure_next_move_accuracy(model, move_seqs, next_moves, device='cuda:0',
                               batch_size=64, hook_fn=None, hook_point=None):
    """Measure next-move prediction accuracy, optionally with an ablation hook."""
    # Filter to positions with valid next moves
    valid_mask = next_moves >= 0
    seqs = [s for s, v in zip(move_seqs, valid_mask) if v]
    targets = next_moves[valid_mask]
    if len(seqs) == 0:
        return 0.0

    max_len = min(max(len(s) for s in seqs), 59)
    correct = 0
    total = 0

    for i in range(0, len(seqs), batch_size):
        batch_seqs = seqs[i:i+batch_size]
        batch_targets = targets[i:i+batch_size]
        batch_lens = [min(len(s), max_len) for s in batch_seqs]

        padded = torch.zeros(len(batch_seqs), max_len, dtype=torch.long)
        for j, seq in enumerate(batch_seqs):
            tokens = [board_pos_to_token(m) for m in seq[:max_len]]
            padded[j, :len(tokens)] = torch.tensor(tokens)
        padded = padded.to(device)

        with torch.no_grad():
            if hook_fn is not None and hook_point is not None:
                logits = model.run_with_hooks(
                    padded,
                    fwd_hooks=[(hook_point, hook_fn)],
                )
            else:
                logits = model(padded)

            # Get prediction at last real position
            for j in range(len(batch_seqs)):
                pred_tok = logits[j, batch_lens[j]-1].argmax().item()
                if pred_tok == batch_targets[j]:
                    correct += 1
                total += 1

    return correct / total if total > 0 else 0.0


def run_ablation(model, probe, layer, concept_name,
                 move_seqs, next_moves, device='cuda:0'):
    """Run encoding≠use ablation for a concept at a specific layer."""
    print(f"  Ablation for {concept_name} @ L{layer}...", flush=True)

    # Baseline accuracy (no ablation)
    baseline_acc = measure_next_move_accuracy(
        model, move_seqs, next_moves, device=device
    )
    print(f"    Baseline next-move acc: {baseline_acc:.4f}", flush=True)

    # Get probe directions and create ablation hook
    probe_dirs = get_probe_direction(probe, device)
    hook_fn = ablation_hook_factory(probe_dirs)
    hook_point = f"blocks.{layer}.hook_resid_post"

    # Ablated accuracy
    ablated_acc = measure_next_move_accuracy(
        model, move_seqs, next_moves, device=device,
        hook_fn=hook_fn, hook_point=hook_point
    )
    print(f"    Ablated next-move acc:  {ablated_acc:.4f}", flush=True)
    delta = ablated_acc - baseline_acc
    print(f"    Δaccuracy: {delta:+.4f}", flush=True)

    return {
        'baseline_acc': baseline_acc,
        'ablated_acc': ablated_acc,
        'delta_acc': delta,
        'n_probe_dirs': int(probe_dirs.shape[0]),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OthelloGPT Phase 2: MLP probes + ablation")
    parser.add_argument("--n-games", type=int, default=5000)
    parser.add_argument("--max-positions", type=int, default=20000)
    parser.add_argument("--output-dir", type=str,
                        default="/nvmessd/lifanhong/LR-env/exp015-othello-phase2")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--d-hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load OthelloGPT ───────────────────────────────────────────────────
    print("="*60, flush=True)
    print("Phase 2: MLP Probes + Ablation for OthelloGPT", flush=True)
    print("="*60, flush=True)

    print("\n[1/5] Loading OthelloGPT...", flush=True)
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    from transformer_lens.utils import download_file_from_hf

    cfg = HookedTransformerConfig(
        n_layers=8, d_model=512, d_head=64, n_heads=8, d_mlp=2048,
        d_vocab=61, n_ctx=59, act_fn="gelu", normalization_type="LNPre",
    )
    model = HookedTransformer(cfg)
    sd = download_file_from_hf("NeelNanda/Othello-GPT-Transformer-Lens", "synthetic_model.pth")
    model.load_state_dict(sd)
    model = model.to(args.device).eval()
    print(f"  Loaded synthetic model: 8 layers, d=512", flush=True)

    # ── 2. Generate data ─────────────────────────────────────────────────────
    print(f"\n[2/5] Generating {args.n_games} games...", flush=True)
    games = generate_games(args.n_games)
    print(f"  Generated {len(games)} valid games", flush=True)

    print(f"  Collecting labels (max {args.max_positions} positions)...", flush=True)
    data = collect_labels(games, args.max_positions)
    n_pos = len(data['state'])
    n_valid_next = int((data['next_moves'] >= 0).sum())
    print(f"  Collected {n_pos} positions ({n_valid_next} with valid next moves)", flush=True)

    # ── 3. Extract hidden states ─────────────────────────────────────────────
    print(f"\n[3/5] Extracting hidden states...", flush=True)
    max_len = min(max(len(s) for s in data['move_seqs']), 59)
    all_hiddens = {l: [] for l in range(9)}

    extract_batch = 64
    for i in tqdm(range(0, n_pos, extract_batch), desc="  Extracting"):
        batch_seqs = data['move_seqs'][i:i+extract_batch]
        batch_lens = [min(len(s), max_len) for s in batch_seqs]

        padded = torch.zeros(len(batch_seqs), max_len, dtype=torch.long)
        for j, seq in enumerate(batch_seqs):
            tokens = [board_pos_to_token(m) for m in seq[:max_len]]
            padded[j, :len(tokens)] = torch.tensor(tokens)
        padded = padded.to(args.device)

        with torch.no_grad():
            _, cache = model.run_with_cache(padded)

        for l in range(8):
            acts = cache[f"blocks.{l}.hook_resid_post"]
            last_acts = torch.stack([acts[j, batch_lens[j]-1] for j in range(len(batch_seqs))])
            all_hiddens[l].append(last_acts.cpu().numpy())

        acts_emb = cache["hook_embed"] + cache["hook_pos_embed"]
        last_emb = torch.stack([acts_emb[j, batch_lens[j]-1] for j in range(len(batch_seqs))])
        all_hiddens[8].append(last_emb.cpu().numpy())

    for l in all_hiddens:
        all_hiddens[l] = np.concatenate(all_hiddens[l], axis=0)
    print(f"  Hidden states shape: {all_hiddens[0].shape}", flush=True)

    d_model = all_hiddens[0].shape[1]
    split = int(0.8 * n_pos)

    # ── 4. MLP Probing ───────────────────────────────────────────────────────
    print(f"\n[4/5] MLP Probing (d_hidden={args.d_hidden}, lr={args.lr}, "
          f"epochs={args.epochs})...", flush=True)

    results = {}
    best_probes = {}  # store best probes for ablation

    for layer in range(9):
        layer_name = f"L{layer}" if layer < 8 else "embed"
        print(f"\n  ── {layer_name} ──", flush=True)
        H = all_hiddens[layer]
        X_tr, X_te = H[:split], H[split:]

        # Board state (64 squares × 3 classes)
        board_acc, per_sq = train_board_state_mlp(
            H, data['state'], device=args.device,
            d_hidden=args.d_hidden, lr=args.lr,
            epochs=args.epochs, batch_size=args.batch_size,
        )
        print(f"  Board state: {board_acc:.4f}", flush=True)

        # Frontier discs (64 binary probes, skip low-variance squares)
        frontier_accs = []
        best_frontier_probe = None
        best_frontier_acc = 0
        for sq in range(64):
            col = data['frontier'][:, sq]
            if col.mean() < 0.02 or col.mean() > 0.98:
                continue
            acc, probe = train_mlp_probe(
                X_tr, col[:split], X_te, col[split:],
                d_in=d_model, n_classes=2,
                d_hidden=args.d_hidden, lr=args.lr,
                epochs=args.epochs, batch_size=args.batch_size,
                device=args.device,
            )
            frontier_accs.append(acc)
            if acc > best_frontier_acc:
                best_frontier_acc = acc
                best_frontier_probe = probe
        frontier_acc = np.mean(frontier_accs) if frontier_accs else 0
        print(f"  Frontier discs: {frontier_acc:.4f} ({len(frontier_accs)} squares)", flush=True)

        # Mobility (regression)
        mob_r2, mob_probe = train_mlp_probe(
            X_tr, data['mobility'][:split], X_te, data['mobility'][split:],
            d_in=d_model, n_classes=1,
            d_hidden=args.d_hidden, lr=args.lr,
            epochs=args.epochs, batch_size=args.batch_size,
            device=args.device, task='regression',
        )
        print(f"  Mobility: R²={mob_r2:.4f}", flush=True)

        # Corner occupancy (4 binary probes)
        corner_accs = []
        for ci in range(4):
            col = data['corner'][:, ci]
            if col.mean() < 0.02 or col.mean() > 0.98:
                continue
            acc, _ = train_mlp_probe(
                X_tr, col[:split], X_te, col[split:],
                d_in=d_model, n_classes=2,
                d_hidden=args.d_hidden, lr=args.lr,
                epochs=args.epochs, batch_size=args.batch_size,
                device=args.device,
            )
            corner_accs.append(acc)
        corner_acc = np.mean(corner_accs) if corner_accs else 0
        print(f"  Corner occupancy: {corner_acc:.4f} ({len(corner_accs)} corners)", flush=True)

        # Edge control (regression)
        edge_r2, _ = train_mlp_probe(
            X_tr, data['edge_control'][:split], X_te, data['edge_control'][split:],
            d_in=d_model, n_classes=1,
            d_hidden=args.d_hidden, lr=args.lr,
            epochs=args.epochs, batch_size=args.batch_size,
            device=args.device, task='regression',
        )
        print(f"  Edge control: R²={edge_r2:.4f}", flush=True)

        # Disc advantage (regression)
        disc_r2, _ = train_mlp_probe(
            X_tr, data['disc_advantage'][:split], X_te, data['disc_advantage'][split:],
            d_in=d_model, n_classes=1,
            d_hidden=args.d_hidden, lr=args.lr,
            epochs=args.epochs, batch_size=args.batch_size,
            device=args.device, task='regression',
        )
        print(f"  Disc advantage: R²={disc_r2:.4f}", flush=True)

        results[layer_name] = {
            'board_state_acc': float(board_acc),
            'board_state_per_square': [float(a) for a in per_sq],
            'frontier_acc': float(frontier_acc),
            'frontier_n_squares': len(frontier_accs),
            'mobility_r2': float(mob_r2),
            'corner_acc': float(corner_acc),
            'corner_n_valid': len(corner_accs),
            'edge_control_r2': float(edge_r2),
            'disc_advantage_r2': float(disc_r2),
        }

        # Track best probes for ablation (skip embed layer)
        if layer < 8:
            best_probes.setdefault('frontier', {'acc': 0})
            if frontier_acc > best_probes['frontier']['acc']:
                best_probes['frontier'] = {
                    'acc': frontier_acc, 'layer': layer, 'probe': best_frontier_probe,
                }
            best_probes.setdefault('mobility', {'r2': -999})
            if mob_r2 > best_probes['mobility']['r2']:
                best_probes['mobility'] = {
                    'r2': mob_r2, 'layer': layer, 'probe': mob_probe,
                }

    # ── 5. Ablation ──────────────────────────────────────────────────────────
    print(f"\n[5/5] Ablation (encoding ≠ use)...", flush=True)
    ablation_results = {}

    for concept in ['frontier', 'mobility']:
        info = best_probes[concept]
        layer = info['layer']
        probe = info['probe']
        metric = info.get('acc', info.get('r2'))
        print(f"\n  {concept}: best layer = L{layer} (metric={metric:.4f})", flush=True)

        if probe is None:
            print(f"    Skipping (no valid probe trained)", flush=True)
            continue

        abl = run_ablation(
            model, probe, layer, concept,
            data['move_seqs'], data['next_moves'],
            device=args.device,
        )
        ablation_results[concept] = {
            'layer': layer,
            'probe_metric': float(metric),
            **abl,
        }

    # ── Save results ─────────────────────────────────────────────────────────
    summary = {
        'config': {
            'n_games': len(games),
            'n_positions': n_pos,
            'probe_type': 'MLP (2-layer)',
            'd_hidden': args.d_hidden,
            'lr': args.lr,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
        },
        'probing_results': results,
        'ablation_results': ablation_results,
    }

    out_path = out_dir / "phase2_results.json"
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # ── Print summary ────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("PHASE 2 RESULTS SUMMARY")
    print("="*80)

    print(f"\n{'Layer':<7} {'Board St.':<10} {'Frontier':<10} {'Mob. R²':<9} "
          f"{'Corner':<9} {'Edge R²':<9} {'Disc R²':<9}")
    print("-"*63)
    for layer in range(9):
        ln = f"L{layer}" if layer < 8 else "embed"
        r = results[ln]
        print(f"{ln:<7} {r['board_state_acc']:.4f}     {r['frontier_acc']:.4f}     "
              f"{r['mobility_r2']:.4f}    {r['corner_acc']:.4f}    "
              f"{r['edge_control_r2']:.4f}    {r['disc_advantage_r2']:.4f}")

    if ablation_results:
        print(f"\nAblation (encoding ≠ use):")
        print(f"{'Concept':<12} {'Layer':<7} {'Baseline':<10} {'Ablated':<10} {'Δacc':<10}")
        print("-"*49)
        for concept, abl in ablation_results.items():
            print(f"{concept:<12} L{abl['layer']:<5} {abl['baseline_acc']:.4f}     "
                  f"{abl['ablated_acc']:.4f}     {abl['delta_acc']:+.4f}")

    print(f"\nResults saved to {out_path}")
    print("="*80)


if __name__ == "__main__":
    main()
