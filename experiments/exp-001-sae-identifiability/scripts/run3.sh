#!/bin/bash
# Exp-001 Run 3: Pythia-1.4B 快速验证 — 模型规模是否是共享率的关键
# 只跑 2 个种子，1M tokens，快速看 MMCS 是否质变
# 峰值内存 ~9GB (安全)
set -e
cd /Users/ziegel/LR
source .venv/bin/activate
unset HF_ENDPOINT

PY="experiments/exp-001-sae-identifiability/scripts/run_pilot.py"
LOG_DIR="experiments/exp-001-sae-identifiability/logs/run3"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Run 3: Pythia-1.4B, d_sae=4096, K=32, 1M tokens, 3 epochs, 2 seeds"
echo "=========================================="

# Step 1: Cache 1M tokens
echo "Step 1: Caching 1M tokens from Pythia-1.4B layer 12..."
python "$PY" --step cache \
    --model EleutherAI/pythia-1.4b-deduped \
    --layer 12 --n-tokens 1000000 \
    2>&1 | tee "$LOG_DIR/cache.log"

# Step 2: Train 2 SAEs sequentially
echo "Step 2: Training 2 SAEs..."
for seed in 0 1; do
    echo ""
    echo "--- Seed $seed ---"
    python "$PY" --step train --n-seeds 1 --seed-offset $seed \
        --model EleutherAI/pythia-1.4b-deduped \
        --layer 12 --n-tokens 1000000 \
        --d-sae 4096 --k 32 --n-epochs 3 --device mps \
        2>&1 | tee "$LOG_DIR/train_seed${seed}.log"
done

# Step 3: Compare
echo ""
echo "Step 3: Comparing..."
python "$PY" --step compare \
    --model EleutherAI/pythia-1.4b-deduped \
    --layer 12 --d-sae 4096 --k 32 --device mps \
    2>&1 | tee "$LOG_DIR/compare.log"

echo ""
echo "=========================================="
echo "Run 3 DONE"
echo "=========================================="
