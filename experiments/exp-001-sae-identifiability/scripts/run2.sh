#!/bin/bash
# Exp-001 Run 2: 更小字典 + 更多数据（串行训练，内存安全）
# 峰值内存 ~16GB (5M tokens 激活 15.4GB + SAE 9MB + batch)
set -e
cd /Users/ziegel/LR
source .venv/bin/activate
unset HF_ENDPOINT

PY="experiments/exp-001-sae-identifiability/scripts/run_pilot.py"
LOG_DIR="experiments/exp-001-sae-identifiability/logs/run2"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Run 2: d_sae=1536, K=32, 5M tokens, 5 epochs"
echo "=========================================="

# Step 1: Cache 5M tokens
echo "Step 1: Caching 5M tokens..."
python "$PY" --step cache --n-tokens 5000000 2>&1 | tee "$LOG_DIR/cache.log"

# Step 2: Train 5 SAEs SEQUENTIALLY (memory safe)
echo "Step 2: Training 5 SAEs sequentially..."
for seed in 0 1 2 3 4; do
    echo ""
    echo "--- Seed $seed ---"
    python "$PY" --step train --n-seeds 1 --seed-offset $seed \
        --d-sae 1536 --k 32 --n-epochs 5 --n-tokens 5000000 --device mps \
        2>&1 | tee "$LOG_DIR/train_seed${seed}.log"
done

# Step 3: Compare
echo ""
echo "Step 3: Comparing features..."
python "$PY" --step compare --d-sae 1536 --k 32 --device mps 2>&1 | tee "$LOG_DIR/compare.log"

echo ""
echo "=========================================="
echo "Run 2 DONE"
echo "=========================================="
