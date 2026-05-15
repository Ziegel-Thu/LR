#!/bin/bash
# Exp-001: SAE 可识别性 Pilot — 并行训练脚本
set -e
cd /Users/ziegel/LR
source .venv/bin/activate

SCRIPT="experiments/exp-001-sae-identifiability/scripts/run_pilot.py"
LOG_DIR="experiments/exp-001-sae-identifiability/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Step 1: Caching activations (2M tokens)"
echo "=========================================="
python "$SCRIPT" --step cache --n-tokens 2000000 2>&1 | tee "$LOG_DIR/cache.log"

echo ""
echo "=========================================="
echo "Step 2: Training 5 SAEs in parallel"
echo "=========================================="
PIDS=()
for seed in 0 1 2 3 4; do
    echo "Starting seed $seed..."
    python "$SCRIPT" --step train --n-seeds 1 --seed-offset $seed --device mps \
        2>&1 | tee "$LOG_DIR/train_seed${seed}.log" &
    PIDS+=($!)
done

echo "Waiting for all training jobs (PIDs: ${PIDS[*]})..."
for pid in "${PIDS[@]}"; do
    wait $pid
    echo "PID $pid finished with exit code $?"
done

echo ""
echo "=========================================="
echo "Step 3: Comparing features"
echo "=========================================="
python "$SCRIPT" --step compare --device mps 2>&1 | tee "$LOG_DIR/compare.log"

echo ""
echo "=========================================="
echo "DONE — results in experiments/exp-001-sae-identifiability/results.json"
echo "=========================================="
