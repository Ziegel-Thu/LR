#!/bin/bash
# Exp-007 Full Run: parallel probe + ablation across 8 GPUs
# Usage: bash launch_full.sh

set -e
cd /beegfs_hdd/data/nfs_share/users/lifanhong/nishome/LR
source /nvmessd/lifanhong/LR-env/venv/bin/activate
export HF_HOME=/nvmessd/lifanhong/LR-env/cache/hf
export HF_HUB_DISABLE_XET=1

WORK_DIR=/nvmessd/lifanhong/LR-env/exp007_full
SCRIPT=experiments/exp-007-encoding-use-gap/scripts/run_full.py

# Step 1: Cache hidden states (single GPU, ~10min)
echo "=== Step 1: Caching hidden states ==="
CUDA_VISIBLE_DEVICES=0 python $SCRIPT cache --work-dir $WORK_DIR

# Step 2+3: Run features in parallel (up to 8 GPUs)
echo "=== Step 2+3: Running features in parallel ==="
FEATURES=(
  is_capitalized is_plural is_high_freq is_non_english
  is_numeric is_punctuation is_short is_stopword
  has_prefix is_rare is_title_case starts_with_space
  is_subword ends_with_ing ends_with_tion
)

GPU=0
PIDS=()
for feat in "${FEATURES[@]}"; do
  echo "  Starting $feat on GPU $GPU"
  CUDA_VISIBLE_DEVICES=$GPU python $SCRIPT feature \
    --feature $feat --work-dir $WORK_DIR \
    > /nvmessd/lifanhong/LR-env/exp007_${feat}.log 2>&1 &
  PIDS+=($!)
  GPU=$(( (GPU + 1) % 8 ))

  # Wait if all 8 GPUs busy
  if [ ${#PIDS[@]} -ge 8 ]; then
    wait ${PIDS[0]}
    PIDS=("${PIDS[@]:1}")
  fi
done

# Wait for all remaining
echo "  Waiting for all features..."
wait "${PIDS[@]}"
echo "  All features done!"

# Step 4: Analyze
echo "=== Step 4: Analyzing ==="
python $SCRIPT analyze --work-dir $WORK_DIR

echo "=== DONE ==="
