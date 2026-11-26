#!/bin/bash
# Round-1 world-model ablation: hierarchy depth / stride at a fixed clip window (num_l1_frames=13,
# span 37 raw steps, so every episode is usable and all runs cost the same per step).
# One single-GPU run per config, all logged to one W&B group.
#
# Usage: bash scripts/launch_ablation_r1.sh   (expects the dataset copy at /tmp/point_hjepa/v1)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source .venv/bin/activate
# Batches pad frames to the largest point count, so memory varies per step; reduce fragmentation.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_ROOT="${RUN_ROOT:-/tmp/point_hjepa/runs}"
GROUP="${GROUP:-abl-r1-depth}"
MAX_STEPS="${MAX_STEPS:-26000}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
BS="${BS:-48}"
COMMIT="$(git rev-parse --short HEAD)$(git diff --quiet || echo -dirty)"

# train_v1_local reads a node-local copy of the dataset; /tmp does not survive moving to another node.
if [ ! -d /tmp/point_hjepa/v1/PullCubeTool-v1 ]; then
  echo "Copying outputs/datasets/v1 -> /tmp/point_hjepa/v1 (~72 GB)"
  mkdir -p /tmp/point_hjepa/v1
  rsync -a --exclude logs --exclude hydra --exclude '*.pid' --exclude README.md \
    outputs/datasets/v1/ /tmp/point_hjepa/v1/
fi

COMMON=(--config-name train_v1_local +num_l1_frames=13
  train.max_steps="$MAX_STEPS" train.epochs=100 train.eval_every="$EVAL_EVERY"
  train.batch_size_per_gpu="$BS" train.num_workers=12 train.diag_num_frames=8192
  wandb.group="$GROUP" "wandb.notes='commit $COMMIT; data /tmp/point_hjepa/v1'")

launch() {
  local gpu=$1 name=$2; shift 2
  local out="$RUN_ROOT/$name"
  mkdir -p "$out"
  echo "Launching $name on GPU $gpu -> $out"
  CUDA_VISIBLE_DEVICES=$gpu setsid nohup python scripts/train_ddp.py "${COMMON[@]}" \
    wandb.name="$name" train.output_dir="$out" hydra.run.dir="$out/hydra" "$@" \
    > "$out/train.log" 2>&1 < /dev/null &
  echo $! > "$out/train.pid"
}

launch 0 r1-L1 num_levels=1 strides=[] context_lengths=[3] \
  loss.lambda_sigreg=[0.08] loss.gamma_idm=[50] "wandb.tags=[L1,T13,fs3]"
launch 1 r1-L2-s2 "wandb.tags=[L2,s2,T13,fs3]"
launch 2 r1-L2-s4 strides=[4] "wandb.tags=[L2,s4,T13,fs3]"
launch 3 r1-L3-s22 num_levels=3 strides=[2,2] context_lengths=[3,3,3] \
  loss.lambda_sigreg=[0.08,0.16,0.16] loss.gamma_idm=[50,50,50] "wandb.tags=[L3,s2-2,T13,fs3]"
echo "All Round-1 runs launched. Logs: $RUN_ROOT/<run>/train.log"
