#!/bin/bash
# SIGReg-weight ablation on the L2 stride-2 world model, same protocol as Round 1 (num_l1_frames=13,
# span 37 raw steps, so every episode is usable and all runs cost the same per step).
# One single-GPU run per config, all logged to one W&B group.
#
# Usage: bash scripts/launch_ablation_sigreg.sh   (copies the dataset to /tmp/point_hjepa/v1 if missing)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source .venv/bin/activate
# Batches pad frames to the largest point count, so memory varies per step; reduce fragmentation.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_ROOT="${RUN_ROOT:-/tmp/point_hjepa/runs}"
GROUP="${GROUP:-abl-r2-sigreg}"
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

# Baseline lambda_sigreg=[0.08, 0.16] is r1-L2-s2 (x1). At x1 the SIGReg term is ~400x smaller than the
# gamma_idm=50 IDM term, so sweep on a log scale up to where the two are comparable.
launch 0 r2-sig-x4 "loss.lambda_sigreg=[0.32,0.64]" "wandb.tags=[L2,s2,T13,fs3,sig-x4]"
launch 1 r2-sig-x16 "loss.lambda_sigreg=[1.28,2.56]" "wandb.tags=[L2,s2,T13,fs3,sig-x16]"
launch 2 r2-sig-x64 "loss.lambda_sigreg=[5.12,10.24]" "wandb.tags=[L2,s2,T13,fs3,sig-x64]"
launch 3 r2-sig-x256 "loss.lambda_sigreg=[20.48,40.96]" "wandb.tags=[L2,s2,T13,fs3,sig-x256]"
echo "All SIGReg runs launched. Logs: $RUN_ROOT/<run>/train.log"
