#!/bin/bash
# Point-H-JEPA (train_ogbench_v1: L2 stride 2, SIGReg x16, frame_skip 5) on the sharded OGBench
# collection, one 2-GPU DDP run per task, same step-based protocol as the ablations (num_l1_frames=13).
#
# Usage: bash scripts/launch_ogbench_train.sh   (expects /tmp/ogb/datasets from launch_ogbench_collection.sh)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_ROOT="${RUN_ROOT:-/tmp/ogb/runs}"
GROUP="${GROUP:-ogb-l2-sig-x16}"
MAX_STEPS="${MAX_STEPS:-30000}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
BS="${BS:-48}"
COMMIT="$(git rev-parse --short HEAD)$(git diff --quiet || echo -dirty)"

launch() {
  local gpus=$1 port=$2 task=$3
  local name="ogb-${task}-l2" out="$RUN_ROOT/ogbench_${task}_l2"
  mkdir -p "$out"
  echo "Launching $name on GPUs $gpus -> $out"
  CUDA_VISIBLE_DEVICES=$gpus setsid nohup torchrun --nproc_per_node=2 --master_port="$port" \
    scripts/train_ddp.py --config-name "train_ogbench_${task}_local" +num_l1_frames=13 \
    train.max_steps="$MAX_STEPS" train.epochs=100 train.eval_every="$EVAL_EVERY" \
    train.batch_size_per_gpu="$BS" train.num_workers=12 train.diag_num_frames=8192 \
    train.output_dir="$out" hydra.run.dir="$out/hydra" \
    wandb.group="$GROUP" wandb.name="$name" "wandb.notes='commit $COMMIT; data /tmp/ogb/datasets'" \
    > "$out/train.log" 2>&1 < /dev/null &
  echo $! > "$out/train.pid"
}

launch 0,1 29511 cube_single
launch 2,3 29512 scene
echo "Logs: $RUN_ROOT/ogbench_<task>_l2/train.log"
