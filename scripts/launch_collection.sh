#!/bin/bash
# Launch sharded ManiSkill3 6D data collection: one process per (task, shard),
# NUM_GPUS shards per task, two processes per GPU. Each shard collects its own
# expert_srmp episodes (distinct seeds) and branches exploratory rollouts from
# its own snapshots, writing independent Zarr stores under OUT_ROOT.
#
# Usage: OUT_ROOT=outputs/datasets/v1 bash scripts/launch_collection.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

OUT_ROOT="${OUT_ROOT:-$REPO/outputs/datasets/v1}"
NUM_GPUS="${NUM_GPUS:-4}"
TASKS="${TASKS:-StackCube-v1 PullCubeTool-v1}"
EXPERT_EPISODES="${EXPERT_EPISODES:-600}"
BRANCHED_EPISODES="${BRANCHED_EPISODES:-2176}"
NUM_ENVS="${NUM_ENVS:-128}"
MIN_FREE_GB="${MIN_FREE_GB:-15}"
VK_ENV="${VK_ENV:-/project/community/sbandred/opt/nvidia-vk-580.159.03/env.sh}"

source .venv/bin/activate
# Vulkan/EGL userspace libs for headless SAPIEN rendering (node driver ships compute-only).
[ -f "$VK_ENV" ] && source "$VK_ENV"

mkdir -p "$OUT_ROOT/logs"
task_idx=0
for task in $TASKS; do
  for shard in $(seq 0 $((NUM_GPUS - 1))); do
    gpu=$shard
    global_shard=$((task_idx * NUM_GPUS + shard))
    seed=$((100000 * (global_shard + 1)))
    out_dir="$OUT_ROOT/$task/shard_$shard"
    log="$OUT_ROOT/logs/${task}_shard_${shard}.log"
    mkdir -p "$out_dir"
    echo "Launching $task shard $shard on GPU $gpu (seed=$seed) -> $out_dir"
    CUDA_VISIBLE_DEVICES=$gpu setsid nohup python -m point_hjepa.data.collect_maniskill3_6d \
      task_id="$task" mode=both \
      expert_episodes="$EXPERT_EPISODES" branched_episodes="$BRANCHED_EPISODES" \
      num_envs="$NUM_ENVS" seed="$seed" \
      xyz_dtype=float16 sort_by_voxel=true min_free_gb="$MIN_FREE_GB" \
      export_ply_count=0 \
      output_dir="$out_dir" hydra.run.dir="$out_dir/hydra" \
      > "$log" 2>&1 < /dev/null &
    echo $! > "$OUT_ROOT/logs/${task}_shard_${shard}.pid"
  done
  task_idx=$((task_idx + 1))
done
echo "All shards launched. Logs: $OUT_ROOT/logs"
