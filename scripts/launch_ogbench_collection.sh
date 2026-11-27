#!/bin/bash
# Sharded OGBench wm_collect: NUM_SHARDS single-threaded processes per task (MuJoCo + EGL rendering is
# CPU-bound and runs ~2x faster per process with one thread), each with its own seed and Zarr store,
# spread round-robin over the GPUs for EGL. JEPA4R's scale: 10,000 episodes per task.
#
# Usage: OUT_ROOT=/tmp/ogb/datasets bash scripts/launch_ogbench_collection.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

OUT_ROOT="${OUT_ROOT:-/tmp/ogb/datasets}"
TASKS="${TASKS:-cube_single scene}"
EPISODES="${EPISODES:-10000}"
NUM_SHARDS="${NUM_SHARDS:-40}"
NUM_GPUS="${NUM_GPUS:-4}"
VK_ENV="${VK_ENV:-/project/community/sbandred/opt/nvidia-vk-580.159.03/env.sh}"

source .venv/bin/activate
[ -f "$VK_ENV" ] && source "$VK_ENV"
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# Unsegmented clouds, cropped to each task's workspace box (drops only far floor / out-of-workspace
# geometry; table, arm and objects stay, no segmentation).
declare -A CROP=(
  [cube_single]="[[0.10,-0.50,-0.02],[0.75,0.50,0.60]]"
  [scene]="[[0.10,-0.55,-0.02],[0.75,0.50,0.60]]"
)

per_shard=$(( (EPISODES + NUM_SHARDS - 1) / NUM_SHARDS ))
mkdir -p "$OUT_ROOT/logs"
task_idx=0
for task in $TASKS; do
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    gpu=$(( shard % NUM_GPUS ))
    seed=$(( 1000000 * (task_idx + 1) + shard ))
    tag=$(printf "%s_shard_%02d" "$task" "$shard")
    out_dir="$OUT_ROOT/$task/$(printf "shard_%02d" "$shard")"
    mkdir -p "$out_dir"
    CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu setsid nohup python -m point_hjepa.data.collect_ogbench_6d \
      task="$task" mode=wm_collect unsegmented=true num_points=4096 \
      num_episodes="$per_shard" seed="$seed" "crop_box=${CROP[$task]}" \
      output_dir="$out_dir" hydra.run.dir="$out_dir/hydra" \
      > "$OUT_ROOT/logs/$tag.log" 2>&1 < /dev/null &
    echo $! > "$OUT_ROOT/logs/$tag.pid"
  done
  echo "Launched $NUM_SHARDS shards x $per_shard episodes for $task"
  task_idx=$((task_idx + 1))
done
echo "Logs: $OUT_ROOT/logs"
