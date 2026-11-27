# Point-H-JEPA

ManiSkill3 GPU-parallel (`physx_cuda`) 6D RGB-XYZ point-cloud data collection and PyTorch DDP implementation of a bottom-up, unsegmented **Point-H-JEPA** world model with `utonia_tiny`, an Attentive Pooling Bridge, Multi-Scale AdaLN-Zero Predictors, Multi-Scale Inverse Dynamics Models (IDM) with stop-gradient isolation, DDP-aware Epps-Pulley `SIGReg`, and a Stage-2 Hindsight 3D Goal Mapper (`pd_joint_delta_pos`, `action_dim = 8`).

## 1. Data Collection & Visual Verification

- **Unsegmented 6D Point Clouds (`[x, y, z, r, g, b]`):** Robot-base frame coordinates (`float32` meters) and `[0, 255]` RGB (`uint8`), preserving tabletop, robot arm, targets, and distractors without segmentation mask trimming.
- **Two Dataset Modes (`Zarr v3`, `BloscCodec(zstd)`):**
  - `expert_srmp`: Collision-free SRMP (VAMP) demonstrations on `StackCube-v1` and `PullCubeTool-v1`, saving `get_state_dict()` snapshots at key trajectory stages (`approach`, `grasp`, `lift`, `transport`, `place_or_pull`).
  - `branched_exploratory`: GPU-parallel `physx_cuda` rollouts restoring SRMP snapshots via `set_state_dict` (with immediate PD drive-target sync) across a mixture of noisy expert (50%), interrupted/held-OU/release-drop (30%), and full-episode OU random exploration (20%).

```bash
# Collect both expert_srmp and branched_exploratory datasets
python -m point_hjepa.data.collect_maniskill3_6d \
  task_id=StackCube-v1 mode=both expert_episodes=20 branched_episodes=100 num_envs=64

python -m point_hjepa.data.collect_maniskill3_6d \
  task_id=PullCubeTool-v1 mode=both expert_episodes=100 branched_episodes=5000 num_envs=128

# Live ManiSkill3 SAPIEN GUI playback across all 6 trajectory modes
python scripts/visualize_rollouts.py --human --loop --fps 20 --max-steps 160

# Collect 6D mode Zarrs and step through 4096-pt clouds interactively in Viser
python scripts/visualize_rollouts.py --save-zarr-dir outputs/pc_inspect --collect-zarr-only --max-steps 160
python scripts/inspect_pointclouds_viser.py --zarrs outputs/pc_inspect/*_modes.zarr --port 8080
```

### Large-scale sharded collection (4 GPUs)

```bash
uv sync --extra dev
# Headless nodes with a compute-only NVIDIA driver need the matching Vulkan/EGL userspace
# libs (extracted from the driver .run with --extract-only); the launcher sources this env file.
source /project/community/sbandred/opt/nvidia-vk-580.159.03/env.sh

# 2 tasks x 4 shards (2 processes per GPU), float16 xyz + voxel-sorted points (~23 KB/frame),
# stops early if free disk drops below MIN_FREE_GB.
OUT_ROOT=outputs/datasets/v1 bash scripts/launch_collection.sh
python scripts/collection_status.py outputs/datasets/v1

# Upload (resumable) after `hf auth login`
python scripts/upload_to_hf.py --repo-id <user>/point-hjepa-maniskill3-6d --folder outputs/datasets/v1
```

## 2. Training (4× H100 DDP & Local Smoke Test)

```bash
# Stage 1: Goal-free Point-H-JEPA world model training (global batch size 256)
torchrun --nproc_per_node=4 scripts/train_ddp.py

# Stage 2: Hindsight 3D point goal mapper training (frozen world model)
python scripts/train_goal_mapper.py \
  goal_mapper.checkpoint=outputs/runs/sanity_l2/checkpoint_latest.pt
```

World-model ablations (one single-GPU run per config, node-local data copy, W&B group per round;
`train.eval_every` runs probes + checkpoint every N steps instead of only at epoch end):

```bash
bash scripts/launch_ablation_r1.sh   # depth/stride sweep: L1, L2 stride 2/4, L3 strides 2,2
```

Local single-GPU smoke test (RTX 5080):

```bash
python scripts/train_ddp.py \
  train.batch_size_per_gpu=4 train.epochs=1 train.max_steps=5 \
  train.num_workers=0 train.diag_num_frames=64 wandb.enabled=false

python scripts/train_goal_mapper.py \
  goal_mapper.checkpoint=outputs/runs/sanity_l2/checkpoint_latest.pt \
  goal_mapper.batch_size=16 goal_mapper.epochs=1 goal_mapper.max_steps=5 \
  train.num_workers=0 wandb.enabled=false
```

## 3. Per-level point-cloud decoders (JEPA4R / PAGE-4D)

A post-hoc decoder per hierarchy level maps any latent `z^(l)` (observed, one-step predicted, or imagined by an open-loop rollout) back to a metric 6D point cloud. Each level's decoder is an exact port of the JEPA4R decoder (arXiv 2611.19771):
- a FiLM set generator turns `z` into 16×16 tokens of width 1024;
- the frozen PAGE-4D trunk (DINOv2 ViT-L + 24 frame/global layers + DPT point head) produces a 224×224 point map;
- a learnable affine map (initialized as a virtual camera) and an rgb branch move it into the target frame.

Training is LP-FT:
- **Stage A** trains only the heads.
- **Stage B** adds rank-4 q/v LoRA on all 72 attentions.

Each level gets its own set generator, affine map, rgb branch and LoRA bank. The 0.94B-param trunk is shared and frozen, which works the same as separate decoders. A level-`l` state is supervised with the observed cloud of the Level-1 frame it summarizes.

```bash
# PAGE-4D code (third_party/PAGE4D @ c4a39ac) + checkpoint_nomask.pt (6.8 GB, checkpoints/page4d/)
python scripts/setup_page4d.py

# Stage A, then stage B, on a frozen Stage-1 checkpoint (configs/decoder/page4d.yaml)
torchrun --nproc_per_node=4 scripts/train_pc_decoder.py --config-name pc_decoder \
  decoder.wm_checkpoint=outputs/runs/sanity_l2/checkpoint_latest.pt
torchrun --nproc_per_node=4 scripts/train_pc_decoder.py --config-name pc_decoder decoder.stage=B \
  decoder.init_from=outputs/runs/pc_decoder/stageA/decoder_latest.pt

# GT | L1 | L2 ... decoded side by side: observed / one-step / imagined latents (viser), or export GIF + PLY
python scripts/visualize_decoded_hierarchy.py \
  --decoder-ckpt outputs/runs/pc_decoder/stageB/decoder_latest.pt \
  --zarr outputs/val_stackcube/StackCube-v1_expert_srmp.zarr --port 8080
python scripts/visualize_decoded_hierarchy.py --decoder-ckpt ... --zarr ... \
  --no-server --export-gif outputs/viz/decoded.gif --gif-mode imagined --export-ply outputs/viz/decoded_ply
```

Validation reports, per level, the two-sided nearest-neighbor error in cm (mean, median, p90, share of points > 3 cm). It does this for both observed latents (`val/l*/obs_*`) and one-step predictions (`val/l*/pred1_*`). The visualizer also prints imagined-rollout error against horizon.

## 4. Verification

```bash
pytest -v
```

## 5. OGBench Data Collection & Benchmark Evaluation (JEPA4R Comparison)

Point-H-JEPA supports collecting **unsegmented** 6D RGB-XYZ point clouds on OGBench manipulation tasks (`action_dim=5`, full scene geometry retained with invisible goal markers excluded) and evaluating frozen hierarchical latents alongside JEPA4R's published baselines (`state`, `jepa4r`, `lewm`, `pldm`, `lejepa`, `visreg`, `vjepa21`).

```bash
# 1. Install OGBench + MuJoCo extras (in a matching env or optional extra) and fetch JEPA4R code + weights
uv pip install -e ".[ogbench]"
python scripts/setup_jepa4r.py --skip-weights
python scripts/setup_jepa4r.py --tasks cube_single scene

# 2. Collect unsegmented 4,096-pt 6D OGBench world-model data (Zarr v3)
python -m point_hjepa.data.collect_ogbench_6d task=cube_single mode=wm_collect unsegmented=true num_points=4096
python -m point_hjepa.data.collect_ogbench_6d task=scene mode=wm_collect unsegmented=true num_points=4096

# Sharded at JEPA4R's scale (10k episodes/task, 40 single-threaded shards/task) on node-local /tmp,
# then one 2-GPU run per task (train_ogbench_<task>_local, num_l1_frames=13, 30k steps)
bash scripts/launch_ogbench_collection.sh
bash scripts/launch_ogbench_train.sh

# 3. Train Stage-1 Point-H-JEPA on OGBench (using winning r2-sig-x16 SIGReg weights [1.28, 2.56])
torchrun --nproc_per_node=4 scripts/train_ddp.py --config-name train_ogbench_v1 \
  datasets='[{path:outputs/datasets/ogbench/cube_single/cube_single_unsegmented.zarr,weight:1.0}]'

# 4. Inspect JEPA4R Table 1 reference baselines or dry-run the evaluation matrix
python scripts/eval_ogbench.py --print-baselines
python scripts/eval_ogbench.py --dry-run --methods point_hjepa jepa4r lewm state --tasks cube_single scene

# 5. Evaluate Point-H-JEPA (frozen concat/l1/l2 latents) and baselines on OGBench
python scripts/eval_ogbench.py \
  --tasks cube_single scene \
  --methods point_hjepa \
  --wm-ckpt outputs/runs/ogbench_cube_single_l2/checkpoint_latest.pt \
  --latent-mode concat --unsegmented
```
