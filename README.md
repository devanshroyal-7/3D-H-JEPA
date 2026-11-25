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

## 2. Training (4× H100 DDP & Local Smoke Test)

```bash
# Stage 1: Goal-free Point-H-JEPA world model training (global batch size 256)
torchrun --nproc_per_node=4 scripts/train_ddp.py

# Stage 2: Hindsight 3D point goal mapper training (frozen world model)
python scripts/train_goal_mapper.py \
  goal_mapper.checkpoint=outputs/runs/sanity_l2/checkpoint_latest.pt
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

## 3. Verification

```bash
pytest -v
```
