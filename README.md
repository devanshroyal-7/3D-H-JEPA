# Point-H-JEPA — ManiSkill3 6D Point-Cloud Data Collection

GPU-parallelized **ManiSkill3** (`physx_cuda`) 6D RGB-XYZ point-cloud collector (`pd_joint_delta_pos`, 8D action space) with SRMP (VAMP) expert demonstrations and state-branched exploratory rollouts.

## Features

- **Unsegmented 6D Point Clouds (`[x, y, z, r, g, b]`)**: Robot-base frame coordinates (`float32` meters) and `[0, 255]` RGB (`uint8`), keeping tabletop, robot arm, targets, and distractors intact (no ground-truth segmentation mask trimming).
- **Two Dataset Modes**:
  - `expert_srmp`: Collision-free SRMP (VAMP) demonstrations on `StackCube-v1` and `PullCubeTool-v1`, saving `get_state_dict()` snapshots at key trajectory stages (`approach`, `grasp`, `lift`, `transport`, `place_or_pull`).
  - `branched_exploratory`: GPU-parallel `physx_cuda` rollouts restoring SRMP snapshots via `set_state_dict` (with immediate PD drive-target synchronization) and executing a 50% noisy expert / 30% interrupted (expert→random, mid-grasp gripper-held OU, release/drop) / 20% OU random exploration mixture.
- **Zarr v3 Storage (`BloscCodec(zstd, clevel=3)`)**: `xyz`, `rgb`, `action (8D)`, `proprio`, `achieved_point`, `ee_point`, `episode_ends`.

## Usage

```bash
# Collect both expert_srmp and branched_exploratory datasets for StackCube-v1
python -m point_hjepa.data.collect_maniskill3_6d \
  task_id=StackCube-v1 mode=both expert_episodes=20 branched_episodes=100 num_envs=64

# Collect for PullCubeTool-v1 (scale num_envs=128..256 on H100)
python -m point_hjepa.data.collect_maniskill3_6d \
  task_id=PullCubeTool-v1 mode=both expert_episodes=100 branched_episodes=5000 num_envs=128

# Run collector unit tests
pytest -v tests/test_collector.py
```
