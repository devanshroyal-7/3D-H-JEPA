# 3D Point H-JEPA

Data collection for a hierarchical, action-conditioned JEPA that plans from point clouds. The repo is built around [H-JEPA](https://arxiv.org/abs/2610.06805) ([code](https://github.com/kevinghst/H-JEPA)). This tree is the ManiSkill3 dataset pipeline: environments, an SRMP motion planner, point-cloud preprocessing, chunked episode storage, and a window sampler that stays inside episode boundaries.

## Layout

```
point_hjepa/data/collect.py      CLI
point_hjepa/data/preprocess.py   filter, base frame, crop, segmentation, FPS
point_hjepa/data/writer.py       async Zarr (HDF5 optional)
point_hjepa/data/dataset.py      context / frameskip / level-2 windows
point_hjepa/data/planners/       SRMP default, mplib optional
configs/                         one YAML per tier, plus the smoke run
tests/
```

## Environment

ManiSkill pins `mplib==0.1.1`, which needs **Python 3.11** and **numpy&lt;2**. Point clouds are rendered even when physics is on the CPU, so Vulkan has to be present. This machine has no NVIDIA GPU; the runs below used lavapipe:

```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
export SAPIEN_NO_DISPLAY=1
```

`point_hjepa.data.envs.ensure_vulkan` sets those when they are unset and CUDA is missing.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
# mplib comes in with mani_skill. SRMP is the planner that collection uses:
pip install "srmp>=0.3.0,<0.4"
```

Versions for the srmp 0.3.0 re-run below: Python 3.11.18, numpy 1.26.4, mani_skill 3.0.1, srmp 0.3.0, mplib 0.1.1.

## Collect

```bash
# 20 PickCube episodes, SRMP, CPU, the run reported below
python -m point_hjepa.data.collect --config configs/smoke_pickcube.yaml

# tier 1
python -m point_hjepa.data.collect --config configs/tier1_pickcube.yaml --num-episodes 200
python -m point_hjepa.data.collect --config configs/tier1_pushcube.yaml --num-episodes 200

# tier 2
python -m point_hjepa.data.collect --config configs/tier2_stackcube.yaml --num-episodes 100
python -m point_hjepa.data.collect --config configs/tier2_pullcubetool.yaml --num-episodes 100
python -m point_hjepa.data.collect --config configs/tier2_peginsertion.yaml --num-episodes 100

# CPU processes (SRMP is one scene per process)
python -m point_hjepa.data.collect --config configs/smoke_pickcube.yaml --workers 4

# optional comparison backend. Not used unless you ask. Failures do not fall back to it.
python -m point_hjepa.data.collect --config configs/smoke_pickcube.yaml --planner mplib
```

Every environment is created with `obs_mode="pointcloud"` and `control_mode="pd_ee_delta_pose"`. `sim_backend: auto` uses the CPU whenever the mix includes expert or noisy episodes, because SRMP reads a CPU SAPIEN scene. `sim_backend: gpu` with those sources raises `SrmpRequiresCpuScene` instead of switching planners. GPU vectorized collection is implemented for random exploration only. **This machine has no CUDA device, so that path was not run.**

Optional camera-pose noise is `randomize_camera: true` (`camera_pos_noise` in meters, applied to the base camera's local position at reset). The smoke config leaves it off.

### Action sources

`source_weights` default to 0.5 expert / 0.3 epsilon-noisy expert / 0.2 random workspace exploration. Counts use largest remainder, so 20 episodes are exactly 10 / 6 / 4. Expert and noisy episodes are SRMP screw plans tracked by closed-loop `pd_ee_delta_pose` actions (the controller's normalized ±0.1 m / ±0.1 rad deltas, including the sign flip from `rot_lower=-0.1`). Noisy actions add `N(0, noise_epsilon)` with `noise_epsilon=0.3`, then clip to [-1, 1]. Exploration samples in the workspace box and is pushed back inside when the TCP leaves it. Episodes are kept when they fail or collide. The env's success flag terminates the task, but collection keeps stepping until `max_steps` so a short success still fills the tier length.

## Dataset

Zarr group (HDF5 via `format: hdf5`):

| array | shape | meaning |
| --- | --- | --- |
| `points` | `(F, N, 3)` float32 | base-frame points, `N=2048` |
| `actions` | `(F, 7)` float32 | action applied at that frame; the last frame is padding |
| `goal_coords` | `(F, G)` float32 | goal from `obs["extra"]` when that mode provides one |
| `robot_state` | `(F, 25)` float32 | `qpos` (9), `qvel` (9), TCP xyz + wxyz |
| `point_count` | `(F,)` int32 | real points before repeat-padding up to `N` |
| `success`, `done` | `(F,)` bool | `done` is true only on the terminal frame |
| `episode_ends` | `(E,)` int64 | exclusive end index into `F` |
| `episode_source` | `(E,)` int8 | 0 expert, 1 noisy, 2 explore |
| `episode_success` | `(E,)` bool | success on the terminal frame |

`rgb` `(F, N, 3)` uint8 is written when `store_rgb: true`.

Preprocessing, per frame: drop points with `w==0` or non-finite xyz, transform into the robot base frame, crop to `workspace_min/max`, keep robot and task-object points, keep at most `table_keep` (256) random table points, drop the ground, then batched farthest-point sampling to `N`. Clouds that are still larger than 4096 are randomly capped before FPS. Voxel pooling is `downsample: voxel`.

PickCube's pointcloud observation includes `extra["goal_pos"]`, so `goal_coords` for the smoke set is `obs.extra:goal_pos`. Push, stack, peg, and pull only put the goal in `extra` when `"state"` is in the obs mode. Those runs record the simulator goal and set `goal_source` to `env`.

### Windows

```python
from point_hjepa.data.dataset import PointCloudWindowDataset

ds = PointCloudWindowDataset(
    "outputs/smoke/pickcube.zarr",
    context_frames=3,   # default
    frameskip=5,        # level-1 stride
    horizon=4,
    level2_stride=10,
    level2_horizon=2,
)
sample = ds[0]
# context_points        (3, 2048, 3)
# context_actions       (2, 5, 7)
# target_points         (4, 2048, 3)
# target_actions        (4, 5, 7)
# level2_context_points (3, 2048, 3)
# level2_target_points  (2, 2048, 3)
# level2_*_actions      stride 10
```

An anchor `t` is kept only when both the level-1 and level-2 index sets lie in `[0, episode_length)`. Windows never cross `episode_ends`.

## SRMP on 0.3.0

SRMP is the default planner. mplib is still optional and is not a fallback. Collection calls `set_base_pose` with the robot base and plans in the world frame. Static geometry is `read_sim`, with `set_base_contact_allowed` so the fixed base may rest on the table. Search fallback uses coal: VAMP's base spheres still overlap that table, and `validity_backend="auto"` then drops back to coal.

Re-checked on srmp 0.3.0, PickCube Panda base at `x = -0.615` m:

On 0.2.2 the screw miss was the world-frame Jacobian ignoring base translation, and the table and held-object checks came from VAMP sphere padding versus exact `is_state_colliding()`. Those workarounds are gone except where 0.3.0 still needs them. `scripts/repro_srmp_read_sim_ground.py` prints the ground-plane pose from that investigation.

1. **`plan_screw` after `set_base_pose` — fixed.** The same nearby-configuration screw that missed by **0.187 m** on 0.2.2 now misses by **3.4 mm** (a reset-pose screw in the smoke probe missed by **1.7 mm**). The base-frame scene shift is gone.
2. **Imported table — fixed by `set_base_contact_allowed`.** `read_sim` still places `panda_link0` against the table, so every state collides until the switch is on. With it on, the reset pose does not collide. The cut-out slab is no longer used for SRMP. mplib still builds that cut-out, because it has no equivalent switch.
3. **Ground plane — not a false collision.** Ground-only clearance at the reset pose is **0.924 m**. Of 103 self-collision-free samples with the hand at `x <= 0` and `z > 0.02`, none collided with the imported plane. The plane is kept. The occupancy grid still starts at `z = -1.2` so that floor is inside the distance field.
4. **Held cube near the table — lift kept.** A grasp 5 mm off the table penetrates by about **4.2 cm** in coal (`distance_to_robot_collision` about −0.042 m), so a collision-checked plan cannot start there. The 5 cm collision-off lift still runs before the transport plan. The old mismatch did not reproduce: when coal reported the state free (hand 10 cm above that cube), wAstar returned a path.

`mesh_sampling_resolution` is the VAMP surface spacing in metres (`make_planner` context). It defaults to **0.01** (srmp's own default) and to **0.03** for large-mesh tasks (`PickClutterYCB-v1`). Set it in the YAML to override. On a 1.2 × 1.2 × 0.4 m box (3072 triangles), building the VAMP scene took **45.08 s** and **12.4 GiB** of RSS at 0.01 m, and **0.358 s** and **420 MiB** at 0.03 m. A validity query went from 0.109 ms to 0.020 ms. PickCube does not use that path (coal `plan_screw`); the setting is what the search fallback and any VAMP check will use.

## Smoke run (20× PickCube-v1, srmp 0.3.0)

`configs/smoke_pickcube.yaml`, one CPU process, SRMP 0.3.0, lavapipe, `N=2048`, no RGB.

| | |
| --- | --- |
| episodes | 20 (10 expert, 6 noisy, 4 explore) |
| frames | 1020 (51 per episode) |
| success | **10/10 expert**, 0/6 noisy, 0/4 explore |
| SRMP plan failures | 2, both on noisy episodes (start state invalid after the arm was driven into collision). Episodes were kept. mplib was not called. |
| wall time | **46.46 s** |
| throughput | **21.95 env steps/s**, 0.430 episodes/s |
| disk | **10.10 MiB** |
| windows | **220**, every one inside its episode |
| goal | `obs.extra:goal_pos` |

A separate 2-episode run with `--workers 2` finished in 16.31 s (6.25 steps/s). Most of that is process and Vulkan startup; the 20-episode serial number is the one to quote for throughput.

One-seed checks, same stack, not the smoke set:

| task | frames | success | plan failures | wall | steps/s |
| --- | --- | --- | --- | --- | --- |
| StackCube-v1 | 151 | yes | 0 | 29.34 s | 5.15 |
| PullCubeTool-v1 | 201 | no | 0 | 17.20 s | 11.69 |
| PegInsertionSide-v1 | 201 | no | 2 | 58.28 s | 3.45 |

SRMP did not block tier 1 or tier 2: all three envs stepped, wrote windows, and stayed on SRMP. Pull and peg did not succeed on seed 0. Their experts are scripted grasps on top of `plan_screw` (peg keeps a top-down TCP orientation), not a tuned insertion/pull solution. Failed episodes are stored.

## Tier 3

Not collected.

- **PickClutterYCB-v1** is registered, but `~/.maniskill` does not contain `tasks/pick_clutter/ycb_train_5k.json.gz` or the YCB meshes. The config exits with the download commands (`mani_skill.utils.download_asset ycb` and `pick_clutter_ycb_configs`). In mani_skill 3.0.1, `PickClutterYCBEnv.evaluate` returns constant `False` and `_get_obs_extra` is empty in pointcloud mode, so success labels and `goal_coords` would not come from the observation even after the assets are present. An expert is implemented for when the assets exist.
- **SetTable-v1** is not registered. ManiSkill-HAB is a separate package and needs ReplicaCAD rearranged scenes plus a mobile manipulator. The Panda SRMP backend does not cover that embodiment. `configs/tier3_settable.yaml` exits with that reason.

## Tests

```bash
pytest
```

Tests cover preprocessing, the `pd_ee_delta_pose` scale, table-cutout geometry (still used by mplib), the 10/6/4 mix, Zarr and HDF5 windows, the SetTable stub, mesh-sampling defaults, and a live `plan_screw` check that a moved base stays within 2 cm on srmp 0.3.0.

## Not verified here

- GPU simulation and the vectorized explore path (`nvidia-smi` reports no device).
- An end-to-end mplib collection. The backend is selectable and plans in the base frame without `set_base_pose`; the smoke run did not call it.
- Camera randomization inside the simulator (the flag was off).
- PickClutterYCB after the asset download, and any ManiSkill-HAB task.
- A success rate for PullCubeTool or PegInsertionSide beyond the single seeds above.
