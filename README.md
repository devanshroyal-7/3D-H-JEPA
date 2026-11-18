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

The project environment is managed by [uv](https://docs.astral.sh/uv/). `.python-version` pins 3.11 and `uv.lock` pins the rest. `uv sync` creates `.venv`, installs the package in editable mode, and installs the `dev` group (pytest).

```bash
# https://docs.astral.sh/uv/getting-started/installation/
uv sync
```

Run tools from that environment with `uv run` (no need to activate `.venv`):

```bash
uv run python -m point_hjepa.data.collect --config configs/smoke_pickcube.yaml
uv run pytest
```

`mplib` is installed because `mani_skill` depends on it. The `mplib` extra exists only to name that pin explicitly: `uv sync --extra mplib`.

Torch is taken from the PyTorch CPU index (`2.15.0+cpu` in the lockfile). ManiSkill's GPU simulator is SAPIEN/PhysX, not this wheel. To build a CUDA torch instead, point `[tool.uv.sources] torch` at a CUDA index and run `uv lock`.

Versions locked for the numbers below: Python 3.11, numpy 1.26.4, mani_skill 3.0.1, srmp 0.2.2, mplib 0.1.1, torch 2.15.0+cpu.

## Collect

```bash
# 20 PickCube episodes, SRMP, CPU, the run reported below
uv run python -m point_hjepa.data.collect --config configs/smoke_pickcube.yaml

# tier 1
uv run python -m point_hjepa.data.collect --config configs/tier1_pickcube.yaml --num-episodes 200
uv run python -m point_hjepa.data.collect --config configs/tier1_pushcube.yaml --num-episodes 200

# tier 2
uv run python -m point_hjepa.data.collect --config configs/tier2_stackcube.yaml --num-episodes 100
uv run python -m point_hjepa.data.collect --config configs/tier2_pullcubetool.yaml --num-episodes 100
uv run python -m point_hjepa.data.collect --config configs/tier2_peginsertion.yaml --num-episodes 100

# CPU processes (SRMP is one scene per process)
uv run python -m point_hjepa.data.collect --config configs/smoke_pickcube.yaml --workers 4

# optional comparison backend. Not used unless you ask. Failures do not fall back to it.
uv run python -m point_hjepa.data.collect --config configs/smoke_pickcube.yaml --planner mplib
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

## SRMP workarounds

SRMP is the default planner (`planner: srmp`). mplib is a separate backend. The production planner does **not** call `set_base_pose`.

Measured on srmp 0.2.2 with the PickCube Panda, whose base is mounted at `x = -0.615` m (`outputs/smoke/pickcube.zarr_report.json`):

1. **`plan_screw` after `set_base_pose`.** Planning to the FK pose of a nearby configuration missed by **0.187 m** (xyz error about `[-0.031, 0.088, 0.162]`). The same target, expressed in the robot base frame with the planner base left at the identity, missed by **0.00169 m**. Collection always uses the identity-base path. A residual retry subtracts a measured bias only when a screw still lands more than 3 cm off, and a failed screw may call SRMP `plan()` (wA*). It does not call mplib.
2. **`read_sim` table.** Before import the reset configuration was not in collision. After `read_sim` it was. Removing the imported table (`scene-0_table-workspace::box0`) cleared the collision. The table mesh spans roughly `z ∈ [-0.92, 0]` and covers the robot mount, so the top is flush with the base. Collection drops that mesh and adds a 3 cm table-top slab with a 0.18 m square cutout around the base, lowered 2 mm so it is not flush with `z = 0`.
3. **`read_sim` ground.** The ground actor is at the origin while its collision plane is offset locally to about `z = -0.92`. The imported plane (`scene-0_ground::plane2`) is discarded. A slab centered at `z = -0.08` is added instead. The occupancy grid origin is `z = -0.5`; the library default starts at `z = 0` and would clamp this geometry onto the base.
4. **Held cube near the table.** After the fingers close, the expert lifts **0.05 m** with collision checking off (`post_grasp_lift`, a Cartesian phase, not `plan_screw`). Only then is the object attached and a collision-aware transport planned.

If the cutout is still in collision at the reset `qpos`, that episode's table slabs are removed and the episode is kept. On the smoke run this branch did not fire (`plan_failures` on the 10 clean expert episodes was 0).

## Smoke run (20× PickCube-v1)

`configs/smoke_pickcube.yaml`, one CPU process, SRMP, lavapipe, `N=2048`, no RGB.

| | |
| --- | --- |
| episodes | 20 (10 expert, 6 noisy, 4 explore) |
| frames | 1020 (51 per episode: 50 steps + terminal obs) |
| success | **10/10 expert**, 0/6 noisy, 0/4 explore |
| SRMP plan failures | 2, both on noisy episodes (start state invalid after the arm was driven into collision). Episodes were kept. Cartesian tracking was used for those phases. mplib was not called. |
| wall time | **51.29 s** (excludes the 0.30 s probe) |
| throughput | **19.89 env steps/s**, 0.390 episodes/s |
| disk | **10,635,222 bytes (10.14 MiB)** |
| windows | **220**, every one inside its episode |
| `point_count` | min 1388, mean 1852, max 2048 (short clouds are repeat-padded) |
| goal | `obs.extra:goal_pos`, shape `(1020, 3)` |
| robot state | `(1020, 25)` |

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
uv run pytest
```

14 tests: preprocessing, the `pd_ee_delta_pose` scale (including the negative `rot_lower`), table cutout geometry, the 10/6/4 mix, Zarr and HDF5 windowing across two episodes, the SetTable stub, and the live `plan_screw` base-pose check (greater than 10 cm with `set_base_pose`, under 3 cm in the base frame).

## Not verified here

- GPU simulation and the vectorized explore path (`nvidia-smi` reports no device).
- An end-to-end mplib collection. The backend is selectable and plans in the base frame without `set_base_pose`; the smoke run did not call it.
- Camera randomization inside the simulator (the flag was off).
- PickClutterYCB after the asset download, and any ManiSkill-HAB task.
- A success rate for PullCubeTool or PegInsertionSide beyond the single seeds above.
