# Point-H-JEPA (Utonia Tiny + Hierarchical JEPA + Goal Mapper)

PyTorch DDP implementation of a bottom-up, unsegmented 6D RGB-XYZ point-cloud Hierarchical Joint-Embedding Predictive Architecture (**Point-H-JEPA**) with `utonia_tiny`, an Attentive Pooling Bridge, Multi-Scale AdaLN-Zero Predictors, Multi-Scale Inverse Dynamics Models (IDM) with stop-gradient isolation, DDP-aware Epps-Pulley `SIGReg`, and a Stage-2 Hindsight 3D Goal Mapper (`pd_joint_delta_pos`, `action_dim = 8`).

## Training (4× H100 DDP)

```bash
# Stage 1: Goal-free Point-H-JEPA world model training (global batch size 256)
torchrun --nproc_per_node=4 scripts/train_ddp.py

# Stage 2: Hindsight 3D point goal mapper training (frozen world model)
python scripts/train_goal_mapper.py \
  goal_mapper.checkpoint=outputs/runs/sanity_l2/checkpoint_latest.pt
```

## Local Single-GPU Smoke Test (RTX 5080)

```bash
python scripts/train_ddp.py \
  train.batch_size_per_gpu=4 train.epochs=1 train.max_steps=5 \
  train.num_workers=0 train.diag_num_frames=64 wandb.enabled=false

python scripts/train_goal_mapper.py \
  goal_mapper.checkpoint=outputs/runs/sanity_l2/checkpoint_latest.pt \
  goal_mapper.batch_size=16 goal_mapper.epochs=1 goal_mapper.max_steps=5 \
  train.num_workers=0 wandb.enabled=false
```

## Verification

```bash
pytest -v tests/test_shapes_and_grads.py
```
