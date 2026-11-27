"""PyTorch downstream OGBench policies matching JEPA4R's GCIQL (Cube) and FLARE-BC (Scene) protocols."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _make_mlp(
    in_dim: int,
    hidden_dims: tuple[int, ...],
    out_dim: int,
    *,
    layer_norm: bool = True,
    batch_norm: bool = False,
    activation: type[nn.Module] = nn.GELU,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    cur = int(in_dim)
    for h in hidden_dims:
        layers.append(nn.Linear(cur, int(h)))
        if batch_norm:
            layers.append(nn.BatchNorm1d(int(h)))
        elif layer_norm:
            layers.append(nn.LayerNorm(int(h)))
        layers.append(activation())
        cur = int(h)
    layers.append(nn.Linear(cur, int(out_dim)))
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------- #
# GCIQL (OGBench Cube tasks)
# --------------------------------------------------------------------------- #


@dataclass
class GCIQLConfig:
    obs_dim: int
    goal_dim: int
    action_dim: int = 5
    hidden_dims: tuple[int, ...] = (512, 512, 512)
    lr: float = 3e-4
    batch_size: int = 1024
    discount: float = 0.99
    tau: float = 0.005
    expectile: float = 0.9
    alpha: float = 1.0
    const_std: bool = True
    goal_tol: float = 0.4
    value_p_cur: float = 0.2
    value_p_traj: float = 0.5
    value_p_rand: float = 0.3


class GCIQLGoalSampler:
    """Hindsight goal & reward sampler matching JEPA4R's ``DistanceInputs``."""

    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        episode_ends: np.ndarray,
        *,
        num_cubes: int = 1,
        features: np.ndarray | None = None,
        cfg: GCIQLConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.observations = np.asarray(observations, dtype=np.float32)
        self.actions = np.asarray(actions, dtype=np.float32)
        self.episode_ends = np.asarray(episode_ends, dtype=np.int64)
        self.num_cubes = int(num_cubes)
        self.features = None if features is None else np.asarray(features, dtype=np.float32)
        self.cfg = cfg or GCIQLConfig(obs_dim=self.observations.shape[1], goal_dim=3 * self.num_cubes)
        self.rng = np.random.default_rng(seed)

        self.size = len(self.observations)
        self.ep_starts = np.concatenate([np.zeros((1,), dtype=np.int64), self.episode_ends[:-1]])
        ep_ids = np.empty(self.size, dtype=np.int64)
        valid_mask = np.ones(self.size, dtype=bool)
        for eid, (s, e) in enumerate(zip(self.ep_starts, self.episode_ends)):
            ep_ids[s:e] = eid
            if e > s:
                valid_mask[e - 1] = False
        self.ep_ids = ep_ids
        self.valid_idxs = np.flatnonzero(valid_mask)
        self.obj_xyz = np.stack(
            [self.observations[:, 19 + 9 * i : 22 + 9 * i] for i in range(self.num_cubes)],
            axis=1,
        )  # [size, num_cubes, 3]

    def _sample_value_goal_idxs(self, idxs: np.ndarray) -> np.ndarray:
        n = len(idxs)
        ep_ids = self.ep_ids[idxs]
        ends = self.episode_ends[ep_ids]
        rem = np.maximum(1, ends - 1 - idxs)

        u = self.rng.random(n)
        goals = idxs.copy()

        # Geometric future step in same episode
        geom_mask = (u >= self.cfg.value_p_cur) & (u < self.cfg.value_p_cur + self.cfg.value_p_traj)
        if np.any(geom_mask):
            offsets = self.rng.geometric(1.0 - self.cfg.discount, size=int(geom_mask.sum()))
            goals[geom_mask] = np.minimum(ends[geom_mask] - 1, idxs[geom_mask] + np.minimum(offsets, rem[geom_mask]))

        # Random step across dataset
        rand_mask = u >= (self.cfg.value_p_cur + self.cfg.value_p_traj)
        if np.any(rand_mask):
            goals[rand_mask] = self.rng.integers(0, self.size, size=int(rand_mask.sum()))
        return goals

    def _sample_actor_goal_idxs(self, idxs: np.ndarray) -> np.ndarray:
        ep_ids = self.ep_ids[idxs]
        ends = self.episode_ends[ep_ids]
        hi = np.maximum(idxs + 1, ends)
        return np.array([int(self.rng.integers(lo, end)) for lo, end in zip(idxs + 1, hi)], dtype=np.int64)

    def _build_obs(self, idxs: np.ndarray) -> np.ndarray:
        if self.features is None:
            return self.observations[idxs]
        return np.concatenate([self.features[idxs], self.observations[idxs, :19]], axis=-1).astype(np.float32)

    def _build_goal(self, idxs: np.ndarray) -> np.ndarray:
        xyz_flat = self.obj_xyz[idxs].reshape(len(idxs), 3 * self.num_cubes)
        if self.features is None:
            return xyz_flat.astype(np.float32)
        return np.concatenate([self.features[idxs], xyz_flat], axis=-1).astype(np.float32)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        sel = self.rng.choice(self.valid_idxs, size=int(batch_size), replace=True)
        val_goal_ids = self._sample_value_goal_idxs(sel)
        act_goal_ids = self._sample_actor_goal_idxs(sel)

        success = np.all(
            np.linalg.norm(self.obj_xyz[sel] - self.obj_xyz[val_goal_ids], axis=-1) <= self.cfg.goal_tol,
            axis=-1,
        )
        rewards = success.astype(np.float32) - 1.0
        masks = 1.0 - success.astype(np.float32)

        return {
            "observations": self._build_obs(sel),
            "next_observations": self._build_obs(sel + 1),
            "actions": self.actions[sel],
            "value_goals": self._build_goal(val_goal_ids),
            "actor_goals": self._build_goal(act_goal_ids),
            "rewards": rewards,
            "masks": masks,
        }


class TorchGCIQLPolicy(nn.Module):
    """PyTorch GCIQL (Expectile V + Twin Q + DDPG+BC Actor) matching ``vendor/gciql.py``."""

    def __init__(self, cfg: GCIQLConfig) -> None:
        super().__init__()
        self.cfg = cfg
        in_og = cfg.obs_dim + cfg.goal_dim
        self.value_net = _make_mlp(in_og, cfg.hidden_dims, 1, layer_norm=True)
        self.critic_q1 = _make_mlp(in_og + cfg.action_dim, cfg.hidden_dims, 1, layer_norm=True)
        self.critic_q2 = _make_mlp(in_og + cfg.action_dim, cfg.hidden_dims, 1, layer_norm=True)
        self.target_q1 = copy.deepcopy(self.critic_q1).requires_grad_(False)
        self.target_q2 = copy.deepcopy(self.critic_q2).requires_grad_(False)
        self.actor_mean = _make_mlp(in_og, cfg.hidden_dims, cfg.action_dim, layer_norm=True)
        self.actor_log_std = nn.Parameter(torch.zeros(cfg.action_dim))
        self.optimizer = torch.optim.Adam(
            [
                *self.value_net.parameters(),
                *self.critic_q1.parameters(),
                *self.critic_q2.parameters(),
                *self.actor_mean.parameters(),
                self.actor_log_std,
            ],
            lr=cfg.lr,
        )

    def _expectile_loss(self, diff: torch.Tensor) -> torch.Tensor:
        weight = torch.where(diff >= 0.0, self.cfg.expectile, 1.0 - self.cfg.expectile)
        return (weight * diff.pow(2)).mean()

    def _q_min(self, q1_net: nn.Module, q2_net: nn.Module, obs: torch.Tensor, goal: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, goal, act], dim=-1)
        return q1_net(x).squeeze(-1), q2_net(x).squeeze(-1)

    def update(self, batch: dict[str, np.ndarray | torch.Tensor], device: torch.device | str = "cpu") -> dict[str, float]:
        dev = torch.device(device)
        obs = torch.as_tensor(batch["observations"], device=dev, dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_observations"], device=dev, dtype=torch.float32)
        act = torch.as_tensor(batch["actions"], device=dev, dtype=torch.float32)
        v_goal = torch.as_tensor(batch["value_goals"], device=dev, dtype=torch.float32)
        a_goal = torch.as_tensor(batch["actor_goals"], device=dev, dtype=torch.float32)
        rew = torch.as_tensor(batch["rewards"], device=dev, dtype=torch.float32)
        mask = torch.as_tensor(batch["masks"], device=dev, dtype=torch.float32)

        with torch.no_grad():
            tq1, tq2 = self._q_min(self.target_q1, self.target_q2, obs, v_goal, act)
            tq = torch.minimum(tq1, tq2)
            next_v = self.value_net(torch.cat([next_obs, v_goal], dim=-1)).squeeze(-1)
            q_target = rew + self.cfg.discount * mask * next_v

        v = self.value_net(torch.cat([obs, v_goal], dim=-1)).squeeze(-1)
        value_loss = self._expectile_loss(tq - v)

        q1, q2 = self._q_min(self.critic_q1, self.critic_q2, obs, v_goal, act)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        mean = self.actor_mean(torch.cat([obs, a_goal], dim=-1))
        std = torch.exp(self.actor_log_std.clamp(-5.0, 2.0)).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        q_actions = torch.clamp(mean if self.cfg.const_std else dist.rsample(), -1.0, 1.0)

        for p in (*self.critic_q1.parameters(), *self.critic_q2.parameters()):
            p.requires_grad_(False)
        qa1, qa2 = self._q_min(self.critic_q1, self.critic_q2, obs, a_goal, q_actions)
        for p in (*self.critic_q1.parameters(), *self.critic_q2.parameters()):
            p.requires_grad_(True)

        qa = torch.minimum(qa1, qa2)
        q_loss = -qa.mean() / (qa.abs().mean().detach() + 1e-6)
        log_prob = dist.log_prob(act).sum(dim=-1)
        bc_loss = -(self.cfg.alpha * log_prob).mean()
        actor_loss = q_loss + bc_loss

        total_loss = value_loss + critic_loss + actor_loss
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            for p, tp in zip(self.critic_q1.parameters(), self.target_q1.parameters()):
                tp.data.lerp_(p.data, self.cfg.tau)
            for p, tp in zip(self.critic_q2.parameters(), self.target_q2.parameters()):
                tp.data.lerp_(p.data, self.cfg.tau)

        return {
            "total_loss": float(total_loss.item()),
            "value_loss": float(value_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "q_loss": float(q_loss.item()),
            "bc_loss": float(bc_loss.item()),
        }

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        goal: np.ndarray,
        *,
        device: torch.device | str = "cpu",
    ) -> np.ndarray:
        self.eval()
        dev = torch.device(device)
        obs_t = torch.as_tensor(obs, device=dev, dtype=torch.float32)
        goal_t = torch.as_tensor(goal, device=dev, dtype=torch.float32)
        squeeze = obs_t.ndim == 1
        if squeeze:
            obs_t, goal_t = obs_t.unsqueeze(0), goal_t.unsqueeze(0)
        mean = self.actor_mean(torch.cat([obs_t, goal_t], dim=-1))
        out = torch.clamp(mean, -1.0, 1.0).cpu().numpy()
        return out[0] if squeeze else out


def train_gciql(
    sampler: GCIQLGoalSampler,
    cfg: GCIQLConfig,
    *,
    train_steps: int = 1_000_000,
    device: str | torch.device = "cpu",
    eval_callback: Any | None = None,
    eval_interval: int = 100_000,
) -> tuple[TorchGCIQLPolicy, list[dict[str, float]]]:
    """Train ``TorchGCIQLPolicy`` on cached latents/states for ``train_steps`` updates."""
    dev = torch.device(device)
    policy = TorchGCIQLPolicy(cfg).to(dev)
    evals: list[dict[str, float]] = []
    for step in range(1, int(train_steps) + 1):
        policy.train()
        batch = sampler.sample(cfg.batch_size)
        policy.update(batch, device=dev)
        if eval_callback is not None and (step % eval_interval == 0 or step == train_steps):
            res = eval_callback(policy, step)
            evals.append({"step": step, **res})
    return policy, evals


def evaluate_gciql_online(
    policy: TorchGCIQLPolicy,
    env_factory: Any,
    encoder: Any,
    extractor: Any,
    *,
    num_tasks: int = 5,
    episodes_per_task: int = 50,
    horizon: int = 200,
    modality: str = "pcd",
    num_cubes: int = 1,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Evaluate a trained GCIQL policy across OGBench Cube's 5 official tasks."""
    per_task: dict[str, float] = {}
    env = env_factory()
    try:
        u = env.unwrapped
        for task_id in range(1, num_tasks + 1):
            successes = 0
            for ep in range(episodes_per_task):
                seed = 970_000_000 + task_id * 100_000 + ep
                captured_goal: dict[str, np.ndarray] = {}
                orig_render = getattr(u, "render", None)
                if modality != "state" and callable(orig_render):
                    active = {"flag": False}

                    def _wrapped_render(*r_args: Any, **r_kwargs: Any) -> Any:
                        img = orig_render(*r_args, **r_kwargs)
                        if (
                            not active["flag"]
                            and not captured_goal
                            and getattr(u, "_render_goal", False)
                        ):
                            active["flag"] = True
                            try:
                                captured_goal.update(
                                    extractor.render_frame(
                                        u._model,
                                        u._data,
                                        return_rgb_224=(modality == "rgb"),
                                    )
                                )
                            finally:
                                active["flag"] = False
                        return img

                    u.render = _wrapped_render

                try:
                    ob, info = env.reset(seed=seed, options={"task_id": task_id, "render_goal": True})
                finally:
                    if orig_render is not None:
                        u.render = orig_render

                goal_state = np.asarray(info["goal"], dtype=np.float32)

                if modality == "state":
                    goal_vec = np.concatenate(
                        [goal_state[19 + 9 * i : 22 + 9 * i] for i in range(num_cubes)], axis=-1
                    ).astype(np.float32)
                else:
                    if "goal_pcd" in info and modality == "pcd":
                        goal_obs = info["goal_pcd"]
                    elif "goal_rgb" in info and modality == "rgb":
                        goal_obs = info["goal_rgb"]
                    elif modality == "pcd" and "xyz" in captured_goal:
                        goal_obs = np.concatenate(
                            [captured_goal["xyz"], captured_goal["rgb"].astype(np.float32)], axis=-1
                        )
                    elif modality == "rgb" and "rgb_224" in captured_goal:
                        goal_obs = captured_goal["rgb_224"]
                    elif modality == "rgb" and "goal_rendered" in info:
                        goal_obs = np.asarray(info["goal_rendered"], dtype=np.uint8)
                    else:
                        raise RuntimeError("Could not capture official goal observation during OGBench reset")
                    z_goal = encoder.encode_one(goal_obs)
                    goal_xyz = np.concatenate(
                        [goal_state[19 + 9 * i : 22 + 9 * i] for i in range(num_cubes)], axis=-1
                    ).astype(np.float32)
                    goal_vec = np.concatenate([z_goal, goal_xyz], axis=-1).astype(np.float32)

                ep_success = False
                for _ in range(horizon):
                    if modality == "state":
                        obs_vec = np.asarray(ob, dtype=np.float32)
                    else:
                        frame = extractor.render_frame(u._model, u._data, return_rgb_224=(modality == "rgb"))
                        frame_obs = (
                            np.concatenate([frame["xyz"], frame["rgb"].astype(np.float32)], axis=-1)
                            if modality == "pcd"
                            else frame["rgb_224"]
                        )
                        z_cur = encoder.encode_one(frame_obs)
                        obs_vec = np.concatenate([z_cur, np.asarray(ob, dtype=np.float32)[:19]], axis=-1)

                    action = policy.act(obs_vec, goal_vec, device=device)
                    ob, _, terminated, truncated, step_info = env.step(action)
                    if step_info.get("success", False):
                        ep_success = True
                    if terminated or truncated:
                        break
                successes += int(ep_success)
            per_task[f"task{task_id}"] = 100.0 * successes / max(1, episodes_per_task)
    finally:
        env.close()
    mean_succ = sum(per_task.values()) / max(1, len(per_task))
    return {"success": mean_succ, "success_per_task": per_task}


# --------------------------------------------------------------------------- #
# FLARE Behavior Cloning (OGBench Scene)
# --------------------------------------------------------------------------- #


@dataclass
class FLAREBCConfig:
    latent_dim: int
    proprio_dim: int = 6
    action_dim: int = 5
    frame_stack: int = 3
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 100


def build_flare_features(
    latents: np.ndarray,
    proprio: np.ndarray,
    *,
    frame_stack: int = 3,
    proprio_dim: int = 6,
) -> np.ndarray:
    """Compute FLARE temporal-difference features ``[z_t, z_t-z_{t-1}, ..., proprio]``."""
    T, D = latents.shape
    if frame_stack < 1:
        raise ValueError(f"Expected frame_stack >= 1, got {frame_stack}")
    idx = np.arange(T)[:, None] - np.arange(frame_stack)[None, :]
    idx = np.clip(idx, 0, T - 1)
    hist = latents[idx]  # [T, frame_stack, D], hist[:, 0]=z_t, hist[:, 1]=z_{t-1}, ...

    parts = [hist[:, 0]]
    for k in range(1, frame_stack):
        parts.append(hist[:, k - 1] - hist[:, k])
    if proprio_dim > 0:
        parts.append(np.asarray(proprio[:, :proprio_dim], dtype=np.float32))
    return np.concatenate(parts, axis=-1).astype(np.float32)


class TorchFLAREBCPolicy(nn.Module):
    """BatchNorm MLP Behavior Cloning policy matching JEPA4R's CortexBench Scene setup."""

    def __init__(self, cfg: FLAREBCConfig) -> None:
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.frame_stack * cfg.latent_dim + cfg.proprio_dim
        self.net = _make_mlp(
            in_dim,
            cfg.hidden_dims,
            cfg.action_dim,
            layer_norm=False,
            batch_norm=True,
            activation=nn.ReLU,
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=cfg.lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def update(self, x: np.ndarray | torch.Tensor, actions: np.ndarray | torch.Tensor, device: torch.device | str = "cpu") -> float:
        dev = torch.device(device)
        x_t = torch.as_tensor(x, device=dev, dtype=torch.float32)
        a_t = torch.as_tensor(actions, device=dev, dtype=torch.float32)
        pred = self.forward(x_t)
        loss = F.mse_loss(pred, a_t)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def act(self, x: np.ndarray, *, device: torch.device | str = "cpu") -> np.ndarray:
        self.eval()
        dev = torch.device(device)
        x_t = torch.as_tensor(x, device=dev, dtype=torch.float32)
        squeeze = x_t.ndim == 1
        if squeeze:
            x_t = x_t.unsqueeze(0)
        out = torch.clamp(self.forward(x_t), -1.0, 1.0).cpu().numpy()
        return out[0] if squeeze else out


def train_flare_bc(
    features: np.ndarray,
    actions: np.ndarray,
    cfg: FLAREBCConfig,
    *,
    device: str | torch.device = "cpu",
    seed: int = 100,
    eval_callback: Any | None = None,
    eval_every: int = 5,
) -> tuple[TorchFLAREBCPolicy, list[dict[str, float]]]:
    """Train ``TorchFLAREBCPolicy`` with mini-batch MSE over ``cfg.epochs`` epochs."""
    rng = np.random.default_rng(seed)
    dev = torch.device(device)
    policy = TorchFLAREBCPolicy(cfg).to(dev)
    n = len(features)
    evals: list[dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        policy.train()
        perm = rng.permutation(n)
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            if len(idx) < 2:
                continue
            policy.update(features[idx], actions[idx], device=dev)
        if eval_callback is not None and (epoch % eval_every == 0 or epoch == cfg.epochs):
            res = eval_callback(policy, epoch)
            evals.append({"epoch": epoch, **res})
    return policy, evals


def evaluate_scene_online(
    policy: TorchFLAREBCPolicy,
    env_factory: Any,
    encoder: Any,
    extractor: Any,
    *,
    task_id: int = 1,
    episodes: int = 25,
    seed_base: int = 100,
    horizon: int = 750,
    modality: str = "pcd",
    frame_stack: int = 3,
    proprio_dim: int = 6,
    device: str | torch.device = "cpu",
) -> float:
    """Evaluate a trained FLARE-BC policy on one OGBench Scene task."""
    env = env_factory(task_id=task_id)
    successes = 0
    try:
        u = env.unwrapped
        for ep in range(episodes):
            seed = seed_base + ep
            env.reset(seed=seed, options={"task_id": task_id})
            hist_z: list[np.ndarray] = []
            ep_success = False
            for _ in range(horizon):
                oi = u.compute_ob_info()
                yaw = float(np.asarray(oi["proprio/effector_yaw"]).reshape(-1)[0])
                proprio = np.concatenate(
                    [
                        np.asarray(oi["proprio/effector_pos"], dtype=np.float32),
                        np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32),
                        np.asarray(oi["proprio/gripper_opening"], dtype=np.float32).reshape(-1),
                    ]
                )[:proprio_dim]

                if modality == "state":
                    z_cur = np.asarray(u.compute_observation(), dtype=np.float32)
                else:
                    frame = extractor.render_frame(u._model, u._data, return_rgb_224=(modality == "rgb"))
                    frame_obs = (
                        np.concatenate([frame["xyz"], frame["rgb"].astype(np.float32)], axis=-1)
                        if modality == "pcd"
                        else frame["rgb_224"]
                    )
                    z_cur = encoder.encode_one(frame_obs)

                hist_z.append(z_cur)
                win = [hist_z[-1 - min(k, len(hist_z) - 1)] for k in range(frame_stack)]
                win_arr = np.stack(win, axis=0)  # [frame_stack, D]
                parts = [win_arr[0]] + [win_arr[k - 1] - win_arr[k] for k in range(1, frame_stack)]
                if proprio_dim > 0:
                    parts.append(proprio)
                feat = np.concatenate(parts, axis=-1).astype(np.float32)

                action = policy.act(feat, device=device)
                _, _, terminated, truncated, info = env.step(action)
                if info.get("success", False):
                    ep_success = True
                if terminated or truncated:
                    break
            successes += int(ep_success)
    finally:
        env.close()
    return 100.0 * successes / max(1, episodes)
