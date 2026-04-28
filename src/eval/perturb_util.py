from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch


PERTURB_MODES = ("none", "random_small", "short_large", "place_slowdown")


@dataclass
class PerturbConfig:
    mode: str


@dataclass
class PerturbContext:
    step_idx: int
    num_envs: int
    device: torch.device
    furniture_name: Optional[str]
    task_name: Optional[str]
    skill_states: list[Optional[str]]
    ee_pos_vel: Optional[torch.Tensor]


@dataclass
class PerturbStats:
    total_steps: int = 0
    applied_steps: int = 0
    applied_env_steps: int = 0
    mode_counts: dict[str, int] = field(default_factory=dict)

    def record(self, mode: str, forces: torch.Tensor):
        self.total_steps += 1
        applied_mask = torch.linalg.norm(forces.detach(), dim=-1) > 0
        applied_env_steps = int(applied_mask.sum().item())
        if applied_env_steps == 0:
            return
        self.applied_steps += 1
        self.applied_env_steps += applied_env_steps
        self.mode_counts[mode] = self.mode_counts.get(mode, 0) + applied_env_steps

    def summary(self) -> dict[str, int | dict[str, int]]:
        return {
            "total_steps": self.total_steps,
            "applied_steps": self.applied_steps,
            "applied_env_steps": self.applied_env_steps,
            "mode_counts": dict(self.mode_counts),
        }


class PerturbRunner:
    """Generate end-effector perturbation forces for evaluation rollouts.

    Keep evaluation CLI intentionally small: edit the defaults below when tuning
    perturbation behavior.
    """

    def __init__(self, mode: str):
        if mode not in PERTURB_MODES:
            raise ValueError(
                f"Invalid perturb mode `{mode}`. Valid modes: {PERTURB_MODES}"
            )

        self.config = PerturbConfig(mode=mode)

        # Tunable defaults. These are intentionally not exposed by evaluate_model.py.
        self.perturb_per_timesteps = 25
        self.perturb_min_force = 5.0
        self.perturb_max_force = 10.0
        self.perturb_delay = 5
        self.perturb_state = "place"
        self.perturb_furniture: Optional[str] = None
        self.perturb_slowdown_gain = 30.0
        self.perturb_down_vel_threshold = 1e-4
        self.perturb_seed = 0

        self.stats = PerturbStats()
        self._generator = torch.Generator()
        self._generator.manual_seed(self.perturb_seed)
        self._short_large_match_steps: Optional[torch.Tensor] = None
        self._short_large_fired: Optional[torch.Tensor] = None

        self._mode_fns: dict[str, Callable[[PerturbContext], torch.Tensor]] = {
            "none": self._zero_force,
            "random_small": self._random_small,
            "short_large": self._short_large,
            "place_slowdown": self._place_slowdown,
        }

    @property
    def mode(self) -> str:
        return self.config.mode

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    @property
    def requires_skill_annotations(self) -> bool:
        return self.mode in {"short_large", "place_slowdown"}

    def reset_episode(self, num_envs: int, device: torch.device):
        self._short_large_match_steps = torch.zeros(
            num_envs, dtype=torch.int64, device=device
        )
        self._short_large_fired = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def compute_force(self, context: PerturbContext) -> torch.Tensor:
        forces = self._mode_fns[self.mode](context)
        self.stats.record(self.mode, forces)
        return forces

    def _zero_force(self, context: PerturbContext) -> torch.Tensor:
        return torch.zeros((context.num_envs, 3), device=context.device)

    def _random_small(self, context: PerturbContext) -> torch.Tensor:
        if context.step_idx % self.perturb_per_timesteps != 0:
            return self._zero_force(context)
        return self._sample_random_forces(
            context.num_envs,
            context.device,
            min_force=0.0,
            max_force=self.perturb_max_force,
        )

    def _short_large(self, context: PerturbContext) -> torch.Tensor:
        self._ensure_short_large_state(context)
        matches = self._matches_target_state(context)
        assert self._short_large_match_steps is not None
        assert self._short_large_fired is not None

        self._short_large_match_steps = torch.where(
            matches,
            self._short_large_match_steps + 1,
            torch.zeros_like(self._short_large_match_steps),
        )
        should_fire = (
            matches
            & ~self._short_large_fired
            & (self._short_large_match_steps >= self.perturb_delay + 1)
        )

        forces = self._zero_force(context)
        if should_fire.any():
            sampled_forces = self._sample_random_forces(
                context.num_envs,
                context.device,
                min_force=self.perturb_min_force,
                max_force=self.perturb_max_force,
            )
            forces[should_fire] = sampled_forces[should_fire]
            self._short_large_fired[should_fire] = True
        return forces

    def _place_slowdown(self, context: PerturbContext) -> torch.Tensor:
        if context.ee_pos_vel is None:
            raise ValueError("place_slowdown perturbation requires ee_pos_vel.")

        ee_pos_vel = context.ee_pos_vel.to(device=context.device, dtype=torch.float32)
        if ee_pos_vel.shape != (context.num_envs, 3):
            raise ValueError(
                "ee_pos_vel must have shape "
                f"({context.num_envs}, 3), got {tuple(ee_pos_vel.shape)}"
            )

        skill_matches = self._matches_skill_suffix(context.skill_states, "place").to(
            device=context.device
        )
        downward_speed = torch.clamp(-ee_pos_vel[:, 2], min=0.0)
        moving_down = downward_speed > self.perturb_down_vel_threshold
        should_apply = skill_matches & moving_down

        forces = self._zero_force(context)
        if not should_apply.any():
            return forces

        speed = torch.linalg.norm(ee_pos_vel, dim=-1).clamp_min(1e-6)
        direction = -ee_pos_vel / speed.unsqueeze(-1)
        magnitude = torch.clamp(
            downward_speed * self.perturb_slowdown_gain,
            max=self.perturb_max_force,
        )
        forces[should_apply] = direction[should_apply] * magnitude[
            should_apply
        ].unsqueeze(-1)
        return forces

    def _ensure_short_large_state(self, context: PerturbContext):
        if (
            self._short_large_match_steps is None
            or self._short_large_fired is None
            or self._short_large_match_steps.shape[0] != context.num_envs
            or self._short_large_match_steps.device != context.device
        ):
            self.reset_episode(context.num_envs, context.device)

    def _matches_target_state(self, context: PerturbContext) -> torch.Tensor:
        furniture_matches = (
            self.perturb_furniture is None
            or context.furniture_name == self.perturb_furniture
            or context.task_name == self.perturb_furniture
        )
        if not furniture_matches:
            return torch.zeros(context.num_envs, dtype=torch.bool, device=context.device)
        return self._matches_skill_suffix(
            context.skill_states, self.perturb_state
        ).to(device=context.device)

    def _matches_skill_suffix(
        self, skill_states: list[Optional[str]], target_state: str
    ) -> torch.Tensor:
        target_state = str(target_state)
        matches = []
        for skill_state in skill_states:
            if skill_state is None:
                matches.append(False)
                continue
            skill_state = str(skill_state)
            matches.append(
                skill_state == target_state or skill_state.endswith(f"-{target_state}")
            )
        return torch.tensor(matches, dtype=torch.bool)

    def _sample_random_forces(
        self,
        num_envs: int,
        device: torch.device,
        min_force: float,
        max_force: float,
    ) -> torch.Tensor:
        directions = torch.randn(
            (num_envs, 3), generator=self._generator, dtype=torch.float32
        )
        directions = directions / torch.linalg.norm(
            directions, dim=-1, keepdim=True
        ).clamp_min(1e-6)
        scales = torch.rand(
            (num_envs, 1), generator=self._generator, dtype=torch.float32
        )
        scales = min_force + scales * (max_force - min_force)
        return (directions * scales).to(device=device)
