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
    modified_steps: int = 0
    modified_env_steps: int = 0
    mode_counts: dict[str, int] = field(default_factory=dict)

    def record_force(self, mode: str, forces: torch.Tensor):
        self.total_steps += 1
        applied_mask = torch.linalg.norm(forces.detach(), dim=-1) > 0
        applied_env_steps = int(applied_mask.sum().item())
        if applied_env_steps == 0:
            return
        self.applied_steps += 1
        self.applied_env_steps += applied_env_steps
        self.mode_counts[mode] = self.mode_counts.get(mode, 0) + applied_env_steps

    def record_action_mod(self, num_envs_modified: int):
        if num_envs_modified == 0:
            return
        self.modified_steps += 1
        self.modified_env_steps += num_envs_modified

    def summary(self) -> dict[str, int | dict[str, int]]:
        return {
            "total_steps": self.total_steps,
            "applied_steps": self.applied_steps,
            "applied_env_steps": self.applied_env_steps,
            "modified_steps": self.modified_steps,
            "modified_env_steps": self.modified_env_steps,
            "mode_counts": dict(self.mode_counts),
        }


class PerturbRunner:
    """Generate end-effector perturbation forces for evaluation rollouts.

    Keep evaluation CLI intentionally small: edit the defaults below when tuning
    perturbation behavior.  Parameters are grouped by perturbation mode so you can
    jump straight to the block that matches the mode you are using.

    Modes that *modify actions* (currently only place_slowdown) produce replayable
    trajectories because the modified action is saved into the rollout.  Modes that
    apply external *forces* (random_small, short_large) do not.
    """

    def __init__(self, mode: str):
        if mode not in PERTURB_MODES:
            raise ValueError(
                f"Invalid perturb mode `{mode}`. Valid modes: {PERTURB_MODES}"
            )

        self.config = PerturbConfig(mode=mode)

        # -- common --
        self.perturb_seed = 0

        # -- random_small --
        self.random_small_interval = 25      # steps between random force applications
        self.random_small_max_force = 10.0    # max force magnitude

        # -- short_large --
        self.short_large_min_force = 5.0              # min force when firing
        self.short_large_max_force = 10.0              # max force when firing
        self.short_large_delay = 5                     # steps to wait after entering target state
        self.short_large_trigger_state = "place"       # skill state that triggers the impulse
        self.short_large_trigger_furniture: Optional[str] = None  # restrict to a specific furniture / task

        # -- place_slowdown --
        self.place_slowdown_speed_ratio = 0.8          # z-axis action scale during place (1.0 = off)
        self.place_slowdown_random_force = 0.0         # max random force overlaid (0 = off)
        self.place_slowdown_random_interval = 25       # steps between random force applications

        self.stats = PerturbStats()
        self._generator = torch.Generator()
        self._generator.manual_seed(self.perturb_seed)
        self._short_large_match_steps: Optional[torch.Tensor] = None
        self._short_large_fired: Optional[torch.Tensor] = None

        self._mode_fns: dict[str, Callable[[PerturbContext], torch.Tensor]] = {
            "none": self._zero_force,
            "random_small": self._random_small,
            "short_large": self._short_large,
            "place_slowdown": self._place_slowdown_random,
        }

    # -- public properties --------------------------------------------------

    @property
    def mode(self) -> str:
        return self.config.mode

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    @property
    def requires_skill_annotations(self) -> bool:
        return self.mode in {"short_large", "place_slowdown"}

    @property
    def modifies_action(self) -> bool:
        return self.mode == "place_slowdown"

    @property
    def applies_force(self) -> bool:
        if self.mode in {"random_small", "short_large"}:
            return True
        if self.mode == "place_slowdown" and self.place_slowdown_random_force > 0:
            return True
        return False

    # -- public methods -----------------------------------------------------

    def reset_episode(self, num_envs: int, device: torch.device):
        self._short_large_match_steps = torch.zeros(
            num_envs, dtype=torch.int64, device=device
        )
        self._short_large_fired = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def compute_force(self, context: PerturbContext) -> torch.Tensor:
        forces = self._mode_fns[self.mode](context)
        self.stats.record_force(self.mode, forces)
        return forces

    def modify_action(
        self, action: torch.Tensor, context: PerturbContext
    ) -> torch.Tensor:
        """Return a modified copy of *action* (replay-safe)."""
        if self.mode == "place_slowdown":
            return self._modify_action_place_slowdown(action, context)
        return action

    # -- force: none --------------------------------------------------------

    def _zero_force(self, context: PerturbContext) -> torch.Tensor:
        return torch.zeros((context.num_envs, 3), device=context.device)

    # -- force: random_small ------------------------------------------------

    def _random_small(self, context: PerturbContext) -> torch.Tensor:
        if context.step_idx % self.random_small_interval != 0:
            return self._zero_force(context)
        return self._sample_random_forces(
            context.num_envs,
            context.device,
            min_force=0.0,
            max_force=self.random_small_max_force,
        )

    # -- force: short_large -------------------------------------------------

    def _short_large(self, context: PerturbContext) -> torch.Tensor:
        self._ensure_short_large_state(context)
        matches = self._matches_short_large_target(context)
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
            & (self._short_large_match_steps >= self.short_large_delay + 1)
        )

        forces = self._zero_force(context)
        if should_fire.any():
            sampled_forces = self._sample_random_forces(
                context.num_envs,
                context.device,
                min_force=self.short_large_min_force,
                max_force=self.short_large_max_force,
            )
            forces[should_fire] = sampled_forces[should_fire]
            self._short_large_fired[should_fire] = True
        return forces

    # -- force: place_slowdown_random ---------------------------------------

    def _place_slowdown_random(self, context: PerturbContext) -> torch.Tensor:
        if (
            self.place_slowdown_random_force <= 0
            or context.step_idx % self.place_slowdown_random_interval != 0
        ):
            return self._zero_force(context)
        return self._sample_random_forces(
            context.num_envs,
            context.device,
            min_force=0.0,
            max_force=self.place_slowdown_random_force,
        )

    # -- action modification ------------------------------------------------

    def _modify_action_place_slowdown(
        self, action: torch.Tensor, context: PerturbContext
    ) -> torch.Tensor:
        if self.place_slowdown_speed_ratio >= 1.0:
            return action

        skill_matches = self._matches_skill_suffix(
            context.skill_states, "place"
        ).to(device=action.device)
        num_modified = int(skill_matches.sum().item())
        self.stats.record_action_mod(num_modified)
        if num_modified == 0:
            return action

        modified = action.clone()
        modified[skill_matches, 2] *= self.place_slowdown_speed_ratio
        return modified

    # -- helpers ------------------------------------------------------------

    def _ensure_short_large_state(self, context: PerturbContext):
        if (
            self._short_large_match_steps is None
            or self._short_large_fired is None
            or self._short_large_match_steps.shape[0] != context.num_envs
            or self._short_large_match_steps.device != context.device
        ):
            self.reset_episode(context.num_envs, context.device)

    def _matches_short_large_target(self, context: PerturbContext) -> torch.Tensor:
        furniture_matches = (
            self.short_large_trigger_furniture is None
            or context.furniture_name == self.short_large_trigger_furniture
            or context.task_name == self.short_large_trigger_furniture
        )
        if not furniture_matches:
            return torch.zeros(context.num_envs, dtype=torch.bool, device=context.device)
        return self._matches_skill_suffix(
            context.skill_states, self.short_large_trigger_state
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
