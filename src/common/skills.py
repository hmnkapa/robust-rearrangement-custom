from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch


SKILL_ORDER = ("pick", "place", "insert", "screw", "push")
SKILL_TO_INDEX = {skill: idx for idx, skill in enumerate(SKILL_ORDER)}
SKILL_TO_ONEHOT = {
    skill: np.eye(len(SKILL_ORDER), dtype=np.float32)[idx]
    for idx, skill in enumerate(SKILL_ORDER)
}


def _normalize_skill_label(skill: Optional[str]) -> Optional[str]:
    if skill is None:
        return None
    if isinstance(skill, bytes):
        skill = skill.decode("utf-8")
    return skill


def skill_to_onehot_tensor(
    skill: Optional[str],
    skill_dim: int,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    skill = _normalize_skill_label(skill)
    onehot = torch.zeros(skill_dim, device=device, dtype=dtype)
    if skill is None:
        return onehot

    if skill not in SKILL_TO_INDEX:
        raise ValueError(
            f"Unknown skill label {skill!r}. Expected one of {SKILL_ORDER}."
        )

    skill_idx = SKILL_TO_INDEX[skill]
    if skill_idx >= skill_dim:
        raise ValueError(
            f"Skill dim {skill_dim} is too small for skill {skill!r} at index {skill_idx}."
        )

    onehot[skill_idx] = 1
    return onehot


def batch_skills_to_onehot_tensor(
    skills: Sequence[Optional[str]],
    skill_dim: int,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if skill_dim == 0:
        return torch.zeros((len(skills), 0), device=device, dtype=dtype)

    if len(skills) == 0:
        return torch.zeros((0, skill_dim), device=device, dtype=dtype)

    return torch.stack(
        [
            skill_to_onehot_tensor(skill, skill_dim, device=device, dtype=dtype)
            for skill in skills
        ],
        dim=0,
    )
