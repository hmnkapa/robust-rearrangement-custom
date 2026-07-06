from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


DESK_POSE_COUNT = 6
DESK_NUM_PAIRS = 4
POSE_DIM = 7
REQUIRED_ROBOT_STATE_KEYS = (
    "joint_positions",
    "gripper_finger_1_pos",
    "gripper_finger_2_pos",
)


def resolve_initial_state_dir(path: str, original_cwd: str | Path) -> Path:
    state_dir = Path(path).expanduser()
    if state_dir.is_absolute():
        return state_dir
    return Path(original_cwd) / state_dir


def load_desk_initial_states_from_dir(state_dir: Path) -> list[dict[str, Any]]:
    if not state_dir.exists():
        raise FileNotFoundError(f"Initial state dir does not exist: {state_dir}")
    if not state_dir.is_dir():
        raise NotADirectoryError(f"Initial state path is not a directory: {state_dir}")

    state_paths = sorted(state_dir.glob("*.pkl"))
    if not state_paths:
        raise ValueError(f"No .pkl initial states found in {state_dir}")

    states = []
    for path in state_paths:
        with path.open("rb") as f:
            state = pickle.load(f)
        validate_two_leg_desk_initial_state(state, source=path)
        states.append(state)
    return states


def sample_initial_states(
    states: list[dict[str, Any]], num_envs: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    if not states:
        raise ValueError("Cannot sample from an empty initial-state list")
    indices = rng.integers(0, len(states), size=num_envs)
    return [states[int(idx)] for idx in indices]


def validate_two_leg_desk_initial_state(
    state: dict[str, Any], source: Path | None = None
) -> None:
    label = f" in {source}" if source is not None else ""
    if not isinstance(state, dict):
        raise ValueError(f"Initial state{label} must be a dict")

    parts_poses = np.asarray(state.get("parts_poses"))
    expected_parts_shape = (DESK_POSE_COUNT * POSE_DIM,)
    if parts_poses.shape != expected_parts_shape:
        raise ValueError(
            f"Initial state{label} parts_poses shape {parts_poses.shape} "
            f"!= {expected_parts_shape}"
        )

    robot_state = state.get("robot_state")
    if not isinstance(robot_state, dict):
        raise ValueError(f"Initial state{label} must contain robot_state dict")
    missing_keys = [key for key in REQUIRED_ROBOT_STATE_KEYS if key not in robot_state]
    if missing_keys:
        raise ValueError(
            f"Initial state{label} missing robot_state keys: {missing_keys}"
        )

    metadata = state.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Initial state{label} must contain metadata dict")

    inserted_mask = _desk_mask(metadata, "desk_inserted_mask", label)
    success_mask = _desk_mask(metadata, "desk_success_mask", label)
    if not np.array_equal(inserted_mask, success_mask):
        raise ValueError(
            f"Initial state{label} desk_inserted_mask and desk_success_mask differ"
        )
    if int(success_mask.sum()) != 2:
        raise ValueError(
            f"Initial state{label} must mark exactly two completed desk legs"
        )

    current_pair_idx = metadata.get("desk_current_pair_idx")
    if current_pair_idx is None:
        raise ValueError(f"Initial state{label} missing desk_current_pair_idx")
    current_pair_idx = int(current_pair_idx)
    if not 0 <= current_pair_idx < DESK_NUM_PAIRS:
        raise ValueError(
            f"Initial state{label} desk_current_pair_idx {current_pair_idx} "
            f"outside [0, {DESK_NUM_PAIRS})"
        )


def _desk_mask(metadata: dict[str, Any], key: str, label: str) -> np.ndarray:
    if key not in metadata:
        raise ValueError(f"Initial state{label} missing metadata.{key}")
    mask = np.asarray(metadata[key], dtype=bool)
    if mask.shape != (DESK_NUM_PAIRS,):
        raise ValueError(
            f"Initial state{label} metadata.{key} shape {mask.shape} "
            f"!= ({DESK_NUM_PAIRS},)"
        )
    return mask
