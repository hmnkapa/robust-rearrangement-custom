#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def _is_array_like(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _shape_str(value: Any) -> str:
    if _is_array_like(value):
        try:
            return str(tuple(int(x) for x in value.shape))
        except Exception:
            return str(value.shape)
    if isinstance(value, (list, tuple)):
        return f"(len={len(value)})"
    if isinstance(value, dict):
        return f"(dict keys={sorted(str(k) for k in value.keys())})"
    if value is None:
        return "(None)"
    if isinstance(value, str):
        return f"(str len={len(value)})"
    return "()"


@dataclass(frozen=True)
class ObsSummary:
    num_observations: int | None
    elem_shapes: dict[str, str]


def _infer_num_obs_and_elem_shape(value: Any) -> tuple[int | None, str]:
    """Infer (N, element_shape) for a single observation field.

    Handles common layouts:
      - np.ndarray with shape (N, ...)
      - list/tuple of length N where each item is array-like
      - single array-like value without time dimension
    """
    if _is_array_like(value):
        shape = tuple(int(x) for x in value.shape)
        if len(shape) >= 2:
            return shape[0], str(shape[1:])
        if len(shape) == 1:
            return None, str(shape)
        return None, str(shape)

    if isinstance(value, (list, tuple)) and len(value) > 0:
        first = value[0]
        if _is_array_like(first):
            try:
                shape = tuple(int(x) for x in first.shape)
            except Exception:
                shape = first.shape
            return len(value), str(shape)
        return len(value), _shape_str(first)

    return None, _shape_str(value)


def _flatten_dict(d: dict[str, Any], prefix: str = "") -> Iterable[tuple[str, Any]]:
    for key, value in d.items():
        key_str = str(key)
        full_key = f"{prefix}/{key_str}" if prefix else key_str
        if isinstance(value, dict):
            yield from _flatten_dict(value, prefix=full_key)
        else:
            yield full_key, value


def _summarize_observations(obs: Any) -> ObsSummary:
    if isinstance(obs, dict):
        num_obs: int | None = None
        elem_shapes: dict[str, str] = {}
        for flat_key, value in _flatten_dict(obs):
            n, elem_shape = _infer_num_obs_and_elem_shape(value)
            if num_obs is None and n is not None:
                num_obs = n
            elem_shapes[flat_key] = elem_shape
        return ObsSummary(num_observations=num_obs, elem_shapes=elem_shapes)

    if isinstance(obs, (list, tuple)) and len(obs) > 0:
        first = obs[0]
        if isinstance(first, dict):
            # element is dict; shapes refer to one element
            elem_shapes: dict[str, str] = {}
            for flat_key, value in _flatten_dict(first):
                _, elem_shape = _infer_num_obs_and_elem_shape(value)
                elem_shapes[flat_key] = elem_shape
            return ObsSummary(num_observations=len(obs), elem_shapes=elem_shapes)
        return ObsSummary(num_observations=len(obs), elem_shapes={"": _shape_str(first)})

    # Fallback: unknown structure
    return ObsSummary(num_observations=None, elem_shapes={"": _shape_str(obs)})


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        try:
            return pickle.load(f)
        except TypeError:
            f.seek(0)
            return pickle.load(f, encoding="latin1")


def _print_top_field(name: str, value: Any) -> None:
    if _is_array_like(value):
        print(f"{name}: shape {_shape_str(value)}")
        return
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            print(f"{name}: shape (0,)")
            return

        first = value[0]
        if _is_array_like(first):
            try:
                elem_shape = tuple(int(x) for x in first.shape)
            except Exception:
                elem_shape = first.shape
            print(f"{name}: shape {(len(value),) + tuple(elem_shape)}")
            return

        if isinstance(first, (int, float, bool)):
            print(f"{name}: shape ({len(value)},)")
            return

        # Fallback: length only.
        print(f"{name}: len {len(value)}")
        return
    print(f"{name}: {value} (type: {type(value)})")


def _print_camera_info_summary(camera_info: Any) -> None:
    if not isinstance(camera_info, dict):
        _print_top_field("camera_info", camera_info)
        return

    print("camera_info:")

    front = camera_info.get("front_camera")
    if isinstance(front, dict):
        print(f"  front_camera: dict keys={sorted(str(k) for k in front.keys())}")
    else:
        print(f"  front_camera: {front}")

    wrist = camera_info.get("wrist_camera")
    if isinstance(wrist, (list, tuple)):
        print(f"  wrist_camera: len {len(wrist)}")
        first_valid = next((item for item in wrist if isinstance(item, dict)), None)
        if first_valid is not None:
            print(
                f"    first_valid_keys: {sorted(str(k) for k in first_valid.keys())}"
            )
    else:
        print(f"  wrist_camera: {wrist}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a pickle and print a compact structure summary.")
    parser.add_argument("path", type=str, help="Path to .pkl/.pickle file")
    args = parser.parse_args()

    path = Path(args.path)
    data = _load_pickle(path)

    if not isinstance(data, dict):
        print(f"root: {type(data)}")
        if _is_array_like(data):
            print(f"  shape {_shape_str(data)}")
        return

    # Match the sample layout as closely as possible.
    if "observations" in data:
        obs_summary = _summarize_observations(data["observations"])
        if obs_summary.num_observations is not None:
            print(f"observations: number of observations {obs_summary.num_observations}")
        else:
            print("observations: number of observations (unknown)")

        for key in sorted(k for k in obs_summary.elem_shapes.keys() if k):
            print(f"  {key}: shape {obs_summary.elem_shapes[key]}")
        if "" in obs_summary.elem_shapes:
            print(f"  (element): shape {obs_summary.elem_shapes['']}")

    for k in ["actions", "rewards"]:
        if k in data:
            _print_top_field(k, data[k])

    if "camera_info" in data:
        _print_camera_info_summary(data["camera_info"])

    for k in ["success", "task", "action_type"]:
        if k in data:
            _print_top_field(k, data[k])

    # Print any remaining top-level fields not covered above.
    covered = {
        "observations",
        "actions",
        "rewards",
        "camera_info",
        "success",
        "task",
        "action_type",
    }
    extra_keys = [k for k in data.keys() if k not in covered]
    for k in sorted(extra_keys):
        _print_top_field(str(k), data[k])


if __name__ == "__main__":
    main()
