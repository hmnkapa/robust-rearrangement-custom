#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gzip
import lzma
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class FrameStats:
    frame_idx: int
    valid_count: int
    valid_ratio: float
    mean: float
    var: float
    std: float
    min_val: float
    max_val: float


def _load_pickle(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    if path.suffix == ".xz":
        with lzma.open(path, "rb") as f:
            return pickle.load(f)
    with path.open("rb") as f:
        return pickle.load(f)


def _iter_depth_frames(observations: Any, depth_key: str) -> Iterable[np.ndarray]:
    if isinstance(observations, list):
        for i, obs in enumerate(observations):
            if not isinstance(obs, dict):
                raise TypeError(f"observations[{i}] is not a dict")
            if depth_key not in obs:
                raise KeyError(f"Missing key '{depth_key}' in observations[{i}]")
            yield np.asarray(obs[depth_key], dtype=np.float64)
        return

    if isinstance(observations, dict):
        if depth_key not in observations:
            raise KeyError(f"Missing key '{depth_key}' in observations dict")
        arr = np.asarray(observations[depth_key], dtype=np.float64)
        if arr.ndim < 3:
            raise ValueError(
                f"Expected observations['{depth_key}'] shape [T,H,W], got {arr.shape}"
            )
        for i in range(arr.shape[0]):
            yield arr[i]
        return

    raise TypeError("Unsupported observations format; expected list[dict] or dict")


def _resolve_sign_mode(values: np.ndarray, sign_policy: str) -> str:
    if sign_policy in ("as_is", "negate"):
        return sign_policy
    non_pos_ratio = float((values <= 0).mean()) if values.size > 0 else 0.0
    return "negate" if non_pos_ratio > 0.5 else "as_is"


def _depth_distance(frame: np.ndarray, sign_mode: str) -> np.ndarray:
    if sign_mode == "negate":
        return -frame
    return frame


def compute_stats(
    depth_frames: list[np.ndarray],
    sign_policy: str = "auto",
) -> tuple[list[FrameStats], dict[str, float | int | str]]:
    if len(depth_frames) == 0:
        raise ValueError("No depth frames found")

    all_finite_vals = np.concatenate([f[np.isfinite(f)] for f in depth_frames])
    sign_mode = _resolve_sign_mode(all_finite_vals, sign_policy)

    frame_stats: list[FrameStats] = []
    total_valid = 0
    total_sum = 0.0
    total_sum_sq = 0.0
    total_pixels = 0

    for i, frame in enumerate(depth_frames):
        dist = _depth_distance(frame, sign_mode)
        valid = np.isfinite(dist) & (dist > 0)
        vals = dist[valid]
        total_pixels += int(frame.size)

        if vals.size == 0:
            stats = FrameStats(
                frame_idx=i,
                valid_count=0,
                valid_ratio=0.0,
                mean=0.0,
                var=0.0,
                std=0.0,
                min_val=0.0,
                max_val=0.0,
            )
        else:
            mean = float(vals.mean())
            var = float(vals.var())  # exact population variance for this frame
            stats = FrameStats(
                frame_idx=i,
                valid_count=int(vals.size),
                valid_ratio=float(vals.size / frame.size),
                mean=mean,
                var=var,
                std=float(np.sqrt(var)),
                min_val=float(vals.min()),
                max_val=float(vals.max()),
            )
            total_valid += int(vals.size)
            total_sum += float(vals.sum())
            total_sum_sq += float((vals * vals).sum())

        frame_stats.append(stats)

    if total_valid == 0:
        traj_mean = 0.0
        traj_var = 0.0
    else:
        traj_mean = total_sum / total_valid
        traj_var = max(total_sum_sq / total_valid - traj_mean * traj_mean, 0.0)

    summary: dict[str, float | int | str] = {
        "sign_mode": sign_mode,
        "n_frames": len(depth_frames),
        "total_pixels": total_pixels,
        "total_valid_pixels": total_valid,
        "valid_ratio_global": (total_valid / total_pixels) if total_pixels > 0 else 0.0,
        "trajectory_mean": float(traj_mean),
        "trajectory_var": float(traj_var),
        "trajectory_std": float(np.sqrt(traj_var)),
    }
    return frame_stats, summary


def _write_csv(path: Path, rows: list[FrameStats]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame_idx",
                "valid_count",
                "valid_ratio",
                "mean",
                "var",
                "std",
                "min",
                "max",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.frame_idx,
                    r.valid_count,
                    f"{r.valid_ratio:.8f}",
                    f"{r.mean:.8f}",
                    f"{r.var:.8f}",
                    f"{r.std:.8f}",
                    f"{r.min_val:.8f}",
                    f"{r.max_val:.8f}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute exact per-frame and full-trajectory depth stats from a rollout pickle."
    )
    parser.add_argument("pickle_path", type=str, help="Path to .pkl/.pkl.xz/.pkl.gz")
    parser.add_argument(
        "--depth-key",
        type=str,
        default="depth_image2",
        choices=["depth_image1", "depth_image2"],
        help="Which depth stream to analyze",
    )
    parser.add_argument(
        "--sign-policy",
        type=str,
        default="auto",
        choices=["auto", "as_is", "negate"],
        help="How to interpret depth sign: auto for Isaac negative-depth compatibility",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Optional output CSV path for per-frame stats",
    )
    parser.add_argument(
        "--print-frames",
        action="store_true",
        help="Print per-frame stats to stdout",
    )
    args = parser.parse_args()

    path = Path(args.pickle_path)
    data = _load_pickle(path)
    if not isinstance(data, dict) or "observations" not in data:
        raise ValueError("Pickle must contain a dict with key 'observations'")

    frames = list(_iter_depth_frames(data["observations"], args.depth_key))
    per_frame, summary = compute_stats(frames, sign_policy=args.sign_policy)

    print(f"file: {path}")
    print(f"depth_key: {args.depth_key}")
    print(f"sign_mode: {summary['sign_mode']}")
    print(f"n_frames: {summary['n_frames']}")
    print(f"valid_ratio_global: {summary['valid_ratio_global']:.8f}")
    print(f"trajectory_mean: {summary['trajectory_mean']:.8f}")
    print(f"trajectory_var:  {summary['trajectory_var']:.8f}")
    print(f"trajectory_std:  {summary['trajectory_std']:.8f}")

    if args.print_frames:
        print("frame_idx,valid_count,valid_ratio,mean,var,std,min,max")
        for r in per_frame:
            print(
                f"{r.frame_idx},{r.valid_count},{r.valid_ratio:.8f},{r.mean:.8f},"
                f"{r.var:.8f},{r.std:.8f},{r.min_val:.8f},{r.max_val:.8f}"
            )

    if args.out_csv is not None:
        out_path = Path(args.out_csv)
        _write_csv(out_path, per_frame)
        print(f"saved_csv: {out_path}")


if __name__ == "__main__":
    main()
