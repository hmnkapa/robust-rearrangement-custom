#!/usr/bin/env python3
import tempfile
from pathlib import Path

import numpy as np
import torch
import zarr

from src.dataset.normalizer import LinearNormalizer
from src.dataset.zarr import (
    balance_episode_manifest_by_frames,
    build_episode_manifest,
    combine_zarr_episode_subset,
    compute_global_minmax_stats,
    split_episode_manifest,
)


def create_synthetic_dataset(path: Path, episode_lengths, domain: str, offset: int):
    total_frames = int(sum(episode_lengths))
    episode_ends = np.cumsum(np.asarray(episode_lengths, dtype=np.int64))

    root = zarr.open_group(str(path), mode="w")
    root.attrs["domain"] = domain
    root.create_dataset("episode_ends", data=episode_ends, shape=episode_ends.shape)
    root.create_dataset(
        "task",
        data=np.asarray([f"task_{offset}_{idx}" for idx in range(len(episode_lengths))]),
        shape=(len(episode_lengths),),
        dtype="<U32",
    )
    root.create_dataset(
        "success",
        data=np.ones(len(episode_lengths), dtype=np.uint8),
        shape=(len(episode_lengths),),
    )
    root.create_dataset(
        "failure_idx",
        data=np.full(len(episode_lengths), -1, dtype=np.int64),
        shape=(len(episode_lengths),),
    )

    robot_state = (
        np.arange(total_frames * 12, dtype=np.float32).reshape(total_frames, 12) + offset
    )
    action_delta = (
        np.arange(total_frames * 10, dtype=np.float32).reshape(total_frames, 10)
        + offset * 0.1
    )
    action_pos = action_delta + 1.0
    skill = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (total_frames, 1))
    parts_poses = (
        np.arange(total_frames * 14, dtype=np.float32).reshape(total_frames, 14)
        + offset * 0.5
    )

    color_image1 = np.full((total_frames, 2, 2, 3), offset, dtype=np.uint8)
    color_image2 = np.full((total_frames, 2, 2, 3), offset + 1, dtype=np.uint8)
    depth_image1 = np.full((total_frames, 2, 2), offset * 0.01, dtype=np.float32)
    depth_image2 = np.full((total_frames, 2, 2), offset * 0.02, dtype=np.float32)

    root.create_dataset("robot_state", data=robot_state, shape=robot_state.shape)
    root.create_dataset("action/delta", data=action_delta, shape=action_delta.shape)
    root.create_dataset("action/pos", data=action_pos, shape=action_pos.shape)
    root.create_dataset("skill", data=skill, shape=skill.shape)
    root.create_dataset("parts_poses", data=parts_poses, shape=parts_poses.shape)
    root.create_dataset("color_image1", data=color_image1, shape=color_image1.shape)
    root.create_dataset("color_image2", data=color_image2, shape=color_image2.shape)
    root.create_dataset("depth_image1", data=depth_image1, shape=depth_image1.shape)
    root.create_dataset("depth_image2", data=depth_image2, shape=depth_image2.shape)
    root.attrs["normalizer_stats"] = {
        "robot_state": {
            "min": robot_state.min(axis=0).tolist(),
            "max": robot_state.max(axis=0).tolist(),
        },
        "action/delta": {
            "min": action_delta.min(axis=0).tolist(),
            "max": action_delta.max(axis=0).tolist(),
        },
        "action/pos": {
            "min": action_pos.min(axis=0).tolist(),
            "max": action_pos.max(axis=0).tolist(),
        },
        "skill": {
            "min": skill.min(axis=0).tolist(),
            "max": skill.max(axis=0).tolist(),
        },
        "parts_poses": {
            "min": parts_poses.min(axis=0).tolist(),
            "max": parts_poses.max(axis=0).tolist(),
        },
    }


def manual_concat(paths, episode_refs, key):
    arrays = []
    for ref in episode_refs:
        dataset = zarr.open(str(paths[ref.path_idx]), mode="r")
        arrays.append(dataset[key][ref.frame_start : ref.frame_end])
    return np.concatenate(arrays, axis=0)


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        path_a = root / "diffik" / "sim" / "task_a" / "rollout" / "low" / "success.zarr"
        path_b = root / "diffik" / "real" / "task_b" / "teleop" / "low" / "success.zarr"
        path_a.parent.mkdir(parents=True, exist_ok=True)
        path_b.parent.mkdir(parents=True, exist_ok=True)

        create_synthetic_dataset(path_a, [3, 2], "sim", 10)
        create_synthetic_dataset(path_b, [4, 1], "real", 20)

        paths = [path_a, path_b]
        manifest = build_episode_manifest(paths)
        assert len(manifest) == 4
        assert sum(ref.frame_count for ref in manifest) == 10

        train_a, val_a = split_episode_manifest(manifest, test_split=0.25, seed=123)
        train_b, val_b = split_episode_manifest(manifest, test_split=0.25, seed=123)
        assert train_a == train_b
        assert val_a == val_b

        shards = balance_episode_manifest_by_frames(train_a, world_size=2)
        flat_shards = [ref for shard in shards for ref in shard]
        assert sorted(flat_shards, key=lambda ref: (ref.path_idx, ref.episode_idx)) == sorted(
            train_a, key=lambda ref: (ref.path_idx, ref.episode_idx)
        )
        assert len(set((ref.path_idx, ref.episode_idx) for ref in flat_shards)) == len(train_a)

        combined_data, _ = combine_zarr_episode_subset(
            paths, train_a, ["robot_state", "action/delta"]
        )
        expected_robot_state = manual_concat(paths, train_a, "robot_state")
        expected_action_delta = manual_concat(paths, train_a, "action/delta")
        np.testing.assert_allclose(combined_data["robot_state"], expected_robot_state)
        np.testing.assert_allclose(combined_data["action/delta"], expected_action_delta)

        stats = compute_global_minmax_stats(
            paths,
            manifest,
            {"robot_state": "robot_state", "action": "action/delta"},
            device=None,
        )
        reference = LinearNormalizer()
        reference.fit(
            {
                "robot_state": torch.from_numpy(
                    manual_concat(paths, manifest, "robot_state")
                ),
                "action": torch.from_numpy(manual_concat(paths, manifest, "action/delta")),
            }
        )
        np.testing.assert_allclose(
            stats["robot_state"]["min"].numpy(),
            reference.stats["robot_state"]["min"].detach().cpu().numpy(),
        )
        np.testing.assert_allclose(
            stats["robot_state"]["max"].numpy(),
            reference.stats["robot_state"]["max"].detach().cpu().numpy(),
        )
        np.testing.assert_allclose(
            stats["action"]["min"].numpy(),
            reference.stats["action"]["min"].detach().cpu().numpy(),
        )
        np.testing.assert_allclose(
            stats["action"]["max"].numpy(),
            reference.stats["action"]["max"].detach().cpu().numpy(),
        )

    print("DDP dataset sharding validation passed.")


if __name__ == "__main__":
    main()
