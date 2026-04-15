from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import zarr
from tqdm import tqdm
from ipdb import set_trace as bp

from src.common.files import get_processed_paths
from src.dataset.base import EpisodeRef

NORMALIZER_STATS_ATTR = "normalizer_stats"


class ZarrSubsetView:
    def __init__(self, zarr_group, include_keys):
        """
        Create a view-like object for a Zarr group, excluding specified keys.
        :param zarr_group: The original Zarr group.
        :param exclude_keys: A set or list of keys to exclude.
        """
        self.zarr_group = zarr_group
        self.include_keys = set(include_keys)

    def __getitem__(self, key):
        return self.zarr_group[key]

    def observation_keys(self):
        """
        Return keys not excluded.
        """
        return [key for key in self.zarr_group.keys() if key in self.include_keys]

    def items(self):
        """
        Return items not excluded.
        """
        return [(key, self.zarr_group[key]) for key in self.observation_keys()]


def dataset_tuple(path: Path) -> Tuple[str, str, str, str]:
    """
    Extract the task, source, randomness, and outcome from a zarr path.
    """
    return path.with_name(path.stem).parts[-4:]


def _resolve_max_episodes(path: Path, max_episodes=None, max_ep_cnt=None):
    if max_ep_cnt is not None:
        f, s, r, o = dataset_tuple(path)
        return max_ep_cnt.get(f, {}).get(s, {}).get(r, {}).get(o, max_episodes)
    return max_episodes


def _coerce_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return _coerce_scalar(value.tolist())
    if isinstance(value, list):
        if len(value) == 1:
            return _coerce_scalar(value[0])
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _feature_min_max(array: np.ndarray):
    if array.ndim == 1:
        return np.min(array, axis=0), np.max(array, axis=0)
    return np.min(array, axis=0), np.max(array, axis=0)


def _deserialize_normalizer_stats(raw_stats) -> Dict[str, Dict[str, np.ndarray]]:
    if raw_stats is None:
        return {}

    return {
        key: {
            "min": np.asarray(value["min"], dtype=np.float32),
            "max": np.asarray(value["max"], dtype=np.float32),
        }
        for key, value in raw_stats.items()
    }


def _init_feature_stats(first_dataset, stats_key_map: Dict[str, str]):
    local_stats = {}
    for stat_key, zarr_key in stats_key_map.items():
        feature_shape = first_dataset[zarr_key].shape[1:]
        local_stats[stat_key] = {
            "min": np.full(feature_shape, np.inf, dtype=np.float64),
            "max": np.full(feature_shape, -np.inf, dtype=np.float64),
        }
    return local_stats


def _update_feature_stats(
    local_stats: Dict[str, Dict[str, np.ndarray]],
    stat_key: str,
    min_value: np.ndarray,
    max_value: np.ndarray,
):
    local_stats[stat_key]["min"] = np.minimum(local_stats[stat_key]["min"], min_value)
    local_stats[stat_key]["max"] = np.maximum(local_stats[stat_key]["max"], max_value)


def _build_read_blocks(items, *, max_block_frames: Optional[int] = None):
    if not items:
        return []

    ordered_items = sorted(
        items,
        key=lambda item: (
            item["ref"].path_idx,
            item["ref"].frame_start,
            item["ref"].episode_idx,
        ),
    )

    blocks = []
    current_block = None

    for item in ordered_items:
        ref = item["ref"]
        can_extend = False
        if current_block is not None:
            new_frame_count = ref.frame_end - current_block["frame_start"]
            can_extend = (
                ref.path_idx == current_block["path_idx"]
                and ref.frame_start == current_block["frame_end"]
                and (
                    max_block_frames is None or new_frame_count <= max_block_frames
                )
            )

        if not can_extend:
            current_block = {
                "path_idx": ref.path_idx,
                "frame_start": ref.frame_start,
                "frame_end": ref.frame_end,
                "items": [item],
            }
            blocks.append(current_block)
            continue

        current_block["frame_end"] = ref.frame_end
        current_block["items"].append(item)

    return blocks


def _scatter_block_segment(output_array, items, segment_start, segment_end, segment_array):
    for item in items:
        ref = item["ref"]
        overlap_start = max(segment_start, ref.frame_start)
        overlap_end = min(segment_end, ref.frame_end)
        if overlap_start >= overlap_end:
            continue

        source_start = overlap_start - segment_start
        source_end = overlap_end - segment_start
        dest_start = item["output_start"] + (overlap_start - ref.frame_start)
        dest_end = dest_start + (overlap_end - overlap_start)
        output_array[dest_start:dest_end] = segment_array[source_start:source_end]


def _balance_blocks_by_frames(blocks: List[dict], world_size: int) -> List[List[dict]]:
    if world_size <= 1:
        return [blocks]

    shards = [[] for _ in range(world_size)]
    shard_loads = [0 for _ in range(world_size)]
    ordered_blocks = sorted(
        blocks,
        key=lambda block: (
            -(block["frame_end"] - block["frame_start"]),
            block["path_idx"],
            block["frame_start"],
        ),
    )

    for block in ordered_blocks:
        shard_idx = min(range(world_size), key=lambda idx: (shard_loads[idx], idx))
        shards[shard_idx].append(block)
        shard_loads[shard_idx] += block["frame_end"] - block["frame_start"]

    for shard in shards:
        shard.sort(key=lambda block: (block["path_idx"], block["frame_start"]))
    return shards


def _block_frame_count(block: dict) -> int:
    return int(block["frame_end"] - block["frame_start"])


def _estimate_transfer_gib(dataset, keys, total_frames: int) -> float:
    if total_frames <= 0:
        return 0.0

    bytes_per_frame = 0
    for key in keys:
        array = dataset[key]
        trailing_shape = array.shape[1:]
        bytes_per_element = np.dtype(array.dtype).itemsize
        bytes_per_frame += int(np.prod(trailing_shape, dtype=np.int64)) * bytes_per_element

    return (bytes_per_frame * total_frames) / float(1024**3)


def _can_use_precomputed_stats(dataset, episode_refs: List[EpisodeRef], stats_key_map):
    if len(episode_refs) == 0:
        return {}

    raw_stats = dataset.attrs.asdict().get(NORMALIZER_STATS_ATTR)
    if raw_stats is None:
        return None

    total_episodes = len(dataset["episode_ends"])
    episode_indices = sorted(ref.episode_idx for ref in episode_refs)
    if len(episode_indices) != total_episodes:
        return None
    if episode_indices != list(range(total_episodes)):
        return None

    stored_stats = _deserialize_normalizer_stats(raw_stats)
    required_keys = set(stats_key_map.values())
    if not required_keys.issubset(stored_stats.keys()):
        return None

    return stored_stats


def build_episode_manifest(
    zarr_paths: Union[List[Path], Path],
    max_episodes=None,
    max_ep_cnt=None,
) -> List[EpisodeRef]:
    if not isinstance(zarr_paths, list):
        zarr_paths = [zarr_paths]

    manifest: List[EpisodeRef] = []
    for path_idx, path in enumerate(zarr_paths):
        dataset = zarr.open(path, mode="r")
        max_ep = _resolve_max_episodes(
            path, max_episodes=max_episodes, max_ep_cnt=max_ep_cnt
        )
        episode_ends = dataset["episode_ends"][:max_ep]
        task = dataset.get("task", dataset.get("furniture"))
        success = dataset["success"][:max_ep]
        domain = str(dataset.attrs["domain"])

        start_idx = 0
        for episode_idx, end_idx in enumerate(episode_ends):
            end_idx = int(end_idx)
            manifest.append(
                EpisodeRef(
                    path_idx=path_idx,
                    episode_idx=episode_idx,
                    frame_start=start_idx,
                    frame_end=end_idx,
                    frame_count=end_idx - start_idx,
                    task=str(_coerce_scalar(task[episode_idx])),
                    success=int(_coerce_scalar(success[episode_idx])),
                    domain=domain,
                )
            )
            start_idx = end_idx

    return manifest


def split_episode_manifest(
    manifest: List[EpisodeRef], test_split: float, seed: int
) -> Tuple[List[EpisodeRef], List[EpisodeRef]]:
    if not 0.0 <= test_split <= 1.0:
        raise ValueError(f"test_split must be in [0, 1], got {test_split}.")

    if len(manifest) == 0:
        return [], []

    rng = np.random.default_rng(int(seed))
    indices = rng.permutation(len(manifest))
    train_episode_count = int(len(manifest) * (1 - test_split))

    train_indices = indices[:train_episode_count]
    val_indices = indices[train_episode_count:]

    train_manifest = [manifest[idx] for idx in train_indices]
    val_manifest = [manifest[idx] for idx in val_indices]
    return train_manifest, val_manifest


def balance_episode_manifest_by_frames(
    manifest: List[EpisodeRef], world_size: int
) -> List[List[EpisodeRef]]:
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}.")

    shards = [[] for _ in range(world_size)]
    shard_loads = [0 for _ in range(world_size)]

    ordered_manifest = sorted(
        manifest,
        key=lambda ref: (-ref.frame_count, ref.path_idx, ref.episode_idx),
    )
    for ref in ordered_manifest:
        shard_idx = min(range(world_size), key=lambda idx: (shard_loads[idx], idx))
        shards[shard_idx].append(ref)
        shard_loads[shard_idx] += ref.frame_count

    return shards


def combine_zarr_episode_subset(
    zarr_paths: Union[List[Path], Path],
    episode_refs: List[EpisodeRef],
    keys,
    progress_desc: Optional[str] = None,
    progress_position: int = 0,
    progress_disable: bool = False,
) -> Tuple[dict, dict]:
    if not isinstance(zarr_paths, list):
        zarr_paths = [zarr_paths]

    opened = {}
    metadata = {}
    domain_idx = dict(sim=0, real=1)
    total_frames = sum(ref.frame_count for ref in episode_refs)
    total_episodes = len(episode_refs)

    if zarr_paths:
        first_dataset = zarr.open(zarr_paths[0], mode="r")
    else:
        raise ValueError("combine_zarr_episode_subset requires at least one zarr path.")

    combined_data = {
        "episode_ends": np.zeros(total_episodes, dtype=np.int64),
        "task": [],
        "success": np.zeros(total_episodes, dtype=np.uint8),
        "domain": np.zeros(total_episodes, dtype=np.uint8),
        "zarr_idx": np.zeros(total_frames, dtype=np.int64),
        "within_zarr_idx": np.zeros(total_frames, dtype=np.int64),
        "failure_idx": np.full(total_episodes, -1, dtype=np.int64),
    }
    for key in keys:
        combined_data[key] = np.zeros(
            (total_frames,) + first_dataset[key].shape[1:], dtype=first_dataset[key].dtype
        )

    per_path_episode_counts = defaultdict(int)
    per_path_frame_counts = defaultdict(int)

    frame_cursor = 0
    output_items = []
    image_chunk_size = min(int(first_dataset.attrs.asdict().get("chunksize", 256)), 256)

    for episode_cursor, ref in enumerate(episode_refs):
        path = zarr_paths[ref.path_idx]
        dataset = opened.get(ref.path_idx)
        if dataset is None:
            dataset = zarr.open(path, mode="r")
            opened[ref.path_idx] = dataset

        frame_start = ref.frame_start
        frame_end = ref.frame_end
        frame_count = ref.frame_count

        combined_data["episode_ends"][episode_cursor] = frame_cursor + frame_count
        combined_data["task"].append(ref.task)
        combined_data["success"][episode_cursor] = ref.success
        combined_data["domain"][episode_cursor] = domain_idx[ref.domain]
        combined_data["zarr_idx"][frame_cursor : frame_cursor + frame_count] = ref.path_idx
        combined_data["within_zarr_idx"][
            frame_cursor : frame_cursor + frame_count
        ] = np.arange(frame_start, frame_end)

        failure_idx = dataset.get("failure_idx")
        if failure_idx is not None and len(failure_idx) > ref.episode_idx:
            combined_data["failure_idx"][episode_cursor] = failure_idx[ref.episode_idx]

        metadata_key = str(dataset_tuple(path))
        per_path_episode_counts[metadata_key] += 1
        per_path_frame_counts[metadata_key] += frame_count
        metadata[metadata_key] = {
            "n_episodes_used": per_path_episode_counts[metadata_key],
            "n_frames_used": per_path_frame_counts[metadata_key],
            "attrs": dataset.attrs.asdict(),
        }

        output_items.append(
            {
                "ref": ref,
                "episode_cursor": episode_cursor,
                "output_start": frame_cursor,
                "output_end": frame_cursor + frame_count,
            }
        )
        frame_cursor += frame_count

    read_blocks = _build_read_blocks(output_items)
    total_block_frames = sum(_block_frame_count(block) for block in read_blocks)
    estimated_gib = _estimate_transfer_gib(first_dataset, keys, total_block_frames)
    load_desc = progress_desc or "Loading shard"
    if estimated_gib > 0:
        load_desc = f"{load_desc} ({estimated_gib:.2f} GiB)"

    block_iterator = tqdm(
        desc=load_desc,
        total=total_block_frames,
        position=progress_position,
        leave=False,
        disable=progress_disable,
        unit="frame",
    )
    for block in read_blocks:
        dataset = opened[block["path_idx"]]
        block_start = block["frame_start"]
        block_end = block["frame_end"]
        block_frames = _block_frame_count(block)

        for key in keys:
            if "image" in key:
                for segment_start in range(block_start, block_end, image_chunk_size):
                    segment_end = min(segment_start + image_chunk_size, block_end)
                    segment_array = dataset[key][segment_start:segment_end]
                    _scatter_block_segment(
                        combined_data[key],
                        block["items"],
                        segment_start,
                        segment_end,
                        segment_array,
                    )
                continue

            segment_array = dataset[key][block_start:block_end]
            _scatter_block_segment(
                combined_data[key],
                block["items"],
                block_start,
                block_end,
                segment_array,
            )
        block_iterator.update(block_frames)
    block_iterator.close()

    return combined_data, metadata


def compute_global_minmax_stats(
    zarr_paths: Union[List[Path], Path],
    episode_refs: List[EpisodeRef],
    stats_key_map: Dict[str, str],
    device: Union[torch.device, None] = None,
    progress_desc: Optional[str] = None,
    progress_position: int = 0,
    progress_disable: bool = False,
) -> Dict[str, Dict[str, torch.Tensor]]:
    if not isinstance(zarr_paths, list):
        zarr_paths = [zarr_paths]

    if len(zarr_paths) == 0:
        raise ValueError("compute_global_minmax_stats requires at least one zarr path.")

    first_dataset = zarr.open(zarr_paths[0], mode="r")
    local_stats = _init_feature_stats(first_dataset, stats_key_map)
    opened = {}
    refs_by_path = defaultdict(list)
    for ref in episode_refs:
        refs_by_path[ref.path_idx].append(ref)

    scan_items = []
    for path_idx, path_refs in refs_by_path.items():
        dataset = zarr.open(zarr_paths[path_idx], mode="r")
        opened[path_idx] = dataset

        precomputed_stats = _can_use_precomputed_stats(dataset, path_refs, stats_key_map)
        if precomputed_stats is not None:
            for stat_key, zarr_key in stats_key_map.items():
                stored_key_stats = precomputed_stats[zarr_key]
                _update_feature_stats(
                    local_stats,
                    stat_key,
                    stored_key_stats["min"],
                    stored_key_stats["max"],
                )
            continue

        for ref in path_refs:
            scan_items.append({"ref": ref})

    if scan_items:
        scan_blocks = _build_read_blocks(scan_items)
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
        local_blocks = _balance_blocks_by_frames(scan_blocks, world_size)[rank]

        total_block_frames = sum(_block_frame_count(block) for block in local_blocks)
        estimated_gib = _estimate_transfer_gib(
            first_dataset,
            list(stats_key_map.values()),
            total_block_frames,
        )
        minmax_desc = progress_desc or "Computing min/max"
        if estimated_gib > 0:
            minmax_desc = f"{minmax_desc} ({estimated_gib:.2f} GiB)"

        block_iterator = tqdm(
            desc=minmax_desc,
            total=total_block_frames,
            position=progress_position,
            leave=False,
            disable=progress_disable,
            unit="frame",
        )
        for block in local_blocks:
            dataset = opened[block["path_idx"]]
            block_start = block["frame_start"]
            block_end = block["frame_end"]
            block_frames = _block_frame_count(block)
            for stat_key, zarr_key in stats_key_map.items():
                array = dataset[zarr_key][block_start:block_end]
                local_min, local_max = _feature_min_max(array)
                _update_feature_stats(local_stats, stat_key, local_min, local_max)
            block_iterator.update(block_frames)
        block_iterator.close()

    reduced_stats: Dict[str, Dict[str, torch.Tensor]] = {}
    for stat_key, key_stats in local_stats.items():
        min_tensor = torch.as_tensor(key_stats["min"], dtype=torch.float64, device=device)
        max_tensor = torch.as_tensor(key_stats["max"], dtype=torch.float64, device=device)

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(min_tensor, op=dist.ReduceOp.MIN)
            dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)

        reduced_stats[stat_key] = {
            "min": min_tensor.cpu().to(dtype=torch.float32),
            "max": max_tensor.cpu().to(dtype=torch.float32),
        }

    return reduced_stats


def combine_zarr_datasets(
    zarr_paths: Union[List[Path], Path],
    keys,
    max_episodes=None,
    max_ep_cnt=None,
) -> Tuple[dict, dict]:
    """
    Combine multiple zarr datasets into a single dataset.

    This function assume some keys are always present:
    - episode_ends: The end index of each episode.
    - task:         The task name for each episode.
    - success:      Whether the episode was successful.

    These are all of the same length, i.e., the number of episodes.
    """

    if not isinstance(zarr_paths, list):
        zarr_paths = [zarr_paths]

    last_episode_end = 0
    n_episodes = 0
    batch_size = 1000
    total_frames = 0
    total_episodes = 0

    metadata = {}

    domain_idx = dict(sim=0, real=1)

    # First pass to calculate total shapes
    for path in zarr_paths:
        # [F]urniture, [S]ource, [R]andomness, [O]utcome
        f, s, r, o = dataset_tuple(path)
        dataset = zarr.open(path, mode="r")

        if max_ep_cnt is not None:
            max_ep = max_ep_cnt.get(f, {}).get(s, {}).get(r, {}).get(o, max_episodes)
        else:
            max_ep = max_episodes

        n_frames_in_dataset = dataset["episode_ends"][:max_ep][-1]
        n_ep_in_dataset = len(dataset["episode_ends"][:max_ep])

        # Add the metadata
        metadata[str(dataset_tuple(path))] = {
            "n_episodes_used": n_ep_in_dataset,
            "n_frames_used": n_frames_in_dataset,
            "attrs": dataset.attrs.asdict(),
        }

        # Add the counts to the totals
        total_frames += n_frames_in_dataset
        total_episodes += n_ep_in_dataset

    combined_data = {
        "episode_ends": np.zeros(total_episodes, dtype=np.int64),
        "task": [],
        "success": np.zeros(total_episodes, dtype=np.uint8),
        # Domain is 0 for sim, 1 for real
        "domain": np.zeros(total_episodes, dtype=np.uint8),
        # Large sharded datasets and long trajectories can easily exceed uint8.
        "zarr_idx": np.zeros(total_frames, dtype=np.int64),
        "within_zarr_idx": np.zeros(total_frames, dtype=np.int64),
    }
    for key in keys:
        combined_data[key] = np.zeros(
            (total_frames,) + dataset[key].shape[1:], dtype=dataset[key].dtype
        )

    for ii, path in enumerate(tqdm(zarr_paths, desc="Loading zarr files")):
        dataset = zarr.open(path, mode="r")
        # Get the max_episodes for this dataset
        max_episodes = metadata[str(dataset_tuple(path))]["n_episodes_used"]
        end_idxs = dataset["episode_ends"][:max_episodes]

        # Add the frame-based data
        for key in tqdm(keys, desc="Loading data", position=1, leave=False):

            # For the image data, we load in batches
            if key.startswith("color_image"):
                for i in tqdm(
                    range(0, end_idxs[-1], batch_size),
                    desc=f"Loading batches for {key}",
                    leave=False,
                    position=2,
                ):
                    end = min(i + batch_size, end_idxs[-1])
                    batch = dataset[key][i:end]
                    combined_data[key][
                        last_episode_end + i : last_episode_end + end
                    ] = batch

            # For the other data, we can load it all at once
            else:
                combined_data[key][
                    last_episode_end : last_episode_end + end_idxs[-1]
                ] = dataset[key][: end_idxs[-1]]

        # Add the episode-based data
        combined_data["episode_ends"][n_episodes : n_episodes + len(end_idxs)] = (
            end_idxs + last_episode_end
        )
        task = dataset.get("task", dataset.get("furniture"))
        combined_data["task"].extend(task[:max_episodes])
        combined_data["success"][n_episodes : n_episodes + len(end_idxs)] = dataset[
            "success"
        ][:max_episodes]

        combined_data["failure_idx"] = dataset.get(
            "failure_idx", np.full_like(end_idxs, -1)
        )
        combined_data["domain"][n_episodes : n_episodes + len(end_idxs)] = domain_idx[
            dataset.attrs["domain"][:max_episodes]
        ]

        combined_data["zarr_idx"][
            last_episode_end : last_episode_end + end_idxs[-1]
        ] = ii
        combined_data["within_zarr_idx"][
            last_episode_end : last_episode_end + end_idxs[-1]
        ] = np.arange(0, end_idxs[-1])

        # Upddate the counters
        last_episode_end += end_idxs[-1]
        n_episodes += len(end_idxs)

    return combined_data, metadata


if __name__ == "__main__":
    zarr_paths = get_processed_paths(
        environment="sim",
        task=None,
        demo_source=["scripted", "teleop"],
        randomness=None,
        demo_outcome="success",
    )
    print(len(zarr_paths))

    keys = [
        "color_image1",
        "color_image2",
        "robot_state",
        "action/delta",
    ]

    combined_data = combine_zarr_datasets(zarr_paths, keys, max_episodes=None)

    print(
        combined_data["robot_state"].shape,
        combined_data["color_image1"].shape,
        combined_data["episode_ends"].shape,
        combined_data["episode_ends"][-1],
    )
