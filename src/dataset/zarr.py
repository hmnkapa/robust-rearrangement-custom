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


def _group_output_items_by_path(output_items: List[dict]) -> Dict[int, List[dict]]:
    grouped_items = defaultdict(list)
    for item in output_items:
        grouped_items[item["ref"].path_idx].append(item)

    for path_idx in grouped_items:
        grouped_items[path_idx].sort(
            key=lambda item: (item["ref"].frame_start, item["ref"].episode_idx)
        )
    return grouped_items


def _dataset_chunk_frames(dataset, key: str, fallback_frames: int = 1000) -> int:
    chunks = getattr(dataset[key], "chunks", None)
    if chunks is None or len(chunks) == 0 or chunks[0] is None:
        return int(fallback_frames)
    return int(chunks[0])


def _scan_chunk_count(scan_end: int, chunk_frames: int) -> int:
    if scan_end <= 0:
        return 0
    return (int(scan_end) + int(chunk_frames) - 1) // int(chunk_frames)


def _copy_chunk_intersections(
    output_array,
    path_items: List[dict],
    chunk_start: int,
    chunk_end: int,
    chunk_array,
    item_idx: int,
) -> int:
    while item_idx < len(path_items) and path_items[item_idx]["ref"].frame_end <= chunk_start:
        item_idx += 1

    scan_idx = item_idx
    while scan_idx < len(path_items):
        item = path_items[scan_idx]
        ref = item["ref"]
        if ref.frame_start >= chunk_end:
            break

        overlap_start = max(chunk_start, ref.frame_start)
        overlap_end = min(chunk_end, ref.frame_end)
        if overlap_start < overlap_end:
            source_start = overlap_start - chunk_start
            source_end = overlap_end - chunk_start
            dest_start = item["output_start"] + (overlap_start - ref.frame_start)
            dest_end = dest_start + (overlap_end - overlap_start)
            output_array[dest_start:dest_end] = chunk_array[source_start:source_end]

        scan_idx += 1

    while item_idx < len(path_items) and path_items[item_idx]["ref"].frame_end <= chunk_end:
        item_idx += 1

    return item_idx


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


def _split_blocks_by_frame_window(blocks: List[dict], window_frames: int) -> List[dict]:
    if window_frames <= 0:
        raise ValueError(f"window_frames must be positive, got {window_frames}.")

    split_blocks = []
    for block in blocks:
        block_start = block["frame_start"]
        block_end = block["frame_end"]
        for window_start in range(block_start, block_end, window_frames):
            window_end = min(window_start + window_frames, block_end)
            split_blocks.append(
                {
                    "path_idx": block["path_idx"],
                    "frame_start": window_start,
                    "frame_end": window_end,
                    "items": [],
                }
            )
    return split_blocks


def _block_frame_count(block: dict) -> int:
    return int(block["frame_end"] - block["frame_start"])


def _key_bytes_per_frame(dataset, key: str) -> int:
    array = dataset[key]
    trailing_shape = array.shape[1:]
    bytes_per_element = np.dtype(array.dtype).itemsize
    return int(np.prod(trailing_shape, dtype=np.int64)) * bytes_per_element


def _estimate_transfer_bytes(dataset, keys, total_frames: int) -> int:
    if total_frames <= 0:
        return 0

    bytes_per_frame = sum(_key_bytes_per_frame(dataset, key) for key in keys)
    return int(bytes_per_frame * total_frames)


def _format_gib(num_bytes: int) -> str:
    return f"{num_bytes / float(1024**3):.2f} GiB"


def _debug_log(message: str, *, enabled: bool = True):
    if enabled:
        tqdm.write(message)


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
    items_by_path = _group_output_items_by_path(output_items)
    load_desc = progress_desc or "Loading shard"
    total_selected_frames = total_frames
    total_scan_frames = sum(
        path_items[-1]["ref"].frame_end
        for path_items in items_by_path.values()
        if path_items
    )
    total_scan_bytes = 0
    for key in keys:
        total_scan_bytes += sum(
            path_items[-1]["ref"].frame_end * _key_bytes_per_frame(opened[path_idx], key)
            for path_idx, path_items in items_by_path.items()
            if path_items
        )
    _debug_log(
        f"{load_desc}: {len(episode_refs)} episodes, {len(keys)} datasets, "
        f"{len(read_blocks)} selected blocks, selected_frames={total_selected_frames}, "
        f"scan_frames={total_scan_frames}, est_read={_format_gib(total_scan_bytes)}",
        enabled=not progress_disable,
    )

    dataset_iterator = tqdm(
        desc=f"{load_desc}: datasets",
        total=len(keys),
        position=progress_position,
        leave=False,
        disable=progress_disable,
        unit="dataset",
    )
    per_key_position = progress_position + 1
    for key in keys:
        key_scan_plan = []
        key_total_scan_frames = 0
        key_total_chunks = 0
        key_total_bytes = sum(
            path_items[-1]["ref"].frame_end * _key_bytes_per_frame(opened[path_idx], key)
            for path_idx, path_items in items_by_path.items()
            if path_items
        )
        for path_idx, path_items in items_by_path.items():
            if not path_items:
                continue

            dataset = opened[path_idx]
            chunk_frames = _dataset_chunk_frames(
                dataset,
                key,
                fallback_frames=int(dataset.attrs.asdict().get("chunksize", 1000)),
            )
            scan_end = path_items[-1]["ref"].frame_end
            scan_chunks = _scan_chunk_count(scan_end, chunk_frames)
            key_total_scan_frames += scan_end
            key_total_chunks += scan_chunks
            key_scan_plan.append((path_idx, scan_end, chunk_frames))

        _debug_log(
            f"{load_desc}: {key} scans {key_total_chunks} source chunks, "
            f"scan_frames={key_total_scan_frames}, selected_frames={total_selected_frames}, "
            f"est_read={_format_gib(key_total_bytes)}",
            enabled=not progress_disable,
        )

        key_desc = f"{load_desc}: batches for {key}"
        if key_total_bytes > 0:
            key_desc = f"{key_desc} ({_format_gib(key_total_bytes)})"

        key_iterator = tqdm(
            desc=key_desc,
            total=key_total_chunks,
            position=per_key_position,
            leave=False,
            disable=progress_disable,
            unit="chunk",
        )

        for path_idx, scan_end, chunk_frames in key_scan_plan:
            path_items = items_by_path[path_idx]
            dataset = opened[path_idx]
            item_idx = 0

            for chunk_start in range(0, scan_end, chunk_frames):
                chunk_end = min(chunk_start + chunk_frames, scan_end)
                chunk_array = dataset[key][chunk_start:chunk_end]
                item_idx = _copy_chunk_intersections(
                    combined_data[key],
                    path_items,
                    chunk_start,
                    chunk_end,
                    chunk_array,
                    item_idx,
                )
                key_iterator.update(1)

        key_iterator.close()
        dataset_iterator.update(1)
    dataset_iterator.close()

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
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    for ref in episode_refs:
        refs_by_path[ref.path_idx].append(ref)

    scan_items = []
    precomputed_paths = []
    fallback_paths = []
    for path_idx, path_refs in refs_by_path.items():
        dataset = zarr.open(zarr_paths[path_idx], mode="r")
        opened[path_idx] = dataset

        precomputed_stats = _can_use_precomputed_stats(dataset, path_refs, stats_key_map)
        if precomputed_stats is not None:
            precomputed_paths.append(str(zarr_paths[path_idx]))
            for stat_key, zarr_key in stats_key_map.items():
                stored_key_stats = precomputed_stats[zarr_key]
                _update_feature_stats(
                    local_stats,
                    stat_key,
                    stored_key_stats["min"],
                    stored_key_stats["max"],
                )
            continue

        fallback_paths.append(str(zarr_paths[path_idx]))
        for ref in path_refs:
            scan_items.append({"ref": ref})

    if scan_items:
        scan_blocks = _build_read_blocks(scan_items)
        scan_window_frames = int(first_dataset.attrs.asdict().get("chunksize", 1000))
        scan_windows = _split_blocks_by_frame_window(scan_blocks, scan_window_frames)
        local_blocks = _balance_blocks_by_frames(scan_windows, world_size)[rank]

        total_block_frames = sum(_block_frame_count(block) for block in local_blocks)
        stats_keys = list(dict.fromkeys(stats_key_map.values()))
        total_bytes = _estimate_transfer_bytes(
            first_dataset,
            stats_keys,
            total_block_frames,
        )
        key_bytes = {key: _key_bytes_per_frame(first_dataset, key) for key in stats_keys}
        minmax_desc = progress_desc or "Computing min/max"
        _debug_log(
            f"{minmax_desc}: fallback scan on rank {rank}, "
            f"precomputed_paths={len(precomputed_paths)}, "
            f"fallback_paths={len(fallback_paths)}, "
            f"datasets={len(stats_keys)}, "
            f"scan_blocks={len(scan_blocks)}, scan_windows={len(scan_windows)}, "
            f"local_blocks={len(local_blocks)}, local_frames={total_block_frames}, "
            f"est_read={_format_gib(total_bytes)}",
            enabled=not progress_disable,
        )
        if fallback_paths:
            _debug_log(
                f"{minmax_desc}: fallback paths={fallback_paths}",
                enabled=not progress_disable,
            )

        dataset_iterator = tqdm(
            desc=f"{minmax_desc}: datasets",
            total=len(stats_keys),
            position=progress_position,
            leave=False,
            disable=progress_disable,
            unit="dataset",
        )
        per_key_position = progress_position + 1
        per_key_minmax = {}
        for zarr_key in stats_keys:
            key_desc = f"{minmax_desc}: {zarr_key}"
            key_total_bytes = total_block_frames * key_bytes[zarr_key]
            if key_total_bytes > 0:
                key_desc = f"{key_desc} ({_format_gib(key_total_bytes)})"

            key_iterator = tqdm(
                desc=key_desc,
                total=total_block_frames,
                position=per_key_position,
                leave=False,
                disable=progress_disable,
                unit="frame",
            )

            feature_shape = first_dataset[zarr_key].shape[1:]
            key_min = np.full(feature_shape, np.inf, dtype=np.float64)
            key_max = np.full(feature_shape, -np.inf, dtype=np.float64)
            for block in local_blocks:
                dataset = opened[block["path_idx"]]
                block_start = block["frame_start"]
                block_end = block["frame_end"]
                array = dataset[zarr_key][block_start:block_end]
                local_min, local_max = _feature_min_max(array)
                key_min = np.minimum(key_min, local_min)
                key_max = np.maximum(key_max, local_max)
                key_iterator.update(block_end - block_start)

            key_iterator.close()
            dataset_iterator.update(1)
            per_key_minmax[zarr_key] = {"min": key_min, "max": key_max}

        dataset_iterator.close()
        for stat_key, zarr_key in stats_key_map.items():
            _update_feature_stats(
                local_stats,
                stat_key,
                per_key_minmax[zarr_key]["min"],
                per_key_minmax[zarr_key]["max"],
            )
    elif progress_desc and not progress_disable:
        _debug_log(
            f"{progress_desc}: using precomputed zarr normalizer stats "
            f"for {len(precomputed_paths)} path(s)",
            enabled=True,
        )

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
