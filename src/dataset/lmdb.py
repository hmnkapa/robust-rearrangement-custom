import json
import os
import struct
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from src.dataset.base import EpisodeRef

try:
    import lmdb
except ImportError:  # pragma: no cover - exercised only when LMDB support is used.
    lmdb = None


NORMALIZER_STATS_ATTR = "normalizer_stats"
LMDB_FORMAT_VERSION = 1

META_KEY = b"__meta__"
EPISODE_INDEX_KEY = b"__episode_index__"
FRAME_PREFIX = b"__frame__/"
EPISODE_DATA_PREFIX = b"__episode_data__/"

IMAGE_KEYS = ("color_image1", "color_image2", "depth_image1", "depth_image2")


def require_lmdb():
    if lmdb is None:
        raise ImportError(
            "LMDB support requires the `lmdb` package. Install it with `pip install lmdb`."
        )
    return lmdb


def dataset_tuple(path: Path) -> Tuple[str, str, str, str]:
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


def _init_feature_stats(first_specs, stats_key_map: Dict[str, str]):
    local_stats = {}
    for stat_key, lmdb_key in stats_key_map.items():
        feature_shape = tuple(first_specs[lmdb_key]["shape"][1:])
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


def json_dumps_bytes(value) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def json_loads_bytes(value: bytes):
    return json.loads(value.decode("utf-8"))


def frame_key(frame_idx: int) -> bytes:
    return FRAME_PREFIX + f"{int(frame_idx):012d}".encode("ascii")


def episode_data_key(episode_idx: int) -> bytes:
    return EPISODE_DATA_PREFIX + f"{int(episode_idx):08d}".encode("ascii")


def build_frame_specs(example_arrays: Dict[str, np.ndarray]) -> Dict[str, dict]:
    specs = {}
    offset = 0
    ordered_keys = []

    for key in IMAGE_KEYS:
        if key not in example_arrays:
            continue
        array = np.ascontiguousarray(example_arrays[key])
        specs[key] = {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "offset": offset,
            "nbytes": int(array.nbytes),
        }
        ordered_keys.append(key)
        offset += int(array.nbytes)

    return {
        "ordered_keys": ordered_keys,
        "specs": specs,
        "total_nbytes": offset,
    }


def pack_frame(images: Dict[str, np.ndarray], frame_specs: Dict[str, dict]) -> bytes:
    parts = []
    for key in frame_specs["ordered_keys"]:
        array = np.ascontiguousarray(images[key])
        spec = frame_specs["specs"][key]
        if list(array.shape) != spec["shape"] or str(array.dtype) != spec["dtype"]:
            raise ValueError(
                f"Frame key {key} has shape/dtype {array.shape}/{array.dtype}, "
                f"expected {tuple(spec['shape'])}/{spec['dtype']}."
            )
        parts.append(array.tobytes(order="C"))
    return b"".join(parts)


def unpack_frame(raw_value: bytes, frame_specs: Dict[str, dict], keys=None) -> Dict[str, np.ndarray]:
    payload = memoryview(raw_value)
    specs = frame_specs["specs"]
    requested_keys = frame_specs["ordered_keys"] if keys is None else keys
    arrays = {}
    for key in requested_keys:
        spec = specs[key]
        start = int(spec["offset"])
        end = start + int(spec["nbytes"])
        arrays[key] = (
            np.frombuffer(payload[start:end], dtype=np.dtype(spec["dtype"]))
            .reshape(tuple(spec["shape"]))
            .copy()
        )
    return arrays


def pack_named_arrays(named_arrays: Dict[str, np.ndarray]) -> bytes:
    payload_parts = []
    header = {"specs": {}}
    offset = 0

    for key, array in named_arrays.items():
        array = np.ascontiguousarray(array)
        nbytes = int(array.nbytes)
        header["specs"][key] = {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "offset": offset,
            "nbytes": nbytes,
        }
        payload_parts.append(array.tobytes(order="C"))
        offset += nbytes

    header_bytes = json_dumps_bytes(header)
    return struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(payload_parts)


def unpack_named_arrays(raw_value: bytes) -> Dict[str, np.ndarray]:
    if len(raw_value) < 8:
        raise ValueError("Corrupted LMDB array payload: missing header length.")

    header_len = struct.unpack("<Q", raw_value[:8])[0]
    header_start = 8
    header_end = header_start + header_len
    header = json_loads_bytes(raw_value[header_start:header_end])
    payload = memoryview(raw_value)[header_end:]

    arrays = {}
    for key, spec in header["specs"].items():
        start = int(spec["offset"])
        end = start + int(spec["nbytes"])
        arrays[key] = (
            np.frombuffer(payload[start:end], dtype=np.dtype(spec["dtype"]))
            .reshape(tuple(spec["shape"]))
            .copy()
        )
    return arrays


def open_lmdb_env(path: Union[str, Path], readonly: bool = True):
    lmdb_module = require_lmdb()
    path = Path(path)
    subdir = path.is_dir() or path.suffix == ".lmdb" or not path.exists()
    if not readonly:
        path.mkdir(parents=True, exist_ok=True)

    return lmdb_module.open(
        str(path),
        subdir=subdir,
        readonly=readonly,
        create=not readonly,
        lock=True,
        readahead=not readonly,
        meminit=not readonly,
        map_size=1 << 40,
        max_readers=2048,
        max_spare_txns=0,
    )


def read_lmdb_meta(path: Union[str, Path]) -> dict:
    env = open_lmdb_env(path, readonly=True)
    try:
        with env.begin(write=False) as txn:
            raw_meta = txn.get(META_KEY)
            if raw_meta is None:
                raise ValueError(f"Missing LMDB metadata in {path}.")
            return json_loads_bytes(raw_meta)
    finally:
        env.close()


def read_lmdb_episode_index(path: Union[str, Path]) -> List[dict]:
    env = open_lmdb_env(path, readonly=True)
    try:
        with env.begin(write=False) as txn:
            raw_index = txn.get(EPISODE_INDEX_KEY)
            if raw_index is None:
                raise ValueError(f"Missing episode index in {path}.")
            return json_loads_bytes(raw_index)
    finally:
        env.close()


def read_lmdb_attrs(path: Union[str, Path]) -> dict:
    return read_lmdb_meta(path)["attrs"]


class LMDBImageStore:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.meta = read_lmdb_meta(self.path)
        self.frame_specs = self.meta["frame_specs"]
        self._env = None
        self._pid = None

    def _ensure_env(self):
        current_pid = os.getpid()
        if self._env is None or self._pid != current_pid:
            if self._env is not None:
                self._env.close()
            self._env = open_lmdb_env(self.path, readonly=True)
            self._pid = current_pid
        return self._env

    def get_frames(self, frame_indices, keys) -> Dict[str, np.ndarray]:
        frame_indices = [int(idx) for idx in frame_indices]
        if not frame_indices:
            specs = self.frame_specs["specs"]
            return {
                key: np.empty((0,) + tuple(specs[key]["shape"]), dtype=np.dtype(specs[key]["dtype"]))
                for key in keys
            }

        env = self._ensure_env()
        outputs = {key: [] for key in keys}
        with env.begin(write=False) as txn:
            for frame_idx in frame_indices:
                raw_value = txn.get(frame_key(frame_idx))
                if raw_value is None:
                    raise KeyError(
                        f"Missing frame {frame_idx} in LMDB dataset {self.path}."
                    )
                decoded = unpack_frame(raw_value, self.frame_specs, keys=keys)
                for key in keys:
                    outputs[key].append(decoded[key])

        return {key: np.stack(values, axis=0) for key, values in outputs.items()}

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
            self._pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        state["_pid"] = None
        return state


def build_episode_manifest(
    lmdb_paths: Union[List[Path], Path],
    max_episodes=None,
    max_ep_cnt=None,
) -> List[EpisodeRef]:
    if not isinstance(lmdb_paths, list):
        lmdb_paths = [lmdb_paths]

    manifest: List[EpisodeRef] = []
    for path_idx, path in enumerate(lmdb_paths):
        meta = read_lmdb_meta(path)
        episode_index = read_lmdb_episode_index(path)
        max_ep = _resolve_max_episodes(path, max_episodes=max_episodes, max_ep_cnt=max_ep_cnt)
        if max_ep is not None:
            episode_index = episode_index[:max_ep]

        domain = str(meta["attrs"]["domain"])
        for episode_idx, episode_meta in enumerate(episode_index):
            manifest.append(
                EpisodeRef(
                    path_idx=path_idx,
                    episode_idx=episode_idx,
                    frame_start=int(episode_meta["frame_start"]),
                    frame_end=int(episode_meta["frame_end"]),
                    frame_count=int(episode_meta["frame_end"]) - int(episode_meta["frame_start"]),
                    task=str(_coerce_scalar(episode_meta["task"])),
                    success=int(_coerce_scalar(episode_meta["success"])),
                    domain=domain,
                )
            )

    return manifest


def _init_combined_data(first_meta, total_frames: int, total_episodes: int, keys):
    lowdim_specs = first_meta["lowdim_specs"]
    frame_specs = first_meta["frame_specs"]["specs"]
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
        if key in lowdim_specs:
            spec = lowdim_specs[key]
        elif key in frame_specs:
            spec = frame_specs[key]
        else:
            raise KeyError(f"Unknown LMDB key {key}.")

        combined_data[key] = np.zeros(
            (total_frames,) + tuple(spec["shape"][1:]),
            dtype=np.dtype(spec["dtype"]),
        )

    return combined_data


def _load_episode_arrays(
    txn,
    episode_idx: int,
    requested_keys,
) -> Dict[str, np.ndarray]:
    raw_value = txn.get(episode_data_key(episode_idx))
    if raw_value is None:
        raise KeyError(f"Missing episode data for episode {episode_idx}.")

    arrays = unpack_named_arrays(raw_value)
    return {key: arrays[key] for key in requested_keys}


def _load_frame_arrays(
    txn,
    frame_start: int,
    frame_end: int,
    frame_specs: Dict[str, dict],
    requested_keys,
) -> Dict[str, np.ndarray]:
    outputs = {key: [] for key in requested_keys}
    for frame_idx in range(frame_start, frame_end):
        raw_value = txn.get(frame_key(frame_idx))
        if raw_value is None:
            raise KeyError(f"Missing frame {frame_idx}.")
        decoded = unpack_frame(raw_value, frame_specs, keys=requested_keys)
        for key in requested_keys:
            outputs[key].append(decoded[key])
    return {key: np.stack(values, axis=0) for key, values in outputs.items()}


def combine_lmdb_episode_subset(
    lmdb_paths: Union[List[Path], Path],
    episode_refs: List[EpisodeRef],
    keys,
    progress_desc: Optional[str] = None,
    progress_position: int = 0,
    progress_disable: bool = False,
) -> Tuple[dict, dict]:
    if not isinstance(lmdb_paths, list):
        lmdb_paths = [lmdb_paths]
    if len(lmdb_paths) == 0:
        raise ValueError("combine_lmdb_episode_subset requires at least one LMDB path.")

    meta_by_path = {idx: read_lmdb_meta(path) for idx, path in enumerate(lmdb_paths)}
    episode_index_by_path = {
        idx: read_lmdb_episode_index(path) for idx, path in enumerate(lmdb_paths)
    }

    total_frames = sum(ref.frame_count for ref in episode_refs)
    total_episodes = len(episode_refs)
    first_meta = meta_by_path[0]
    combined_data = _init_combined_data(first_meta, total_frames, total_episodes, keys)
    metadata = {}
    per_path_episode_counts = defaultdict(int)
    per_path_frame_counts = defaultdict(int)
    domain_idx = dict(sim=0, real=1)

    requested_lowdim_keys = [
        key for key in keys if key in first_meta["lowdim_specs"]
    ]
    requested_image_keys = [
        key for key in keys if key in first_meta["frame_specs"]["specs"]
    ]

    env_cache = {}
    frame_cursor = 0

    episode_iterator = tqdm(
        episode_refs,
        desc=progress_desc or "Loading LMDB episodes",
        position=progress_position,
        leave=False,
        disable=progress_disable,
        unit="episode",
    )
    for episode_cursor, ref in enumerate(episode_iterator):
        env = env_cache.get(ref.path_idx)
        if env is None:
            env = open_lmdb_env(lmdb_paths[ref.path_idx], readonly=True)
            env_cache[ref.path_idx] = env

        meta = meta_by_path[ref.path_idx]
        episode_meta = episode_index_by_path[ref.path_idx][ref.episode_idx]
        frame_start = ref.frame_start
        frame_end = ref.frame_end
        frame_count = ref.frame_count

        combined_data["episode_ends"][episode_cursor] = frame_cursor + frame_count
        combined_data["task"].append(ref.task)
        combined_data["success"][episode_cursor] = ref.success
        combined_data["domain"][episode_cursor] = domain_idx[ref.domain]
        combined_data["zarr_idx"][frame_cursor : frame_cursor + frame_count] = ref.path_idx
        combined_data["within_zarr_idx"][frame_cursor : frame_cursor + frame_count] = np.arange(
            frame_start, frame_end
        )

        metadata_key = str(dataset_tuple(lmdb_paths[ref.path_idx]))
        per_path_episode_counts[metadata_key] += 1
        per_path_frame_counts[metadata_key] += frame_count
        metadata[metadata_key] = {
            "n_episodes_used": per_path_episode_counts[metadata_key],
            "n_frames_used": per_path_frame_counts[metadata_key],
            "attrs": meta["attrs"],
        }

        with env.begin(write=False) as txn:
            if requested_lowdim_keys:
                episode_arrays = _load_episode_arrays(txn, ref.episode_idx, requested_lowdim_keys)
                for key, value in episode_arrays.items():
                    if value.shape[0] != frame_count:
                        raise ValueError(
                            f"Episode {ref.episode_idx} key {key} has {value.shape[0]} "
                            f"frames, expected {frame_count}."
                        )
                    combined_data[key][frame_cursor : frame_cursor + frame_count] = value

            if requested_image_keys:
                frame_arrays = _load_frame_arrays(
                    txn,
                    frame_start,
                    frame_end,
                    meta["frame_specs"],
                    requested_image_keys,
                )
                for key, value in frame_arrays.items():
                    combined_data[key][frame_cursor : frame_cursor + frame_count] = value

        failure_idx = episode_meta.get("failure_idx")
        if failure_idx is not None:
            combined_data["failure_idx"][episode_cursor] = int(failure_idx)

        frame_cursor += frame_count

    for env in env_cache.values():
        env.close()

    return combined_data, metadata


def combine_lmdb_datasets(
    lmdb_paths: Union[List[Path], Path],
    keys,
    max_episodes=None,
    max_ep_cnt=None,
) -> Tuple[dict, dict]:
    manifest = build_episode_manifest(
        lmdb_paths,
        max_episodes=max_episodes,
        max_ep_cnt=max_ep_cnt,
    )
    return combine_lmdb_episode_subset(lmdb_paths, manifest, keys)


def _can_use_precomputed_stats(meta, episode_refs: List[EpisodeRef], stats_key_map):
    if len(episode_refs) == 0:
        return {}

    raw_stats = meta["attrs"].get(NORMALIZER_STATS_ATTR)
    if raw_stats is None:
        return None

    total_episodes = int(meta["attrs"]["n_episodes"])
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


def compute_global_minmax_stats(
    lmdb_paths: Union[List[Path], Path],
    episode_refs: List[EpisodeRef],
    stats_key_map: Dict[str, str],
    device: Union[torch.device, None] = None,
    progress_desc: Optional[str] = None,
    progress_position: int = 0,
    progress_disable: bool = False,
) -> Dict[str, Dict[str, torch.Tensor]]:
    if not isinstance(lmdb_paths, list):
        lmdb_paths = [lmdb_paths]
    if len(lmdb_paths) == 0:
        raise ValueError("compute_global_minmax_stats requires at least one LMDB path.")

    first_meta = read_lmdb_meta(lmdb_paths[0])
    local_stats = _init_feature_stats(first_meta["lowdim_specs"], stats_key_map)

    refs_by_path = defaultdict(list)
    for ref in episode_refs:
        refs_by_path[ref.path_idx].append(ref)

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    path_iterator = tqdm(
        sorted(refs_by_path.items()),
        desc=progress_desc or f"[Rank {rank}] LMDB min/max",
        position=progress_position,
        leave=False,
        disable=progress_disable,
        unit="path",
    )

    for path_idx, path_refs in path_iterator:
        path = lmdb_paths[path_idx]
        meta = read_lmdb_meta(path)

        precomputed_stats = _can_use_precomputed_stats(meta, path_refs, stats_key_map)
        if precomputed_stats is not None:
            for stat_key, lmdb_key in stats_key_map.items():
                stored_key_stats = precomputed_stats[lmdb_key]
                _update_feature_stats(
                    local_stats,
                    stat_key,
                    stored_key_stats["min"],
                    stored_key_stats["max"],
                )
            continue

        env = open_lmdb_env(path, readonly=True)
        try:
            with env.begin(write=False) as txn:
                for ref in path_refs:
                    episode_arrays = _load_episode_arrays(
                        txn,
                        ref.episode_idx,
                        list(dict.fromkeys(stats_key_map.values())),
                    )
                    for stat_key, lmdb_key in stats_key_map.items():
                        local_min, local_max = _feature_min_max(episode_arrays[lmdb_key])
                        _update_feature_stats(local_stats, stat_key, local_min, local_max)
        finally:
            env.close()

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
