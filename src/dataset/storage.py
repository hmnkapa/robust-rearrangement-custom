import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import zarr

from src.dataset import lmdb as lmdb_backend
from src.dataset import zarr as zarr_backend


def detect_dataset_format(path: Union[str, Path]) -> str:
    path = Path(path)
    if path.suffix == ".zarr" or (path.is_dir() and (path / ".zgroup").exists()):
        return "zarr"
    if path.suffix == ".lmdb" or (path.is_dir() and (path / "data.mdb").exists()):
        return "lmdb"
    raise ValueError(f"Unsupported dataset path format: {path}")


def ensure_homogeneous_dataset_format(dataset_paths: Union[List[Path], Path]) -> str:
    if not isinstance(dataset_paths, list):
        dataset_paths = [dataset_paths]
    if len(dataset_paths) == 0:
        raise ValueError("At least one dataset path is required.")

    formats = {detect_dataset_format(path) for path in dataset_paths}
    if len(formats) != 1:
        raise ValueError(
            f"Mixed dataset formats are not supported: {sorted(formats)}"
        )
    return formats.pop()


def dataset_tuple(path: Path):
    return path.with_name(path.stem).parts[-4:]


def combine_datasets(dataset_paths, keys, max_episodes=None, max_ep_cnt=None):
    dataset_format = ensure_homogeneous_dataset_format(dataset_paths)
    if dataset_format == "zarr":
        return zarr_backend.combine_zarr_datasets(
            dataset_paths,
            keys,
            max_episodes=max_episodes,
            max_ep_cnt=max_ep_cnt,
        )
    return lmdb_backend.combine_lmdb_datasets(
        dataset_paths,
        keys,
        max_episodes=max_episodes,
        max_ep_cnt=max_ep_cnt,
    )


def combine_episode_subset(
    dataset_paths,
    episode_refs,
    keys,
    progress_desc: Optional[str] = None,
    progress_position: int = 0,
    progress_disable: bool = False,
):
    dataset_format = ensure_homogeneous_dataset_format(dataset_paths)
    if dataset_format == "zarr":
        return zarr_backend.combine_zarr_episode_subset(
            dataset_paths,
            episode_refs,
            keys,
            progress_desc=progress_desc,
            progress_position=progress_position,
            progress_disable=progress_disable,
        )
    return lmdb_backend.combine_lmdb_episode_subset(
        dataset_paths,
        episode_refs,
        keys,
        progress_desc=progress_desc,
        progress_position=progress_position,
        progress_disable=progress_disable,
    )


def build_episode_manifest(dataset_paths, max_episodes=None, max_ep_cnt=None):
    dataset_format = ensure_homogeneous_dataset_format(dataset_paths)
    if dataset_format == "zarr":
        return zarr_backend.build_episode_manifest(
            dataset_paths,
            max_episodes=max_episodes,
            max_ep_cnt=max_ep_cnt,
        )
    return lmdb_backend.build_episode_manifest(
        dataset_paths,
        max_episodes=max_episodes,
        max_ep_cnt=max_ep_cnt,
    )


def split_episode_manifest(manifest, test_split: float, seed: int):
    return zarr_backend.split_episode_manifest(manifest, test_split, seed)


def balance_episode_manifest_by_frames(manifest, world_size: int):
    return zarr_backend.balance_episode_manifest_by_frames(manifest, world_size)


def compute_global_minmax_stats(
    dataset_paths,
    episode_refs,
    stats_key_map: Dict[str, str],
    device=None,
    progress_desc: Optional[str] = None,
    progress_position: int = 0,
    progress_disable: bool = False,
):
    dataset_format = ensure_homogeneous_dataset_format(dataset_paths)
    if dataset_format == "zarr":
        return zarr_backend.compute_global_minmax_stats(
            dataset_paths,
            episode_refs,
            stats_key_map,
            device=device,
            progress_desc=progress_desc,
            progress_position=progress_position,
            progress_disable=progress_disable,
        )
    return lmdb_backend.compute_global_minmax_stats(
        dataset_paths,
        episode_refs,
        stats_key_map,
        device=device,
        progress_desc=progress_desc,
        progress_position=progress_position,
        progress_disable=progress_disable,
    )


def read_dataset_attrs(path: Union[str, Path]) -> dict:
    dataset_format = detect_dataset_format(path)
    if dataset_format == "zarr":
        return zarr.open(path, mode="r").attrs.asdict()
    return lmdb_backend.read_lmdb_attrs(path)


def summarize_manifest_metadata(data_paths: List[Path], episode_refs) -> Dict[str, dict]:
    metadata = {}
    attrs_cache = {}
    for ref in episode_refs:
        metadata_key = str(dataset_tuple(data_paths[ref.path_idx]))
        if ref.path_idx not in attrs_cache:
            attrs_cache[ref.path_idx] = read_dataset_attrs(data_paths[ref.path_idx])

        if metadata_key not in metadata:
            metadata[metadata_key] = {
                "n_episodes_used": 0,
                "n_frames_used": 0,
                "attrs": attrs_cache[ref.path_idx],
            }

        metadata[metadata_key]["n_episodes_used"] += 1
        metadata[metadata_key]["n_frames_used"] += ref.frame_count

    return metadata


def resolve_load_into_memory(
    requested_value,
    dataset_paths,
    observation_type: str,
) -> bool:
    if isinstance(requested_value, str):
        normalized = requested_value.strip().lower()
        if normalized == "auto":
            requested_value = None
        elif normalized in {"true", "1", "yes"}:
            requested_value = True
        elif normalized in {"false", "0", "no"}:
            requested_value = False
        else:
            raise ValueError(
                f"Unsupported load_into_memory value: {requested_value!r}"
            )

    if requested_value is not None:
        return bool(requested_value)

    dataset_format = ensure_homogeneous_dataset_format(dataset_paths)
    if dataset_format == "lmdb" and observation_type in {"image", "rgbd"}:
        return False
    return True


class ZarrImageStore:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self._dataset = None
        self._pid = None

    def _ensure_dataset(self):
        current_pid = os.getpid()
        if self._dataset is None or self._pid != current_pid:
            self._dataset = zarr.open(self.path, mode="r")
            self._pid = current_pid
        return self._dataset

    def get_frames(self, frame_indices, keys):
        dataset = self._ensure_dataset()
        frame_indices = np.asarray(frame_indices, dtype=np.int64)
        if frame_indices.size == 0:
            return {
                key: np.empty((0,) + dataset[key].shape[1:], dtype=dataset[key].dtype)
                for key in keys
            }

        is_contiguous = np.array_equal(
            frame_indices,
            np.arange(frame_indices[0], frame_indices[0] + len(frame_indices)),
        )
        outputs = {}
        for key in keys:
            if is_contiguous:
                outputs[key] = dataset[key][frame_indices[0] : frame_indices[-1] + 1]
            else:
                outputs[key] = np.stack(
                    [dataset[key][int(frame_idx)] for frame_idx in frame_indices],
                    axis=0,
                )
        return outputs

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_dataset"] = None
        state["_pid"] = None
        return state


def build_lazy_image_stores(dataset_paths):
    if not isinstance(dataset_paths, list):
        dataset_paths = [dataset_paths]

    dataset_format = ensure_homogeneous_dataset_format(dataset_paths)
    if dataset_format == "zarr":
        return [ZarrImageStore(path) for path in dataset_paths]
    return [lmdb_backend.LMDBImageStore(path) for path in dataset_paths]
