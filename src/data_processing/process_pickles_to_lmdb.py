import argparse
import os
import random
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from src.common.files import (
    expand_lmdb_shard_paths,
    get_processed_path,
    get_raw_paths,
    lmdb_shard_path,
)
from src.data_processing.process_pickles import (
    NORMALIZER_STATS_KEYS,
    TIMESERIES_KEYS,
    compute_normalizer_stats_from_dict,
    merge_normalizer_stats,
    process_pickle_file,
    serialize_normalizer_stats,
)
from src.dataset.lmdb import (
    EPISODE_INDEX_KEY,
    LMDB_FORMAT_VERSION,
    META_KEY,
    build_frame_specs,
    episode_data_key,
    frame_key,
    json_dumps_bytes,
    open_lmdb_env,
    pack_frame,
    pack_named_arrays,
    require_lmdb,
    read_lmdb_episode_index,
    read_lmdb_meta,
)
from src.visualization.render_mp4 import unpickle_data


LOWDIM_KEYS = tuple(key for key in TIMESERIES_KEYS if key not in {
    "color_image1",
    "color_image2",
    "depth_image1",
    "depth_image2",
})
IMAGE_KEYS = ("color_image1", "color_image2", "depth_image1", "depth_image2")


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def log_lmdb_storage_layout(frame_specs, lowdim_specs, resize_image: bool):
    print(
        "[INFO] LMDB storage format: images are stored as raw per-frame byte payloads "
        "(no compression), low-dimensional arrays are stored once per episode."
    )
    print(f"[INFO] resize_image={resize_image} (default: False)")
    print("[INFO] Image frame layout:")
    for key in frame_specs["ordered_keys"]:
        spec = frame_specs["specs"][key]
        print(
            f"[INFO]   {key}: dtype={spec['dtype']}, shape={tuple(spec['shape'])}, "
            f"nbytes/frame={spec['nbytes']} ({format_bytes(int(spec['nbytes']))})"
        )
    print(
        "[INFO]   total image bytes/timestep="
        f"{frame_specs['total_nbytes']} ({format_bytes(int(frame_specs['total_nbytes']))})"
    )
    print("[INFO] Low-dimensional episode arrays:")
    for key in LOWDIM_KEYS:
        spec = lowdim_specs[key]
        print(
            f"[INFO]   {key}: dtype={spec['dtype']}, shape={tuple(spec['shape'])}"
        )


def log_episode_storage_debug(
    episode_data,
    frame_specs,
    packed_lowdim_nbytes: int,
):
    episode_length = int(episode_data["episode_length"])
    image_nbytes = int(frame_specs["total_nbytes"]) * episode_length
    total_nbytes = image_nbytes + packed_lowdim_nbytes
    print(
        f"[DEBUG] Example episode storage: timesteps={episode_length}, "
        f"image_bytes={image_nbytes} ({format_bytes(image_nbytes)}), "
        f"lowdim_bytes={packed_lowdim_nbytes} ({format_bytes(packed_lowdim_nbytes)}), "
        f"total={total_nbytes} ({format_bytes(total_nbytes)})"
    )
    for key in IMAGE_KEYS:
        array = np.asarray(episode_data[key])
        print(
            f"[DEBUG]   {key}: shape={tuple(array.shape)}, dtype={array.dtype}, "
            f"bytes/episode={array.nbytes} ({format_bytes(int(array.nbytes))})"
        )


def log_batch_storage_debug(
    batch_index: int,
    total_batches: int,
    batch_timesteps: int,
    batch_episodes: int,
    batch_image_bytes: int,
    batch_lowdim_bytes: int,
    running_total_bytes: int,
):
    batch_total_bytes = batch_image_bytes + batch_lowdim_bytes
    avg_bytes_per_timestep = (
        batch_total_bytes / batch_timesteps if batch_timesteps > 0 else 0.0
    )
    print(
        f"[DEBUG] Batch {batch_index}/{total_batches} payload estimate: "
        f"episodes={batch_episodes}, timesteps={batch_timesteps}, "
        f"image={format_bytes(batch_image_bytes)}, "
        f"lowdim={format_bytes(batch_lowdim_bytes)}, "
        f"total={format_bytes(batch_total_bytes)}, "
        f"avg/timestep={format_bytes(int(round(avg_bytes_per_timestep)))}, "
        f"running_total={format_bytes(running_total_bytes)}"
    )


def parse_task_episode_limits(entries: List[str]) -> Dict[str, int]:
    limits = {}
    for entry in entries or []:
        if "=" not in entry:
            raise ValueError(
                f"Invalid --task-episode-limit entry {entry!r}. Expected TASK=COUNT."
            )
        task, count = entry.split("=", 1)
        task = task.strip()
        if not task:
            raise ValueError(f"Invalid task name in entry {entry!r}.")
        limits[task] = int(count)
    return limits


def parse_datetime_from_pickle_name(path: Path) -> Optional[datetime]:
    name = path.name
    patterns = (
        (
            r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}(?:\.\d+)?",
            ("%Y-%m-%dT%H-%M-%S.%f", "%Y-%m-%dT%H-%M-%S"),
        ),
        (
            r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:\.\d+)?",
            ("%Y-%m-%d_%H-%M-%S.%f", "%Y-%m-%d_%H-%M-%S"),
        ),
        (
            r"\d{8}_\d{6}(?:\.\d+)?",
            ("%Y%m%d_%H%M%S.%f", "%Y%m%d_%H%M%S"),
        ),
    )
    for pattern, formats in patterns:
        for match in re.findall(pattern, name):
            for time_format in formats:
                try:
                    return datetime.strptime(match, time_format)
                except ValueError:
                    continue
    return None


def pickle_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def newest_pickle_sort_key(path: Path):
    parsed_time = parse_datetime_from_pickle_name(path)
    timestamp = (
        parsed_time.timestamp() if parsed_time is not None else pickle_mtime(path)
    )
    return (-timestamp, str(path))


def order_pickle_paths(
    paths: List[Path],
    randomize_order: bool,
    rng: random.Random,
) -> List[Path]:
    paths = list(paths)
    if randomize_order:
        rng.shuffle(paths)
        return paths
    return sorted(paths, key=newest_pickle_sort_key)


def pickle_identity(path: Path) -> str:
    path = Path(path).expanduser()
    try:
        resolved_path = path.resolve()
    except OSError:
        resolved_path = path.absolute()

    for candidate in (resolved_path, path):
        parts = candidate.parts
        if "raw" in parts:
            return "/".join(parts[parts.index("raw") + 1 :])
    return str(resolved_path)


def absolute_pickle_path(path: Path) -> str:
    path = Path(path).expanduser()
    try:
        return str(path.resolve())
    except OSError:
        return str(path.absolute())


def unnumbered_lmdb_base_path(path: Path) -> Path:
    match = re.match(r"^(?P<stem>.*)-\d+$", path.stem)
    if match:
        return path.with_name(f"{match.group('stem')}{path.suffix}")
    return path


def next_lmdb_shard_path(base_path: Path) -> Path:
    shard_index = 1
    while True:
        candidate = lmdb_shard_path(base_path, shard_index)
        if not candidate.exists():
            return candidate
        shard_index += 1


def infer_shard_index(base_path: Path, output_path: Path) -> Optional[int]:
    match = re.match(
        rf"^{re.escape(base_path.stem)}-(\d+){re.escape(base_path.suffix)}$",
        output_path.name,
    )
    if match:
        return int(match.group(1))
    return None


def resolve_output_path(
    base_path: Path,
    overwrite: bool,
    explicit_output_dir: bool,
) -> Path:
    if explicit_output_dir:
        return base_path
    if overwrite:
        return lmdb_shard_path(base_path, 1)
    return next_lmdb_shard_path(base_path)


def string_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, bytes):
        return [value.decode("utf-8")]
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in value
        ]
    return [str(value)]


def read_used_pickle_files(path: Path) -> List[str]:
    used_pickle_files = []
    try:
        meta = read_lmdb_meta(path)
        attrs = meta.get("attrs", {})
        used_pickle_files.extend(string_list(attrs.get("pickle_files")))
        used_pickle_files.extend(string_list(attrs.get("selected_pickle_files")))
        used_pickle_files.extend(string_list(attrs.get("pickle_paths")))
        used_pickle_files.extend(string_list(attrs.get("selected_pickle_paths")))
    except Exception as exc:
        print(f"[WARNING] Could not read LMDB metadata from {path}: {exc}")

    try:
        episode_index = read_lmdb_episode_index(path)
        used_pickle_files.extend(
            str(episode_meta["pickle_file"])
            for episode_meta in episode_index
            if "pickle_file" in episode_meta
        )
    except Exception as exc:
        print(f"[WARNING] Could not read LMDB episode index from {path}: {exc}")

    return used_pickle_files


def confirm_no_duplicate_pickles(
    selected_pickle_files: List[str],
    selected_pickle_paths: List[str],
    existing_lmdb_paths: List[Path],
):
    if not existing_lmdb_paths:
        return

    selected = set(selected_pickle_files) | set(selected_pickle_paths)
    duplicates_by_lmdb = {}
    for lmdb_path in existing_lmdb_paths:
        used = set(read_used_pickle_files(lmdb_path))
        overlap = sorted(selected & used)
        if overlap:
            duplicates_by_lmdb[lmdb_path] = overlap

    if not duplicates_by_lmdb:
        return

    print("[WARNING] Some selected pickle files already appear in existing LMDB shards:")
    for lmdb_path, overlaps in duplicates_by_lmdb.items():
        print(f"[WARNING]   {lmdb_path}")
        for pickle_file in overlaps[:20]:
            print(f"[WARNING]     {pickle_file}")
        if len(overlaps) > 20:
            print(f"[WARNING]     ... and {len(overlaps) - 20} more")

    if not sys.stdin.isatty():
        raise RuntimeError(
            "Duplicate pickle files were detected and stdin is not interactive; aborting."
        )

    answer = input("Continue and write a shard with duplicate pickle files? [y/N] ")
    if answer.strip().lower() != "y":
        raise RuntimeError("Aborted because selected pickle files were already used.")


def log_first_pickle_shape(pickle_paths: List[Path]):
    total_files = len(pickle_paths)
    if total_files == 0:
        print("[WARNING] No pickle files found for the specified criteria.")
        return

    first_pickle_data = unpickle_data(pickle_paths[0])
    print("[INFO] Shape of the first pickle file's data:")
    for key, value in first_pickle_data.items():
        if key in {"success", "task", "action_type"}:
            print(f"{key}: {value} (type: {type(value)})")
        elif key in {"rewards", "actions"}:
            print(f"{key}: shape {np.shape(value)}")
        elif key == "observations":
            print(f"{key}: number of observations {len(value)}")
            if len(value) > 0:
                for obs_key, obs_value in value[0].items():
                    if obs_key == "robot_state" and isinstance(obs_value, dict):
                        for sub_key, sub_value in obs_value.items():
                            print(f"  robot_state/{sub_key}: shape {np.shape(sub_value)}")
                    elif isinstance(obs_value, np.ndarray):
                        print(f"  {obs_key}: shape {obs_value.shape}")
                    else:
                        print(f"  {obs_key}: type {type(obs_value)}")


def gather_pickle_paths(args, task_episode_limits: Dict[str, int]) -> List[Path]:
    rng = random.Random(args.random_seed)

    if args.input_dir is not None:
        if task_episode_limits:
            raise ValueError(
                "--task-episode-limit is not supported together with --input-dir."
            )
        input_dir = Path(args.input_dir).expanduser().resolve()
        if not input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        pickle_paths = order_pickle_paths(
            list(input_dir.rglob("*.pkl*")),
            randomize_order=args.randomize_order,
            rng=rng,
        )
        print(f"Using explicit input directory: {input_dir}")
        return pickle_paths

    tasks = args.task if isinstance(args.task, list) else [args.task]
    selected_paths = []

    for task in tasks:
        task_paths = order_pickle_paths(
            get_raw_paths(
                controller=args.controller,
                domain=args.domain,
                task=task,
                demo_source=args.source,
                randomness=args.randomness,
                demo_outcome=args.demo_outcome,
                suffix=args.suffix,
            ),
            randomize_order=args.randomize_order,
            rng=rng,
        )

        task_limit = task_episode_limits.get(task)
        if task_limit is not None:
            task_paths = task_paths[:task_limit]

        print(f"[INFO] Task {task}: selected {len(task_paths)} pickle files")
        selected_paths.extend(task_paths)

    if args.randomize_order:
        rng.shuffle(selected_paths)
    else:
        selected_paths = order_pickle_paths(
            selected_paths,
            randomize_order=False,
            rng=rng,
        )

    return selected_paths


def process_batch(batch_paths, noop_threshold, n_cpus, resize_image):
    if n_cpus <= 1:
        return [
            process_pickle_file(
                path,
                noop_threshold=noop_threshold,
                calculate_pos_action_from_delta=True,
                resize_image=resize_image,
            )
            for path in batch_paths
        ]

    with ThreadPoolExecutor(max_workers=n_cpus) as executor:
        return list(
            executor.map(
                lambda path: process_pickle_file(
                    path,
                    noop_threshold=noop_threshold,
                    calculate_pos_action_from_delta=True,
                    resize_image=resize_image,
                ),
                batch_paths,
            )
        )


def build_lowdim_specs(example_episode_data):
    return {
        key: {
            "dtype": str(np.asarray(example_episode_data[key]).dtype),
            "shape": list(np.asarray(example_episode_data[key]).shape),
        }
        for key in LOWDIM_KEYS
    }


def ensure_removed_output_path(output_path: Path):
    if not output_path.exists():
        return

    if output_path.is_dir():
        shutil.rmtree(output_path)
    else:
        output_path.unlink()


def main():
    require_lmdb()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controller",
        "-c",
        type=str,
        required=True,
        choices=["osc", "diffik"],
    )
    parser.add_argument(
        "--domain",
        "-d",
        type=str,
        choices=["sim", "real", "distillation"],
        required=True,
    )
    parser.add_argument(
        "--task",
        "-f",
        type=str,
        nargs="+",
        required=True,
        help="One or more task names. Multiple tasks will be merged into one LMDB.",
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        choices=["scripted", "rollout", "teleop", "augmentation"],
        required=True,
    )
    parser.add_argument(
        "--randomness",
        "-r",
        type=str,
        choices=["low", "low_perturb", "med", "med_perturb", "high", "high_perturb"],
        required=True,
    )
    parser.add_argument(
        "--demo-outcome",
        "-o",
        type=str,
        choices=["success", "failure", "partial_success"],
        required=True,
    )
    parser.add_argument("--suffix", type=str, default=None)
    parser.add_argument("--output-suffix", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--num-pickles",
        type=int,
        default=None,
        help="Maximum number of newest pickle files to process in this shard.",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--randomize-order", action="store_true")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--n-cpus", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--map-size-gb", type=int, default=1024)
    parser.add_argument(
        "--resize-image",
        action="store_true",
        help="Resize images to standard dimensions (240x320x3).",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Path to the directory containing pkl files",
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to save the LMDB directory",
        default=None,
    )
    parser.add_argument(
        "--task-episode-limit",
        type=str,
        nargs="*",
        default=None,
        help="Per-task episode limits, for example: one_leg=100 round_table=50",
    )
    parser.add_argument(
        "--debug-storage-stats",
        action="store_true",
        help="Print detailed LMDB payload estimates for image and low-dimensional data.",
    )
    args = parser.parse_args()

    assert not args.randomize_order or args.offset == 0, "Cannot offset with randomize"
    if args.offset < 0:
        raise ValueError(f"--offset must be non-negative, got {args.offset}.")
    if args.num_pickles is not None and args.num_pickles <= 0:
        raise ValueError(f"--num-pickles must be positive, got {args.num_pickles}.")

    task_episode_limits = parse_task_episode_limits(args.task_episode_limit)
    pickle_paths = gather_pickle_paths(args, task_episode_limits)
    log_first_pickle_shape(pickle_paths)

    start = args.offset
    end = (
        args.offset + args.num_pickles
        if args.num_pickles is not None
        else len(pickle_paths)
    )
    pickle_paths = pickle_paths[start:end]
    print(f"Found {len(pickle_paths)} pickle files after filtering")
    if len(pickle_paths) == 0:
        raise ValueError("No pickle files selected; refusing to create an empty LMDB dataset.")

    selected_pickle_files = [pickle_identity(path) for path in pickle_paths]
    selected_pickle_paths = [absolute_pickle_path(path) for path in pickle_paths]

    explicit_output_dir = args.output_dir is not None
    if args.output_dir is not None:
        base_output_path = Path(args.output_dir).expanduser().resolve()
        print(f"Using explicit output path: {base_output_path}")
    else:
        base_output_path = get_processed_path(
            controller=args.controller,
            domain=args.domain,
            task=args.task,
            demo_source=args.source,
            randomness=args.randomness,
            demo_outcome=args.demo_outcome,
            suffix=args.output_suffix,
            dataset_format="lmdb",
        )

    duplicate_scan_base = unnumbered_lmdb_base_path(base_output_path)
    existing_lmdb_paths = expand_lmdb_shard_paths(duplicate_scan_base)
    output_path = resolve_output_path(
        base_output_path,
        overwrite=args.overwrite,
        explicit_output_dir=explicit_output_dir,
    )
    shard_index = infer_shard_index(duplicate_scan_base, output_path)

    print(f"Base output path: {base_output_path}")
    print(f"Resolved output path: {output_path}")
    if output_path.exists():
        if not args.overwrite:
            raise ValueError(
                f"Output path already exists: {output_path}. Use --overwrite to overwrite."
            )

    confirm_no_duplicate_pickles(
        selected_pickle_files=selected_pickle_files,
        selected_pickle_paths=selected_pickle_paths,
        existing_lmdb_paths=existing_lmdb_paths,
    )

    if output_path.exists():
        ensure_removed_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    noop_threshold = 0.0
    n_cpus = min(os.cpu_count() or 1, args.n_cpus)
    batch_size = max(1, args.batch_size)
    time_created = datetime.now().astimezone().isoformat()
    env = open_lmdb_env(output_path, readonly=False)
    env.set_mapsize(int(args.map_size_gb) * (1024**3))

    episode_index = []
    normalizer_stats = {}
    frame_specs = None
    lowdim_specs = None
    global_frame_idx = 0
    global_episode_idx = 0
    selected_task_counts = {task: 0 for task in args.task}
    running_payload_bytes = 0

    total_batches = (len(pickle_paths) + batch_size - 1) // batch_size if pickle_paths else 0
    print(
        f"Processing pickle files with {n_cpus} CPUs, batch_size={batch_size}, "
        f"noop_threshold={noop_threshold}, total_batches={total_batches}"
    )

    for batch_start in range(0, len(pickle_paths), batch_size):
        batch_paths = pickle_paths[batch_start : batch_start + batch_size]
        batch_results = process_batch(
            batch_paths,
            noop_threshold=noop_threshold,
            n_cpus=n_cpus,
            resize_image=args.resize_image,
        )
        batch_image_bytes = 0
        batch_lowdim_bytes = 0
        batch_timesteps = 0

        with env.begin(write=True) as txn:
            for episode_data in batch_results:
                if frame_specs is None:
                    frame_specs = build_frame_specs(
                        {
                            key: episode_data[key][0]
                            for key in IMAGE_KEYS
                        }
                    )
                    lowdim_specs = build_lowdim_specs(episode_data)
                    log_lmdb_storage_layout(
                        frame_specs,
                        lowdim_specs,
                        resize_image=args.resize_image,
                    )

                episode_length = int(episode_data["episode_length"])
                frame_start = global_frame_idx
                frame_end = frame_start + episode_length

                lowdim_payload = {
                    key: np.asarray(episode_data[key]) for key in LOWDIM_KEYS
                }
                packed_lowdim_payload = pack_named_arrays(lowdim_payload)
                txn.put(
                    episode_data_key(global_episode_idx),
                    packed_lowdim_payload,
                )
                packed_lowdim_nbytes = len(packed_lowdim_payload)
                batch_lowdim_bytes += packed_lowdim_nbytes
                batch_image_bytes += int(frame_specs["total_nbytes"]) * episode_length
                batch_timesteps += episode_length

                if args.debug_storage_stats and global_episode_idx == 0:
                    log_episode_storage_debug(
                        episode_data,
                        frame_specs,
                        packed_lowdim_nbytes,
                    )

                for local_frame_idx in range(episode_length):
                    frame_payload = {
                        key: episode_data[key][local_frame_idx]
                        for key in IMAGE_KEYS
                    }
                    txn.put(
                        frame_key(global_frame_idx + local_frame_idx),
                        pack_frame(frame_payload, frame_specs),
                    )

                episode_index.append(
                    {
                        "episode_idx": global_episode_idx,
                        "frame_start": frame_start,
                        "frame_end": frame_end,
                        "task": episode_data["task"],
                        "success": int(episode_data["success"]),
                        "pickle_file": episode_data["pickle_file"],
                    }
                )
                selected_task_counts.setdefault(episode_data["task"], 0)
                selected_task_counts[episode_data["task"]] += 1

                episode_stats = compute_normalizer_stats_from_dict(lowdim_payload)
                merge_normalizer_stats(normalizer_stats, episode_stats)

                global_frame_idx = frame_end
                global_episode_idx += 1

        running_payload_bytes += batch_image_bytes + batch_lowdim_bytes

        print(
            f"[INFO] Written batch {batch_start // batch_size + 1}/{total_batches}, "
            f"timesteps so far: {global_frame_idx}, episodes so far: {global_episode_idx}"
        )
        if args.debug_storage_stats:
            log_batch_storage_debug(
                batch_index=batch_start // batch_size + 1,
                total_batches=total_batches,
                batch_timesteps=batch_timesteps,
                batch_episodes=len(batch_results),
                batch_image_bytes=batch_image_bytes,
                batch_lowdim_bytes=batch_lowdim_bytes,
                running_total_bytes=running_payload_bytes,
            )

    serialized_normalizer_stats = serialize_normalizer_stats(normalizer_stats)
    attrs = {
        "time_created": time_created,
        "time_finished": datetime.now().astimezone().isoformat(),
        "noop_threshold": noop_threshold,
        "rotation_mode": "rot_6d",
        "n_episodes": global_episode_idx,
        "n_timesteps": global_frame_idx,
        "mean_episode_length": (
            round(global_frame_idx / global_episode_idx) if global_episode_idx else 0
        ),
        "calculated_pos_action_from_delta": True,
        "randomize_order": args.randomize_order,
        "random_seed": args.random_seed,
        "pickle_order": "random" if args.randomize_order else "newest",
        "offset": args.offset,
        "num_pickles": args.num_pickles,
        "selected_pickle_count": len(pickle_paths),
        "pickle_files": selected_pickle_files,
        "pickle_paths": selected_pickle_paths,
        "shard_index": shard_index,
        "shard_path": str(output_path),
        "demo_source": args.source,
        "controller": args.controller,
        "domain": args.domain if args.domain == "real" else "sim",
        "task": args.task if len(args.task) > 1 else args.task[0],
        "tasks": args.task,
        "selected_task_counts": selected_task_counts,
        "randomness": args.randomness,
        "demo_outcome": args.demo_outcome,
        "suffix": args.suffix,
        "output_suffix": args.output_suffix,
        "storage_format": "lmdb",
        "normalizer_stats": serialized_normalizer_stats,
        "normalizer_stats_keys": list(NORMALIZER_STATS_KEYS),
    }
    meta = {
        "format": "robust_rearrangement_lmdb",
        "format_version": LMDB_FORMAT_VERSION,
        "attrs": attrs,
        "frame_specs": frame_specs,
        "lowdim_specs": lowdim_specs or {},
    }

    with env.begin(write=True) as txn:
        txn.put(META_KEY, json_dumps_bytes(meta))
        txn.put(EPISODE_INDEX_KEY, json_dumps_bytes(episode_index))

    env.sync()
    env.close()
    if args.debug_storage_stats:
        print(
            "[DEBUG] Final LMDB payload estimate (metadata excluded): "
            f"{format_bytes(running_payload_bytes)} across {global_frame_idx} timesteps"
        )
    print("[INFO] LMDB processing complete.")


if __name__ == "__main__":
    main()
