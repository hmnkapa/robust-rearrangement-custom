import argparse
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from src.common.files import get_processed_path, get_raw_paths
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
)
from src.visualization.render_mp4 import unpickle_data


LOWDIM_KEYS = tuple(key for key in TIMESERIES_KEYS if key not in {
    "color_image1",
    "color_image2",
    "depth_image1",
    "depth_image2",
})


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
    if args.input_dir is not None:
        if task_episode_limits:
            raise ValueError(
                "--task-episode-limit is not supported together with --input-dir."
            )
        input_dir = Path(args.input_dir).expanduser().resolve()
        if not input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        pickle_paths = sorted(input_dir.rglob("*.pkl*"))
        print(f"Using explicit input directory: {input_dir}")
        return pickle_paths

    tasks = args.task if isinstance(args.task, list) else [args.task]
    selected_paths = []
    rng = random.Random(args.random_seed)

    for task in tasks:
        task_paths = sorted(
            get_raw_paths(
                controller=args.controller,
                domain=args.domain,
                task=task,
                demo_source=args.source,
                randomness=args.randomness,
                demo_outcome=args.demo_outcome,
                suffix=args.suffix,
            )
        )

        if args.randomize_order:
            rng.shuffle(task_paths)

        task_limit = task_episode_limits.get(task)
        if task_limit is not None:
            task_paths = task_paths[:task_limit]

        print(f"[INFO] Task {task}: selected {len(task_paths)} pickle files")
        selected_paths.extend(task_paths)

    if args.randomize_order:
        rng.shuffle(selected_paths)

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
    parser.add_argument("--max-files", type=int, default=None)
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
    args = parser.parse_args()

    assert not args.randomize_order or args.offset == 0, "Cannot offset with randomize"

    task_episode_limits = parse_task_episode_limits(args.task_episode_limit)
    pickle_paths = gather_pickle_paths(args, task_episode_limits)
    log_first_pickle_shape(pickle_paths)

    start = args.offset
    end = args.offset + args.max_files if args.max_files is not None else len(pickle_paths)
    pickle_paths = pickle_paths[start:end]
    print(f"Found {len(pickle_paths)} pickle files after filtering")
    if len(pickle_paths) == 0:
        raise ValueError("No pickle files selected; refusing to create an empty LMDB dataset.")

    if args.output_dir is not None:
        output_path = Path(args.output_dir).expanduser().resolve()
        print(f"Using explicit output path: {output_path}")
    else:
        output_path = get_processed_path(
            controller=args.controller,
            domain=args.domain,
            task=args.task,
            demo_source=args.source,
            randomness=args.randomness,
            demo_outcome=args.demo_outcome,
            suffix=args.output_suffix,
            dataset_format="lmdb",
        )

    print(f"Output path: {output_path}")
    if output_path.exists():
        if not args.overwrite:
            raise ValueError(
                f"Output path already exists: {output_path}. Use --overwrite to overwrite."
            )
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

        with env.begin(write=True) as txn:
            for episode_data in batch_results:
                if frame_specs is None:
                    frame_specs = build_frame_specs(
                        {
                            key: episode_data[key][0]
                            for key in ("color_image1", "color_image2", "depth_image1", "depth_image2")
                        }
                    )
                    lowdim_specs = build_lowdim_specs(episode_data)

                episode_length = int(episode_data["episode_length"])
                frame_start = global_frame_idx
                frame_end = frame_start + episode_length

                lowdim_payload = {
                    key: np.asarray(episode_data[key]) for key in LOWDIM_KEYS
                }
                txn.put(
                    episode_data_key(global_episode_idx),
                    pack_named_arrays(lowdim_payload),
                )

                for local_frame_idx in range(episode_length):
                    frame_payload = {
                        key: episode_data[key][local_frame_idx]
                        for key in ("color_image1", "color_image2", "depth_image1", "depth_image2")
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

        print(
            f"[INFO] Written batch {batch_start // batch_size + 1}/{total_batches}, "
            f"timesteps so far: {global_frame_idx}, episodes so far: {global_episode_idx}"
        )

    serialized_normalizer_stats = serialize_normalizer_stats(normalizer_stats)
    attrs = {
        "time_created": time_created,
        "time_finished": datetime.now().astimezone().isoformat(),
        "noop_threshold": noop_threshold,
        "rotation_mode": "rot_6d",
        "n_episodes": global_episode_idx,
        "n_timesteps": global_frame_idx,
        "mean_episode_length": round(global_frame_idx / global_episode_idx) if global_episode_idx else 0,
        "calculated_pos_action_from_delta": True,
        "randomize_order": args.randomize_order,
        "random_seed": args.random_seed,
        "demo_source": args.source,
        "controller": args.controller,
        "domain": args.domain if args.domain == "real" else "sim",
        "task": args.task if len(args.task) > 1 else args.task[0],
        "tasks": args.task,
        "selected_task_counts": selected_task_counts,
        "randomness": args.randomness,
        "demo_outcome": args.demo_outcome,
        "suffix": args.suffix,
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
    print("[INFO] LMDB processing complete.")


if __name__ == "__main__":
    main()
