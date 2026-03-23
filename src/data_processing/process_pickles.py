import argparse
import array
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import random
from typing import List

import numpy as np
import torch
import zarr
from furniture_bench.robot.robot_state import filter_and_concat_robot_state
from numcodecs import Blosc, JSON
from tqdm import tqdm, trange
from src.common.types import Trajectory
from src.common.files import get_processed_path, get_raw_paths
from src.visualization.render_mp4 import unpickle_data
from src.common.geometry import (
    np_proprioceptive_quat_to_6d_rotation,
    np_action_quat_to_6d_rotation,
    np_apply_quat,
)
from src.data_processing.utils import resize, resize_crop
from src.data_processing.utils import clip_quat_xyzw_magnitude

from ipdb import set_trace as bp  # noqa
import sys

SKILL_ORDER = ("pick", "place", "insert", "screw", "push")
SKILL_TO_ONEHOT = {
    skill: np.eye(len(SKILL_ORDER), dtype=np.float32)[idx]
    for idx, skill in enumerate(SKILL_ORDER)
}


# === Modified Function to Initialize Zarr Store with Full Dimensions ===
def initialize_zarr_store(out_path, full_data_shapes, chunksize=32):
    """
    Initialize the Zarr store with full dimensions for each dataset.
    """
    z = zarr.open(str(out_path), mode="w")
    z.attrs["time_created"] = datetime.now().astimezone().isoformat()

    # Define the compressor
    # compressor = Blosc(cname="zstd", clevel=9, shuffle=Blosc.BITSHUFFLE)
    compressor = Blosc(cname="lz4", clevel=5)

    # Initialize datasets with full shapes
    for name, shape, dtype in full_data_shapes:
        if "image" in name:  # Apply compression to image data
            z.create_dataset(
                name,
                shape=shape,
                dtype=dtype,
                chunks=(chunksize,) + shape[1:],
                compressor=compressor,
            )
        elif dtype == object:
            z.create_dataset(
            name,
            shape=shape,
            dtype=dtype,
            chunks=shape,
            object_codec=JSON(),
            )
        else:
            z.create_dataset(
            name, shape=shape, dtype=dtype, chunks=shape
            )

    return z


def process_pickle_file(
    pickle_path: Path,
    noop_threshold: float,
    calculate_pos_action_from_delta: bool = False,
    resize_image: bool = False,
):
    """
    Process a single pickle file and return processed data.
    """
    data: Trajectory = unpickle_data(pickle_path)
    obs = data["observations"]

    action_delta_quat = np.array(data["actions"], dtype=np.float32)
    assert (
        action_delta_quat.shape[-1] == 8
    ), "Expecting actions to be 8D (pos, quat, gripper)"

    if len(obs) == len(action_delta_quat) + 1:
        # The simulator data collection stores the observation received after
        # the last action. We need to remove this observation to match the lengths
        obs = obs[:-1]
    if len(obs) == len(action_delta_quat):
        # In the real world, we apparently don't do that
        pass
    else:
        raise ValueError(
            f"Observations and actions have different lengths: {len(obs)} vs {len(action_delta_quat)}"
        )

    # Extract the observations from the pickle file and convert to 6D rotation
    color_image1 = np.array([o["color_image1"] for o in obs], dtype=np.uint8)
    color_image2 = np.array([o["color_image2"] for o in obs], dtype=np.uint8)

    # Backward compatibility: older pickles may not include depth images.
    sample_depth1 = obs[0].get("depth_image1", None)
    sample_depth2 = obs[0].get("depth_image2", None)
    default_depth_shape = color_image1.shape[1:3]
    if sample_depth1 is not None and sample_depth2 is not None:
        depth_image1 = np.array([o["depth_image1"] for o in obs], dtype=np.float32)
        depth_image2 = np.array([o["depth_image2"] for o in obs], dtype=np.float32)
    else:
        print(f"[WARN] Missing depth images in {pickle_path}, filling zeros for depth_image1/2.")
        depth_image1 = np.zeros((len(obs),) + default_depth_shape, dtype=np.float32)
        depth_image2 = np.zeros((len(obs),) + default_depth_shape, dtype=np.float32)

    assert (
        color_image1.shape == color_image2.shape
    ), "Color images have different shapes"

    assert (
        depth_image1.shape == depth_image2.shape
    ), "Depth images have different shapes"

    if resize_image:
        if color_image1.shape[1:] != (240, 320, 3):
            # Resize only if the shape is not already correct
            color_image1 = resize(color_image1)
            color_image2 = resize_crop(color_image2)

        # Ensure the shape is consistent with the expected Zarr dataset shape
        assert color_image1.shape[1:] == (240, 320, 3), f"Unexpected shape for color_image1: {color_image1.shape[1:]}"
        # assert color_image2.shape[1:] == (240, 320, 3), f"Unexpected shape for color_image2: {color_image2.shape[1:]}"
    else:
        print("[INFO] Skipping image resizing as --resize-image is not set.")

    if isinstance(obs[0]["robot_state"], dict):
        # Convert the robot state to a numpy array
        robot_state_quat = np.array(
            [filter_and_concat_robot_state(o["robot_state"]) for o in obs],
            dtype=np.float32,
        )
    else:
        robot_state_quat = np.array([o["robot_state"] for o in obs], dtype=np.float32)

    robot_state_6d = np_proprioceptive_quat_to_6d_rotation(robot_state_quat)
    parts_poses = (
        np.array([o["parts_poses"] for o in obs], dtype=np.float32)
        if "parts_poses" in obs[0]
        else np.array([], dtype=np.float32)
    )

    # TODO: Make sure this is rectified in the controller-end and
    # Clip xyz delta position actions to ±0.025
    action_delta_quat[:, :3] = np.clip(action_delta_quat[:, :3], -0.025, 0.025)

    # figure out what to do with the corrupted raw data
    # For now, clip the z-axis rotation to 0.35
    action_delta_quat[:, 3:7] = clip_quat_xyzw_magnitude(
        action_delta_quat[:, 3:7], clip_mag=0.35
    )

    # Take the sign of the gripper action
    action_delta_quat[:, -1] = np.sign(action_delta_quat[:, -1])

    # Calculate the position actions
    if calculate_pos_action_from_delta:
        action_pos = np.concatenate(
            [
                robot_state_quat[:, :3] + action_delta_quat[:, :3],
                np_apply_quat(robot_state_quat[:, 3:7], action_delta_quat[:, 3:7]),
                # Append the gripper action
                action_delta_quat[:, -1:],
            ],
            axis=1,
        )
        action_pos_6d = np_action_quat_to_6d_rotation(action_pos)

    else:
        raise NotImplementedError(
            "This script only supports calculating position actions from delta actions."
        )

    # Convert delta action to use 6D rotation
    action_delta_6d = np_action_quat_to_6d_rotation(action_delta_quat)

    # Extract the rewards from the pickle file
    reward = (
        np.array(data["rewards"], dtype=np.float32)
        if "rewards" in data
        else np.zeros(len(action_delta_6d))
    )
    reward = reward[:len(action_delta_6d)]

    # Use observation-level skill labels as the authoritative source.
    skill = np.zeros((len(obs), len(SKILL_ORDER)), dtype=np.float32)
    for idx, observation in enumerate(obs):
        skill_label = observation.get("skill")
        if skill_label is None:
            continue
        if isinstance(skill_label, bytes):
            skill_label = skill_label.decode("utf-8")
        if skill_label not in SKILL_TO_ONEHOT:
            raise ValueError(
                f"Unknown skill label {skill_label!r} in {pickle_path}. "
                f"Expected one of {SKILL_ORDER}."
            )
        skill[idx] = SKILL_TO_ONEHOT[skill_label]

    augment_states = (
        data["augment_states"] if "augment_states" in data else np.zeros_like(reward)
    )

    # Sanity check that all arrays are the same length
    assert len(robot_state_6d) == len(
        action_delta_6d
    ), f"Mismatch in {pickle_path}, lengths differ by {len(robot_state_6d) - len(action_delta_6d)}"

    assert len(reward) == len(
        action_delta_6d
    ), f"Reward mismatch in {pickle_path}, lengths differ by {len(reward) - len(action_delta_6d)}"

    # Extract the pickle file name as the path after `raw` in the path
    pickle_file = "/".join(pickle_path.parts[pickle_path.parts.index("raw") + 1 :])

    task = data.get("task", data.get("furniture"))

    processed_data = {
        "robot_state": robot_state_6d,
        "color_image1": color_image1,
        "color_image2": color_image2,
        "depth_image1": depth_image1,
        "depth_image2": depth_image2,
        "action/delta": action_delta_6d,
        "action/pos": action_pos_6d,
        "reward": reward,
        "skill": skill,
        "augment_states": augment_states,
        "parts_poses": parts_poses,
        "episode_length": len(action_delta_6d),
        "task": task,
        "success": 1 if data["success"] == "partial_success" else int(data["success"]),
        "pickle_file": pickle_file,
    }

    return processed_data


def parallel_process_pickle_files(
    pickle_paths,
    noop_threshold,
    num_threads,
    calculate_pos_action_from_delta=False,
    resize_image=False,
):
    """
    Process all pickle files in parallel and aggregate results.
    """
    # Initialize empty data structures to hold aggregated data
    aggregated_data = {
        "robot_state": [],
        "color_image1": [],
        "color_image2": [],
        "depth_image1": [],
        "depth_image2": [],
        "action/delta": [],
        "action/pos": [],
        "reward": [],
        "skill": [],
        "augment_states": [],
        "parts_poses": [],
        "episode_ends": [],
        "task": [],
        "success": [],
        "pickle_file": [],
    }

    def aggregate_data(data):
        for key in data:
            if key == "episode_length":
                # Calculate and append to episode_ends
                last_end = (
                    aggregated_data["episode_ends"][-1]
                    if len(aggregated_data["episode_ends"]) > 0
                    else 0
                )
                aggregated_data["episode_ends"].append(last_end + data[key])
            else:
                aggregated_data[key].append(data[key])

    if num_threads == 1:
        # Run synchronous version
        for path in tqdm(pickle_paths, desc="Processing files"):
            data = process_pickle_file(
                path, noop_threshold, calculate_pos_action_from_delta, resize_image
            )
            aggregate_data(data)
    else:
        # Run threaded version
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(
                    process_pickle_file,
                    path,
                    noop_threshold,
                    calculate_pos_action_from_delta,
                    resize_image,
                )
                for path in pickle_paths
            ]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing files"
            ):
                data = future.result()
                aggregate_data(data)

    # Convert lists to numpy arrays for numerical data
    for key in tqdm(
        [
            "robot_state",
            "color_image1",
            "color_image2",
            "depth_image1",
            "depth_image2",
            "action/delta",
            "action/pos",
            "reward",
            "skill",
            "parts_poses",
            "augment_states",
        ],
        desc="Converting lists to numpy arrays",
    ):
        aggregated_data[key] = np.concatenate(aggregated_data[key])

    return aggregated_data


def write_to_zarr_store(z, key, value):
    """
    Function to write data to a Zarr store.
    """
    z[key][:] = value


def parallel_write_to_zarr(z, aggregated_data, num_threads):
    """
    Write aggregated data to the Zarr store in parallel.
    """
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for key, value in aggregated_data.items():
            # Schedule the writing of each dataset
            futures.append(executor.submit(write_to_zarr_store, z, key, value))

        # Wait for all futures to complete and track progress
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Writing to Zarr store"
        ):
            future.result()


# === Entry Point of the Script ===
if __name__ == "__main__":
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
        required=True,
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
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
    )
    parser.add_argument("--output-suffix", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--randomize-order", action="store_true")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--n-cpus", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--resize-image", action="store_true", help="Resize images to standard dimensions (240x320x3)")
    parser.add_argument("--input-dir", type=str, help="Path to the directory containing pkl files", default=None)
    parser.add_argument("--output-dir", type=str, help="Path to save the zarr file", default=None)
    args = parser.parse_args()

    assert not args.randomize_order or args.offset == 0, "Cannot offset with randomize"

    if args.input_dir is not None:
        input_dir = Path(args.input_dir).expanduser().resolve()
        if not input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        pickle_paths: List[Path] = sorted(input_dir.rglob("*.pkl*"))
        print(f"Using explicit input directory: {input_dir}")
    else:
        pickle_paths = sorted(
            get_raw_paths(
                controller=args.controller,
                domain=args.domain,
                task=args.task,
                demo_source=args.source,
                randomness=args.randomness,
                demo_outcome=args.demo_outcome,
                suffix=args.suffix,
            )
        )

    # Output the shape of the first pickle file
    total_files = len(pickle_paths)
    if total_files > 0:
        first_pickle_data = unpickle_data(pickle_paths[0])
        print("[INFO] Shape of the first pickle file's data:")
        for key, value in first_pickle_data.items():
            if key == "success" or key == "task" or key == "action_type":
                print(f"{key}: {value} (type: {type(value)})")
            elif key == "rewards" or key == "actions":
                print(f"{key}: shape {np.shape(value)}")
            elif key == "observations":
                print(f"{key}: number of observations {len(value)}")
                if len(value) > 0:
                    for obs_key, obs_value in value[0].items():
                        if obs_key == "robot_state":
                            for sub_key, sub_value in obs_value.items():
                                print(f"  robot_state/{sub_key}: shape {np.shape(sub_value)}")
                        elif isinstance(obs_value, np.ndarray):
                            print(f"  {obs_key}: shape {obs_value.shape}")
                        else:
                            print(f"  {obs_key}: type {type(obs_value)}")
            else:
                print("[WARNING] No pickle files found for the specified criteria.")

    if args.randomize_order:
        print(f"Using random seed: {args.random_seed}")
        random.seed(args.random_seed)
        random.shuffle(pickle_paths)
    start = args.offset
    end = (
        args.offset + args.max_files
        if args.max_files is not None
        else len(pickle_paths)
    )
    pickle_paths = pickle_paths[start:end]

    print(f"Found {len(pickle_paths)} pickle files")

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
        )

    print(f"Output path: {output_path}")

    if output_path.exists() and not args.overwrite:
        raise ValueError(
            f"Output path already exists: {output_path}. Use --overwrite to overwrite."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process all pickle files
    chunksize = args.chunk_size
    noop_threshold = 0.0
    n_cpus = min(os.cpu_count(), args.n_cpus)
    batch_size = args.batch_size

    # If batch processing requested, do a lightweight scan first to determine total shapes
    if batch_size > 0 and batch_size < len(pickle_paths):
        print(f"[INFO] Using batch processing with batch_size={batch_size}")
        total_timesteps = 0
        episode_lengths = []
        tasks_meta = []
        successes_meta = []
        pickle_files_meta = []
        # For determining per-step dims
        img_shape = None
        parts_pose_dim = None
        robot_state_dim = None

        for p in tqdm(pickle_paths, desc="Scanning pickle files for shapes"):
            data = unpickle_data(p)

            obs = data["observations"]
            actions = data["actions"]
            # Adjust for possible extra last obs
            if len(obs) == len(actions) + 1:
                obs = obs[:-1]
            ep_len = len(actions)
            episode_lengths.append(ep_len)
            total_timesteps += ep_len
            tasks_meta.append(data.get("task", data.get("furniture")))
            successes_meta.append(1 if data.get("success") == "partial_success" else int(data.get("success", 0)))
            pickle_file_rel = "/".join(p.parts[p.parts.index("raw") + 1 :]) if "raw" in p.parts else p.name
            pickle_files_meta.append(pickle_file_rel)
            if robot_state_dim is None:
                rs = obs[0]["robot_state"]
                if isinstance(rs, dict):
                    rs_vec = filter_and_concat_robot_state(rs)
                else:
                    rs_vec = rs
                robot_state_dim = rs_vec.shape[-1] + 2  # will become 6D rotation later but safe placeholder
            if parts_pose_dim is None and "parts_poses" in obs[0]:
                parts_pose_dim = len(obs[0]["parts_poses"])
            if img_shape is None and "color_image1" in obs[0]:
                img_shape = obs[0]["color_image1"].shape
        if parts_pose_dim is None:
            parts_pose_dim = 0
        if img_shape is None:
            # No images in this dataset (state-only); set dummy shape (total_timesteps,0,0,0)
            img_shape = (0, 0, 0)

        # Build full_data_shapes for zarr initialization
        # Process the first pickle file to determine data shapes
        sample_data = process_pickle_file(
            pickle_paths[0],
            noop_threshold=0.0,
            calculate_pos_action_from_delta=True,
            resize_image=args.resize_image,
        )

        # Define full_data_shapes based on the sample data and total_timesteps
        full_data_shapes = [
            ("robot_state", (total_timesteps,) + sample_data["robot_state"].shape[1:], np.float32),
            ("color_image1", (total_timesteps,) + sample_data["color_image1"].shape[1:], np.uint8),
            ("color_image2", (total_timesteps,) + sample_data["color_image2"].shape[1:], np.uint8),
            ("depth_image1", (total_timesteps,) + sample_data["depth_image1"].shape[1:], np.float32),
            ("depth_image2", (total_timesteps,) + sample_data["depth_image2"].shape[1:], np.float32),
            ("action/delta", (total_timesteps,) + sample_data["action/delta"].shape[1:], np.float32),
            ("action/pos", (total_timesteps,) + sample_data["action/pos"].shape[1:], np.float32),
            ("parts_poses", (total_timesteps,) + sample_data["parts_poses"].shape[1:], np.float32),
            ("reward", (total_timesteps,), np.float32),
            ("skill", (total_timesteps,) + sample_data["skill"].shape[1:], np.float32),
            ("augment_states", (total_timesteps,), np.float32),
            ("episode_ends", (len(episode_lengths),), np.uint32),
            ("task", (len(episode_lengths),), str),
            ("success", (len(episode_lengths),), np.uint8),
            ("pickle_file", (len(episode_lengths),), str),
        ]

        # Output the full_data_shapes for debugging or inspection
        print("Full data shapes:")
        for name, shape, dtype in full_data_shapes:
            print(f"{name}: shape={shape}, dtype={dtype}")
        sys.stdout.flush()  # Ensure the output is flushed immediately after printing

        # Initialize zarr store early
        z = initialize_zarr_store(output_path, full_data_shapes, chunksize=chunksize)

        # Fill episode-level metadata arrays
        cum_end = 0
        episode_ends_arr = []
        for L in episode_lengths:
            cum_end += L
            episode_ends_arr.append(cum_end)
        z["episode_ends"][:] = np.array(episode_ends_arr, dtype=np.uint32)
        z["task"][:] = np.array(tasks_meta, dtype=object)
        z["success"][:] = np.array(successes_meta, dtype=np.uint8)
        z["pickle_file"][:] = np.array(pickle_files_meta, dtype=object)

        # Process in batches and write slices
        write_ptr = 0
        for start_i in range(0, len(pickle_paths), batch_size):
            batch_paths = pickle_paths[start_i : start_i + batch_size]
            batch_timeseries = {
                "robot_state": [],
                "color_image1": [],
                "color_image2": [],
                "depth_image1": [],
                "depth_image2": [],
                "action/delta": [],
                "action/pos": [],
                "reward": [],
                "skill": [],
                "augment_states": [],
                "parts_poses": [],
            }
            for p in batch_paths:
                data = process_pickle_file(
                    p,
                    noop_threshold=0.0,
                    calculate_pos_action_from_delta=True,
                    resize_image=args.resize_image,
                )
                for k in batch_timeseries.keys():
                    batch_timeseries[k].append(data[k])
            # Concatenate this batch
            for k in batch_timeseries.keys():
                if batch_timeseries[k]:
                    batch_timeseries[k] = np.concatenate(batch_timeseries[k])
                else:
                    # Create empty array with correct trailing dims
                    if k in ["robot_state", "action/delta", "action/pos", "parts_poses", "skill"]:
                        batch_timeseries[k] = np.empty((0, batch_timeseries[k][0].shape[1] if batch_timeseries[k] else 0), dtype=np.float32)
                    else:
                        batch_timeseries[k] = np.empty((0,), dtype=np.float32)
            batch_len = batch_timeseries["action/delta"].shape[0]
            end_ptr = write_ptr + batch_len
            # Write slice to zarr
            for k, arr in batch_timeseries.items():
                z[k][write_ptr:end_ptr] = arr
            write_ptr = end_ptr
            print(f"[INFO] Written batch {start_i//batch_size + 1}, timesteps so far: {write_ptr}/{total_timesteps}")

        # Update metadata attrs and exit early (skip original full aggregation path)
        z.attrs["time_finished"] = datetime.now().astimezone().isoformat()
        z.attrs["noop_threshold"] = 0.0
        z.attrs["chunksize"] = chunksize
        z.attrs["rotation_mode"] = "rot_6d"
        z.attrs["n_episodes"] = len(z["episode_ends"])
        z.attrs["n_timesteps"] = total_timesteps
        z.attrs["mean_episode_length"] = round(total_timesteps / len(z["episode_ends"]))
        z.attrs["calculated_pos_action_from_delta"] = True
        z.attrs["randomize_order"] = args.randomize_order
        z.attrs["random_seed"] = args.random_seed
        z.attrs["demo_source"] = args.source
        z.attrs["controller"] = args.controller
        z.attrs["domain"] = args.domain if args.domain == "real" else "sim"
        z.attrs["task"] = args.task
        z.attrs["randomness"] = args.randomness
        z.attrs["demo_outcome"] = args.demo_outcome
        z.attrs["suffix"] = args.suffix
        print("[INFO] Batch processing complete.")
        exit(0)

    print(
        f"Processing pickle files with {n_cpus} CPUs, chunksize={chunksize}, noop_threshold={noop_threshold}\n"
        f"randomize_order={args.randomize_order}, random_seed={args.random_seed}\n"
        f"from file nr. {start} to {end} out of {total_files}"
    )

    all_data = parallel_process_pickle_files(
        pickle_paths,
        noop_threshold,
        n_cpus,
        calculate_pos_action_from_delta=True,
        resize_image=args.resize_image,
    )

    # Define the full shapes for each dataset
    full_data_shapes = [
        # These are of length: number of timesteps
        ("robot_state", all_data["robot_state"].shape, np.float32),
        ("color_image1", all_data["color_image1"].shape, np.uint8),
        ("color_image2", all_data["color_image2"].shape, np.uint8),
        ("depth_image1", all_data["depth_image1"].shape, np.float32),
        ("depth_image2", all_data["depth_image2"].shape, np.float32),
        ("action/delta", all_data["action/delta"].shape, np.float32),
        ("action/pos", all_data["action/pos"].shape, np.float32),
        ("parts_poses", all_data["parts_poses"].shape, np.float32),
        ("reward", all_data["reward"].shape, np.float32),
        ("skill", all_data["skill"].shape, np.float32),
        ("augment_states", all_data["augment_states"].shape, np.float32),
        # These are of length: number of episodes
        ("episode_ends", (len(all_data["episode_ends"]),), np.uint32),
        ("task", (len(all_data["task"]),), str),
        ("success", (len(all_data["success"]),), np.uint8),
        ("pickle_file", (len(all_data["pickle_file"]),), str),
    ]

    # Output the full_data_shapes for debugging or inspection
    print("Full data shapes:")
    for name, shape, dtype in full_data_shapes:
        print(f"{name}: shape={shape}, dtype={dtype}")
    
    
    # Initialize Zarr store with full dimensions
    z = initialize_zarr_store(output_path, full_data_shapes, chunksize=chunksize)

    # Write the data to the Zarr store
    it = tqdm(all_data)
    for name in it:
        it.set_description(f"Writing data to zarr: {name}")
        dataset = z[name]
        data = all_data[name]

        for i in trange(0, len(data), chunksize, desc="Writing chunks", leave=False):
            dataset[i : i + chunksize] = data[i : i + chunksize]

    # Update final metadata
    z.attrs["time_finished"] = datetime.now().astimezone().isoformat()
    z.attrs["noop_threshold"] = noop_threshold
    z.attrs["chunksize"] = chunksize
    z.attrs["rotation_mode"] = "rot_6d"
    z.attrs["n_episodes"] = len(z["episode_ends"])
    z.attrs["n_timesteps"] = len(z["action/delta"])
    z.attrs["mean_episode_length"] = round(
        len(z["action/delta"]) / len(z["episode_ends"])
    )
    z.attrs["calculated_pos_action_from_delta"] = True
    z.attrs["randomize_order"] = args.randomize_order
    z.attrs["random_seed"] = args.random_seed
    z.attrs["demo_source"] = args.source
    z.attrs["controller"] = args.controller
    z.attrs["domain"] = args.domain if args.domain == "real" else "sim"
    z.attrs["task"] = args.task
    z.attrs["randomness"] = args.randomness
    z.attrs["demo_outcome"] = args.demo_outcome
    z.attrs["suffix"] = args.suffix
