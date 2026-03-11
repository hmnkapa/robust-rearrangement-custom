#!/usr/bin/env python3

import argparse
import pickle
import shutil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R
import zarr
from tqdm import tqdm

from src.common.geometry import (
    np_rot_6d_to_isaac_quat,
    np_action_6d_to_quat,
    np_apply_quat,
)


def _build_default_paths(
    task: str,
    randomness: str,
    demo_source: str,
    env: str,
    processed_root: Path,
    raw_root: Path,
) -> Tuple[Path, Path]:
    input_zarr = (
        processed_root
        / "diffik"
        / env
        / task
        / demo_source
        / randomness
        / "success.zarr"
    )
    output_dir = (
        raw_root
        / "diffik"
        / env
        / task
        / demo_source
        / randomness
        / "success"
    )
    return input_zarr, output_dir


def _maybe_get_group(z: zarr.hierarchy.Group, name: str) -> Optional[zarr.hierarchy.Group]:
    try:
        g = z[name]
        if isinstance(g, zarr.hierarchy.Group):
            return g
    except Exception:
        return None
    return None


def _maybe_get_array(z: zarr.hierarchy.Group, name: str):
    try:
        arr = z[name]
        if isinstance(arr, zarr.core.Array):
            return arr
    except Exception:
        return None
    return None


def _split_robot_state(robot_state_vec: np.ndarray):
    # After process_pickle, robot_state is 16D:
    # [ee_pos(3), ee_rot6d(6), ee_pos_vel(3), ee_ori_vel(3), gripper_width(1)]
    # Some legacy datasets might store 14D:
    # [ee_pos(3), ee_quat(4), ee_pos_vel(3), ee_ori_vel(3), gripper_width(1)]
    dim = robot_state_vec.shape[-1]
    if dim == 16:
        ee_pos = robot_state_vec[..., 0:3]
        ee_rot_6d = robot_state_vec[..., 3:9]
        ee_pos_vel = robot_state_vec[..., 9:12]
        ee_ori_vel = robot_state_vec[..., 12:15]
        gripper_width = robot_state_vec[..., 15:16]
        ee_quat = np_rot_6d_to_isaac_quat(ee_rot_6d)
        return ee_pos, ee_quat, ee_pos_vel, ee_ori_vel, gripper_width
    if dim == 14:
        ee_pos = robot_state_vec[..., 0:3]
        ee_quat = robot_state_vec[..., 3:7]
        ee_pos_vel = robot_state_vec[..., 7:10]
        ee_ori_vel = robot_state_vec[..., 10:13]
        gripper_width = robot_state_vec[..., 13:14]
        return ee_pos, ee_quat, ee_pos_vel, ee_ori_vel, gripper_width
    raise ValueError(f"Unexpected robot_state dimension: {dim}")


def _actions_from_6d(action_6d: np.ndarray) -> np.ndarray:
    if action_6d.shape[-1] == 10:
        return np_action_6d_to_quat(action_6d)
    if action_6d.shape[-1] == 8:
        return action_6d
    raise ValueError(f"Unexpected action dimension: {action_6d.shape[-1]}")


def _pos_actions_from_delta_6d(
    action_delta_6d: np.ndarray, robot_state_vec: np.ndarray
) -> np.ndarray:
    ee_pos, ee_quat, _ee_pos_vel, _ee_ori_vel, _gripper_width = _split_robot_state(
        robot_state_vec
    )
    delta_pos = action_delta_6d[..., :3]
    delta_rot_6d = action_delta_6d[..., 3:9]
    delta_gripper = action_delta_6d[..., 9:10]

    delta_quat = np_rot_6d_to_isaac_quat(delta_rot_6d)
    new_quat = np_apply_quat(ee_quat, delta_quat)

    return np.concatenate([ee_pos + delta_pos, new_quat, delta_gripper], axis=-1)


def _delta_actions_from_pos_6d(
    action_pos_6d: np.ndarray, robot_state_vec: np.ndarray
) -> np.ndarray:
    ee_pos, ee_quat, _ee_pos_vel, _ee_ori_vel, _gripper_width = _split_robot_state(
        robot_state_vec
    )
    action_pos_quat = np_action_6d_to_quat(action_pos_6d)

    pos_action_quat = R.from_quat(action_pos_quat[:, 3:7])
    pos_quat = R.from_quat(ee_quat)
    delta_action_quat = (pos_quat.inv() * pos_action_quat).as_quat()
    delta_action_pos = action_pos_quat[:, :3] - ee_pos

    return np.concatenate(
        [delta_action_pos, delta_action_quat, action_pos_quat[:, -1:]], axis=1
    )


def zarr_to_pickles(
    input_zarr: Path,
    output_dir: Path,
    pad_last_observation: bool = True,
    action_type_out: str = "pos",
    if_exist: str = "overwrite",
):
    z = zarr.open_group(str(input_zarr), mode="r")

    episode_ends = _maybe_get_array(z, "episode_ends")
    if episode_ends is None:
        raise ValueError("Missing episode_ends in zarr")
    episode_ends = episode_ends[:]

    task_arr = _maybe_get_array(z, "task")
    success_arr = _maybe_get_array(z, "success")
    pickle_file_arr = _maybe_get_array(z, "pickle_file")

    action_group = _maybe_get_group(z, "action")
    if action_group is None:
        raise ValueError("Missing action group in zarr")

    if action_type_out == "pos":
        action_key = "pos"
    elif action_type_out == "delta":
        action_key = "delta"
    else:
        raise ValueError(f"Unsupported action_type_out: {action_type_out}")

    if action_key not in action_group:
        raise ValueError(f"Action group missing '{action_key}'")

    action_arr = action_group[action_key]
    reward_arr = _maybe_get_array(z, "reward")

    color_image1 = _maybe_get_array(z, "color_image1")
    color_image2 = _maybe_get_array(z, "color_image2")
    parts_poses = _maybe_get_array(z, "parts_poses")
    robot_state = _maybe_get_array(z, "robot_state")

    if color_image1 is None or color_image2 is None:
        raise ValueError("Missing color_image1 or color_image2 in zarr")
    if robot_state is None:
        raise ValueError("Missing robot_state in zarr")

    if if_exist == "overwrite" and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prev_end = 0
    for ep_idx, end in enumerate(tqdm(episode_ends, desc="episodes")):
        end = int(end)
        start = int(prev_end)
        prev_end = end

        # Slice episode data
        ci1 = color_image1[start:end]
        ci2 = color_image2[start:end]
        pp = parts_poses[start:end] if parts_poses is not None else None
        rs = robot_state[start:end]
        actions_6d = action_arr[start:end]
        rewards = reward_arr[start:end] if reward_arr is not None else None

        ee_pos, ee_quat, ee_pos_vel, ee_ori_vel, gripper_width = _split_robot_state(rs)
        gripper_f1 = gripper_width / 2.0
        gripper_f2 = gripper_width / 2.0

        # process_pickle only keeps a 16D proprioceptive subset, so joint_* values are not recoverable.
        # Fill with zeros to satisfy expected raw-pickle schema.
        joint_positions = np.zeros((len(rs), 7), dtype=np.float32)
        joint_velocities = np.zeros((len(rs), 7), dtype=np.float32)
        joint_torques = np.zeros((len(rs), 9), dtype=np.float32)

        obs_list = []
        for i in range(len(rs)):
            obs = {
                "color_image1": ci1[i],
                "color_image2": ci2[i],
                "parts_poses": pp[i] if pp is not None else np.zeros((0,), dtype=np.float32),
                "robot_state": {
                    "ee_ori_vel": ee_ori_vel[i],
                    "ee_pos": ee_pos[i],
                    "ee_pos_vel": ee_pos_vel[i],
                    "ee_quat": ee_quat[i],
                    "gripper_finger_1_pos": gripper_f1[i],
                    "gripper_finger_2_pos": gripper_f2[i],
                    "gripper_width": gripper_width[i],
                    "joint_positions": joint_positions[i],
                    "joint_torques": joint_torques[i],
                    "joint_velocities": joint_velocities[i],
                },
            }
            obs_list.append(obs)

        if pad_last_observation and obs_list:
            obs_list.append(obs_list[-1])

        if action_type_out == "pos":
            # Match eval_model/save_raw_rollout behavior: store delta actions even when action_type is pos.
            actions = _delta_actions_from_pos_6d(actions_6d, rs)
        elif action_type_out == "delta":
            # Store delta actions directly.
            actions = _actions_from_6d(actions_6d)
        else:
            raise ValueError(f"Unsupported action_type_out: {action_type_out}")

        data = {
            "observations": obs_list,
            "actions": actions,
            "rewards": rewards if rewards is not None else np.zeros(len(actions), dtype=np.float32),
            "success": bool(success_arr[ep_idx]) if success_arr is not None else True,
            "task": str(task_arr[ep_idx]) if task_arr is not None else None,
            "action_type": action_type_out,
        }

        if pickle_file_arr is not None:
            name = Path(str(pickle_file_arr[ep_idx])).name
        else:
            name = f"episode_{ep_idx:04d}.pkl"
        if not name.endswith(".pkl"):
            name = name + ".pkl"
        out_path = output_dir / name

        if out_path.exists():
            if if_exist == "append":
                stem = out_path.stem
                suffix = out_path.suffix
                idx = 1
                while True:
                    candidate = out_path.with_name(f"{stem}_{idx}{suffix}")
                    if not candidate.exists():
                        out_path = candidate
                        break
                    idx += 1
            elif if_exist != "overwrite":
                raise ValueError(f"Unsupported if_exist: {if_exist}")

        with out_path.open("wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser(description="Extract per-episode pickles from a diffik success.zarr.")
    parser.add_argument("--task", type=str, help="Task name, e.g., one_leg")
    parser.add_argument("--randomness", type=str, help="Randomness level, e.g., low")
    parser.add_argument("--demo-source", type=str, default="teleop", help="demo source: teleop|rollout")
    parser.add_argument("--env", type=str, default="sim", help="environment: sim|real")
    parser.add_argument(
        "--input-zarr",
        type=str,
        default=None,
        help="Path to success.zarr. If omitted, derived from task/randomness.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for pickles. If omitted, derived from task/randomness.",
    )
    parser.add_argument(
        "--processed-root",
        type=str,
        default="/data/hy/robust-rearrangement/data/processed",
        help="Base processed root for default path building.",
    )
    parser.add_argument(
        "--raw-root",
        type=str,
        default="/data/hy/robust-rearrangement/raw/raw",
        help="Base raw root for default path building.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Deprecated. Same as --if-exist overwrite.",
    )
    parser.add_argument(
        "--no-pad-last-observation",
        action="store_true",
        help="Do not append the last observation to make obs=len(actions)+1.",
    )
    parser.add_argument(
        "--action-type",
        type=str,
        default="pos",
        choices=["pos", "delta"],
        help="Action type to save in output pickle.",
    )
    parser.add_argument(
        "--if-exist",
        type=str,
        default="overwrite",
        choices=["overwrite", "append"],
        help="What to do if output file already exists.",
    )

    args = parser.parse_args()

    if args.input_zarr is None or args.output_dir is None:
        if not args.task or not args.randomness:
            raise ValueError("Provide --input-zarr/--output-dir or --task/--randomness")
        input_zarr, output_dir = _build_default_paths(
            args.task,
            args.randomness,
            args.demo_source,
            args.env,
            Path(args.processed_root),
            Path(args.raw_root),
        )
    else:
        input_zarr = Path(args.input_zarr)
        output_dir = Path(args.output_dir)

    if not input_zarr.exists():
        raise FileNotFoundError(f"Input zarr not found: {input_zarr}")

    zarr_to_pickles(
        input_zarr=input_zarr,
        output_dir=output_dir,
        pad_last_observation=not args.no_pad_last_observation,
        action_type_out=args.action_type,
        if_exist="overwrite" if args.overwrite else args.if_exist,
    )


if __name__ == "__main__":
    main()
