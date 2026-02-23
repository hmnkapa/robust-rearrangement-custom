from datetime import datetime
from typing import List
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.visualization.render_mp4 import pickle_data
from src.common.types import Trajectory, Observation
from src.common.geometry import np_action_6d_to_quat

from ipdb import set_trace as bp
from src.visualization.render_mp4 import create_in_memory_mp4, depth2heatmap


def save_raw_rollout(
    robot_states: np.ndarray,
    imgs1: np.ndarray,
    imgs2: np.ndarray,
    depth_image1: np.ndarray,
    depth_image2: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    parts_poses: np.ndarray,
    success: bool,
    task: str,
    action_type: str,
    rollout_save_dir: Path,
    compress_pickles: bool = False,
    have_img_obs:  bool = False,
    have_depth_obs: bool = False,
    pcs: List[np.ndarray] = None,
):
    observations: List[Observation] = list()

    # If pcs is None, create a list of Nones with the same length as robot_states
    if pcs is None:
        pcs = [None] * len(robot_states)

    for robot_state, image1, image2, depth1, depth2, parts_pose, pc in zip(
        robot_states, imgs1, imgs2, depth_image1, depth_image2, parts_poses, pcs
    ):
        observations.append(
            {
                "robot_state": robot_state,
                "color_image1": image1,
                "color_image2": image2,
                "depth_image1": depth1,
                "depth_image2": depth2,
                "parts_poses": parts_pose,
                "point_cloud": pc,
            }
        )

    if action_type == "pos":

        if actions.shape[1] == 10:
            # If we've used rot_6d convert to quat
            actions = np_action_6d_to_quat(actions)
        
        assert actions.shape[1] == 8

        # Get the action quat
        pos_action_quat = R.from_quat(actions[:, 3:7])

        # Get the position quat from the robot state
        pos_quat = R.from_quat([rs["ee_quat"] for rs in robot_states[:-1]])

        # The action quat was calculated as pos_quat * action_quat
        # Calculate the delta quat between the pos_quat and the action_quat
        delta_action_quat = pos_quat.inv() * pos_action_quat

        # Also calculate the delta position
        delta_action_pos = actions[:, :3] - np.array(
            [rs["ee_pos"] for rs in robot_states[:-1]]
        )

        # Insert the delta quat into the actions
        actions = np.concatenate(
            [delta_action_pos, delta_action_quat.as_quat(), actions[:, -1:]], axis=1
        )

    data: Trajectory = {
        "observations": observations,
        "actions": actions.tolist(),
        "rewards": rewards.tolist(),
        "success": success,
        "task": task,
        "action_type": action_type,
    }

    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
    output_path = rollout_save_dir / ("success" if success else "failure")
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path / f"{timestamp}.pkl"

    if compress_pickles:
        output_path = output_path.with_suffix(".pkl.xz")

    pickle_data(data, output_path)


    # Additionally save MP4 videos for video1 and video2
    if have_img_obs:
        # Ensure output directory exists (with success/failure subdirectory)
        status_dir = Path(rollout_save_dir) / ("success" if success else "failure")
        status_dir.mkdir(parents=True, exist_ok=True)

        # Create MP4 bytes for each camera stream
        mp4_cam1 = create_in_memory_mp4(imgs1, fps=20)
        mp4_cam2 = create_in_memory_mp4(imgs2, fps=20)

        # Build filenames
        cam1_path = status_dir / f"{timestamp}_cam1.mp4"
        cam2_path = status_dir / f"{timestamp}_cam2.mp4"

        # Write files
        with open(cam1_path, "wb") as f1:
            f1.write(mp4_cam1.getvalue() if hasattr(mp4_cam1, "getvalue") else mp4_cam1)
        with open(cam2_path, "wb") as f2:
            f2.write(mp4_cam2.getvalue() if hasattr(mp4_cam2, "getvalue") else mp4_cam2)

    # Additionally save depth videos as MP4 for video1 and video2
    if have_depth_obs:
        # Ensure output directory exists (with success/failure subdirectory)
        status_dir = Path(rollout_save_dir) / ("success" if success else "failure")
        status_dir.mkdir(parents=True, exist_ok=True)

        # Create MP4 bytes for each camera stream
        depth1_heatmap_frames = depth2heatmap(depth_image1)
        depth2_heatmap_frames = depth2heatmap(depth_image2)

        mp4_dep1 = create_in_memory_mp4(depth1_heatmap_frames, fps=20)
        mp4_dep2 = create_in_memory_mp4(depth2_heatmap_frames, fps=20)

        # Build filenames
        dep1_path = status_dir / f"{timestamp}_dep1.mp4"
        dep2_path = status_dir / f"{timestamp}_dep2.mp4"

        # Write files
        with open(dep1_path, "wb") as f1:
            f1.write(mp4_dep1.getvalue() if hasattr(mp4_dep1, "getvalue") else mp4_dep2)
        with open(dep2_path, "wb") as f2:
            f2.write(mp4_dep2.getvalue() if hasattr(mp4_dep2, "getvalue") else mp4_dep2)