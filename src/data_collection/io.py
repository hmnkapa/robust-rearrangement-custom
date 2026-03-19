from datetime import datetime
from typing import List
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.visualization.render_mp4 import pickle_data
from src.common.types import Trajectory, Observation
from src.common.geometry import np_action_6d_to_quat

from ipdb import set_trace as bp
from src.visualization.render_mp4 import (
    create_in_memory_mp4,
    depth2heatmap,
    analyze_depth_smoothness,
)
from src.eval.skill_annotation_util import draw_skill_on_image


def _write_depth_smoothness_report(report_path: Path, camera_name: str, smoothness: dict):
    lines = [
        f"camera={camera_name}",
        f"depth_sign_mode={smoothness.get('depth_sign_mode', 'as_is')}",
        f"valid_pixel_ratio_global={smoothness.get('valid_pixel_ratio_global', 0.0):.6f}",
        f"global_min_p1={smoothness['global_min']:.6f}",
        f"global_max_p99={smoothness['global_max']:.6f}",
        f"jump_threshold_p95={smoothness['threshold']:.6f}",
        f"jump_frames={smoothness['n_jumps']}/{max(smoothness['n_frames'] - 1, 0)}",
        "frame_idx,valid_ratio,depth_mean,depth_p95,delta_mean,delta_p95,delta_max,status",
    ]
    for row in smoothness["per_frame"]:
        status = "JUMP" if row["is_jump"] else "OK"
        lines.append(
            f"{row['frame']},{row['valid_ratio']:.6f},{row['depth_mean']:.6f},"
            f"{row['depth_p95']:.6f},{row['delta_mean']:.6f},{row['delta_p95']:.6f},"
            f"{row['delta_max']:.6f},{status}"
        )
    report_path.write_text("\n".join(lines) + "\n")


def save_raw_rollout(
    robot_states: np.ndarray,
    imgs1: np.ndarray,
    imgs2: np.ndarray,
    depth_image1: np.ndarray,
    depth_image2: np.ndarray,
    skills: List[str],
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
    skill_on_image: bool = False,
):
    observations: List[Observation] = list()

    # If pcs is None, create a list of Nones with the same length as robot_states
    if pcs is None:
        pcs = [None] * len(robot_states)

    if skills is None:
        skills = [None] * len(robot_states)

    for robot_state, image1, image2, depth1, depth2, parts_pose, pc, skill in zip(
        robot_states, imgs1, imgs2, depth_image1, depth_image2, parts_poses, pcs, skills
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
                "skill": skill,
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

        imgs2_for_video = imgs2.copy()
        if skill_on_image:
            n_annotated = min(len(imgs2_for_video), len(skills))
            for frame_idx in range(n_annotated):
                skill = skills[frame_idx]
                if skill is None:
                    continue
                imgs2_for_video[frame_idx] = draw_skill_on_image(
                    imgs2_for_video[frame_idx], skill
                )

        # Create MP4 bytes for each camera stream
        mp4_cam1 = create_in_memory_mp4(imgs1, fps=20)
        mp4_cam2 = create_in_memory_mp4(imgs2_for_video, fps=20)

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

        smooth1 = analyze_depth_smoothness(depth_image1)
        smooth2 = analyze_depth_smoothness(depth_image2)
        report1_path = status_dir / f"{timestamp}_dep1_smoothness.txt"
        report2_path = status_dir / f"{timestamp}_dep2_smoothness.txt"
        _write_depth_smoothness_report(report1_path, "depth_image1", smooth1)
        _write_depth_smoothness_report(report2_path, "depth_image2", smooth2)
        print(
            f"[DepthSmoothness] dep1 jump_frames={smooth1['n_jumps']}/"
            f"{max(smooth1['n_frames'] - 1, 0)}, threshold={smooth1['threshold']:.6f}, "
            f"report={report1_path}"
        )
        print(
            f"[DepthSmoothness] dep2 jump_frames={smooth2['n_jumps']}/"
            f"{max(smooth2['n_frames'] - 1, 0)}, threshold={smooth2['threshold']:.6f}, "
            f"report={report2_path}"
        )

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
            f1.write(mp4_dep1.getvalue() if hasattr(mp4_dep1, "getvalue") else mp4_dep1)
        with open(dep2_path, "wb") as f2:
            f2.write(mp4_dep2.getvalue() if hasattr(mp4_dep2, "getvalue") else mp4_dep2)
