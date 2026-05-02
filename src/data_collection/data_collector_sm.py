"""Define data collection class that rollout the environment, get action from the interface (e.g., teleoperation, automatic scripts), and save data."""

import json
import os
import pickle
import select
import struct
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Union, List, Dict

import cv2
from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv
from furniture_bench.envs.furniture_rl_sim_env import FurnitureRLSimEnv
from furniture_bench.envs.observation import FULL_OBS

# from src.gym.furniture_sim_env import FurnitureSimEnv, FurnitureRLSimEnv
import gym
from src.gym import turn_off_april_tags
import torch
from tqdm import tqdm, trange
from ipdb import set_trace as bp

from furniture_bench.device.device_interface import DeviceInterface
from furniture_bench.sim_config import sim_config
from src.data_processing.utils import resize, resize_crop
from furniture_bench.envs.initialization_mode import Randomness
from furniture_bench.utils.scripted_demo_mod import scale_scripted_action
from src.visualization.render_mp4 import pickle_data, unpickle_data


from src.data_collection.collect_enum import CollectEnum

import time
from multiprocessing.managers import SharedMemoryManager

import numpy as np
import scipy.spatial.transform as st
from furniture_bench.device.spacemouse.spacemouse_shared_memory import Spacemouse

from src.common.files import SCAN_ASSET_FB_ROOT


def precise_wait(t_end: float, slack_time: float = 0.001, time_func=time.monotonic):
    t_start = time_func()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time_func() < t_end:
            pass
    return


class DataCollectorSpaceMouse:
    """Demonstration collection class.
    `pkl` files have resized images while `mp4` / `png` files save raw camera inputs.
    """

    DEFAULT_SM_POS_SPEED = 0.54
    DEFAULT_SM_ROT_SPEED = 2.8

    env: FurnitureSimEnv

    def __init__(
        self,
        data_path: str,
        device_interface: DeviceInterface,
        furniture: str,
        draw_marker: bool,
        randomness: Randomness.LOW,
        compute_device_id: int,
        graphics_device_id: int,
        save_failure: bool = False,
        num_demos: int = 100,
        resize_sim_img: bool = True,
        verbose: bool = True,
        show_pbar: bool = False,
        ctrl_mode: str = "diffik",
        ee_laser: bool = True,
        right_multiply_rot: bool = True,
        compress_pickles: bool = False,
        resume_trajectory_paths: Union[List[str], None] = None,
        sample_perturbations: bool = False,
        sm_pos_speed: float = None,
        sm_rot_speed: float = None,
        teleop_setting: int = 1,
        show_teleop_cameras: bool = False,
    ):
        """
        Args:
            data_path (str): Path to save data.
            device_interface (DeviceInterface): Keyboard and/or Oculus interface.
            furniture (str): Name of the furniture.
            draw_marker (bool): Whether to draw AprilTag marker.
            manual_label (bool): Whether to manually label the reward.
            scripted (bool): Whether to use scripted function for getting action.
            randomness (str): Initialization randomness level.
            gpu_id (int): GPU ID.
            save_failure (bool): Whether to save failure trajectories.
            num_demos (int): The maximum number of demonstrations to collect in this run. Internal loop will be terminated when this number is reached.
            ctrl_mode (str): 'osc' (joint torque, with operation space control) or 'diffik' (joint impedance, with differential inverse kinematics control)
            ee_laser (bool): If True, show a line coming from the end-effector in the viewer
            right_multiply_rot (bool): If True, convert rotation actions (delta rot) assuming they're applied as RIGHT multiplys (local rotations)
            sm_pos_speed (float): SpaceMouse max translational speed in meters per second. Default matches the old src 0.3 speed with 1.8x scalar.
            sm_rot_speed (float): SpaceMouse max rotational speed in radians per second. Default matches the old src 0.7 speed with 4x scalar.
            teleop_setting (int): 1 matches existing src world-frame position/rotation control. 2 matches furniture-bench end-effector-frame control.
            show_teleop_cameras (bool): If True, show an OpenCV preview window for teleoperation cameras.
        """
        assert ctrl_mode == "diffik", "Highly suspicious that this is not 'diffik'"

        self.teleop_preview_stdin = None
        self.teleop_preview_stdout = None
        self.teleop_preview_process = None
        self.teleop_preview_frame_count = 0

        if not draw_marker:
            turn_off_april_tags()

        print("[NB] Creating 4 envs for debugging purposes")
        obs_keys = list(FULL_OBS)
        for depth_key in ("depth_image1", "depth_image2"):
            if depth_key not in obs_keys:
                obs_keys.append(depth_key)
        self.env = FurnitureRLSimEnv(
            furniture=furniture,
            obs_keys=obs_keys,
            headless=False,
            max_env_steps=3_000,  # Arbitrary number
            num_envs=1,
            act_rot_repr="quat",
            action_type="delta",
            manual_done=True,
            resize_img=resize_sim_img,
            np_step_out=False,
            channel_first=False,
            randomness=randomness,
            compute_device_id=compute_device_id,
            graphics_device_id=graphics_device_id,
            ctrl_mode=ctrl_mode,
            ee_laser=ee_laser,
        )

        self.data_path = Path(data_path)
        self.device_interface = device_interface
        self.furniture = furniture
        self.num_demos = num_demos

        self.traj_counter = 0
        self.num_success = 0
        self.num_fail = 0

        self.save_failure = save_failure
        self.resize_sim_img = resize_sim_img
        self.compress_pickles = compress_pickles
        self.resume_trajectory_paths = resume_trajectory_paths
        self.sample_perturbations = sample_perturbations
        if sm_pos_speed is not None and sm_pos_speed <= 0:
            raise ValueError("sm_pos_speed must be positive.")
        if sm_rot_speed is not None and sm_rot_speed <= 0:
            raise ValueError("sm_rot_speed must be positive.")
        if teleop_setting not in (1, 2):
            raise ValueError("teleop_setting must be 1 or 2.")
        self.sm_pos_speed = (
            self.DEFAULT_SM_POS_SPEED if sm_pos_speed is None else sm_pos_speed
        )
        self.sm_rot_speed = (
            self.DEFAULT_SM_ROT_SPEED if sm_rot_speed is None else sm_rot_speed
        )
        self.teleop_setting = teleop_setting
        if teleop_setting == 1:
            self.sm_pos_frame = "world"
            self.sm_rot_frame = "world"
            self.ee_rot_sign = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        else:
            self.sm_pos_frame = "ee"
            self.sm_rot_frame = "ee"
            self.ee_rot_sign = np.array([1.0, -1.0, -1.0], dtype=np.float32)
        self.show_teleop_cameras = show_teleop_cameras
        self.teleop_camera_keys = ("color_image2", "color_image1")
        self.teleop_camera_labels = (
            "Fixed camera (color_image2)",
            "Wrist camera (color_image1)",
        )
        self.teleop_display_scale = 4.0
        self.teleop_window_name = "Teleop Camera View"
        self.teleop_window_scale = 0.3
        self.teleop_window_pos = (0, 0)

        self.iter_idx = 0

        self.pos_bounds_m = 0.025
        self.ori_bounds_deg = 20

        self.verbose = verbose
        self.pbar = None if not show_pbar else tqdm(total=self.num_demos)

        # Set the number of timesteps we will continue to record actions after a gripper action is taken
        # This is taken up from 10 because we saw some "teleportation" of the gripper when we were recording
        self.record_latency_when_grasping = 8

        # Parameters for controlling the time it takes for the robot to settle at the start of a trajectory
        self.start_delay = 2  # seconds
        self.robot_settled = False
        self.starttime = datetime.now()

        # Variable controlling if we're currently in recording mode
        self.recording = True

        # our flags
        self.right_multiply_rot = right_multiply_rot

        self._reset_collector_buffer()

        self.metadata = {
            "max_force_magnitude": self.env.max_force_magnitude,
            "max_torque_magnitude": self.env.max_torque_magnitude,
            "max_obstacle_offset": self.env.max_obstacle_offset,
            "franka_joint_rand_lim_deg": self.env.franka_joint_rand_lim_deg,
            "ctrl_mode": ctrl_mode,
            "pos_bounds_m": self.pos_bounds_m,
            "ori_bounds_deg": self.ori_bounds_deg,
            "record_latency_when_grasping": self.record_latency_when_grasping,
            "start_delay": self.start_delay,
            "right_multiply_rot": self.right_multiply_rot,
            "sample_perturbations": self.sample_perturbations,
            "max_pos_speed": self.sm_pos_speed,
            "max_rot_speed": self.sm_rot_speed,
            "teleop_setting": self.teleop_setting,
            "sm_pos_frame": self.sm_pos_frame,
            "sm_rot_frame": self.sm_rot_frame,
        }

    def _squeeze_and_numpy(
        self, d: Dict[str, Union[torch.Tensor, np.ndarray, float, int, None]]
    ):
        """
        Recursively squeeze and convert tensors to numpy arrays
        Convert scalars to floats
        Leave NoneTypes alone
        """
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = self._squeeze_and_numpy(v)

            elif v is None:
                continue

            elif isinstance(v, (torch.Tensor, np.ndarray)):
                if isinstance(v, torch.Tensor):
                    v = v.cpu().numpy()
                d[k] = v.squeeze()

            elif k == "rewards":
                d[k] = float(v)
            elif k == "skills":
                d[k] = int(v)
            else:
                raise ValueError(f"Unsupported type: {type(v)}")

        return d

    def _camera_image_to_rgb(self, img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        img = np.asarray(img).squeeze()
        if img.ndim != 3:
            return None
        if img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
            img = np.moveaxis(img, 0, -1)
        if img.shape[-1] == 4:
            img = img[..., :3]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def _start_teleop_camera_preview(self):
        if self.teleop_preview_process is not None:
            return

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self.teleop_preview_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "src.data_collection.teleop_camera_preview",
                "--window-name",
                self.teleop_window_name,
                "--window-scale",
                str(self.teleop_window_scale),
                "--window-x",
                str(self.teleop_window_pos[0]),
                "--window-y",
                str(self.teleop_window_pos[1]),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        self.teleop_preview_stdin = self.teleop_preview_process.stdin
        self.teleop_preview_stdout = self.teleop_preview_process.stdout
        self.teleop_preview_frame_count = 0

    def _stop_teleop_camera_preview(self):
        if self.teleop_preview_stdin is not None:
            try:
                self.teleop_preview_stdin.write(struct.pack("!I", 0))
                self.teleop_preview_stdin.flush()
            except Exception:
                pass
            try:
                self.teleop_preview_stdin.close()
            except Exception:
                pass
        if self.teleop_preview_process is not None:
            try:
                self.teleop_preview_process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.teleop_preview_process.kill()
                self.teleop_preview_process.wait(timeout=1.0)
        self.teleop_preview_stdin = None
        self.teleop_preview_stdout = None
        self.teleop_preview_process = None
        self.teleop_preview_frame_count = 0

    def _show_teleop_cameras(self, obs):
        if not self.show_teleop_cameras:
            return

        frames = []
        for key, label in zip(self.teleop_camera_keys, self.teleop_camera_labels):
            if key not in obs:
                continue
            img = self._camera_image_to_rgb(obs[key])
            if img is None:
                continue
            img = img.copy()
            cv2.putText(
                img,
                label,
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            frames.append(img)

        if not frames:
            return

        min_h = min(frame.shape[0] for frame in frames)
        display_h = int(min_h * self.teleop_display_scale)
        frames = [
            cv2.resize(
                frame,
                (int(frame.shape[1] * display_h / frame.shape[0]), display_h),
            )
            for frame in frames
        ]
        img = np.hstack(frames)

        if self.teleop_preview_process is None:
            self._start_teleop_camera_preview()

        try:
            payload = pickle.dumps(img, protocol=pickle.HIGHEST_PROTOCOL)
            self.teleop_preview_stdin.write(struct.pack("!I", len(payload)))
            self.teleop_preview_stdin.write(payload)
            self.teleop_preview_stdin.flush()
        except Exception as exc:
            self.verbose_print(
                f"[data collection] Failed to send teleop camera frame, disabling preview: {exc}"
            )
            self.show_teleop_cameras = False
            self._stop_teleop_camera_preview()
            return

        if self.teleop_preview_stdout is None:
            return

        ack_timeout = 4.0 if self.teleop_preview_frame_count == 0 else 1.0
        deadline = time.monotonic() + ack_timeout
        ack = None
        while time.monotonic() < deadline:
            timeout = max(0.0, deadline - time.monotonic())
            ready, _, _ = select.select([self.teleop_preview_stdout], [], [], timeout)
            if not ready:
                break

            line = self.teleop_preview_stdout.readline()
            if not line:
                break
            message = line.decode("utf-8", errors="replace").strip()
            if message == "ok":
                ack = "ok"
                break
            if message.startswith("error:"):
                ack = message
                break

        if ack != "ok":
            if ack is None:
                self.verbose_print(
                    "[data collection] Teleop camera preview did not respond, disabling it."
                )
            else:
                self.verbose_print(
                    f"[data collection] Teleop camera preview failed, disabling it: {ack}"
                )
            self.show_teleop_cameras = False
            self._stop_teleop_camera_preview()
            return

        self.teleop_preview_frame_count += 1

    def _resize_depth_obs(self, depth_image, crop=False):
        if depth_image is None:
            return None

        if isinstance(depth_image, torch.Tensor):
            depth = depth_image
            if depth.ndim == 2:
                depth = depth.unsqueeze(0)
            if depth.ndim == 3:
                depth = depth.unsqueeze(-1)
            resized = resize_crop(depth) if crop else resize(depth)
            return resized.squeeze(-1)

        if isinstance(depth_image, np.ndarray):
            depth = depth_image
            if depth.ndim == 2:
                depth = depth[None, ...]
            if depth.ndim == 3:
                depth = depth[..., None]
            resized = resize_crop(depth) if crop else resize(depth)
            return np.squeeze(resized, axis=-1)

        raise ValueError(f"Unsupported depth image type: {type(depth_image)}")

    def collect(self):
        self.verbose_print("[data collection] Start collecting the data!")

        from collections import namedtuple

        args = namedtuple(
            "Args",
            [
                "frequency",
                "command_latency",
                "deadzone",
                "max_pos_speed",
                "max_rot_speed",
            ],
        )

        args.frequency = 10
        args.command_latency = 0.01
        args.deadzone = 0.05
        args.max_pos_speed = self.sm_pos_speed
        args.max_rot_speed = self.sm_rot_speed

        frequency = args.frequency
        dt = 1 / frequency
        command_latency = args.command_latency
        pos_bounds_m = max(self.pos_bounds_m, args.max_pos_speed / frequency)
        ori_bounds_deg = max(
            self.ori_bounds_deg, np.rad2deg(args.max_rot_speed / frequency)
        )

        # Add these to the metadata
        self.metadata["frequency"] = frequency
        self.metadata["command_latency"] = command_latency
        self.metadata["deadzone"] = args.deadzone
        self.metadata["max_pos_speed"] = args.max_pos_speed
        self.metadata["max_rot_speed"] = args.max_rot_speed
        self.metadata["pos_bounds_m"] = pos_bounds_m
        self.metadata["ori_bounds_deg"] = ori_bounds_deg
        self.verbose_print(
            "[data collection] SpaceMouse limits: "
            f"frequency={frequency}Hz, "
            f"max_pos_speed={args.max_pos_speed:.3f}m/s, "
            f"max_rot_speed={args.max_rot_speed:.3f}rad/s, "
            f"teleop_setting={self.teleop_setting}, "
            f"pos_frame={self.sm_pos_frame}, "
            f"rot_frame={self.sm_rot_frame}, "
            f"pos_step_bound={pos_bounds_m:.4f}m, "
            f"rot_step_bound={ori_bounds_deg:.2f}deg"
        )

        obs = self.reset()
        next_obs = obs
        done = False

        target_pose_rv, gripper_width, gripper_open, grasp_flag = self.set_target_pose()

        def pose_rv2mat(pose_rv):
            pose_mat = np.eye(4)
            pose_mat[:-1, -1] = pose_rv[:3]
            pose_mat[:-1, :-1] = st.Rotation.from_rotvec(pose_rv[3:]).as_matrix()
            return pose_mat

        def to_isaac_dpose_from_abs(
            current_pose_mat, goal_pose_mat, grasp_flag, device, rm=True
        ):
            """
            Convert from absolute current and desired pose to delta pose

            Args:
                rm (bool): 'rm' stands for 'right multiplication' - If True, assume commands send as right multiply (local rotations)
            """
            if rm:
                delta_rot_mat = (
                    np.linalg.inv(current_pose_mat[:-1, :-1]) @ goal_pose_mat[:-1, :-1]
                )
            else:
                delta_rot_mat = goal_pose_mat[:-1:-1] @ np.linalg.inv(
                    current_pose_mat[:-1, :-1]
                )

            dpos = goal_pose_mat[:-1, -1] - current_pose_mat[:-1, -1]
            target_translation = torch.from_numpy(dpos).float().to(device)

            target_rot = st.Rotation.from_matrix(delta_rot_mat)
            target_quat_xyzw = torch.from_numpy(target_rot.as_quat()).float().to(device)
            target_dpose = torch.cat(
                (target_translation, target_quat_xyzw, grasp_flag), dim=-1
            ).reshape(1, -1)
            return target_dpose

        target_pose_last_action_rv = None
        ready_to_grasp = True
        steps_since_grasp = 0

        with SharedMemoryManager() as shm_manager:
            with Spacemouse(shm_manager=shm_manager, deadzone=args.deadzone) as sm:
                t_start = time.monotonic()
                self.iter_idx = 0

                prev_keyboard_gripper = -1
                global_start_time = time.time()
                while self.num_success < self.num_demos:

                    # calculate timing
                    t_cycle_end = t_start + (self.iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    precise_wait(t_sample)

                    # get teleop command
                    sm_state = sm.get_motion_state_transformed()
                    dpos = sm_state[:3] * (args.max_pos_speed / frequency)
                    drot_xyz = sm_state[3:] * (args.max_rot_speed / frequency)
                    if self.sm_rot_frame == "ee":
                        drot_xyz *= self.ee_rot_sign
                    drot = st.Rotation.from_euler("xyz", drot_xyz)

                    (
                        keyboard_action,
                        collect_enum,
                    ) = self.device_interface.get_action()  # from the keyboard

                    if collect_enum == CollectEnum.PAUSE:
                        self.recording = False
                        self.verbose_print("Paused recording")
                    elif collect_enum == CollectEnum.CONTINUE:
                        self.recording = True
                        self.verbose_print("Continued recording")

                    # If undo action is taken, undo the last 10 actions.
                    if collect_enum == CollectEnum.UNDO:
                        # Go back in time by removing transitions from the buffer and setting simulator state
                        self.undo_actions()

                        # Set the right target poses for the state after undoing actions
                        (
                            target_pose_rv,
                            gripper_width,
                            gripper_open,
                            grasp_flag,
                        ) = self.set_target_pose()
                        target_pose_last_action_rv = None

                        continue

                    if np.allclose(dpos, 0.0) and np.allclose(drot_xyz, 0.0):
                        action_taken = False
                        if target_pose_last_action_rv is None:
                            translation, quat_xyzw = self.env.get_ee_pose()
                            translation, quat_xyzw = (
                                translation.cpu().numpy().squeeze(),
                                quat_xyzw.cpu().numpy().squeeze(),
                            )
                            rotvec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
                            target_pose_last_action_rv = np.array(
                                [*translation, *rotvec]
                            )
                    else:
                        action_taken = True
                        target_pose_last_action_rv = None

                    steps_since_grasp += 1
                    if steps_since_grasp > self.record_latency_when_grasping:
                        ready_to_grasp = True
                    if steps_since_grasp < self.record_latency_when_grasping:
                        action_taken = True

                    kb_grasp = prev_keyboard_gripper != keyboard_action[-1]
                    sm_grasp = (
                        sm.is_button_pressed(0) or sm.is_button_pressed(1)
                    ) and ready_to_grasp
                    if kb_grasp or sm_grasp:
                        # env.gripper_close() if gripper_open else env.gripper_open()
                        grasp_flag = -1 * grasp_flag
                        gripper_open = not gripper_open

                        ready_to_grasp = False
                        steps_since_grasp = 0
                    prev_keyboard_gripper = keyboard_action[-1]

                    target_rot = st.Rotation.from_rotvec(target_pose_rv[3:])
                    if self.sm_pos_frame == "ee":
                        dpos[[1, 2]] *= -1
                        dpos = target_rot.apply(dpos)

                    new_target_pose_rv = target_pose_rv.copy()
                    new_target_pose_rv[:3] += dpos
                    if self.sm_rot_frame == "world":
                        new_target_rot = drot * target_rot
                    elif self.sm_rot_frame == "ee":
                        new_target_rot = target_rot * drot
                    else:
                        raise ValueError(f"Invalid sm_rot_frame: {self.sm_rot_frame}")
                    new_target_pose_rv[3:] = new_target_rot.as_rotvec()

                    target_pose_mat = pose_rv2mat(target_pose_rv)
                    if target_pose_last_action_rv is not None:
                        new_target_pose_mat = pose_rv2mat(target_pose_last_action_rv)
                    else:
                        new_target_pose_mat = pose_rv2mat(new_target_pose_rv)

                    # convert this into the furniture bench info we need
                    action = to_isaac_dpose_from_abs(
                        current_pose_mat=target_pose_mat,
                        goal_pose_mat=new_target_pose_mat,
                        grasp_flag=grasp_flag,
                        device=self.env.device,
                        rm=self.right_multiply_rot,
                    )

                    if not (np.allclose(keyboard_action[:6], 0.0)):
                        action[0, :7] = (
                            torch.from_numpy(keyboard_action[:7])
                            .float()
                            .to(action.device)
                        )
                        action_taken = True
                        target_pose_last_action_rv = None

                    action = scale_scripted_action(
                        action.detach().cpu().clone(),
                        pos_bounds_m=pos_bounds_m,
                        ori_bounds_deg=ori_bounds_deg,
                        device=self.env.device,
                    )

                    skill_complete = int(collect_enum == CollectEnum.SKILL)
                    if skill_complete == 1:
                        self.skill_set.append(skill_complete)

                    if collect_enum == CollectEnum.TERMINATE:
                        self.verbose_print("Terminate the program.")
                        break

                    # An episode is done.
                    if done or collect_enum in [
                        CollectEnum.SUCCESS,
                        CollectEnum.SUCCESS_RECORD,
                        CollectEnum.FAIL,
                    ]:
                        global_total_time = time.time() - global_start_time
                        print(f"Time elapsed: {global_total_time} seconds.")
                        self.store_transition(next_obs)

                        if (
                            done and not self.env.furnitures[0].all_assembled()
                        ) or collect_enum is CollectEnum.FAIL:
                            collect_enum = CollectEnum.FAIL
                            if self.save_failure:
                                self.verbose_print("Saving failure trajectory.")
                                obs = self.save_and_reset(collect_enum, {})
                            else:
                                self.verbose_print(
                                    "Failed to assemble the furniture, reset without saving."
                                )
                                obs = self.reset()
                            self.num_fail += 1
                        else:
                            # if done:
                            #     collect_enum = CollectEnum.SUCCESS
                            print(f"CollectEnum: {collect_enum}")
                            obs = self.save_and_reset(collect_enum, {})
                            self.num_success += 1
                            self.update_pbar()

                        self.traj_counter += 1
                        self.verbose_print(
                            f"Success: {self.num_success}, Fail: {self.num_fail}"
                        )

                        next_obs = obs
                        done = False

                        steps_since_grasp = 0
                        ready_to_grasp = True
                        target_pose_last_action_rv = None
                        prev_keyboard_gripper = -1
                        self.iter_idx = 0
                        t_start = time.monotonic()
                        (
                            target_pose_rv,
                            gripper_width,
                            gripper_open,
                            grasp_flag,
                        ) = self.set_target_pose()

                        continue

                    # Execute action.
                    # if action_taken: # We can do this if we want to apply random perturbations when demonstrating
                    next_obs, rew, done, info = self.env.step(
                        action,
                        sample_perturbations=action_taken and self.sample_perturbations,
                    )
                    self._show_teleop_cameras(next_obs)

                    if rew == 1:
                        self.last_reward_idx = len(self.transitions)

                    # Error handling.
                    if not info["obs_success"]:
                        self.verbose_print(
                            "Getting observation failed, save trajectory."
                        )
                        obs = self.save_and_reset(CollectEnum.FAIL, info)
                        next_obs = obs
                        done = False
                        steps_since_grasp = 0
                        ready_to_grasp = True
                        target_pose_last_action_rv = None
                        prev_keyboard_gripper = -1
                        self.iter_idx = 0
                        t_start = time.monotonic()
                        (
                            target_pose_rv,
                            gripper_width,
                            gripper_open,
                            grasp_flag,
                        ) = self.set_target_pose()
                        continue

                    # Logging a step.
                    if action_taken:
                        # Store a transition.
                        if info["action_success"]:
                            self.store_transition(obs, action, rew, skill_complete)

                            # Intrinsic rotation
                            translation, quat_xyzw = self.env.get_ee_pose()
                            translation, quat_xyzw = (
                                translation.cpu().numpy().squeeze(),
                                quat_xyzw.cpu().numpy().squeeze(),
                            )

                    obs = next_obs

                    # target_pose = new_target_pose
                    translation, quat_xyzw = self.env.get_ee_pose()
                    translation, quat_xyzw = (
                        translation.cpu().numpy().squeeze(),
                        quat_xyzw.cpu().numpy().squeeze(),
                    )
                    rotvec = st.Rotation.from_quat(quat_xyzw).as_rotvec()

                    target_pose_rv = np.array([*translation, *rotvec])

                    # SM wait
                    precise_wait(t_cycle_end)
                    self.iter_idx += 1

                    if (not self.robot_settled) and (
                        (datetime.now() - self.starttime).seconds > self.start_delay
                    ):
                        self.robot_settled = True
                        print("Robot settled")

                self.verbose_print(
                    f"Collected {self.traj_counter} / {self.num_demos} successful trajectories!"
                )

    def set_target_pose(self):
        translation, quat_xyzw = self.env.get_ee_pose()
        translation, quat_xyzw = (
            translation.cpu().numpy().squeeze(),
            quat_xyzw.cpu().numpy().squeeze(),
        )
        gripper_width = self.env.gripper_width()
        rotvec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
        target_pose_rv = np.array([*translation, *rotvec])
        if isinstance(gripper_width, torch.Tensor):
            gripper_open = bool((gripper_width >= 0.05).view(-1)[0].item())
        else:
            gripper_open = bool(gripper_width >= 0.05)
        grasp_flag = torch.tensor(
            [-1 if gripper_open else 1], dtype=torch.float32, device=self.env.device
        )

        return target_pose_rv, gripper_width, gripper_open, grasp_flag

    def undo_actions(self):
        self.verbose_print("Undo the last 10 actions.")
        if not self.transitions:
            self.verbose_print("Nothing to undo.")
            return

        # Remove the last 10 transitions from the buffer but keep at least one
        self.transitions = self.transitions[:1] + self.transitions[1:-10]

        # Set the environment to the state before the last 10 actions.
        self.env.reset_env_to(env_idx=0, state=self.transitions[-1]["observations"])
        self.env.refresh()

    def store_transition(
        self, obs, action=None, rew=None, skill_complete=None, setup_phase=False
    ):
        """Store the observation, action, and reward."""
        if (not setup_phase) and ((not self.robot_settled) or (not self.recording)):
            # Don't store anything until the robot has settled
            # Without this, we get ~8 useless actions at the start of every trajectory
            return

        # We want to resize the images while tensors for maximum compatibility with the rest of the code
        n_ob = {}
        n_ob["color_image1"] = resize(obs["color_image1"])
        n_ob["color_image2"] = resize_crop(obs["color_image2"])
        n_ob["depth_image1"] = self._resize_depth_obs(obs.get("depth_image1"))
        n_ob["depth_image2"] = self._resize_depth_obs(
            obs.get("depth_image2"), crop=True
        )
        n_ob["robot_state"] = obs["robot_state"]
        n_ob["parts_poses"] = obs["parts_poses"]

        if action is not None:
            if isinstance(action, torch.Tensor):
                action = action.squeeze().cpu().numpy()
            elif isinstance(action, np.ndarray):
                action = action.squeeze()
            else:
                raise ValueError(f"Unsupported action type: {type(action)}")

        if rew is not None:
            if isinstance(rew, torch.Tensor):
                rew = rew.item()
            elif isinstance(rew, np.ndarray):
                rew = rew.item()
            elif isinstance(rew, float):
                rew = rew
            elif isinstance(rew, int):
                rew = float(rew)

        transition = {
            "observations": n_ob,
            "actions": action,
            "rewards": rew,
            "skills": skill_complete,
        }

        # Treat the whole transition as a dictionary, and squeeze all the tensors and make scalars into floats
        transition = self._squeeze_and_numpy(transition)
        self.transitions.append(transition)

        # We'll update the steps counter whenever we store an observation
        if not setup_phase:
            print(
                f"{[self.step_counter]} assembled: {self.env.furniture.assembled_set} "
                f"num assembled: {len(self.env.furniture.assembled_set)} "
                f"Skill: {len(self.skill_set)}."
            )

    @property
    def step_counter(self):
        return len(self.transitions)

    def save_and_reset(self, collect_enum: CollectEnum, info):
        """Saves the collected data and reset the environment."""
        self.save(collect_enum, info)
        self.verbose_print(f"Saved {self.traj_counter} trajectories in this run.")
        return self.reset()

    def reset(self):
        self.verbose_print("[data collection] Resetting environment...", flush=True)
        obs = self.env.reset()
        self._reset_collector_buffer()
        self._show_teleop_cameras(obs)

        if self.resume_trajectory_paths:
            obs = self.load_state()

        self.verbose_print("Start collecting the data!")

        self.starttime = datetime.now()
        self.robot_settled = False
        self.recording = True
        return obs

    def _reset_collector_buffer(self):
        # Now, observations, actions, rewards, and skall_complete flags are stored as transition "tuples"
        self.transitions = []

        self.last_reward_idx = -1
        self.skill_set = []

    def load_state(self):
        """
        Load the state of the environment from a one_leg trajectory
        from the currently first pickle in the resume_trajectory_paths list
        """

        # Get the state dict at the end of a one_leg trajectory
        trajectory_path = self.resume_trajectory_paths.pop(0)
        print("Loading state from:")
        print(trajectory_path)

        state = unpickle_data(trajectory_path)

        self.env.reset_env_to(env_idx=0, state=state["observations"][-1])
        self.env.refresh()

        # Add all the data so far in the trajectory to the collect buffer
        for i in trange(len(state["observations"]), desc="Hydrating state"):
            self.store_transition(
                obs={
                    "color_image1": np.array(state["observations"][i]["color_image1"]),
                    "color_image2": np.array(state["observations"][i]["color_image2"]),
                    "depth_image1": state["observations"][i].get("depth_image1"),
                    "depth_image2": state["observations"][i].get("depth_image2"),
                    "robot_state": state["observations"][i]["robot_state"],
                    "parts_poses": np.array(state["observations"][i]["parts_poses"]),
                },
                action=state["actions"][i] if i < len(state["actions"]) else None,
                rew=state["rewards"][i] if i < len(state["rewards"]) else None,
                skill_complete=state["skills"][i] if i < len(state["skills"]) else None,
                setup_phase=True,
            )

        return self.transitions.pop()["observations"]

    def save(self, collect_enum: CollectEnum, info):
        print(f"Length of trajectory: {len(self.transitions)}")

        # Save transitions with resized images.
        data = {}
        data["observations"] = [t["observations"] for t in self.transitions]
        data["actions"] = [t["actions"] for t in self.transitions][:-1]
        data["rewards"] = [t["rewards"] for t in self.transitions][:-1]
        data["skills"] = [t["skills"] for t in self.transitions][:-1]
        data["success"] = (
            True
            if collect_enum in [CollectEnum.SUCCESS, CollectEnum.SUCCESS_RECORD]
            else False
        )
        data["furniture"] = self.furniture
        data["metadata"] = self.metadata

        if "error" in info:
            data["error_description"] = info["error"].value
            data["error"] = True
        else:
            data["error"] = False
            data["error_description"] = ""

        # Save data.
        demo_path = self.data_path / ("success" if data["success"] else "failure")
        demo_path.mkdir(parents=True, exist_ok=True)

        path = demo_path / f"{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.pkl"

        if self.compress_pickles:
            # Add the suffix .gz if we are compressing the pickle files
            path = path.with_suffix(".pkl.xz")

        pickle_data(data, path)

        print(f"Data saved at {path}")

        if collect_enum == CollectEnum.SUCCESS_RECORD:
            # Save the final poses as rewards to use
            # print(f"Here to get part poses for reward computation")
            # from IPython import embed

            # embed()
            # assert False

            furn_name = self.env.furniture_name
            base_assembly_file = str(
                SCAN_ASSET_FB_ROOT
                / sim_config["furniture"][furn_name]["assembly_json_fname"]
            )
            assembly_pair_idxs = self.env.pairs_to_assemble
            for i, pair_tuple in enumerate(assembly_pair_idxs):
                ref_idx = pair_tuple[0]
                moved_idx = pair_tuple[1]
                ref_pose_final = data["observations"][-1]["parts_poses"][
                    int(7 * ref_idx) : int(7 * ref_idx) + 7
                ]
                moved_pose_final = data["observations"][-1]["parts_poses"][
                    int(7 * moved_idx) : int(7 * moved_idx) + 7
                ]

                # write these poses to file
                suffix = path.parts[-1].replace(".pkl", f"_{i}.json")
                json_fname = base_assembly_file.replace(".json", f"_{suffix}")
                json_dict = self._build_assembly_json_dict(
                    ref_pose_final, moved_pose_final
                )
                print(f"Saving to json file: {json_fname}")
                with open(json_fname, "w") as f:
                    json.dump(json_dict, f)

    def verbose_print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def update_pbar(self):
        if self.pbar is not None:
            self.pbar.update(1)

    def __del__(self):
        self._stop_teleop_camera_preview()
        del self.env

        if self.device_interface is not None:
            self.device_interface.close()

    def _build_assembly_json_dict(self, ref_pose, moved_pose):
        def pose2mat(pose):
            pose_mat = np.eye(4)
            pose_mat[:-1, :-1] = st.Rotation.from_quat(pose[3:7]).as_matrix()
            pose_mat[:-1, -1] = pose[:3]
            return pose_mat

        ref_pose_mat = pose2mat(ref_pose)
        moved_pose_mat = pose2mat(moved_pose)

        json_dict = dict(
            data=dict(
                reference=dict(pose=ref_pose_mat.reshape(-1).tolist()),
                moved=dict(pose=moved_pose_mat.reshape(-1).tolist()),
            ),
        )
        return json_dict
