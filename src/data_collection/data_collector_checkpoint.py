"""Data collector with checkpoint/resume (断点续采) support.

Extends DataCollectorSpaceMouse to allow saving a checkpoint mid-trajectory
and later restoring the full scene state (robot joints, gripper, parts poses)
to that checkpoint.
"""

from datetime import datetime

import numpy as np
import scipy.spatial.transform as st
import torch

from src.data_collection.data_collector_sm import DataCollectorSpaceMouse
from src.data_collection.collect_enum import CollectEnum


class DataCollectorCheckpoint(DataCollectorSpaceMouse):
    """Data collector that supports checkpoint/resume during teleop."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.checkpoint_state = None
        self.checkpoint_idx = None

    def set_checkpoint(self):
        """Save the current environment state as a checkpoint."""
        if not self.transitions:
            print("[checkpoint] No transitions to checkpoint.")
            return
        self.checkpoint_state = self.transitions[-1]["observations"]
        self.checkpoint_idx = len(self.transitions) - 1
        print(f"[checkpoint] Checkpoint set at step {self.checkpoint_idx}")

    def resume_checkpoint(self):
        """Restore environment to the last checkpoint state.

        Resets robot joints, gripper, and all furniture part poses,
        then trims the transition buffer to match.
        """
        if self.checkpoint_state is None:
            print("[checkpoint] No checkpoint to resume from.")
            return

        self.transitions = self.transitions[: self.checkpoint_idx + 1]
        self.env.reset_env_to(env_idx=0, state=self.checkpoint_state)
        self.env.refresh()
        self.starttime = datetime.now()
        self.robot_settled = False
        print(f"[checkpoint] Resumed to checkpoint at step {self.checkpoint_idx}")

    def collect(self):
        """Main collection loop with checkpoint/resume support.

        Overrides the parent collect() to handle checkpoint_pressed and
        resume_pressed flags from KeyboardInterfaceCheckpoint.
        """
        from collections import namedtuple

        args = namedtuple(
            "Args",
            ["frequency", "command_latency", "deadzone", "max_pos_speed", "max_rot_speed"],
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
            if rm:
                delta_rot_mat = (
                    np.linalg.inv(current_pose_mat[:-1, :-1]) @ goal_pose_mat[:-1, :-1]
                )
            else:
                delta_rot_mat = goal_pose_mat[:-1, :-1] @ np.linalg.inv(
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

        import time as time_module
        from furniture_bench.device.spacemouse.spacemouse_shared_memory import Spacemouse
        from multiprocessing.managers import SharedMemoryManager
        from src.data_collection.data_collector_sm import precise_wait
        from furniture_bench.utils.scripted_demo_mod import scale_scripted_action

        with SharedMemoryManager() as shm_manager:
            with Spacemouse(shm_manager=shm_manager, deadzone=args.deadzone) as sm:
                t_start = time_module.monotonic()
                self.iter_idx = 0

                prev_keyboard_gripper = -1
                global_start_time = time_module.time()
                while self.num_success < self.num_demos:

                    t_cycle_end = t_start + (self.iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    precise_wait(t_sample)

                    sm_state = sm.get_motion_state_transformed()
                    dpos = sm_state[:3] * (args.max_pos_speed / frequency)
                    drot_xyz = sm_state[3:] * (args.max_rot_speed / frequency)
                    if self.sm_rot_frame == "ee":
                        drot_xyz *= self.ee_rot_sign
                    drot = st.Rotation.from_euler("xyz", drot_xyz)

                    (
                        keyboard_action,
                        collect_enum,
                    ) = self.device_interface.get_action()

                    # --- Checkpoint / Resume handling ---
                    if hasattr(self.device_interface, "checkpoint_pressed") and self.device_interface.checkpoint_pressed:
                        self.device_interface.checkpoint_pressed = False
                        self.set_checkpoint()
                        continue

                    if hasattr(self.device_interface, "resume_pressed") and self.device_interface.resume_pressed:
                        self.device_interface.resume_pressed = False
                        self.resume_checkpoint()
                        target_pose_rv, gripper_width, gripper_open, grasp_flag = self.set_target_pose()
                        target_pose_last_action_rv = None
                        continue
                    # -----------------------------------

                    if collect_enum == CollectEnum.PAUSE:
                        self.recording = False
                        self.verbose_print("Paused recording")
                    elif collect_enum == CollectEnum.CONTINUE:
                        self.recording = True
                        self.verbose_print("Continued recording")

                    if collect_enum == CollectEnum.UNDO:
                        self.undo_actions()
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

                    if done or collect_enum in [
                        CollectEnum.SUCCESS,
                        CollectEnum.SUCCESS_RECORD,
                        CollectEnum.FAIL,
                    ]:
                        global_total_time = time_module.time() - global_start_time
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
                        t_start = time_module.monotonic()
                        (
                            target_pose_rv,
                            gripper_width,
                            gripper_open,
                            grasp_flag,
                        ) = self.set_target_pose()

                        continue

                    next_obs, rew, done, info = self.env.step(
                        action,
                        sample_perturbations=action_taken and self.sample_perturbations,
                    )
                    self._show_teleop_cameras(next_obs)

                    if rew == 1:
                        self.last_reward_idx = len(self.transitions)

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
                        t_start = time_module.monotonic()
                        (
                            target_pose_rv,
                            gripper_width,
                            gripper_open,
                            grasp_flag,
                        ) = self.set_target_pose()
                        continue

                    if action_taken:
                        if info["action_success"]:
                            self.store_transition(obs, action, rew, skill_complete)

                            translation, quat_xyzw = self.env.get_ee_pose()
                            translation, quat_xyzw = (
                                translation.cpu().numpy().squeeze(),
                                quat_xyzw.cpu().numpy().squeeze(),
                            )

                    obs = next_obs

                    translation, quat_xyzw = self.env.get_ee_pose()
                    translation, quat_xyzw = (
                        translation.cpu().numpy().squeeze(),
                        quat_xyzw.cpu().numpy().squeeze(),
                    )
                    rotvec = st.Rotation.from_quat(quat_xyzw).as_rotvec()

                    target_pose_rv = np.array([*translation, *rotvec])

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

    def reset(self):
        """Reset environment and clear checkpoint state."""
        obs = super().reset()
        self.checkpoint_state = None
        self.checkpoint_idx = None
        return obs
