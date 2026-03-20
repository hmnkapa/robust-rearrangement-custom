from gymnasium import Env
from omegaconf import DictConfig  # noqa: F401
import torch

import collections

import numpy as np
from tqdm import tqdm, trange
from ipdb import set_trace as bp  # noqa: F401

from typing import Dict, Optional, Union
from pathlib import Path

from src.behavior.base import Actor
from src.visualization.render_mp4 import create_in_memory_mp4
from src.common.context import suppress_all_output
from src.common.tasks import task2idx
from src.common.files import get_processed_path, trajectory_save_dir
from src.data_collection.io import save_raw_rollout
from src.data_processing.utils import filter_and_concat_robot_state
from src.data_processing.utils import resize, resize_crop
from tensordict import TensorDict

from copy import deepcopy

import wandb
import zarr
from datetime import datetime

from src.eval.skill_annotation_util import (
    draw_guidance_point_on_image,
    draw_skill_on_image,
    get_annotation_bundle,
    reset_skill_annotator,
)


RolloutStats = collections.namedtuple(
    "RolloutStats",
    [
        "success_rate",
        "n_success",
        "n_rollouts",
        "epoch_idx",
        "rollout_max_steps",
        "total_return",
        "total_reward",
    ],
)

RolloutSaveValues = collections.namedtuple(
    "RolloutSaveValues",
    [
        "robot_states",
        "imgs1",
        "imgs2",
        "actions",
        "rewards",
        "parts_poses",
        "point_clouds",
        "depth_image1",
        "depth_image2",
        "skills",
        "guidance_points",
        "guidance_points_2d",
        "camera_infos",
    ],
)


def resize_image(obs, key):
    try:
        obs[key] = resize(obs[key])
    except KeyError:
        pass

def resize_depth(obs, key):
    # key : [B, H, W]
    depth_image = obs[key].unsqueeze(-1) # [B, H, W, C]
    try:
        obs[key] = resize(depth_image).squeeze(-1)
    except KeyError:
        pass

def resize_crop_image(obs, key):
    try:
        obs[key] = resize_crop(obs[key])
    except KeyError:
        pass

def resize_crop_depth(obs, key):
    # key : [B, H, W]
    depth_image = obs[key].unsqueeze(-1) # [B, H, W, C]
    try:
        obs[key] = resize_crop(depth_image).squeeze(-1)
    except KeyError:
        pass

def squeeze_and_numpy(d: Dict[str, Union[torch.Tensor, np.ndarray, float, int, None]]):
    """
    Recursively squeeze and convert tensors to numpy arrays
    Convert scalars to floats
    Leave NoneTypes alone
    """
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = squeeze_and_numpy(v)

        elif v is None:
            continue

        elif isinstance(v, (torch.Tensor, np.ndarray)):
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            d[k] = v.squeeze()

        else:
            raise ValueError(f"Unsupported type: {type(v)}")

    return d


def tensordict_to_list_of_dicts(tensordict):
    list_of_dicts = []
    keys = list(tensordict.keys())
    num_elements = tensordict[keys[0]].shape[0]

    for i in range(num_elements):
        dict_element = {}
        for key in keys:
            dict_element[key] = tensordict[key][i].cpu().numpy()
        list_of_dicts.append(dict_element)

    return list_of_dicts


class SuccessTqdm(tqdm):
    def __init__(
        self,
        num_envs: int,
        n_rollouts: int,
        task_name: str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.num_envs = num_envs
        self.n_rollouts = n_rollouts
        self.task_name = task_name
        self.round = 0
        self.success_in_prev_rounds = 0

    def pbar_desc(self, n_success: int):
        total = self.round * self.num_envs
        n_success += self.success_in_prev_rounds
        success_rate = n_success / total if total > 0 else 0
        self.set_description(
            f"Performing rollouts ({self.task_name}): "
            f"round {self.round}/{self.n_rollouts//self.num_envs}, "
            f"success: {n_success}/{total} ({success_rate:.1%})"
        )

    def before_round(self, n_success: int):
        self.success_in_prev_rounds = n_success
        self.round += 1

        self.pbar_desc(0)


def rollout(
    env: Env,
    actor: Actor,
    rollout_max_steps: int,
    pbar: SuccessTqdm = None,
    resize_video: bool = True,
    n_parts_assemble: int = 1,
    save_rollouts: bool = False,
    pc_generator = None,
    annotate_skill: bool = False,
    skill_on_image: bool = False,
    annotate_wrist_camera: bool = True,
) -> Optional[RolloutSaveValues]:
    # get first observation
    with suppress_all_output(False):
        obs = env.reset()
        actor.reset()
    if annotate_skill:
        reset_skill_annotator(env)

    video_obs = deepcopy(obs)
    previous_skill = None
    initial_annotation = (
        get_annotation_bundle(
            env,
            previous_skill,
            annotate_wrist_camera=annotate_wrist_camera,
            resize_images=resize_video,
        )
        if annotate_skill
        else None
    )
    initial_skill = initial_annotation["skill"] if initial_annotation is not None else None
    initial_guidance_point = (
        None if initial_annotation is None else initial_annotation["guidance_point"]
    )
    initial_guidance_point_2d = (
        {} if initial_annotation is None else initial_annotation["guidance_point_2d"]
    )
    if initial_skill is not None:
        previous_skill = initial_skill
    if annotate_skill:
        initial_debug = {} if initial_annotation is None else initial_annotation.get("debug", {})
        if initial_debug:
            print(
                f"[skill-debug] step=0 idx={initial_debug.get('assemble_idx')} "
                f"part={initial_debug.get('active_part')} phase={initial_debug.get('phase')} "
                f"skill={initial_skill}",
                flush=True,
            )
        else:
            print(f"[skill-debug] step=0 skill={initial_skill}", flush=True)
        print(
            f"[guidance-debug] step=0 gp={initial_guidance_point} gp_2d={initial_guidance_point_2d}",
            flush=True,
        )

    # Resize the images in the observation if they exist
    resize_image(obs, "color_image1")
    resize_crop_image(obs, "color_image2")
    # Resize the depth image
    resize_depth(obs, "depth_image1")
    resize_crop_depth(obs, "depth_image2")

    if resize_video:
        resize_image(video_obs, "color_image1")
        resize_crop_image(video_obs, "color_image2")
        resize_depth(video_obs, "depth_image1")
        resize_crop_depth(video_obs, "depth_image2")

    if annotate_skill and initial_annotation is not None:
        initial_guidance = initial_annotation["guidance_point_2d"]
        if "color_image2" in video_obs:
            img2 = video_obs["color_image2"].cpu().numpy()
            # print(
            #     f"[guidance-draw-debug] step=0 image=color_image2 shape={img2.shape} uv={initial_guidance.get('color_image2')}",
            #     flush=True,
            # )
            img2 = draw_guidance_point_on_image(img2, initial_guidance.get("color_image2"))
            video_obs["color_image2"] = torch.from_numpy(img2).to(video_obs["color_image2"].device)
        if annotate_wrist_camera and "color_image1" in video_obs:
            img1 = video_obs["color_image1"].cpu().numpy()
            # print(
            #     f"[guidance-draw-debug] step=0 image=color_image1 shape={img1.shape} uv={initial_guidance.get('color_image1')}",
            #     flush=True,
            # )
            img1 = draw_guidance_point_on_image(img1, initial_guidance.get("color_image1"))
            video_obs["color_image1"] = torch.from_numpy(img1).to(video_obs["color_image1"].device)

    # save initial visualization and rewards
    robot_states = [TensorDict(video_obs["robot_state"], batch_size=env.num_envs)]
    imgs1 = [] if "color_image1" not in video_obs else [video_obs["color_image1"].cpu()]
    imgs2 = [] if "color_image2" not in video_obs else [video_obs["color_image2"].cpu()]
    depth_image1 = [] if "depth_image1" not in video_obs else [video_obs["depth_image1"]]
    depth_image2 = [] if "depth_image2" not in video_obs else [video_obs["depth_image2"]]
    parts_poses = [video_obs["parts_poses"].cpu()]
    skills = [initial_skill]
    guidance_points = [
        None if initial_annotation is None else initial_annotation["guidance_point"]
    ]
    guidance_points_2d = [
        {} if initial_annotation is None else initial_annotation["guidance_point_2d"]
    ]
    camera_infos = [
        {} if initial_annotation is None else initial_annotation["camera_info"]
    ]
    actions = list()
    rewards = torch.zeros((env.num_envs, rollout_max_steps), dtype=torch.float32)
    done = torch.zeros((env.num_envs, 1), dtype=torch.bool, device="cuda")
    
    # Collect point clouds if pc_generator is provided
    point_clouds = []  # List of lists: [[env0_step0, env1_step0, ...], [env0_step1, ...]]
    if pc_generator is not None:
        pcs_step = pc_generator.generate_transformed_cropped_point_cloud_for_all_env()
        # Add point cloud to obs for actor
        if len(pcs_step) > 0:
            obs["point_cloud"] = torch.stack(pcs_step)
            
        pcs_step_np = []
        for env_idx, pc in enumerate(pcs_step):
            pc_np = pc.detach().cpu().numpy()
            if pc_np.shape[0] == 0:
                print(f"[DEBUG] Empty point cloud: env={env_idx}, step=0 (initial)")
            pcs_step_np.append(pc_np)
        point_clouds.append(pcs_step_np)

    step_idx = 0

    # TODO - figure out how to fix this
    actor.normalizer = actor.normalizer.to(actor.device)
    actor.model = actor.model.to(actor.device)

    while not done.all():
        # Convert from robot state dict to robot state tensor
        if not getattr(actor, "expects_raw_robot_state", False):
            obs["robot_state"] = env.filter_and_concat_robot_state(obs["robot_state"])

        # Get the next actions from the actor
        action_pred = actor.action(obs)
        
        # print("[DEBUG] action: ", action_pred)
        # print("[DEBUG] gripper action: ", action_pred[:, 7])
        # action_pred = torch.tensor(actions[step_idx], device="cuda").unsqueeze(0)
        # action_pred = actor.normalizer(action_pred, "action", forward=False)

        obs, reward, done, _ = env.step(action_pred, sample_perturbations=False)

        # Generate point clouds for the new observation
        if pc_generator is not None:
            pcs_step = pc_generator.generate_transformed_cropped_point_cloud_for_all_env()
            if len(pcs_step) > 0:
                obs["point_cloud"] = torch.stack(pcs_step)
        else:
            pcs_step = None

        video_obs = deepcopy(obs)
        current_annotation = (
            get_annotation_bundle(
                env,
                previous_skill,
                annotate_wrist_camera=annotate_wrist_camera,
                resize_images=resize_video,
            )
            if annotate_skill
            else None
        )
        current_skill = current_annotation["skill"] if current_annotation is not None else None
        current_guidance_point = (
            None if current_annotation is None else current_annotation["guidance_point"]
        )
        current_guidance_point_2d = (
            {} if current_annotation is None else current_annotation["guidance_point_2d"]
        )
        if current_skill is not None:
            previous_skill = current_skill
        if annotate_skill:
            current_debug = {} if current_annotation is None else current_annotation.get("debug", {})
            if current_debug:
                print(
                    f"[skill-debug] step={step_idx + 1} idx={current_debug.get('assemble_idx')} "
                    f"part={current_debug.get('active_part')} phase={current_debug.get('phase')} "
                    f"skill={current_skill}",
                    flush=True,
                )
            else:
                print(f"[skill-debug] step={step_idx + 1} skill={current_skill}", flush=True)
            print(
                f"[guidance-debug] step={step_idx + 1} gp={current_guidance_point} gp_2d={current_guidance_point_2d}",
                flush=True,
            )

        # Resize the images in the observation if they exist
        resize_image(obs, "color_image1")
        resize_crop_image(obs, "color_image2")
        resize_depth(obs, "depth_image1")
        resize_crop_depth(obs, "depth_image2")

        # Save observations for the policy
        if resize_video:
            resize_image(video_obs, "color_image1")
            resize_crop_image(video_obs, "color_image2")
            resize_depth(video_obs, "depth_image1")
            resize_crop_depth(video_obs, "depth_image2")

        if annotate_skill:
            guidance_for_draw = current_annotation["guidance_point_2d"]
            if "color_image2" in video_obs:
                img2 = video_obs["color_image2"].cpu().numpy()
                # print(
                #     f"[guidance-draw-debug] step={step_idx + 1} image=color_image2 shape={img2.shape} uv={guidance_for_draw.get('color_image2')}",
                #     flush=True,
                # )
                img2 = draw_guidance_point_on_image(
                    img2,
                    guidance_for_draw.get("color_image2"),
                )
                video_obs["color_image2"] = torch.from_numpy(img2).to(obs["color_image2"].device)
            if annotate_wrist_camera and "color_image1" in video_obs:
                img1 = video_obs["color_image1"].cpu().numpy()
                # print(
                #     f"[guidance-draw-debug] step={step_idx + 1} image=color_image1 shape={img1.shape} uv={guidance_for_draw.get('color_image1')}",
                #     flush=True,
                # )
                img1 = draw_guidance_point_on_image(
                    img1,
                    guidance_for_draw.get("color_image1"),
                )
                video_obs["color_image1"] = torch.from_numpy(img1).to(obs["color_image1"].device)

        # Store the results for visualization and logging
        if save_rollouts:
            robot_states.append(
                TensorDict(video_obs["robot_state"], batch_size=env.num_envs)
            )
            if "color_image1" in video_obs:
                imgs1.append(video_obs["color_image1"].cpu())
            if "color_image2" in video_obs:
                imgs2.append(video_obs["color_image2"].cpu())
            if "depth_image1" in video_obs:
                depth_image1.append(video_obs["depth_image1"])
            if "depth_image2" in video_obs:
                depth_image2.append(video_obs["depth_image2"])
            actions.append(action_pred.cpu())
            parts_poses.append(video_obs["parts_poses"].cpu())
            skills.append(current_skill)
            guidance_points.append(
                None if current_annotation is None else current_annotation["guidance_point"]
            )
            guidance_points_2d.append(
                {} if current_annotation is None else current_annotation["guidance_point_2d"]
            )
            camera_infos.append(
                {} if current_annotation is None else current_annotation["camera_info"]
            )

            # Collect point clouds at each step
            if pcs_step is not None:
                pcs_step_np = []
                for env_idx, pc in enumerate(pcs_step):
                    pc_np = pc.detach().cpu().numpy()
                    if pc_np.shape[0] == 0:
                        current_success = (rewards[:, :step_idx+1].sum(dim=1) == n_parts_assemble)[env_idx].item()
                        print(f"[DEBUG] Empty point cloud: env={env_idx}, step={step_idx+1}, success_so_far={current_success}")
                    pcs_step_np.append(pc_np)
                point_clouds.append(pcs_step_np)

        # Always store rewards as they are used to calculate success
        rewards[:, step_idx] = reward.squeeze().cpu()

        # update progress bar
        step_idx += 1
        if pbar is not None:
            pbar.set_postfix(step=step_idx)
            n_success = (rewards.sum(dim=1) == n_parts_assemble).sum().item()
            pbar.pbar_desc(n_success)
            pbar.update()

        if step_idx >= rollout_max_steps:
            done = torch.ones((env.num_envs, 1), dtype=torch.bool, device="cuda")

        if done.all():
            break

    # Reorganize point_clouds from [step][env] to [env][step]
    if pc_generator is not None and point_clouds:
        # point_clouds is [[env0_s0, env1_s0, ...], [env0_s1, env1_s1, ...], ...]
        # Convert to [[env0_s0, env0_s1, ...], [env1_s0, env1_s1, ...], ...]
        num_steps = len(point_clouds)
        num_envs = len(point_clouds[0]) if point_clouds else 0
        pcs_per_env = []
        for env_idx in range(num_envs):
            pcs_per_env.append([point_clouds[step][env_idx] for step in range(num_steps)])
    else:
        pcs_per_env = None

    # print(f"[DEBUG] imgs1 shape: {(torch.stack(imgs1, dim=1) if imgs1 else []).shape}", flush=True)
    # print(f"[DEBUG] imgs2 shape: {(torch.stack(imgs2, dim=1) if imgs2 else []).shape}", flush=True)
    # print(f"[DEBUG] depth_image2 shape: {(torch.stack(depth_image2, dim=1) if depth_image2 else []).shape}", flush=True)
    # for i, t in enumerate(depth_image2):
    #     print(f"Index {i} device: {t.device}")

    return RolloutSaveValues(
        torch.stack(robot_states, dim=1) if robot_states else [],
        torch.stack(imgs1, dim=1) if imgs1 else [],
        torch.stack(imgs2, dim=1) if imgs2 else [],
        torch.stack(actions, dim=1) if actions else [],
        rewards,
        torch.stack(parts_poses, dim=1) if parts_poses else [],
        pcs_per_env,
        torch.stack(depth_image1, dim=1) if depth_image1 else [],
        torch.stack(depth_image2, dim=1) if depth_image2 else [],
        skills,
        guidance_points,
        guidance_points_2d,
        camera_infos,
    )


@torch.no_grad()
def calculate_success_rate(
    env: Env,
    actor: Actor,
    n_rollouts: int,
    rollout_max_steps: int,
    epoch_idx: int,
    discount: float = 0.99,
    rollout_save_dir: Optional[Path] = None,
    save_rollouts_to_wandb: bool = False,
    save_failures: bool = False,
    n_parts_assemble: Optional[int] = None,
    compress_pickles: bool = False,
    resize_video: bool = True,
    n_steps_padding: int = 30,
    break_on_n_success: bool = False,
    stop_after_n_success: int = 0,
    record_first_state_only: bool = False,
    pc_generator = None,
    annotate_skill: bool = False,
    skill_on_image: bool = False,
    annotate_wrist_camera: bool = True,
) -> RolloutStats:

    pbar = SuccessTqdm(
        num_envs=env.num_envs,
        n_rollouts=n_rollouts,
        task_name=env.task_name,
        total=rollout_max_steps * (n_rollouts // env.num_envs),
        desc="Performing rollouts",
        leave=True,
        unit="step",
    )

    if n_parts_assemble is None:
        n_parts_assemble = env.n_parts_assemble

    tbl = wandb.Table(
        columns=["rollout", "success", "epoch", "reward", "return", "steps"]
    )

    n_success = 0
    total_reward = 0
    episode_returns = []
    table_rows = []

    save_rollouts = rollout_save_dir is not None or save_rollouts_to_wandb

    # For record_first_state_only
    if record_first_state_only:
        first_robot_states = []
        first_part_poses = []
        first_success = []

    pbar.pbar_desc(n_success)
    for i in range(n_rollouts // env.num_envs):
        # Update the progress bar
        pbar.before_round(n_success)

        # Perform a rollout with the current model
        rollout_data: RolloutSaveValues = rollout(
            env,
            actor,
            rollout_max_steps,
            pbar=pbar,
            resize_video=resize_video,
            n_parts_assemble=n_parts_assemble,
            save_rollouts=save_rollouts,
            pc_generator=pc_generator,
            annotate_skill=annotate_skill,
            skill_on_image=skill_on_image,
            annotate_wrist_camera=annotate_wrist_camera,
        )

        # Calculate the success rate
        success_flags = rollout_data.rewards.sum(dim=1) == n_parts_assemble
        n_success += success_flags.sum().item()

        # Save the results from the rollout immediately
        if save_rollouts:
            have_img_obs = rollout_data.imgs1 is not None and len(rollout_data.imgs1) > 0
            have_depth_obs = rollout_data.depth_image1 is not None and len(rollout_data.depth_image1) > 0

            for env_idx in range(env.num_envs):
                robot_states = tensordict_to_list_of_dicts(rollout_data.robot_states[env_idx])
                actions = rollout_data.actions[env_idx].numpy()
                rewards = rollout_data.rewards[env_idx].numpy()
                parts_poses = rollout_data.parts_poses[env_idx].numpy()
                skills = [s for s in rollout_data.skills]
                guidance_points = [g for g in rollout_data.guidance_points]
                guidance_points_2d = [g for g in rollout_data.guidance_points_2d]
                camera_infos = [c for c in rollout_data.camera_infos]
                success = success_flags[env_idx].item()
                task = env.furniture_name
                
                # Get point clouds for this env (list of arrays per step)
                pcs_for_rollout = rollout_data.point_clouds[env_idx] if rollout_data.point_clouds is not None else None

                # Calculate episode return
                episode_return = np.sum(rewards * discount ** np.arange(len(rewards)))
                episode_returns.append(episode_return)
                total_reward += np.sum(rewards)

                if record_first_state_only:
                    first_robot_states.append(robot_states[0])
                    first_part_poses.append(parts_poses[0])
                    first_success.append(success)
                    continue

                video1 = (
                    rollout_data.imgs1[env_idx].numpy()
                    if have_img_obs
                    else np.zeros((len(robot_states), 2, 2, 3), dtype=np.uint8)
                )
                video2 = (
                    rollout_data.imgs2[env_idx].numpy()
                    if have_img_obs
                    else np.zeros((len(robot_states), 2, 2, 3), dtype=np.uint8)
                )
                video2_for_video = video2.copy()
                if annotate_skill and skill_on_image:
                    n_annotated = min(len(video2_for_video), len(skills))
                    for frame_idx in range(n_annotated):
                        skill = skills[frame_idx]
                        if skill is None:
                            continue
                        video2_for_video[frame_idx] = draw_skill_on_image(
                            video2_for_video[frame_idx], skill
                        )
                depth_video1 = (
                    rollout_data.depth_image1[env_idx].cpu().numpy()
                    if have_depth_obs
                    else np.zeros((len(robot_states), 2, 2, 3), dtype=np.uint8)
                )
                depth_video2 = (
                    rollout_data.depth_image2[env_idx].cpu().numpy()
                    if have_depth_obs
                    else np.zeros((len(robot_states), 2, 2, 3), dtype=np.uint8)
                )

                # Number of steps until success
                n_steps = (
                    np.where(rewards == 1)[0][-1] + 1 if success else rollout_max_steps
                )
                n_steps += n_steps_padding
                trim_start_steps = 0

                # Stack the two videos side by side
                if have_img_obs:
                    video = np.concatenate([video1, video2_for_video], axis=2)[
                        trim_start_steps:n_steps
                    ]
                    video = create_in_memory_mp4(video, fps=20)

                if save_rollouts_to_wandb and have_img_obs:
                    table_rows.append(
                        [
                            wandb.Video(video, fps=20, format="mp4"),
                            success,
                            epoch_idx,
                            np.sum(rewards),
                            episode_return,
                            n_steps,
                        ]
                    )

                if rollout_save_dir is not None and (save_failures or success):
                    # Trim point clouds to match n_steps
                    pcs_trimmed = None
                    if pcs_for_rollout is not None:
                        pcs_trimmed = pcs_for_rollout[trim_start_steps : n_steps + 1]
                    save_raw_rollout(
                        robot_states=robot_states[trim_start_steps : n_steps + 1],
                        imgs1=video1[trim_start_steps : n_steps + 1],
                        imgs2=video2[trim_start_steps : n_steps + 1],
                        depth_image1=depth_video1[trim_start_steps : n_steps + 1],
                        depth_image2=depth_video2[trim_start_steps : n_steps + 1],
                        parts_poses=parts_poses[trim_start_steps : n_steps + 1],
                        skills=skills[trim_start_steps : n_steps + 1],
                        guidance_points=guidance_points[trim_start_steps : n_steps + 1],
                        guidance_points_2d=guidance_points_2d[trim_start_steps : n_steps + 1],
                        camera_infos=camera_infos[trim_start_steps : n_steps + 1],
                        actions=actions[trim_start_steps:n_steps],
                        rewards=rewards[trim_start_steps:n_steps],
                        success=success,
                        task=task,
                        action_type=env.action_type,
                        rollout_save_dir=rollout_save_dir,
                        compress_pickles=compress_pickles,
                        have_img_obs=have_img_obs,
                        have_depth_obs=have_depth_obs,
                        pcs=pcs_trimmed,
                        skill_on_image=skill_on_image,
                    )

        if break_on_n_success and n_success >= stop_after_n_success:
            print(
                f"Current number of success {n_success} greater than breaking threshold {stop_after_n_success}. Breaking"
            )
            break

    # Handle record_first_state_only after all rollouts
    if record_first_state_only and rollout_save_dir is not None:
        first_state_npz = str(rollout_save_dir / "first_states.npz")
        print(f"Saving first states to: {first_state_npz}")
        np.savez(
            first_state_npz,
            robot_states=np.asarray(first_robot_states),
            part_poses=np.asarray(first_part_poses),
            success=np.asarray(first_success),
        )

    # Handle wandb table after all rollouts
    if save_rollouts_to_wandb and table_rows:
        table_rows = sorted(table_rows, key=lambda x: x[4], reverse=True)
        for row in table_rows:
            tbl.add_data(*row)
        if wandb.run is not None:
            wandb.log(
                {
                    "rollouts": tbl,
                    "epoch": epoch_idx,
                }
            )

    pbar.close()

    return RolloutStats(
        success_rate=n_success / n_rollouts,
        n_success=n_success,
        n_rollouts=n_rollouts,
        epoch_idx=epoch_idx,
        rollout_max_steps=rollout_max_steps,
        total_return=np.sum(episode_returns) if episode_returns else 0,
        total_reward=total_reward,
    )


def do_rollout_evaluation(
    config: DictConfig,
    env: Env,
    save_rollouts_to_file: bool,
    save_rollouts_to_wandb: bool,
    actor: Actor,
    best_success_rate: float,
    epoch_idx: int,
) -> float:
    rollout_save_dir = None

    if save_rollouts_to_file:
        rollout_save_dir = trajectory_save_dir(
            controller=env.ctrl_mode,
            environment="sim",
            task=config.task,
            demo_source="rollout",
            randomness=config.randomness,
            # Don't create here because we have to do it when we save anyway
            create=False,
        )

    actor.set_task(task2idx[config.task])

    rollout_stats = calculate_success_rate(
        env,
        actor,
        n_rollouts=config.rollout.count,
        rollout_max_steps=config.rollout.max_steps,
        epoch_idx=epoch_idx,
        discount=config.discount,
        rollout_save_dir=rollout_save_dir,
        save_rollouts_to_wandb=save_rollouts_to_wandb,
        save_failures=config.rollout.save_failures,
    )
    success_rate = rollout_stats.success_rate
    best_success_rate = max(best_success_rate, success_rate)
    mean_return = rollout_stats.total_return / rollout_stats.n_rollouts

    # Log the success rate to wandb
    wandb.log(
        {
            "success_rate": success_rate,
            "best_success_rate": best_success_rate,
            "epoch_mean_return": mean_return,
            "n_success": rollout_stats.n_success,
            "n_rollouts": rollout_stats.n_rollouts,
            "epoch": epoch_idx,
        }
    )

    return best_success_rate
