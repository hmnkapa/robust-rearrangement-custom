from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Union

import numpy as np
import torch

import src.common.geometry as C
from src.common.control import ControlMode
from src.dataset.base import BaseSequenceDataset, DatasetShardSpec, EpisodeRef
from src.dataset.normalizer import LinearNormalizer
from src.dataset.storage import build_lazy_image_stores, combine_datasets, combine_episode_subset

from ipdb import set_trace as bp


def create_sample_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append(
                [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx, i]
            )
    if not indices:
        return np.zeros((0, 5), dtype=np.int64)
    return np.array(indices, dtype=np.int64)


def sample_sequence(
    train_data: Dict[str, torch.Tensor],
    sequence_length: int,
    buffer_start_idx: int,
    buffer_end_idx: int,
    sample_start_idx: int,
    sample_end_idx: int,
) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = torch.zeros(
                size=(sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype
            )
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


def float_tensor_from_numpy(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).to(dtype=torch.float32)


def load_combined_data(
    dataset: BaseSequenceDataset,
    keys,
    *,
    data_subset=None,
    max_episode_count=None,
):
    return dataset._load_combined_data(
        keys,
        full_loader=combine_datasets,
        subset_loader=combine_episode_subset,
        data_subset=data_subset,
        max_episode_count=max_episode_count,
    )


def apply_relative_normalizer_bounds(
    dataset: BaseSequenceDataset,
    combined_data,
    pred_horizon: int,
):
    max_delta_action = np.max(np.abs(combined_data["action/delta"][:, :3]))
    dataset.normalizer.stats.action.min[:3] = -max_delta_action * pred_horizon
    dataset.normalizer.stats.action.max[:3] = max_delta_action * pred_horizon
    dataset.normalizer.stats.action.min[3:] = -1.0
    dataset.normalizer.stats.action.max[3:] = 1.0
    dataset.normalizer.stats.robot_state.min[:9] = -1.0
    dataset.normalizer.stats.robot_state.max[:9] = 1.0


class ImageDataset(BaseSequenceDataset):
    def __init__(
        self,
        dataset_paths: Union[List[Path], Path],
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
        data_subset: Optional[int] = None,
        predict_past_actions: bool = False,
        control_mode: ControlMode = ControlMode.delta,
        pad_after: bool = True,
        max_episode_count: Union[dict, None] = None,
        minority_class_power: bool = False,
        load_into_memory: bool = True,
        episode_refs: Optional[List[EpisodeRef]] = None,
        normalizer: Optional[LinearNormalizer] = None,
        shard_spec: Optional[DatasetShardSpec] = None,
    ):
        super().__init__(
            dataset_paths=dataset_paths,
            pred_horizon=pred_horizon,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            predict_past_actions=predict_past_actions,
            control_mode=control_mode,
            pad_after=pad_after,
            episode_refs=episode_refs,
            normalizer=normalizer,
            shard_spec=shard_spec,
        )
        self.minority_class_power = minority_class_power
        self.load_into_memory = load_into_memory
        self.non_image_keys = ["robot_state", "action/pos", "action/delta", "skill"]
        self.image_keys = ["color_image1", "color_image2"]

        control_mode_key = "pos" if control_mode == ControlMode.relative else control_mode

        if self.load_into_memory:
            load_into_memory_start_perf = perf_counter()
            combined_data, metadata = load_combined_data(
                self,
                self.non_image_keys + self.image_keys,
                data_subset=data_subset,
                max_episode_count=max_episode_count,
            )
            self.load_into_memory_seconds = perf_counter() - load_into_memory_start_perf
        else:
            combined_data, metadata = load_combined_data(
                self,
                self.non_image_keys,
                data_subset=data_subset,
                max_episode_count=max_episode_count,
            )
            self.image_stores = build_lazy_image_stores(self.dataset_paths)

        self._set_episode_metadata(combined_data, metadata)
        self.train_data = {
            "robot_state": float_tensor_from_numpy(combined_data["robot_state"]),
            "action": float_tensor_from_numpy(combined_data[f"action/{control_mode_key}"]),
            "skill": float_tensor_from_numpy(combined_data["skill"]),
        }
        self._fit_normalizer(self.train_data)

        if self.control_mode == ControlMode.relative:
            if not self._using_external_normalizer:
                apply_relative_normalizer_bounds(self, combined_data, pred_horizon)
        else:
            for key in self.normalizer.keys():
                self.train_data[key] = self.normalizer(self.train_data[key], key, forward=True)

        if self.load_into_memory:
            self.train_data["color_image1"] = torch.from_numpy(
                combined_data["color_image1"]
            ).permute(0, 3, 1, 2)
            self.train_data["color_image2"] = torch.from_numpy(
                combined_data["color_image2"]
            ).permute(0, 3, 1, 2)

        self.train_data["zarr_idx"] = torch.from_numpy(combined_data["zarr_idx"])
        self.train_data["within_zarr_idx"] = torch.from_numpy(combined_data["within_zarr_idx"])

        self._build_indices(create_sample_indices)
        self.skills = combined_data["skill"].astype(np.uint8)
        self.action_dim = self.train_data["action"].shape[-1]
        self.robot_state_dim = self.train_data["robot_state"].shape[-1]
        self.skill_dim = self.train_data["skill"].shape[-1]
        self._apply_minority_class_power(self.minority_class_power)

    def __getitem__(self, idx):
        (
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
            demo_idx,
        ) = self.indices[idx]

        nsample = sample_sequence(
            train_data=self.train_data,
            sequence_length=self.sequence_length,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )
        if not self.load_into_memory:
            frame_indices = nsample["within_zarr_idx"][: self.obs_horizon].cpu().numpy()
            zarr_indices = nsample["zarr_idx"][: self.obs_horizon].cpu().numpy()
            if np.any(zarr_indices != zarr_indices[0]):
                raise ValueError("Lazy image loading expects all observation frames to come from one dataset.")

            frame_batch = self.image_stores[int(zarr_indices[0])].get_frames(
                frame_indices,
                self.image_keys,
            )
            nsample["color_image1"] = torch.from_numpy(frame_batch["color_image1"]).permute(0, 3, 1, 2)
            nsample["color_image2"] = torch.from_numpy(frame_batch["color_image2"]).permute(0, 3, 1, 2)

        nsample["color_image1"] = nsample["color_image1"][: self.obs_horizon, :]
        nsample["color_image2"] = nsample["color_image2"][: self.obs_horizon, :]
        nsample["robot_state"] = nsample["robot_state"][: self.obs_horizon, :]
        nsample["skill"] = nsample["skill"][: self.obs_horizon, :]
        nsample["action"] = nsample["action"][self.first_action_idx : self.final_action_idx, :].clone()

        if self.control_mode == ControlMode.relative:
            curr_ee_pos = nsample["robot_state"][-1, :3]
            curr_ee_6d = nsample["robot_state"][-1, 3:9]
            curr_ee_quat_xyzw = C.rotation_6d_to_quaternion_xyzw(curr_ee_6d)
            nsample["action"][:, :3] = nsample["action"][:, :3] - curr_ee_pos

            if torch.any(torch.isnan(nsample["action"][:, :3])) or torch.any(
                torch.abs(nsample["action"][:, :3]) > 1.0
            ):
                print("Relative pos action has NaN or elements bigger than 1")

            action_quat_xyzw = C.rotation_6d_to_quaternion_xyzw(nsample["action"][:, 3:9])
            action_quat_xyzw = C.quaternion_multiply(
                C.quaternion_invert(curr_ee_quat_xyzw), action_quat_xyzw
            )
            nsample["action"][:, 3:9] = C.quaternion_to_rotation_6d(action_quat_xyzw)
            nsample["action"] = self.normalizer(nsample["action"], "action", forward=True)
            nsample["robot_state"] = self.normalizer(
                nsample["robot_state"], "robot_state", forward=True
            )

        nsample["success"] = torch.IntTensor([self.successes[demo_idx]])
        nsample["domain"] = torch.IntTensor([self.domain[demo_idx]])
        return nsample


class RGBDDataset(BaseSequenceDataset):
    def __init__(
        self,
        dataset_paths: Union[List[Path], Path],
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
        data_subset: Optional[int] = None,
        predict_past_actions: bool = False,
        control_mode: ControlMode = ControlMode.delta,
        pad_after: bool = True,
        max_episode_count: Union[dict, None] = None,
        minority_class_power: bool = False,
        load_into_memory: bool = True,
        episode_refs: Optional[List[EpisodeRef]] = None,
        normalizer: Optional[LinearNormalizer] = None,
        shard_spec: Optional[DatasetShardSpec] = None,
    ):
        super().__init__(
            dataset_paths=dataset_paths,
            pred_horizon=pred_horizon,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            predict_past_actions=predict_past_actions,
            control_mode=control_mode,
            pad_after=pad_after,
            episode_refs=episode_refs,
            normalizer=normalizer,
            shard_spec=shard_spec,
        )
        self.minority_class_power = minority_class_power
        self.load_into_memory = load_into_memory
        self.non_image_keys = ["robot_state", "action/pos", "action/delta", "skill"]
        self.image_keys = ["color_image1", "color_image2"]
        self.depth_keys = ["depth_image1", "depth_image2"]

        control_mode_key = "pos" if control_mode == ControlMode.relative else control_mode

        if self.load_into_memory:
            load_into_memory_start_perf = perf_counter()
            combined_data, metadata = load_combined_data(
                self,
                self.non_image_keys + self.image_keys + self.depth_keys,
                data_subset=data_subset,
                max_episode_count=max_episode_count,
            )
            self.load_into_memory_seconds = perf_counter() - load_into_memory_start_perf
        else:
            combined_data, metadata = load_combined_data(
                self,
                self.non_image_keys,
                data_subset=data_subset,
                max_episode_count=max_episode_count,
            )
            self.image_stores = build_lazy_image_stores(self.dataset_paths)

        self._set_episode_metadata(combined_data, metadata)
        self.train_data = {
            "robot_state": float_tensor_from_numpy(combined_data["robot_state"]),
            "action": float_tensor_from_numpy(combined_data[f"action/{control_mode_key}"]),
            "skill": float_tensor_from_numpy(combined_data["skill"]),
        }
        self._fit_normalizer(self.train_data)

        if self.control_mode == ControlMode.relative:
            if not self._using_external_normalizer:
                apply_relative_normalizer_bounds(self, combined_data, pred_horizon)
        else:
            for key in self.normalizer.keys():
                self.train_data[key] = self.normalizer(self.train_data[key], key, forward=True)

        if self.load_into_memory:
            self.train_data["color_image1"] = torch.from_numpy(
                combined_data["color_image1"]
            ).permute(0, 3, 1, 2)
            self.train_data["color_image2"] = torch.from_numpy(
                combined_data["color_image2"]
            ).permute(0, 3, 1, 2)
            self.train_data["depth_image1"] = torch.from_numpy(
                combined_data["depth_image1"]
            ).to(dtype=torch.float32).unsqueeze(1)
            self.train_data["depth_image2"] = torch.from_numpy(
                combined_data["depth_image2"]
            ).to(dtype=torch.float32).unsqueeze(1)

        self.train_data["zarr_idx"] = torch.from_numpy(combined_data["zarr_idx"])
        self.train_data["within_zarr_idx"] = torch.from_numpy(combined_data["within_zarr_idx"])

        self._build_indices(create_sample_indices)
        self.skills = combined_data["skill"].astype(np.uint8)
        self.action_dim = self.train_data["action"].shape[-1]
        self.robot_state_dim = self.train_data["robot_state"].shape[-1]
        self.skill_dim = self.train_data["skill"].shape[-1]
        self._apply_minority_class_power(self.minority_class_power)

    def __getitem__(self, idx):
        (
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
            demo_idx,
        ) = self.indices[idx]

        nsample = sample_sequence(
            train_data=self.train_data,
            sequence_length=self.sequence_length,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )
        if not self.load_into_memory:
            frame_indices = nsample["within_zarr_idx"][: self.obs_horizon].cpu().numpy()
            zarr_indices = nsample["zarr_idx"][: self.obs_horizon].cpu().numpy()
            if np.any(zarr_indices != zarr_indices[0]):
                raise ValueError("Lazy RGBD loading expects all observation frames to come from one dataset.")

            frame_batch = self.image_stores[int(zarr_indices[0])].get_frames(
                frame_indices,
                self.image_keys + self.depth_keys,
            )
            nsample["color_image1"] = torch.from_numpy(frame_batch["color_image1"]).permute(0, 3, 1, 2)
            nsample["color_image2"] = torch.from_numpy(frame_batch["color_image2"]).permute(0, 3, 1, 2)
            nsample["depth_image1"] = torch.from_numpy(frame_batch["depth_image1"]).to(dtype=torch.float32).unsqueeze(1)
            nsample["depth_image2"] = torch.from_numpy(frame_batch["depth_image2"]).to(dtype=torch.float32).unsqueeze(1)

        nsample["color_image1"] = nsample["color_image1"][: self.obs_horizon, :]
        nsample["color_image2"] = nsample["color_image2"][: self.obs_horizon, :]
        nsample["depth_image1"] = nsample["depth_image1"][: self.obs_horizon, :]
        nsample["depth_image2"] = nsample["depth_image2"][: self.obs_horizon, :]
        nsample["robot_state"] = nsample["robot_state"][: self.obs_horizon, :]
        nsample["skill"] = nsample["skill"][: self.obs_horizon, :]
        nsample["action"] = nsample["action"][self.first_action_idx : self.final_action_idx, :].clone()

        if self.control_mode == ControlMode.relative:
            curr_ee_pos = nsample["robot_state"][-1, :3]
            curr_ee_6d = nsample["robot_state"][-1, 3:9]
            curr_ee_quat_xyzw = C.rotation_6d_to_quaternion_xyzw(curr_ee_6d)
            nsample["action"][:, :3] = nsample["action"][:, :3] - curr_ee_pos

            if torch.any(torch.isnan(nsample["action"][:, :3])) or torch.any(
                torch.abs(nsample["action"][:, :3]) > 1.0
            ):
                print("Relative pos action has NaN or elements bigger than 1")

            action_quat_xyzw = C.rotation_6d_to_quaternion_xyzw(nsample["action"][:, 3:9])
            action_quat_xyzw = C.quaternion_multiply(
                C.quaternion_invert(curr_ee_quat_xyzw), action_quat_xyzw
            )
            nsample["action"][:, 3:9] = C.quaternion_to_rotation_6d(action_quat_xyzw)
            nsample["action"] = self.normalizer(nsample["action"], "action", forward=True)
            nsample["robot_state"] = self.normalizer(
                nsample["robot_state"], "robot_state", forward=True
            )

        nsample["success"] = torch.IntTensor([self.successes[demo_idx]])
        nsample["domain"] = torch.IntTensor([self.domain[demo_idx]])
        return nsample


class StateDataset(BaseSequenceDataset):
    def __init__(
        self,
        dataset_paths: Union[List[Path], Path],
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
        data_subset: int = None,
        predict_past_actions: bool = False,
        control_mode: ControlMode = ControlMode.delta,
        pad_after: bool = True,
        max_episode_count: Union[dict, None] = None,
        task: str = None,
        add_relative_pose: bool = False,
        normalizer: Optional[LinearNormalizer] = None,
        include_future_obs: bool = False,
        episode_refs: Optional[List[EpisodeRef]] = None,
        shard_spec: Optional[DatasetShardSpec] = None,
    ):
        super().__init__(
            dataset_paths=dataset_paths,
            pred_horizon=pred_horizon,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            predict_past_actions=predict_past_actions,
            control_mode=control_mode,
            pad_after=pad_after,
            episode_refs=episode_refs,
            normalizer=normalizer,
            shard_spec=shard_spec,
        )
        self.include_future_obs = include_future_obs

        load_into_memory_start_perf = perf_counter()
        combined_data, metadata = load_combined_data(
            self,
            ["parts_poses", "robot_state", f"action/{control_mode}"],
            data_subset=data_subset,
            max_episode_count=max_episode_count,
        )
        self.load_into_memory_seconds = perf_counter() - load_into_memory_start_perf

        self._set_episode_metadata(combined_data, metadata)
        self.train_data = {
            "parts_poses": float_tensor_from_numpy(combined_data["parts_poses"]),
            "robot_state": float_tensor_from_numpy(combined_data["robot_state"]),
            "action": float_tensor_from_numpy(combined_data[f"action/{control_mode}"]),
        }

        self._fit_normalizer(self.train_data)

        if task == "place-tabletop":
            self._make_tabletop_goal()

        for key in self.normalizer.keys():
            self.train_data[key] = self.normalizer(self.train_data[key], key, forward=True)

        self.train_data["obs"] = torch.cat(
            [self.train_data["robot_state"], self.train_data["parts_poses"]], dim=-1
        )

        if add_relative_pose:
            parts_poses, robot_state = (
                self.train_data["parts_poses"],
                self.train_data["robot_state"],
            )
            n_frames = parts_poses.shape[0]
            n_parts = parts_poses.shape[1] // 7
            ee_pos = robot_state[:, None, :3]
            ee_quat_xyzw = C.rot_6d_to_isaac_quat(robot_state[:, 3:9]).view(n_frames, 1, 4)
            ee_pose = torch.cat([ee_pos, ee_quat_xyzw], dim=-1)
            parts_pose = parts_poses.view(n_frames, n_parts, 7)
            rel_pose = C.pose_error(ee_pose, parts_pose)
            self.train_data["rel_poses"] = rel_pose.view(n_frames, -1)
            self.train_data["obs"] = torch.cat(
                [self.train_data["obs"], self.train_data["rel_poses"]], dim=-1
            )

        rewards = torch.zeros_like(self.train_data["robot_state"][:, 0])
        rewards[self.episode_ends - 1] = 1.0

        gamma = 0.99
        returns = []
        episode_edges = [0] + self.episode_ends.tolist()
        for start, end in zip(episode_edges[:-1], episode_edges[1:]):
            ep_rewards = rewards[start:end]
            timesteps = torch.arange(len(ep_rewards), device=ep_rewards.device)
            discounts = gamma**timesteps
            ep_returns = (
                torch.flip(
                    torch.cumsum(torch.flip(ep_rewards * discounts, dims=[0]), dim=0),
                    dims=[0],
                )
                / discounts
            )
            returns.append(ep_returns)

        if returns:
            self.train_data["returns"] = torch.cat(returns)
        else:
            self.train_data["returns"] = torch.zeros(0, dtype=torch.float32)
        self._build_indices(create_sample_indices)

        self.action_dim = self.train_data["action"].shape[-1]
        self.robot_state_dim = self.train_data["robot_state"].shape[-1]
        self.parts_poses_dim = self.train_data["parts_poses"].shape[-1]
        self.obs_dim = (self.robot_state_dim + self.parts_poses_dim) * self.obs_horizon
        self.last_obs = self.obs_horizon if not self.include_future_obs else self.sequence_length

        del self.train_data["robot_state"]
        del self.train_data["parts_poses"]
        if add_relative_pose:
            del self.train_data["rel_poses"]

    def __getitem__(self, idx):
        (
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
            demo_idx,
        ) = self.indices[idx]

        nsample = sample_sequence(
            train_data=self.train_data,
            sequence_length=self.sequence_length,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )
        nsample["action"] = nsample["action"][self.first_action_idx : self.final_action_idx, :]
        nsample["obs"] = nsample["obs"][: self.last_obs, :]
        nsample["returns"] = nsample["returns"][self.first_action_idx : self.final_action_idx].sum()
        return nsample

    def _make_tabletop_goal(self):
        episode_edges = np.array([0] + self.episode_ends.tolist())
        tabletop_goal = torch.tensor([0.0819, 0.2866, -0.0157])
        new_episode_starts = []
        new_episode_ends = []
        curr_cumulate_timesteps = 0
        self.episode_ends = []

        for prev_ee, curr_ee in zip(episode_edges[:-1], episode_edges[1:]):
            for i in range(prev_ee, curr_ee):
                if torch.allclose(
                    self.train_data["parts_poses"][i, :3], tabletop_goal, atol=1e-2
                ):
                    new_episode_starts.append(prev_ee)
                    end = i + 10
                    new_episode_ends.append(end)
                    curr_cumulate_timesteps += end - prev_ee
                    self.episode_ends.append(curr_cumulate_timesteps)
                    break

        for key in self.train_data:
            data_slices = [
                self.train_data[key][start:end]
                for start, end in zip(new_episode_starts, new_episode_ends)
            ]
            if data_slices:
                self.train_data[key] = torch.cat(data_slices)
            else:
                self.train_data[key] = self.train_data[key][:0]

        self.episode_ends = torch.tensor(self.episode_ends)
