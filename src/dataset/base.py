from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import torch

from src.common.control import ControlMode
from src.dataset.normalizer import LinearNormalizer


@dataclass(frozen=True)
class EpisodeRef:
    path_idx: int
    episode_idx: int
    frame_start: int
    frame_end: int
    frame_count: int
    task: str
    success: int
    domain: str


@dataclass(frozen=True)
class DatasetShardSpec:
    enabled: bool = False
    rank: int = 0
    world_size: int = 1
    split: str = "full"
    balance: str = "none"
    is_validation: bool = False


class BaseSequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_paths: Union[List[Path], Path],
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
        predict_past_actions: bool = False,
        control_mode: ControlMode = ControlMode.delta,
        pad_after: bool = True,
        episode_refs: Optional[Sequence[EpisodeRef]] = None,
        normalizer: Optional[LinearNormalizer] = None,
        shard_spec: Optional[DatasetShardSpec] = None,
    ):
        if isinstance(dataset_paths, list):
            self.dataset_paths = dataset_paths
        else:
            self.dataset_paths = [dataset_paths]

        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.predict_past_actions = predict_past_actions
        self.control_mode = control_mode
        self.pad_after = pad_after
        self.episode_refs = list(episode_refs) if episode_refs is not None else None
        self.shard_spec = shard_spec or DatasetShardSpec()
        self.load_into_memory_seconds = 0.0

        self.normalizer = LinearNormalizer()
        self._using_external_normalizer = normalizer is not None
        if normalizer is not None:
            self.normalizer.load_state_dict(normalizer.state_dict())
            self.normalizer.cpu()

        self.sequence_length = (
            pred_horizon if predict_past_actions else obs_horizon + pred_horizon - 1
        )
        self.first_action_idx = 0 if predict_past_actions else self.obs_horizon - 1
        self.final_action_idx = self.first_action_idx + self.pred_horizon

    def _load_combined_data(
        self,
        keys,
        *,
        full_loader: Callable,
        subset_loader: Callable,
        data_subset=None,
        max_episode_count=None,
    ):
        if self.episode_refs is not None:
            episode_refs = self.episode_refs
            if data_subset is not None:
                episode_refs = episode_refs[:data_subset]
            return subset_loader(self.dataset_paths, episode_refs, keys)

        return full_loader(
            self.dataset_paths,
            keys,
            max_episodes=data_subset,
            max_ep_cnt=max_episode_count,
        )

    def _set_episode_metadata(self, combined_data, metadata):
        self.episode_ends = combined_data["episode_ends"]
        self.metadata = metadata
        print(f"Loading dataset of {len(self.episode_ends)} episodes:")
        for path, data in metadata.items():
            print(
                f"  {path}: {data['n_episodes_used']} episodes, {data['n_frames_used']}"
            )

        self.successes = combined_data.get(
            "success", np.zeros(len(self.episode_ends), dtype=np.uint8)
        ).astype(np.uint8)
        self.failure_idx = combined_data.get(
            "failure_idx", np.full(len(self.episode_ends), -1, dtype=np.int64)
        )
        self.domain = combined_data.get(
            "domain", np.zeros(len(self.episode_ends), dtype=np.uint8)
        )

    def _build_indices(self, create_sample_indices_fn):
        self.indices = create_sample_indices_fn(
            episode_ends=self.episode_ends,
            sequence_length=self.sequence_length,
            pad_before=self.obs_horizon - 1,
            pad_after=self.action_horizon - 1 if self.pad_after else 0,
        )
        self.n_samples = len(self.indices)

    def _fit_normalizer(self, train_data):
        if not self._using_external_normalizer:
            self.normalizer.fit(train_data)

    def _apply_minority_class_power(self, minority_class_power):
        if not minority_class_power:
            return

        sim_indices = []
        real_indices = []

        for i, (_, _, _, _, demo_idx) in enumerate(self.indices):
            if self.domain[demo_idx] == 0:
                sim_indices.append(i)
            else:
                real_indices.append(i)

        sim_indices = np.array(sim_indices)
        real_indices = np.array(real_indices)

        sim_samples = len(sim_indices)
        real_samples = len(real_indices)
        if sim_samples == 0 or real_samples == 0:
            return

        class_samples = np.array([sim_samples, real_samples])
        total_samples = len(self.indices)

        print(
            f"Ratio of real to sim samples before upsampling: {real_samples/sim_samples:.2f}"
        )

        class_weights = np.power(class_samples, 1 / minority_class_power)
        class_weights = class_weights / np.sum(class_weights)
        desired_class_samples = total_samples * class_weights

        print(
            f"Ratio of real to sim samples after upsampling: {desired_class_samples[1]/desired_class_samples[0]:.2f}"
        )

        minority_class = np.argmin(class_samples)
        additional_samples_needed = int(
            desired_class_samples[minority_class] - class_samples[minority_class]
        )

        if additional_samples_needed <= 0:
            return

        source_indices = real_indices if minority_class == 1 else sim_indices
        additional_indices = np.random.choice(
            source_indices, size=additional_samples_needed, replace=True
        )
        additional_samples = self.indices[additional_indices]
        self.indices = np.concatenate((self.indices, additional_samples))
        self.n_samples = len(self.indices)

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def __len__(self):
        return len(self.indices)

    def train(self):
        pass

    def eval(self):
        pass
