from typing import Tuple, Union

import torch
from omegaconf import DictConfig

from src.behavior.base import Actor
from src.models import get_diffusion_backbone
from src.models.dit_policy_objectives import FlowMatchingObjective


class FMTPolicy(Actor):
    def __init__(
        self,
        device: Union[str, torch.device],
        cfg: DictConfig,
    ) -> None:
        super().__init__(device, cfg)
        actor_cfg = cfg.actor

        self.model = get_diffusion_backbone(
            action_dim=self.action_dim,
            obs_dim=self.obs_dim,
            actor_config=actor_cfg,
        ).to(device)
        self.flow_matching = FlowMatchingObjective(
            sigma_min=actor_cfg.sigma_min,
            num_integration_steps=actor_cfg.num_integration_steps,
            timestep_sampling=actor_cfg.timestep_sampling,
            timestep_alpha=actor_cfg.timestep_sampling_alpha,
            timestep_beta=actor_cfg.timestep_sampling_beta,
            timestep_s=actor_cfg.timestep_sampling_s,
            integration_method=actor_cfg.integration_method,
        )

    def _normalized_action(self, nobs: torch.Tensor) -> torch.Tensor:
        B = nobs.shape[0]

        if not self.flatten_obs and len(nobs.shape) == 2:
            nobs = nobs.reshape(B, self.obs_horizon, self.obs_dim)

        return self.flow_matching.sample(
            model=self.model,
            shape=(B, self.pred_horizon, self.action_dim),
            conditioning_vec=nobs.float(),
            device=self.device,
            dtype=nobs.dtype,
        )

    def compute_loss(self, batch) -> Tuple[torch.Tensor, dict]:
        obs_cond = self._training_obs(batch, flatten=self.flatten_obs)
        naction = batch["action"]

        loss = self.flow_matching.compute_loss(
            model=self.model,
            action=naction,
            conditioning_vec=obs_cond.float(),
            loss_fn=self.loss_fn,
        )
        loss = loss.mean(dim=[1, 2]).unsqueeze(1)

        if self.rescale_loss_for_domain:
            domain = batch["domain"].squeeze().long()
            if domain.dim() == 0:
                domain = domain.unsqueeze(0)
            class_sizes = torch.bincount(domain, minlength=int(domain.max().item()) + 1)
            class_weights = torch.pow(class_sizes.clamp_min(1).float(), -1.0 / 2)
            class_weights = class_weights / class_weights.sum()
            loss *= class_weights[domain].unsqueeze(1)

        loss = loss.mean()
        losses = {"bc_loss": loss.item()}

        if self.camera_2_vib is not None:
            mu, log_var = batch["mu"], batch["log_var"]
            vib_loss = self.camera_2_vib.kl_divergence(mu, log_var)
            losses["vib_loss"] = vib_loss.item()
            vib_loss = torch.clamp(vib_loss, max=1)
            loss += self.vib_front_feature_beta * vib_loss

        if self.confusion_loss_beta > 0:
            confusion_loss = batch["confusion_loss"]
            losses["confusion_loss"] = confusion_loss.item()
            loss += self.confusion_loss_beta * confusion_loss

        return loss, losses
