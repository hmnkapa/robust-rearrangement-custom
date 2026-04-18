#!/usr/bin/env python
# Adapted from:
# https://github.com/brysonjones/multitask_dit_policy
#
# Original copyright 2025 Bryson Jones.
# Licensed under the Apache License, Version 2.0.
#
# This local copy keeps only the flow-matching objective and sampler logic needed
# by this repository. Diffusion scheduling stays in src.behavior.diffusion.

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FlowMatchingObjective:
    def __init__(
        self,
        sigma_min: float = 0.0,
        num_integration_steps: int = 100,
        timestep_sampling: str = "beta",
        timestep_alpha: float = 1.5,
        timestep_beta: float = 1.0,
        timestep_s: float = 0.999,
        integration_method: str = "euler",
    ):
        self.sigma_min = sigma_min
        self.num_integration_steps = num_integration_steps
        self.timestep_sampling = timestep_sampling
        self.timestep_alpha = timestep_alpha
        self.timestep_beta = timestep_beta
        self.timestep_s = timestep_s
        self.integration_method = integration_method

    def sample_t(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if self.timestep_sampling == "uniform":
            t = torch.rand(batch_size, device=device, dtype=dtype)
        elif self.timestep_sampling == "beta":
            alpha = torch.tensor(
                self.timestep_alpha,
                device=device,
                dtype=dtype,
            )
            beta = torch.tensor(
                self.timestep_beta,
                device=device,
                dtype=dtype,
            )
            dist = torch.distributions.Beta(alpha, beta)
            u = dist.sample((batch_size,))
            t = self.timestep_s * (1.0 - u)
        else:
            raise ValueError(f"Unknown timestep sampling: {self.timestep_sampling}")

        return t

    def interpolate(self, action: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        t = t.view(-1, *([1] * (action.dim() - 1)))
        return (1 - (1 - self.sigma_min) * t) * noise + t * action

    def target_velocity(self, action: Tensor, noise: Tensor) -> Tensor:
        return action - (1 - self.sigma_min) * noise

    def compute_loss(
        self,
        model: nn.Module,
        action: Tensor,
        conditioning_vec: Tensor,
        loss_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    ) -> Tensor:
        batch_size = action.shape[0]
        noise = torch.randn_like(action)
        t = self.sample_t(batch_size, action.device, action.dtype)
        x_t = self.interpolate(action, noise, t)
        target = self.target_velocity(action, noise)
        pred = model(sample=x_t, timestep=t, global_cond=conditioning_vec)

        if loss_fn is None:
            return F.mse_loss(pred, target, reduction="none")
        return loss_fn(pred, target)

    def _model_velocity(
        self,
        model: nn.Module,
        sample: Tensor,
        t: Tensor,
        conditioning_vec: Tensor,
    ) -> Tensor:
        return model(sample=sample, timestep=t, global_cond=conditioning_vec)

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape,
        conditioning_vec: Tensor,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        x = torch.randn(shape, device=device, dtype=dtype)
        dt = 1.0 / self.num_integration_steps

        for step_idx in range(self.num_integration_steps):
            t_value = step_idx / self.num_integration_steps
            t = torch.full((shape[0],), t_value, device=device, dtype=dtype)
            if self.integration_method == "euler":
                v = self._model_velocity(model, x, t, conditioning_vec)
                x = x + dt * v
            elif self.integration_method == "rk4":
                k1 = self._model_velocity(model, x, t, conditioning_vec)
                t_mid = torch.full((shape[0],), t_value + 0.5 * dt, device=device, dtype=dtype)
                k2 = self._model_velocity(model, x + 0.5 * dt * k1, t_mid, conditioning_vec)
                k3 = self._model_velocity(model, x + 0.5 * dt * k2, t_mid, conditioning_vec)
                t_next = torch.full((shape[0],), t_value + dt, device=device, dtype=dtype)
                k4 = self._model_velocity(model, x + dt * k3, t_next, conditioning_vec)
                x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            else:
                raise ValueError(f"Unknown integration method: {self.integration_method}")

        return x
