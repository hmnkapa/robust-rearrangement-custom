import os
from pathlib import Path

from ipdb import set_trace as bp


import random
import time
from typing import Optional

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.behavior.diffusion import DiffusionPolicy
from src.behavior.residual_diffusion import ResidualDiffusionPolicy
from src.behavior.residual_mlp import ResidualMlpPolicy
from src.eval.eval_utils import get_model_from_api_or_cached
from diffusers.optimization import get_scheduler


from src.gym.env_rl_wrapper import RLPolicyEnvWrapper
from src.common.config_util import merge_base_bc_config_with_root_config
from src.gym.observation import DEFAULT_STATE_OBS

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

import wandb
from wandb.apis.public.runs import Run
from wandb.errors.util import CommError

from src.gym import get_rl_env
import gymnasium as gym

# Register the eval resolver for omegaconf
OmegaConf.register_new_resolver("eval", eval)


def _task_overrides_cfg() -> DictConfig:
    """Return only explicit Hydra task overrides as an OmegaConf object."""
    try:
        overrides = HydraConfig.get().overrides.task
    except ValueError:
        return OmegaConf.create()

    dotlist = []
    for override in overrides:
        if not override or override.startswith("hydra.") or override.startswith("~"):
            continue
        if "=" not in override:
            continue
        dotlist.append(override.lstrip("+"))
    return OmegaConf.from_dotlist(dotlist)


def _resolve_checkpoint_path(path: str) -> Path:
    checkpoint_path = Path(path).expanduser()
    if checkpoint_path.is_absolute():
        return checkpoint_path
    return Path(hydra.utils.get_original_cwd()) / checkpoint_path


def _resume_checkpoint_path(cfg: DictConfig) -> Optional[Path]:
    resume_cfg = cfg.get("resume")
    if resume_cfg is None:
        return None
    checkpoint_path = resume_cfg.get("checkpoint_path")
    if checkpoint_path in (None, ""):
        return None
    return _resolve_checkpoint_path(str(checkpoint_path))


def _load_local_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    print(f"Loading local resume checkpoint from {path}")
    return torch.load(path, map_location="cpu")


def _merge_checkpoint_config_with_overrides(
    cfg: DictConfig, checkpoint_state_dict: dict
) -> DictConfig:
    checkpoint_cfg = checkpoint_state_dict.get("config")
    if checkpoint_cfg is None:
        return cfg

    checkpoint_cfg = OmegaConf.create(checkpoint_cfg)
    OmegaConf.set_struct(checkpoint_cfg, False)
    merged_cfg = OmegaConf.merge(checkpoint_cfg, _task_overrides_cfg())
    OmegaConf.set_struct(merged_cfg, False)
    return merged_cfg


def _is_eval_iteration(iteration: int, cfg: DictConfig) -> bool:
    return (iteration - int(cfg.eval_first)) % cfg.eval_interval == 0


def _global_step_from_iteration(iteration: int, cfg: DictConfig) -> int:
    completed_train_iterations = sum(
        not _is_eval_iteration(idx, cfg) for idx in range(1, iteration + 1)
    )
    return int(completed_train_iterations * cfg.batch_size)


def _load_training_state(
    *,
    agent: nn.Module,
    optimizer_actor: optim.Optimizer,
    optimizer_critic: optim.Optimizer,
    lr_scheduler_actor,
    lr_scheduler_critic,
    state_dict: dict,
) -> None:
    model_state_dict = state_dict["model_state_dict"]
    if "actor_logstd" in model_state_dict:
        agent.residual_policy.load_state_dict(model_state_dict)
    else:
        agent.load_state_dict(model_state_dict)

    optimizer_actor.load_state_dict(state_dict["optimizer_actor_state_dict"])
    optimizer_critic.load_state_dict(state_dict["optimizer_critic_state_dict"])
    lr_scheduler_actor.load_state_dict(state_dict["scheduler_actor_state_dict"])
    lr_scheduler_critic.load_state_dict(state_dict["scheduler_critic_state_dict"])


@torch.no_grad()
def calculate_advantage(
    values: torch.Tensor,
    next_value: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_done: torch.Tensor,
    steps_per_iteration: int,
    discount: float,
    gae_lambda: float,
):
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(steps_per_iteration)):
        if t == steps_per_iteration - 1:
            nextnonterminal = 1.0 - next_done.to(torch.float)
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1].to(torch.float)
            nextvalues = values[t + 1]

        delta = rewards[t] + discount * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = (
            delta + discount * gae_lambda * nextnonterminal * lastgaelam
        )
    returns = advantages + values
    return advantages, returns


@hydra.main(
    config_path="../config",
    config_name="base_residual_rl",
    version_base="1.2",
)
def main(cfg: DictConfig):

    OmegaConf.set_struct(cfg, False)

    resume_checkpoint_path = _resume_checkpoint_path(cfg)
    run_state_dict = None
    if resume_checkpoint_path is not None:
        run_state_dict = _load_local_checkpoint(resume_checkpoint_path)
        cfg = _merge_checkpoint_config_with_overrides(cfg, run_state_dict)

    if (job_id := os.environ.get("SLURM_JOB_ID")) is not None:
        cfg.slurm_job_id = job_id

    # Ensure exactly one of cfg.base_policy.wandb_id or cfg.base_policy.wt_path is set
    if resume_checkpoint_path is None:
        assert (
            sum(
                [
                    cfg.base_policy.wandb_id is not None,
                    cfg.base_policy.wt_path is not None,
                ]
            )
            == 1
        ), "Exactly one of base_policy.wandb_id or base_policy.wt_path must be set"

    # Check if we are continuing a run
    run_exists = False
    if resume_checkpoint_path is None and cfg.wandb.continue_run_id is not None:
        try:
            run: Run = wandb.Api().run(
                f"{cfg.wandb.project}/{cfg.wandb.continue_run_id}"
            )
            run_exists = True
        except (ValueError, CommError):
            pass

    if run_exists:
        print(f"Continuing run {cfg.wandb.continue_run_id}, {run.name}")

        run_id = cfg.wandb.continue_run_id
        run_path = f"{cfg.wandb.project}/{run_id}"

        # Load the weights from the run
        cfg, wts = get_model_from_api_or_cached(
            run_path, "latest", wandb_mode=cfg.wandb.mode
        )

        # Update the cfg.continue_run_id to the run_id
        cfg.wandb.continue_run_id = run_id

        base_cfg = cfg.base_policy
        merge_base_bc_config_with_root_config(cfg, base_cfg)

        print(f"Loading weights from {wts}")

        run_state_dict = torch.load(wts)

        # Set the best test loss and success rate to the one from the run
        try:
            best_eval_success_rate = run.summary["eval/best_eval_success_rate"]
        except KeyError:
            best_eval_success_rate = run.summary["eval/success_rate"]

        iteration = run.summary["iteration"]
        global_step = run.lastHistoryStep
        sps = run.summary.get("charts/SPS", run.summary.get("training/SPS", 0))
        training_cum_time = sps * global_step
        run_name = run.name

    elif resume_checkpoint_path is not None:
        print(f"Resuming run from local checkpoint {resume_checkpoint_path}")

        base_cfg = cfg.base_policy
        merge_base_bc_config_with_root_config(cfg, base_cfg)

        if "actor_name" not in cfg or cfg.actor_name is None:
            cfg.actor_name = f"residual_{cfg.base_policy.actor.name}"
        if cfg.seed is None:
            cfg.seed = random.randint(0, 2**32 - 1)

        iteration = int(run_state_dict.get("iteration", 0))
        global_step = int(
            run_state_dict.get(
                "global_step", _global_step_from_iteration(iteration, cfg)
            )
        )
        best_eval_success_rate = run_state_dict.get("best_eval_success_rate")
        if best_eval_success_rate is None:
            best_eval_success_rate = (
                run_state_dict.get("success_rate", 0.0)
                if resume_checkpoint_path.name == "actor_chkpt_best_success_rate.pt"
                else 0.0
            )
        best_eval_success_rate = float(best_eval_success_rate)
        training_cum_time = float(run_state_dict.get("training_cum_time", 0.0))
        run_name = resume_checkpoint_path.parent.name

    else:
        global_step = 0
        iteration = 0
        best_eval_success_rate = 0.0
        training_cum_time = 0

        # Load the behavior cloning actor
        if cfg.base_policy.wandb_id is not None:
            base_cfg, base_wts = get_model_from_api_or_cached(
                cfg.base_policy.wandb_id,
                wt_type=cfg.base_policy.wt_type,
                wandb_mode=cfg.wandb.mode,
            )
        elif cfg.base_policy.wt_path is not None:
            base_wts = cfg.base_policy.wt_path
            base_cfg: DictConfig = OmegaConf.create(torch.load(base_wts)["config"])
        else:
            raise ValueError("No base policy provided")

        merge_base_bc_config_with_root_config(cfg, base_cfg)
        cfg.actor_name = f"residual_{cfg.base_policy.actor.name}"

        if cfg.seed is None:
            cfg.seed = random.randint(0, 2**32 - 1)

        run_name = f"{int(time.time())}__{cfg.actor_name}_ppo__{cfg.seed}"

    if "task" not in cfg.env:
        cfg.env.task = "one_leg"

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    gpu_id = cfg.gpu_id
    device = torch.device(f"cuda:{gpu_id}")

    env: gym.Env = get_rl_env(
        gpu_id=gpu_id,
        act_rot_repr=cfg.control.act_rot_repr,
        action_type=cfg.control.control_mode,
        april_tags=False,
        concat_robot_state=True,
        ctrl_mode=cfg.control.controller,
        obs_keys=DEFAULT_STATE_OBS,
        task=cfg.env.task,
        # compute_device_id=gpu_id,
        # graphics_device_id=gpu_id,
        headless=cfg.headless,
        num_envs=cfg.num_envs,
        observation_space="state",
        randomness=cfg.env.randomness,
        max_env_steps=100_000_000,
        desk_insert_reward=cfg.env.desk_insert_reward,
        desk_success_reward=cfg.env.desk_success_reward,
        desk_twist_target_deg=cfg.env.desk_twist_target_deg,
        desk_twist_round_deg=cfg.env.desk_twist_round_deg,
        desk_twist_total_reward=cfg.env.desk_twist_total_reward,
        desk_twist_axis_sign=cfg.env.desk_twist_axis_sign,
        desk_contact_reward_weight=cfg.env.desk_contact_reward_weight,
        desk_release_reward_weight=cfg.env.desk_release_reward_weight,
        desk_contact_reward_scale=cfg.env.desk_contact_reward_scale,
        desk_contact_threshold=cfg.env.desk_contact_threshold,
        desk_release_contact_threshold=cfg.env.desk_release_contact_threshold,
        desk_contact_key_y=cfg.env.desk_contact_key_y,
        desk_contact_surface=cfg.env.desk_contact_surface,
        desk_twist_delta_clip_deg=cfg.env.desk_twist_delta_clip_deg,
        desk_twist_progress_threshold_deg=cfg.env.desk_twist_progress_threshold_deg,
        desk_no_progress_limit=cfg.env.desk_no_progress_limit,
        desk_wrist_limit_margin_rad=cfg.env.desk_wrist_limit_margin_rad,
        desk_wrist_reset_threshold_rad=cfg.env.desk_wrist_reset_threshold_rad,
    )

    n_parts_to_assemble = env.n_parts_assemble
    is_desk_task = cfg.env.task == "desk"

    if cfg.base_policy.actor.name == "diffusion":
        agent = ResidualDiffusionPolicy(device, base_cfg)
    elif cfg.base_policy.actor.name == "mlp":
        agent = ResidualMlpPolicy(device, base_cfg)
    else:
        raise ValueError(f"Unknown actor type: {cfg.base_policy.actor}")

    agent.to(device)
    agent.eval()

    # Set the inference steps of the actor
    if isinstance(agent, DiffusionPolicy):
        agent.inference_steps = 4

    env: RLPolicyEnvWrapper = RLPolicyEnvWrapper(
        env,
        max_env_steps=cfg.num_env_steps,
        normalize_reward=cfg.normalize_reward,
        reset_on_success=cfg.reset_on_success,
        reset_on_failure=cfg.reset_on_failure,
        reward_clip=cfg.clip_reward,
        sample_perturbations=cfg.sample_perturbations,
        device=device,
    )

    optimizer_actor = optim.AdamW(
        agent.actor_parameters,
        lr=cfg.learning_rate_actor,
        betas=cfg.get("optimizer_betas_actor", (0.9, 0.999)),
        eps=1e-5,
        weight_decay=1e-6,
    )

    lr_scheduler_actor = get_scheduler(
        name=cfg.lr_scheduler.name,
        optimizer=optimizer_actor,
        num_warmup_steps=cfg.lr_scheduler.actor_warmup_steps,
        num_training_steps=cfg.num_iterations,
    )

    optimizer_critic = optim.AdamW(
        agent.critic_parameters,
        lr=cfg.learning_rate_critic,
        eps=1e-5,
        weight_decay=1e-6,
    )

    lr_scheduler_critic = get_scheduler(
        name=cfg.lr_scheduler.name,
        optimizer=optimizer_critic,
        num_warmup_steps=cfg.lr_scheduler.critic_warmup_steps,
        num_training_steps=cfg.num_iterations,
    )

    if run_state_dict is not None:
        _load_training_state(
            agent=agent,
            optimizer_actor=optimizer_actor,
            optimizer_critic=optimizer_critic,
            lr_scheduler_actor=lr_scheduler_actor,
            lr_scheduler_critic=lr_scheduler_critic,
            state_dict=run_state_dict,
        )
    else:
        agent.load_base_state_dict(base_wts)

    residual_policy = agent.residual_policy

    has_pretrained_wts = (
        "pretrained_wts" in cfg.actor.residual_policy
        and cfg.actor.residual_policy.pretrained_wts
    )
    if run_state_dict is None and has_pretrained_wts:
        print(
            f"Loading pretrained weights from {cfg.actor.residual_policy.pretrained_wts}"
        )
        run_state_dict = torch.load(cfg.actor.residual_policy.pretrained_wts)
        _load_training_state(
            agent=agent,
            optimizer_actor=optimizer_actor,
            optimizer_critic=optimizer_critic,
            lr_scheduler_actor=lr_scheduler_actor,
            lr_scheduler_critic=lr_scheduler_critic,
            state_dict=run_state_dict,
        )

    steps_per_iteration = cfg.data_collection_steps

    print(f"Total timesteps: {cfg.total_timesteps}, batch size: {cfg.batch_size}")
    print(
        f"Mini-batch size: {cfg.minibatch_size}, num iterations: {cfg.num_iterations}"
    )

    print(OmegaConf.to_yaml(cfg, resolve=True))

    run = wandb.init(
        id=cfg.wandb.continue_run_id,
        resume=None if cfg.wandb.continue_run_id is None else "allow",
        project=cfg.wandb.project,
        entity=cfg.wandb.get("entity", None),
        config=OmegaConf.to_container(cfg, resolve=True),
        name=run_name,
        save_code=True,
        mode=cfg.wandb.mode if not cfg.debug else "disabled",
    )

    obs: torch.Tensor = torch.zeros(
        (
            steps_per_iteration,
            cfg.num_envs,
            residual_policy.obs_dim,
        )
    )
    actions = torch.zeros((steps_per_iteration, cfg.num_envs) + env.action_space.shape)
    logprobs = torch.zeros((steps_per_iteration, cfg.num_envs))
    rewards = torch.zeros((steps_per_iteration, cfg.num_envs))
    assembly_rewards = torch.zeros((steps_per_iteration, cfg.num_envs))
    insert_rewards = torch.zeros((steps_per_iteration, cfg.num_envs))
    twist_rewards = torch.zeros((steps_per_iteration, cfg.num_envs))
    contact_rewards = torch.zeros((steps_per_iteration, cfg.num_envs))
    release_rewards = torch.zeros((steps_per_iteration, cfg.num_envs))
    success_rewards = torch.zeros((steps_per_iteration, cfg.num_envs))
    dones = torch.zeros((steps_per_iteration, cfg.num_envs))
    values = torch.zeros((steps_per_iteration, cfg.num_envs))

    start_time = time.time()

    next_done = torch.zeros(cfg.num_envs)
    next_obs = env.reset()
    agent.reset()

    # Create model save dir
    model_save_dir: Path = Path("models") / wandb.run.name
    model_save_dir.mkdir(parents=True, exist_ok=True)

    def _info_reward(info, key, reward):
        value = info.get(key, torch.zeros_like(reward))
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, device=reward.device)
        return value.view(-1).detach().cpu()

    while global_step < cfg.total_timesteps:
        iteration += 1
        print(f"Iteration: {iteration}/{cfg.num_iterations}")
        print(f"Run name: {run_name}")
        iteration_start_time = time.time()

        # If eval first flag is set, we will evaluate the model before doing any training
        eval_mode = (iteration - int(cfg.eval_first)) % cfg.eval_interval == 0

        # Also reset the env to have more consistent results
        if eval_mode or cfg.reset_every_iteration:
            next_obs = env.reset()
            agent.reset()

        print(f"Eval mode: {eval_mode}")

        for step in range(0, steps_per_iteration):
            if not eval_mode:
                # Only count environment steps during training
                global_step += cfg.num_envs

            # Get the base normalized action
            base_naction = agent.base_action_normalized(next_obs)

            # Process the obs for the residual policy
            next_nobs = agent.process_obs(next_obs)
            next_residual_nobs = torch.cat([next_nobs, base_naction], dim=-1)

            dones[step] = next_done
            obs[step] = next_residual_nobs

            with torch.no_grad():
                residual_naction_samp, logprob, _, value, naction_mean = (
                    residual_policy.get_action_and_value(next_residual_nobs)
                )

            residual_naction = residual_naction_samp if not eval_mode else naction_mean
            naction = base_naction + residual_naction * residual_policy.action_scale

            action = agent.normalizer(naction, "action", forward=False)
            next_obs, reward, next_done, truncated, info = env.step(action)

            if cfg.truncation_as_done:
                next_done = next_done | truncated

            values[step] = value.flatten().cpu()
            actions[step] = residual_naction.cpu()
            logprobs[step] = logprob.cpu()
            rewards[step] = reward.view(-1).cpu()
            assembly_rewards[step] = _info_reward(info, "assembly_reward", reward)
            insert_rewards[step] = _info_reward(info, "desk_insert_reward", reward)
            twist_rewards[step] = _info_reward(info, "desk_twist_reward", reward)
            contact_rewards[step] = _info_reward(info, "desk_contact_reward", reward)
            release_rewards[step] = _info_reward(info, "desk_release_reward", reward)
            success_rewards[step] = _info_reward(info, "desk_success_reward", reward)
            next_done = next_done.view(-1).cpu()

            if step > 0 and (env_step := step * 1) % 100 == 0:
                print(
                    f"env_step={env_step}, global_step={global_step}, "
                    f"mean_reward={rewards[:step+1].sum(dim=0).mean().item()}, "
                    f"mean_twist_reward={twist_rewards[:step+1].sum(dim=0).mean().item()} "
                    f"fps={env_step * cfg.num_envs / (time.time() - iteration_start_time):.2f}"
                )

        # Calculate the success rate
        # Find the rewards that are not zero
        # Env is successful if it received a reward more than or equal to n_parts_to_assemble
        if is_desk_task:
            success_reward_unit = max(float(cfg.env.desk_success_reward), 1e-6)
            progress_units = success_rewards / success_reward_unit
        else:
            progress_units = assembly_rewards
        env_success = progress_units.sum(dim=0) >= n_parts_to_assemble
        mean_reward = rewards.sum(dim=0).mean().item()
        mean_assembly_reward = assembly_rewards.sum(dim=0).mean().item()
        mean_insert_reward = insert_rewards.sum(dim=0).mean().item()
        mean_twist_reward = twist_rewards.sum(dim=0).mean().item()
        mean_contact_reward = contact_rewards.sum(dim=0).mean().item()
        mean_release_reward = release_rewards.sum(dim=0).mean().item()
        mean_success_reward = success_rewards.sum(dim=0).mean().item()
        success_rate = env_success.float().mean().item()

        if success_rate > 0:
            # Calculate the share of timesteps that come from successful trajectories that account for the success rate and the varying number of timesteps per trajectory
            # Count total timesteps in successful trajectories
            timesteps_in_success = progress_units[:, env_success]

            # Find index of last reward in each trajectory
            # This has all timesteps including and after episode is done
            success_dones = timesteps_in_success.cumsum(dim=0) >= n_parts_to_assemble
            last_reward_idx = success_dones.int().argmax(dim=0)

            # Calculate the total number of timesteps in successful trajectories
            total_timesteps_in_success = (last_reward_idx + 1).sum().item()

            # Calculate the share of successful timesteps
            success_timesteps_share = total_timesteps_in_success / rewards.numel()

            # Mean successful episode length
            mean_success_episode_length = (
                total_timesteps_in_success / env_success.sum().item()
            )
            max_success_episode_length = last_reward_idx.max().item()
        else:
            success_timesteps_share = 0
            mean_success_episode_length = 0
            max_success_episode_length = 0

        print(
            f"SR: {success_rate:.4%}, SPS: {steps_per_iteration * cfg.num_envs / (time.time() - iteration_start_time):.2f}"
            f", STS: {success_timesteps_share:.4%}, MSEL: {mean_success_episode_length:.2f}"
        )

        if eval_mode:
            # If we are in eval mode, we don't need to do any training, so log the result and continue

            # Save the model if the evaluation success rate improves
            if success_rate > best_eval_success_rate:
                best_eval_success_rate = success_rate
                model_path = str(model_save_dir / f"actor_chkpt_best_success_rate.pt")
                torch.save(
                    {
                        # Save the weights of the residual policy (base + residual)
                        "model_state_dict": agent.state_dict(),
                        "optimizer_actor_state_dict": optimizer_actor.state_dict(),
                        "optimizer_critic_state_dict": optimizer_critic.state_dict(),
                        "scheduler_actor_state_dict": lr_scheduler_actor.state_dict(),
                        "scheduler_critic_state_dict": lr_scheduler_critic.state_dict(),
                        "config": OmegaConf.to_container(cfg, resolve=True),
                        "success_rate": success_rate,
                        "success_timesteps_share": success_timesteps_share,
                        "best_eval_success_rate": best_eval_success_rate,
                        "iteration": iteration,
                        "global_step": global_step,
                        "training_cum_time": training_cum_time,
                    },
                    model_path,
                )

                wandb.save(model_path)
                print(f"Evaluation success rate improved. Model saved to {model_path}")

            wandb.log(
                {
                    "eval/success_rate": success_rate,
                    "eval/mean_reward": mean_reward,
                    "eval/mean_assembly_reward": mean_assembly_reward,
                    "eval/mean_insert_reward": mean_insert_reward,
                    "eval/mean_twist_reward": mean_twist_reward,
                    "eval/mean_contact_reward": mean_contact_reward,
                    "eval/mean_release_reward": mean_release_reward,
                    "eval/mean_success_reward": mean_success_reward,
                    "eval/best_eval_success_rate": best_eval_success_rate,
                    "iteration": iteration,
                },
                step=global_step,
            )
            # Start the data collection again
            # NOTE: We're not resetting here now, that happens before the next
            # iteration only if the reset_every_iteration flag is set
            continue

        b_obs = obs.reshape((-1, residual_policy.obs_dim))
        b_actions = actions.reshape((-1,) + env.action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_values = values.reshape(-1)

        # Get the base normalized action
        # Process the obs for the residual policy
        base_naction = agent.base_action_normalized(next_obs)
        next_nobs = agent.process_obs(next_obs)
        next_residual_nobs = torch.cat([next_nobs, base_naction], dim=-1)
        next_value = residual_policy.get_value(next_residual_nobs).reshape(1, -1).cpu()

        # bootstrap value if not done
        advantages, returns = calculate_advantage(
            values,
            next_value,
            rewards,
            dones,
            next_done,
            steps_per_iteration,
            cfg.discount,
            cfg.gae_lambda,
        )

        b_advantages = advantages.reshape(-1).cpu()
        b_returns = returns.reshape(-1).cpu()

        # Optimizing the policy and value network
        b_inds = np.arange(cfg.batch_size)
        clipfracs = []
        for epoch in trange(cfg.update_epochs, desc="Policy update"):
            early_stop = False

            np.random.shuffle(b_inds)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_inds = b_inds[start:end]

                # Get the minibatch and place it on the device
                mb_obs = b_obs[mb_inds].to(device)
                mb_actions = b_actions[mb_inds].to(device)
                mb_logprobs = b_logprobs[mb_inds].to(device)
                mb_advantages = b_advantages[mb_inds].to(device)
                mb_returns = b_returns[mb_inds].to(device)
                mb_values = b_values[mb_inds].to(device)

                # Calculate the loss
                _, newlogprob, entropy, newvalue, action_mean = (
                    residual_policy.get_action_and_value(mb_obs, mb_actions)
                )
                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                    ]

                if cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                policy_loss = 0

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - mb_returns) ** 2
                    v_clipped = mb_values + torch.clamp(
                        newvalue - mb_values,
                        -cfg.clip_coef,
                        cfg.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean() * cfg.ent_coef

                ppo_loss = pg_loss - entropy_loss

                # Add the auxiliary regularization loss
                residual_l1_loss = torch.mean(torch.abs(action_mean))
                residual_l2_loss = torch.mean(torch.square(action_mean))

                # Normalize the losses so that each term has the same scale
                if iteration > cfg.n_iterations_train_only_value:

                    # Scale the losses using the calculated scaling factors
                    policy_loss += ppo_loss
                    policy_loss += cfg.residual_l1 * residual_l1_loss
                    policy_loss += cfg.residual_l2 * residual_l2_loss

                # Total loss
                loss: torch.Tensor = policy_loss + v_loss * cfg.vf_coef

                optimizer_actor.zero_grad()
                optimizer_critic.zero_grad()

                loss.backward()
                nn.utils.clip_grad_norm_(
                    residual_policy.parameters(), cfg.max_grad_norm
                )

                optimizer_actor.step()
                optimizer_critic.step()

                if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                    print(
                        f"Early stopping at epoch {epoch} due to reaching max kl: {approx_kl:.4f} > {cfg.target_kl:.4f}"
                    )
                    early_stop = True
                    break

            if early_stop:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        action_norms = torch.norm(b_actions[:, :3], dim=-1).cpu()

        training_cum_time += time.time() - iteration_start_time
        sps = int(global_step / training_cum_time) if training_cum_time > 0 else 0

        wandb.log(
            {
                "training/learning_rate_actor": optimizer_actor.param_groups[0]["lr"],
                "training/learning_rate_critic": optimizer_critic.param_groups[0]["lr"],
                "training/SPS": sps,
                "charts/rewards": rewards.sum().item(),
                "charts/mean_reward": mean_reward,
                "charts/assembly_rewards": assembly_rewards.sum().item(),
                "charts/mean_assembly_reward": mean_assembly_reward,
                "charts/insert_rewards": insert_rewards.sum().item(),
                "charts/mean_insert_reward": mean_insert_reward,
                "charts/twist_rewards": twist_rewards.sum().item(),
                "charts/mean_twist_reward": mean_twist_reward,
                "charts/contact_rewards": contact_rewards.sum().item(),
                "charts/mean_contact_reward": mean_contact_reward,
                "charts/release_rewards": release_rewards.sum().item(),
                "charts/mean_release_reward": mean_release_reward,
                "charts/success_rewards": success_rewards.sum().item(),
                "charts/mean_success_reward": mean_success_reward,
                "charts/success_rate": success_rate,
                "charts/success_timesteps_share": success_timesteps_share,
                "charts/mean_success_episode_length": mean_success_episode_length,
                "charts/max_success_episode_length": max_success_episode_length,
                "charts/action_norm_mean": action_norms.mean(),
                "charts/action_norm_std": action_norms.std(),
                "values/advantages": b_advantages.mean().item(),
                "values/returns": b_returns.mean().item(),
                "values/values": b_values.mean().item(),
                "values/mean_logstd": residual_policy.actor_logstd.mean().item(),
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/total_loss": loss.item(),
                "losses/entropy_loss": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": np.mean(clipfracs),
                "losses/explained_variance": explained_var,
                "losses/residual_l1": residual_l1_loss.item(),
                "losses/residual_l2": residual_l2_loss.item(),
                "histograms/values": wandb.Histogram(values),
                "histograms/returns": wandb.Histogram(b_returns),
                "histograms/advantages": wandb.Histogram(b_advantages),
                "histograms/logprobs": wandb.Histogram(logprobs),
                "histograms/rewards": wandb.Histogram(rewards),
                "histograms/assembly_rewards": wandb.Histogram(assembly_rewards),
                "histograms/insert_rewards": wandb.Histogram(insert_rewards),
                "histograms/twist_rewards": wandb.Histogram(twist_rewards),
                "histograms/contact_rewards": wandb.Histogram(contact_rewards),
                "histograms/release_rewards": wandb.Histogram(release_rewards),
                "histograms/success_rewards": wandb.Histogram(success_rewards),
                "histograms/action_norms": wandb.Histogram(action_norms),
            },
            step=global_step,
        )

        # Step the learning rate scheduler
        lr_scheduler_actor.step()
        lr_scheduler_critic.step()

        # Checkpoint every cfg.checkpoint_interval steps
        if cfg.checkpoint_interval > 0 and iteration % cfg.checkpoint_interval == 0:
            model_path = str(model_save_dir / f"actor_chkpt_{iteration}.pt")
            torch.save(
                {
                    "model_state_dict": agent.state_dict(),
                    "optimizer_actor_state_dict": optimizer_actor.state_dict(),
                    "optimizer_critic_state_dict": optimizer_critic.state_dict(),
                    "scheduler_actor_state_dict": lr_scheduler_actor.state_dict(),
                    "scheduler_critic_state_dict": lr_scheduler_critic.state_dict(),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "success_rate": success_rate,
                    "best_eval_success_rate": best_eval_success_rate,
                    "iteration": iteration,
                    "global_step": global_step,
                    "training_cum_time": training_cum_time,
                },
                model_path,
            )

            wandb.save(model_path)
            print(f"Model saved to {model_path}")

        # Print some stats at the end of the iteration
        print(
            f"Iteration {iteration}/{cfg.num_iterations}, global step {global_step}, SPS {sps}"
        )

    print(f"Training finished in {(time.time() - start_time):.2f}s")


if __name__ == "__main__":
    main()
