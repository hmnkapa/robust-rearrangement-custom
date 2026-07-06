#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Prefer this checkout over editable installs that may point at an older clone.
export PYTHONPATH="$REPO_ROOT/furniture-bench:$REPO_ROOT/furniture-bench/r3m:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CUDA_VISIBLE_DEVICES=7 python "$SCRIPT_DIR/auto_resume_residual_ppo.py" \
    --workdir "$REPO_ROOT" \
    --restart-delay 10 \
    -- \
    python -m src.train.residual_ppo \
    base_policy.wt_path=/home/hy/lq/outputs/2026-05-06/12-39-53.760711/models/crimson-microwave-10/actor_chkpt_last.pt \
    env.task=desk \
    env.randomness=low \
    env.initial_state_dir=two_leg_yaw \
    env.initial_state_desk_phase=top_yaw_complete \
    num_env_steps=3000 \
    num_envs=2048 \
    update_epochs=100 \
    normalize_reward=false \
    total_timesteps=1500000000 \
    num_minibatches=16 \
    wandb.entity=null \
    checkpoint_interval=1 \
    debug=false \
    residual_l1=0.001 \
    residual_l2=0.001 \
    ent_coef=0.001 \
    env.desk_insert_reward=2.0 \
    env.desk_success_reward=2.0 \
    env.desk_twist_total_reward=2.0 \
    env.desk_contact_reward_weight=0.15 \
    env.desk_proximity_reward_weight=0.10 \
    env.desk_release_reward_weight=0.10 \
    env.desk_proximity_reward_scale=0.02 \
    env.desk_twist_axis_sign=-1.0 \
    env.desk_twist_target_deg=200.0 \
    env.desk_top_yaw_total_reward=0.0 \
    env.desk_top_yaw_bonus_reward=0.0 \
    env.desk_contact_reward_max_attempts=6
