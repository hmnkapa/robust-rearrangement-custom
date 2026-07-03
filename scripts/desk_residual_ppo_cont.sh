#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Prefer this checkout over editable installs that may point at an older clone.
export PYTHONPATH="$REPO_ROOT/furniture-bench:$REPO_ROOT/furniture-bench/r3m:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CUDA_VISIBLE_DEVICES=2 python "$SCRIPT_DIR/auto_resume_residual_ppo.py" \
    --workdir "$REPO_ROOT" \
    --restart-delay 10 \
    -- \
    python -m src.train.residual_ppo \
    init_from.checkpoint_path=/home/hy/lq/robust-rearrangement-custom-fix-contact-reward/actor_chkpt_98.pt \
    env.task=desk \
    env.randomness=low \
    num_env_steps=4000 \
    num_envs=2048 \
    update_epochs=50 \
    normalize_reward=false \
    total_timesteps=2000000000 \
    num_minibatches=8 \
    wandb.entity=null \
    checkpoint_interval=1 \
    debug=false \
    residual_l1=0.001 \
    residual_l2=0.001 \
    ent_coef=0.001 \
    env.desk_insert_reward=15.0 \
    env.desk_success_reward=2.0 \
    env.desk_twist_total_reward=2.0 \
    env.desk_contact_reward_weight=0.06 \
    env.desk_release_reward_weight=0.02 \
    env.desk_twist_axis_sign=-1.0 \
    env.desk_twist_target_deg=200.0 \
    env.desk_top_yaw_total_reward=2.0 \
    env.desk_contact_reward_max_attempts=6
