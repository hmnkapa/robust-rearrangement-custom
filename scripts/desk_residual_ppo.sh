#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Prefer this checkout over editable installs that may point at an older clone.
export PYTHONPATH="$REPO_ROOT/furniture-bench:$REPO_ROOT/furniture-bench/r3m:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CUDA_VISIBLE_DEVICES=1 python "$SCRIPT_DIR/auto_resume_residual_ppo.py" \
    --workdir "$REPO_ROOT" \
    --restart-delay 10 \
    -- \
    python -m src.train.residual_ppo \
    base_policy.wt_path=/home/hy/lq/outputs/2026-05-06/12-39-53.760711/models/crimson-microwave-10/actor_chkpt_last.pt \
    env.task=desk \
    env.randomness=low \
    num_env_steps=2000 \
    normalize_reward=false \
    total_timesteps=1000000000 \
    wandb.entity=null \
    checkpoint_interval=1 \
    debug=false \
    residual_l1=0.001 \
    residual_l2=0.001 \
    ent_coef=0.001 \
    env.desk_leg_rot_reward_weight=0.5
