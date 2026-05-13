# Residual RL — Wandb 指标说明与四任务 Reward 统计

> 论文: [From Imitation to Refinement: Residual RL for Precise Assembly](https://arxiv.org/abs/2407.16677)
>
> 代码: `src/train/residual_ppo.py`, `src/train/residual_ppo_w_bc.py`
>
> 本文档统计了 one_leg, round_table, lamp, desk 四个任务的 residual RL 训练 reward 与成功率，并逐项解释 wandb 记录的所有指标。

---

## 1. 奖励函数设计

Residual RL 使用**奖励函数** `_reward()` 于 [furniture_rl_sim_env.py:2291](furniture-bench/furniture_bench/envs/furniture_rl_sim_env.py#L2291):

- **每成功组装一对部件 = +1.0** (sparse reward)
- 需连续 `assembly_confirm_frames`(3~5) 帧都在装配阈值内，才判定 "新组装完成"，防止瞬时碰撞误判
- 已组装过的部件不再产生 reward（`already_assembled` mask）
- 训练中 `normalize_reward: true`（默认），用 `RunningMeanStdClip` 做 reward normalization，clip 在 ±5.0
- 判定成功: `sum(rewards) >= n_parts_to_assemble`

| 任务 | n_parts_to_assemble | max reward/episode | 单 reward 值 |
|------|-------------------|-------------------|-------------|
| one_leg | 1 | 1.0 | 1.0 |
| lamp | 2 | 2.0 | 1.0 |
| round_table | 2 | 2.0 | 1.0 |
| desk | 4 | 4.0 | 1.0 |

---

## 2. 四任务训练结果（来自 RPPO Checkpoints）

数据来源: `checkpoints/rppo/{task}/low/actor_chkpt.pt`

| 指标 | one_leg | lamp | round_table | desk |
|------|---------|------|-------------|------|
| **training success_rate** | **97.46%** | **98.24%** | **95.99%** | **无训练数据** |
| success_timesteps_share | 58.1% | 23.9% | 13.3% | — |
| 训练 iteration | 306 | 571 | 921 | — |
| reward/成功 episode | 1.0 | 2.0 | 2.0 | — |
| 训练耗时 | ~18h | — | — | — |

### 分析

- **one_leg 收敛最快** (306 iter, 58% step 占比): 1 个部件对，探索负担最小
- **lamp 收敛稳健** (571 iter, 24% step 占比): 2 个部件有**有序依赖**（先 bulb 后 shade），base policy 提供了较强的先验
- **round_table 收敛最慢** (921 iter, 13% step 占比): 2 个部件，table_top 的 push 阶段需要精确对齐，耗时最长
- **desk 未训练**: `checkpoints/bc/` 和 `checkpoints/rppo/` 均无 desk 目录，wandb 无相关 run

### Round_table Eval 详细数据

本地 JSON log 路径: `logs/evaluate_model/round_table/`

**RPPO Actor 评估（state-based residual policy）**:

| Checkpoint | n_rollouts | n_success | success_rate | total_reward | avg_reward/成功 |
|-----------|-----------|----------|-------------|-------------|---------------|
| actor_chkpt (04-28 13:20) | 200 | 198 | **99.0%** | 398.0 | 2.01 |
| actor_chkpt (05-12 14:57) | 212 | 200 | **94.3%** | 410.0 | 2.05 |
| actor_chkpt (04-28 23:22) | 1 | 1 | 100% | 2.0 | 2.0 |
| actor_chkpt (04-28 13:30) | 4 | 3 | 75.0% | 7.0 | 2.33 |
| actor_chkpt (05-07 17:45) | 1 | 0 | 0.0% | 0.0 | — |

**RGBD Skill 策略评估（image-based，residual diffusion）**:

| Checkpoint | n_rollouts | n_success | success_rate |
|-----------|-----------|----------|-------------|
| rgbd_skill_dit_200traj | 36 | 23 | **63.9%** |
| rgbd_skill_fmt_200traj | 36 | 19 | 52.8% |
| rgbd_skill_dit_100traj | 36 | 18 | 50.0% |
| rgbd_skill_fmt_100traj | 36 | 13 | 36.1% |

**瓶颈环节**（RPPO actor best checkpoint 的 per-skill 成功率）: `base-leg-screw` (99.5%) 和 `leg-top-pick` (100%) 基本良好，RGBD skill 策略中 `base-leg-screw` (85~96%) 是主要瓶颈。

---

## 3. Residual PPO 训练流程

### 3.1 架构

```
Base Policy (BC, frozen)          Residual Policy (MLP, trainable)
        |                                    |
   a_base = π_base(s)         a_res ~ N(μ_res(s, a_base), σ²)
        |                                    |
        +----------- merged ------------------+
                     |
          a = a_base + α × a_res    (α = action_scale = 0.1)
```

- **Base policy**: 预训练的 BC 策略（DiffusionPolicy 或 MLPPolicy），frozen，提供基础行为
- **Residual policy**: 小型 MLP（2 层 256 维），输入 `[normalized_obs, base_action]`，输出残差动作
- **Actor mean head**: 输出 deterministic residual `action_mean`
- **Actor log_std**: 独立可学习参数（默认 `learn_std=false`，固定 `init_logstd=-1`），控制探索噪声
- **Critic**: 独立 MLP（2 层 256 维），输出 scalar value V(s)

### 3.2 数据量

| 参数 | 值 | 说明 |
|------|-----|------|
| 并行环境数 | 1,024 | Isaac Gym GPU 并行 |
| 每轮收集步数 (`num_env_steps`) | 700 / 1000 | 每个 env 每 iteration 执行的步数 |
| **batch_size** | 716,800 / 1,024,000 | = num_envs × num_env_steps，每轮收集的 transition 数 |
| minibatch_size | = batch_size | 不拆 minibatch（`num_minibatches=1`） |
| PPO 更新轮数 (`update_epochs`) | 50 | 同一批数据重复用 50 次 |
| 总环境步数 (`total_timesteps`) | 1,000,000,000 | 1B |
| 总 iteration | ~1,395 (700步时) / ~976 (1000步时) | = 1e9 / batch_size |
| eval 频率 | 每 5 轮 | eval 时确定性执行（`action=action_mean`，不加噪声） |

### 3.3 伪代码

```python
# ===== 初始化 =====
env = create_parallel_envs(num_envs=1024, task="one_leg")  # 1024 envs on GPU
n_parts = env.n_parts_assemble  # 1 / 2 / 4

# 加载 BC base policy (frozen)
base_policy = load_bc_checkpoint(wandb_id).eval()

# 初始化 residual policy (小 MLP, 可训练)
residual = ResidualPolicy(
    obs_dim = |normalized_obs| + |base_action|,  # e.g. 58 + 7 = 65
    actor=[256, 256], critic=[256, 256],
    init_logstd=-1.0, learn_std=False,           # 固定 σ=exp(-1)≈0.37
    action_scale=0.1,
)
opt_actor  = AdamW(actor_params,  lr=3e-4)
opt_critic = AdamW(critic_params, lr=5e-3)

obs = env.reset()

# ===== 主循环 =====
for iteration in range(num_iterations):  # ~1000-1400

    # Step 1: 数据收集 (700 or 1000 steps × 1024 envs)
    buffer = []
    for step in range(num_env_steps):
        with torch.no_grad():
            a_base = base_policy.base_action_normalized(obs)         # (1024, 7)
            nobs  = base_policy.process_obs(obs)                     # (1024, 58)
        residual_obs = torch.cat([nobs, a_base], dim=-1)             # (1024, 65)

        a_res, logprob, entropy, value, action_mean = \
            residual.get_action_and_value(residual_obs)
        # a_res ~ N(action_mean, exp(init_logstd))  固定 std 的高斯采样
        # eval 模式直接用 action_mean (deterministic)

        a = a_base + 0.1 * a_res  # action_scale=0.1
        a_unnorm = base_policy.normalizer(a, "action", forward=False)
        next_obs, reward, done, truncated, info = env.step(a_unnorm)
        # reward ∈ {0, 1}, 1 = 新组装一对部件

        buffer.append((residual_obs, a_res, logprob, reward, done, value))

    # Step 2: 计算 GAE + Returns
    next_value = residual.get_value(next_residual_obs)
    advantages, returns = GAE(
        values=buffer.values, next_value=next_value,
        rewards=buffer.rewards, dones=buffer.dones,
        gamma=0.999, lambda_=0.95,
    )

    # Step 3: 训练成功率
    env_success = (buffer.rewards > 0).sum(dim=0) >= n_parts
    success_rate = env_success.float().mean()

    # Step 4: PPO 更新 (同一批数据 50 epochs)
    b_obs, b_actions, b_logprobs, b_values, b_adv, b_ret = flatten(buffer)  # → (B,)
    for epoch in range(50):
        # Forward
        _, new_logprob, entropy, new_value, action_mean = \
            residual.get_action_and_value(b_obs, b_actions)

        # PPO clipped policy loss (Actor)
        ratio = exp(new_logprob - b_logprobs)
        pg_loss = max(-adv·ratio, -adv·clip(ratio, 0.8, 1.2)).mean()

        # Value loss (Critic)
        v_loss = 0.5 * MSE(new_value, b_returns)

        # Residual regularization (Actor)
        res_l1 = mean(|action_mean|)    # 鼓励稀疏
        res_l2 = mean(action_mean²)     # 鼓励小幅度

        # Total loss: 同时反向传播更新 Actor + Critic
        total_loss = pg_loss + 1.0 * v_loss + 0.0 * res_l1 + 0.0 * res_l2

        opt_actor.zero_grad(); opt_critic.zero_grad()
        total_loss.backward()
        clip_grad_norm(residual.parameters(), 1.0)
        opt_actor.step(); opt_critic.step()

        # KL early stop: 防止策略崩溃
        approx_kl = mean((ratio - 1) - logratio)
        if approx_kl > 0.1: break  # target_kl

    # Step 5: Eval (每 5 轮)
    if iteration % 5 == 0:
        eval_sr = rollout_eval(deterministic=True)
        if eval_sr > best_eval_sr: save_checkpoint()

    # Step 6: Wandb logging
    wandb.log({...})  # 完整列表见 §4

    # Step 7: 重置环境 (reset_every_iteration=true)
    obs = env.reset()
```

### 3.4 关键设计要点

1. **Actor 和 Critic 同时更新**: `total_loss.backward()` 对 `residual.parameters()` 统一求梯度（actor_mean、critic、actor_logstd 从同一个 optimizer 更新）
2. **不做 minibatch 拆分**: `num_minibatches=1`，全量 batch 一次 forward/backward
3. **每轮重置环境**: `reset_every_iteration=true` 保证 episodes 干净开始
4. **KL early stop**: 新旧策略差异过大（KL>0.1）提前结束当前 iteration 的 PPO epoch
5. **前 N 轮只训 Critic**: `n_iterations_train_only_value=0`（默认），即一开始就同步训

---

## 4. Wandb 指标完整说明

所有指标在 [residual_ppo.py:604-638](src/train/residual_ppo.py#L604-L638) 记录，按 `global_step` (环境步数) 为 x 轴。

### 4.1 Training 组（训练元信息）

| 指标 | 含义 | 正常范围 |
|------|------|---------|
| `training/learning_rate_actor` | Actor 当前学习率，cosine decay | 3e-4 → 0 |
| `training/learning_rate_critic` | Critic 当前学习率，cosine decay | 5e-3 → 0 |
| `training/SPS` | 每秒环境步数 (Steps Per Second)，训练吞吐量 | 取决于硬件 |

### 4.2 Charts 组（核心训练曲线）

| 指标 | 含义 | 正常范围 |
|------|------|---------|
| `charts/rewards` | 当前 iteration 所有 env 的 reward **总和**。1.0 表示成功组装一对部件。one_leg 理论最大值 = num_envs × 1 = 1024; lamp/round_table = num_envs × 2 = 2048 | 0 → 逐步增大 |
| `charts/success_rate` | **训练成功率**: 多少比例的 env 完成了所有部件组装。判定: `sum(rewards, dim=0) >= n_parts_to_assemble` | 目标 >0.9 |
| `charts/success_timesteps_share` | 成功 episode 中用到的时间步占比。值越低说明策略越高效。round_table 仅 13% 说明大量步数耗在非成功 episode | 逐趋下降 |
| `charts/mean_success_episode_length` | 成功 episode 的绝对平均步长 | 趋近最低可行值 |
| `charts/max_success_episode_length` | 成功 episode 中最长步长 | 逐渐下降 |
| `charts/action_norm_mean` | **残差动作**前 3 维 (平移) 范数均值。衡量 residual 输出幅度 | 应较小 (~0.01) |
| `charts/action_norm_std` | 残差动作范数的标准差，衡量 residual 输出的波动 | 应较小 |

### 4.3 Values 组（PPO value/advantage 估计）

| 指标 | 含义 |
|------|------|
| `values/advantages` | GAE 优势函数均值，应接近 0 |
| `values/returns` | Discounted return 均值，Critic 要拟合的目标 |
| `values/values` | **Critic 网络预测的 V(s) 均值**，应追踪 returns |
| `values/mean_logstd` | **Actor 输出的 log_std 均值**。默认 `init_logstd=-1`、`learn_std=false`，恒定为 -1。`std = exp(-1) ≈ 0.37`，乘以 `action_scale=0.1` 得到有效噪声 ≈0.037 |

### 4.4 Losses 组（损失函数）

| 指标 | 对象 | 公式 / 含义 |
|------|------|------------|
| `losses/value_loss` | **Critic** | `0.5 × MSE(newvalue, returns)` |
| `losses/policy_loss` | **Actor** | PPO clipped objective: `max(-adv×ratio, -adv×clip(ratio, 1-ε, 1+ε))`，ε=0.2 |
| `losses/total_loss` | **两者** | `= policy_loss + residual_l1 + residual_l2 + vf_coef × value_loss` |
| `losses/entropy_loss` | **Actor** | `ent_coef × mean(entropy)`，鼓励探索 |
| `losses/old_approx_kl` | **Actor** | KL 近似 (low-variance): `mean(-logratio)` |
| `losses/approx_kl` | **Actor** | KL 近似: `mean((ratio-1) - logratio)`。触发 early stop 的阈值是 `target_kl=0.1` |
| `losses/clipfrac` | **Actor** | PPO clip 比例: `|ratio - 1| > 0.2` 的样本占比。接近 0 更新平稳；持续 >0.2 需降低 lr |
| `losses/explained_variance` | **Critic** | `1 - Var(returns - values) / Var(returns)`。接近 1 说明 Critic 拟合好，<0.5 说明 Critic 没学到东西 |
| `losses/residual_l1` | **Actor** | `mean(|action_mean|)`。鼓励 residual 输出**稀疏**（大部分维度为 0） |
| `losses/residual_l2` | **Actor** | `mean(action_mean²)`。鼓励 residual 输出**小幅度**，对 base policy 做最小修正 |

### 4.5 Histograms 组（分布直方图）

| 指标 | 内容 |
|------|------|
| `histograms/values` | Value 预测值分布 |
| `histograms/returns` | Discounted return 分布 |
| `histograms/advantages` | GAE 优势估计分布 |
| `histograms/logprobs` | Log probability 分布 |
| `histograms/rewards` | **Reward 分布**（sparse 0/1，看正样本密度） |
| `histograms/action_norms` | 残差动作范数分布 |

### 4.6 Eval 组（仅在 eval iteration 记录）

记录于 [residual_ppo.py:451-458](src/train/residual_ppo.py#L451-L458)，每 `eval_interval=5` 轮一次。

| 指标 | 含义 |
|------|------|
| `eval/success_rate` | 当前 checkpoint 的 eval rollout 成功率（确定性执行 `action=action_mean`，不加噪声） |
| `eval/best_eval_success_rate` | 历史最优 eval 成功率，用于保存 best model |

---

## 5. 四任务实际训练参数对比

### 5.1 默认参数 (base_residual_rl.yaml)

| 超参数 | 默认值 | 说明 |
|--------|--------|------|
| num_envs | 1024 | 并行环境数 |
| num_env_steps | 700 | 每轮数据收集步数 |
| batch_size | 716,800 | = 700 × 1024 |
| update_epochs | 50 | PPO 每轮更新轮数 |
| discount (γ) | 0.999 | 折扣因子 |
| gae_lambda (λ) | 0.95 | GAE λ |
| clip_coef (ε) | 0.2 | PPO clip 范围 |
| target_kl | 0.1 | KL early stop 阈值 |
| learning_rate_actor | 3e-4 | Actor 学习率 |
| learning_rate_critic | 5e-3 | Critic 学习率 |
| max_grad_norm | 1.0 | 梯度 clip |
| ent_coef | 0.0 | 熵系数 |
| vf_coef | 1.0 | Value loss 权重 |
| normalize_reward | true | Reward normalization |
| init_logstd | -1.0 | 初始 log std |
| learn_std | false | 是否学习 std |
| action_scale | 0.1 | 残差动作缩放 |
| residual_l1 | 0.0 | L1 正则系数 |
| residual_l2 | 0.0 | L2 正则系数 |

### 5.2 各任务实际训练参数（来自 checkpoint config + slurm 脚本）

| 参数 | one_leg | lamp | round_table | **desk (未训练)** |
|------|---------|------|-------------|-------------------|
| num_env_steps | **700** | **1000** | **1000** | **???** |
| batch_size | 716,800 | 1,024,000 | 1,024,000 | — |
| ent_coef | **0.001** | 0 | 0 | — |
| learn_std | **True** | False | False | — |
| init_logstd | **-0.9** | -1.0 | -1.0 | — |
| normalize_reward | **False** | False | False | — |
| task_timeout (理论 max) | 1,000 | 2,000 | 2,000 | **4,000** |
| num_env_steps / timeout | **70%** | **50%** | **50%** | — |

关键发现:

- **one_leg 使用了更积极的探索**: `learn_std=true`, `init_logstd=-0.9` (σ=0.41), `ent_coef=0.001`。说明 1 部件简单任务上适当增加探索有助于收敛
- **lamp/round_table 关闭了探索**: `learn_std=false`, `init_logstd=-1`, `ent_coef=0`。2 部件任务依赖 base policy 的先验，减少无意义的随机探索
- **lamp/round_table 的 num_env_steps=1000**（默认 700 被 override），与 task_timeout=2000 的比例为 50%
- **三个任务都关掉了 `normalize_reward`**（覆盖默认的 true），用 raw sparse 0/1 reward

---

## 6. Residual PPO w/ BC 额外指标

`residual_ppo_w_bc.py` 在 residual PPO 基础上同时做 BC 训练 base policy。额外记录:

| 指标 | 含义 |
|------|------|
| `base_bc/demo_loss` | BC loss on demo data |
| `base_bc/buffer_loss` | BC loss on replay buffer (successful RPPO rollouts) |
| `base_bc/total_loss` | demo_loss + buffer_loss |
| `base_bc/replay_buffer_size` | 成功 trajectory replay buffer 大小 |

---

## 7. Desk 未收敛原因总结

Desk 与已收敛任务的对比：

| | one_leg | lamp | round_table | **desk** |
|---|---|---|---|---|
| 部件对数 | 1 | 2 | 2 | **4** |
| phases | 5 | 7 | 8 | **16** |
| task_timeout | 1,000 | 2,000 | 2,000 | **4,000** |
| 实际 num_env_steps | 700 | 1,000 | 1,000 | **700 (默认)** ❌ |
| num_env_steps / timeout | 70% | 50% | 50% | **17.5%** ❌ |
| BC base policy | 有 | 有 | 有 | **无** |
| RPPO checkpoint | 有 | 有 | 有 | **无** |
| 探索配置 | learn_std=true | learn_std=false | learn_std=false | **默认=false** |

**核心原因（按优先级）**:

1. **num_env_steps 严重偏小**: 默认 700 步只占 desk 4000 步 timeout 的 17.5%，无法在单 episode 内完成 4 次组装。应至少调到 2000
2. **无 BC base policy**: `checkpoints/bc/` 下无 desk，需要先训练一个 desk BC 策略作为 base
3. **4 部件稀疏奖励 + 固定小噪声**: `learn_std=false`, `init_logstd=-1`，有效噪声仅 0.037
4. **desk 每条腿有 4 个装配位姿** (`assembled_rel_poses` 含 4 元素列表)，增加了多样性
