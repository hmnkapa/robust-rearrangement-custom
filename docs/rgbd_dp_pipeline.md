# RGBD 2D Diffusion Policy Pipeline (Replay -> Pickle -> Zarr -> Train -> Eval)

本文档给出仓库内可直接执行的端到端流程，主线是 **RGBD + 2D Diffusion Policy**。

## 1. 环境准备

```bash
# 仓库根目录
cd /data/hy/robust-rearrangement

# 建议使用项目环境
conda activate rr

# 数据路径（按你的机器修改）
export DATA_DIR_RAW=/data/hy/robust-rearrangement/raw
export DATA_DIR_PROCESSED=/data/hy/robust-rearrangement/data/processed

# 可选：wandb
export WANDB_ENTITY=<your_wandb_entity>
```

## 2. Replay 采集/重放并生成 Raw Pickle

下面示例将 replay 结果存到标准 raw 目录。你需要准备输入轨迹 `--pickle-path`。

```bash
python -m src.eval.replay_parallel \
  --pickle-path /abs/path/to/source_traj.pkl \
  --task one_leg \
  --gpu 0 \
  --num-envs 1 \
  --randomness low \
  --act-rot-repr quat \
  --headless
```

如果你还需要导出 DP3 点云 pickle（非本文主线，可选）：

```bash
python -m src.eval.replay_parallel \
  --pickle-path /abs/path/to/source_traj.pkl \
  --task one_leg \
  --gpu 0 \
  --num-envs 1 \
  --randomness low \
  --act-rot-repr quat \
  --headless \
  --save-pc-for-dp3 \
  --pc-points 4096 \
  --pc-downsample-mode random \
  --pc-out-dir /data/hy/robust-rearrangement/raw/raw/diffik/sim/one_leg/rollout/low/pc/success
```

## 3. 处理 Pickle 到 Zarr

### 3.1 标准路径模式（推荐）

```bash
python -m src.data_processing.process_pickles \
  -c diffik \
  -d sim \
  -f one_leg \
  -s rollout \
  -r low \
  -o success \
  --suffix rgbd \
  --output-suffix rgbd \
  --overwrite \
  --n-cpus 8 \
  --batch-size 20 \
  --resize-image
```

### 3.2 显式输入/输出路径模式

`--input-dir` 指向包含 `.pkl` 或 `.pkl.xz` 的目录；`--output-dir` 为目标 zarr 路径。

```bash
python -m src.data_processing.process_pickles \
  -c diffik \
  -d sim \
  -f one_leg \
  -s rollout \
  -r low \
  -o success \
  --input-dir /abs/path/to/raw_pickles \
  --output-dir /abs/path/to/one_leg_low_rgbd.zarr \
  --overwrite \
  --n-cpus 8 \
  --batch-size 20 \
  --resize-image
```

## 4. 训练 RGBD 2D Diffusion Policy

```bash
python -m src.train.bc \
  +experiment=rgbd/diff_unet \
  task=one_leg \
  randomness=low \
  demo_source=rollout \
  data.demo_outcome=success \
  data.suffix=rgbd \
  wandb.project=<your_project> \
  dryrun=false
```

可选快速自检（不真正训练）：

```bash
python -m src.train.bc --cfg job +experiment=rgbd/diff_unet task=one_leg randomness=low wandb.project=<your_project>
```

## 5. Eval（RGBD 模型）

### 5.1 用 WandB run-id

```bash
python -m src.eval.evaluate_model \
  --run-id <wandb_project>/<wandb_run_id> \
  --n-envs 32 \
  --n-rollouts 128 \
  -f one_leg \
  --randomness low \
  --action-type pos \
  --observation-space image \
  --wt-type best_success_rate \
  --if-exists append
```

### 5.2 用本地 checkpoint

```bash
python -m src.eval.evaluate_model \
  --wt-path /abs/path/to/actor_chkpt_best_success_rate.pt \
  --n-envs 32 \
  --n-rollouts 128 \
  -f one_leg \
  --randomness low \
  --action-type pos \
  --observation-space image \
  --if-exists append
```

## 6. 常见问题排查

- `KeyError: depth_image1`：旧数据缺少深度图；当前流程会自动补零 depth 并继续处理。
- Hydra 配置报错：`experiment` 组请用 `+experiment=rgbd/diff_unet`。
- 训练前检查 zarr：确认包含 `color_image1/2` 和 `depth_image1/2`。
- Isaac/CUDA 初始化告警：若只是 `--cfg job` 或 `--help`，通常不影响配置检查。

## 7. 与 Notion 流程对齐（待补充）

参考文档：
- https://www.notion.so/FB-DP3-2d76aab8287c801a81b6df3b7821599b?source=copy_link

当前版本基于仓库可验证脚本编写；如你补充 Notion 中的差异点（参数、目录结构、评估设置），可在本节追加“仓库命令 <-> Notion 步骤”对照表。
