# src teleop 使用说明

`src/data_collection/teleop.py` 用于通过 SpaceMouse 在 `src` 数据路径下采集 FurnitureRLSimEnv 轨迹。启动一次脚本可以连续采集多条成功轨迹，直到成功数量达到 `--num-demos`。

## 运行命令

从仓库根目录运行：

```bash
export DATA_DIR_RAW=/data/hy/robust-rearrangement/raw

python -m src.data_collection.teleop \
  --furniture desk \
  --randomness low \
  --num-demos 1 \
  --ctrl-mode diffik \
  --sm-pos-speed 0.54 \
  --sm-rot-speed 2.8 \
  --teleop-setting 1 \
  --show-teleop-cameras
```

使用扰动采集时：

```bash
python -m src.data_collection.teleop \
  --furniture one_leg \
  --randomness low \
  --num-demos 1 \
  --ctrl-mode diffik \
  --sample-perturbations
```

## 操作流程

1. 确认 SpaceMouse 驱动和服务已启动。

```bash
sudo systemctl start spacenavd
```

2. 启动采集命令。

3. 脚本 reset 后会自动进入待采集状态，不需要再按 `s` 开始。此时可以直接使用 SpaceMouse 或键盘开始操作；只有在机器人完成起始稳定、并且确实发生 teleop 动作时，transition 才会被记录。

4. 使用 SpaceMouse 控制末端位姿。默认不打开相机预览窗口；如果需要 OpenCV 预览，可添加 `--show-teleop-cameras`，显示 front camera 和 wrist camera。预览窗口通过单独的 helper 进程同步刷新，每个控制 step 都会等待该帧显示完成后再继续；窗口会按当前 preview 图缩小到 `0.3x` 并放到左上角，不改变保存到 pkl 的图像。

5. 使用 SpaceMouse 按钮或键盘 `z` 切换夹爪开合。

6. 当前轨迹成功后，按 `t` 保存为成功轨迹。

7. 当前轨迹失败后，按 `n` 标记失败。默认不保存失败轨迹；加 `--save-failure` 才保存。

8. 脚本保存并 reset 后，会自动进入下一条轨迹的待采集状态。

9. 成功轨迹数达到 `--num-demos` 后，脚本结束。

## 常用按键

`t`：标记当前轨迹成功并保存。

`y`：标记成功并额外记录最终装配 pose json。

`n`：标记当前轨迹失败。只有加 `--save-failure` 时才保存失败轨迹。

`` ` ``：标记一个 skill 完成。

数字键 `0` 到 `9`：手动 reward 标注。

`z`：切换夹爪开合；SpaceMouse 按钮也可以切换夹爪。

`p` / `c`：暂停 / 继续记录 transition。

`b`：撤销最近 10 个 transition，并 reset 到撤销后的最后一个 observation。

`w/s/a/d/q/e`：键盘平移控制，可作为 SpaceMouse 外的备用输入。

`i/k/j/l/u/o`：键盘旋转控制，可作为 SpaceMouse 外的备用输入。

`[` 和 `]`：调整键盘控制步长。

## 常用参数

`--furniture`：任务名，例如 `one_leg`、`lamp`、`round_table`、`mug_rack`、`factory_peg_hole`。

`--randomness`：初始化随机程度，可选 `low`、`med`、`high`。

`--num-demos`：目标成功轨迹数量。

`--ctrl-mode`：底层控制器。当前 src collector 只支持 `diffik`。

`--gpu-id`：仿真使用的 GPU id，同时用于 compute 和 graphics。

`--save-failure`：保存失败轨迹。

`--draw-marker`：显示 AprilTag marker；默认关闭 marker。

`--no-ee-laser`：关闭仿真中末端执行器的辅助 laser。

`--sample-perturbations`：在有 teleop 动作的 step 后，对家具 parts 采样随机扰动，并把数据保存到 `<randomness>_perturb` 目录。

`--resume-dir`：从目录中递归读取已有 `.pkl` / `.pkl.xz`，随机抽取最多 `--num-demos` 条；每次 reset 后加载其中一条轨迹的最终 observation 作为仿真初始状态，并把原轨迹内容 hydrate 到当前采集 buffer，用于从已有阶段继续采集后续阶段。

`--sm-pos-speed`：SpaceMouse 平移速度上限，单位是 m/s。默认 `0.54`，等价于旧 src 逻辑的 `0.3 * 1.8`。

`--sm-rot-speed`：SpaceMouse 旋转速度上限，单位是 rad/s。默认 `2.8`，等价于旧 src 逻辑的 `0.7 * 4`。

`--teleop-setting`：SpaceMouse 采集预设，可选 `1` 或 `2`。`1` 对齐旧 src 行为，平移和旋转都在 world/base 坐标系。`2` 对齐 furniture-bench 行为，平移和旋转都在 end effector 坐标系，ee 旋转符号为 `[x=+1, y=-1, z=-1]`，并按当前 furniture-bench 实现翻转 ee 平移的 `dpos[1]` 和 `dpos[2]`。

`--show-teleop-cameras`：显示 OpenCV 预览窗口。默认关闭。开启后会启动一个同步刷新的 preview helper 进程。

## Replay 单条轨迹

回放单条已采集轨迹：

```bash
python -m src.eval.replay \
  --pickle-path /data/hy/robust-rearrangement/raw/raw/diffik/sim/one_leg/teleop/low/success/2026-05-02T21-03-16.pkl \
  --task one_leg \
  --action-src action \
  --visualize
```

如果只想无窗口检查 replay 是否能跑通，可以改成：

```bash
python -m src.eval.replay \
  --pickle-path /data/hy/robust-rearrangement/raw/raw/diffik/sim/one_leg/teleop/low/success/2026-05-02T21-03-16.pkl \
  --task one_leg \
  --action-src action \
  --headless
```

## 保存路径

脚本使用 `DATA_DIR_RAW` 自动生成保存路径：

```text
$DATA_DIR_RAW/raw/<ctrl-mode>/sim/<furniture>/teleop/<randomness>/success/<timestamp>.pkl
```

例如：

```text
$DATA_DIR_RAW/raw/diffik/sim/one_leg/teleop/low/success/2026-05-02T12-30-00.pkl
```

如果打开 `--sample-perturbations`，`<randomness>` 会变成 `low_perturb`、`med_perturb` 或 `high_perturb`。

## 单条 pkl 内容

src 版默认就是 pickle-only：只保存 `.pkl`，不额外保存 mp4/png。每个 `.pkl` 是一条 trajectory，主要字段包括：

```python
{
    "observations": [...],
    "actions": [...],
    "rewards": [...],
    "skills": [...],
    "success": True or False,
    "furniture": "one_leg",
    "metadata": {...},
    "error": False,
    "error_description": "",
}
```

每个 observation 通常包含：

```python
{
    "color_image1": ...,  # Wrist camera
    "color_image2": ...,  # Front camera
    "robot_state": ...,
    "parts_poses": ...,
}
```
