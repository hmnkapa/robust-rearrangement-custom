# 批量 Eval 实验操作手册

## 1. 实验名称与超参对照表

**每个实验的差异参数**：

| # | 实验 | RUN_ID | ANNOTATE_SKILL | GP_ON_IMAGE | SKILL_ON_IMAGE | GP_COLORED |
|---|------|--------|:---:|:---:|:---:|:---:|
| 1 | rgbd+only skill | | true | false | true | false |
| 2 | rgbd |  | true | false | true | false |
| 3 | rgbd+colored GP |  | true | true | true | true |
| 4 | rgbd+GP |  | true | true | true | false |
| 5 | rgbd+GP+skill |  | true | true | true | false |
| 6 | rgb |  | true | false | true | false |

RUN_ID 的对应要求用户输入。

## 2. auto_eval.sh 参数说明

脚本位置：`~/projects/gpu-snatcher/auto_eval.sh`

```bash
# 核心参数
RUN_ID="xxx"                  # wandb run_name，决定下载哪个 checkpoint
EVAL_ANNOTATE_SKILL=true/false       # 启用 skill 标注 (需 image obs)
EVAL_GUIDANCE_POINT_ON_IMAGE=true/false  # guidance point 可视化
EVAL_SKILL_ON_IMAGE=true/false       # skill 文字覆盖在 rollout 视频上
EVAL_GUIDANCE_POINT_COLORED=true/false   # 按 skill 着色 guidance point

# 跳过下载直接用本地 checkpoint
OVERWRITE_WT_PATH="/path/to/local.pt"   # 设置后自动跳过 download 步骤
STEPS=(eval)                            # 注释掉 download，只跑 eval

# 正常下载+eval
OVERWRITE_WT_PATH=""                    # 清空，恢复自动拼接路径
STEPS=(download eval)                   # 恢复下载步骤
```

EVAL flag 组合规则：
- `ANNOTATE_SKILL=false` → 无 step 级成功率统计，rollout 视频无标注
- `ANNOTATE_SKILL=true` → 有 `[skill-debug]` stdout 日志 + assembly step 统计
- `SKILL_ON_IMAGE=true` → 需要 `ANNOTATE_SKILL=true`，在视频画面上叠加 skill 文字
- `GUIDANCE_POINT_ON_IMAGE=true` → 在视频上画 guidance point 圆点
- `GUIDANCE_POINT_COLORED=true` → guidance point 按 skill 颜色编码（pick=黄, place=红）

**注意**：
- `--observation-space state` 时不支持 skill annotation（env 无 camera 属性）
- `--observation-space image` 才能用 EVAL flag
- image eval 每个实验产生 ~60G rollout 数据，需定期清理

## 3. 批量 Eval 执行流程

### 3.1 准备

```bash
# 确认环境
conda activate rr
cd ~/projects/gpu-snatcher
# 确认 SSH 可达
ssh -o ConnectTimeout=5 zju_4090_228 echo ok
# 检查磁盘（每个实验需 ~60G）
df -h /
```

### 3.2 串行执行（手动）

对每个实验，依次：

```bash
# 1. 修改 auto_eval.sh 中的 RUN_ID 和 EVAL flag
vim auto_eval.sh

# 2. 运行
bash auto_eval.sh 2>&1 | tee /tmp/exp<N>_<name>.log

# 3. 提取结果
grep -E "Success rate \((one_leg|round_table|lamp|all tasks)\)" /tmp/exp<N>_<name>.log
grep -A10 "Assembly step success rates" /tmp/exp<N>_<name>.log
```

### 3.3 串行执行（Claude Code 自动化）

告诉 Claude Code：

```
我要批量跑 eval 实验，项目路径 ~/projects/gpu-snatcher，
实验参数见 ~/projects/robust-rearrangement-custom/reports/claude/batch_eval_guide.md。
请串行执行，每个完成后自动推进下一个，每 10 分钟自检汇报进度。
```

Claude Code 会自动：
1. 读取参数表
2. 依次修改 `auto_eval.sh`
3. 启动后台任务（`run_in_background: true`）
4. 设 Monitor 捕捉成功率 + 错误
5. Cron 每 10 分钟检查进度 + 磁盘 + 成功率
6. 全部完成后汇总到 `eval_summary.md`

### 3.4 磁盘清理

每个 image-based 实验产生 ~50-60G rollout pickle 数据在：
```
~/projects/robust-rearrangement-custom/data/raw/diffik/sim/{task}/rollout/low/{suffix}/
```

清理策略（每实验每 task 保留 10 个 pkl）：
```bash
BASE="$HOME/projects/robust-rearrangement-custom/data/raw/diffik/sim"
for suffix in rgbd-only-skill rgbd rgbd-skill-colored rgbd-skill; do
    for task in one_leg round_table lamp; do
        dir="$BASE/$task/rollout/low/$suffix"
        [ -d "$dir" ] || continue
        mapfile -t files < <(find "$dir" -name "*.pkl" -type f | sort)
        if [ ${#files[@]} -gt 10 ]; then
            for ((i=10; i<${#files[@]}; i++)); do rm -f "${files[$i]}"; done
        fi
    done
done
```

## 4. 监控自检流程

### 4.1 Cron 检查脚本模式

```bash
echo "=== $(date) disk: $(df -h / | awk 'NR==2{print $5,$4}') procs: $(pgrep -c evaluate_model) ==="

if grep -q "Evaluation finished successfully" /tmp/exp<N>.log 2>/dev/null; then
    echo "EXP: COMPLETED"
    grep -E "Success rate \((one_leg|round_table|lamp|all tasks)\)" /tmp/exp<N>.log | tail -4
else
    echo "EXP: RUNNING"
    echo "task: $(tail -5 /tmp/exp<N>.log | grep -oP '\([a-z_]+\):' | tail -1)"
    echo "latest: $(grep -oP 'success: \d+/\d+ \(\d+\.?\d*%\)' /tmp/exp<N>.log | tail -1)"
fi
```

### 4.2 关键监控指标

| 指标 | 命令 |
|------|------|
| 磁盘剩余 | `df -h /` |
| eval 进程数 | `pgrep -c evaluate_model` |
| 当前任务 | `tail -5 log \| grep -oP '\([a-z_]+\):'` |
| 当前轮次成功率 | `grep -oP 'success: \d+/\d+ \(\d+\.?\d*%\)' log \| tail -1` |
| 是否完成 | `grep -q "Evaluation finished successfully" log` |
| 是否有错误 | `grep -c "ERROR\|Traceback" log` |

### 4.3 典型异常处理

| 异常 | 现象 | 处理 |
|------|------|------|
| SCP 超时 | 日志卡在 "Downloading checkpoint via scp" | kill 进程，检查 SSH，重跑或设 OVERWRITE_WT_PATH |
| 磁盘满 | 返回 ENOSPC 或磁盘 >95% | 执行清理脚本（见 3.4） |
| `front_cam_pos` 错误 | `--observation-space state` + `--annotate-skill` | 改用 `image` observation space |
| 进程僵死 | pgrep 返回 0 但无报错 | 检查 Bash 超时，改用 `run_in_background: true` |

## 5. 需要收集的结果

每个实验完成后需提取以下数据：

### 5.1 总体/单任务成功率

从 stdout 日志提取：
```bash
grep -E "Success rate \((one_leg|round_table|lamp|all tasks)\)" /tmp/exp<N>.log
```

格式：`Success rate (lamp): 25.00% (9/36)`

### 5.2 Assembly Step 成功率（需启用 annotate-skill）

从 stdout 日志提取：
```bash
grep -A10 "Assembly step success rates" /tmp/exp<N>.log
```

每个 task 输出类似：
```
Assembly step success rates (lamp):
  base-bulb: 25.00% (9/36)
  base-hood: 100.00% (5/5)
```

**各任务 sub-step 列表**：
| 任务 | sub-steps |
|------|-----------|
| one_leg | top-leg |
| round_table | top-leg, leg-base |
| lamp | base-bulb, base-hood |

### 5.3 stdout 日志保留

每个实验保留完整日志以备查：
```bash
tee /tmp/exp<N>_<name>.log
```

## 6. 结果汇总模板

参考 `~/projects/robust-rearrangement-custom/reports/multi_task_condition_eval.md`，需包含：

### 6.1 总览表

| # | 实验 | RUN_ID | one_leg | round_table | lamp | **Overall** |
|---|------|--------|:---:|:---:|:---:|:---:|
| 1 | ... | ... | X% (n/36) | X% (n/36) | X% (n/36) | **X% (n/108)** |

### 6.2 Assembly Step 表（仅 enable annotate-skill 的实验）

**one_leg**：top-leg 成功率
**round_table**：top-leg → leg-base 级联成功率
**lamp**：base-bulb → base-hood 级联成功率

### 6.3 结论

- one_leg / round_table / lamp 三个任务的难度排序
- 各 condition 对比（best/worst overall）
- 关键子步骤发现（如 base-hood 100%）
- 关于 guidance point / skill / colored / depth 的消融结论

完整示例见：`~/projects/robust-rearrangement-custom/reports/multi_task_condition_eval.md`

```markdown
| # | 实验 | RUN_ID | one_leg | round_table | lamp | Overall |
|---|------|--------|:---:|:---:|:---:|:---:|
| 1 | rgbd+only skill | iconic-surf-2 | 0.00% (0/36) | 0.00% (0/36) | 11.11% (4/36) | 3.70% (4/108) |
| 2 | rgbd | unique-durian-3 | 0.00% (0/36) | 0.00% (0/36) | 22.22% (8/36) | 7.41% (8/108) |
| ... | ... | ... | ... | ... | ... | ... |
```

结果写入：`~/projects/robust-rearrangement-custom/reports/`，Notion 同步更新。
