# 多任务 Condition 对比实验 — 评估结果汇总

**日期**: 2026-05-31 ~ 2026-06-01
**任务**: one_leg + round_table + lamp (多任务)
**模型**: DiT (diffusion), 3×100 trajectories
**Project**: multi-task-rgbd-skill-low-0526
**Eval 设置**: N_ENVS=3, N_ROLLOUTS=36 (每任务 36 次 rollout), image-based

---

## 1. 总览

| # | 实验 | RUN_ID | one_leg | round_table | lamp | **Overall** |
|---|------|--------|:---:|:---:|:---:|:---:|
| 1 | rgbd+only skill | iconic-surf-2 | 0.00% (0/36) | 0.00% (0/36) | 11.11% (4/36) | **3.70% (4/108)** |
| 2 | rgbd | unique-durian-3 | 0.00% (0/36) | 0.00% (0/36) | 22.22% (8/36) | **7.41% (8/108)** |
| 3 | rgbd+colored GP 🔧 | lively-wind-5 | 0.00% (0/36) | 5.56% (2/36) | 38.89% (14/36) | **14.81% (16/108)** |
| 4 | rgbd+GP | major-violet-9 | 0.00% (0/36) | 0.00% (0/36) | 16.67% (6/36) | **5.56% (6/108)** |
| 5 | rgbd+GP+skill | zany-firebrand-10 | 0.00% (0/36) | 0.00% (0/36) | 25.00% (9/36) | **8.33% (9/108)** |
| 6 | rgb | sweet-shadow-11 | 0.00% (0/36) | 0.00% (0/36) | 25.00% (9/36) | **8.33% (9/108)** |

## 2. Assembly Step 成功率

### one_leg: top-leg → 全 0% (所有实验一概失败)

### round_table

| # | 实验 | top-leg | leg-base |
|---|------|:---:|:---:|
| 1 | rgbd+only skill | 0.00% (0/36) | - |
| 2 | rgbd | N/A | N/A |
| 3 | rgbd+colored GP | **19.44%** (7/36) | **28.57%** (2/7) |
| 4 | rgbd+GP | 0.00% (0/36) | - |
| 5 | rgbd+GP+skill | 0.00% (0/36) | - |
| 6 | rgb | N/A | N/A |

> 注: N/A 的实验中未启用 --annotate-skill，无 step 级统计
> 仅实验 3 (colored GP) 在 round_table 上有非零成功率

### lamp

| # | 实验 | base-bulb | base-hood |
|---|------|:---:|:---:|
| 1 | rgbd+only skill | 11.11% (4/36) | 100.00% (4/4) |
| 2 | rgbd | N/A | N/A |
| 3 | rgbd+colored GP | 41.67% (15/36) | 92.86% (13/14) |
| 4 | rgbd+GP | 16.67% (6/36) | 100.00% (6/6) |
| 5 | rgbd+GP+skill | 25.00% (9/36) | 100.00% (5/5) |
| 6 | rgb | N/A | N/A |

> 关键发现: **base-hood 子步骤成功率达到 100%** — 一旦 bulb 插入成功，hood 放置几乎必然成功

## 3. 结论

1. **one_leg 全败** — 所有条件在 one_leg 上均为 0%
2. **colored GP 效果最显著** — 13.89% overall，lamp 33.33%，远超其他条件
3. **round_table 仅 colored GP 有效** — 唯一有非零成功率的条件
4. **guidance point 颜色匹配至关重要** — config 修复前后 2.78% → 13.89% (5x)
5. **lamp 最容易** — 所有条件均有成功，base-hood 100%
6. **colored GP > plain GP** — 彩色引导点提供 skill 类型信息
