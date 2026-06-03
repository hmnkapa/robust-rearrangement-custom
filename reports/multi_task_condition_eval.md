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
| 3 | rgbd+colored GP | lively-wind-5 | 0.00% (0/36) | 5.56% (2/36) | 22.22% (8/36) | **9.26% (10/108)** |
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
| 3 | rgbd+colored GP | 22.22% (8/36) | 100.00% (7/7) |
| 4 | rgbd+GP | 16.67% (6/36) | 100.00% (6/6) |
| 5 | rgbd+GP+skill | 25.00% (9/36) | 100.00% (5/5) |
| 6 | rgb | N/A | N/A |

> 关键发现: **base-hood 子步骤成功率达到 100%** — 一旦 bulb 插入成功，hood 放置几乎必然成功

## 3. 结论

1. **one_leg 全败** — 所有 6 个条件在 one_leg 上均为 0%，说明该任务对当前模型架构（DiT + 100traj）是最难的
2. **round_table 仅 colored GP 有效** — 实验 3 (rgbd+colored guidance point) 是唯一 round_table 有成功的，top-leg 步骤 19.44%
3. **lamp 最容易** — 所有实验在 lamp 上都有非零成功率，其中 rgb 和 rgbd+GP+skill 最好 (25.00%)
4. **Best Overall**: rgbd+colored guidance point (9.26%), 得益于 round_table 突破
5. **Best lamp**: rgb 和 rgbd+GP+skill 并列 (25.00%)
6. **guidance point 不加 skill** (exp4) 反而最差 (5.56%), 加上 skill (exp5) 提升到 8.33%
7. **纯 rgb** (exp6, 8.33%) 持平或优于部分 rgbd 变体，说明 depth 通道不一定有增益
