# LMDB I/O 性能诊断报告

**日期**: 2026-05-11
**服务器**: `zju_4090_243`
**Conda 环境**: `rr`
**数据路径**: `/data/hy/robust-rearrangement-custom/data/processed/diffik/sim/round_table/rollout/low/success/`

---

## 1. 环境信息

### 1.1 硬件

| 组件 | 详情 |
|------|------|
| CPU | 256 核 (2× EPYC) |
| 内存 | 503 GB (buff/cache ~377 GB, free ~38 GB) |
| GPU | 8× RTX 4090 (24 GB) |
| 系统盘 (sdb) | 7TB NVMe RAID (MR9560-16i) |
| 数据盘 (sda) | 15TB 机械硬盘 (Seagate Exos ST16000NM000J, `rotational=1`) |
| 文件系统 | ext4, relatime |

### 1.2 LMDB 数据集

6 个 `rgbd-skill` shard，总计 **~638 GB，635,851 帧**：

| Shard | Episodes | Frames | 大小 | 每帧 |
|-------|----------|--------|------|------|
| rgbd-skill.lmdb | 200 | 115,126 | 115.5 GB | 1,050 KB |
| rgbd-skill-1.lmdb | 199 | 106,028 | 106.4 GB | 1,050 KB |
| rgbd-skill-2.lmdb | 199 | 106,083 | 106.5 GB | 1,050 KB |
| rgbd-skill-3.lmdb | 193 | 102,727 | 103.1 GB | 1,050 KB |
| rgbd-skill-4.lmdb | 191 | 102,178 | 102.5 GB | 1,050 KB |
| rgbd-skill-5.lmdb | 194 | 103,709 | 104.1 GB | 1,050 KB |

每帧含 4 个 key：

| Key | Shape | Dtype | 大小 |
|-----|-------|-------|------|
| color_image1 | (240, 320, 3) | uint8 | 230 KB |
| color_image2 | (240, 320, 3) | uint8 | 230 KB |
| depth_image1 | (240, 320) | float32 | 307 KB |
| depth_image2 | (240, 320) | float32 | 307 KB |

> 注：深度图是 (240, 320) 二维 float32，非 (240, 320, 1)

### 1.3 训练配置 (来自 gpu-snatcher)

```
+experiment=rgbd/dit
data.storage_format=lmdb
data.load_into_memory=false
data.dataloader_workers=4          # 每个 DDP rank 的 worker 数
training.batch_size=512            # 全局 batch size (每 rank = 256)
data.ddp_shard_enabled=true
--nproc_per_node=2                 # 2 卡训练
```

---

## 2. 基准测试方法

脚本 `scripts/benchmark_lmdb_io.py` 直接复用项目 `src.dataset.lmdb.LMDBImageStore`，访问模式与训练一致：

- 每个 worker 子进程打开独立的 LMDB env（PID 检测）
- 随机选择帧索引，调用 `store.get_frames([idx], keys)`
- 模拟 `RGBDDataset.__getitem__` 的惰性加载路径

测试参数: duration=30s, warmup=5s, mode=rgbd (4 keys)

---

## 3. 实验结果

### 3.1 单进程基线 — 确认单 LMDB 最大 IO 吞吐

| Workers | Readers | Samples/s | MB/s | RSS Total (MB) |
|---------|---------|-----------|------|----------------|
| 1 | 1 | 2.7 | 2.7 | 385 |
| 2 | 2 | 5.1 | 5.3 | 768 |
| 4 | 4 | 8.0 | 8.2 | 1,509 |
| 8 | 8 | 11.9 | 12.2 | 2,973 |
| 16 | 16 | 15.4 | 15.8 | 5,851 |

**结论**: 吞吐随 worker 数次线性增长，16 workers 时趋近 HDD 上限。每个额外 worker 增加 ~360 MB RSS。

### 3.2 单进程多 LMDB vs 单 LMDB（4 workers）

| LMDB 数 | Samples/s | MB/s | vs 单 LMDB |
|---------|-----------|------|-----------|
| 1 | 8.0 | 8.2 | 基线 |
| 6 | 6.7 | 6.9 | -16% |

**结论**: 6 个 LMDB 比 1 个慢约 16%，差异远小于用户报告的 "10x"。但这是在热缓存下的结果，冷缓存差异可能更大。

### 3.3 多进程同读单 LMDB — 定位问题 1

| Procs | W/Proc | 总 Readers | Samples/s | MB/s | RSS (MB) |
|-------|--------|-----------|-----------|------|----------|
| 1 | 8 | 8 | 11.9 | 12.2 | 2,973 |
| 2 | 4 | 8 | 12.5 | 12.8 | 2,983 |
| 2 | 8 | 16 | 17.3 | 17.7 | 5,828 |
| 4 | 4 | 16 | 17.5 | 17.9 | 5,936 |
| 4 | 8 | 32 | 21.0 | 21.6 | 10,702 |

**关键对比**:

| 对比 | 总 Readers | Samples/s | 差异 |
|------|-----------|-----------|------|
| 1p/8w vs 2p/4w | 8 | 11.9 vs 12.5 | +5% |
| 1p/16w vs 2p/8w vs 4p/4w | 16 | 15.4 vs 17.3 vs 17.5 | ~+12% |

**结论**: 相同总 Reader 数下，进程数对吞吐影响微小（<12%）。瓶颈在总 Reader 数而非进程结构。

### 3.4 多进程多 LMDB — 模拟真实训练并行

| Procs | Shards | Mode | 总 Readers | Samples/s | MB/s | 场景 |
|-------|--------|------|-----------|-----------|------|------|
| 2 | 2 | split | 8 | 14.8 | 15.2 | 2 卡各独占 1 LMDB |
| 4 | 6 | split | 16 | 18.4 | 18.9 | 4 卡各独占 ~1.5 LMDB |
| 4 | 6 | shared | 16 | 17.0 | 17.4 | 4 卡共享 6 LMDB（真实场景） |

**结论**: split 模式（各进程独占不同 LMDB）比 shared 模式快约 8-10%。

### 3.5 控制变量 — 固定 16 Readers，变进程数

| Procs | Shards | Mode | Samples/s | MB/s |
|-------|--------|------|-----------|------|
| 1 | 1 | shared | 15.4 | 15.8 |
| 1 | 6 | shared | 19.7 | 20.2 |
| 4 | 6 | shared | 17.0 | 17.4 |

**结论**: 多 shard 在高 Reader 数下反而略有优势（更多并行文件访问），但与进程数无关。

---

## 4. iostat 磁盘级分析

### 4.1 基准测试期间 sda（数据盘 HDD）状态

| 指标 | 值 |
|------|-----|
| 读 IOPS | **5,000-7,000 /s** |
| 读吞吐 | **20-27 MB/s** |
| **磁盘利用率** | **87-93%** |
| 每次读大小 | **恰好 4.0 KB**（LMDB 页面大小） |
| 读延迟 (await) | 1-4 ms |
| 队列深度 (aqu-sz) | 7-21 |

### 4.2 根因分析

LMDB 使用 B-tree 存储，默认页面大小 4KB。读取一帧（1,050 KB）需要：

```
B-tree 内部节点查找:  3-4 次 × 4KB = 12-16 KB   (通常命中 page cache)
叶子数据页读取:       ~262 次 × 4KB = 1,048 KB  (大部分未缓存)
─────────────────────────────────────────
每帧磁盘 I/O:         ~1,048 KB
每帧总 I/O:           ~1,064 KB
```

机械硬盘随机 4KB 读取的物理上限约 **6,000-7,000 IOPS**，对应 **24-28 MB/s**。这与 iostat 观测值完全吻合。

应用层吞吐（~17-21 MB/s）略低于磁盘吞吐（~25 MB/s），差额来自：
- B-tree 内部节点未命中的额外读取
- LMDB 元数据和锁开销
- Python/numpy 解包开销

### 4.3 Page Cache 状态

- 总 page cache: ~377 GB
- 数据集总量: ~638 GB（6 shard）
- 缓存覆盖率: ~59%

单 shard（116 GB）可以完全缓存。6 shard（638 GB）只能部分缓存，B-tree 内部页竞争加剧。

---

## 5. 综合诊断

### 5.1 问题 1：多进程同读单 LMDB 为什么会慢？

**根因不是 LMDB 锁或 reader contention，而是 HDD 随机读物理上限。**

- 2 卡训练 (2p×4w = 8 readers): ~12.5 MB/s, 磁盘利用率 ~70-80% — 勉强可接受
- 两个 2 卡训练 (4p×4w = 16 readers): ~17.5 MB/s, 磁盘利用率 **~90%** — IO 成为瓶颈
- 训练吞吐需求: 512 frames/batch ÷ 17.5 frames/s ≈ **29 秒/batch**，远超 GPU 计算时间

增加 dataloader workers 无效是因为 HDD 已经饱和，更多 reader 只会增加队列深度而不能提高吞吐。

### 5.2 问题 2：多 LMDB 为什么会更慢？

**实测差异约 16%（热缓存），远小于报告的 "10x"。可能的原因：**

1. **冷缓存效应**：6 个 LMDB = 6 套 B-tree，内部节点总数 6×。冷启动时需要从 HDD 加载更多元数据
2. **Page cache 竞争**：638 GB 数据 vs 377 GB page cache。多 LMDB 场景下 B-tree 内部页更容易被逐出
3. **文件间寻道**：HDD 在 6 个文件间切换时磁头寻道开销比单文件更高
4. **GPU 内存竞争**：真实训练时 GPU 显存占用 (~20GB/卡) 会进一步压缩 page cache 可用空间

报告的 "10x" 可能发生在冷缓存 + GPU 训练同时运行的场景。

---

## 6. 建议方案

| 优先级 | 方案 | 预期效果 | 实施难度 |
|--------|------|---------|---------|
| **P0** | 将 LMDB 数据迁移到 sdb (NVMe RAID) | 随机读 IOPS 从 6K→500K+，吞吐 10-50x | 低 |
| **P1** | 合并 LMDB shard，减少 B-tree 数量 | 减少 page cache 压力，冷启动更快 | 中 |
| **P2** | 使用 `vmtouch` 预热 page cache | 训练启动前预加载关键 B-tree 页面 | 低 |
| **P3** | 考虑替代存储格式（numpy memmap / zarr） | 绕过 LMDB B-tree 开销，顺序读更友好 | 高 |

### 6.1 立即可做

```bash
# 方案 P0: 在 sdb 上创建数据目录并复制
mkdir -p /mnt/nvme/hy/robust-rearrangement-custom/data/processed/diffik/sim/round_table/rollout/low/success/
cp -r /data/hy/robust-rearrangement-custom/data/processed/diffik/sim/round_table/rollout/low/success/rgbd-skill*.lmdb \
     /mnt/nvme/hy/.../

# 设置环境变量指向新路径
export DATA_DIR_PROCESSED=/mnt/nvme/hy/robust-rearrangement-custom/data/
```

注意：sdb 空间只有 7TB，当前数据集 ~638 GB，需确认剩余空间充足。

### 6.2 短期缓解（不改存储）

```bash
# 训练前预热 page cache
vmtouch -t /data/hy/robust-rearrangement-custom/data/processed/diffik/sim/round_table/rollout/low/success/rgbd-skill*.lmdb/data.mdb

# 减少 dataloader workers（避免过度竞争 HDD）
data.dataloader_workers=2  # 而不是 4 或 8
```

---

## 7. 附录

### 7.1 基准脚本使用

```bash
python scripts/benchmark_lmdb_io.py \
  --lmdb-dir <LMDB目录> \
  --shard-glob "rgbd-skill*.lmdb" \
  --shard-subset 0 \           # 只使用第 0 个 shard
  --num-processes 2 \          # 模拟 DDP rank 数
  --workers-per-process 4 \    # 每个 rank 的 workers
  --mode rgbd \                # image / rgbd
  --shard-mode shared \        # shared / split
  --duration 30 \
  --warmup 5 \
  --system-monitor \           # 后台 iostat/vmstat
  --output results.json
```

### 7.2 原始数据

所有 JSON 结果文件位于服务器:
```
/mnt/nas/share/home/hy/robust-rearrangement-custom/benchmark_results/
```
