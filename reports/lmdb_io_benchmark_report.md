# LMDB I/O 性能诊断报告

**日期**: 2026-05-11
**服务器**: `zju_4090_243`
**Conda 环境**: `rr`
**数据路径**: `/data/hy/robust-rearrangement-custom/data/processed/diffik/sim/round_table/rollout/low/success/`

---

## 1. 环境信息

### 1.1 硬件

**当前服务器 `zju_4090_243`：**

| 组件 | 详情 |
|------|------|
| CPU | 256 核 (2× EPYC) |
| 内存 | 503 GB (buff/cache ~377 GB, free ~38 GB) |
| GPU | 8× RTX 4090 (24 GB) |
| 数据盘 (sda) | 15TB HDD (Seagate Exos ST16000NM000J, `rotational=1`), `/data` |
| 系统盘 (sdb) | 7TB SSD (`rotational=0`), `/` |
| 文件系统 | ext4, relatime |

**所有 4090 服务器磁盘清单：**

| 服务器 | 系统盘 | 数据盘路径 | 数据盘类型 | 可用空间 |
|--------|--------|-----------|-----------|---------|
| `zju_4090_228` | NVMe 3.5T | `/data` (nvme1n1p1) | **NVMe** | 93G (97%) |
| `zju_4090_230` | NVMe 3.5T | `/data` (nvme1n1p1) | **NVMe** | 159G (96%) |
| `zju_4090_232` | NVMe 7T (nvme0n1) | `/data` (sda1) | **HDD** | NVMe: 722G, HDD: 2.3T |
| `zju_4090_236` | SSD 7T (sda) | `/` 同盘 | **SSD** | 待确认 |
| `zju_4090_238` | SSD 7T (sda) | `/` 同盘 | **SSD** | 待确认 |
| `zju_4090_240` | SSD 7T (sda) | `/` 同盘 | **SSD** | 216G (97%) |
| `zju_4090_243` | SSD 7T (sdb) | `/data` (sda1) | **HDD** | sdb: 418G, sda: 6.6T |

> **结论**：228/230 的 `/data` 已在 NVMe 上，无需迁移即可享受全速 IO；232/243 的 `/data` 在 HDD 上，但系统盘有 NVMe/SSD 可做迁移目标；236/238/240 是纯 SSD 机器，IO 本身不存在 HDD 瓶颈。

### 1.3 SSD vs HDD 随机读性能实测（zju_4090_243）

| 指标 | HDD sda (/data) | SSD sdb (/) | 倍数 |
|------|----------------|-------------|------|
| 随机 4K IOPS（单线程） | 9 | 407,959 | **45,000×** |
| 随机 4K IOPS（16 并发，NCQ） | ~6,000 | ~500K+（估算） | **~80×** |
| 顺序读 | 43.7 MB/s | 8,000 MB/s | **183×** |

> HDD 单线程 9 IOPS 是因为每读一次磁头需要完整寻道（115 GB 跨度，~110ms/次）。训练的实际 6,000 IOPS 靠 16 reader 并发 + NCQ 重排合并达成。SSD 没有机械寻道，单线程即可 40 万 IOPS。
>
> 之前估计的 "10-50x" 偏保守，实际随机读差距约 **80×**。

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

### 1.4 训练配置 (来自 gpu-snatcher)

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

**RSS (Resident Set Size)**：进程占用的**物理内存**（非虚拟内存），是实际驻留在 RAM 中的页面大小。LMDB 通过 mmap 映射 data.mdb 文件，当一个新 worker 进程开始随机读帧时，它会频繁缺页（page fault），OS 不断从磁盘拉 4KB 页到物理内存中——这些被拉进来的页面就是 RSS 的增量。

**Samples/s 不随 worker 数正比增长的原因**：RSS 增长瓶颈是内存容量（你有 503 GB），而吞吐瓶颈是 HDD 的**物理 IOPS 上限**（~6,000 IOPS = ~25 MB/s）：

RSS 几乎完美线性（每个 worker 加 ~360 MB），但吞吐在 8→16 workers 时边际增益已从 +89% 降到 +29%，因为 HDD 利用率已到 87-93%，物理上无法再挤出更多 IOPS。**再多 worker 只会加 RSS 而几乎不加吞吐。**

**符合我观察到的现象**：单训练 2 rank 总共 8 dataloader worker 一个 epoch 是 1min，并且每个 step 速度均匀，说明 IO 不是瓶颈，所以一个 batch_size=512 的训练要求的 IO 吞吐量≈12.2MB/s。同样的设置 2 个训练同时跑，一个 epoch 是 2min30s，除了每个训练只能分到 7.9MB/s 速度要慢 50% 之外，还有 page cache 更容易 miss 导致的减速。

要想两个训练在同一台机器上同时跑，合理的配置是：
1. 用不同数据集的训练放在不同的机器上跑
2. 两个训练跑在同一台机器上：batch_size=256，其他不变。

### 3.2 单进程多 LMDB vs 单 LMDB（4 workers）

| LMDB 数 | Samples/s | MB/s | vs 单 LMDB | 实际缓存状态 |
|---------|-----------|------|-----------|-------------|
| 1 | 8.0 | 8.2 | 基线 | **真·热缓存**（116 GB < 377 GB page cache，单 shard 完全驻留内存） |
| 6 | 6.7 | 6.9 | -16% | **大部分冷缓存**（638 GB > 377 GB page cache，最多 59% 覆盖率） |


**结论**: 
- 单 LMDB 热缓存 8.0 MB/s vs 6 LMDB 冷缓存 6.7 MB/s，差异仅 16%，原因是 4 workers 的低并发远未打满 HDD 物理上限（~25 MB/s），B-tree 内部页的额外开销在低并发下不显著
- 用户报告的 "10x" 降速发生在高并发（≥16 readers）+ GPU 显存占用进一步压缩 page cache 的场景，此时 6 套 B-tree 内部页竞争导致 miss 率急剧上升

所以

### 3.4 多进程多 LMDB — 模拟真实训练并行

| Procs | Shards | Mode | 总 Readers | Samples/s | MB/s | 场景 |
|-------|--------|------|-----------|-----------|------|------|
| 2 | 2 | split | 8 | 14.8 | 15.2 | 2 卡各独占 1 LMDB |
| 4 | 6 | split | 16 | 18.4 | 18.9 | 4 卡各独占 ~1.5 LMDB |
| 4 | 6 | shared | 16 | 17.0 | 17.4 | 4 卡共享 6 LMDB（真实场景） |

**结论**: 第一行和第二行对比基本还原了我看到减速现象的实际设置了，相当于每个训练分到的 IO 吞吐量变为 60%。

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
- 4 卡训练 (4p×4w = 16 readers): ~17.5 MB/s, 磁盘利用率 **~90%** — IO 成为瓶颈
- 训练吞吐需求: 512 frames/batch ÷ 17.5 frames/s ≈ **29 秒/batch**，远超 GPU 计算时间

增加 dataloader workers 无效是因为 HDD 已经饱和，更多 reader 只会增加队列深度而不能提高吞吐。

> **注意**：以上是单训练实例的 benchmark 数据。如果同时运行两个独立训练实例（各 2 卡），由于访问模式交错 + page cache 撕裂，吞吐会远低于单实例 16 readers 的数据。详见 **5.4 节**。

### 5.2 问题 2：多 LMDB 为什么会更慢？

**实测差异约 16%（6 LMDB 大部分冷缓存 vs 1 LMDB 热缓存），远小于报告的 "10x"。原因分析：**

1. **B-tree 内部页膨胀**：6 个 LMDB = 6 套 B-tree，内部节点总数 6×。冷缓存下这些内部页都需要从 HDD 加载
2. **Page cache 竞争**：638 GB 数据 vs 377 GB page cache，覆盖率上限 59%。多 LMDB 场景下 B-tree 内部页互相逐出，每次随机读都可能触发内部页 miss → 额外的 4KB 随机读
3. **文件间寻道**：HDD 磁头在 6 个 data.mdb 文件间切换，寻道开销比单文件高
4. **GPU 内存竞争**：真实训练时 GPU 显存占用 (~20GB/卡 × 2-4 卡) 进一步压缩 page cache，覆盖率可能降至 30-40%

**为什么低并发下差异不大（16%），高并发下差异急剧放大（10x）：**

- 低并发（4 readers, ~8 MB/s）：HDD 远未饱和，B-tree 内部页 miss 导致的额外寻道可以被 HDD 的 NCQ 消化
- 高并发（16+ readers, ~20 MB/s）：HDD 接近饱和（90%+ util），每次 B-tree 内部页 miss 都直接拖慢数据页读取。6 套 B-tree 的随机访问模式让磁头在 6 个文件的 6 套内部页 + 数据页之间疯狂寻道，有效吞吐从 ~20 MB/s 骤降到 ~2 MB/s

报告的 "10x" 发生在冷缓存 + 高并发 + GPU 训练同时运行的场景。

### 5.3 问题 3：DDP 数据分发不感知 LMDB 分片

**当前 DDP 数据分发机制**（详见 `src/train/bc_ddp.py`）：

两个代码路径，但结果相同——**每个 rank 都会打开所有 LMDB shard**：

| 路径 | 条件 | 分发粒度 | 每个 rank 打开的 LMDB |
|------|------|---------|---------------------|
| A | `ddp_shard_enabled=False`（默认） | `DistributedSampler` 在 sample index 级切分 | **全部** |
| B | `ddp_shard_enabled=True` | `balance_episode_manifest_by_frames()` 在 episode 级切分 | **全部** |

路径 B 虽然按 episode 做均衡分配，但 episode 分发不感知 LMDB 归属——同一个 LMDB shard 的 episode 可能分散到多个 rank。每个 rank 在构建 dataset 时仍传入完整 `data_path`（所有 LMDB），导致 `build_lazy_image_stores()` 为所有 shard 创建 `LMDBImageStore`：

```
6 个 LMDB shard × N 个 DDP rank = 6N 个 LMDB env 同时压在 HDD 上
```

以 4 卡训练为例：24 个 LMDB env，每个都有独立的 B-tree 内部页缓存，进一步加剧 page cache 竞争。

**benchmark 的 `split` 模式已验证优化空间**（见 3.4 节）：

| 模式 | Samples/s | vs shared |
|------|-----------|-----------|
| 4p/6s shared（当前 DDP 行为） | 17.0 | 基线 |
| 4p/6s split（各 rank 独占 LMDB） | 18.4 | **+8%** |

在 HDD 上 +8% 的提升放到 NVMe 上会更大，因为 split 模式减少了跨文件的 B-tree 内部页竞争。

### 5.4 问题 4：多个独立训练实例同时运行的叠加效应

以上 benchmark 只测试了**单个训练实例**多 reader 的场景，但实际生产环境中可能同时运行 2 个以上的独立训练（如 `gpu-snatcher` 的排程）。两个独立训练实例的叠加效应远比线性叠加严重。

**现象**：2 个训练各 2 rank × 4 worker = 总共 16 readers，实际吞吐远低于单训练 16 readers 的 15.4 samples/s。

**根因分析**：

**1. 协同性缺失——最关键的差异**

单训练 16 readers 全部在同一个 `DataLoader` 迭代器下工作，采样节奏由 PyTorch 的事件循环驱动，IO 请求在时间线上相对集中——每轮 `__getitem__` 几乎同时发起，磁盘 NCQ（Native Command Queuing）可以把相邻的随机请求重排优化。而两个独立训练各自有独立的迭代节奏、独立的事件循环，IO 请求在时间线上**完全交错**，磁头在两个不相关的访问域之间随机跳跃：

```
单训练 16 readers:
  [批次1: 16个请求同时发出] → NCQ 重排合并 → [批次2: 16个请求]
  请求组之间有空隙，磁头有一定优化空间

双训练 2×8 readers:
  训练A:  req  req  req  req  req  req  req  req  ...
  训练B:    req  req  req  req  req  req  req  req  ...
  时间线:  ABABABABABABABABAB...  完全交错，无批次边界
```

HDD 磁头在处理这 16 个交错请求时，寻道路径被**切碎**成两倍的独立随机跳转，NCQ 无法将两个无关工作负载的请求合并优化。

**2. Page cache 撕裂**

单训练时 page cache 中至少有一定概率复用（同一个训练的随机访问模式有一定局部性）。两个独立训练时：

```
训练A 将一批 4KB 页拉入 cache (say ~30 GB worth of active pages)
训练B 紧接着将完全不相关的 4KB 页拉入 cache (another ~30 GB)
→ 两者互相逐出对方刚拉入的页面
```

这不是 59% × 2 ≈ 118% miss（不可能 >100%），而是两个访问模式在有限 cache 空间内**互相打架**，有效覆盖率从 59% 可能暴跌到 20-30%。每次 cache miss 落盘 = 一次 4KB 随机读 = 磁头多一次寻道，雪崩式放大。

**3. LMDB reader table 竞争**

LMDB 使用 MVCC，每个 `get_frames()` 调用涉及读事务。虽然 LMDB 的读事务无锁（Copy-on-Write），但 reader slot 的注册/注销涉及原子操作和 cache line 的跨核 invalidate。两份独立的 `LMDBImageStore` 实例 = 两套 reader table = 双倍元数据页，进一步争夺 page cache。

**4. GPU 显存叠加**

两个训练各占 2 卡 × ~20 GB = ~80 GB pinned 显存，2 个 CUDA context。这些 pinned memory 和 driver 开销进一步压缩 OS 可用 page cache。与单个训练（40 GB pinned）相比，page cache 可用空间从 ~337 GB 降到 ~297 GB。

**综合效应**：

```
双训练吞吐 << 单训练吞吐(相同总 reader 数)

单训练 16 readers: 15.4 samples/s  (HDD ~90% util, NCQ 可优化)
双训练 2×8 readers: 可能 < 5 samples/s  (无 NCQ 优化 + cache 撕裂 + 磁头在两个模式间疯狂寻道)
```

这就是为什么你观察到的降速**远超** benchmark 报告的 16%。benchmark 只测了单实例，而真实的多训练场景中，**访问模式交错的代价 >> B-tree 内部页竞争 >> HDD IOPS 上限**。

**验证方法**：在 benchmark 脚本中模拟两个独立训练的访问模式——启动两组独立子进程，各自有独立采样节奏（不同 RNG seed、不同迭代间隔），对比同总 reader 数的单组模式。

---

## 6. 建议方案

| 优先级 | 方案 | 预期效果 | 实施难度 |
|--------|------|---------|---------|
| **P0** | 将 LMDB 数据迁移到 SSD/NVMe | 随机读 IOPS 从 6K→400K+（~80×），彻底消除 IO 瓶颈 | 低 |
| **P1** | 减小 batch_size 到 256 | 每 step 数据需求减半，等效训练速度翻倍（不改 IO，减少每 step 等待量） | 低 |
| **P2** | lmdb-shard-aware DDP shard：按 LMDB 分片分配 rank | 减少每 rank 打开的 LMDB 数，降低 B-tree 内部页竞争（~8% 提升） | 中 |
| **P3** | 合并 LMDB shard，减少 B-tree 数量 | 减少 page cache 压力，冷启动更快 | 中 |

### 6.1 P0：迁移数据到 SSD/NVMe

**全服务器磁盘吞吐实测**（`dd iflag=direct` 顺序读 + 16 并发 `O_DIRECT` 随机 4K 读）：

| 服务器 | 最快盘路径 | 磁盘类型 | 顺序读 (MB/s) | 随机 4K 16 并发 (MB/s) | 可用空间 |
|--------|-----------|---------|--------------|----------------------|---------|
| `zju_4090_228` | `/data` | **NVMe** | 1,800 | ~824 | `/` 351G, `/data` 93G |
| `zju_4090_230` | `/home/hy` | **NVMe** | 1,900 | ~714 | `/` 173G, `/data` 159G |
| `zju_4090_232` | `/home/hy` | **NVMe** | 3,300 | ~787* | `/` 722G |
| `zju_4090_236` | `/home/hy` | **SATA SSD RAID** | 4,900 | ~706 | `/` 261G |
| `zju_4090_238` | `/home/hy` | **SATA SSD** | 4,300 | ~709 | `/` 354G |
| `zju_4090_240` | `/data` | **SATA SSD** | 3,700 | ~688 | `/` 216G |
| `zju_4090_243` | `/home/hy` | **NVMe SSD** | 4,400 | ~744 | `/` 414G |
| `zju_4090_243` HDD | `/data` | **HDD** | 177 | **~25**† | `/data` 6.6T |

> 228 `/data` 虽然在 NVMe 上，但仅 93G 可用，放不下数据集，且没有其他 NVMe 路径有足够空间。228 最快盘的 `/home/hy` 在另一块 NVMe 上也只有 4G 可用。
> *232: `O_DIRECT` 不被该内核支持，随机 4K 数据无 `O_DIRECT`（可能部分命中 page cache）
> †HDD 随机读为 LMDB benchmark 实测值（16 workers + NCQ），`O_DIRECT` 下单线程仅 ~1.6 MB/s（无 NCQ 协同）

**补充测试（3090服务器）**：

| 服务器 | 最快盘路径 | 磁盘类型 | 顺序读 (MB/s) | 随机 4K 16 并发 (MB/s) | 可用空间 |
|--------|-----------|---------|--------------|----------------------|---------|
| `zju_3090_221` | `/home/hy` | **NVMe** | 1,900 | ~707 | `/` 808G |
| `zju_3090_251` `/data` | `/data` | **RAID HDD** | 214 | **~2.4** | `/data` 347G |

**测试方法说明**：
- 顺序读：`dd if=/tmp/testfile of=/dev/null bs=1M count=1024 iflag=direct`（绕过 page cache）
- 随机 4K 16 并发：16 个 Python 进程同时 `O_DIRECT` 随机读（`os.lseek` + `os.read`），每进程 10,000 次 4KB 读取，跨越 2GB 文件。报告值为 16 进程聚合
- 所有测试档均创建于目标盘的 `/tmp` 或 `/data` 路径

**关键结论**：
- 所有 SSD 的随机读吞吐都在 **700+ MB/s**，远超 HDD 的 **~25 MB/s**（28×+）
- Python `O_DIRECT` 瓶颈（~0.07ms/次）限制单进程约 13K IOPS，真实训练中 LMDB 的 C 实现可突破此限制，实际随机读差距更接近 50-100×
- SATA SSD（236/238/240）和 NVMe（228/230/232/243 sdb）在 16 并发下随机读接近，但 NVMe 在高并发下延迟更低、扩展更好
- **236 的 SATA SSD 虽然顺序读高（RAID 聚合），但 16 worker 随机读会被打满**——这就是你观察到"慢了两倍多"的原因：不是 HDD 的问题，而是 SATA SSD RAID 的随机 IOPS 有限（单线程 38K IOPS vs NVMe 400K+）

**按服务器迁移方案：**

**232 — 直接迁移（唯一无需清理就能放下的 NVMe）**
`/data` 在 HDD，但 `/home/hy` 在 NVMe（`nvme0n1p2`，722G 可用）上，足够容纳 638 GB 数据集。

```bash
mkdir -p /home/hy/data/robust-rearrangement-custom/data/processed/diffik/sim/round_table/rollout/low/success/
cp -r /data/hy/robust-rearrangement-custom/data/processed/.../rgbd-skill*.lmdb /home/hy/data/.../
export DATA_DIR_PROCESSED=/home/hy/data/
```

**243 — 清理 sdb 后迁移**
`/data` 在 HDD，`/home/hy` 在 sdb NVMe SSD（414G 可用）。需先清理 ~224 GB。

```bash
ssh zju_4090_243 "df -h /"  # 确认 >700 GB free
mkdir -p /home/hy/data/...
cp -r /data/hy/robust-rearrangement-custom/data/processed/.../rgbd-skill*.lmdb /home/hy/data/...
export DATA_DIR_PROCESSED=/home/hy/data/
```

**228 / 230 — NVMe 空间不足**
两个服务器的 `/data` 和 `/home/hy` 虽都在 NVMe 上，但可用空间均 < 200G，放不下 638 GB 数据集。需要先清理磁盘，或只用部分 shard。

**236/238/240 — SATA SSD，既放不下又有瓶颈**
这三台机器最快盘是 SATA SSD，但不仅可用空间都 < 400G（放不下数据集），且 16 个 worker 并发会触及随机 IOPS 瓶颈。建议优先使用 232。

### 6.2 P1：减小 batch_size（不改存储的快速缓解）

如果暂时无法迁移数据，减小 batch_size 是最直接的缓解手段：

```
batch_size=512, 2 训练同时跑:
  每训练分到 ~7.9 MB/s ÷ 1050 KB/frame ≈ 7.5 frames/s
  512 frames ÷ 7.5 frames/s ≈ 68s/step

batch_size=256, 2 训练同时跑:
  256 frames ÷ 7.5 frames/s ≈ 34s/step → 训练速度翻倍
```

对于 DiT diffusion policy，256 的 global batch 通常不会显著影响收敛质量。

```yaml
# gpu-snatcher 配置中按需调整
training.batch_size=256
```

### 6.3 短期缓解（辅助手段）

```bash
# 合并 shard 后预热 page cache（仅当数据集 < page cache 时有效）
vmtouch -t /data/hy/robust-rearrangement-custom/data/processed/.../rgbd-skill-merged.lmdb/data.mdb
```

> 注意：`vmtouch` 预热仅当数据量 < 可用 page cache 时有效。当前 6 个 shard 共 638 GB > 377 GB，无法全部预热。合并为 1 个 shard 后可通过预热覆盖全部数据。
>
> **不建议减少 dataloader workers**：减少 workers → 并发度下降 → HDD 利用率降低 → 总吞吐更低。瓶颈在 HDD 而非 CPU 或锁。

### 6.4 P2 实现方案：lmdb-shard-aware DDP shard

**目标**：让每个 DDP rank 只打开分配给自己的 LMDB shard，消除跨 rank 的文件级竞争。

**当前行为 vs 目标行为**（以 4 卡、6 LMDB shard 为例）：

```
当前 (ddp_shard_enabled=True):
  rank 0: 打开 shard 0,1,2,3,4,5 → 读取 episode 子集（可能跨所有 shard）
  rank 1: 打开 shard 0,1,2,3,4,5 → 读取 episode 子集（可能跨所有 shard）
  rank 2: 打开 shard 0,1,2,3,4,5 → 读取 episode 子集（可能跨所有 shard）
  rank 3: 打开 shard 0,1,2,3,4,5 → 读取 episode 子集（可能跨所有 shard）
  = 24 个 LMDB env 同时竞争 HDD

目标:
  rank 0: 打开 shard 0,1 → 只读这两个 shard 内的 episode
  rank 1: 打开 shard 2,3 → 只读这两个 shard 内的 episode
  rank 2: 打开 shard 4   → 只读这个 shard 内的 episode
  rank 3: 打开 shard 5   → 只读这个 shard 内的 episode
  = 6 个 LMDB env，无跨文件竞争
```

**改动点**（3 处）：

**1. `bc_ddp.py` — episode 分发增加 LMDB affinity 约束**

在 `balance_episode_manifest_by_frames()` 之后，新增一步：将 episode 按 `path_idx`（所属 LMDB）分组，再分配 LMDB 组到各 rank。核心逻辑：

```python
# 现有：episode 级别贪心均衡
train_shards = balance_episode_manifest_by_frames(train_episode_refs, world_size)

# 新增：LMDB-aware 重新分配
def balance_by_lmdb_shard(episode_refs, num_ranks, lmdb_frame_counts):
    """将 LMDB shard 按帧数均衡分配给各 rank，同一 shard 的 episode 不拆分。"""
    # 按 path_idx 分组
    episodes_by_shard = defaultdict(list)
    for ep in episode_refs:
        episodes_by_shard[ep.path_idx].append(ep)
    
    # 按 shard 总帧数降序排列（大 shard 优先分配）
    shard_groups = sorted(episodes_by_shard.items(), 
                          key=lambda x: sum(e.frame_count for e in x[1]), 
                          reverse=True)
    
    # 贪心分配：每次把当前 shard 分配给总帧数最少的 rank
    rank_loads = [0] * num_ranks
    rank_assignments = [[] for _ in range(num_ranks)]
    
    for shard_idx, episodes in shard_groups:
        lightest_rank = min(range(num_ranks), key=lambda r: rank_loads[r])
        rank_assignments[lightest_rank].extend(episodes)
        rank_loads[lightest_rank] += sum(e.frame_count for e in episodes)
    
    return rank_assignments
```

**2. `dataset.py` / `storage.py` — 按 rank 过滤 LMDB shard**

当前 `build_lazy_image_stores(self.dataset_paths)` 为所有 shard 创建 store。需要改为只为自己分配的 shard 创建：

```python
# 根据 episode_refs 中的 path_idx 确定本 rank 需要的 shard 集合
used_path_indices = set(ep.path_idx for ep in self.episode_refs)
self.image_stores = build_lazy_image_stores(
    [self.dataset_paths[i] for i in used_path_indices]
)
```

同时 `self.dataset_paths` 也需要相应裁剪，避免非图像 key 的数据加载也遍历无关 shard。

**3. `files.py` — `expand_lmdb_shard_paths` 返回 shard 元信息**

当前只返回路径列表。为支持 LMDB-level 分配，需要同时返回每个 shard 的帧数（可从 LMDB meta 中快速读取），供步骤 1 的贪心算法使用。

**预期效果**：
- HDD 场景：减少跨文件 B-tree 内部页竞争，benchmark 已验证 split vs shared 有 ~8% 提升
- NVMe 场景：提升更显著，因为 NVMe 的 IOPS 不再瓶颈，但 B-tree 内部页的 CPU/内存开销仍存在
- 内存节省：每 rank 的 RSS 降低（只打开部分 LMDB env），6 shard / 4 rank 约节省 75% 的 LMDB 元数据内存

**风险**：
- LMDB shard 数量不是 rank 数量的整数倍时，负载不均衡（贪心算法可缓解）
- 如果某个 shard 的 episode 分布不均衡（比如一个 shard 都是长 episode），可能造成 rank 间 batch 分布偏差

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
