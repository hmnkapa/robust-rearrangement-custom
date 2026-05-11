#!/usr/bin/env python
"""
LMDB I/O benchmark simulating train.bc_ddp + RGBDDataset access patterns.

Each dataloader worker opens its own LMDBImageStore (PID-based env),
randomly samples frames, and records throughput/latency/memory metrics.

Architecture:
  Main process
    -> Spawns (num_processes * workers_per_process) worker subprocesses
    -> Each worker tagged with (rank_id, worker_id)
    -> Workers report stats via multiprocessing.Queue
    -> Optional system monitor thread (iostat/vmstat)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path so we can import src.dataset.lmdb
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMAGE_MODE_KEYS = {
    "image": ["color_image1", "color_image2"],
    "rgbd": ["color_image1", "color_image2", "depth_image1", "depth_image2"],
    "state": [],
}


def _find_lmdb_shards(lmdb_dir: Path, shard_glob: str) -> List[Path]:
    """Find all LMDB shards in a directory, sorted by name."""
    shards = sorted(lmdb_dir.glob(shard_glob))
    # Filter: must be a directory containing data.mdb
    shards = [s for s in shards if s.is_dir() and (s / "data.mdb").exists()]
    return shards


def _select_shards(all_shards: List[Path], subset_spec: Optional[str]) -> List[Path]:
    """Select a subset of shards.

    subset_spec can be:
      - None or "all": use all shards
      - "0,1,2": comma-separated indices (0-based)
      - "3": single index
    """
    if subset_spec is None or subset_spec.lower() == "all":
        return list(all_shards)
    indices = [int(x.strip()) for x in subset_spec.split(",")]
    selected = []
    for i in indices:
        if i < 0 or i >= len(all_shards):
            raise ValueError(f"Shard index {i} out of range (0..{len(all_shards)-1})")
        selected.append(all_shards[i])
    return selected


def _assign_shards_to_ranks(
    shards: List[Path], num_ranks: int, mode: str
) -> List[List[Path]]:
    """Distribute shards across ranks.

    mode:
      - "shared": every rank gets all shards (simulates shared LMDB access)
      - "split": shards are divided as evenly as possible across ranks
    """
    if mode == "shared":
        return [list(shards) for _ in range(num_ranks)]
    elif mode == "split":
        assignments = [[] for _ in range(num_ranks)]
        for i, shard in enumerate(shards):
            assignments[i % num_ranks].append(shard)
        return assignments
    else:
        raise ValueError(f"Unknown shard_mode: {mode}")


def _read_frame_meta(lmdb_path: Path) -> Tuple[int, int, List[str]]:
    """Return (total_frames, frame_nbytes, ordered_keys) for an LMDB."""
    from src.dataset.lmdb import read_lmdb_meta, read_lmdb_episode_index

    meta = read_lmdb_meta(lmdb_path)
    episodes = read_lmdb_episode_index(lmdb_path)
    total_frames = sum(int(ep["frame_end"]) - int(ep["frame_start"]) for ep in episodes)
    frame_specs = meta["frame_specs"]
    frame_nbytes = frame_specs["total_nbytes"]
    ordered_keys = list(frame_specs["ordered_keys"])
    return total_frames, frame_nbytes, ordered_keys


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------


def _worker_run(
    rank_id: int,
    worker_id: int,
    shard_paths: List[Path],
    image_keys: List[str],
    duration: float,
    warmup: float,
    stats_queue: mp.Queue,
    stop_event: mp.Event,  # unused, kept for future use
):
    """Worker process: open LMDB, randomly sample frames, record stats."""
    import psutil

    from src.dataset.lmdb import LMDBImageStore

    # Open one LMDBImageStore per shard
    stores: List[Tuple[LMDBImageStore, int, int]] = []  # (store, frame_start, frame_count)
    for shard_path in shard_paths:
        store = LMDBImageStore(shard_path)
        from src.dataset.lmdb import read_lmdb_episode_index

        episodes = read_lmdb_episode_index(shard_path)
        frame_count = sum(int(ep["frame_end"]) - int(ep["frame_start"]) for ep in episodes)
        stores.append((store, 0, frame_count))

    if not stores:
        stats_queue.put({"error": f"Rank {rank_id} worker {worker_id}: no shards assigned"})
        return

    # Determine which keys are actually available
    available_keys = stores[0][0].frame_specs["ordered_keys"]
    actual_keys = [k for k in image_keys if k in available_keys]
    if not actual_keys:
        actual_keys = list(available_keys)
    frame_nbytes = stores[0][0].frame_specs["total_nbytes"]

    # Per-key nbytes for accurate throughput
    key_nbytes = 0
    specs = stores[0][0].frame_specs["specs"]
    for k in actual_keys:
        if k in specs:
            key_nbytes += int(specs[k]["nbytes"])
    if key_nbytes == 0:
        key_nbytes = frame_nbytes

    pid = os.getpid()
    proc = psutil.Process(pid)

    latencies: List[float] = []
    total_samples = 0
    total_bytes = 0
    errors = 0
    start_time = time.perf_counter()
    warmup_start = start_time

    # Pre-compute cumulative frame counts for weighted random shard selection
    cum_frames = []
    cum = 0
    for _, _, fc in stores:
        cum += fc
        cum_frames.append(cum)

    try:
        while time.perf_counter() - start_time < duration + warmup:
            # Randomly select a shard (weighted by frame count)
            r = np.random.randint(0, cum)
            store_idx = 0
            for i, cf in enumerate(cum_frames):
                if r < cf:
                    store_idx = i
                    break
            store, _, frame_count = stores[store_idx]

            # Randomly select a frame
            frame_idx = np.random.randint(0, frame_count)

            # Read frame
            t0 = time.perf_counter()
            try:
                frames = store.get_frames([frame_idx], actual_keys)
            except Exception:
                errors += 1
                continue
            elapsed = time.perf_counter() - t0

            total_samples += 1
            total_bytes += key_nbytes

            if time.perf_counter() > warmup_start + warmup:
                latencies.append(elapsed)

            if errors > 100:
                break

    except Exception as e:
        stats_queue.put({"error": f"Rank {rank_id} worker {worker_id}: {e}"})
        return
    finally:
        for store, _, _ in stores:
            try:
                store.close()
            except Exception:
                pass

    elapsed_total = time.perf_counter() - start_time - warmup
    if elapsed_total <= 0:
        elapsed_total = 0.001

    # Collect per-process metrics
    try:
        mem = proc.memory_info()
        rss_mb = mem.rss / (1024 * 1024)
        cpu_pct = proc.cpu_percent(interval=0)
    except Exception:
        rss_mb = 0.0
        cpu_pct = 0.0

    latencies_sorted = sorted(latencies) if latencies else [0.0]
    n = len(latencies_sorted)

    stats_queue.put(
        {
            "rank_id": rank_id,
            "worker_id": worker_id,
            "pid": pid,
            "shard_count": len(stores),
            "samples": total_samples,
            "bytes_total": total_bytes,
            "elapsed": elapsed_total,
            "samples_per_sec": total_samples / elapsed_total,
            "mb_per_sec": (total_bytes / (1024 * 1024)) / elapsed_total,
            "latency_p50": float(latencies_sorted[int(n * 0.5)]),
            "latency_p95": float(latencies_sorted[int(n * 0.95)]) if n > 1 else float(latencies_sorted[0]),
            "latency_p99": float(latencies_sorted[int(n * 0.99)]) if n > 1 else float(latencies_sorted[0]),
            "latency_mean": float(np.mean(latencies_sorted)) if latencies_sorted else 0.0,
            "latency_min": float(latencies_sorted[0]),
            "latency_max": float(latencies_sorted[-1]),
            "rss_mb": rss_mb,
            "cpu_pct": cpu_pct,
            "errors": errors,
        }
    )


# ---------------------------------------------------------------------------
# System monitor
# ---------------------------------------------------------------------------


def _system_monitor_thread(output_dir: Path, tag: str, stop_event: threading.Event):
    """Run iostat and vmstat in the background, write logs to output_dir."""
    log_dir = output_dir / "system_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Try with ts (timestamp prefix), fallback to plain
    ts_cmd = "ts '%H:%M:%.S'"

    # iostat
    iostat_path = shutil.which("iostat")
    if iostat_path:
        iostat_log = log_dir / f"iostat_{tag}.log"
        cmd = f"{iostat_path} -x 1 2>&1"
        # Try ts first
        ts_found = shutil.which("ts")
        if ts_found:
            cmd = f"{iostat_path} -x 1 2>&1 | {ts_found} '%H:%M:%.S'"
        with open(iostat_log, "w") as f:
            proc = subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
    else:
        proc = None

    # vmstat
    vmstat_path = shutil.which("vmstat")
    vmstat_proc = None
    if vmstat_path:
        vmstat_log = log_dir / f"vmstat_{tag}.log"
        cmd = f"{vmstat_path} 1 2>&1"
        ts_found = shutil.which("ts")
        if ts_found:
            cmd = f"{vmstat_path} 1 2>&1 | {ts_found} '%H:%M:%.S'"
        with open(vmstat_log, "w") as f:
            vmstat_proc = subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)

    stop_event.wait()

    for p in [proc, vmstat_proc]:
        if p is not None:
            try:
                p.terminate()
                p.wait(timeout=5)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Stats aggregation
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    config: Dict[str, Any] = field(default_factory=dict)
    workers: List[Dict[str, Any]] = field(default_factory=list)
    aggregate: Dict[str, Any] = field(default_factory=dict)
    system_info: Dict[str, Any] = field(default_factory=dict)


def _aggregate_stats(
    worker_stats: List[Dict],
    config: Dict[str, Any],
    elapsed: float,
    system_info: Dict[str, Any],
) -> BenchmarkResult:
    """Aggregate worker-level stats into a summary."""
    if not worker_stats:
        return BenchmarkResult(config=config, system_info=system_info)

    total_samples = sum(w["samples"] for w in worker_stats)
    total_bytes = sum(w["bytes_total"] for w in worker_stats)
    total_errors = sum(w["errors"] for w in worker_stats)

    all_latencies = []
    for w in worker_stats:
        # We don't store raw latencies per worker to save memory; use per-worker aggregates
        pass

    rss_values = [w["rss_mb"] for w in worker_stats if w.get("rss_mb", 0) > 0]
    cpu_values = [w["cpu_pct"] for w in worker_stats if w.get("cpu_pct", 0) > 0]

    # Group by rank
    by_rank = defaultdict(lambda: {"samples": 0, "bytes": 0, "elapsed": 0, "workers": 0})
    for w in worker_stats:
        r = by_rank[w["rank_id"]]
        r["samples"] += w["samples"]
        r["bytes"] += w["bytes_total"]
        r["workers"] += 1
    for r in by_rank.values():
        r["elapsed"] = max(w["elapsed"] for w in worker_stats if w["rank_id"] == worker_stats[0]["rank_id"])
        r["samples_per_sec"] = r["samples"] / max(r["elapsed"], 0.001)
        r["mb_per_sec"] = (r["bytes"] / (1024 * 1024)) / max(r["elapsed"], 0.001)

    return BenchmarkResult(
        config=config,
        workers=worker_stats,
        aggregate={
            "total_samples": total_samples,
            "total_bytes": total_bytes,
            "total_errors": total_errors,
            "elapsed_sec": elapsed,
            "total_samples_per_sec": total_samples / max(elapsed, 0.001),
            "total_mb_per_sec": (total_bytes / (1024 * 1024)) / max(elapsed, 0.001),
            "avg_samples_per_sec_per_worker": total_samples / max(elapsed, 0.001) / len(worker_stats),
            "avg_rss_mb": float(np.mean(rss_values)) if rss_values else 0.0,
            "max_rss_mb": float(np.max(rss_values)) if rss_values else 0.0,
            "total_rss_mb": float(np.sum(rss_values)) if rss_values else 0.0,
            "avg_cpu_pct": float(np.mean(cpu_values)) if cpu_values else 0.0,
            "by_rank": dict(by_rank),
        },
        system_info=system_info,
    )


# ---------------------------------------------------------------------------
# Drop caches (requires sudo)
# ---------------------------------------------------------------------------


def _drop_caches():
    """Attempt to drop page cache. Requires passwordless sudo or will fail."""
    try:
        subprocess.run(
            ["sudo", "sh", "-c", "sync && echo 3 > /proc/sys/vm/drop_caches"],
            timeout=10,
            capture_output=True,
        )
        print("[info] Dropped page caches")
    except Exception as e:
        print(f"[warn] Could not drop caches: {e}")


# ---------------------------------------------------------------------------
# Collect system info
# ---------------------------------------------------------------------------


def _collect_system_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    info["hostname"] = platform.node()
    info["platform"] = platform.platform()
    info["python"] = sys.version

    # CPU
    info["cpu_count_logical"] = os.cpu_count()
    try:
        import psutil

        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        mem = psutil.virtual_memory()
        info["memory_total_gb"] = mem.total / (1024**3)
        info["memory_available_gb"] = mem.available / (1024**3)
    except Exception:
        pass

    # Disk
    try:
        df = shutil.disk_usage("/data")
        info["disk_total_gb"] = df.total / (1024**3)
        info["disk_free_gb"] = df.free / (1024**3)
    except Exception:
        pass

    return info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_benchmark(
    *,
    lmdb_dir: Path,
    all_shards: List[Path],
    selected_shards: List[Path],
    num_processes: int,
    workers_per_process: int,
    image_keys: List[str],
    duration: float,
    warmup: float,
    shard_mode: str,
    output: Optional[Path],
    system_monitor: bool,
    cold_cache: bool,
    allocate_gpu_memory: bool,
    gpu_ids: Optional[List[int]],
) -> BenchmarkResult:
    """Run a single benchmark configuration."""

    # Assign shards to ranks
    rank_shards = _assign_shards_to_ranks(selected_shards, num_processes, shard_mode)

    # Allocate GPU memory if requested (creates CUDA context to simulate training)
    if allocate_gpu_memory and gpu_ids:
        try:
            import torch

            for gpu_id in gpu_ids[:num_processes]:
                torch.cuda.set_device(gpu_id)
                # Allocate ~2GB to simulate model memory
                _ = torch.zeros(512 * 1024 * 1024, device=f"cuda:{gpu_id}")
            print(f"[info] Allocated GPU memory on devices {gpu_ids[:num_processes]}")
        except Exception as e:
            print(f"[warn] Could not allocate GPU memory: {e}")

    # Drop caches if requested
    if cold_cache:
        _drop_caches()
        time.sleep(2)

    system_info = _collect_system_info()
    system_info["lmdb_shards_used"] = [str(s) for s in selected_shards]
    system_info["total_shards_available"] = len(all_shards)
    system_info["shard_mode"] = shard_mode

    # Start system monitor
    stop_monitor = threading.Event()
    monitor_thread = None
    if system_monitor:
        tag = f"p{num_processes}_w{workers_per_process}_s{len(selected_shards)}_{shard_mode}"
        monitor_thread = threading.Thread(
            target=_system_monitor_thread,
            args=(output.parent if output else Path("."), tag, stop_monitor),
            daemon=True,
        )
        monitor_thread.start()

    # Spawn all workers
    ctx = mp.get_context("spawn")
    stats_queue = ctx.Queue()
    stop_event = ctx.Event()
    processes: List[mp.Process] = []

    total_workers = num_processes * workers_per_process

    if workers_per_process == 0:
        # num_workers=0 mode: each rank reads directly in its own subprocess
        for rank_id in range(num_processes):
            shards_for_rank = rank_shards[rank_id]
            p = ctx.Process(
                target=_worker_run,
                args=(
                    rank_id,
                    0,
                    shards_for_rank,
                    image_keys,
                    duration,
                    warmup,
                    stats_queue,
                    stop_event,
                ),
            )
            p.start()
            processes.append(p)
        total_workers = num_processes
    else:
        for rank_id in range(num_processes):
            shards_for_rank = rank_shards[rank_id]
            for worker_id in range(workers_per_process):
                p = ctx.Process(
                    target=_worker_run,
                    args=(
                        rank_id,
                        worker_id,
                        shards_for_rank,
                        image_keys,
                        duration,
                        warmup,
                        stats_queue,
                        stop_event,
                    ),
                )
                p.start()
                processes.append(p)

    print(
        f"[bench] Starting: {num_processes} ranks x {workers_per_process} workers "
        f"= {total_workers} readers, {len(selected_shards)} shards, "
        f"mode={shard_mode}, duration={duration}s, warmup={warmup}s"
    )

    # Wait for all workers
    t0 = time.perf_counter()
    for p in processes:
        p.join()
    elapsed = time.perf_counter() - t0

    # Stop system monitor
    if monitor_thread is not None:
        stop_monitor.set()
        monitor_thread.join(timeout=5)

    # Collect stats
    worker_stats: List[Dict] = []
    while True:
        try:
            worker_stats.append(stats_queue.get_nowait())
        except Exception:
            break

    result = _aggregate_stats(worker_stats, {
        "lmdb_dir": str(lmdb_dir),
        "num_shards": len(selected_shards),
        "shard_mode": shard_mode,
        "num_processes": num_processes,
        "workers_per_process": workers_per_process,
        "total_workers": total_workers,
        "image_keys": image_keys,
        "duration": duration,
        "warmup": warmup,
        "cold_cache": cold_cache,
        "allocate_gpu_memory": allocate_gpu_memory,
    }, elapsed, system_info)

    # Print summary
    agg = result.aggregate
    print(f"  -> {agg['total_samples_per_sec']:.1f} samples/s, "
          f"{agg['total_mb_per_sec']:.1f} MB/s, "
          f"{agg['total_rss_mb']:.0f} MB RSS total, "
          f"{agg['avg_cpu_pct']:.1f}% avg CPU")

    # Write output
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "config": result.config,
            "aggregate": result.aggregate,
            "workers": result.workers,
            "system_info": result.system_info,
        }

        # Clean non-serializable types
        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_clean(v) for v in obj]
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        output_data = _clean(output_data)
        with open(output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"  -> Results saved to {output}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="LMDB I/O benchmark — simulates train.bc_ddp data loading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--lmdb-dir",
        type=Path,
        required=True,
        help="Directory containing LMDB shards",
    )
    parser.add_argument(
        "--shard-glob",
        default="*.lmdb",
        help="Glob pattern for LMDB shard directories (default: *.lmdb)",
    )
    parser.add_argument(
        "--shard-subset",
        default=None,
        help="Comma-separated shard indices to use, or 'all' (default: all)",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=1,
        help="Number of rank processes to simulate (default: 1)",
    )
    parser.add_argument(
        "--workers-per-process",
        type=int,
        default=4,
        help="Number of dataloader workers per rank (default: 4)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30,
        help="Benchmark duration per run in seconds (default: 30)",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=5,
        help="Warmup duration in seconds (default: 5)",
    )
    parser.add_argument(
        "--mode",
        choices=["image", "rgbd", "state"],
        default="rgbd",
        help="Observation mode: image (2 keys), rgbd (4 keys), state (0 keys) (default: rgbd)",
    )
    parser.add_argument(
        "--image-keys",
        nargs="*",
        default=None,
        help="Explicit image keys to read (overrides --mode)",
    )
    parser.add_argument(
        "--shard-mode",
        choices=["shared", "split"],
        default="shared",
        help="How to assign shards to ranks: shared (all ranks see all shards) "
        "or split (divide shards across ranks) (default: shared)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--system-monitor",
        action="store_true",
        help="Start background iostat/vmstat monitoring",
    )
    parser.add_argument(
        "--cold-cache",
        action="store_true",
        help="Drop page caches before benchmark (requires sudo)",
    )
    parser.add_argument(
        "--allocate-gpu-memory",
        action="store_true",
        help="Allocate GPU memory to simulate training memory pressure",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated GPU device IDs (e.g., '0,1')",
    )
    parser.add_argument(
        "--print-summary-only",
        action="store_true",
        help="Only print summary, no JSON output",
    )

    args = parser.parse_args()

    # Resolve image keys
    if args.image_keys is not None:
        image_keys = args.image_keys
    else:
        image_keys = IMAGE_MODE_KEYS.get(args.mode, [])
    if not image_keys and args.mode == "rgbd":
        image_keys = IMAGE_MODE_KEYS["rgbd"]

    # Find shards
    all_shards = _find_lmdb_shards(args.lmdb_dir, args.shard_glob)
    if not all_shards:
        print(f"[error] No LMDB shards found in {args.lmdb_dir} (glob: {args.shard_glob})")
        sys.exit(1)
    print(f"[info] Found {len(all_shards)} LMDB shard(s) in {args.lmdb_dir}")

    selected_shards = _select_shards(all_shards, args.shard_subset)
    print(f"[info] Using {len(selected_shards)} shard(s): {[s.name for s in selected_shards]}")

    # Validate available keys against first shard
    try:
        _, _, available_keys = _read_frame_meta(selected_shards[0])
        missing = [k for k in image_keys if k not in available_keys]
        if missing:
            print(f"[warn] Requested keys {missing} not in LMDB. Available: {available_keys}")
            image_keys = [k for k in image_keys if k in available_keys]
        print(f"[info] Image keys: {image_keys}")
    except Exception as e:
        print(f"[error] Failed to read LMDB metadata: {e}")
        sys.exit(1)

    # Resolve GPU IDs
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]

    # Determine output path
    output = args.output
    if output is None and not args.print_summary_only:
        tag = f"p{args.num_processes}_w{args.workers_per_process}_s{len(selected_shards)}_{args.shard_mode}_{args.mode}"
        output = Path(f"benchmark_results/bench_{tag}.json")

    run_benchmark(
        lmdb_dir=args.lmdb_dir,
        all_shards=all_shards,
        selected_shards=selected_shards,
        num_processes=args.num_processes,
        workers_per_process=args.workers_per_process,
        image_keys=image_keys,
        duration=args.duration,
        warmup=args.warmup,
        shard_mode=args.shard_mode,
        output=output,
        system_monitor=args.system_monitor,
        cold_cache=args.cold_cache,
        allocate_gpu_memory=args.allocate_gpu_memory,
        gpu_ids=gpu_ids,
    )


if __name__ == "__main__":
    main()
