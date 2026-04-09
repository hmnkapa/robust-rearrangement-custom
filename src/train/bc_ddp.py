import os
import random
import re
import shutil
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Optional, Tuple, Union

import hydra
import numpy as np
import torch
import torch.distributed as dist
import wandb
from diffusers.optimization import get_scheduler
from gymnasium import Env
from omegaconf import DictConfig, OmegaConf
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from src.behavior import get_actor
from src.behavior.base import Actor
from src.common.earlystop import EarlyStopper
from src.common.files import get_processed_paths, path_override
from src.common.hydra import to_native
from src.common.pytorch_util import dict_to_device
from src.dataset.dataloader import AsyncDevicePrefetchLoader, build_dataloader
from src.dataset.dataset import ImageDataset, RGBDDataset, StateDataset
from src.eval.eval_utils import get_model_from_api_or_cached
from src.eval.rollout import do_rollout_evaluation
from src.gym import get_rl_env
from src.models.ema import SwitchEMA
from wandb.errors.util import CommError
from wandb_osh.hooks import TriggerWandbSyncHook, _comm_default_dir


def configure_runtime_tmpdir() -> Path:
    # W&B launches a local service that writes port files into a temp directory.
    # On shared machines, TMPDIR can sometimes point at a deleted path after a move
    # or a stale shell session; fall back to a stable writable location.
    candidate_dirs = []
    for env_name in ("TMPDIR", "TEMP", "TMP"):
        value = os.environ.get(env_name)
        if value:
            candidate_dirs.append(Path(value).expanduser())

    candidate_dirs.extend(
        [
            Path("/tmp") / os.environ.get("USER", "user") / "robust-rearrangement",
            Path.cwd() / ".tmp",
        ]
    )

    for candidate_dir in candidate_dirs:
        try:
            candidate_dir.mkdir(parents=True, exist_ok=True)
            probe_file = candidate_dir / ".wandb_tmp_probe"
            with open(probe_file, "w"):
                pass
            probe_file.unlink()

            tmp_dir = str(candidate_dir)
            tempfile.tempdir = tmp_dir
            for env_name in ("TMPDIR", "TEMP", "TMP"):
                os.environ[env_name] = tmp_dir

            wandb_osh_dir = candidate_dir / "wandb_osh"
            wandb_osh_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("WANDB_OSH_COMM_DIR", str(wandb_osh_dir))
            return candidate_dir
        except OSError:
            continue

    raise RuntimeError("Unable to configure a writable temporary directory for W&B.")


RUNTIME_TMP_DIR = configure_runtime_tmpdir()

trigger_sync = TriggerWandbSyncHook(
    communication_dir=os.environ.get("WANDB_OSH_COMM_DIR", _comm_default_dir),
)


print("=== Activate TF32 training? Deactivated for now...")
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True


def ddp_setup():
    required_env = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing_env = [name for name in required_env if name not in os.environ]
    if missing_env:
        raise RuntimeError(
            "bc_ddp.py must be launched with torchrun so that "
            f"{', '.join(missing_env)} are set."
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def is_main_process(rank: int) -> bool:
    return rank == 0


def broadcast_object(obj, src: int = 0):
    object_list = [obj]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]


def distributed_mean(value_sum: float, count: int, device: torch.device) -> float:
    tensor = torch.tensor([value_sum, count], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if tensor[1].item() == 0:
        return float("nan")
    return (tensor[0] / tensor[1]).item()


def load_state_dict_from_path(path: str):
    return torch.load(path, map_location="cpu")


def _checkpoint_epoch(path: Path) -> int:
    match = re.search(r"actor_chkpt_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def _extract_path_datetime(path: Path) -> Optional[datetime]:
    path_str = str(path)
    parsed_times = []

    for pattern, time_format in (
        (r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d+", "%Y-%m-%d_%H-%M-%S.%f"),
        (r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", "%Y-%m-%d_%H-%M-%S"),
    ):
        for match in re.findall(pattern, path_str):
            try:
                parsed_times.append(datetime.strptime(match, time_format))
            except ValueError:
                continue

    split_datetime_matches = re.findall(
        r"(\d{4}-\d{2}-\d{2})[\\/](\d{2}-\d{2}-\d{2}(?:\.\d+)?)",
        path_str,
    )
    for date_part, time_part in split_datetime_matches:
        if "." in time_part:
            time_format = "%Y-%m-%d_%H-%M-%S.%f"
        else:
            time_format = "%Y-%m-%d_%H-%M-%S"
        try:
            parsed_times.append(
                datetime.strptime(f"{date_part}_{time_part}", time_format)
            )
        except ValueError:
            continue

    if not parsed_times:
        return None

    return max(parsed_times)


def _checkpoint_priority(path: Path) -> int:
    if path.name == "actor_chkpt_last.pt":
        return 5
    if re.match(r"actor_chkpt_latest_\d+\.pt$", path.name):
        return 4
    if re.match(r"actor_chkpt_\d+\.pt$", path.name):
        return 3
    if path.name == "actor_chkpt_best_test_loss.pt":
        return 2
    if re.match(r"actor_chkpt_best_test_loss_\d+\.pt$", path.name):
        return 1
    if path.name == "actor_chkpt_best_success_rate.pt":
        return 0
    return -1


def _path_sort_key(path: Path) -> Tuple[float, float]:
    path_datetime = _extract_path_datetime(path)
    mtime = path.stat().st_mtime
    return (
        path_datetime.timestamp() if path_datetime is not None else float("-inf"),
        mtime,
    )


def _resolve_checkpoint_in_run_dir(run_dir: Path) -> Optional[Path]:
    possible_files = [
        run_dir / "actor_chkpt_last.pt",
        run_dir / "actor_chkpt_best_test_loss.pt",
        run_dir / "actor_chkpt_best_success_rate.pt",
    ]
    possible_files.extend(run_dir.glob("actor_chkpt_*.pt"))

    checkpoint_candidates = []
    seen_paths = set()
    for checkpoint_path in possible_files:
        if not checkpoint_path.exists():
            continue

        checkpoint_key = str(checkpoint_path.resolve())
        if checkpoint_key in seen_paths:
            continue
        seen_paths.add(checkpoint_key)

        checkpoint_candidates.append(
            (
                _checkpoint_priority(checkpoint_path),
                _checkpoint_epoch(checkpoint_path),
                checkpoint_path.stat().st_mtime,
                checkpoint_path,
            )
        )

    if not checkpoint_candidates:
        return None

    checkpoint_candidates.sort()
    return checkpoint_candidates[-1][3]


def parse_wandb_run_reference(
    run_reference: str,
    default_project: Optional[str] = None,
    default_entity: Optional[str] = None,
) -> Tuple[Optional[str], str, str]:
    parts = [part for part in run_reference.split("/") if part]

    if len(parts) == 3:
        entity, project, run_id = parts
    elif len(parts) == 2:
        entity = default_entity
        project, run_id = parts
    elif len(parts) == 1:
        entity = default_entity
        project = default_project
        run_id = parts[0]
    else:
        raise ValueError(f"Invalid W&B run reference: '{run_reference}'")

    if project is None:
        raise ValueError(
            "W&B project is required when continue_run_id is not a full run path."
        )

    return entity, project, run_id


def get_wandb_run_paths(project: str, run_id: str, entity: Optional[str] = None):
    paths = []
    if entity:
        paths.append(f"{entity}/{project}/{run_id}")
    paths.append(f"{project}/{run_id}")
    return paths


def find_local_resume_checkpoint(
    model_save_dir: Union[str, Path], run_name_candidates, search_roots
) -> Optional[Path]:
    model_save_dir = Path(model_save_dir)
    for run_name in run_name_candidates:
        if not run_name:
            continue

        candidate_dirs = []
        seen_dirs = set()

        for root in search_roots:
            root = Path(root)
            base_model_dir = (
                model_save_dir if model_save_dir.is_absolute() else root / model_save_dir
            )

            direct_run_dir = base_model_dir / run_name
            if direct_run_dir.exists():
                direct_run_dir_key = str(direct_run_dir.resolve())
                if direct_run_dir_key not in seen_dirs:
                    seen_dirs.add(direct_run_dir_key)
                    candidate_dirs.append(direct_run_dir)

            outputs_root = root / "outputs"
            if not outputs_root.exists():
                continue

            search_pattern = str(model_save_dir / run_name)
            for output_run_dir in outputs_root.glob(f"**/{search_pattern}"):
                if not output_run_dir.exists():
                    continue

                output_run_dir_key = str(output_run_dir.resolve())
                if output_run_dir_key in seen_dirs:
                    continue
                seen_dirs.add(output_run_dir_key)
                candidate_dirs.append(output_run_dir)

        if not candidate_dirs:
            continue

        candidate_dirs.sort(key=_path_sort_key, reverse=True)
        for run_dir in candidate_dirs:
            checkpoint_path = _resolve_checkpoint_in_run_dir(run_dir)
            if checkpoint_path is not None:
                return checkpoint_path

    return None


def resolve_resume_payload(cfg: DictConfig, is_main: bool):
    payload = {
        "cfg_container": OmegaConf.to_container(cfg, resolve=True),
        "state_dict_path": None,
        "resume_message": None,
    }

    if is_main and cfg.wandb.continue_run_id is not None:
        run = None
        run_exists = False
        remote_error = None

        continue_run_reference = cfg.wandb.continue_run_id
        run_entity, run_project, run_id = parse_wandb_run_reference(
            continue_run_reference, cfg.wandb.project, cfg.wandb.get("entity")
        )
        wandb_mode = cfg.wandb.mode
        data_paths_override = cfg.data.data_paths_override
        original_cwd = Path(hydra.utils.get_original_cwd())
        run_paths = get_wandb_run_paths(run_project, run_id, run_entity)
        run_path = run_paths[0]
        run_name_candidates = [cfg.wandb.name, run_id]

        for candidate_run_path in run_paths:
            try:
                run = wandb.Api().run(candidate_run_path)
                run_exists = True
                run_path = candidate_run_path
                run_name_candidates.insert(0, run.name)
                break
            except (ValueError, CommError) as exc:
                remote_error = exc

        resumed_cfg = None
        state_dict_path = None
        resume_message = None

        if run_exists:
            weights_path = None
            try:
                resumed_cfg, weights_path = get_model_from_api_or_cached(
                    run_path, "last", wandb_mode=wandb_mode
                )
            except Exception:
                try:
                    resumed_cfg, weights_path = get_model_from_api_or_cached(
                        run_path, "latest", wandb_mode=wandb_mode
                    )
                except Exception as exc:
                    remote_error = exc

            if weights_path is not None:
                state_dict = load_state_dict_from_path(weights_path)
                epoch_idx = state_dict.get("epoch", run.summary.get("epoch", 0))
                resumed_cfg.training.start_epoch = epoch_idx
                state_dict_path = str(Path(weights_path).resolve())
                resume_message = f"Continuing run {run_id}, {run.name} from wandb checkpoint"

        if state_dict_path is None:
            local_checkpoint_path = find_local_resume_checkpoint(
                cfg.training.model_save_dir,
                run_name_candidates,
                search_roots=[Path.cwd(), original_cwd],
            )

            if local_checkpoint_path is None:
                remote_error_message = (
                    f" W&B lookup failed: {remote_error}" if remote_error is not None else ""
                )
                raise FileNotFoundError(
                    "Could not resume training. No checkpoint was found in W&B or in "
                    f"'{cfg.training.model_save_dir}' or under '{original_cwd / 'outputs'}' "
                    f"for run '{run_id}'."
                    f"{remote_error_message}"
                )

            state_dict = load_state_dict_from_path(str(local_checkpoint_path))
            checkpoint_cfg = state_dict.get("config")
            if checkpoint_cfg is None:
                raise KeyError(
                    f"Local checkpoint {local_checkpoint_path} does not contain a saved config."
                )

            resumed_cfg = OmegaConf.create(checkpoint_cfg)
            epoch_idx = state_dict.get("epoch", 0)
            resumed_cfg.training.start_epoch = epoch_idx + 1
            state_dict_path = str(local_checkpoint_path.resolve())
            resume_message = (
                f"Continuing run {run_id} from local checkpoint {local_checkpoint_path}"
            )

        if run is not None:
            resumed_cfg.wandb.project = run.project
            resumed_cfg.wandb.entity = run.entity
        resumed_cfg.wandb.continue_run_id = run_id
        resumed_cfg.wandb.mode = wandb_mode
        resumed_cfg.data.data_paths_override = data_paths_override

        payload = {
            "cfg_container": OmegaConf.to_container(resumed_cfg, resolve=True),
            "state_dict_path": state_dict_path,
            "resume_message": resume_message,
        }

    return broadcast_object(payload if is_main else None, src=0)


def resolve_seed(cfg: DictConfig, is_main: bool) -> int:
    seed = cfg.get("seed")
    if seed is None and is_main:
        seed = int(np.random.randint(0, 2**32 - 1))
    seed = broadcast_object(seed if is_main else None, src=0)

    OmegaConf.set_struct(cfg, False)
    cfg.seed = int(seed)
    OmegaConf.set_struct(cfg, True)

    return int(seed)


def resolve_remote_checkpoint_path(
    checkpoint_run_id: Optional[str], is_main: bool
) -> Optional[str]:
    checkpoint_path = None

    if is_main and checkpoint_run_id is not None:
        api = wandb.Api()
        run = api.run(checkpoint_run_id)
        checkpoint_path = str(
            Path(
                [f for f in run.files() if f.name.endswith(".pt")][0]
                .download(exist_ok=True)
                .name
            ).resolve()
        )

    return broadcast_object(checkpoint_path if is_main else None, src=0)


def set_dryrun_params(cfg: DictConfig):
    if cfg.dryrun:
        OmegaConf.set_struct(cfg, False)
        cfg.training.steps_per_epoch = 10 if cfg.training.steps_per_epoch != -1 else -1
        cfg.data.data_subset = 5
        cfg.data.dataloader_workers = 0
        cfg.training.sample_every = 1
        cfg.training.eval_every = 1

        if cfg.rollout.rollouts:
            cfg.rollout.every = 1
            cfg.rollout.loss_threshold = float("inf")

        cfg.wandb.mode = "disabled"

        OmegaConf.set_struct(cfg, True)


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def emit_timing_metrics(timing_metrics):
    if not timing_metrics:
        return

    for key, value in timing_metrics.items():
        print(f"TIMING {key}={value:.6f}")

    wandb.summary.update(timing_metrics)


def get_wandb_init_dir() -> Optional[str]:
    wandb_dir = os.environ.get("WANDB_DIR")
    if not wandb_dir:
        return None

    Path(wandb_dir).mkdir(parents=True, exist_ok=True)
    return wandb_dir


def get_wandb_run_dir_name(run, configured_name: Optional[str]) -> str:
    if run.name:
        return run.name
    if configured_name:
        return configured_name
    if getattr(run, "id", None):
        return run.id
    return f"wandb-run-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')}"


def can_write_to_dir(path: Union[str, Path]) -> bool:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    probe_file = path / ".wandb_access_probe"
    try:
        with open(probe_file, "w"):
            pass
        return True
    except OSError:
        return False
    finally:
        try:
            probe_file.unlink()
        except OSError:
            pass


@contextmanager
def patch_wandb_access_checks(paths):
    normalized_paths = set()
    for path in paths:
        if not path:
            continue
        path_obj = Path(path)
        if can_write_to_dir(path_obj):
            normalized_paths.add(str(path_obj.resolve()))

    if not normalized_paths:
        yield
        return

    original_access = os.access

    def patched_access(path, mode, *args, **kwargs):
        try:
            resolved_path = str(Path(path).resolve())
        except (OSError, RuntimeError, TypeError, ValueError):
            resolved_path = None

        if resolved_path in normalized_paths:
            if mode & os.W_OK:
                return True
            if mode & os.R_OK:
                return True
            if mode & os.X_OK:
                return True

        return original_access(path, mode, *args, **kwargs)

    os.access = patched_access
    try:
        yield
    finally:
        os.access = original_access


def log_action_mse(log_dict, category, pred_action, gt_action):
    B, T, _ = pred_action.shape
    pred_action = pred_action.view(B, T, -1, 10)
    gt_action = gt_action.view(B, T, -1, 10)
    log_dict[f"action_sample/{category}_action_mse_error"] = (
        torch.nn.functional.mse_loss(pred_action, gt_action)
    )
    log_dict[f"action_sample/{category}_action_mse_error_pos"] = (
        torch.nn.functional.mse_loss(pred_action[..., :3], gt_action[..., :3])
    )
    log_dict[f"action_sample/{category}_action_mse_error_rot"] = (
        torch.nn.functional.mse_loss(pred_action[..., 3:9], gt_action[..., 3:9])
    )
    log_dict[f"action_sample/{category}_action_mse_error_width"] = (
        torch.nn.functional.mse_loss(pred_action[..., 9], gt_action[..., 9])
    )


def build_save_dict(
    cfg: DictConfig,
    actor: DDP,
    best_test_loss: float,
    best_success_rate: float,
    epoch_idx: int,
    global_step: int,
    optimizers,
    lr_schedulers,
):
    save_dict = {
        "model_state_dict": actor.module.state_dict(),
        "best_test_loss": best_test_loss,
        "best_success_rate": best_success_rate,
        "epoch": epoch_idx,
        "global_step": global_step,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    for (name, opt), scheduler in zip(optimizers, lr_schedulers):
        save_dict[f"{name}_optimizer_state_dict"] = opt.state_dict()
        save_dict[f"{name}_scheduler_state_dict"] = scheduler.state_dict()

    return save_dict


@hydra.main(config_path="../config", config_name="base")
def main(cfg: DictConfig):
    job_start_perf = perf_counter()
    rank, local_rank, world_size = ddp_setup()
    main_process = is_main_process(rank)
    run = None

    try:
        set_dryrun_params(cfg)
        OmegaConf.resolve(cfg)

        resume_payload = resolve_resume_payload(cfg, main_process)
        cfg = OmegaConf.create(resume_payload["cfg_container"])
        OmegaConf.resolve(cfg)

        base_seed = resolve_seed(cfg, main_process)
        process_seed = base_seed + rank
        torch.manual_seed(process_seed)
        np.random.seed(process_seed)
        random.seed(process_seed)

        if main_process and resume_payload["resume_message"] is not None:
            print(resume_payload["resume_message"])

        device = torch.device("cuda", local_rank)
        env: Optional[Env] = None
        state_dict = None

        best_test_loss = float("inf")
        test_loss_mean = float("inf")
        best_success_rate = 0.0
        prev_best_success_rate = 0.0
        global_step = 0
        is_resuming = resume_payload["state_dict_path"] is not None

        if resume_payload["state_dict_path"] is not None:
            state_dict = load_state_dict_from_path(resume_payload["state_dict_path"])
            best_test_loss = state_dict.get("best_test_loss", float("inf"))
            test_loss_mean = best_test_loss
            best_success_rate = state_dict.get("best_success_rate", 0.0)
            prev_best_success_rate = best_success_rate
            global_step = state_dict.get("global_step", 0) or 0

        if cfg.training.batch_size % world_size != 0:
            raise ValueError(
                "In bc_ddp.py, training.batch_size is treated as the global batch size. "
                f"Got training.batch_size={cfg.training.batch_size} and WORLD_SIZE={world_size}."
            )

        per_rank_batch_size = cfg.training.batch_size // world_size
        data_init_start_perf = perf_counter()

        OmegaConf.set_struct(cfg, False)
        if (job_id := os.environ.get("SLURM_JOB_ID")) is not None:
            cfg.slurm_job_id = job_id
        cfg.training.world_size = world_size
        cfg.training.rank = rank
        cfg.training.local_rank = local_rank
        cfg.training.per_rank_batch_size = per_rank_batch_size
        OmegaConf.set_struct(cfg, True)

        if main_process:
            print(OmegaConf.to_yaml(cfg))

        if cfg.data.data_paths_override is None:
            data_path = get_processed_paths(
                controller=to_native(cfg.control.controller),
                domain=to_native(cfg.data.environment),
                task=to_native(cfg.data.task),
                demo_source=to_native(cfg.data.demo_source),
                randomness=to_native(cfg.data.randomness),
                demo_outcome=to_native(cfg.data.demo_outcome),
                suffix=to_native(cfg.data.suffix),
            )
        else:
            data_path = path_override(cfg.data.data_paths_override)

        if main_process:
            print(f"Using data from {data_path}")

        dataset: Union[ImageDataset, StateDataset, RGBDDataset]

        if cfg.observation_type == "image":
            dataset = ImageDataset(
                dataset_paths=data_path,
                pred_horizon=cfg.data.pred_horizon,
                obs_horizon=cfg.data.obs_horizon,
                action_horizon=cfg.data.action_horizon,
                data_subset=cfg.data.data_subset,
                control_mode=cfg.control.control_mode,
                predict_past_actions=cfg.data.predict_past_actions,
                pad_after=cfg.data.get("pad_after", True),
                max_episode_count=cfg.data.get("max_episode_count", None),
                minority_class_power=cfg.data.get("minority_class_power", False),
                load_into_memory=cfg.data.get("load_into_memory", True),
            )
        elif cfg.observation_type == "rgbd":
            dataset = RGBDDataset(
                dataset_paths=data_path,
                pred_horizon=cfg.data.pred_horizon,
                obs_horizon=cfg.data.obs_horizon,
                action_horizon=cfg.data.action_horizon,
                data_subset=cfg.data.data_subset,
                control_mode=cfg.control.control_mode,
                predict_past_actions=cfg.data.predict_past_actions,
                pad_after=cfg.data.get("pad_after", True),
                max_episode_count=cfg.data.get("max_episode_count", None),
                minority_class_power=cfg.data.get("minority_class_power", False),
                load_into_memory=cfg.data.get("load_into_memory", True),
            )
        elif cfg.observation_type == "state":
            dataset = StateDataset(
                dataset_paths=data_path,
                pred_horizon=cfg.data.pred_horizon,
                obs_horizon=cfg.data.obs_horizon,
                action_horizon=cfg.data.action_horizon,
                data_subset=cfg.data.data_subset,
                control_mode=cfg.control.control_mode,
                predict_past_actions=cfg.data.predict_past_actions,
                pad_after=cfg.data.get("pad_after", True),
                max_episode_count=cfg.data.get("max_episode_count", None),
                include_future_obs=cfg.data.include_future_obs,
            )
        else:
            raise ValueError(f"Unknown observation type: {cfg.observation_type}")

        train_size = int(len(dataset) * (1 - cfg.data.test_split))
        test_size = len(dataset) - train_size
        if main_process:
            print(
                f"Splitting dataset into {train_size} train and {test_size} test samples."
            )
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        OmegaConf.set_struct(cfg, False)
        cfg.robot_state_dim = dataset.robot_state_dim
        cfg.action_dim = dataset.action_dim
        if hasattr(dataset, "skill_dim"):
            cfg.skill_dim = dataset.skill_dim
        if cfg.observation_type == "state":
            cfg.parts_poses_dim = dataset.parts_poses_dim
        OmegaConf.set_struct(cfg, True)

        actor: Actor = get_actor(cfg, device)
        actor.set_normalizer(dataset.normalizer)
        actor.to(device)

        OmegaConf.set_struct(cfg, False)
        cfg.data_path = [str(f) for f in data_path]
        cfg.n_episodes = len(dataset.episode_ends)
        cfg.n_samples = dataset.n_samples
        cfg.timestep_obs_dim = actor.timestep_obs_dim
        OmegaConf.set_struct(cfg, True)

        remote_checkpoint_path = resolve_remote_checkpoint_path(
            cfg.training.load_checkpoint_run_id, main_process
        )
        if remote_checkpoint_path is not None:
            if main_process:
                print(f"Loading checkpoint from {cfg.training.load_checkpoint_run_id}")
            actor.load_state_dict(load_state_dict_from_path(remote_checkpoint_path))

        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        trainloader = build_dataloader(
            dataset=train_dataset,
            batch_size=per_rank_batch_size,
            num_workers=cfg.data.dataloader_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            persistent_workers=cfg.data.get("persistent_workers", False),
            prefetch_factor=cfg.data.get("prefetch_factor", None),
            sampler=train_sampler,
            steps_per_epoch=cfg.training.steps_per_epoch,
        )

        testloader = None
        if main_process:
            test_steps_per_epoch = (
                max(int(round(cfg.training.steps_per_epoch * cfg.data.test_split)), 1)
                if cfg.training.steps_per_epoch != -1
                else -1
            )
            testloader = build_dataloader(
                dataset=test_dataset,
                batch_size=per_rank_batch_size,
                num_workers=cfg.data.dataloader_workers,
                shuffle=True,
                pin_memory=True,
                drop_last=False,
                persistent_workers=cfg.data.get("persistent_workers", False),
                prefetch_factor=cfg.data.get("prefetch_factor", None),
                steps_per_epoch=test_steps_per_epoch,
            )

        async_device_prefetch_enabled = bool(
            cfg.data.get("async_device_prefetch", False) and device.type == "cuda"
        )
        if async_device_prefetch_enabled:
            trainloader = AsyncDevicePrefetchLoader(trainloader, device)
            if testloader is not None:
                testloader = AsyncDevicePrefetchLoader(testloader, device)
            if main_process:
                print(f"Async device prefetch enabled across {world_size} DDP ranks.")

        timing_metrics = {
            "timing/data_init_seconds": perf_counter() - data_init_start_perf,
        }

        def prepare_batch(batch):
            if async_device_prefetch_enabled:
                return batch
            return dict_to_device(batch, device)

        actor = DDP(
            actor,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

        opt_noise = torch.optim.AdamW(
            params=actor.module.actor_parameters(),
            lr=cfg.training.actor_lr,
            weight_decay=cfg.regularization.weight_decay,
        )
        lr_scheduler = get_scheduler(
            name=cfg.lr_scheduler.name,
            optimizer=opt_noise,
            num_warmup_steps=cfg.lr_scheduler.warmup_steps,
            num_training_steps=len(trainloader) * cfg.training.num_epochs,
        )

        optimizers = [("actor", opt_noise)]
        lr_schedulers = [lr_scheduler]

        if cfg.observation_type in {"image", "rgbd"}:
            opt_encoder = torch.optim.AdamW(
                params=actor.module.encoder_parameters(),
                lr=cfg.training.encoder_lr,
                weight_decay=cfg.regularization.weight_decay,
            )
            lr_scheduler_encoder = get_scheduler(
                name=cfg.lr_scheduler.name,
                optimizer=opt_encoder,
                num_warmup_steps=cfg.lr_scheduler.encoder_warmup_steps,
                num_training_steps=len(trainloader) * cfg.training.num_epochs,
            )
            optimizers.append(("encoder", opt_encoder))
            lr_schedulers.append(lr_scheduler_encoder)

        if state_dict is not None:
            if "model_state_dict" in state_dict:
                actor.module.load_state_dict(state_dict["model_state_dict"])
                for (name, opt), scheduler in zip(optimizers, lr_schedulers):
                    if f"{name}_optimizer_state_dict" in state_dict:
                        opt.load_state_dict(state_dict[f"{name}_optimizer_state_dict"])
                    if f"{name}_scheduler_state_dict" in state_dict:
                        scheduler.load_state_dict(
                            state_dict[f"{name}_scheduler_state_dict"]
                        )
            else:
                actor.module.load_state_dict(state_dict)

            if main_process and cfg.wandb.continue_run_id is not None:
                print(f"Loaded weights from run {cfg.wandb.continue_run_id}")

        ema = None
        if cfg.training.ema.use:
            ema = SwitchEMA(actor.module, cfg.training.ema.decay)
            ema.register()

        early_stopper = EarlyStopper(
            patience=cfg.early_stopper.patience,
            smooth_factor=cfg.early_stopper.smooth_factor,
        )
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        wandb_init_dir = get_wandb_init_dir()

        def save_checkpoint(path: Path):
            torch.save(
                build_save_dict(
                    cfg,
                    actor,
                    best_test_loss,
                    best_success_rate,
                    epoch_idx,
                    global_step,
                    optimizers,
                    lr_schedulers,
                ),
                str(path),
            )

        if main_process:
            with patch_wandb_access_checks([wandb_init_dir, tempfile.gettempdir()]):
                run = wandb.init(
                    id=cfg.wandb.continue_run_id,
                    name=cfg.wandb.name,
                    resume=None if cfg.wandb.continue_run_id is None else "allow",
                    project=cfg.wandb.project,
                    entity=cfg.wandb.get("entity"),
                    config=config_dict,
                    mode=cfg.wandb.mode,
                    notes=cfg.wandb.notes,
                    dir=wandb_init_dir,
                )

            if cfg.wandb.continue_run_id is not None:
                if run.id != cfg.wandb.continue_run_id:
                    raise RuntimeError(
                        "W&B initialized an unexpected run id: "
                        f"expected '{cfg.wandb.continue_run_id}', got '{run.id}'."
                    )
                if not run.resumed:
                    raise RuntimeError(
                        f"W&B did not resume existing run '{cfg.wandb.continue_run_id}'. "
                        "With resume='allow', this means W&B started a new run with the same "
                        "id instead of attaching to prior history. Refusing to continue."
                    )

            if cfg.wandb.watch_model:
                run.watch(actor.module, log="all", log_freq=1000)

            run_dir_name = get_wandb_run_dir_name(run, cfg.wandb.name)

            print(f"Run name: {run_dir_name}")
            print(f"Run storage location: {run.dir}")
            wandb.config.update(config_dict)

            dataset_stats = {
                "num_samples_train": int(dataset.n_samples * (1 - cfg.data.test_split)),
                "num_samples_test": dataset.n_samples
                - int(dataset.n_samples * (1 - cfg.data.test_split)),
                "num_episodes_train": int(
                    len(dataset.episode_ends) * (1 - cfg.data.test_split)
                ),
                "num_episodes_test": int(
                    len(dataset.episode_ends) * cfg.data.test_split
                ),
                "dataset_metadata": dataset.metadata,
                "world_size": world_size,
                "global_batch_size": cfg.training.batch_size,
                "per_rank_batch_size": per_rank_batch_size,
            }
            wandb.summary.update(dataset_stats)

            starttime = now()
            wandb.summary["start_time"] = starttime

            model_dir_name = run_dir_name
            if is_resuming:
                model_dir_name = (
                    f"{run_dir_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')}"
                )

            model_save_dir = Path(cfg.training.model_save_dir) / model_dir_name
            model_save_dir.mkdir(parents=True, exist_ok=True)

            print(f"Job started at: {starttime}")
            print(f"This process has access to {os.cpu_count()} CPUs.")
            train_loop_start_perf = perf_counter()
            timing_metrics["timing/time_to_train_loop_seconds"] = (
                train_loop_start_perf - job_start_perf
            )
            wandb.summary.update(timing_metrics)
        else:
            model_save_dir = None
            train_loop_start_perf = None

        dist.barrier()

        early_stop = False
        epoch_durations = []
        timing_epoch_10_logged = False
        checkpoint_archive_interval = cfg.training.save_per_epoch
        pbar_desc = (
            f"Epoch ({cfg.task}, {cfg.observation_type}"
            f"{f', {cfg.vision_encoder.model}' if cfg.observation_type in {'image', 'rgbd'} else ''})"
        )

        if main_process:
            epoch_iter = trange(
                cfg.training.start_epoch,
                cfg.training.num_epochs,
                initial=cfg.training.start_epoch,
                total=cfg.training.num_epochs,
                desc=pbar_desc,
            )
        else:
            epoch_iter = range(cfg.training.start_epoch, cfg.training.num_epochs)

        for epoch_idx in epoch_iter:
            epoch_start_perf = perf_counter() if main_process else None
            epoch_log = {"epoch": epoch_idx}
            train_metric_sums = defaultdict(float)
            train_metric_counts = defaultdict(int)
            train_loss_keys = set()

            actor.train()
            train_sampler.set_epoch(epoch_idx)

            tepoch = tqdm(
                trainloader,
                desc=f"Training [rank {rank}]",
                leave=False,
                total=len(trainloader),
                disable=not main_process,
            )
            for batch in tepoch:
                for _, opt in optimizers:
                    opt.zero_grad()

                batch = prepare_batch(batch)
                loss, losses_log = actor(batch)
                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    actor.parameters(),
                    max_norm=1.0 + 1e3 * (1 - cfg.training.clip_grad_norm),
                )

                for (_, opt), scheduler in zip(optimizers, lr_schedulers):
                    opt.step()
                    scheduler.step()

                if ema is not None:
                    ema.update()

                loss_cpu = loss.item()
                train_metric_sums["epoch_loss"] += loss_cpu
                train_metric_counts["epoch_loss"] += 1
                train_metric_sums["grad_norm"] += grad_norm.item()
                train_metric_counts["grad_norm"] += 1

                for key, value in losses_log.items():
                    train_loss_keys.add(key)
                    train_metric_sums[key] += float(value)
                    train_metric_counts[key] += 1

                global_step += 1

                if main_process:
                    tepoch.set_postfix(loss=loss_cpu)

            tepoch.close()

            epoch_log["epoch_loss"] = distributed_mean(
                train_metric_sums["epoch_loss"],
                train_metric_counts["epoch_loss"],
                device,
            )
            epoch_log["train_grad_norm"] = distributed_mean(
                train_metric_sums["grad_norm"],
                train_metric_counts["grad_norm"],
                device,
            )

            for key in sorted(train_loss_keys):
                epoch_log[f"train_{key}"] = distributed_mean(
                    train_metric_sums[key], train_metric_counts[key], device
                )

            for name, opt in optimizers:
                epoch_log[f"{name}_lr"] = opt.param_groups[0]["lr"]

            if (
                main_process
                and cfg.training.eval_every > 0
                and (epoch_idx + 1) % cfg.training.eval_every == 0
            ):
                actor.eval()

                if ema is not None:
                    ema.apply_shadow()

                eval_losses_log = defaultdict(list)
                test_loss = []

                test_tepoch = tqdm(testloader, desc="Validation", leave=False)
                for test_batch in test_tepoch:
                    with torch.no_grad():
                        test_batch = prepare_batch(test_batch)
                        test_loss_val, losses_log = actor.module.compute_loss(test_batch)

                        test_loss_cpu = test_loss_val.item()
                        test_loss.append(test_loss_cpu)
                        test_tepoch.set_postfix(loss=test_loss_cpu)

                        for key, value in losses_log.items():
                            eval_losses_log[key].append(value)

                test_tepoch.close()

                epoch_log["test_epoch_loss"] = test_loss_mean = np.mean(test_loss)
                for key, value in eval_losses_log.items():
                    epoch_log[f"test_{key}"] = np.mean(value)

                if (
                    cfg.rollout.rollouts
                    and (epoch_idx + 1) % cfg.rollout.every == 0
                    and np.mean(test_loss_mean) < cfg.rollout.loss_threshold
                ):
                    if env is None:
                        env = get_rl_env(
                            local_rank,
                            task=cfg.rollout.task,
                            num_envs=cfg.rollout.num_envs,
                            randomness=cfg.rollout.randomness,
                            observation_space=cfg.observation_type,
                            resize_img=False,
                            act_rot_repr=cfg.control.act_rot_repr,
                            action_type=cfg.control.control_mode,
                            parts_poses_in_robot_frame=cfg.rollout.parts_poses_in_robot_frame,
                            headless=True,
                            verbose=True,
                        )

                    best_success_rate = do_rollout_evaluation(
                        config=cfg,
                        env=env,
                        save_rollouts_to_file=cfg.rollout.save_rollouts,
                        save_rollouts_to_wandb=False,
                        actor=actor.module,
                        best_success_rate=best_success_rate,
                        epoch_idx=epoch_idx,
                    )

                if (
                    cfg.training.sample_every > 0
                    and (epoch_idx + 1) % cfg.training.sample_every == 0
                ):
                    with torch.no_grad():
                        train_sampling_batch = prepare_batch(next(iter(trainloader)))
                        pred_action = actor.module.action_pred(train_sampling_batch)
                        gt_action = actor.module.normalizer(
                            train_sampling_batch["action"], "action", forward=False
                        )
                        log_action_mse(epoch_log, "train", pred_action, gt_action)

                        val_sampling_batch = prepare_batch(next(iter(testloader)))
                        gt_action = actor.module.normalizer(
                            val_sampling_batch["action"], "action", forward=False
                        )
                        pred_action = actor.module.action_pred(val_sampling_batch)
                        log_action_mse(epoch_log, "val", pred_action, gt_action)

                if (
                    cfg.training.store_best_test_loss_model
                    and test_loss_mean < best_test_loss
                ):
                    best_test_loss = test_loss_mean
                    save_checkpoint(model_save_dir / "actor_chkpt_best_test_loss.pt")

                if (
                    cfg.training.store_best_success_rate_model
                    and best_success_rate > prev_best_success_rate
                ):
                    prev_best_success_rate = best_success_rate
                    save_checkpoint(model_save_dir / "actor_chkpt_best_success_rate.pt")

                if (
                    cfg.training.checkpoint_interval > 0
                    and (epoch_idx + 1) % cfg.training.checkpoint_interval == 0
                ):
                    save_checkpoint(model_save_dir / f"actor_chkpt_{epoch_idx}.pt")

                if ema is not None:
                    ema.restore()

                early_stop = early_stopper.update(test_loss_mean)
                epoch_log["early_stopper/counter"] = early_stopper.counter
                epoch_log["early_stopper/best_loss"] = early_stopper.best_loss
                epoch_log["early_stopper/ema_loss"] = early_stopper.ema_loss

            if main_process and cfg.training.store_last_model:
                save_checkpoint(model_save_dir / "actor_chkpt_last.pt")

            if (
                main_process
                and checkpoint_archive_interval > 0
                and (epoch_idx + 1) % checkpoint_archive_interval == 0
            ):
                save_checkpoint(model_save_dir / f"actor_chkpt_latest_{epoch_idx + 1}.pt")

                best_test_loss_checkpoint = model_save_dir / "actor_chkpt_best_test_loss.pt"
                if best_test_loss_checkpoint.exists():
                    shutil.copy2(
                        best_test_loss_checkpoint,
                        model_save_dir / f"actor_chkpt_best_test_loss_{epoch_idx + 1}.pt",
                    )

            if ema is not None and cfg.training.ema.switch:
                ema.copy_to_model()

            if main_process:
                wandb.log(epoch_log, step=global_step)
                epoch_iter.set_postfix(
                    time=now(),
                    loss=epoch_log["epoch_loss"],
                    test_loss=test_loss_mean,
                    best_success_rate=best_success_rate,
                    stopper_counter=early_stopper.counter,
                )

                if (
                    cfg.wandb.mode == "offline"
                    and (epoch_idx % cfg.wandb.get("osh_sync_interval", 1)) == 0
                ):
                    trigger_sync()

            dist.barrier()
            early_stop_tensor = torch.tensor(
                [1 if early_stop else 0], device=device, dtype=torch.int32
            )
            dist.broadcast(early_stop_tensor, src=0)
            early_stop = bool(early_stop_tensor.item())

            if main_process:
                epoch_durations.append(perf_counter() - epoch_start_perf)
                if not timing_epoch_10_logged and len(epoch_durations) >= 10:
                    timing_metrics["timing/time_to_epoch_10_total_seconds"] = (
                        perf_counter() - job_start_perf
                    )
                    timing_metrics["timing/time_to_epoch_10_train_loop_seconds"] = (
                        perf_counter() - train_loop_start_perf
                    )
                    timing_metrics["timing/avg_epoch_seconds_first_10"] = float(
                        np.mean(epoch_durations[:10])
                    )
                    emit_timing_metrics(timing_metrics)
                    timing_epoch_10_logged = True

            if early_stop:
                if main_process:
                    print(
                        f"Early stopping at epoch {epoch_idx} as test loss did not improve "
                        f"for {early_stopper.patience} epochs."
                    )
                break

        if main_process and hasattr(epoch_iter, "close"):
            epoch_iter.close()

        if main_process and not timing_epoch_10_logged:
            emit_timing_metrics(timing_metrics)

        if run is not None:
            wandb.finish()
    finally:
        if dist.is_available() and dist.is_initialized():
            destroy_process_group()


if __name__ == "__main__":
    main()
