import os
import random
import re
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Optional, Tuple, Union

import hydra
import numpy as np
import torch
import wandb
from diffusers.optimization import get_scheduler
from gymnasium import Env
from ipdb import set_trace as bp
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import random_split
from tqdm import tqdm, trange

from src.behavior import get_actor
from src.behavior.base import Actor
from src.common.earlystop import EarlyStopper
from src.common.files import get_processed_paths, path_override
from src.common.hydra import to_native
from src.common.pytorch_util import dict_to_device
from src.dataset.dataloader import AsyncDevicePrefetchLoader, build_dataloader
from src.dataset.dataset import ImageDataset, StateDataset, RGBDDataset
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
            # cfg.rollout.num_rollouts = 1
            cfg.rollout.loss_threshold = float("inf")
            # cfg.rollout.max_steps = 10

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


def _checkpoint_epoch(path: Path) -> int:
    match = re.search(
        r"actor_chkpt_(?:latest_|best_test_loss_)?(\d+)\.pt$",
        path.name,
    )
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
):
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


def resolve_resume_state(
    cfg: DictConfig,
) -> Tuple[Optional[DictConfig], Optional[dict], Optional[str]]:
    if cfg.wandb.continue_run_id is None:
        return None, None, None

    run = None
    run_exists = False
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
    remote_error = None

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
    state_dict = None
    resume_message = None

    if run_exists:
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
                weights_path = None

        if weights_path is not None:
            state_dict = torch.load(weights_path, map_location="cpu")
            resume_message = (
                f"Continuing run {run_id}, {run.name} from wandb checkpoint"
            )

    if state_dict is None:
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

        state_dict = torch.load(local_checkpoint_path, map_location="cpu")
        checkpoint_cfg = state_dict.get("config")
        if checkpoint_cfg is None:
            raise KeyError(
                f"Local checkpoint {local_checkpoint_path} does not contain a saved config."
            )

        resumed_cfg = OmegaConf.create(checkpoint_cfg)
        resume_message = (
            f"Continuing run {run_id} from local checkpoint {local_checkpoint_path}"
        )

    if run is not None:
        resumed_cfg.wandb.project = run.project
        resumed_cfg.wandb.entity = run.entity
    resumed_cfg.wandb.continue_run_id = run_id
    resumed_cfg.wandb.mode = wandb_mode
    resumed_cfg.data.data_paths_override = data_paths_override

    epoch_idx = state_dict.get("epoch", 0)
    resumed_cfg.training.start_epoch = epoch_idx + 1

    return resumed_cfg, state_dict, resume_message


# @hydra.main(config_path="../config/bc", config_name="base")
@hydra.main(config_path="../config", config_name="base")
def main(cfg: DictConfig):
    set_dryrun_params(cfg)
    OmegaConf.resolve(cfg)
    job_start_perf = perf_counter()

    # Set the random seed
    if cfg.get("seed") is None:
        OmegaConf.set_struct(cfg, False)
        cfg.seed = np.random.randint(0, 2**32 - 1)
        OmegaConf.set_struct(cfg, True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    print(OmegaConf.to_yaml(cfg))
    env: Optional[Env] = None
    device = torch.device(
        f"cuda:{cfg.training.gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    resumed_cfg, state_dict, resume_message = resolve_resume_state(cfg)
    is_resuming = resumed_cfg is not None

    if is_resuming:
        cfg = resumed_cfg
        print(resume_message)

        best_test_loss = state_dict.get("best_test_loss", float("inf"))
        test_loss_mean = best_test_loss
        best_success_rate = state_dict.get("best_success_rate", 0)
        global_step = state_dict.get("global_step", 0)
        prev_best_success_rate = best_success_rate
    else:
        # Train loop
        best_test_loss = float("inf")
        test_loss_mean = float("inf")
        best_success_rate = 0
        prev_best_success_rate = 0
        global_step = 0

    data_init_start_perf = perf_counter()
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

    print(f"Using data from {data_path}")

    dataset: Union[ImageDataset, StateDataset]

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

    # Split the dataset into train and test (effective, meaning that this is after upsampling)
    train_size = int(len(dataset) * (1 - cfg.data.test_split))
    test_size = len(dataset) - train_size
    print(f"Splitting dataset into {train_size} train and {test_size} test samples.")
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    OmegaConf.set_struct(cfg, False)
    if (job_id := os.environ.get("SLURM_JOB_ID")) is not None:
        cfg.slurm_job_id = job_id

    cfg.robot_state_dim = dataset.robot_state_dim
    cfg.action_dim = dataset.action_dim
    if hasattr(dataset, "skill_dim"):
        cfg.skill_dim = dataset.skill_dim

    if cfg.observation_type == "state":
        cfg.parts_poses_dim = dataset.parts_poses_dim

    # Create the policy network
    actor: Actor = get_actor(
        cfg,
        device,
    )
    actor.set_normalizer(dataset.normalizer)
    actor.to(device)

    # Set the data path in the cfg object
    cfg.data_path = [str(f) for f in data_path]

    # Update the cfg object with the action dimension
    cfg.n_episodes = len(dataset.episode_ends)
    cfg.n_samples = dataset.n_samples

    # Update the cfg object with the observation dimension
    cfg.timestep_obs_dim = actor.timestep_obs_dim
    OmegaConf.set_struct(cfg, True)

    if cfg.training.load_checkpoint_run_id is not None:
        api = wandb.Api()
        run = api.run(cfg.training.load_checkpoint_run_id)
        model_path = (
            [f for f in run.files() if f.name.endswith(".pt")][0]
            .download(exist_ok=True)
            .name
        )
        print(f"Loading checkpoint from {cfg.training.load_checkpoint_run_id}")
        actor.load_state_dict(torch.load(model_path))

    # Create dataloaders
    trainloader = build_dataloader(
        dataset=train_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.dataloader_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        persistent_workers=cfg.data.get("persistent_workers", False),
        prefetch_factor=cfg.data.get("prefetch_factor", None),
        steps_per_epoch=cfg.training.steps_per_epoch,
    )

    test_steps_per_epoch = (
        max(int(round(cfg.training.steps_per_epoch * cfg.data.test_split)), 1)
        if cfg.training.steps_per_epoch != -1
        else -1
    )
    testloader = build_dataloader(
        dataset=test_dataset,
        batch_size=cfg.training.batch_size,
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
        testloader = AsyncDevicePrefetchLoader(testloader, device)
        print(f"Async device prefetch enabled on {device}.")

    timing_metrics = {
        "timing/data_init_seconds": perf_counter() - data_init_start_perf,
    }

    def prepare_batch(batch):
        if async_device_prefetch_enabled:
            return batch
        return dict_to_device(batch, device)

    # Create lists for optimizers and lr schedulers

    opt_noise = torch.optim.AdamW(
        params=actor.actor_parameters(),
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

    if cfg.observation_type == "image" or cfg.observation_type == "rgbd":

        opt_encoder = torch.optim.AdamW(
            params=actor.encoder_parameters(),
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
            actor.load_state_dict(state_dict["model_state_dict"])
            for (name, opt), scheduler in zip(optimizers, lr_schedulers):
                opt.load_state_dict(state_dict[f"{name}_optimizer_state_dict"])
                scheduler.load_state_dict(state_dict[f"{name}_scheduler_state_dict"])

        else:
            actor.load_state_dict(state_dict)

        print(f"Loaded weights from run {cfg.wandb.continue_run_id}")

    if cfg.training.ema.use:
        ema = SwitchEMA(actor, cfg.training.ema.decay)
        ema.register()

    early_stopper = EarlyStopper(
        patience=cfg.early_stopper.patience,
        smooth_factor=cfg.early_stopper.smooth_factor,
    )
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb_init_dir = get_wandb_init_dir()

    # Init wandb
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
        run.watch(actor, log="all", log_freq=1000)

    run_dir_name = get_wandb_run_dir_name(run, cfg.wandb.name)

    # Print the run name and storage location
    print(f"Run name: {run_dir_name}")
    print(f"Run storage location: {run.dir}")

    # In sweeps, the init is ignored, so to make sure that the cfg is saved correctly
    # to wandb we need to log it manually
    wandb.config.update(config_dict)

    # save stats to wandb and update the cfg object
    train_size = int(dataset.n_samples * (1 - cfg.data.test_split))
    test_size = dataset.n_samples - train_size

    dataset_stats = {
        "num_samples_train": train_size,
        "num_samples_test": test_size,
        "num_episodes_train": int(
            len(dataset.episode_ends) * (1 - cfg.data.test_split)
        ),
        "num_episodes_test": int(len(dataset.episode_ends) * cfg.data.test_split),
        "dataset_metadata": dataset.metadata,
    }

    # Add the dataset stats to the wandb summary
    wandb.summary.update(dataset_stats)

    starttime = now()
    wandb.summary["start_time"] = starttime

    # Create model save dir
    model_dir_name = run_dir_name
    if is_resuming:
        model_dir_name = (
            f"{run_dir_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')}"
        )

    model_save_dir = Path(cfg.training.model_save_dir) / model_dir_name
    model_save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Job started at: {starttime}")
    train_loop_start_perf = perf_counter()
    timing_metrics["timing/time_to_train_loop_seconds"] = (
        train_loop_start_perf - job_start_perf
    )
    wandb.summary.update(timing_metrics)

    early_stop = False
    epoch_durations = []
    timing_epoch_10_logged = False

    pbar_desc = f"Epoch ({cfg.task}, {cfg.observation_type}{f', {cfg.vision_encoder.model}' if cfg.observation_type == 'image' else ''})"

    tglobal = trange(
        cfg.training.start_epoch,
        cfg.training.num_epochs,
        initial=cfg.training.start_epoch,
        total=cfg.training.num_epochs,
        desc=pbar_desc,
    )

    checkpoint_archive_interval = cfg.training.save_per_epoch

    def build_save_dict():
        save_dict = {
            "model_state_dict": actor.state_dict(),
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

    def save_checkpoint(path: Path):
        torch.save(build_save_dict(), str(path))

    for epoch_idx in tglobal:
        epoch_start_perf = perf_counter()
        epoch_loss = list()
        test_loss = list()

        epoch_log = {
            "epoch": epoch_idx,
        }

        train_losses_log = defaultdict(list)

        # batch loop
        actor.train()
        tepoch = tqdm(trainloader, desc="Training", leave=False, total=len(trainloader))
        for batch in tepoch:
            # Zero the gradients in all optimizers
            for _, opt in optimizers:
                opt.zero_grad()

            # Get a batch on device and compute loss and gradients
            batch = prepare_batch(batch)
            loss, losses_log = actor.compute_loss(batch)
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=1.0 + 1e3 * (1 - cfg.training.clip_grad_norm),
            )

            # Step the optimizers and schedulers
            for (_, opt), scheduler in zip(optimizers, lr_schedulers):
                opt.step()
                scheduler.step()

            if cfg.training.ema.use:
                ema.update()

            # Log the loss and gradients
            loss_cpu = loss.item()

            train_losses_log["grad_norm"] = grad_norm.item()

            for k, v in losses_log.items():
                train_losses_log[k].append(v)

            epoch_loss.append(loss_cpu)

            # Update the global step
            global_step += 1

            tepoch.set_postfix(loss=loss_cpu)

        tepoch.close()

        epoch_log["epoch_loss"] = np.mean(epoch_loss)

        for k, v in train_losses_log.items():
            epoch_log[f"train_{k}"] = np.mean(v)

        # Add the learning rates to the log
        for name, opt in optimizers:
            epoch_log[f"{name}_lr"] = opt.param_groups[0]["lr"]

        if (
            cfg.training.eval_every > 0
            and (epoch_idx + 1) % cfg.training.eval_every == 0
        ):
            # Evaluation loop
            actor.eval()

            if cfg.training.ema.use:
                ema.apply_shadow()

            eval_losses_log = defaultdict(list)

            test_tepoch = tqdm(testloader, desc="Validation", leave=False)
            for test_batch in test_tepoch:
                with torch.no_grad():
                    # device transfer for test_batch
                    test_batch = prepare_batch(test_batch)

                    # Get test loss
                    test_loss_val, losses_log = actor.compute_loss(test_batch)

                    # logging
                    test_loss_cpu = test_loss_val.item()
                    test_loss.append(test_loss_cpu)
                    test_tepoch.set_postfix(loss=test_loss_cpu)

                    # Append the losses to the log
                    for k, v in losses_log.items():
                        eval_losses_log[k].append(v)

            test_tepoch.close()

            epoch_log["test_epoch_loss"] = test_loss_mean = np.mean(test_loss)
            # Update the epoch log with the mean of the evaluation losses

            for k, v in eval_losses_log.items():
                epoch_log[f"test_{k}"] = np.mean(v)

            if (
                cfg.rollout.rollouts
                and (epoch_idx + 1) % cfg.rollout.every == 0
                and np.mean(test_loss_mean) < cfg.rollout.loss_threshold
            ):
                # Do not load the environment until we successfuly made it this far
                if env is None:
                    env = get_rl_env(
                        cfg.training.gpu_id,
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
                    actor=actor,
                    best_success_rate=best_success_rate,
                    epoch_idx=epoch_idx,
                )

            # Save the model if the test loss is the best so far
            if (
                cfg.training.store_best_test_loss_model
                and test_loss_mean < best_test_loss
            ):
                best_test_loss = test_loss_mean
                save_path = str(model_save_dir / f"actor_chkpt_best_test_loss.pt")
                save_checkpoint(Path(save_path))
                # wandb.save(save_path)

            # Save the model if the success rate is the best so far
            if (
                cfg.training.store_best_success_rate_model
                and best_success_rate > prev_best_success_rate
            ):
                prev_best_success_rate = best_success_rate
                save_path = str(model_save_dir / f"actor_chkpt_best_success_rate.pt")
                save_checkpoint(Path(save_path))
                # wandb.save(save_path)

            if (
                cfg.training.checkpoint_interval > 0
                and (epoch_idx + 1) % cfg.training.checkpoint_interval == 0
            ):
                save_path = str(model_save_dir / f"actor_chkpt_{epoch_idx}.pt")
                save_checkpoint(Path(save_path))
                # wandb.save(save_path)

            # Run diffusion sampling on a training batch
            if (
                cfg.training.sample_every > 0
                and (epoch_idx + 1) % cfg.training.sample_every == 0
            ):

                with torch.no_grad():
                    # sample trajectory from training set, and evaluate difference
                    train_sampling_batch = prepare_batch(next(iter(trainloader)))
                    pred_action = actor.action_pred(train_sampling_batch)
                    gt_action = actor.normalizer(
                        train_sampling_batch["action"], "action", forward=False
                    )
                    log_action_mse(epoch_log, "train", pred_action, gt_action)

                    val_sampling_batch = prepare_batch(next(iter(testloader)))
                    gt_action = actor.normalizer(
                        val_sampling_batch["action"], "action", forward=False
                    )
                    pred_action = actor.action_pred(val_sampling_batch)
                    log_action_mse(epoch_log, "val", pred_action, gt_action)

            # If using EMA, restore the model
            if cfg.training.ema.use:
                ema.restore()

            # Since we now have a new test loss, we can update the early stopper
            early_stop = early_stopper.update(test_loss_mean)
            epoch_log["early_stopper/counter"] = early_stopper.counter
            epoch_log["early_stopper/best_loss"] = early_stopper.best_loss
            epoch_log["early_stopper/ema_loss"] = early_stopper.ema_loss

        # We store the last model at the end of each epoch for better checkpointing
        if cfg.training.store_last_model:
            save_path = str(model_save_dir / f"actor_chkpt_last.pt")
            save_checkpoint(Path(save_path))
            # wandb.save(save_path)

        if checkpoint_archive_interval > 0 and (epoch_idx + 1) % checkpoint_archive_interval == 0:
            save_checkpoint(model_save_dir / f"actor_chkpt_latest_{epoch_idx + 1}.pt")

            best_test_loss_checkpoint = model_save_dir / "actor_chkpt_best_test_loss.pt"
            if best_test_loss_checkpoint.exists():
                shutil.copy2(
                    best_test_loss_checkpoint,
                    model_save_dir
                    / f"actor_chkpt_best_test_loss_{epoch_idx + 1}.pt",
                )

        # If switch is enabled, copy the the shadow to the model at the end of each epoch
        if cfg.training.ema.use and cfg.training.ema.switch:
            ema.copy_to_model()

        # Log epoch stats
        wandb.log(epoch_log, step=global_step)
        tglobal.set_postfix(
            time=now(),
            loss=epoch_log["epoch_loss"],
            test_loss=test_loss_mean,
            best_success_rate=best_success_rate,
            stopper_counter=early_stopper.counter,
        )

        # If we are in offline mode, trigger the sync
        if (
            cfg.wandb.mode == "offline"
            and (epoch_idx % cfg.wandb.get("osh_sync_interval", 1)) == 0
        ):
            trigger_sync()

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

        # Now that everything is logged and restored, we can check if we need to stop
        if early_stop:
            print(
                f"Early stopping at epoch {epoch_idx} as test loss did not improve for {early_stopper.patience} epochs."
            )
            break

    tglobal.close()
    if not timing_epoch_10_logged:
        emit_timing_metrics(timing_metrics)
    wandb.finish()


if __name__ == "__main__":
    main()
