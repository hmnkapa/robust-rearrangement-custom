import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
import time

from src.pc_util.point_cloud_generator import PointCloudGenerator

# 需要在 torch 之前加载 isaacgym 本地扩展以避免段错误
# Preload isaacgym native extensions before importing torch to avoid segfaults
try:
    import isaacgym  # noqa: F401
    from isaacgym import gymapi as _ig_gymapi  # noqa: F401
    from isaacgym import gymtorch as _ig_gymtorch  # noqa: F401
except Exception as _e:
    print(f"[WARN] isaacgym preload failed or unavailable: {_e}")

from gymnasium import Env
import torch  # needs to be after isaac gym imports
from omegaconf import DictConfig, OmegaConf
from src.behavior.base import Actor  # noqa
from src.behavior.base import model_requires_skill_input, model_uses_guidance_point, model_uses_guidance_point_colored
from src.behavior.diffusion import DiffusionPolicy  # noqa
from src.eval.rollout import calculate_success_rate
from src.behavior import get_actor
from src.common.tasks import task2idx, task_timeout
from src.common.files import trajectory_save_dir
from src.gym import get_rl_env
from src.eval.eval_utils import load_model_weights
from src.eval.perturb_util import PERTURB_MODES, PerturbRunner

from typing import Any, Dict, List, Optional
from ipdb import set_trace as bp  # noqa
import wandb
from wandb import Api
from wandb.sdk.wandb_run import Run

_wandb_api: Optional[Api] = None


class LocalCheckpointWrapper:
    def __init__(self, checkpoint_path: str, map_location: Optional[torch.device] = None):
        self.checkpoint_path = Path(checkpoint_path)

        # Load checkpoint using provided map_location (from main's device) to avoid CUDA mismatches
        if map_location is None:
            self.checkpoint = torch.load(self.checkpoint_path)
        else:
            self.checkpoint = torch.load(self.checkpoint_path, map_location=map_location)

        self.config: DictConfig = OmegaConf.create(self.checkpoint["config"])
        self.name = self.checkpoint_path.stem
        self.id = self.name
        self.project = "local_evaluation"
        self.entity = "local"
        self.summary = {}

    @property
    def state(self):
        return "finished"

    def update(self):
        # In a real WandB run, this would push updates to the server
        # For local evaluation, we'll just save the summary to a JSON file
        import json

        summary_path = self.checkpoint_path.with_suffix(".summary.json")
        with open(summary_path, "w") as f:
            json.dump(self.summary, f, indent=2)
        print(f"Updated summary saved to {summary_path}")

    def file(self, name: str):
        # This method would normally return a WandB file object
        # For local evaluation, we'll return a dummy object with a download method
        class DummyFile:
            def __init__(self, path):
                self.path = path
                self.name = os.path.basename(path)

            def download(self, replace=True):
                # For local files, we don't need to download anything
                pass

        return DummyFile(self.checkpoint_path)

    def files(self):
        # This method would normally return an iterator of WandB file objects
        # For local evaluation, we'll return an iterator with just the checkpoint file
        yield self.file(self.checkpoint_path.name)

    def get(self, key: str, default: Any = None) -> Any:
        # This method mimics the behavior of wandb.run.config.get()
        return self.config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        # This method allows accessing config items using square bracket notation
        return self.config[key]


def get_wandb_api() -> Api:
    global _wandb_api
    if _wandb_api is None:
        _wandb_api = Api()
    return _wandb_api


def validate_args(args: argparse.Namespace):
    tasks = args.task if isinstance(args.task, list) else [args.task]
    rollout_after_success_values = (
        args.rollout_after_success
        if isinstance(args.rollout_after_success, list)
        else [args.rollout_after_success]
    )

    assert (
        sum(
            [
                args.run_id is not None,
                args.sweep_id is not None,
                args.project_id is not None,
                args.wt_path is not None,
            ]
        )
        == 1
    ), "Exactly one of run-id, sweep-id, project-id must be provided"
    assert args.run_state is None or all(
        [
            state in ["running", "finished", "failed", "crashed"]
            for state in args.run_state
        ]
    ), (
        "Invalid run-state: "
        f"{args.run_state}. Valid options are: None, running, finished, failed, crashed"
    )

    assert not args.leaderboard, "Leaderboard mode is not supported as of now"

    assert not args.store_video_wandb or args.wandb, "store-video-wandb requires wandb"
    assert not args.skill_on_image or args.annotate_skill, (
        "--skill-on-image requires --annotate-skill"
    )
    assert all(
        value >= 0 for value in rollout_after_success_values
    ), "--rollout-after-success must be non-negative"
    assert len(rollout_after_success_values) in (
        1,
        len(tasks),
    ), (
        "--rollout-after-success must provide either one value for all tasks "
        "or one value per task in the same order as --task"
    )


def resolve_rollout_after_success_by_task(
    tasks: List[str], rollout_after_success_values: List[int]
) -> Dict[str, int]:
    if len(rollout_after_success_values) == 1:
        value = int(rollout_after_success_values[0])
        return {task: value for task in tasks}

    return {
        task: int(value)
        for task, value in zip(tasks, rollout_after_success_values)
    }


def get_runs(args: argparse.Namespace, map_location: Optional[torch.device] = None) -> List[Run]:
    # Clear the cache to make sure we get the latest runs
    if args.wt_path:

        run = LocalCheckpointWrapper(args.wt_path, map_location=map_location)
        # 输出 checkpoint.config
        try:
            print("checkpoint.config:")
            print(OmegaConf.to_yaml(run.config))
        except Exception:
            print(run.config)
        runs = [run]

    else:
        api = get_wandb_api()
        api.flush()
        if args.sweep_id:
            runs: List[Run] = list(api.sweep(args.sweep_id).runs)
        elif args.run_id:
            runs: List[Run] = [api.run(run_id) for run_id in args.run_id]
        elif args.project_id:
            runs: List[Run] = list(api.runs(args.project_id))
        else:
            raise ValueError

        # Filter out the runs based on the run state
        if args.run_state:
            runs = [run for run in runs if run.state in args.run_state]

    return runs


def vision_encoder_field_hotfix(run, config):
    if isinstance(config.vision_encoder, str):
        # Read in the vision encoder config from the `vision_encoder` config group and set it
        OmegaConf.set_readonly(config, False)
        config.vision_encoder = OmegaConf.load(
            f"src/config/vision_encoder/{config.vision_encoder}.yaml"
        )
        OmegaConf.set_readonly(config, True)

        # Write it back to the run
        run.config = OmegaConf.to_container(config, resolve=True)
        run.update()


def convert_state_dict(state_dict):
    if not any(k.startswith("encoder1.0") for k in state_dict.keys()) and not any(
        k.startswith("encoder1.model.nets.3") for k in state_dict.keys()
    ):
        print("Dict already in the correct format")
        return

    # Change all instances of "encoder1.0" to "encoder1" and "encoder2.0" to "encoder2"
    # and all instances of "encoder1.1" to encoder1_proj and "encoder2.1" to "encoder2_proj"
    for k in list(state_dict.keys()):
        if k.startswith("encoder1.0"):
            new_k = k.replace("encoder1.0", "encoder1")
            state_dict[new_k] = state_dict.pop(k)
        elif k.startswith("encoder2.0"):
            new_k = k.replace("encoder2.0", "encoder2")
            state_dict[new_k] = state_dict.pop(k)
        elif k.startswith("encoder1.1"):
            new_k = k.replace("encoder1.1", "encoder1_proj")
            state_dict[new_k] = state_dict.pop(k)
        elif k.startswith("encoder2.1"):
            new_k = k.replace("encoder2.1", "encoder2_proj")
            state_dict[new_k] = state_dict.pop(k)


def format_success_rate(n_success: int, n_rollouts: int) -> str:
    if n_rollouts <= 0:
        return "n/a (0/0)"
    return f"{n_success / n_rollouts:.2%} ({n_success}/{n_rollouts})"


def _coerce_count_dict(counts: Optional[Dict[str, Any]]) -> Dict[str, int]:
    if not isinstance(counts, dict):
        return {}

    normalized_counts: Dict[str, int] = {}
    for key, value in counts.items():
        try:
            normalized_counts[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return normalized_counts


def _merge_count_dicts(
    existing_counts: Optional[Dict[str, Any]],
    new_counts: Optional[Dict[str, Any]],
) -> Dict[str, int]:
    merged_counts: Dict[str, int] = {}
    for source in (_coerce_count_dict(existing_counts), _coerce_count_dict(new_counts)):
        for key, value in source.items():
            merged_counts[key] = merged_counts.get(key, 0) + value
    return merged_counts


def _build_success_rate_dict(
    reached_counts: Dict[str, int],
    completion_counts: Dict[str, int],
) -> Dict[str, float]:
    return {
        key: (completion_counts.get(key, 0) / value if value > 0 else 0.0)
        for key, value in reached_counts.items()
    }


def _print_skill_progress_stats(
    task: str,
    state_counts: Dict[str, int],
    skill_completion_counts: Dict[str, int],
    step_counts: Dict[str, int],
    step_completion_counts: Dict[str, int],
):
    if not state_counts:
        print(f"Skill-state statistics ({task}): unavailable")
        return

    print(f"Reached skill states ({task}):")
    for state_label, reached in state_counts.items():
        print(f"  {state_label}: {reached}")

    print(f"Skill success rates ({task}):")
    for state_label, reached in state_counts.items():
        completed = skill_completion_counts.get(state_label, 0)
        print(
            f"  {state_label}: "
            f"{completed / reached:.2%} ({completed}/{reached})"
        )

    if not step_counts:
        print(f"Assembly step success rates ({task}): unavailable")
        return

    print(f"Assembly step success rates ({task}):")
    for step_label, reached in step_counts.items():
        completed = step_completion_counts.get(step_label, 0)
        print(
            f"  {step_label}: "
            f"{completed / reached:.2%} ({completed}/{reached})"
        )


def _build_progress_summary(
    state_counts: Dict[str, int],
    skill_completion_counts: Dict[str, int],
    step_counts: Dict[str, int],
    step_completion_counts: Dict[str, int],
) -> Dict[str, Dict[str, Any]]:
    return {
        "skill_state_counts": dict(state_counts),
        "skill_completion_counts": dict(skill_completion_counts),
        "skill_success_rates": _build_success_rate_dict(
            state_counts, skill_completion_counts
        ),
        "assembly_step_counts": dict(step_counts),
        "assembly_step_completion_counts": dict(step_completion_counts),
        "assembly_step_success_rates": _build_success_rate_dict(
            step_counts, step_completion_counts
        ),
    }


def _default_eval_logs_dir() -> Path:
    log_dir = Path.cwd() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _safe_path_part(value: str) -> str:
    safe = str(value).strip()
    safe = re.sub(r"[^A-Za-z0-9_.+-]+", "_", safe)
    return safe.strip("._") or "unknown"


def _resolve_checkpoint_name(args: argparse.Namespace) -> str:
    if args.wt_path:
        return Path(args.wt_path).stem

    wt_type = getattr(args, "wt_type", None)
    if wt_type is None:
        return "unknown_checkpoint"

    wt_name = Path(str(wt_type)).stem.strip()
    if not wt_name:
        return "unknown_checkpoint"
    if wt_name.startswith("actor_chkpt_") or wt_name.startswith("student_chkpt_"):
        return wt_name
    if wt_name == "latest":
        return "actor_chkpt_latest"
    return f"actor_chkpt_{wt_name}"


def _write_eval_stats_log(
    *,
    log_dir: Path,
    task_name: str,
    checkpoint_name: str,
    payload: Dict[str, Any],
) -> Path:
    timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")
    directory_parts = ["evaluate_model", _safe_path_part(task_name), _safe_path_part(checkpoint_name)]

    target_dir = log_dir.joinpath(*directory_parts)
    target_dir.mkdir(parents=True, exist_ok=True)

    log_path = target_dir / f"{timestamp}.json"
    duplicate_idx = 1
    while log_path.exists():
        log_path = target_dir / f"{timestamp}_{duplicate_idx:02d}.json"
        duplicate_idx += 1

    with open(log_path, "w") as f:
        import json

        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"Saved evaluation stats log to: {log_path}")
    return log_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=False, nargs="*")
    parser.add_argument("--wt-path", type=str, default=None)

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--n-rollouts", type=int, default=1)
    parser.add_argument("--randomness", type=str, default="low")
    parser.add_argument(
        "--task",
        "-f",
        type=str,
        nargs="+",
        choices=[
            "one_leg",
            "lamp",
            "round_table",
            "desk",
            "square_table",
            "cabinet",
            "mug_rack",
            "factory_peg_hole",
            "bimanual_insertion",
        ],
        required=True,
    )
    parser.add_argument("--n-parts-assemble", type=int, default=None)

    parser.add_argument("--save-rollouts", action="store_true")
    parser.add_argument("--save-failures", action="store_true")
    parser.add_argument("--store-full-resolution-video", action="store_true")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--leaderboard", action="store_true")

    # Define what should be done if the success rate fields are already present
    parser.add_argument(
        "--if-exists",
        type=str,
        choices=["skip", "overwrite", "append", "error"],
        default="error",
    )
    parser.add_argument(
        "--run-state",
        type=str,
        default=None,
        choices=["running", "finished", "failed", "crashed"],
        nargs="*",
    )

    # For batch evaluating runs from a sweep or a project
    parser.add_argument("--sweep-id", type=str, default=None)
    parser.add_argument("--project-id", type=str, default=None)

    parser.add_argument("--continuous-mode", action="store_true")
    parser.add_argument(
        "--continuous-interval",
        type=int,
        default=60,
        help="Pause interval before next evaluation",
    )
    parser.add_argument("--ignore-currently-evaluating-flag", action="store_true")

    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--store-video-wandb", action="store_true")
    parser.add_argument("--eval-top-k", type=int, default=None)
    parser.add_argument(
        "--action-type", type=str, default="pos", choices=["delta", "pos", "relative"]
    )
    parser.add_argument("--prioritize-fewest-rollouts", action="store_true")
    parser.add_argument("--multitask", action="store_true")
    parser.add_argument("--compress-pickles", action="store_true")
    parser.add_argument(
        "--output-only-pickle",
        action="store_true",
        help="When saving rollouts locally, only write pickle files and skip txt/mp4 side outputs.",
    )
    parser.add_argument("--max-rollouts", type=int, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose assembly debug output from the evaluation environments.",
    )
    parser.add_argument("--max-rollout-steps", type=int, default=None)
    parser.add_argument("--april-tags", action="store_true")

    parser.add_argument(
        "--observation-space", choices=["image", "state"], default="state"
    )
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--wt-type", type=str, default="best_success_rate")

    parser.add_argument("--stop-after-n-success", type=int, default=0)
    parser.add_argument("--break-on-n-success", action="store_true")
    parser.add_argument(
        "--target-successes",
        type=int,
        default=0,
        help="If >0, keep running rollouts until this many successes are collected. "
        "n-rollouts acts as the batch size per iteration.",
    )
    parser.add_argument(
        "--perturb-mode",
        type=str,
        choices=PERTURB_MODES,
        default="none",
        help="End-effector perturbation mode. All non-mode parameters live in src/eval/perturb_util.py.",
    )
    parser.add_argument(
        "--rollout-after-success",
        type=int,
        nargs="+",
        default=[0],
        help="After the success step, continue rollout for this many additional frames before stopping. Max rollout steps still apply.",
    )
    parser.add_argument(
        "--full-length-rollout",
        action="store_true",
        help="Continue each rollout until max rollout steps even after success, and save the full trajectory.",
    )
    parser.add_argument("--record-for-coverage", action="store_true")
    parser.add_argument("--annotate-skill", action="store_true")
    parser.add_argument("--guidance-point-on-image", action="store_true")
    parser.add_argument("--guidance-point-colored", action="store_true")
    parser.add_argument("--skill-on-image", action="store_true")

    parser.add_argument("--save-rollouts-suffix", type=str, default="")
    parser.add_argument(
        "--rollout-suffix-model-name",
        type=str,
        default=None,
        help=(
            "Optional extra subdirectory inserted into the raw rollout save path "
            "(placed after task-group suffix and before success/failure)."
        ),
    )
    parser.add_argument(
        "--task-group",
        type=str,
        default=None,
        help="Optional task group name used for rollout path suffix when running multitask via subprocesses.",
    )
    parser.add_argument(
        "--task-summary-out",
        type=str,
        default=None,
        help="Optional path to write JSON summary for this task run.",
    )

    # params for RGBD
    parser.add_argument("--save-depth-image", action="store_true", help="Enable depth images collection")
    # additional params for point cloud observation
    parser.add_argument("--save-pc-for-dp3", action="store_true", help="Enable point cloud generation and pickle export for DP3")
    parser.add_argument("--pc-points", type=int, default=4096, help="Downsampled point count for generated point clouds")
    parser.add_argument("--pc-bbox-half-extent", type=float, nargs="+", default=[0.2], help="Half-extent of the cubic bbox for point cloud cropping (in meters). Can be scalar or [x, y, z]")
    parser.add_argument(
        "--pc-downsample-mode",
        type=str,
        default="random",
        choices=["random", "uniform", "fps"],
        help="Downsample mode for point cloud generation",
    )
    parser.add_argument(
        "--pc-bbox-crop-mode",
        type=str,
        default="eepose-centered",
        choices=["eepose-centered", "fixed-scene"],
        help="Mode for 3D bbox cropping: 'eepose-centered' (default) or 'fixed-scene'",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Validate the arguments
    validate_args(args)

    # Make the device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    tasks: List[str] = args.task if isinstance(args.task, list) else [args.task]
    rollout_after_success_by_task = resolve_rollout_after_success_by_task(
        tasks, args.rollout_after_success
    )
    if len(tasks) > 1:
        args.multitask = True
    task_group = (
        args.task_group
        if args.task_group is not None
        else ("+".join(tasks) if len(tasks) > 1 else None)
    )
    eval_logs_dir = _default_eval_logs_dir()

    # If multiple tasks are provided, run each task in a fresh process to avoid IsaacGym core dumps.
    is_child = os.environ.get("RR_EVAL_MT_CHILD", "0") == "1"
    if len(tasks) > 1 and not is_child:
        base_argv = []
        skip_next = False
        raw_argv = sys.argv[1:]
        i = 0
        while i < len(raw_argv):
            arg = raw_argv[i]
            if skip_next:
                skip_next = False
                i += 1
                continue
            if arg in ["-f", "--task"]:
                i += 1
                while i < len(raw_argv) and not raw_argv[i].startswith("-"):
                    i += 1
                continue
            if arg == "--rollout-after-success":
                i += 1
                while i < len(raw_argv) and not raw_argv[i].startswith("-"):
                    i += 1
                continue
            if arg in ["--task-group", "--task-summary-out"]:
                skip_next = True
                i += 1
                continue
            base_argv.append(arg)
            i += 1

        total_success_all_tasks = 0
        total_rollouts_all_tasks = 0
        per_task_summaries = {}

        for task in tasks:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
                tmp.write(b"{}")
                summary_path = tmp.name

            cmd = [
                sys.executable,
                "-m",
                "src.eval.evaluate_model",
                *base_argv,
                "-f",
                task,
                "--rollout-after-success",
                str(rollout_after_success_by_task[task]),
            ]
            if task_group is not None:
                cmd.extend(["--task-group", task_group])
            cmd.extend(["--task-summary-out", summary_path])

            child_env = os.environ.copy()
            child_env["RR_EVAL_MT_CHILD"] = "1"
            result = subprocess.run(cmd, env=child_env)
            if result.returncode != 0:
                print(f"[ERROR] Task {task} failed with return code {result.returncode}")
                sys.exit(result.returncode)

            try:
                import json

                with open(summary_path, "r") as f:
                    summary = json.load(f)
                task_n_success = summary.get("n_success", 0)
                task_n_rollouts = summary.get("n_rollouts", 0)
                per_task_summaries[task] = summary
                total_success_all_tasks += task_n_success
                total_rollouts_all_tasks += task_n_rollouts
            finally:
                try:
                    os.remove(summary_path)
                except OSError:
                    pass

        print("Final success rate summary:")
        for task in tasks:
            task_summary = per_task_summaries.get(
                task, {"n_success": 0, "n_rollouts": 0}
            )
            print(
                f"Success rate ({task}): "
                f"{format_success_rate(task_summary['n_success'], task_summary['n_rollouts'])}"
            )
        print(
            "Success rate (all tasks): "
            f"{format_success_rate(total_success_all_tasks, total_rollouts_all_tasks)}"
        )
        multitask_payload = {
            "task_group": task_group,
            "checkpoint_name": _resolve_checkpoint_name(args),
            "n_success": total_success_all_tasks,
            "n_rollouts": total_rollouts_all_tasks,
            "success_rate": (
                total_success_all_tasks / total_rollouts_all_tasks
                if total_rollouts_all_tasks > 0
                else None
            ),
            "per_task": per_task_summaries,
        }
        _write_eval_stats_log(
            log_dir=eval_logs_dir,
            task_name=task_group or "multitask",
            checkpoint_name=_resolve_checkpoint_name(args),
            payload=multitask_payload,
        )
        sys.exit(0)

    primary_task = tasks[0]
    primary_spf = f"{primary_task}/" if args.multitask else ""

    # Get the environment(s)
    # TODO: This needs to be changed to enable recreation the env for each run
    print(
        f"Creating the environment(s) with action_type {args.action_type} (this needs to be changed to enable recreation the env for each run)"
    )
    env: Optional[Env] = None
    env_task: Optional[str] = None
    run: Optional[Run] = None

    summary_total_success = 0
    summary_total_rollouts = 0
    summary_state_counts: Dict[str, int] = {}
    summary_skill_completion_counts: Dict[str, int] = {}
    summary_step_counts: Dict[str, int] = {}
    summary_step_completion_counts: Dict[str, int] = {}

    # Start the evaluation loop
    print(f"Starting evaluation loop in continuous mode: {args.continuous_mode}")
    try:
        while True:
            runs = get_runs(args, map_location=device)

            if args.eval_top_k is not None:
                runs = sorted(
                    runs,
                    key=lambda current_run: current_run.summary.get(
                        primary_spf + "success_rate", 0
                    ),
                    reverse=True,
                )[: args.eval_top_k]

            runs = sorted(
                runs,
                key=lambda current_run: current_run.summary.get(
                    primary_spf + "n_rollouts", 0
                ),
            )

            print(f"Found {len(runs)} runs to evaluate:")
            for current_run in runs:
                print(
                    f"    Run: {current_run.name}: "
                    f"{current_run.summary.get(primary_spf + 'n_rollouts', 0)}, "
                    f"{current_run.summary.get(primary_spf + 'success_rate', None)}"
                )

            for run in runs:
                if not args.wt_path:
                    api = get_wandb_api()
                    api.flush()
                    run = api.run("/".join([run.project, run.id]))

                if (
                    run.config.get("currently_evaluating", False)
                    and not args.ignore_currently_evaluating_flag
                ):
                    print(f"Run: {run.name} is currently being evaluated, skipping")
                    continue

                if args.wandb:
                    print(
                        f"Setting currently_evaluating flag to true for run: {run.name}"
                    )
                    run.config["currently_evaluating"] = True
                    run.update()

                test_epoch_loss = run.summary.get("test_epoch_loss", None)
                print(
                    f"Evaluating run: {run.name} at test_epoch_loss: {test_epoch_loss}"
                )

                cfg = OmegaConf.create(run.config)
                assert cfg.control.control_mode == args.action_type

                print(OmegaConf.to_yaml(cfg))

                if "base_policy" in cfg:
                    print("Applying residual field hotfix")
                    cfg.action_dim = cfg.base_policy.action_dim

                if "student_policy" in cfg:
                    print("Applying dagger field hotfix")
                    cfg.action_dim = cfg.student_policy.action_dim

                if "critic" in cfg:
                    print("Applying critic field hotfix")
                    cfg.actor.critic = cfg.critic
                    cfg.actor.init_logstd = cfg.init_logstd
                    cfg.discount = cfg.base_policy.discount

                requires_skill_input = model_requires_skill_input(cfg)
                uses_guidance_point = model_uses_guidance_point(cfg)
                uses_guidance_point_colored = model_uses_guidance_point_colored(cfg)
                actor_name = cfg.actor_name if "actor_name" in cfg else cfg.actor.name
                print(
                    "Skill input requirement: "
                    f"{requires_skill_input} "
                    f"(observation_type={cfg.observation_type}, "
                    f"actor={actor_name}, skill_dim={cfg.get('skill_dim', None)})"
                )
                if args.action_horizon is not None:
                    cfg.actor.action_horizon = args.action_horizon
                    print(f"Overriding action_horizon to {args.action_horizon}")

                actor: Actor = get_actor(cfg=cfg, device=device)

                if isinstance(actor, DiffusionPolicy):
                    actor.inference_steps = 4

                if args.wt_path:
                    if cfg.actor.name != "dp3":
                        actor.load_state_dict(run.checkpoint["model_state_dict"])
                    actor.eval()
                    actor.to(device)
                else:
                    actor = load_model_weights(run=run, actor=actor, wt_type=args.wt_type)

                if actor is None:
                    print(
                        f"Skipping run: {run.name} as no weights for wt_type: {args.wt_type} was not found"
                    )
                    if args.wandb:
                        run.config["currently_evaluating"] = False
                        run.update()
                    continue

                total_success = 0
                total_rollouts = 0
                checkpoint_name = _resolve_checkpoint_name(args)

                for task in tasks:
                    task_rollout_after_success = rollout_after_success_by_task[task]
                    spf = f"{task}/" if args.multitask else ""
                    rollout_max_steps = (
                        task_timeout(task, n_parts=args.n_parts_assemble)
                        if args.max_rollout_steps is None
                        else args.max_rollout_steps
                    )

                    if args.max_rollouts is not None:
                        if run.summary.get(spf + "n_rollouts", 0) >= args.max_rollouts:
                            print(
                                f"Run: {run.name} task {task} has already been evaluated "
                                f"{run.summary.get(spf + 'n_rollouts', 0)} times, skipping"
                            )
                            continue

                    how_update = "overwrite"
                    if run.summary.get(spf + "success_rate", None) is not None:
                        if args.if_exists == "skip":
                            print(
                                f"Run: {run.name} task {task} has already been evaluated, skipping"
                            )
                            continue
                        if args.if_exists == "error":
                            raise ValueError(
                                f"Run: {run.name} task {task} has already been evaluated"
                            )
                        if args.if_exists == "overwrite":
                            print(
                                f"Run: {run.name} task {task} has already been evaluated, overwriting"
                            )
                            how_update = "overwrite"
                        elif args.if_exists == "append":
                            print(
                                f"Run: {run.name} task {task} has already been evaluated, appending"
                            )
                            how_update = "append"

                    suffix = args.save_rollouts_suffix
                    if actor_name == "dp3":
                        suffix = "dp3"
                    if args.save_pc_for_dp3:
                        suffix = f"pc/{args.pc_points}/{args.pc_downsample_mode}"
                    if args.save_depth_image:
                        suffix = "rgbd"
                    if args.annotate_skill:
                        if args.guidance_point_on_image:
                            if args.guidance_point_colored:
                                suffix = "rgbd-skill-colored" if args.save_depth_image else "rgb-skill-colored"
                            else:
                                suffix = "rgbd-skill" if args.save_depth_image else "rgb-skill"
                        else:
                            suffix = "rgbd-only-skill" if args.save_depth_image else "rgb-only-skill"
                    if task_group:
                        suffix = f"{suffix}/{task_group}" if suffix else task_group
                    if args.rollout_suffix_model_name is not None:
                        model_name = args.rollout_suffix_model_name.strip()
                        if model_name:
                            suffix = f"{suffix}/{model_name}" if suffix else model_name

                    save_dir = (
                        trajectory_save_dir(
                            controller="diffik",
                            domain="sim",
                            task=task,
                            demo_source="rollout",
                            randomness=args.randomness,
                            suffix=suffix,
                            create=True,
                        )
                        if args.save_rollouts
                        else None
                    )
                    rollout_path_hint = (
                        save_dir
                        if save_dir is not None
                        else (
                            Path("diffik")
                            / "sim"
                            / task
                            / "rollout"
                            / args.randomness
                            / suffix
                        )
                    )

                    if args.store_video_wandb:
                        wandb.init(
                            project=run.project,
                            entity=run.entity,
                            id=run.id,
                            resume="allow",
                        )

                    if env is None or env_task != task:
                        if env is not None and env_task != task:
                            close_fn = getattr(env, "close", None)
                            if callable(close_fn):
                                close_fn()
                            env = None
                            env_task = None

                        env_obs_keys = None
                        if (
                            args.save_pc_for_dp3
                            or args.save_depth_image
                            or actor_name == "dp3"
                        ):
                            from src.gym import FULL_OBS

                            env_obs_keys = list(FULL_OBS)
                            if args.observation_space == "state":
                                env_obs_keys = [
                                    key
                                    for key in env_obs_keys
                                    if "color_image" not in key
                                ]
                            if "depth_image1" not in env_obs_keys:
                                env_obs_keys.append("depth_image1")
                            if "depth_image2" not in env_obs_keys:
                                env_obs_keys.append("depth_image2")

                        env = get_rl_env(
                            gpu_id=args.gpu,
                            task=task,
                            num_envs=args.n_envs,
                            randomness=args.randomness,
                            observation_space=args.observation_space,
                            max_env_steps=5_000,
                            resize_img=False,
                            act_rot_repr=cfg.control.act_rot_repr,
                            action_type=args.action_type,
                            april_tags=args.april_tags,
                            verbose=args.verbose,
                            debug=args.debug,
                            headless=not args.visualize,
                            obs_keys=env_obs_keys,
                        )
                        env_task = task

                    pc_generator = None
                    if args.save_pc_for_dp3 or actor_name == "dp3":
                        bbox_ext = args.pc_bbox_half_extent
                        if isinstance(bbox_ext, list) and len(bbox_ext) == 1:
                            bbox_ext = bbox_ext[0]
                        pc_generator = PointCloudGenerator(
                            env=env,
                            camera_name="front",
                            max_points=args.pc_points,
                            bbox_half_extent=bbox_ext,
                            bbox_crop_mode=args.pc_bbox_crop_mode,
                        )

                    print(
                        f"Starting rollout of run: {run.name} task: {task} "
                        f"(rollout_after_success={task_rollout_after_success}, "
                        f"perturb_mode={args.perturb_mode})"
                    )
                    actor.set_task(task2idx[task])
                    perturb_runner = (
                        None
                        if args.perturb_mode == "none"
                        else PerturbRunner(args.perturb_mode)
                    )
                    rollout_stats = calculate_success_rate(
                        actor=actor,
                        env=env,
                        n_rollouts=args.n_rollouts,
                        target_successes=args.target_successes or None,
                        rollout_max_steps=rollout_max_steps,
                        epoch_idx=0,
                        discount=cfg.discount,
                        rollout_save_dir=save_dir,
                        save_failures=args.save_failures,
                        n_parts_assemble=args.n_parts_assemble,
                        compress_pickles=args.compress_pickles,
                        resize_video=not args.store_full_resolution_video,
                        break_on_n_success=args.break_on_n_success,
                        stop_after_n_success=args.stop_after_n_success,
                        rollout_after_success=task_rollout_after_success,
                        full_length_rollout=args.full_length_rollout,
                        record_first_state_only=args.record_for_coverage,
                        pc_generator=pc_generator,
                        annotate_skill=args.annotate_skill,
                        annotate_guidance_point=uses_guidance_point,
                        guidance_point_on_image=args.guidance_point_on_image,
                        guidance_point_colored=args.guidance_point_colored,
                        model_guidance_point_colored=uses_guidance_point_colored,
                        skill_on_image=args.skill_on_image,
                        provide_skill_input=requires_skill_input,
                        collect_skill_stats=args.annotate_skill,
                        output_only_pickle=args.output_only_pickle,
                        perturb_runner=perturb_runner,
                    )

                    if args.store_video_wandb:
                        wandb.finish()

                    success_rate = rollout_stats.success_rate
                    print(
                        f"Success rate ({task}): "
                        f"{format_success_rate(rollout_stats.n_success, rollout_stats.n_rollouts)}"
                    )
                    # DEBUG: print training-equivalent action_norm_mean
                    if hasattr(actor, '_train_metric_norms') and actor._train_metric_norms:
                        import numpy as np
                        tmn = np.array(actor._train_metric_norms)
                        tmn_n = np.array(actor._train_metric_norms_with_noise)
                        bpn = np.array(actor._base_pos_norms)
                        srpn = np.array(actor._scaled_res_pos_norms)
                        print(f"[TRAIN_METRIC] action_norm_mean (mean, pos3d, unscaled) = {tmn.mean():.4f} std={tmn.std():.4f}")
                        print(f"[TRAIN_METRIC] action_norm_mean (with noise, pos3d, unscaled) = {tmn_n.mean():.4f} std={tmn_n.std():.4f}")
                        print(f"[TRAIN_METRIC] base_action pos3d norm (normalized) = {bpn.mean():.4f}")
                        print(f"[TRAIN_METRIC] residual pos3d norm (scaled by action_scale) = {srpn.mean():.4f}")
                        print(f"[TRAIN_METRIC] residual scaled / base (pos3d) = {srpn.mean()/bpn.mean()*100:.1f}%")
                    _print_skill_progress_stats(
                        task=task,
                        state_counts=rollout_stats.state_counts,
                        skill_completion_counts=rollout_stats.skill_completion_counts,
                        step_counts=rollout_stats.step_counts,
                        step_completion_counts=rollout_stats.step_completion_counts,
                    )
                    task_payload = {
                        "run_name": run.name,
                        "run_id": getattr(run, "id", None),
                        "task": task,
                        "task_group": task_group,
                        "checkpoint_name": checkpoint_name,
                        "n_success": rollout_stats.n_success,
                        "n_rollouts": rollout_stats.n_rollouts,
                        "success_rate": success_rate,
                        "rollout_max_steps": rollout_stats.rollout_max_steps,
                        "total_return": rollout_stats.total_return,
                        "total_reward": rollout_stats.total_reward,
                        "rollout_path_hint": str(rollout_path_hint),
                        "perturb_mode": args.perturb_mode,
                        **_build_progress_summary(
                            state_counts=rollout_stats.state_counts,
                            skill_completion_counts=rollout_stats.skill_completion_counts,
                            step_counts=rollout_stats.step_counts,
                            step_completion_counts=rollout_stats.step_completion_counts,
                        ),
                    }
                    if perturb_runner is not None:
                        task_payload["perturb_stats"] = perturb_runner.stats.summary()
                    _write_eval_stats_log(
                        log_dir=eval_logs_dir,
                        task_name=task,
                        checkpoint_name=checkpoint_name,
                        payload=task_payload,
                    )

                    if args.wandb:
                        print("Writing to wandb...")
                        s: dict = run.summary

                        if how_update == "overwrite":
                            state_counts = dict(rollout_stats.state_counts)
                            skill_completion_counts = dict(
                                rollout_stats.skill_completion_counts
                            )
                            step_counts = dict(rollout_stats.step_counts)
                            step_completion_counts = dict(
                                rollout_stats.step_completion_counts
                            )
                        elif how_update == "append":
                            state_counts = _merge_count_dicts(
                                s.get(spf + "skill_state_counts"),
                                rollout_stats.state_counts,
                            )
                            skill_completion_counts = _merge_count_dicts(
                                s.get(spf + "skill_completion_counts"),
                                rollout_stats.skill_completion_counts,
                            )
                            step_counts = _merge_count_dicts(
                                s.get(spf + "assembly_step_counts"),
                                rollout_stats.step_counts,
                            )
                            step_completion_counts = _merge_count_dicts(
                                s.get(spf + "assembly_step_completion_counts"),
                                rollout_stats.step_completion_counts,
                            )
                        else:
                            raise ValueError(f"Invalid how_update: {how_update}")

                        progress_summary = _build_progress_summary(
                            state_counts=state_counts,
                            skill_completion_counts=skill_completion_counts,
                            step_counts=step_counts,
                            step_completion_counts=step_completion_counts,
                        )

                        if how_update == "overwrite":
                            s[spf + "success_rate"] = success_rate
                            s[spf + "n_success"] = rollout_stats.n_success
                            s[spf + "n_rollouts"] = rollout_stats.n_rollouts
                            s[spf + "total_return"] = rollout_stats.total_return
                            s[spf + "average_return"] = (
                                rollout_stats.total_return / rollout_stats.n_rollouts
                            )
                            s[spf + "total_reward"] = rollout_stats.total_reward
                            s[spf + "average_reward"] = (
                                rollout_stats.total_reward / rollout_stats.n_rollouts
                            )
                        elif how_update == "append":
                            s[spf + "n_success"] = s.get(spf + "n_success", 0) + rollout_stats.n_success
                            s[spf + "n_rollouts"] = s.get(spf + "n_rollouts", 0) + rollout_stats.n_rollouts
                            s[spf + "success_rate"] = (
                                s[spf + "n_success"] / s[spf + "n_rollouts"]
                            )
                            s[spf + "total_return"] = (
                                s.get(spf + "total_return", 0) + rollout_stats.total_return
                            )
                            s[spf + "average_return"] = (
                                s[spf + "total_return"] / s[spf + "n_rollouts"]
                            )
                            s[spf + "total_reward"] = (
                                s.get(spf + "total_reward", 0) + rollout_stats.total_reward
                            )
                            s[spf + "average_reward"] = (
                                s[spf + "total_reward"] / s[spf + "n_rollouts"]
                            )

                        for key, value in progress_summary.items():
                            s[spf + key] = value

                        run.update()
                    else:
                        print("Not writing to wandb")

                    total_success += rollout_stats.n_success
                    total_rollouts += rollout_stats.n_rollouts
                    summary_total_success += rollout_stats.n_success
                    summary_total_rollouts += rollout_stats.n_rollouts
                    summary_state_counts = _merge_count_dicts(
                        summary_state_counts, rollout_stats.state_counts
                    )
                    summary_skill_completion_counts = _merge_count_dicts(
                        summary_skill_completion_counts,
                        rollout_stats.skill_completion_counts,
                    )
                    summary_step_counts = _merge_count_dicts(
                        summary_step_counts, rollout_stats.step_counts
                    )
                    summary_step_completion_counts = _merge_count_dicts(
                        summary_step_completion_counts,
                        rollout_stats.step_completion_counts,
                    )

                if total_rollouts > 0:
                    print(
                        "Success rate (all tasks): "
                        f"{format_success_rate(total_success, total_rollouts)}"
                    )

                if args.wandb:
                    run.config["currently_evaluating"] = False
                    run.update()

                if args.prioritize_fewest_rollouts:
                    break

            if not args.continuous_mode:
                break

            print(
                f"Sleeping for {args.continuous_interval} seconds before checking for new runs..."
            )
            time.sleep(args.continuous_interval)

        if args.task_summary_out:
            import json

            with open(args.task_summary_out, "w") as f:
                progress_summary = _build_progress_summary(
                    state_counts=summary_state_counts,
                    skill_completion_counts=summary_skill_completion_counts,
                    step_counts=summary_step_counts,
                    step_completion_counts=summary_step_completion_counts,
                )
                json.dump(
                    {
                        "task": primary_task if len(tasks) == 1 else task_group,
                        "n_success": summary_total_success,
                        "n_rollouts": summary_total_rollouts,
                        "success_rate": (
                            summary_total_success / summary_total_rollouts
                            if summary_total_rollouts > 0
                            else None
                        ),
                        **progress_summary,
                    },
                    f,
                )
    finally:
        if args.wandb and run is not None:
            print("Exiting the evaluation loop")
            print("Unsetting the currently_evaluating flag")
            run.config["currently_evaluating"] = False
            run.update()
