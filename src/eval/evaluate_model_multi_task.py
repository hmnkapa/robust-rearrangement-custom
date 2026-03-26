import argparse
import os
import sys
import subprocess
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
from src.behavior.base import model_requires_skill_input
from src.behavior.diffusion import DiffusionPolicy  # noqa
from src.eval.rollout import calculate_success_rate
from src.behavior import get_actor
from src.common.tasks import task2idx, task_timeout
from src.common.files import trajectory_save_dir
from src.gym import get_rl_env
from src.eval.eval_utils import load_model_weights

from typing import Any, List, Optional
from ipdb import set_trace as bp  # noqa
import wandb
from wandb import Api
from wandb.sdk.wandb_run import Run

api = Api()


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


def validate_args(args: argparse.Namespace):
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
    # Always treat as multitask for this entrypoint
    parser.add_argument("--compress-pickles", action="store_true")
    parser.add_argument("--max-rollouts", type=int, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
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
        "--full-length-rollout",
        action="store_true",
        help="Continue each rollout until max rollout steps even after success, and save the full trajectory.",
    )
    parser.add_argument("--record-for-coverage", action="store_true")

    parser.add_argument("--save-rollouts-suffix", type=str, default="")
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

    args.multitask = True
    tasks: List[str] = args.task if isinstance(args.task, list) else [args.task]
    task_group = args.task_group if args.task_group is not None else "+".join(tasks)

    # If multiple tasks are provided, run each task in a fresh process to avoid IsaacGym core dumps.
    is_child = os.environ.get("RR_EVAL_MT_CHILD", "0") == "1"
    if len(tasks) > 1 and not is_child:
        base_argv = []
        skip_next = False
        i = 0
        while i < len(sys.argv[1:]):
            arg = sys.argv[1:][i]
            if skip_next:
                skip_next = False
                i += 1
                continue
            if arg in ["-f", "--task"]:
                # Skip all task values until next flag
                i += 1
                while i < len(sys.argv[1:]) and not sys.argv[1:][i].startswith("-"):
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

        for task in tasks:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
                summary_path = tmp.name

            cmd = [
                sys.executable,
                "-m",
                "src.eval.evaluate_model_multi_task",
                *base_argv,
                "-f",
                task,
                "--task-group",
                task_group,
                "--task-summary-out",
                summary_path,
            ]
            env = os.environ.copy()
            env["RR_EVAL_MT_CHILD"] = "1"
            result = subprocess.run(cmd, env=env)
            if result.returncode != 0:
                print(f"[ERROR] Task {task} failed with return code {result.returncode}")
                sys.exit(result.returncode)

            try:
                import json

                with open(summary_path, "r") as f:
                    summary = json.load(f)
                total_success_all_tasks += summary.get("n_success", 0)
                total_rollouts_all_tasks += summary.get("n_rollouts", 0)
            finally:
                try:
                    os.remove(summary_path)
                except OSError:
                    pass

        if total_rollouts_all_tasks > 0:
            overall_success_rate = total_success_all_tasks / total_rollouts_all_tasks
            print(
                f"Success rate (all tasks): {overall_success_rate:.2%} ({total_success_all_tasks}/{total_rollouts_all_tasks})"
            )
        sys.exit(0)

    # Summary prefix, shortened to spf for brevity downstream
    primary_task = tasks[0]
    primary_spf = f"{primary_task}/" + "" if args.multitask else ""

    # Get the environment(s)
    # TODO: This needs to be changed to enable recreation the env for each run
    print(
        f"Creating the environment(s) with action_type {args.action_type} (this needs to be changed to enable recreation the env for each run)"
    )
    env: Optional[Env] = None
    env_task: Optional[str] = None

    # Start the evaluation loop
    print(f"Starting evaluation loop in continuous mode: {args.continuous_mode}")
    try:
        while True:
            # Get the run(s) to test, using main's device for map_location
            runs = get_runs(args, map_location=device)

            # For now, filter out only the runs with strictly positive success rates to add more runs to them to get a better estimate
            if args.eval_top_k is not None:
                # Get the top k runs
                runs = sorted(
                    runs,
                    key=lambda run: run.summary.get(primary_spf + "success_rate", 0),
                    reverse=True,
                )[: args.eval_top_k]

            # Also, evaluate the ones with the fewest rollouts first (if they have any)
            runs = sorted(
                runs,
                key=lambda run: run.summary.get(primary_spf + "n_rollouts", 0),
            )

            print(f"Found {len(runs)} runs to evaluate:")
            for run in runs:
                print(
                    f"    Run: {run.name}: {run.summary.get(primary_spf + 'n_rollouts', 0)}, {run.summary.get(primary_spf + 'success_rate', None)}"
                )
            for run in runs:
                # First, we must flush the api and request the run again in case the information is stale
                if not args.wt_path:
                    api.flush()
                    run = api.run("/".join([run.project, run.id]))

                # Check if the run is currently being evaluated
                if (
                    run.config.get("currently_evaluating", False)
                    and not args.ignore_currently_evaluating_flag
                ):
                    print(f"Run: {run.name} is currently being evaluated, skipping")
                    continue

                # If in overwrite set the currently_evaluating flag to true runs can cooperate better in skip mode
                if args.wandb:
                    print(
                        f"Setting currently_evaluating flag to true for run: {run.name}"
                    )
                    run.config["currently_evaluating"] = True
                    run.update()

                # Get the current `test_epoch_loss` from the run
                test_epoch_loss = run.summary.get("test_epoch_loss", None)
                print(
                    f"Evaluating run: {run.name} at test_epoch_loss: {test_epoch_loss}"
                )

                cfg = OmegaConf.create(run.config)

                # Check that we didn't set the wrong action type and pose representation
                assert cfg.control.control_mode == args.action_type

                print(OmegaConf.to_yaml(cfg))

                # Temporary fix for residual missing field
                if "base_policy" in cfg:
                    print("Applying residual field hotfix")
                    cfg.action_dim = cfg.base_policy.action_dim

                # Temporary fix for dagger missing field
                if "student_policy" in cfg:
                    print("Applying dagger field hotfix")
                    cfg.action_dim = cfg.student_policy.action_dim

                # Temporary fix for critic missing field in actor config
                if "critic" in cfg:
                    print("Applying critic field hotfix")
                    cfg.actor.critic = cfg.critic
                    cfg.actor.init_logstd = cfg.init_logstd
                    cfg.discount = cfg.base_policy.discount

                # Make the actor
                actor: Actor = get_actor(cfg=cfg, device=device)

                # Set the inference steps of the actor
                if isinstance(actor, DiffusionPolicy):
                    actor.inference_steps = 4

                if args.wt_path:
                    if cfg.actor.name != "dp3":
                        actor.load_state_dict(run.checkpoint["model_state_dict"])
                    actor.eval()
                    actor.to(device)

                else:
                    actor: Optional[Actor] = load_model_weights(
                        run=run, actor=actor, wt_type=args.wt_type
                    )

                if actor is None:
                    print(
                        f"Skipping run: {run.name} as no weights for wt_type: {args.wt_type} was not found"
                    )
                    continue

                actor_name = cfg.actor_name if "actor_name" in cfg else cfg.actor.name

                total_success = 0
                total_rollouts = 0

                task_total_success = 0
                task_total_rollouts = 0

                for task in tasks:
                    spf = f"{task}/" + "" if args.multitask else ""

                    # Set the timeout
                    rollout_max_steps = (
                        task_timeout(task, n_parts=args.n_parts_assemble)
                        if args.max_rollout_steps is None
                        else args.max_rollout_steps
                    )

                    # Check if the number of rollouts this run has is greater than the max_rollouts
                    if args.max_rollouts is not None:
                        if run.summary.get(spf + "n_rollouts", 0) >= args.max_rollouts:
                            print(
                                f"Run: {run.name} task {task} has already been evaluated {run.summary.get(spf + 'n_rollouts', 0)} times, skipping"
                            )
                            continue

                    # Check if the run has already been evaluated for this task
                    how_update = "overwrite"
                    if run.summary.get(spf + "success_rate", None) is not None:
                        if args.if_exists == "skip":
                            print(f"Run: {run.name} task {task} has already been evaluated, skipping")
                            continue
                        elif args.if_exists == "error":
                            raise ValueError(f"Run: {run.name} task {task} has already been evaluated")
                        elif args.if_exists == "overwrite":
                            print(
                                f"Run: {run.name} task {task} has already been evaluated, overwriting"
                            )
                            how_update = "overwrite"
                        elif args.if_exists == "append":
                            print(f"Run: {run.name} task {task} has already been evaluated, appending")
                            how_update = "append"

                    suffix = args.save_rollouts_suffix
                    if actor_name == "dp3":
                        suffix = "dp3"

                    if args.save_pc_for_dp3:
                        suffix = f"pc/{args.pc_points}/{args.pc_downsample_mode}"

                    if args.save_depth_image:
                        suffix = f"rgbd"

                    if task_group:
                        suffix = f"{suffix}/{task_group}" if suffix else task_group
                    
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

                    if args.store_video_wandb:
                        # For the run table with videos to be saved to wandb,
                        # a run needs to be active, so we initialie run here
                        wandb.init(
                            project=run.project,
                            entity=run.entity,
                            id=run.id,
                            resume="allow",
                        )

                    # Only actually load the environment after we know we've got at least one run to evaluate
                    if env is None or env_task != task:
                        if env is not None and env_task != task:
                            close_fn = getattr(env, "close", None)
                            if callable(close_fn):
                                close_fn()
                            env = None
                            env_task = None

                        # Prepare obs_keys with depth_image2 if needed for point cloud generation
                        env_obs_keys = None
                        if args.save_pc_for_dp3 or args.save_depth_image or actor_name == "dp3":
                            # Need to include depth_image2 for point cloud generation
                            from src.gym import FULL_OBS
                            env_obs_keys = list(FULL_OBS)
                            if args.observation_space == "state":
                                # Filter out color images but keep depth
                                env_obs_keys = [key for key in env_obs_keys if "color_image" not in key]
                            # Ensure depth_image is included
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
                            headless=not args.visualize,
                            obs_keys=env_obs_keys,  # Pass prepared obs_keys
                        )
                        env_task = task

                    # 点云 obs 相关
                    pc_generator = None
                    if args.save_pc_for_dp3 or actor_name == "dp3":
                        # depth_image2 should already be in obs_keys from env creation
                        # Handle list vs scalar for bbox
                        bbox_ext = args.pc_bbox_half_extent
                        if isinstance(bbox_ext, list) and len(bbox_ext) == 1:
                            bbox_ext = bbox_ext[0]
                        pc_generator = PointCloudGenerator(
                            env=env, 
                            camera_name="front", 
                            max_points=args.pc_points, 
                            bbox_half_extent=bbox_ext,
                            bbox_crop_mode=args.pc_bbox_crop_mode
                        )

                    # Perform the rollouts
                    print(f"Starting rollout of run: {run.name} task: {task}")
                    actor.set_task(task2idx[task])
                    requires_skill_input = model_requires_skill_input(cfg)
                    rollout_stats = calculate_success_rate(
                        actor=actor,
                        env=env,
                        n_rollouts=args.n_rollouts,
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
                        full_length_rollout=args.full_length_rollout,
                        record_first_state_only=args.record_for_coverage,
                        pc_generator=pc_generator,
                        provide_skill_input=requires_skill_input,
                    )

                    if args.store_video_wandb:
                        # Close the run to save the videos
                        wandb.finish()

                    success_rate = rollout_stats.success_rate

                    print(
                        f"Success rate ({task}): {success_rate:.2%} ({rollout_stats.n_success}/{rollout_stats.n_rollouts})"
                    )

                    if args.wandb:
                        print("Writing to wandb...")

                        s: dict = run.summary

                        # Set the summary fields
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
                        else:
                            raise ValueError(f"Invalid how_update: {how_update}")

                        # Set the currently_evaluating flag to false
                        run.config["currently_evaluating"] = False

                        # Update the run to save the summary fields
                        run.update()

                    else:
                        print("Not writing to wandb")

                    total_success += rollout_stats.n_success
                    total_rollouts += rollout_stats.n_rollouts
                    task_total_success += rollout_stats.n_success
                    task_total_rollouts += rollout_stats.n_rollouts

                if total_rollouts > 0:
                    overall_success_rate = total_success / total_rollouts
                    print(
                        f"Success rate (all tasks): {overall_success_rate:.2%} ({total_success}/{total_rollouts})"
                    )

                if args.task_summary_out and task_total_rollouts > 0:
                    import json

                    with open(args.task_summary_out, "w") as f:
                        json.dump(
                            {
                                "task": tasks[0] if tasks else None,
                                "n_success": task_total_success,
                                "n_rollouts": task_total_rollouts,
                            },
                            f,
                        )

                # If we prioritize the runs with the fewest rollouts, break after the first run
                # so that we can sort the runs according to the number of rollouts and evaluate them again
                if args.prioritize_fewest_rollouts:
                    break

            # If not in continuous mode, break
            if not args.continuous_mode:
                break

            # Sleep for the interval
            print(
                f"Sleeping for {args.continuous_interval} seconds before checking for new runs..."
            )
            time.sleep(args.continuous_interval)
    finally:
        # Unset the "currently_evaluating" flag
        if args.wandb:
            print("Exiting the evaluation loop")
            print("Unsetting the currently_evaluating flag")
            run.config["currently_evaluating"] = False
            run.update()
