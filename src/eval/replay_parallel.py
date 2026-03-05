import argparse
import lzma
import pickle
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.common.context import suppress_all_output
from src.gym import get_rl_env
from src.pc_util.point_cloud_generator import PointCloudGenerator

TASK_CHOICES = [
    "one_leg",
    "lamp",
    "round_table",
    "desk",
    "square_table",
    "cabinet",
    "mug_rack",
    "factory_peg_hole",
    "bimanual_insertion",
]


def _load_pickle(path: Path) -> Dict[str, Any]:
    if path.suffix == ".xz" or path.name.endswith(".pkl.xz"):
        with lzma.open(path, "rb") as f:
            return pickle.load(f)
    with open(path, "rb") as f:
        return pickle.load(f)


def _to_numpy_single(x, env_idx: int):
    if hasattr(x, "detach"):
        arr = x.detach()
        if arr.ndim > 0 and arr.shape[0] > env_idx:
            arr = arr[env_idx]
        return arr.cpu().numpy()

    arr = np.asarray(x)
    if arr.ndim > 0 and arr.shape[0] > env_idx:
        return arr[env_idx]
    return arr


def _extract_obs_single(step_obs: Dict[str, Any], env_idx: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    robot_state = step_obs.get("robot_state")
    if isinstance(robot_state, dict):
        out["robot_state"] = {
            k: _to_numpy_single(v, env_idx) for k, v in robot_state.items()
        }
    elif robot_state is not None:
        out["robot_state"] = _to_numpy_single(robot_state, env_idx)

    for key in ["color_image1", "color_image2", "depth_image1", "depth_image2", "parts_poses"]:
        if key in step_obs:
            out[key] = _to_numpy_single(step_obs[key], env_idx)

    return out


def _build_pos_action_from_observation(obs_item: Dict[str, Any]) -> np.ndarray:
    rs = obs_item.get("robot_state")
    if not isinstance(rs, dict):
        raise RuntimeError("Expected robot_state dict when action-src=pos.")

    ee_pos = np.asarray(rs["ee_pos"], dtype=np.float32).reshape(-1)
    ee_quat = np.asarray(rs["ee_quat"], dtype=np.float32).reshape(-1)
    gw = rs.get("gripper_width", 0.0)
    gw = float(np.asarray(gw).reshape(-1)[0])

    # Normalize gripper width [0, 0.065] -> [-1, 1]
    grip = -1.0 * (2.0 * (gw / 0.065) - 1.0)

    return np.concatenate([ee_pos, ee_quat, np.array([grip], dtype=np.float32)], axis=0)


def _make_action_tensor(actions: List[np.ndarray], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.stack(actions, axis=0), dtype=torch.float32, device=device)


def _run_batch(
    env,
    device: torch.device,
    batch_items: List[Tuple[Path, Dict[str, Any], int]],
    action_src: str,
    save_pc_for_dp3: bool,
    pc_points: int,
    pc_downsample_mode: str,
    pc_generator: Optional[PointCloudGenerator],
) -> List[Dict[str, Any]]:
    """
    Run one batch in parallel (len=batch_items envs).

    Returns one result dict per input item:
      {
        "path": Path,
        "attempt": int,
        "success": bool,
        "trajectory": Optional[Dict],
        "dp3": Optional[Dict],
        "error": Optional[str],
      }
    """
    n_envs = len(batch_items)

    contexts: List[Dict[str, Any]] = []
    reset_states: List[Dict[str, Any]] = []

    for path, data, attempt in batch_items:
        observations = data.get("observations", [])
        actions = data.get("actions", [])

        if len(observations) == 0:
            contexts.append(
                {
                    "path": path,
                    "attempt": attempt,
                    "error": "empty observations",
                    "active": False,
                    "success": False,
                }
            )
            reset_states.append({})
            continue

        start_idx = 5 if len(observations) > 5 else 0
        init_state = observations[start_idx]

        contexts.append(
            {
                "path": path,
                "attempt": attempt,
                "source_observations": observations,
                "source_actions": actions,
                "t": start_idx,
                "active": True,
                "success": False,
                "error": None,
                "new_observations": [],
                "new_actions": [],
                "new_rewards": [],
                "pc_list": [],
                "img_list": [],
                "state_list": [],
                "action_list": [],
            }
        )
        reset_states.append(init_state)

    reset_obs = env.reset_to(reset_states)

    # Keep the fresh reset observation so saved trajectories match rollout shape:
    # len(observations) = len(actions) + 1.
    if isinstance(reset_obs, dict):
        for env_idx, ctx in enumerate(contexts):
            if ctx.get("active", False):
                ctx["new_observations"].append(_extract_obs_single(reset_obs, env_idx))

    # Keep stepping while any env still has pending trajectory to replay.
    while any(ctx.get("active", False) for ctx in contexts):
        step_actions: List[np.ndarray] = []

        for ctx in contexts:
            if not ctx.get("active", False):
                step_actions.append(np.zeros(8, dtype=np.float32))
                continue

            try:
                t = ctx["t"]
                observations = ctx["source_observations"]
                actions = ctx["source_actions"]

                if action_src == "action":
                    if t >= len(actions):
                        ctx["active"] = False
                        step_actions.append(np.zeros(8, dtype=np.float32))
                        continue
                    action = np.asarray(actions[t], dtype=np.float32).reshape(-1)
                else:
                    if t >= len(observations):
                        ctx["active"] = False
                        step_actions.append(np.zeros(8, dtype=np.float32))
                        continue
                    action = _build_pos_action_from_observation(observations[t])

                if action.shape[0] != 8:
                    raise RuntimeError(f"Invalid action dim={action.shape[0]} at step {t}")

                step_actions.append(action)
            except Exception as e:  # noqa: BLE001
                ctx["active"] = False
                ctx["error"] = f"action_build_failed: {e}"
                step_actions.append(np.zeros(8, dtype=np.float32))

        action_tensor = _make_action_tensor(step_actions, device=device)
        step_obs, reward, done, _info = env.step(action_tensor)

        for env_idx, ctx in enumerate(contexts):
            if not ctx.get("active", False):
                continue

            try:
                obs_item = _extract_obs_single(step_obs, env_idx)
                rew_item = float(_to_numpy_single(reward, env_idx))
                done_item = bool(_to_numpy_single(done, env_idx))

                ctx["new_observations"].append(obs_item)
                ctx["new_actions"].append(step_actions[env_idx].astype(np.float32))
                ctx["new_rewards"].append(rew_item)

                if save_pc_for_dp3 and pc_generator is not None:
                    pc = pc_generator.generate_transformed_cropped_point_cloud(
                        env_idx=env_idx,
                        max_points=pc_points,
                        downsample_mode=pc_downsample_mode,
                        visualize=False,
                    )
                    ctx["pc_list"].append(pc.detach().cpu().numpy())
                    ctx["img_list"].append(obs_item.get("color_image2"))
                    ctx["state_list"].append(obs_item.get("robot_state"))
                    ctx["action_list"].append(step_actions[env_idx].astype(np.float32))

                if done_item:
                    ctx["success"] = True
                    ctx["active"] = False

                ctx["t"] += 1
                if ctx["t"] >= len(ctx["source_observations"]) and not ctx["success"]:
                    ctx["active"] = False
            except Exception as e:  # noqa: BLE001
                ctx["active"] = False
                ctx["error"] = f"step_collect_failed: {e}"

    results: List[Dict[str, Any]] = []
    for ctx in contexts:
        result: Dict[str, Any] = {
            "path": ctx["path"],
            "attempt": ctx["attempt"],
            "success": bool(ctx.get("success", False)),
            "trajectory": None,
            "dp3": None,
            "error": ctx.get("error"),
        }

        if result["success"]:
            result["trajectory"] = {
                "observations": ctx["new_observations"],
                "actions": [a.tolist() for a in ctx["new_actions"]],
                "rewards": ctx["new_rewards"],
                "success": True,
                "task": ctx["source_observations"][0].get("task", None),
                "action_type": "pos" if action_src == "pos" else "delta",
            }
            if result["trajectory"]["task"] is None:
                # Fall back to folder-level task name in caller when saving.
                result["trajectory"].pop("task")

            if save_pc_for_dp3:
                result["dp3"] = {
                    "state": ctx["state_list"],
                    "action": ctx["action_list"],
                    "point_cloud": ctx["pc_list"],
                    "img": ctx["img_list"],
                }

        results.append(result)

    return results


def _save_success_result(
    result: Dict[str, Any],
    task: str,
    output_pickle_folder: Path,
    save_pc_for_dp3: bool,
    pc_points: int,
    pc_downsample_mode: str,
) -> Path:
    output_pickle_folder.mkdir(parents=True, exist_ok=True)

    src_path: Path = result["path"]
    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out_name = f"{src_path.stem}_replay_{ts}.pkl"
    out_path = output_pickle_folder / out_name

    traj = result["trajectory"]
    if "task" not in traj:
        traj["task"] = task

    with open(out_path, "wb") as f:
        pickle.dump(traj, f)

    if save_pc_for_dp3 and result.get("dp3") is not None:
        pc_name = f"{src_path.stem}_pc_{pc_points}_{pc_downsample_mode}_{ts}.pkl"
        pc_path = output_pickle_folder / pc_name
        with open(pc_path, "wb") as f:
            pickle.dump(result["dp3"], f)

    return out_path


def _discover_pickles(input_folder: Path) -> List[Path]:
    pickles = sorted(input_folder.rglob("*.pkl"))
    pickles += sorted(input_folder.rglob("*.pkl.xz"))
    # Remove duplicates when .pkl search catches .pkl.xz names unexpectedly.
    unique = sorted(set(pickles))
    return unique


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Parallel replay over a folder of trajectory pickles."
    )
    parser.add_argument(
        "--input-pickle-folder",
        type=str,
        required=True,
        help="Folder containing input trajectory pickles (.pkl/.pkl.xz).",
    )
    parser.add_argument(
        "--output-pickle-folder",
        type=str,
        required=True,
        help="Folder to save successful replay-generated pickles.",
    )
    parser.add_argument("--task", "-f", type=str, required=True, choices=TASK_CHOICES)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--max-success-replay", type=int, default=50)
    parser.add_argument("--max-retry", type=int, default=3)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--randomness", type=str, default="low")
    parser.add_argument(
        "--action-src",
        type=str,
        default="pos",
        choices=["pos", "action"],
        help="'pos': build action from stored robot_state; 'action': use stored action.",
    )
    parser.add_argument("--max-env-steps", type=int, default=5000)
    parser.add_argument(
        "--act-rot-repr", type=str, default="quat", choices=["quat", "rot_6d", "axis"]
    )

    parser.add_argument("--save-pc-for-dp3", action="store_true")
    parser.add_argument("--pc-points", type=int, default=4096)
    parser.add_argument("--pc-bbox-half-extent", type=float, default=0.2)
    parser.add_argument(
        "--pc-downsample-mode",
        type=str,
        default="random",
        choices=["random", "uniform", "fps"],
    )

    args = parser.parse_args(argv)

    input_folder = Path(args.input_pickle_folder).expanduser().resolve()
    output_folder = Path(args.output_pickle_folder).expanduser().resolve()

    if not input_folder.exists():
        raise ValueError(f"Input folder does not exist: {input_folder}")

    candidate_pickles = _discover_pickles(input_folder)
    if not candidate_pickles:
        print(f"No pickle files found in {input_folder}")
        return

    print(f"Found {len(candidate_pickles)} pickles in {input_folder}")
    print(
        f"Target success count={args.max_success_replay}, "
        f"parallel envs={args.num_envs}, max_retry={args.max_retry}"
    )

    queue: Deque[Tuple[Path, int]] = deque((p, 0) for p in candidate_pickles)
    success_count = 0
    failed_count = 0

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    while queue and success_count < args.max_success_replay:
        batch_meta: List[Tuple[Path, int]] = []
        while queue and len(batch_meta) < args.num_envs:
            batch_meta.append(queue.popleft())

        batch_items: List[Tuple[Path, Dict[str, Any], int]] = []
        for path, attempt in batch_meta:
            try:
                data = _load_pickle(path)
                batch_items.append((path, data, attempt))
            except Exception as e:  # noqa: BLE001
                print(f"[LOAD_FAIL] {path}: {e}")
                if attempt + 1 < args.max_retry:
                    queue.append((path, attempt + 1))
                else:
                    failed_count += 1

        if not batch_items:
            continue

        n_envs = len(batch_items)
        with suppress_all_output(False):
            env = get_rl_env(
                gpu_id=args.gpu,
                task=args.task,
                num_envs=n_envs,
                randomness=args.randomness,
                max_env_steps=args.max_env_steps,
                observation_space="image",
                act_rot_repr=args.act_rot_repr,
                action_type=("pos" if args.action_src == "pos" else "delta"),
                april_tags=False,
                verbose=False,
                headless=not args.visualize and args.headless,
            )

        pc_generator = None
        if args.save_pc_for_dp3:
            extra_obs_keys = []
            if "depth_image2" not in env.obs_keys:
                extra_obs_keys.append("depth_image2")
            if extra_obs_keys:
                env.obs_keys = list(env.obs_keys) + extra_obs_keys
                env.set_camera()
            pc_generator = PointCloudGenerator(
                env=env,
                camera_name="front",
                max_points=args.pc_points,
                bbox_half_extent=args.pc_bbox_half_extent,
            )

        results = _run_batch(
            env=env,
            device=device,
            batch_items=batch_items,
            action_src=args.action_src,
            save_pc_for_dp3=args.save_pc_for_dp3,
            pc_points=args.pc_points,
            pc_downsample_mode=args.pc_downsample_mode,
            pc_generator=pc_generator,
        )

        if hasattr(env, "close"):
            env.close()

        for result in results:
            src_path = result["path"]
            attempt = int(result["attempt"])
            if result["success"]:
                saved = _save_success_result(
                    result=result,
                    task=args.task,
                    output_pickle_folder=output_folder,
                    save_pc_for_dp3=args.save_pc_for_dp3,
                    pc_points=args.pc_points,
                    pc_downsample_mode=args.pc_downsample_mode,
                )
                success_count += 1
                print(f"[SUCCESS {success_count}/{args.max_success_replay}] {src_path} -> {saved}")
                if success_count >= args.max_success_replay:
                    break
            else:
                err = result.get("error")
                if attempt + 1 < args.max_retry:
                    queue.append((src_path, attempt + 1))
                    print(
                        f"[RETRY {attempt + 1}/{args.max_retry - 1}] {src_path} "
                        f"reason={err}"
                    )
                else:
                    failed_count += 1
                    print(f"[FAILED] {src_path} reason={err}")

    print(
        "Replay batch finished: "
        f"success={success_count}, failed={failed_count}, remaining={len(queue)}"
    )


if __name__ == "__main__":
    main()
