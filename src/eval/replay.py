import argparse
import pickle
from pathlib import Path
from typing import List
from datetime import datetime

import numpy as np
import torch

from src.gym import get_rl_env
from src.common.context import suppress_all_output
from src.visualization.render_mp4 import create_in_memory_mp4
from src.pc_util.point_cloud_generator import PointCloudGenerator


def _to_numpy_first(x, env_idx: int = 0):
    """Extract env_idx slice and convert to numpy, handling torch tensors and numpy arrays."""
    if hasattr(x, "cpu"):
        arr = x
        if arr.ndim > 1:
            arr = arr[env_idx]
        return arr.cpu().numpy()
    x_np = np.asarray(x)
    if x_np.ndim > 1:
        x_np = x_np[env_idx]
    return x_np


def _extract_robot_state(obs: dict, env_idx: int = 0):
    rs = obs.get("robot_state")
    if rs is None:
        return None
    if isinstance(rs, dict):
        return {k: _to_numpy_first(v, env_idx) for k, v in rs.items()}
    return _to_numpy_first(rs, env_idx)


def _extract_image(obs: dict, key: str, env_idx: int = 0):
    img = obs.get(key)
    if img is None:
        return None
    return _to_numpy_first(img, env_idx)


def _done_flag(done, env_idx: int = 0) -> bool:
    if done is None:
        return False
    if hasattr(done, "cpu"):
        arr = done
        if arr.ndim > 1:
            arr = arr[env_idx]
        return bool(arr.cpu().numpy().item())
    arr = np.asarray(done)
    if arr.ndim > 1:
        arr = arr[env_idx]
    return bool(arr.item())


def _to_tensor_action(action: np.ndarray, num_envs: int, device: torch.device) -> torch.Tensor:
    """Ensure action is (num_envs, action_dim) torch.float32 on device."""
    ac = torch.as_tensor(action, dtype=torch.float32, device=device)
    if ac.ndim == 1:
        ac = ac.unsqueeze(0)
    if ac.shape[0] != num_envs:
        ac = ac.repeat(num_envs, 1)
    return ac


def get_world_ee_pose(env):
    """Return (ee_pos_world, ee_quat_world) as numpy arrays using env.rb_states."""
    # rb_states: (num_rigid_bodies, 13) in SIM/global space
    hand_pos = env.rb_states[env.ee_idxs, :3]
    hand_quat = env.rb_states[env.ee_idxs, 3:7]
    # world pose already in rb_states; no need to subtract base
    return hand_pos.cpu().numpy(), hand_quat.cpu().numpy()


def add_thick_lines(env, env_idx: int, segments: np.ndarray, color: np.ndarray, width: float = 0.003, copies: int = 2):
    """Draw thicker-looking lines by duplicating each segment with small perpendicular offsets.
    segments: (N, 6) array of [x1,y1,z1,x2,y2,z2]
    color: (1,3) RGB color
    width: base offset magnitude in meters
    copies: number of offset copies on each side
    """
    # For each segment, compute direction and a perp vector to offset.
    all_lines = []
    for seg in segments:
        p1 = seg[:3]
        p2 = seg[3:]
        dir_v = p2 - p1
        norm = np.linalg.norm(dir_v)
        if norm < 1e-9:
            continue
        dir_v = dir_v / norm
        # Find a perpendicular vector (not colinear)
        perp = np.cross(dir_v, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(dir_v, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        perp = perp / (np.linalg.norm(perp) + 1e-8)
        # Build offsets: center + symmetric copies
        offsets = [0.0] + [width * k for k in range(1, copies + 1)] + [-width * k for k in range(1, copies + 1)]
        for off in offsets:
            o = perp * off
            all_lines.append(np.concatenate([p1 + o, p2 + o]))
    if not all_lines:
        return
    lines_arr = np.asarray(all_lines, dtype=np.float32)
    # Ensure color array matches number of lines; tile if a single RGB is provided
    if color.ndim == 2 and color.shape[0] == 1:
        color_arr = np.repeat(color.astype(np.float32), repeats=lines_arr.shape[0], axis=0)
    elif color.ndim == 1 and color.shape[0] == 3:
        color_arr = np.repeat(color.reshape(1, 3).astype(np.float32), repeats=lines_arr.shape[0], axis=0)
    else:
        # Assume already per-line colors
        color_arr = color.astype(np.float32)
    env.isaac_gym.add_lines(env.viewer, env.envs[env_idx], len(lines_arr), lines_arr, color_arr)


def render_eepose_arrow(step_obs, env, length=0.05):
    if not (hasattr(env, "viewer") and env.viewer is not None):
        raise RuntimeError("Env viewer not available for rendering EE pose arrow.")
    import numpy as np
    from furniture_bench.controllers.control_utils import quat2mat
    ee_pos_w, ee_quat_w = get_world_ee_pose(env)
    # print("[DEBUG] ee_pose: ", ee_pos_w[0])
    # base_pos_w = env.rb_states[env.franka_base_index, :3].cpu().numpy()
    # print("[DEBUG] base_pose:", base_pos_w)
    colors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    for env_idx in range(env.num_envs):
        ee_pos = ee_pos_w[env_idx]
        ee_quat = ee_quat_w[env_idx]
        dir_z = quat2mat(torch.as_tensor(ee_quat)).cpu().numpy()[:, 2]
        dir_z = dir_z / (np.linalg.norm(dir_z) + 1e-8)
        base = np.asarray(ee_pos, dtype=np.float32)
        tip = base + dir_z * length
        perp = np.cross(dir_z, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(dir_z, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        perp = perp / (np.linalg.norm(perp) + 1e-8)
        left = tip - dir_z * 0.02 + perp * 0.01
        right = tip - dir_z * 0.02 - perp * 0.01
        shaft = np.array([np.concatenate([base, tip])], dtype=np.float32)
        head_l = np.array([np.concatenate([tip, left])], dtype=np.float32)
        head_r = np.array([np.concatenate([tip, right])], dtype=np.float32)
        add_thick_lines(env, env_idx, shaft, colors, width=5e-4, copies=3)
        add_thick_lines(env, env_idx, head_l, colors, width=5e-4, copies=3)
        add_thick_lines(env, env_idx, head_r, colors, width=5e-4, copies=3)


def render_world_axes(env, img: np.ndarray, origin=np.array([-0.1, -0.1, 0.4], dtype=np.float32), length=1.0) -> np.ndarray:
    """Draw XYZ axes projected onto the front camera image.
    img: (H,W,3) uint8 array (RGB)
    P: projection matrix (4x4 or batched/tensor)
    V: view matrix (4x4 or batched/tensor)
    Returns a new image with axes drawn.
    """
    H, W = img.shape[:2]

    # Normalize P, V to numpy 4x4
    def _to_4x4_np(mat):
        if hasattr(mat, "detach"):
            mat = mat.detach().cpu().numpy()
        else:
            mat = np.asarray(mat)
        # If batched, take the first
        if mat.ndim == 3:
            mat = mat[0]
        # If flattened length 16, reshape
        if mat.ndim == 1 and mat.size == 16:
            mat = mat.reshape(4, 4)
        if mat.shape != (4, 4):
            raise ValueError(f"Expected 4x4 matrix, got shape {mat.shape}")
        return mat.astype(np.float32)

    P, V = env.get_front_projection_view_matrix()
    P_np = _to_4x4_np(P)
    V_np = _to_4x4_np(V)

    def project_point(pt_world):
        pw = np.array([pt_world[0], pt_world[1], pt_world[2], 1.0], dtype=np.float32).reshape(4, 1)
        clip = P_np @ (V_np @ pw)
        if clip[3] == 0:  # 这里 clip[3] 是负数
            return None
        ndc = clip[:3] / clip[3]
        x = int((1.0 - (ndc[1] * 0.5 + 0.5)) * W)
        y = int((ndc[0] * 0.5 + 0.5) * H)
        return (x, y)

    axes = {
        "x": (np.array([1.0, 0.0, 0.0], dtype=np.float32), (255, 0, 0)),
        "y": (np.array([0.0, 1.0, 0.0], dtype=np.float32), (0, 255, 0)),
        "z": (np.array([0.0, 0.0, 1.0], dtype=np.float32), (0, 0, 255)),
    }

    canvas = img.copy()
    import cv2
    for key, (dir_vec, color_rgb) in axes.items():
        d = dir_vec / (np.linalg.norm(dir_vec) + 1e-8)
        p0 = project_point(origin)
        p1 = project_point(origin + d * length)
        if p0 is None or p1 is None:
            continue
        cv2.arrowedLine(canvas, p0, p1, color_rgb, thickness=3, tipLength=0.2)
    return canvas


def main(argv=None):
    parser = argparse.ArgumentParser(description="Replay a saved pickle trajectory with pos control")
    parser.add_argument("--pickle-path", type=str, required=False)
    parser.add_argument("--task", "-f", type=str, required=True,
                        choices=["one_leg","lamp","round_table","desk","square_table","cabinet","mug_rack","factory_peg_hole","bimanual_insertion"]) 
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--randomness", type=str, default="low")
    parser.add_argument("--action-src", type=str, default="pos", choices=["pos", "action"], help="Source of action for replay: 'pos' (compute from observations) or 'action' (use recorded actions)")
    parser.add_argument("--max-env-steps", type=int, default=5000)
    parser.add_argument("--act-rot-repr", type=str, default="quat", choices=["quat","rot_6d","axis"])  # pos control uses quat here
    parser.add_argument("--record", action="store_true", help="Save replay videos to local sim_replay folder")
    parser.add_argument("--visualize-eepose", action="store_true", help="Draw EE pose arrow after each step")
    parser.add_argument("--visualize-axis", action="store_true", help="Draw world coordinate axes at origin")
    parser.add_argument("--debug-action", action="store_true", help="Run debug action: move EE +X,+Y,+Z (10 steps each) with overlays")
    parser.add_argument("--save-pc-for-dp3", action="store_true", help="Enable point cloud generation and pickle export for DP3")
    parser.add_argument("--reset-eepos-steps", type=int, default=30, help="If joint_positions near zero, apply absolute ee_pos/ee_quat control for N steps after reset")
    parser.add_argument("--reset-eepos-eps", type=float, default=1e-4, help="Epsilon to detect zero joint_positions for reset-to-ee-pos fallback")
    parser.add_argument("--init-state-frame", type=int, default=5, help="Observation index used as init_state for reset_to")
    parser.add_argument("--pc-points", type=int, default=4096, help="Downsampled point count for generated point clouds")
    parser.add_argument("--pc-bbox-half-extent", type=float, default=0.2, help="Half-extent of the cubic bbox for point cloud cropping (in meters)")
    parser.add_argument(
        "--pc-downsample-mode",
        type=str,
        default="random",
        choices=["random", "uniform", "fps"],
        help="Downsample mode for point cloud generation",
    )
    parser.add_argument(
        "--pc-out-dir",
        type=str,
        default="/data/hy/robust-rearrangement/raw/raw/diffik/sim/one_leg/rollout/low/pc/success",
        help="Output dir for replay pickle with point clouds when task succeeds",
    )
    args = parser.parse_args(argv)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load pickle
    pkl_path = Path(args.pickle_path)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Create RL env with pos control (diffik ctrl by default)
    if args.action_src == "action":
        action_type = "delta"
        # action_type = data["action_type"]
    else:
        action_type = "pos"
    with suppress_all_output(False):
        env = get_rl_env(
            gpu_id=args.gpu,
            task=args.task,
            num_envs=args.num_envs,
            randomness=args.randomness,
            max_env_steps=args.max_env_steps,
            observation_space="image",  # will be ignored for reset_to if state dict provided
            act_rot_repr=args.act_rot_repr,
            action_type=action_type,
            april_tags=False,
            verbose=False,
            headless=not args.visualize and args.headless,
        )

    # Ensure depth/seg tensors exist for point cloud generation when enabled
    pc_generator = None
    if args.save_pc_for_dp3:
        extra_obs_keys = []
        if "depth_image2" not in env.obs_keys:
            extra_obs_keys.append("depth_image2")
        if extra_obs_keys:
            env.obs_keys = list(env.obs_keys) + extra_obs_keys
            env.set_camera()
        pc_generator = PointCloudGenerator(env=env, camera_name="front", max_points=args.pc_points, bbox_half_extent=args.pc_bbox_half_extent)

    # Debug action mode: route to run_debug_action
    if args.debug_action:
        run_debug_action(env=env, device=device, args=args)
        return

    observations: List[dict] = data["observations"]
    actions: List[List[float]] = data["actions"]  # kept for length reference only

    # Reset env to initial observation (single env assumed)
    # FurnitureSim.reset_to expects a list of per-env states
    init_state_idx = max(0, min(args.init_state_frame, len(observations) - 1))
    init_state = observations[init_state_idx]
    def _print_value(name, v):
        try:
            arr = np.asarray(v)
            if arr.ndim == 0:
                print(f"    {name}: {arr.item()}")
                return
            arr = np.round(arr.astype(float), 2)
            print(f"    {name}: {arr.tolist()}")
        except Exception:
            print(f"    {name}: {v}")

    print("[DEBUG] reset_to init_state contents:")
    print(f"  keys: {sorted(init_state.keys())}")
    if "robot_state" in init_state:
        rs = init_state["robot_state"]
        if isinstance(rs, dict):
            print(f"  robot_state keys: {sorted(rs.keys())}")
            for k in sorted(rs.keys()):
                _print_value(f"robot_state/{k}", rs[k])
        else:
            _print_value("robot_state", rs)
    if "parts_poses" in init_state:
        _print_value("parts_poses", init_state["parts_poses"])
    env.reset_to([init_state])

    def _maybe_recover_eepos_after_reset(state):
        rs = state.get("robot_state") if isinstance(state, dict) else None
        if not isinstance(rs, dict):
            return
        jp = rs.get("joint_positions", None)
        if jp is None:
            return
        jp_arr = np.asarray(jp, dtype=np.float32).reshape(-1)
        if jp_arr.size == 0:
            return
        if np.all(np.abs(jp_arr) < args.reset_eepos_eps):
            print("[INFO] Detected near-zero joint_positions after reset; applying absolute EE pose control for stabilization.")
            ee_pos = rs.get("ee_pos", None)
            ee_quat = rs.get("ee_quat", None)
            if ee_pos is None or ee_quat is None:
                return
            ee_pos = np.asarray(ee_pos, dtype=np.float32).reshape(-1)
            ee_quat = np.asarray(ee_quat, dtype=np.float32).reshape(-1)
            gw = rs.get("gripper_width", 0.0)
            gw = float(np.asarray(gw).reshape(-1)[0]) if isinstance(gw, (np.ndarray, list)) else float(gw)
            grip = -1 * (2 * (gw / 0.065) - 1)
            pos_action = np.concatenate([ee_pos, ee_quat, np.array([grip], dtype=np.float32)], axis=0)

            prev_action_type = env.get_action_type()
            env.set_action_type("pos")
            try:
                cur_pos_t, cur_quat_t = env.get_ee_pose()
                cur_pos = cur_pos_t.cpu().numpy()[0]
                cur_quat = cur_quat_t.cpu().numpy()[0]
                target_quat = ee_quat.astype(np.float32)
                target_quat_wxyz = np.array(
                    [target_quat[3], target_quat[0], target_quat[1], target_quat[2]],
                    dtype=np.float32,
                )
                dot_xyzw = float(np.abs(np.dot(cur_quat, target_quat)))
                dot_wxyz = float(np.abs(np.dot(cur_quat, target_quat_wxyz)))
                print("[DEBUG] reset ee-pos fallback")
                print(f"  current ee_pos: {np.round(cur_pos, 3).tolist()}")
                print(f"  target ee_pos:  {np.round(ee_pos, 3).tolist()}")
                print(f"  current ee_quat(xyzw): {np.round(cur_quat, 3).tolist()}")
                print(f"  target ee_quat(xyzw):  {np.round(target_quat, 3).tolist()}")
                print(f"  target ee_quat(wxyz):  {np.round(target_quat_wxyz, 3).tolist()}")
                print(f"  quat dot |xyzw|={dot_xyzw:.3f} |wxyz|={dot_wxyz:.3f}")
            except Exception as e:
                print(f"[DEBUG] reset ee-pos fallback: failed to read current ee pose: {e}")
            for _ in range(args.reset_eepos_steps):
                ac_t = _to_tensor_action(pos_action, num_envs=args.num_envs, device=device)
                env.step(ac_t)
            env.set_action_type(prev_action_type)

    _maybe_recover_eepos_after_reset(init_state)

    # Optional buffers for recording
    imgs1_list = []
    imgs2_list = []
    pc_list = []
    state_list = []
    action_list = []
    img_list = []

    # Use robot_state to compute absolute pos actions; iterate over subsequent observations
    # We step from obs[0] to obs[-1], using obs[t] as target pose at step t
    last_done = None
    for t in range(init_state_idx, len(observations)):
        if t < len(actions):
            formatted_action = [f"{x:.3f}" for x in actions[t]]
            print(f"[DEBUG] Step {t}: Original Action in Pickle: {formatted_action}")

        if args.action_src == "action":
            # Use the recorded action directly
            if t >= len(actions):
                print(f"[INFO] Reached end of recorded actions at step {t}")
                break
            # Ensure action is float32 numpy array
            pos_action = np.array(actions[t], dtype=np.float32)
            
            # If explicit action is used, we might want to ensure gripper is handled correctly if it wasn't in the action?
            # Assuming action in pickle is the full action required by environment.
            
        else:
            # Default "pos": Compute action from observation (force state)
            obs = observations[t]
            rs = obs.get("robot_state")
            if not isinstance(rs, dict):
                raise RuntimeError("Expected robot_state dict in observations for pos control replay")
            # Extract ee_pos (3,) and ee_quat (4,) from stored robot_state
            ee_pos = rs["ee_pos"] if not isinstance(rs["ee_pos"], list) else np.asarray(rs["ee_pos"], dtype=np.float32)
            ee_quat = rs["ee_quat"] if not isinstance(rs["ee_quat"], list) else np.asarray(rs["ee_quat"], dtype=np.float32)
            ee_pos = np.asarray(ee_pos, dtype=np.float32).reshape(-1)
            ee_quat = np.asarray(ee_quat, dtype=np.float32).reshape(-1)
            # Infer gripper action from gripper_width: open (-1) if width increased, else close (+1).
            gw = rs.get("gripper_width", 0.0)
            gw = float(np.asarray(gw).reshape(-1)[0]) if isinstance(gw, (np.ndarray, list)) else float(gw)
            # Normalize gw from [0, 0.065] to [-1, 1]
            grip = -1 * (2 * (gw / 0.065) - 1)
            pos_action = np.concatenate([ee_pos, ee_quat, np.array([grip], dtype=np.float32)], axis=0)

        ac_t = _to_tensor_action(pos_action, num_envs=args.num_envs, device=device)
        # Step the env (pos control) and record images from the step output
        step_obs, _rew, _done, _info = env.step(ac_t)
        last_done = _done
        if args.save_pc_for_dp3 and pc_generator is not None:
            pc = pc_generator.generate_transformed_cropped_point_cloud(
                env_idx=0,
                max_points=args.pc_points,
                downsample_mode=args.pc_downsample_mode,
                visualize=True,
            )
            pc_list.append(pc.cpu().numpy())
            img_list.append(_extract_image(step_obs, "color_image2", env_idx=0) if isinstance(step_obs, dict) else None)
            state_list.append(_extract_robot_state(step_obs, env_idx=0) if isinstance(step_obs, dict) else None)
            action_list.append(pos_action.astype(np.float32))
        # Render helpers
        if args.visualize_eepose:
            render_eepose_arrow(step_obs, env, length=0.05)
        if args.record:
            # Prepare front camera image and optionally overlay world axes before storing
            front_img = None
            if isinstance(step_obs, dict) and "color_image2" in step_obs:
                front_img = np.array(step_obs["color_image2"][0].cpu())
            if front_img is not None and args.visualize_axis:
                front_img = render_world_axes(env, front_img, origin=np.array([-0.1, -0.1, 0.5], dtype=np.float32), length=0.1)
            # Append frames
            if isinstance(step_obs, dict):
                if "color_image1" in step_obs:
                    imgs1_list.append(np.array(step_obs["color_image1"][0].cpu()))
                if front_img is not None:
                    imgs2_list.append(front_img)
                elif "color_image2" in step_obs:
                    imgs2_list.append(np.array(step_obs["color_image2"][0].cpu()))

    # After replay, save videos if requested
    if args.record:
        # Stack lists into arrays of shape (T, H, W, C)
        have_img1 = len(imgs1_list) > 0
        have_img2 = len(imgs2_list) > 0
        if have_img1:
            mp4_cam1 = create_in_memory_mp4(imgs1_list, fps=20)
        if have_img2:
            mp4_cam2 = create_in_memory_mp4(imgs2_list, fps=20)
        # Build output dir sim_replay
        out_dir = Path("sim_record")
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if have_img1:
            cam1_path = out_dir / f"{timestamp}_cam1.mp4"
            with open(cam1_path, "wb") as f1:
                f1.write(mp4_cam1.getvalue() if hasattr(mp4_cam1, "getvalue") else mp4_cam1)
        if have_img2:
            cam2_path = out_dir / f"{timestamp}_cam2.mp4"
            with open(cam2_path, "wb") as f2:
                f2.write(mp4_cam2.getvalue() if hasattr(mp4_cam2, "getvalue") else mp4_cam2)

    print("Replay finished.")

    # Save augmented pickle if task succeeded and point cloud saving enabled
    success = _done_flag(last_done)
    if success and args.save_pc_for_dp3:
        out_dir = Path(args.pc_out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Name file as timestamp + point count
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_name = f"{ts}_{args.pc_points}_{args.pc_downsample_mode}.pkl"
        out_path = out_dir / out_name
        with open(out_path, "wb") as f:
            pickle.dump(
                {
                    "state": state_list,
                    "action": action_list,
                    "point_cloud": pc_list,
                    "img": img_list,
                },
                f,
            )
        print(f"Saved replay with point clouds to {out_path}")
    elif success:
        print("Success without pc export (save_pc_for_dp3 disabled).")


def run_debug_action(env=None, device=None, args=None):
    """Execute debug action sequence: move EE along +X,+Y,+Z (10 steps each),
    overlay current action text, default-enable visualize_eepose and visualize_axis,
    and optionally record videos to sim_record.
    """
    import cv2
    from datetime import datetime
    args.visualize_eepose = True
    args.visualize_axis = True

    # Collect frames
    imgs1_list = []
    imgs2_list = []

    # Get starting EE pose (use env.get_ee_pose for current sim pose)
    ee_pos0_t, ee_quat0_t = env.get_ee_pose()
    ee_pos0 = ee_pos0_t.cpu().numpy()[0]
    ee_quat0 = ee_quat0_t.cpu().numpy()[0]

    phases = [
        ("move +X", np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        ("move -X", np.array([-1.0, 0.0, 0.0], dtype=np.float32)),
        ("move +Y", np.array([0.0, 1.0, 0.0], dtype=np.float32)),
        ("move -Y", np.array([0.0, -1.0, 0.0], dtype=np.float32)),
        ("move +Z", np.array([0.0, 0.0, 1.0], dtype=np.float32)),
        ("move -Z", np.array([0.0, 0.0, -1.0], dtype=np.float32)),
    ]
    step_size = 0.006

    total_steps = 0
    for label, axis in phases:
        for k in range(30):
            # Always base off current EE pose to avoid drift
            ee_pos_curr_t, ee_quat_curr_t = env.get_ee_pose()
            ee_pos_curr = ee_pos_curr_t.cpu().numpy()[0]
            ee_quat_curr = ee_quat_curr_t.cpu().numpy()[0]
            target_pos = ee_pos_curr + axis * (step_size)
            pos_action = np.concatenate([
                target_pos.astype(np.float32),
                ee_quat_curr.astype(np.float32),
                np.array([-1.0], dtype=np.float32),
            ])
            ac_t = _to_tensor_action(pos_action, num_envs=args.num_envs, device=device)
            step_obs, _rew, _done, _info = env.step(ac_t)
            total_steps += 1

            # Visualize EE arrow
            render_eepose_arrow(step_obs, env, length=0.08)

            # Prepare front image and overlay axis + text
            front_img = None
            if isinstance(step_obs, dict) and "color_image2" in step_obs:
                front_img = np.array(step_obs["color_image2"][0].cpu())
            if front_img is not None:
                P, V = env.get_front_projection_view_matrix()
                front_img = render_world_axes(env, front_img, origin=np.array([-0.1, -0.1, 0.4], dtype=np.float32), length=0.2)
                cv2.putText(front_img, f"{label} step {k+1}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)

            # Append frames
            if isinstance(step_obs, dict):
                if "color_image1" in step_obs:
                    imgs1_list.append(np.array(step_obs["color_image1"][0].cpu()))
                if front_img is not None:
                    imgs2_list.append(front_img)
                elif "color_image2" in step_obs:
                    raw2 = np.array(step_obs["color_image2"][0].cpu())
                    cv2.putText(raw2, f"{label} step {k+1}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                    imgs2_list.append(raw2)

    print(f"Debug action finished, total steps: {total_steps}")

    # Save videos
    have_img1 = len(imgs1_list) > 0
    have_img2 = len(imgs2_list) > 0
    if have_img1:
        mp4_cam1 = create_in_memory_mp4(imgs1_list, fps=20)
    if have_img2:
        mp4_cam2 = create_in_memory_mp4(imgs2_list, fps=20)
    out_dir = Path("sim_record")
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if have_img1:
        cam1_path = out_dir / f"{timestamp}_cam1.mp4"
        with open(cam1_path, "wb") as f1:
            f1.write(mp4_cam1.getvalue() if hasattr(mp4_cam1, "getvalue") else mp4_cam1)
    if have_img2:
        cam2_path = out_dir / f"{timestamp}_cam2.mp4"
        with open(cam2_path, "wb") as f2:
            f2.write(mp4_cam2.getvalue() if hasattr(mp4_cam2, "getvalue") else mp4_cam2)

    print("Replay finished.")


if __name__ == "__main__":
    main()
