from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional

import cv2
import numpy as np
import torch

import furniture_bench.controllers.control_utils as C
from furniture_bench.furniture import furniture_factory
from furniture_bench.furniture.parts.lamp_base import LampBase
from furniture_bench.furniture.parts.table_top import TableTop
from furniture_bench.furniture.parts.round_table_top import RoundTableTop
from furniture_bench.utils.pose import rot_mat


VALID_SKILLS = {"pick", "place", "insert", "screw", "push"}
OUTPUT_HEIGHT = 240
OUTPUT_WIDTH = 320


def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_torch(x, device=None, dtype=None):
    if torch.is_tensor(x):
        out = x.clone()
        if device is not None:
            out = out.to(device)
        if dtype is not None:
            out = out.to(dtype=dtype)
        return out
    return torch.as_tensor(x, device=device, dtype=dtype)


def _homogeneous_from_pos_rot(pos, rot):
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = rot
    mat[:3, 3] = pos
    return mat


def _camera_intrinsics(width: int, height: int, horizontal_fov: float) -> np.ndarray:
    fov_rad = math.radians(float(horizontal_fov))
    fx = width / (2.0 * math.tan(fov_rad / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _resize_intrinsics(
    intrinsics: np.ndarray,
    raw_width: int,
    raw_height: int,
    out_width: int = OUTPUT_WIDTH,
    out_height: int = OUTPUT_HEIGHT,
) -> np.ndarray:
    sx = out_width / raw_width
    sy = out_height / raw_height
    resized = intrinsics.copy().astype(np.float32)
    resized[0, 0] *= sx
    resized[1, 1] *= sy
    resized[0, 2] *= sx
    resized[1, 2] *= sy
    return resized


def _resize_crop_intrinsics(
    intrinsics: np.ndarray,
    raw_width: int,
    raw_height: int,
    out_width: int = OUTPUT_WIDTH,
    out_height: int = OUTPUT_HEIGHT,
) -> np.ndarray:
    aspect_ratio = raw_width / raw_height
    new_width = int(out_height * aspect_ratio)
    crop_size = max(0, (new_width - out_width) // 2)
    sx = new_width / raw_width
    sy = out_height / raw_height

    resized = intrinsics.copy().astype(np.float32)
    resized[0, 0] *= sx
    resized[1, 1] *= sy
    resized[0, 2] = resized[0, 2] * sx - crop_size
    resized[1, 2] *= sy
    return resized


def _front_camera_to_sim_local(cam_pos, cam_target) -> np.ndarray:
    cam_pos = _to_numpy(cam_pos).astype(np.float32)
    cam_target = _to_numpy(cam_target).astype(np.float32)
    forward = cam_target - cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    up_ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, up_ref)
    if np.linalg.norm(right) < 1e-6:
        up_ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, up_ref)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)
    rot = np.stack([right, up, forward], axis=1)
    return _homogeneous_from_pos_rot(cam_pos, rot)


def _wrist_camera_to_sim_local(annotation_inputs) -> np.ndarray:
    ee_pos_sim = _to_numpy(annotation_inputs["ee_pos_sim"]).astype(np.float32)
    ee_quat_sim = _to_numpy(annotation_inputs["ee_quat_sim"]).astype(np.float32)
    wrist_offset_pos = _to_numpy(annotation_inputs["wrist_cam_offset_pos"]).astype(np.float32)
    wrist_offset_euler = _to_numpy(annotation_inputs["wrist_cam_offset_euler"]).astype(np.float32)

    ee_rot = _to_numpy(C.quat2mat(torch.as_tensor(ee_quat_sim))).astype(np.float32)
    ee_pose = _homogeneous_from_pos_rot(ee_pos_sim, ee_rot)
    wrist_offset = _homogeneous_from_pos_rot(
        wrist_offset_pos,
        rot_mat(wrist_offset_euler.tolist(), hom=False).astype(np.float32),
    )
    return (ee_pose @ wrist_offset).astype(np.float32)


def _project_sim_local_to_image(
    point_sim_local: np.ndarray,
    intrinsics: np.ndarray,
    sim_local_to_camera: np.ndarray,
    image_width: int,
    image_height: int,
) -> Optional[np.ndarray]:
    point = np.ones(4, dtype=np.float32)
    point[:3] = point_sim_local.astype(np.float32)
    point_cam = sim_local_to_camera @ point
    if point_cam[2] <= 1e-8:
        return None

    point_cv = point_cam[:3].copy()
    point_cv[1] = -point_cv[1]
    point_img = intrinsics @ point_cv
    u = point_img[0] / (point_img[2] + 1e-8)
    v = point_img[1] / (point_img[2] + 1e-8)
    if u < 0 or u >= image_width or v < 0 or v >= image_height:
        return None
    return np.array(
        [int(round(float(u))), int(round(float(v)))],
        dtype=np.int32,
    )


def project_3d_to_2d(
    point_sim_local: np.ndarray,
    camera_info: Dict[str, np.ndarray],
) -> Optional[np.ndarray]:
    if point_sim_local is None:
        return None
    image_size = camera_info["image_size"]
    uv = _project_sim_local_to_image(
        point_sim_local,
        camera_info["intrinsics"],
        camera_info["sim_local_to_camera"],
        int(image_size[0]),
        int(image_size[1]),
    )
    if uv is None:
        return None

    image_width = int(image_size[0])
    image_height = int(image_size[1])
    if uv[0] < 0 or uv[0] >= image_width or uv[1] < 0 or uv[1] >= image_height:
        return None
    return uv.astype(np.int32)


def _build_camera_info(
    annotation_inputs,
    resize_images: bool,
    annotate_wrist_camera: bool,
) -> Dict[str, Dict[str, np.ndarray]]:
    camera_info = {}
    camera_cfgs = annotation_inputs["camera_cfgs"]

    front_cfg = camera_cfgs["front"]
    front_intrinsics = _camera_intrinsics(
        front_cfg["width"], front_cfg["height"], front_cfg["horizontal_fov"]
    )
    if resize_images:
        front_intrinsics = _resize_crop_intrinsics(
            front_intrinsics,
            front_cfg["width"],
            front_cfg["height"],
        )
        front_image_size = np.array([OUTPUT_WIDTH, OUTPUT_HEIGHT], dtype=np.int32)
    else:
        front_image_size = np.array([front_cfg["width"], front_cfg["height"]], dtype=np.int32)
    front_camera_to_sim_local = _front_camera_to_sim_local(
        annotation_inputs["front_cam_pos"],
        annotation_inputs["front_cam_target"],
    )
    camera_info["color_image2"] = {
        "image_size": front_image_size,
        "intrinsics": front_intrinsics.astype(np.float32),
        "camera_to_sim_local": front_camera_to_sim_local.astype(np.float32),
        "sim_local_to_camera": np.linalg.inv(front_camera_to_sim_local).astype(np.float32),
    }

    if annotate_wrist_camera:
        wrist_cfg = camera_cfgs["wrist"]
        wrist_intrinsics = _camera_intrinsics(
            wrist_cfg["width"], wrist_cfg["height"], wrist_cfg["horizontal_fov"]
        )
        if resize_images:
            wrist_intrinsics = _resize_intrinsics(
                wrist_intrinsics,
                wrist_cfg["width"],
                wrist_cfg["height"],
            )
            wrist_image_size = np.array([OUTPUT_WIDTH, OUTPUT_HEIGHT], dtype=np.int32)
        else:
            wrist_image_size = np.array([wrist_cfg["width"], wrist_cfg["height"]], dtype=np.int32)
        wrist_camera_to_sim_local = _wrist_camera_to_sim_local(annotation_inputs)
        camera_info["color_image1"] = {
            "image_size": wrist_image_size,
            "intrinsics": wrist_intrinsics.astype(np.float32),
            "camera_to_sim_local": wrist_camera_to_sim_local.astype(np.float32),
            "sim_local_to_camera": np.linalg.inv(wrist_camera_to_sim_local).astype(np.float32),
        }

    return camera_info


def _get_env_offset(env, env_idx: int, base_pos_global: torch.Tensor) -> torch.Tensor:
    franka_origin = _to_torch(
        np.asarray(env.franka_from_origin_mat, dtype=np.float32)[:3, 3],
        device=base_pos_global.device,
        dtype=base_pos_global.dtype,
    )
    return base_pos_global - franka_origin


def _make_env_local_annotation_inputs(env, env_idx: int, annotation_inputs):
    if "base_pos" in annotation_inputs and annotation_inputs["base_pos"] is not None:
        base_pos_global = _to_torch(
            annotation_inputs["base_pos"], device=env.device, dtype=torch.float32
        )
    else:
        base_pos_global = env.rb_states[env.base_idxs[env_idx], :3].clone().to(torch.float32)

    env_offset = _get_env_offset(env, env_idx, base_pos_global)

    local_inputs = dict(annotation_inputs)
    local_inputs["env_idx"] = env_idx
    local_inputs["env_offset"] = env_offset.clone()

    if "camera_cfgs" not in local_inputs and hasattr(env, "camera_cfgs"):
        local_inputs["camera_cfgs"] = env.camera_cfgs
    elif "camera_cfgs" not in local_inputs and hasattr(env, "camera_cfg"):
        local_inputs["camera_cfgs"] = {
            "front": {
                "width": env.camera_cfg.width,
                "height": env.camera_cfg.height,
                "near_plane": env.camera_cfg.near_plane,
                "far_plane": env.camera_cfg.far_plane,
                "horizontal_fov": env.camera_cfg.horizontal_fov,
            },
            "wrist": {
                "width": env.camera_cfg.width,
                "height": env.camera_cfg.height,
                "near_plane": env.camera_cfg.near_plane,
                "far_plane": env.camera_cfg.far_plane,
                "horizontal_fov": 55.0 if getattr(env, "resize_img", False) else 69.4,
            },
        }

    if "front_cam_pos" not in local_inputs and hasattr(env, "front_cam_pos"):
        local_inputs["front_cam_pos"] = np.asarray(env.front_cam_pos, dtype=np.float32)
    if "front_cam_target" not in local_inputs and hasattr(env, "front_cam_target"):
        local_inputs["front_cam_target"] = np.asarray(env.front_cam_target, dtype=np.float32)
    if "wrist_cam_offset_pos" not in local_inputs and hasattr(env, "wrist_cam_offset_pos"):
        local_inputs["wrist_cam_offset_pos"] = np.asarray(env.wrist_cam_offset_pos, dtype=np.float32)
    if "wrist_cam_offset_euler" not in local_inputs and hasattr(env, "wrist_cam_offset_euler"):
        local_inputs["wrist_cam_offset_euler"] = np.asarray(env.wrist_cam_offset_euler, dtype=np.float32)

    local_inputs["base_pos"] = (base_pos_global - env_offset).clone()

    if "ee_pos_sim" in local_inputs and local_inputs["ee_pos_sim"] is not None:
        local_inputs["ee_pos_sim"] = (
            _to_torch(local_inputs["ee_pos_sim"], device=env.device, dtype=torch.float32)
            - env_offset
        )
    elif hasattr(env, "ee_idxs"):
        local_inputs["ee_pos_sim"] = env.rb_states[env.ee_idxs[env_idx], :3].clone() - env_offset

    if "ee_quat_sim" not in local_inputs and hasattr(env, "ee_idxs"):
        local_inputs["ee_quat_sim"] = env.rb_states[env.ee_idxs[env_idx], 3:7].clone()

    for key in ["left_finger_pos", "right_finger_pos"]:
        if key in local_inputs and local_inputs[key] is not None:
            local_inputs[key] = (
                _to_torch(local_inputs[key], device=env.device, dtype=torch.float32) - env_offset
            )

    compact_indices = []
    compact_part_idxs = {}
    for name, part_indices in env.part_idxs.items():
        if len(part_indices) <= env_idx:
            continue
        compact_part_idxs[name] = [len(compact_indices)]
        compact_indices.append(int(part_indices[env_idx]))

    if compact_indices:
        compact_rb_states = env.rb_states[compact_indices].clone()
        compact_rb_states[:, :3] -= env_offset.unsqueeze(0)
    else:
        compact_rb_states = env.rb_states.new_zeros((0, env.rb_states.shape[1]))

    local_inputs["rb_states"] = compact_rb_states
    local_inputs["part_idxs"] = compact_part_idxs

    return local_inputs


@dataclass
class SkillAnnotator:
    furniture_name: str
    previous_skill: Optional[str] = None
    previous_guidance_point_robot: Optional[np.ndarray] = None

    def __post_init__(self):
        self.furniture = furniture_factory(self.furniture_name)
        self.furniture.reset()
        self.assemble_idx = 0

    def _reset_parts(self):
        self.furniture.reset()

    def reset(self):
        self.furniture.reset()
        self.assemble_idx = 0
        self.previous_skill = None
        self.previous_guidance_point_robot = None

    def _assembled(self, annotation_inputs, part_idx1, part_idx2):
        pair = (part_idx1, part_idx2)
        if "assembled_mask" in annotation_inputs and annotation_inputs["assembled_mask"] is not None:
            try:
                pair_idx = self.furniture.should_be_assembled.index(pair)
                assembled_mask = annotation_inputs["assembled_mask"]
                assembled = bool(assembled_mask[pair_idx].item())
                # if self.furniture_name == "round_table":
                #     print(
                #         "[util assembled_debug] "
                #         f"source=env_mask pair={pair} pair_idx={pair_idx} assembled={assembled} "
                #         f"assembled_mask={assembled_mask.tolist()}"
                #     )
                return assembled
            except ValueError:
                pass

        part1 = self.furniture.parts[part_idx1]
        part2 = self.furniture.parts[part_idx2]
        rb_states = annotation_inputs["rb_states"]
        part_idxs = annotation_inputs["part_idxs"]

        part1_pose = C.to_homogeneous(
            rb_states[part_idxs[part1.name]][0][:3],
            C.quat2mat(rb_states[part_idxs[part1.name]][0][3:7]),
        )
        part2_pose = C.to_homogeneous(
            rb_states[part_idxs[part2.name]][0][:3],
            C.quat2mat(rb_states[part_idxs[part2.name]][0][3:7]),
        )
        rel_pose = torch_inv(part1_pose) @ part2_pose
        assembled_rel_poses = self.furniture.assembled_rel_poses[(part_idx1, part_idx2)]
        assembled = self.furniture.assembled(rel_pose.cpu().numpy(), assembled_rel_poses)
        # if self.furniture_name == "round_table":
        #     print(
        #         "[util assembled_debug] "
        #         f"source=local_recompute pair={pair} assembled={assembled}"
        #     )
        return assembled

    def _update_part1_skill_state(self, part, annotation_inputs):
        if isinstance(part, (RoundTableTop, LampBase)):
            skill_state = part.update_skill_state(
                annotation_inputs["ee_pos"],
                annotation_inputs["ee_quat"],
                annotation_inputs["rb_states"],
                annotation_inputs["part_idxs"],
                annotation_inputs["sim_to_april_mat"],
                annotation_inputs["april_to_robot_mat"],
                annotation_inputs["left_finger_pos"],
                annotation_inputs["right_finger_pos"],
                annotation_inputs["left_finger_force"],
                annotation_inputs["right_finger_force"],
            )
        else:
            skill_state = part.update_skill_state(
                annotation_inputs["ee_pos"],
                annotation_inputs["ee_quat"],
                annotation_inputs["gripper_width"],
                annotation_inputs["rb_states"],
                annotation_inputs["part_idxs"],
                annotation_inputs["sim_to_april_mat"],
                annotation_inputs["april_to_robot_mat"],
                annotation_inputs["left_finger_pos"],
                annotation_inputs["right_finger_pos"],
                annotation_inputs["left_finger_force"],
                annotation_inputs["right_finger_force"],
                annotation_inputs["part_contact_forces"].get(part.name),
            )
        skill = part.get_skill_label()
        guidance_point_robot = part.get_guidance_point()
        return skill_state, skill, guidance_point_robot

    def _update_operated_part(self, part, annotation_inputs, assemble_to_name, assembled):
        skill_state = part.update_skill_state(
            annotation_inputs["ee_pos"],
            annotation_inputs["ee_quat"],
            annotation_inputs["gripper_width"],
            annotation_inputs["rb_states"],
            annotation_inputs["part_idxs"],
            annotation_inputs["sim_to_april_mat"],
            annotation_inputs["april_to_robot_mat"],
            annotation_inputs["left_finger_pos"],
            annotation_inputs["right_finger_pos"],
            annotation_inputs["left_finger_force"],
            annotation_inputs["right_finger_force"],
            annotation_inputs["part_contact_forces"].get(part.name),
            assemble_to_name,
            assembled=assembled,
        )
        skill = part.get_skill_label()
        guidance_point_robot = part.get_guidance_point()
        return skill_state, skill, guidance_point_robot

    def _reset_next_pair(self, pair_idx):
        if pair_idx >= len(self.furniture.should_be_assembled):
            return
        part1_idx, part2_idx = self.furniture.should_be_assembled[pair_idx]
        part1 = self.furniture.parts[part1_idx]
        part2 = self.furniture.parts[part2_idx]

        should_skip_part1_reset = (
            self.furniture_name == "lamp"
            and getattr(part1, "name", None) == "lamp_base"
            and getattr(part1, "pre_assemble_done", False)
        )
        if not should_skip_part1_reset:
            reset_fn = getattr(part1, "reset", None)
            if callable(reset_fn):
                reset_fn()
            else:
                reset_skill_state_fn = getattr(part1, "reset_skill_state", None)
                if callable(reset_skill_state_fn):
                    reset_skill_state_fn()

        reset_skill_state_fn = getattr(part2, "reset_skill_state", None)
        if callable(reset_skill_state_fn):
            reset_skill_state_fn()

    def step(
        self,
        env,
        env_idx: int = 0,
        annotate_wrist_camera: bool = False,
        resize_images: bool = True,
    ):
        if self.furniture_name not in {"one_leg", "round_table", "lamp"}:
            return {
                "skill": None,
                "guidance_point": None,
                "guidance_point_2d": {},
                "camera_info": {},
            }

        annotation_inputs = _make_env_local_annotation_inputs(
            env, env_idx, env.get_skill_annotation_inputs(env_idx=env_idx)
        )
        incoming_assemble_idx = annotation_inputs.get("current_assemble_idx")
        num_pairs = len(self.furniture.should_be_assembled)
        if incoming_assemble_idx is not None:
            incoming_assemble_idx = int(incoming_assemble_idx)
            if self.assemble_idx >= num_pairs or incoming_assemble_idx < self.assemble_idx:
                self.assemble_idx = incoming_assemble_idx
        camera_info = _build_camera_info(
            annotation_inputs,
            resize_images=resize_images,
            annotate_wrist_camera=annotate_wrist_camera,
        )
        if self.assemble_idx >= num_pairs:
            return {
                "skill": self.previous_skill,
                "guidance_point": self.previous_guidance_point_robot,
                "guidance_point_2d": {},
                "camera_info": camera_info,
            }

        part1_idx, part2_idx = self.furniture.should_be_assembled[self.assemble_idx]
        part1 = self.furniture.parts[part1_idx]
        part2 = self.furniture.parts[part2_idx]

        skill_state = None
        skill = None
        guidance_point_robot = None
        debug_info = {
            "assemble_idx": self.assemble_idx,
            "active_part": None,
            "phase": None,
        }

        part1_phase_active = (
            not getattr(part1, "pre_assemble_done", True)
            and getattr(part1, "skill_state", None) != "done"
        )
        if part1_phase_active:
            skill_state, skill, guidance_point_robot = self._update_part1_skill_state(
                part1, annotation_inputs
            )
            if skill_state == "done":
                part1.pre_assemble_done = True
            debug_info["active_part"] = part1.name
            debug_info["phase"] = "pre_assemble"

        assembled = self._assembled(annotation_inputs, part1_idx, part2_idx)
        part1_phase_complete = getattr(part1, "pre_assemble_done", True) or (
            getattr(part1, "skill_state", None) == "done"
        )
        if part1_phase_complete:
            skill_state, skill, guidance_point_robot = self._update_operated_part(
                part2,
                annotation_inputs,
                assemble_to_name=part1.name,
                assembled=assembled,
            )
            debug_info["active_part"] = part2.name
            debug_info["phase"] = "assemble"

        if assembled:
            self.assemble_idx = min(self.assemble_idx + 1, num_pairs)
            self._reset_next_pair(self.assemble_idx)

        if incoming_assemble_idx is not None:
            self.assemble_idx = max(self.assemble_idx, incoming_assemble_idx)

        if skill is None or skill_state == "done":
            skill = self.previous_skill
            guidance_point_robot = self.previous_guidance_point_robot
        else:
            self.previous_skill = skill
            if guidance_point_robot is not None:
                self.previous_guidance_point_robot = _to_numpy(guidance_point_robot).astype(np.float32)

        guidance_point = None
        if guidance_point_robot is not None:
            guidance_point = (
                _to_numpy(guidance_point_robot).astype(np.float32)
                + _to_numpy(annotation_inputs["base_pos"]).astype(np.float32)
            )

        guidance_point_2d = {}
        if guidance_point is not None:
            for image_key, cam in camera_info.items():
                uv = project_3d_to_2d(guidance_point, cam)
                guidance_point_2d[image_key] = None if uv is None else uv.astype(np.float32)
        else:
            for image_key in camera_info.keys():
                guidance_point_2d[image_key] = None

        return {
            "skill": skill,
            "guidance_point": guidance_point,
            "guidance_point_2d": guidance_point_2d,
            "camera_info": camera_info,
            "debug": debug_info,
        }


def torch_inv(mat):
    return mat.inverse()


def reset_skill_annotator(env, env_idx: Optional[int] = None):
    furniture_name = getattr(env, "furniture_name", "")
    num_envs = max(int(getattr(env, "num_envs", 1)), 1)
    if env_idx is None:
        env._skill_annotators = [SkillAnnotator(furniture_name) for _ in range(num_envs)]
        env._skill_annotator = env._skill_annotators[0]
        return

    annotators = getattr(env, "_skill_annotators", None)
    if annotators is None or len(annotators) != num_envs:
        reset_skill_annotator(env)
        annotators = env._skill_annotators

    annotators[env_idx] = SkillAnnotator(furniture_name)
    env._skill_annotator = annotators[0]


def _get_or_create_skill_annotator(env, env_idx: int) -> SkillAnnotator:
    num_envs = max(int(getattr(env, "num_envs", 1)), 1)
    annotators = getattr(env, "_skill_annotators", None)
    if (
        annotators is None
        or len(annotators) != num_envs
        or any(getattr(a, "furniture_name", None) != getattr(env, "furniture_name", None) for a in annotators)
    ):
        reset_skill_annotator(env)
        annotators = env._skill_annotators

    env._skill_annotator = annotators[0]
    return annotators[env_idx]


def get_annotation_bundle_for_env(
    env,
    env_idx: int,
    previous_skill: str | None = None,
    annotate_wrist_camera: bool = False,
    resize_images: bool = True,
):
    if getattr(env, "furniture_name", None) not in {"one_leg", "round_table", "lamp"}:
        return {
            "skill": None,
            "guidance_point": None,
            "guidance_point_2d": {},
            "camera_info": {},
            "debug": {},
        }

    annotator = _get_or_create_skill_annotator(env, env_idx)

    if previous_skill is not None and annotator.previous_skill is None:
        annotator.previous_skill = previous_skill

    return annotator.step(
        env,
        env_idx=env_idx,
        annotate_wrist_camera=annotate_wrist_camera,
        resize_images=resize_images,
    )


def get_annotation_bundle_all_envs(
    env,
    previous_skills: Optional[list[str | None]] = None,
    annotate_wrist_camera: bool = False,
    resize_images: bool = True,
):
    num_envs = max(int(getattr(env, "num_envs", 1)), 1)
    bundles = []
    for env_idx in range(num_envs):
        previous_skill = None
        if previous_skills is not None and env_idx < len(previous_skills):
            previous_skill = previous_skills[env_idx]
        bundles.append(
            get_annotation_bundle_for_env(
                env,
                env_idx=env_idx,
                previous_skill=previous_skill,
                annotate_wrist_camera=annotate_wrist_camera,
                resize_images=resize_images,
            )
        )
    return bundles


def get_annotation_bundle(
    env,
    previous_skill: str | None = None,
    annotate_wrist_camera: bool = False,
    resize_images: bool = True,
):
    return get_annotation_bundle_for_env(
        env,
        env_idx=0,
        previous_skill=previous_skill,
        annotate_wrist_camera=annotate_wrist_camera,
        resize_images=resize_images,
    )


def get_skill_label(
    env,
    previous_skill: str | None = None,
    annotate_wrist_camera: bool = False,
    resize_images: bool = True,
) -> str | None:
    bundle = get_annotation_bundle(
        env,
        previous_skill=previous_skill,
        annotate_wrist_camera=annotate_wrist_camera,
        resize_images=resize_images,
    )
    return bundle["skill"]


def draw_skill_on_image(image: np.ndarray, skill: str) -> np.ndarray:
    if skill not in VALID_SKILLS:
        return image

    annotated = image.copy()
    text = f"skill: {skill}"
    origin = (10, 28)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2

    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    top_left = (6, 6)
    bottom_right = (top_left[0] + text_w + 12, top_left[1] + text_h + baseline + 12)

    cv2.rectangle(annotated, top_left, bottom_right, (0, 0, 0), thickness=-1)
    cv2.putText(
        annotated,
        text,
        origin,
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return annotated


def draw_guidance_point_on_image(image: np.ndarray, guidance_point_2d) -> np.ndarray:
    if guidance_point_2d is None:
        return image

    uv = _to_numpy(guidance_point_2d).astype(np.int32)
    annotated = image.copy()
    point_radius = 2
    point_alpha = 0.5
    point_color = (255, 0, 0)

    def _draw_point(frame: np.ndarray, center: tuple[int, int]) -> np.ndarray:
        overlay = frame.copy()
        cv2.circle(
            overlay,
            center,
            point_radius,
            point_color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        return cv2.addWeighted(overlay, point_alpha, frame, 1.0 - point_alpha, 0.0)

    if annotated.ndim == 4:
        if annotated.shape[0] != 1:
            return annotated
        frame = annotated[0]
        height, width = frame.shape[:2]
        center = (int(uv[0]), int(uv[1]))
        if center[0] < 0 or center[0] >= width or center[1] < 0 or center[1] >= height:
            return annotated
        annotated[0] = _draw_point(frame, center)
        return annotated

    height, width = annotated.shape[:2]
    center = (int(uv[0]), int(uv[1]))
    if center[0] < 0 or center[0] >= width or center[1] < 0 or center[1] >= height:
        return annotated
    return _draw_point(annotated, center)
