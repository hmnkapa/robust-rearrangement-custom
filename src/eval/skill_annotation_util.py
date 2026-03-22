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

    def step(self, env, annotate_wrist_camera: bool = False, resize_images: bool = True):
        if self.furniture_name not in {"one_leg", "round_table", "lamp"}:
            return {
                "skill": None,
                "guidance_point": None,
                "guidance_point_2d": {},
                "camera_info": {},
            }
        if getattr(env, "num_envs", 1) != 1:
            return {
                "skill": None,
                "guidance_point": None,
                "guidance_point_2d": {},
                "camera_info": {},
            }

        annotation_inputs = env.get_skill_annotation_inputs()
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


def reset_skill_annotator(env):
    env._skill_annotator = SkillAnnotator(getattr(env, "furniture_name", ""))


def get_annotation_bundle(
    env,
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

    annotator = getattr(env, "_skill_annotator", None)
    if annotator is None or getattr(annotator, "furniture_name", None) != env.furniture_name:
        reset_skill_annotator(env)
        annotator = env._skill_annotator

    if previous_skill is not None and annotator.previous_skill is None:
        annotator.previous_skill = previous_skill

    return annotator.step(
        env,
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

    if annotated.ndim == 4:
        if annotated.shape[0] != 1:
            return annotated
        frame = annotated[0]
        height, width = frame.shape[:2]
        center = (int(uv[0]), int(uv[1]))
        if center[0] < 0 or center[0] >= width or center[1] < 0 or center[1] >= height:
            return annotated
        cv2.circle(frame, center, 6, (255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
        annotated[0] = frame
        return annotated

    height, width = annotated.shape[:2]
    center = (int(uv[0]), int(uv[1]))
    if center[0] < 0 or center[0] >= width or center[1] < 0 or center[1] >= height:
        return annotated
    cv2.circle(annotated, center, 6, (255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
    return annotated
