from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

import furniture_bench.controllers.control_utils as C
from furniture_bench.furniture import furniture_factory


VALID_SKILLS = {"pick", "place", "insert", "screw", "push"}


@dataclass
class SkillAnnotator:
    furniture_name: str
    previous_skill: Optional[str] = None

    def __post_init__(self):
        self.furniture = furniture_factory(self.furniture_name)
        self.furniture.reset()
        self._reset_parts()
        self.assemble_idx = 0

    def _reset_parts(self):
        for part in self.furniture.parts:
            reset_fn = getattr(part, "reset_skill_state", None)
            if callable(reset_fn):
                reset_fn()

    def reset(self):
        self._reset_parts()
        self.assemble_idx = 0
        self.previous_skill = None

    def _assembled(self, annotation_inputs, part_idx1, part_idx2):
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
        return self.furniture.assembled(rel_pose.cpu().numpy(), assembled_rel_poses)

    def step(self, env) -> Optional[str]:
        if self.furniture_name != "one_leg":
            return None
        if getattr(env, "num_envs", 1) != 1:
            return None

        annotation_inputs = env.get_skill_annotation_inputs()
        table_top = self.furniture.parts[0]
        leg = self.furniture.parts[4]

        if table_top.skill_state != "done":
            skill_state = table_top.update_skill_state(
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
            skill = table_top.get_skill_label()
        else:
            assembled = self._assembled(annotation_inputs, 0, 4)
            skill_state = leg.update_skill_state(
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
                table_top.name,
                assembled=assembled,
            )
            if assembled:
                self.assemble_idx = 1
            skill = leg.get_skill_label()

        if skill is None:
            return self.previous_skill
        if skill_state == "done":
            return self.previous_skill

        self.previous_skill = skill
        return skill


def torch_inv(mat):
    return mat.inverse()


def reset_skill_annotator(env):
    env._skill_annotator = SkillAnnotator(getattr(env, "furniture_name", ""))


def get_skill_label(env, previous_skill: str | None = None) -> str | None:
    if getattr(env, "furniture_name", None) != "one_leg":
        return None

    annotator = getattr(env, "_skill_annotator", None)
    if annotator is None or getattr(annotator, "furniture_name", None) != env.furniture_name:
        reset_skill_annotator(env)
        annotator = env._skill_annotator

    if previous_skill is not None and annotator.previous_skill is None:
        annotator.previous_skill = previous_skill

    return annotator.step(env)


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
