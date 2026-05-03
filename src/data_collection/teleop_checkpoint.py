"""Teleop entry point with checkpoint/resume (断点续采) support.

Key bindings (differs from original teleop):
    c  - set checkpoint (save current robot + parts state)
    r  - resume to last checkpoint (restore robot joints, gripper, parts poses)
    p  - toggle pause/resume recording
    t  - mark trajectory as success
    y  - mark trajectory as success + record final part poses
    n  - mark trajectory as failure
    b  - undo last 10 actions
    z  - toggle gripper
    [ ] - adjust movement step size
"""

import argparse
import random

from pathlib import Path

from pynput.keyboard import KeyCode

from src.data_collection.data_collector_checkpoint import DataCollectorCheckpoint
from src.data_collection.keyboard_interface import KeyboardInterface
from src.data_collection.collect_enum import CollectEnum
from furniture_bench.envs.initialization_mode import Randomness
from src.common.files import trajectory_save_dir
from src.gym import turn_off_april_tags
import gym


SUPPORTED_TELEOP_FURNITURE = [
    "one_leg",
    "lamp",
    "round_table",
    "square_table",
    "desk",
    "cabinet",
    "factory_peg_hole",
    "factory_nut_bolt",
]


class KeyboardInterfaceCheckpoint(KeyboardInterface):
    """Keyboard interface that adds checkpoint/resume keys.

    Overrides get_action to intercept the parent's CONTINUE (c key) and
    RESET (r key) enums, converting them to checkpoint/resume flags.
    The p key toggles between pause and continue.
    """

    def __init__(self):
        super().__init__()
        self.checkpoint_pressed = False
        self.resume_pressed = False
        self._paused = False

    def reset(self):
        super().reset()
        self.checkpoint_pressed = False
        self.resume_pressed = False
        self._paused = False

    def on_press(self, k):
        """Intercept c and r keys before delegating to parent."""
        try:
            k_char = k.char
            if k_char is None:
                return
            k_char = k_char.lower()

            if self.waiting_for_start and k_char == "s":
                with self._start_signal_lock:
                    self._start_signal_count += 1
                return

            if k_char == "c":
                gym.logger.info("Checkpoint pressed")
                self.checkpoint_pressed = True
                return

            if k_char == "r":
                gym.logger.info("Resume pressed")
                self.resume_pressed = True
                return

            if k_char == "p":
                gym.logger.info("Pause toggle pressed")
                self._paused = not self._paused
                if self._paused:
                    self.key_enum = CollectEnum.PAUSE
                else:
                    self.key_enum = CollectEnum.CONTINUE
                return

            # Delegate all other keys to the parent handler.
            # Reconstruct a KeyCode object for pynput compatibility.
            super().on_press(KeyCode.from_char(k_char))
        except AttributeError:
            pass


def main():
    parser = argparse.ArgumentParser(description="Collect IL data with checkpoint/resume")
    parser.add_argument(
        "--furniture",
        help="Name of the furniture",
        choices=SUPPORTED_TELEOP_FURNITURE,
        required=True,
    )
    parser.add_argument(
        "--save-failure",
        action="store_true",
        help="Save failure trajectories.",
    )
    parser.add_argument("--randomness", default="low", choices=["low", "med", "high"])
    parser.add_argument("--gpu-id", default=0, type=int)
    parser.add_argument("--num-demos", default=100, type=int)
    parser.add_argument(
        "--ctrl-mode",
        type=str,
        help="Type of low level controller to use.",
        choices=["osc", "diffik"],
        required=True,
    )
    parser.add_argument(
        "--draw-marker",
        action="store_true",
        help="If set, will draw an AprilTag marker on the furniture",
    )
    parser.add_argument(
        "--no-ee-laser",
        action="store_false",
        help="If set, will not show the laser coming from the end effector",
        dest="ee_laser",
    )
    parser.add_argument(
        "--resume-dir",
        type=str,
        help="Directory to resume trajectories from",
        default=None,
    )
    parser.add_argument(
        "--sample-perturbations",
        action="store_true",
    )
    parser.add_argument(
        "--sm-pos-speed",
        type=float,
        default=None,
        help=(
            "Override SpaceMouse max translational speed in meters per second. "
            "Default is 0.54 for the src diffik collector."
        ),
    )
    parser.add_argument(
        "--sm-rot-speed",
        type=float,
        default=None,
        help=(
            "Override SpaceMouse max rotational speed in radians per second. "
            "Default is 2.8 for the src diffik collector."
        ),
    )
    parser.add_argument(
        "--teleop-setting",
        type=int,
        choices=[1, 2],
        default=1,
        help=(
            "Teleoperation preset. 1 matches the existing src behavior "
            "(world-frame position/rotation). 2 matches furniture-bench "
            "(end-effector-frame position/rotation with adjusted signs)."
        ),
    )
    parser.add_argument(
        "--show-teleop-cameras",
        action="store_true",
        help="Show an OpenCV preview window for fixed and wrist cameras.",
    )

    args = parser.parse_args()

    if not args.draw_marker:
        turn_off_april_tags()

    keyboard_device_interface = KeyboardInterfaceCheckpoint()
    keyboard_device_interface.print_usage()
    print("==========Checkpoint Keys==========")
    print("c: set checkpoint (save current state)")
    print("r: resume to last checkpoint")
    print("p: toggle pause/resume recording")
    print("===================================")

    randomness = Randomness.str_to_enum(args.randomness)

    data_path: Path = trajectory_save_dir(
        controller=args.ctrl_mode,
        domain="sim",
        task=args.furniture,
        demo_source="teleop",
        randomness=args.randomness + ("_perturb" if args.sample_perturbations else ""),
    )

    if args.resume_dir is not None:
        pickle_paths = list(Path(args.resume_dir).rglob("*.pkl*"))
        random.shuffle(pickle_paths)
        pickle_paths = pickle_paths[: args.num_demos]
        print("loaded num trajectories", len(pickle_paths))
    else:
        pickle_paths = None

    data_collector = DataCollectorCheckpoint(
        data_path=data_path,
        device_interface=keyboard_device_interface,
        furniture=args.furniture,
        draw_marker=args.draw_marker,
        resize_sim_img=True,
        randomness=randomness,
        compute_device_id=args.gpu_id,
        graphics_device_id=args.gpu_id,
        save_failure=args.save_failure,
        num_demos=args.num_demos,
        ctrl_mode=args.ctrl_mode,
        ee_laser=args.ee_laser,
        compress_pickles=False,
        resume_trajectory_paths=pickle_paths,
        sample_perturbations=args.sample_perturbations,
        sm_pos_speed=args.sm_pos_speed,
        sm_rot_speed=args.sm_rot_speed,
        teleop_setting=args.teleop_setting,
        show_teleop_cameras=args.show_teleop_cameras,
    )
    data_collector.collect()


if __name__ == "__main__":
    main()
