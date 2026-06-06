#!/usr/bin/env python3
"""Restart residual PPO training from the latest local checkpoint after crashes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Iterable, Optional, Sequence


CHECKPOINT_RE = re.compile(r"^actor_chkpt_(\d+)\.pt$")
RUN_NAME_RE = re.compile(r"Run name:\s*(?P<name>\S+)")
SAVED_CHECKPOINT_RE = re.compile(r"Model saved to (?P<path>.+actor_chkpt_\d+\.pt)")
WANDB_RUN_URL_RE = re.compile(r"/runs/(?P<id>[A-Za-z0-9_-]+)")
WANDB_LOCAL_RE = re.compile(r"wandb[/\\]run-[0-9_]+-(?P<id>[A-Za-z0-9_-]+)")

REQUIRED_CHECKPOINT_KEYS = {
    "model_state_dict",
    "optimizer_actor_state_dict",
    "optimizer_critic_state_dict",
    "scheduler_actor_state_dict",
    "scheduler_critic_state_dict",
    "iteration",
}


@dataclass
class RunState:
    run_name: Optional[str] = None
    wandb_run_id: Optional[str] = None


def checkpoint_iteration(path: Path) -> Optional[int]:
    match = CHECKPOINT_RE.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def is_valid_checkpoint(path: Path) -> bool:
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False

    if not isinstance(payload, dict):
        return False
    return REQUIRED_CHECKPOINT_KEYS.issubset(payload.keys())


def find_latest_checkpoint(run_dir: Path) -> Optional[Path]:
    if not run_dir.exists():
        return None

    candidates = []
    for path in run_dir.iterdir():
        iteration = checkpoint_iteration(path)
        if iteration is not None:
            candidates.append((iteration, path))

    for _, path in sorted(candidates, reverse=True):
        if is_valid_checkpoint(path):
            return path
    return None


def cleanup_old_checkpoints(run_dir: Path) -> tuple[Optional[Path], list[Path]]:
    """Keep the latest valid checkpoint and every 100-iteration checkpoint."""
    latest = find_latest_checkpoint(run_dir)
    if latest is None:
        return None, []

    deleted = []
    for path in run_dir.iterdir():
        iteration = checkpoint_iteration(path)
        if iteration is None or path == latest or iteration % 100 == 0:
            continue
        try:
            path.unlink()
            deleted.append(path)
        except FileNotFoundError:
            continue
    return latest, deleted


def _override_key(arg: str) -> Optional[str]:
    if "=" not in arg:
        return None
    key = arg.split("=", 1)[0]
    return key.lstrip("+")


def get_hydra_override(command: Sequence[str], key: str) -> Optional[str]:
    for arg in command:
        if _override_key(arg) == key:
            return arg.split("=", 1)[1]
    return None


def replace_or_append_hydra_override(
    command: Sequence[str], key: str, value: object
) -> list[str]:
    override = f"{key}={value}"
    updated = list(command)
    for index, arg in enumerate(updated):
        if _override_key(arg) == key:
            updated[index] = override
            return updated
    updated.append(override)
    return updated


def build_resume_command(
    base_command: Sequence[str],
    checkpoint_path: Path,
    wandb_run_id: Optional[str],
) -> list[str]:
    command = replace_or_append_hydra_override(
        base_command, "resume.checkpoint_path", checkpoint_path
    )
    if wandb_run_id:
        command = replace_or_append_hydra_override(
            command, "wandb.continue_run_id", wandb_run_id
        )
    return command


def update_run_state_from_line(state: RunState, line: str) -> None:
    run_match = RUN_NAME_RE.search(line)
    if run_match is not None:
        state.run_name = run_match.group("name")

    url_match = WANDB_RUN_URL_RE.search(line)
    if url_match is not None:
        state.wandb_run_id = url_match.group("id")
        return

    local_match = WANDB_LOCAL_RE.search(line)
    if local_match is not None:
        state.wandb_run_id = local_match.group("id")


def checkpoint_path_from_line(line: str, workdir: Path) -> Optional[Path]:
    match = SAVED_CHECKPOINT_RE.search(line)
    if match is None:
        return None
    path = Path(match.group("path").strip())
    if not path.is_absolute():
        path = workdir / path
    return path


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _quote_command(command: Iterable[str]) -> str:
    return " ".join(command)


class AutoResumeRunner:
    def __init__(
        self,
        *,
        workdir: Path,
        log_path: Path,
        restart_delay: float,
    ) -> None:
        self.workdir = workdir
        self.log_path = log_path
        self.restart_delay = restart_delay
        self.child: Optional[subprocess.Popen[str]] = None
        self.stop_requested = False
        self.stop_signal: Optional[int] = None

    def install_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, _frame) -> None:
        self.stop_requested = True
        self.stop_signal = int(signum)
        child = self.child
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    def _log_wrapper(self, log_file, message: str) -> None:
        line = f"[auto-resume] {message}\n"
        sys.stdout.write(line)
        sys.stdout.flush()
        log_file.write(line)
        log_file.flush()

    def run_once(self, command: Sequence[str], state: RunState, log_file) -> int:
        self._log_wrapper(log_file, f"launching: {_quote_command(command)}")
        preexec_fn = os.setsid if hasattr(os, "setsid") else None
        self.child = subprocess.Popen(
            list(command),
            cwd=self.workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            preexec_fn=preexec_fn,
        )

        assert self.child.stdout is not None
        for line in self.child.stdout:
            update_run_state_from_line(state, line)
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()

            saved_checkpoint = checkpoint_path_from_line(line, self.workdir)
            if saved_checkpoint is not None:
                self.cleanup_run_checkpoints(saved_checkpoint.parent, log_file)

        return_code = self.child.wait()
        self.child = None
        self._log_wrapper(log_file, f"process exited with code {return_code}")
        if state.run_name:
            self.cleanup_run_checkpoints(self.workdir / "models" / state.run_name, log_file)
        return return_code

    def cleanup_run_checkpoints(self, run_dir: Path, log_file) -> None:
        latest, deleted = cleanup_old_checkpoints(run_dir)
        if latest is None or not deleted:
            return
        self._log_wrapper(
            log_file,
            f"kept checkpoint {latest}; removed {len(deleted)} old checkpoint(s)",
        )

    def sleep_before_restart(self, log_file) -> None:
        if self.restart_delay <= 0:
            return
        self._log_wrapper(log_file, f"restarting in {self.restart_delay:g}s")
        deadline = time.time() + self.restart_delay
        while time.time() < deadline and not self.stop_requested:
            time.sleep(min(0.2, deadline - time.time()))

    def exit_code_after_signal(self) -> int:
        if self.stop_signal is None:
            return 1
        return 128 + int(self.stop_signal)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restart residual PPO from the latest local checkpoint."
    )
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--log-dir", type=Path, default=Path("auto_resume_logs"))
    parser.add_argument("--restart-delay", type=float, default=10.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing training command after --")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    workdir = args.workdir.expanduser().resolve()
    log_dir = args.log_dir
    if not log_dir.is_absolute():
        log_dir = workdir / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"auto_resume_{_timestamp()}.log"

    original_command = list(args.command)
    state = RunState()
    resume_count = 0

    initial_resume = get_hydra_override(original_command, "resume.checkpoint_path")
    if initial_resume:
        original_command = replace_or_append_hydra_override(
            original_command, "resume.checkpoint_path", initial_resume
        )
        state.run_name = Path(initial_resume).expanduser().parent.name
        resume_count = 1
    initial_wandb_id = get_hydra_override(original_command, "wandb.continue_run_id")
    if initial_wandb_id:
        state.wandb_run_id = initial_wandb_id

    runner = AutoResumeRunner(
        workdir=workdir,
        log_path=log_path,
        restart_delay=float(args.restart_delay),
    )
    runner.install_signal_handlers()

    attempt = 0
    command = original_command
    with log_path.open("a", encoding="utf-8") as log_file:
        runner._log_wrapper(log_file, f"log file: {log_path}")
        if initial_resume:
            runner._log_wrapper(
                log_file,
                f"initial resume checkpoint: {initial_resume} (resume #{resume_count})",
            )
        while True:
            if attempt > 0 and state.run_name:
                run_dir = workdir / "models" / state.run_name
                checkpoint_path = find_latest_checkpoint(run_dir)
                if checkpoint_path is not None:
                    resume_count += 1
                    command = build_resume_command(
                        original_command, checkpoint_path, state.wandb_run_id
                    )
                    runner._log_wrapper(
                        log_file,
                        f"resuming from checkpoint: {checkpoint_path} (resume #{resume_count})",
                    )
                else:
                    command = original_command
                    runner._log_wrapper(
                        log_file,
                        f"no valid checkpoint under {run_dir}; restarting original command",
                    )
            elif attempt > 0:
                command = original_command
                runner._log_wrapper(
                    log_file,
                    "run name is unknown; restarting original command",
                )

            return_code = runner.run_once(command, state, log_file)
            if return_code == 0:
                runner._log_wrapper(log_file, f"total resume count: {resume_count}")
                return 0
            if runner.stop_requested:
                runner._log_wrapper(log_file, f"total resume count: {resume_count}")
                return runner.exit_code_after_signal()

            attempt += 1
            runner.sleep_before_restart(log_file)
            if runner.stop_requested:
                runner._log_wrapper(log_file, f"total resume count: {resume_count}")
                return runner.exit_code_after_signal()


if __name__ == "__main__":
    raise SystemExit(main())
