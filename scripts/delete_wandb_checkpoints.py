#!/usr/bin/env python3
"""List or delete checkpoint files stored as W&B run files.

The training scripts in this repo mostly upload checkpoints with
``wandb.save(path)``. Those files appear under each run's Files tab, commonly as
``models/<run-name>/actor_chkpt_<step>.pt``.

By default this script only prints a dry-run. Add ``--delete`` to actually
delete the selected files.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable, List, Optional, Tuple
from urllib.parse import urlparse


DEFAULT_PREFIXES = ("actor_chkpt", "student_chkpt")
NAMED_SUFFIXES = {
    "last",
    "best_test_loss",
    "best_success_rate",
    "best_val_action_mse_error",
}
ARCHIVED_NAMED_PREFIXES = (
    "latest",
    "best_test_loss",
    "best_success_rate",
    "best_val_action_mse_error",
)


@dataclass(frozen=True)
class Candidate:
    run_path: str
    run_name: str
    file_name: str
    size: int
    prefix: str
    suffix: str
    step: Optional[int]
    updated_at: object
    file_obj: object

    @property
    def is_named(self) -> bool:
        return self.step is None and self.suffix in NAMED_SUFFIXES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run or delete W&B checkpoint files uploaded with wandb.save(). "
            "Use --delete to actually remove files."
        )
    )
    parser.add_argument(
        "project",
        help=(
            "W&B project path, either <entity>/<project>, <entity>/<project>/runs, "
            "or a full W&B run/project URL."
        ),
    )
    parser.add_argument(
        "--run",
        action="append",
        dest="runs",
        default=[],
        help=(
            "Run id or full run path/URL to scan. Can be repeated. "
            "If omitted, scans all project runs."
        ),
    )
    parser.add_argument(
        "--run-regex",
        help="Only scan runs whose id or display name matches this regex.",
    )
    parser.add_argument(
        "--state",
        help="Only scan runs in this state, for example finished, crashed, or failed.",
    )
    parser.add_argument(
        "--pattern",
        default="%chkpt%.pt",
        help=(
            "W&B Run.files() LIKE pattern. Default: %(default)s. "
            "Use %%actor_chkpt%%.pt or %%student_chkpt%%.pt to narrow."
        ),
    )
    parser.add_argument(
        "--prefix",
        action="append",
        dest="prefixes",
        default=[],
        help=(
            "Checkpoint filename prefix to match. Can be repeated. "
            f"Default: {', '.join(DEFAULT_PREFIXES)}."
        ),
    )
    parser.add_argument(
        "--keep-latest-n",
        type=int,
        default=1,
        help="Keep this many newest numbered checkpoints per run and prefix.",
    )
    parser.add_argument(
        "--keep-step-multiple",
        type=int,
        default=100,
        help=(
            "Keep numbered checkpoints whose step is a positive multiple of this "
            "value, per run and prefix. Set to 0 to disable. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--include-named",
        action="store_true",
        help="Also delete named checkpoints such as actor_chkpt_best_success_rate.pt.",
    )
    parser.add_argument(
        "--min-step",
        type=int,
        help="Only delete numbered checkpoints with step >= this value.",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        help="Only delete numbered checkpoints with step <= this value.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Stop after selecting this many files for deletion.",
    )
    parser.add_argument(
        "--max-delete-count",
        type=int,
        default=500,
        help=(
            "Safety guard for --delete. Refuse to delete more than this many "
            "files unless set to 0. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete selected files. Without this flag, only prints a dry-run.",
    )
    return parser.parse_args()


def load_wandb_module():
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The wandb package is not installed in this Python environment. "
            "Install it or run this script from the training environment."
        ) from exc
    return wandb


def normalize_wandb_path(path: str) -> Tuple[str, List[str]]:
    """Return a W&B API path and any run ids embedded in an app URL/path."""
    path = path.strip().rstrip("/")
    if "://" in path:
        path = urlparse(path).path.strip("/")

    parts = [part for part in path.split("/") if part]
    if parts and parts[0] in {"app", "wandb.ai"}:
        parts = parts[1:]

    if "runs" in parts:
        runs_index = parts.index("runs")
        project = "/".join(parts[:runs_index])
        embedded_runs = parts[runs_index + 1 :]
        return project, embedded_runs

    return "/".join(parts), []


def normalize_project_arg(project: str, run_ids: List[str]) -> Tuple[str, List[str]]:
    project_path, embedded_runs = normalize_wandb_path(project)
    return project_path, embedded_runs + list(run_ids)


def normalize_run_arg(project: str, run_id: str) -> str:
    run_path, embedded_runs = normalize_wandb_path(run_id)
    if embedded_runs:
        return f"{run_path}/{embedded_runs[0]}"
    if "/" in run_path:
        return run_path
    return f"{project}/{run_path}"


def checkpoint_regex(prefixes: Iterable[str]) -> re.Pattern:
    prefix_alt = "|".join(re.escape(prefix) for prefix in prefixes)
    archive_alt = "|".join(re.escape(prefix) for prefix in ARCHIVED_NAMED_PREFIXES)
    named_alt = "|".join(re.escape(suffix) for suffix in NAMED_SUFFIXES)
    return re.compile(
        rf"^(?P<prefix>{prefix_alt})_(?:(?P<named>{named_alt})|"
        rf"(?:(?P<archive>{archive_alt})_)?(?P<step>\d+))\.pt$"
    )


def run_path(run: object) -> str:
    entity = getattr(run, "entity", None)
    project = getattr(run, "project", None)
    run_id = getattr(run, "id", None)
    if entity and project and run_id:
        return f"{entity}/{project}/{run_id}"
    if project and run_id:
        return f"{project}/{run_id}"
    return str(run)


def iter_runs(api: object, project: str, run_ids: List[str]) -> Iterable[object]:
    if run_ids:
        for run_id in run_ids:
            run_ref = normalize_run_arg(project, run_id)
            print(f"loading run {run_ref}", file=sys.stderr, flush=True)
            try:
                yield api.run(run_ref)
            except Exception as exc:
                print(f"warning: could not load run {run_ref}: {exc}", file=sys.stderr)
        return

    print(f"loading runs from project {project}", file=sys.stderr, flush=True)
    yield from api.runs(project)


def collect_candidates(args: argparse.Namespace) -> List[Candidate]:
    wandb = load_wandb_module()
    api = wandb.Api()
    prefixes = args.prefixes or list(DEFAULT_PREFIXES)
    file_re = checkpoint_regex(prefixes)
    run_re = re.compile(args.run_regex) if args.run_regex else None
    candidates = []

    for run in iter_runs(api, args.project, args.runs):
        if args.state and getattr(run, "state", None) != args.state:
            continue

        rid = getattr(run, "id", "")
        rname = getattr(run, "name", "") or ""
        if run_re and not (run_re.search(rid) or run_re.search(rname)):
            continue

        before_count = len(candidates)
        current_run_path = run_path(run)
        print(
            f"scanning files for {current_run_path} ({rname or rid})",
            file=sys.stderr,
            flush=True,
        )
        for file_obj in run.files(pattern=args.pattern):
            file_name = file_obj.name
            base_name = PurePosixPath(file_name).name
            match = file_re.match(base_name)
            if not match:
                continue

            step_text = match.group("step")
            step = int(step_text) if step_text is not None else None
            if step is not None:
                if args.min_step is not None and step < args.min_step:
                    continue
                if args.max_step is not None and step > args.max_step:
                    continue

            suffix = match.group("named") or match.group("archive") or "periodic"
            candidates.append(
                Candidate(
                    run_path=current_run_path,
                    run_name=rname,
                    file_name=file_name,
                    size=int(getattr(file_obj, "size", 0) or 0),
                    prefix=match.group("prefix"),
                    suffix=suffix,
                    step=step,
                    updated_at=getattr(file_obj, "updated_at", None),
                    file_obj=file_obj,
                )
            )

        print(
            f"matched {len(candidates) - before_count} checkpoint file(s) "
            f"in {current_run_path}",
            file=sys.stderr,
            flush=True,
        )

    return candidates


def choose_deletions(
    candidates: List[Candidate], args: argparse.Namespace
) -> Tuple[List[Candidate], List[Candidate]]:
    kept = []
    selected = []
    numbered_by_run_prefix = {}

    for candidate in candidates:
        if candidate.is_named and not args.include_named:
            kept.append(candidate)
            continue
        if candidate.step is None:
            selected.append(candidate)
            continue

        key = (candidate.run_path, candidate.prefix)
        numbered_by_run_prefix.setdefault(key, []).append(candidate)

    for group in numbered_by_run_prefix.values():
        group.sort(
            key=lambda item: (
                item.step if item.step is not None else -1,
                str(item.updated_at or ""),
                item.file_name,
            ),
            reverse=True,
        )

        latest_pool = []
        for item in group:
            if (
                args.keep_step_multiple > 0
                and item.step is not None
                and item.step > 0
                and item.step % args.keep_step_multiple == 0
            ):
                kept.append(item)
            else:
                latest_pool.append(item)

        keep_count = max(args.keep_latest_n, 0)
        kept.extend(latest_pool[:keep_count])
        selected.extend(latest_pool[keep_count:])

    selected.sort(key=lambda item: (item.run_path, item.file_name))
    if args.limit is not None:
        kept.extend(selected[args.limit :])
        selected = selected[: args.limit]
    return selected, kept


def format_size(size: int) -> str:
    if size <= 0:
        return "unknown"
    gib = size / 1024 / 1024 / 1024
    if gib >= 1:
        return f"{gib:.2f} GiB"
    return f"{size / 1024 / 1024:.1f} MiB"


def print_plan(selected: List[Candidate], kept: List[Candidate], delete: bool) -> None:
    selected_size = sum(item.size for item in selected)
    kept_size = sum(item.size for item in kept)
    mode = "DELETE" if delete else "DRY-RUN"
    print(f"[{mode}] selected {len(selected)} file(s), {format_size(selected_size)}")
    print(f"[{mode}] kept {len(kept)} matched file(s), {format_size(kept_size)}")

    for item in selected:
        print(
            f"{'delete' if delete else 'would delete'}\t"
            f"{format_size(item.size)}\t{item.run_path}\t{item.file_name}"
        )


def validate_args(args: argparse.Namespace) -> None:
    if args.keep_latest_n < 0:
        raise SystemExit("--keep-latest-n must be non-negative")
    if args.keep_step_multiple < 0:
        raise SystemExit("--keep-step-multiple must be non-negative")
    if args.limit is not None and args.limit < 0:
        raise SystemExit("--limit must be non-negative")
    if args.max_delete_count < 0:
        raise SystemExit("--max-delete-count must be non-negative")
    if not args.project:
        raise SystemExit("project path is empty")


def main() -> int:
    args = parse_args()
    args.project, args.runs = normalize_project_arg(args.project, args.runs)
    validate_args(args)

    candidates = collect_candidates(args)
    selected, kept = choose_deletions(candidates, args)
    print_plan(selected, kept, args.delete)

    if not args.delete:
        print("No files were deleted. Re-run with --delete to apply this plan.")
        return 0

    if args.max_delete_count and len(selected) > args.max_delete_count:
        raise SystemExit(
            f"Refusing to delete {len(selected)} files because --max-delete-count "
            f"is {args.max_delete_count}. Increase it or set it to 0."
        )

    for index, item in enumerate(selected, start=1):
        print(
            f"deleting {index}/{len(selected)}: {item.run_path} {item.file_name}",
            file=sys.stderr,
            flush=True,
        )
        item.file_obj.delete()

    print(f"Deleted {len(selected)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
