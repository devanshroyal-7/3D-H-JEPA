"""Fetch AIGeeksGroup/JEPA4R code and (optionally) OGBench baseline checkpoints.

- Clones https://github.com/AIGeeksGroup/JEPA4R into ``third_party/JEPA4R``.
- Optionally pins a commit with ``--commit``.
- Downloads per-task baseline checkpoints from the Hugging Face model repo
  ``AIGeeksGroup/JEPA4R`` (revision ``TASK_HF_BRANCHES[task]``) into
  ``checkpoints/jepa4r/<task>/``.

Usage:
    python scripts/setup_jepa4r.py --skip-weights
    python scripts/setup_jepa4r.py --tasks cube_single scene
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

JEPA4R_REPO = "https://github.com/AIGeeksGroup/JEPA4R"
JEPA4R_HF_REPO = "AIGeeksGroup/JEPA4R"

TASK_HF_BRANCHES: dict[str, str] = {
    "cube_single": "ogbench-cube-single",
    "cube_double": "ogbench-cube-double",
    "cube_triple": "ogbench-cube-triple",
    "cube_quadruple": "ogbench-cube-quadruple",
    "scene": "ogbench-scene",
    "puzzle_3x3": "ogbench-puzzle-3x3",
    "puzzle_4x4": "ogbench-puzzle-4x4",
    "puzzle_4x5": "ogbench-puzzle-4x5",
}


def resolve_branch(task: str) -> str:
    """Return the Hugging Face branch name for an OGBench task."""
    key = task.strip().lower().replace("-", "_")
    if key not in TASK_HF_BRANCHES:
        raise ValueError(
            f"Unknown OGBench task {task!r}. Expected one of {sorted(TASK_HF_BRANCHES)}"
        )
    return TASK_HF_BRANCHES[key]


def setup_code(dest: Path, commit: str | None = None) -> Path:
    """Clone (if needed) and optionally check out a commit in ``third_party/JEPA4R``."""
    if not (dest / ".git").is_dir():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", JEPA4R_REPO, str(dest)], check=True)
    if commit:
        subprocess.run(["git", "-C", str(dest), "checkout", "-q", commit], check=True)
    rev = subprocess.check_output(
        ["git", "-C", str(dest), "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()
    print(f"[jepa4r] code at {dest} @ {rev}")
    return dest


def setup_weights(dest_root: Path, tasks: list[str]) -> list[Path]:
    """Download baseline checkpoints from ``AIGeeksGroup/JEPA4R`` per task branch."""
    from huggingface_hub import snapshot_download

    downloaded: list[Path] = []
    for task in tasks:
        branch = resolve_branch(task)
        task_dir = dest_root / task.strip().lower().replace("-", "_")
        task_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=JEPA4R_HF_REPO,
            repo_type="model",
            revision=branch,
            local_dir=str(task_dir),
        )
        print(f"[jepa4r] weights for {task} ({branch}) at {task_dir}")
        downloaded.append(task_dir)
    return downloaded


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument(
        "--tasks",
        nargs="+",
        default=["cube_single", "scene"],
        help=f"OGBench tasks to fetch weights for ({', '.join(sorted(TASK_HF_BRANCHES))})",
    )
    ap.add_argument("--commit", type=str, default=None, help="Optional git commit/tag to checkout")
    ap.add_argument("--skip-code", action="store_true")
    ap.add_argument("--skip-weights", action="store_true")
    args = ap.parse_args(argv)

    if not args.skip_code:
        setup_code(args.root / "third_party" / "JEPA4R", commit=args.commit)
    if not args.skip_weights:
        setup_weights(args.root / "checkpoints" / "jepa4r", args.tasks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
