"""Summarize progress of sharded collection runs: episodes, steps and disk size per Zarr store.

    python scripts/collection_status.py outputs/datasets/v1
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import zarr


def _dir_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/datasets/v1")
    total_gb = 0.0
    totals: dict[str, list[int]] = {}
    for store in sorted(root.glob("*/shard_*/*.zarr")):
        g = zarr.open_group(str(store), mode="r")
        eps = int(g["episode_ends"].shape[0])
        steps = int(g["episode_ends"][-1]) if eps else 0
        size = _dir_size_gb(store)
        total_gb += size
        kind = store.name.split("_", 1)[1].removesuffix(".zarr")
        t = totals.setdefault(f"{store.parts[-3]} {kind}", [0, 0])
        t[0] += eps
        t[1] += steps
        print(f"{store.relative_to(root)}: {eps} episodes, {steps} steps, {size:.2f} GB")
    print("---")
    for k, (eps, steps) in totals.items():
        print(f"{k}: {eps} episodes, {steps} steps")
    print(f"Total on disk: {total_gb:.1f} GB; free: {shutil.disk_usage(root).free / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
