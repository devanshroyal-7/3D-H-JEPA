"""Published JEPA4R OGBench Table 1 baselines and Markdown comparison formatting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

BASELINE_METHODS: tuple[str, ...] = (
    "state",
    "jepa4r",
    "lewm",
    "pldm",
    "lejepa",
    "visreg",
    "vjepa21",
)

# Exact published numbers from third_party/JEPA4R/paper/expected/tab1_ogbench.yaml
# Success (%) of downstream frozen-latent policies across OGBench tasks.
TAB1_OGBENCH_REFERENCE: dict[str, dict[str, float]] = {
    "pldm": {
        "cube_single": 46.8,
        "cube_double": 3.6,
        "cube_triple": 4.8,
        "scene": 46.1,
        "puzzle_3x3": 100.0,
        "puzzle_4x4": 95.6,
        "puzzle_4x5": 21.6,
        "cube_quadruple": 0.0,
    },
    "lejepa": {
        "cube_single": 0.0,
        "cube_double": 0.0,
        "cube_triple": 0.0,
        "scene": 41.1,
        "puzzle_3x3": 96.8,
        "puzzle_4x4": 99.2,
        "puzzle_4x5": 96.4,
        "cube_quadruple": 0.0,
    },
    "visreg": {
        "cube_single": 1.2,
        "cube_double": 0.0,
        "cube_triple": 0.0,
        "scene": 40.5,
        "puzzle_3x3": 1.2,
        "puzzle_4x4": 0.4,
        "puzzle_4x5": 0.0,
        "cube_quadruple": 0.0,
    },
    "vjepa21": {
        "cube_single": 6.0,
        "cube_double": 0.4,
        "cube_triple": 0.0,
        "scene": 38.4,
        "puzzle_3x3": 100.0,
        "puzzle_4x4": 99.6,
        "puzzle_4x5": 32.8,
        "cube_quadruple": 0.0,
    },
    "lewm": {
        "cube_single": 24.4,
        "cube_double": 1.2,
        "cube_triple": 0.0,
        "scene": 40.8,
        "puzzle_3x3": 100.0,
        "puzzle_4x4": 99.6,
        "puzzle_4x5": 26.4,
        "cube_quadruple": 0.0,
    },
    "jepa4r": {
        "cube_single": 78.4,
        "cube_double": 41.6,
        "cube_triple": 18.4,
        "scene": 64.8,
        "puzzle_3x3": 100.0,
        "puzzle_4x4": 95.6,
        "puzzle_4x5": 68.0,
        "cube_quadruple": 0.0,
    },
}


def format_comparison_table(
    eval_results: Mapping[str, Mapping[str, float]] | Sequence[Mapping[str, Any]],
    tasks: Sequence[str],
    *,
    include_reference: bool = True,
) -> str:
    """Format evaluated methods alongside JEPA4R Table 1 published baselines as Markdown."""
    rows: list[tuple[str, Mapping[str, float]]] = []
    if isinstance(eval_results, Mapping):
        for label, scores in eval_results.items():
            rows.append((str(label), scores))
    else:
        for item in eval_results:
            label = str(item.get("label") or item.get("method") or "run")
            scores = item.get("results_by_task") or item.get("scores") or {}
            rows.append((label, scores))

    if include_reference:
        for ref_name in ("jepa4r", "pldm", "lewm", "lejepa", "vjepa21", "visreg"):
            rows.append((f"[ref] {ref_name}", TAB1_OGBENCH_REFERENCE[ref_name]))

    task_list = list(tasks)
    header = ["Method", *task_list, "Mean", "Delta vs JEPA4R"]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |",
    ]

    jepa_ref = TAB1_OGBENCH_REFERENCE["jepa4r"]
    ref_vals = [jepa_ref[t] for t in task_list if t in jepa_ref]
    ref_mean = sum(ref_vals) / len(ref_vals) if ref_vals else None

    for label, scores in rows:
        cells = [label]
        vals: list[float] = []
        for t in task_list:
            if t in scores and scores[t] is not None:
                v = float(scores[t])
                vals.append(v)
                cells.append(f"{v:.1f}")
            else:
                cells.append("—")
        if vals:
            mean_v = sum(vals) / len(vals)
            cells.append(f"{mean_v:.1f}")
            if ref_mean is not None and len(vals) == len(ref_vals):
                delta = mean_v - ref_mean
                cells.append(f"{delta:+.1f}")
            else:
                cells.append("—")
        else:
            cells.extend(["—", "—"])
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)
