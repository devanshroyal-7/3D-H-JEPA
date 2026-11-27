"""Unified OGBench evaluation and JEPA4R baseline comparison CLI.

Supports:
- Printing published JEPA4R Table 1 reference baselines (``--print-baselines``)
- Dry-run inspection of planned evaluations and cache paths (``--dry-run``)
- Evaluating frozen ``PointHJEPA`` checkpoints (unsegmented 4,096-pt by default,
  across ``l1``, ``l2``, ``l3``, ``concat``, or ``concat_12`` latent modes)
- Evaluating JEPA4R paper baselines (``state``, ``jepa4r``, ``lewm``, ``pldm``,
  ``lejepa``, ``visreg``, ``vjepa21``, or ``all_baselines``) via either our
  PyTorch downstream harness (``--backend torch``) or JEPA4R's native scripts
  (``--backend jepa4r_native``).

Examples:
    python scripts/eval_ogbench.py --print-baselines
    python scripts/eval_ogbench.py --dry-run --methods point_hjepa jepa4r lewm state --tasks cube_single scene
    python scripts/eval_ogbench.py --tasks cube_single scene --methods point_hjepa --wm-ckpt outputs/runs/ogbench_cube_single_l2/checkpoint_latest.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow running directly as `python scripts/eval_ogbench.py` without PYTHONPATH=.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from point_hjepa.data.collect_ogbench_6d import OGBENCH_TASKS, normalize_ogbench_task
from point_hjepa.eval.baselines import (
    BASELINE_METHODS,
    TAB1_OGBENCH_REFERENCE,
    format_comparison_table,
)
from point_hjepa.eval.ogbench_harness import _encoder_cache_tag, run_ogbench_evaluation
from scripts.setup_jepa4r import TASK_HF_BRANCHES

METHOD_CHOICES = ("point_hjepa", *BASELINE_METHODS, "all_baselines")
LATENT_MODES = ("l1", "l2", "l3", "concat", "concat_12")


def _expand_methods(raw_methods: list[str]) -> list[str]:
    out: list[str] = []
    for m in raw_methods:
        if m == "all_baselines":
            for b in BASELINE_METHODS:
                if b not in out:
                    out.append(b)
        elif m not in out:
            out.append(m)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--tasks",
        nargs="+",
        default=["cube_single", "scene"],
        help=f"OGBench tasks ({', '.join(sorted(OGBENCH_TASKS))})",
    )
    ap.add_argument(
        "--methods",
        nargs="+",
        default=["point_hjepa"],
        choices=METHOD_CHOICES,
        help="Methods to evaluate or compare",
    )
    ap.add_argument("--wm-ckpt", type=Path, default=None, help="Path to Stage-1 PointHJEPA checkpoint")
    ap.add_argument("--latent-mode", choices=LATENT_MODES, default="concat")

    seg_group = ap.add_mutually_exclusive_group()
    seg_group.add_argument(
        "--unsegmented",
        dest="unsegmented",
        action="store_true",
        default=True,
        help="Use unsegmented full-scene point clouds for Point-H-JEPA (default)",
    )
    seg_group.add_argument(
        "--segmented",
        dest="unsegmented",
        action="store_false",
        help="Strip table/floor background (matching JEPA4R preprocessing)",
    )
    ap.add_argument("--num-points", type=int, default=None, help="Points per cloud (default: 4096 unseg, 1200 seg)")
    ap.add_argument("--baseline-ckpt", type=Path, default=None, help="Optional explicit baseline checkpoint dir/file")
    ap.add_argument("--jepa4r-root", type=Path, default=_REPO_ROOT / "third_party" / "JEPA4R")
    ap.add_argument("--checkpoints-root", type=Path, default=_REPO_ROOT / "checkpoints" / "jepa4r")
    ap.add_argument("--backend", choices=("torch", "jepa4r_native"), default="torch")
    ap.add_argument("--train-steps", type=int, default=1_000_000, help="GCIQL training updates (Cube tasks)")
    ap.add_argument("--bc-epochs", type=int, default=100, help="FLARE-BC training epochs (Scene task)")
    ap.add_argument("--episodes-per-task", type=int, default=None, help="Online eval episodes per task")
    ap.add_argument("--cache-dir", type=Path, default=_REPO_ROOT / "outputs" / "ogbench_cache")
    ap.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "outputs" / "ogbench_eval")
    ap.add_argument("--print-baselines", action="store_true", help="Print JEPA4R Table 1 reference baselines and exit")
    ap.add_argument("--dry-run", action="store_true", help="Validate config, print planned runs & reference table, exit")
    ap.add_argument("--wandb", action="store_true", help="Log comparison results to W&B")
    return ap.parse_args(argv)


def _print_reference_section(tasks: list[str]) -> str:
    table = format_comparison_table({}, tasks, include_reference=True)
    lines = [
        "=== JEPA4R Published OGBench Table 1 Reference (Success %) ===",
        table,
        "",
        "=== Hugging Face Checkpoint Branches (AIGeeksGroup/JEPA4R) ===",
    ]
    for t in tasks:
        branch = TASK_HF_BRANCHES.get(t, "N/A")
        lines.append(f"  - {t}: {branch} (policy={OGBENCH_TASKS[t]['policy_type']}, action_dim={OGBENCH_TASKS[t]['action_dim']})")
    text = "\n".join(lines)
    print(text)
    return text


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tasks = [normalize_ogbench_task(t) for t in args.tasks]
    methods = _expand_methods(list(args.methods))

    if args.print_baselines:
        all_tasks = list(OGBENCH_TASKS.keys()) if args.tasks == ["cube_single", "scene"] else tasks
        _print_reference_section(all_tasks)
        return 0

    if args.dry_run:
        print("=== OGBench Evaluation Plan (Dry Run) ===")
        print(f"Tasks        : {tasks}")
        print(f"Methods      : {methods}")
        print(f"Backend      : {args.backend}")
        print(f"Point clouds : unsegmented={args.unsegmented}, num_points={args.num_points or (4096 if args.unsegmented else 1200)}")
        print(f"Cache dir    : {args.cache_dir}")
        print(f"Output dir   : {args.out_dir}")
        print("\nPlanned jobs:")
        for t in tasks:
            spec = OGBENCH_TASKS[t]
            for m in methods:
                pts = args.num_points or (4096 if (m == "point_hjepa" and args.unsegmented) else 1200)
                tag = _encoder_cache_tag(m, latent_mode=args.latent_mode, unsegmented=args.unsegmented, num_points=pts)
                ckpt_info = str(args.wm_ckpt) if m == "point_hjepa" else str(args.baseline_ckpt or (args.checkpoints_root / t / m))
                ref = TAB1_OGBENCH_REFERENCE.get(m, {}).get(t)
                ref_str = f" | paper_ref={ref:.1f}%" if ref is not None else ""
                print(
                    f"  - task={t} method={m} policy={spec['policy_type']} "
                    f"cache={args.cache_dir / t / tag} ckpt={ckpt_info}{ref_str}"
                )
        print()
        _print_reference_section(tasks)
        return 0

    if "point_hjepa" in methods and args.wm_ckpt is None and args.backend == "torch":
        raise ValueError("--wm-ckpt is required when evaluating method 'point_hjepa'")

    results_by_method: dict[str, dict[str, float]] = {}
    detailed: list[dict[str, Any]] = []
    for m in methods:
        label = f"point_hjepa ({args.latent_mode}, {'unseg' if args.unsegmented else 'seg'})" if m == "point_hjepa" else m
        results_by_method[label] = {}
        for t in tasks:
            res = run_ogbench_evaluation(
                t,
                m,
                wm_ckpt=args.wm_ckpt,
                latent_mode=args.latent_mode,
                unsegmented=args.unsegmented,
                num_points=args.num_points,
                baseline_ckpt=args.baseline_ckpt,
                jepa4r_root=args.jepa4r_root,
                checkpoints_root=args.checkpoints_root,
                cache_dir=args.cache_dir,
                out_dir=args.out_dir,
                backend=args.backend,
                train_steps=args.train_steps,
                bc_epochs=args.bc_epochs,
                episodes_per_task=args.episodes_per_task,
            )
            detailed.append(res)
            if "success" in res and res["success"] is not None:
                results_by_method[label][t] = float(res["success"])

    table_md = format_comparison_table(results_by_method, tasks, include_reference=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "tasks": tasks,
        "methods": methods,
        "backend": args.backend,
        "unsegmented": args.unsegmented,
        "latent_mode": args.latent_mode,
        "results": results_by_method,
        "detailed": detailed,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2) + "\n")
    (args.out_dir / "comparison.md").write_text(table_md + "\n")
    print(table_md)

    if args.wandb:
        import wandb

        run = wandb.init(
            entity="djonnala-carnegie-mellon-university",
            project="3D-H-JEPA",
            job_type="ogbench_eval",
            config={
                "tasks": tasks,
                "methods": methods,
                "backend": args.backend,
                "unsegmented": args.unsegmented,
                "latent_mode": args.latent_mode,
                "wm_ckpt": str(args.wm_ckpt) if args.wm_ckpt else None,
            },
        )
        log_dict: dict[str, Any] = {}
        for label, scores in results_by_method.items():
            safe = label.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
            for t, val in scores.items():
                log_dict[f"ogbench/{safe}/{t}"] = val
            if scores:
                log_dict[f"ogbench/{safe}/mean"] = sum(scores.values()) / len(scores)
        run.log(log_dict)
        run.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
