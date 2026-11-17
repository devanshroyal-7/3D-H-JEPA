"""Motion-planner backends. SRMP is the default; mplib is optional."""

from __future__ import annotations

from point_hjepa.data.planners.base import MotionPlanner, PlanResult


def create_planner(name: str) -> MotionPlanner:
    key = (name or "srmp").lower()
    if key == "srmp":
        from point_hjepa.data.planners.srmp_backend import SrmpPlanner

        return SrmpPlanner()
    if key == "mplib":
        from point_hjepa.data.planners.mplib_backend import MplibPlanner

        return MplibPlanner()
    raise ValueError(f"Unknown planner {name!r}. Choose 'srmp' (default) or 'mplib'.")


__all__ = ["MotionPlanner", "PlanResult", "create_planner"]
