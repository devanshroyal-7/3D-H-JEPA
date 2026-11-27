"""OGBench evaluation harness, baseline encoders, and frozen-latent downstream policies."""

from point_hjepa.eval.baselines import (
    BASELINE_METHODS,
    TAB1_OGBENCH_REFERENCE,
    format_comparison_table,
)
from point_hjepa.eval.encoders import (
    FrozenObsEncoder,
    JEPA4RBaselineEncoder,
    PointHJEPAEncoder,
    StateOnlyEncoder,
    build_encoder,
    load_frozen_world_model,
)
from point_hjepa.eval.ogbench_harness import (
    cache_ogbench_features,
    run_ogbench_evaluation,
)
from point_hjepa.eval.ogbench_policies import (
    FLAREBCConfig,
    GCIQLConfig,
    GCIQLGoalSampler,
    TorchFLAREBCPolicy,
    TorchGCIQLPolicy,
    build_flare_features,
    evaluate_gciql_online,
    evaluate_scene_online,
    train_flare_bc,
    train_gciql,
)

__all__ = [
    "BASELINE_METHODS",
    "TAB1_OGBENCH_REFERENCE",
    "format_comparison_table",
    "FrozenObsEncoder",
    "PointHJEPAEncoder",
    "StateOnlyEncoder",
    "JEPA4RBaselineEncoder",
    "build_encoder",
    "load_frozen_world_model",
    "GCIQLConfig",
    "GCIQLGoalSampler",
    "TorchGCIQLPolicy",
    "FLAREBCConfig",
    "TorchFLAREBCPolicy",
    "build_flare_features",
    "train_gciql",
    "evaluate_gciql_online",
    "train_flare_bc",
    "evaluate_scene_online",
    "cache_ogbench_features",
    "run_ogbench_evaluation",
]
