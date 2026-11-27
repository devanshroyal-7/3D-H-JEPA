"""JEPA4R-style per-level point-cloud decoders (PAGE-4D trunk) for Point-H-JEPA latents."""

from point_hjepa.models.pc_decoder.hierarchical_decoder import HierarchicalPointDecoder, LevelHead
from point_hjepa.models.pc_decoder.losses import chamfer, nn_distance, nn_point_errors, sliced_wasserstein
from point_hjepa.models.pc_decoder.set_generator import FiLMBlock, SetDecoder

__all__ = [
    "FiLMBlock",
    "HierarchicalPointDecoder",
    "LevelHead",
    "SetDecoder",
    "chamfer",
    "nn_distance",
    "nn_point_errors",
    "sliced_wasserstein",
]
