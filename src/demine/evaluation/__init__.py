"""Khung kiểm chứng bốn tầng."""

from .calibration import calibration_report, reliability_curve
from .consistency import check_consistency
from .detection_metrics import evaluate_detections
from .prioritisation import (
    accident_coverage,
    clearance_efficiency_curve,
    normalised_rank,
    priority_index,
)
from .spatial_cv import block_kfold, make_spatial_blocks, spatial_holdout, transfer_split

__all__ = [
    "calibration_report",
    "reliability_curve",
    "check_consistency",
    "evaluate_detections",
    "accident_coverage",
    "clearance_efficiency_curve",
    "normalised_rank",
    "priority_index",
    "block_kfold",
    "make_spatial_blocks",
    "spatial_holdout",
    "transfer_split",
]
