"""Tầng một — dữ liệu, địa hình và mô phỏng."""

from .clearance import (
    AccidentData,
    ClearanceData,
    simulate_accidents,
    simulate_clearance,
)
from .dataset import (
    detections_to_grid,
    read_ground_truth,
    read_tile_metadata,
    write_dataset,
)
from .geo import Grid
from .imagery import Tile, build_tiles
from .sorties import Scene, simulate_scene
from .terrain import Terrain, build_terrain

__all__ = [
    "AccidentData",
    "ClearanceData",
    "simulate_accidents",
    "simulate_clearance",
    "detections_to_grid",
    "read_ground_truth",
    "read_tile_metadata",
    "write_dataset",
    "Grid",
    "Tile",
    "build_tiles",
    "Scene",
    "simulate_scene",
    "Terrain",
    "build_terrain",
]
