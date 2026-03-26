from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class NavPhase(str, Enum):
    IDLE = "idle"
    SEARCHING = "searching"
    TRACKING = "tracking"
    SUCCESS = "success"
    BLOCKED = "blocked"
    NOT_FOUND = "not_found"
    CANCELLED = "cancelled"


@dataclass
class TargetHint:
    label: str
    rgb_range: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None
    min_pixels: int = 150
    bbox: tuple[int, int, int, int] | None = None
    point_xy: tuple[int, int] | None = None


@dataclass
class Detection:
    found: bool
    confidence: float = 0.0
    center_px: tuple[int, int] | None = None
    bbox_xyxy: tuple[int, int, int, int] | None = None
    distance_m: float | None = None
    position_robot_m: tuple[float, float, float] | None = None
    area_pixels: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    rgb: np.ndarray | None = None
    depth_m: np.ndarray | None = None
    occupancy: np.ndarray | None = None
    pose_xy_yaw: tuple[float, float, float] = (0.0, 0.0, 0.0)
    timestamp: float = 0.0


@dataclass
class NavigationConfig:
    success_distance_m: float = 0.8
    success_heading_deg: float = 10.0
    obstacle_margin_m: float = 0.45
    max_search_turns: int = 8
    search_turn_deg: float = 45.0
    target_relocation_m: float = 0.75
    max_steps: int = 64
    camera_fx: float = 525.0
    camera_fy: float = 525.0
    camera_cx: float | None = None
    camera_cy: float | None = None
    action_forward_m: float = 0.35
    action_turn_deg: float = 30.0
    occupancy_resolution_m: float = 0.10
    monocular_reference_area_px: float = 1600.0
    monocular_reference_distance_m: float = 1.2
    monocular_min_distance_m: float = 0.35
    monocular_max_distance_m: float = 4.0


@dataclass
class NavigationState:
    target_label: str | None = None
    phase: NavPhase = NavPhase.IDLE
    message: str = "idle"
    steps: int = 0
    search_turns_completed: int = 0
    last_detection: Detection | None = None
    target_world_xy: tuple[float, float] | None = None
    closest_reachable_xy: tuple[float, float] | None = None
    closest_reachable_distance_m: float | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_label": self.target_label,
            "phase": self.phase.value,
            "message": self.message,
            "steps": self.steps,
            "search_turns_completed": self.search_turns_completed,
            "target_world_xy": self.target_world_xy,
            "closest_reachable_xy": self.closest_reachable_xy,
            "closest_reachable_distance_m": self.closest_reachable_distance_m,
            "last_detection": None
            if self.last_detection is None
            else {
                "found": self.last_detection.found,
                "confidence": self.last_detection.confidence,
                "center_px": self.last_detection.center_px,
                "bbox_xyxy": self.last_detection.bbox_xyxy,
                "distance_m": self.last_detection.distance_m,
                "position_robot_m": self.last_detection.position_robot_m,
                "area_pixels": self.last_detection.area_pixels,
                "metadata": self.last_detection.metadata,
            },
            "history_tail": self.history[-10:],
        }
