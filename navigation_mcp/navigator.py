from __future__ import annotations

import heapq
import math
import time
from typing import Any

import numpy as np

from .bridge import ActionCommand, RobotBridge
from .models import (
    Detection,
    NavPhase,
    NavigationConfig,
    NavigationState,
    Observation,
    TargetHint,
)
from .perception import TargetDetector


class NavigationEngine:
    def __init__(self, bridge: RobotBridge, config: NavigationConfig | None = None):
        self.bridge = bridge
        self.config = config or NavigationConfig()
        self.detector = TargetDetector(self.config)
        self.state = NavigationState()
        self.target_hint: TargetHint | None = None
        self.injected_observation: Observation | None = None

    def set_target(
        self,
        target_label: str,
        success_distance_m: float | None = None,
        success_heading_deg: float | None = None,
        detection_hint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.state = NavigationState(
            target_label=target_label,
            phase=NavPhase.SEARCHING,
            message=f"searching for {target_label}",
        )
        if success_distance_m is not None:
            self.config.success_distance_m = success_distance_m
        if success_heading_deg is not None:
            self.config.success_heading_deg = success_heading_deg
        self.target_hint = self._build_hint(target_label, detection_hint)
        return self.state.to_dict()

    def load_observation(self, observation: Observation) -> dict[str, Any]:
        self.injected_observation = observation
        return {"ok": True, "loaded": True}

    def cancel(self) -> dict[str, Any]:
        self.bridge.stop()
        self.state.phase = NavPhase.CANCELLED
        self.state.message = "navigation cancelled"
        return self.state.to_dict()

    def get_status(self) -> dict[str, Any]:
        return self.state.to_dict()

    def step(self) -> dict[str, Any]:
        if self.target_hint is None or self.state.target_label is None:
            raise ValueError("navigation target is not set")
        if self.state.phase in {NavPhase.SUCCESS, NavPhase.BLOCKED, NavPhase.NOT_FOUND, NavPhase.CANCELLED}:
            return self.state.to_dict()

        obs = self._consume_observation()
        detection = self.detector.detect(obs, self.target_hint)
        self.state.last_detection = detection
        self.state.steps += 1

        if detection.found:
            self.state.phase = NavPhase.TRACKING
            return self._track_target(obs, detection)
        return self._search_or_fail()

    def run_until_done(self, timeout_s: float = 30.0, step_delay_s: float = 0.0) -> dict[str, Any]:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            result = self.step()
            if result["phase"] in {
                NavPhase.SUCCESS.value,
                NavPhase.BLOCKED.value,
                NavPhase.NOT_FOUND.value,
                NavPhase.CANCELLED.value,
            }:
                return result
            if self.state.steps >= self.config.max_steps:
                self.state.phase = NavPhase.BLOCKED
                self.state.message = "maximum navigation steps reached"
                break
            if step_delay_s > 0.0:
                time.sleep(step_delay_s)
        return self.state.to_dict()

    def _consume_observation(self) -> Observation:
        if self.injected_observation is not None:
            obs = self.injected_observation
            self.injected_observation = None
            return obs
        return self.bridge.get_observation()

    def _track_target(self, obs: Observation, detection: Detection) -> dict[str, Any]:
        if detection.position_robot_m is None or detection.distance_m is None:
            self.state.message = "target detected but depth unavailable"
            self.state.history.append({"event": "depth_missing"})
            return self.state.to_dict()

        pose_x, pose_y, pose_yaw = obs.pose_xy_yaw
        target_world = self._robot_point_to_world(pose_x, pose_y, pose_yaw, detection.position_robot_m)

        if self.state.target_world_xy is not None:
            relocation = np.linalg.norm(np.array(target_world) - np.array(self.state.target_world_xy))
            if relocation > self.config.target_relocation_m:
                self.state.history.append(
                    {"event": "target_relocated", "distance_m": float(relocation)}
                )
                self.state.message = f"target relocated by {relocation:.2f} m, replanning"
        self.state.target_world_xy = target_world
        angle_deg = math.degrees(math.atan2(detection.position_robot_m[1], detection.position_robot_m[0]))
        distance_source = str(detection.metadata.get("distance_source", ""))

        if distance_source == "depth" and detection.distance_m <= self.config.success_distance_m:
            if abs(angle_deg) > self.config.success_heading_deg:
                if angle_deg > 0:
                    exec_result = self.bridge.execute(
                        ActionCommand(kind="turn_left", value=min(abs(angle_deg), self.config.action_turn_deg))
                    )
                else:
                    exec_result = self.bridge.execute(
                        ActionCommand(kind="turn_right", value=min(abs(angle_deg), self.config.action_turn_deg))
                    )
                self.state.message = "within success distance, aligning to face target"
                self.state.history.append(
                    {
                        "event": "success_alignment",
                        "angle_deg": angle_deg,
                        "result": exec_result,
                    }
                )
                return self.state.to_dict()
            self.bridge.stop()
            self.state.phase = NavPhase.SUCCESS
            self.state.message = (
                f"target reached within {self.config.success_distance_m:.2f} m and aligned"
            )
            self.state.history.append(
                {"event": "success", "distance_m": detection.distance_m, "angle_deg": angle_deg}
            )
            return self.state.to_dict()
        if distance_source != "depth":
            self.state.history.append(
                {
                    "event": "tracking_without_depth_success_gate",
                    "distance_m": detection.distance_m,
                    "distance_source": distance_source or "unknown",
                }
            )

        plan = self._plan_tracking_motion(obs, detection)
        if plan is None:
            self.state.closest_reachable_xy = self._closest_reachable_xy(obs)
            self.state.closest_reachable_distance_m = None
            self.bridge.stop()
            self.state.phase = NavPhase.BLOCKED
            self.state.message = "no reachable free space toward target"
            self.state.history.append({"event": "blocked"})
            return self.state.to_dict()

        planned_command = plan["command"]
        prior_closest_xy = self.state.closest_reachable_xy
        self.state.closest_reachable_xy = plan.get("closest_world_xy")
        self.state.closest_reachable_distance_m = plan.get("closest_target_distance_m")
        if (
            plan["status"] == "closest_approach"
            and prior_closest_xy is not None
            and self._pose_near_xy(obs.pose_xy_yaw, prior_closest_xy)
        ):
            self.bridge.stop()
            self.state.phase = NavPhase.BLOCKED
            self.state.message = "reached closest reachable position but target remains obstructed"
            self.state.history.append(
                {
                    "event": "blocked",
                    "closest_target_distance_m": self.state.closest_reachable_distance_m,
                    "closest_reachable_xy": self.state.closest_reachable_xy,
                }
            )
            return self.state.to_dict()
        if plan["status"] == "blocked":
            self.bridge.stop()
            self.state.phase = NavPhase.BLOCKED
            self.state.message = "reached closest reachable position but target remains obstructed"
            self.state.history.append(
                {
                    "event": "blocked",
                    "closest_target_distance_m": self.state.closest_reachable_distance_m,
                    "closest_reachable_xy": self.state.closest_reachable_xy,
                }
            )
            return self.state.to_dict()

        exec_result = self.bridge.execute(planned_command)
        if plan["status"] == "detour_goal":
            self.state.message = "tracking target via obstacle detour"
            self.state.history.append(
                {"event": "detour_command", "command": planned_command.kind, "value": planned_command.value, "result": exec_result}
            )
        elif plan["status"] == "closest_approach":
            self.state.message = "approaching closest reachable position before blocked assessment"
            self.state.history.append(
                {
                    "event": "closest_approach_command",
                    "command": planned_command.kind,
                    "value": planned_command.value,
                    "closest_target_distance_m": self.state.closest_reachable_distance_m,
                    "result": exec_result,
                }
            )
        else:
            self.state.message = "tracking target"
            self.state.history.append({"event": "command", "result": exec_result})
        return self.state.to_dict()

    def _search_or_fail(self) -> dict[str, Any]:
        if self.state.search_turns_completed >= self.config.max_search_turns:
            self.bridge.stop()
            self.state.phase = NavPhase.NOT_FOUND
            self.state.message = "target not observed after one full clockwise scan"
            self.state.history.append({"event": "not_found"})
            return self.state.to_dict()

        exec_result = self.bridge.execute(ActionCommand(kind="turn_right", value=self.config.search_turn_deg))
        if not exec_result.get("ok", False):
            self.state.phase = NavPhase.BLOCKED
            self.state.message = "search rotation command did not produce expected odometry change"
            self.state.history.append(
                {
                    "event": "search_turn_failed",
                    "turn_deg": self.config.search_turn_deg,
                    "result": exec_result,
                }
            )
            return self.state.to_dict()
        self.state.search_turns_completed += 1
        self.state.phase = NavPhase.SEARCHING
        self.state.message = (
            f"target missing, rotating clockwise {self.config.search_turn_deg:.0f} deg"
        )
        self.state.history.append(
            {
                "event": "search_turn",
                "turn_index": self.state.search_turns_completed,
                "turn_deg": self.config.search_turn_deg,
                "result": exec_result,
            }
        )
        return self.state.to_dict()

    def _build_hint(self, target_label: str, detection_hint: dict[str, Any] | None) -> TargetHint:
        if not detection_hint:
            return TargetHint(label=target_label)
        rgb_range = detection_hint.get("rgb_range")
        bbox = detection_hint.get("bbox")
        point_xy = detection_hint.get("point_xy")
        return TargetHint(
            label=target_label,
            rgb_range=None if rgb_range is None else (tuple(rgb_range[0]), tuple(rgb_range[1])),
            min_pixels=int(detection_hint.get("min_pixels", 150)),
            bbox=None if bbox is None else tuple(int(v) for v in bbox),
            point_xy=None if point_xy is None else tuple(int(v) for v in point_xy),
        )

    def _line_of_sight_clear(self, obs: Observation, target_distance_m: float) -> bool:
        if obs.occupancy is None:
            return True
        occ = obs.occupancy
        rows, cols = occ.shape[:2]
        meters = np.arange(self.config.obstacle_margin_m, target_distance_m, self.config.occupancy_resolution_m)
        for dist in meters:
            gx = int(round(dist / self.config.occupancy_resolution_m))
            gy = rows // 2
            if 0 <= gy < rows and 0 <= gx < cols and occ[gy, gx] > 0:
                return False
        return True

    def _plan_tracking_motion(self, obs: Observation, detection: Detection) -> dict[str, Any] | None:
        if detection.position_robot_m is None or detection.distance_m is None:
            return None

        direct_command = self._direct_tracking_command(detection)
        if obs.occupancy is None or self._line_of_sight_clear(obs, detection.distance_m):
            return {"status": "direct", "command": direct_command}

        plan = self._plan_path_in_occupancy(obs.occupancy, detection)
        if plan is None:
            return None

        path = plan["path"]
        closest_cell = plan["closest_cell"]
        closest_x_m, closest_y_m = self._grid_to_robot_xy(obs.occupancy, closest_cell)
        closest_world_xy = self._robot_point_to_world(
            obs.pose_xy_yaw[0],
            obs.pose_xy_yaw[1],
            obs.pose_xy_yaw[2],
            (closest_x_m, closest_y_m, 0.0),
        )

        if len(path) <= 1:
            return {
                "status": "blocked",
                "command": ActionCommand(kind="stop", value=0.0),
                "closest_world_xy": closest_world_xy,
                "closest_target_distance_m": plan["closest_target_distance_m"],
            }

        lookahead = path[min(len(path) - 1, 3)]
        target_x_m, target_y_m = self._grid_to_robot_xy(obs.occupancy, lookahead)
        heading_deg = math.degrees(math.atan2(target_y_m, target_x_m))
        if heading_deg > 10:
            command = ActionCommand(kind="turn_left", value=min(heading_deg, self.config.action_turn_deg))
        elif heading_deg < -10:
            command = ActionCommand(kind="turn_right", value=min(-heading_deg, self.config.action_turn_deg))
        else:
            move_dist = math.hypot(target_x_m, target_y_m)
            move_dist = min(self.config.action_forward_m, move_dist)
            if move_dist <= 0.10:
                command = direct_command
            else:
                command = ActionCommand(kind="forward", value=max(0.10, move_dist))
        return {
            "status": plan["status"],
            "command": command,
            "closest_world_xy": closest_world_xy,
            "closest_target_distance_m": plan["closest_target_distance_m"],
        }

    def _direct_tracking_command(self, detection: Detection) -> ActionCommand:
        assert detection.position_robot_m is not None
        assert detection.distance_m is not None
        angle_deg = math.degrees(math.atan2(detection.position_robot_m[1], detection.position_robot_m[0]))
        if angle_deg > 10:
            return ActionCommand(kind="turn_left", value=min(angle_deg, self.config.action_turn_deg))
        if angle_deg < -10:
            return ActionCommand(kind="turn_right", value=min(-angle_deg, self.config.action_turn_deg))
        move_dist = min(self.config.action_forward_m, detection.distance_m - self.config.success_distance_m)
        return ActionCommand(kind="forward", value=max(0.10, move_dist))

    def _plan_path_in_occupancy(
        self,
        occupancy: np.ndarray,
        detection: Detection,
    ) -> dict[str, Any] | None:
        rows, cols = occupancy.shape[:2]
        start = (rows // 2, 0)
        if occupancy[start] > 0:
            return None

        target_row, target_col = self._robot_xy_to_grid(occupancy, detection.position_robot_m[0], detection.position_robot_m[1])
        goal_radius = max(1, int(round(self.config.success_distance_m / self.config.occupancy_resolution_m)))
        candidate_goals: list[tuple[int, int]] = []
        for row in range(rows):
            for col in range(cols):
                if occupancy[row, col] > 0:
                    continue
                dist_to_target = math.hypot(row - target_row, col - target_col)
                if dist_to_target > goal_radius:
                    continue
                if col > target_col + 1:
                    continue
                candidate_goals.append((row, col))

        clear_goals = {
            goal for goal in candidate_goals
            if self._goal_has_clearance(occupancy, goal)
        }

        def min_goal_distance(cell: tuple[int, int], goals: set[tuple[int, int]]) -> float:
            return min(math.hypot(cell[0] - goal[0], cell[1] - goal[1]) for goal in goals)

        frontier: list[tuple[float, float, tuple[int, int]]] = [(0.0, 0.0, start)]
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        cost_so_far: dict[tuple[int, int], float] = {start: 0.0}
        neighbors = [(-1, 0), (1, 0), (0, 1), (-1, 1), (1, 1)]
        best_cell = start
        best_distance = float("inf")

        while frontier:
            _, current_cost, current = heapq.heappop(frontier)
            if current_cost > cost_so_far[current]:
                continue
            current_distance = math.hypot(current[0] - target_row, current[1] - target_col)
            if current_distance < best_distance:
                best_distance = current_distance
                best_cell = current
            if current in clear_goals:
                return {
                    "status": "detour_goal",
                    "path": self._reconstruct_path(came_from, current),
                    "closest_cell": current,
                    "closest_target_distance_m": current_distance * self.config.occupancy_resolution_m,
                }
            for d_row, d_col in neighbors:
                nxt = (current[0] + d_row, current[1] + d_col)
                if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols):
                    continue
                if occupancy[nxt] > 0 or not self._goal_has_clearance(occupancy, nxt):
                    continue
                step_cost = math.hypot(d_row, d_col)
                new_cost = current_cost + step_cost
                if new_cost >= cost_so_far.get(nxt, float("inf")):
                    continue
                cost_so_far[nxt] = new_cost
                came_from[nxt] = current
                heuristic = 0.0 if not clear_goals else min_goal_distance(nxt, clear_goals)
                heapq.heappush(frontier, (new_cost + heuristic, new_cost, nxt))

        if best_distance == float("inf"):
            return None
        return {
            "status": "closest_approach",
            "path": self._reconstruct_path(came_from, best_cell),
            "closest_cell": best_cell,
            "closest_target_distance_m": best_distance * self.config.occupancy_resolution_m,
        }

    def _goal_has_clearance(self, occupancy: np.ndarray, cell: tuple[int, int]) -> bool:
        rows, cols = occupancy.shape[:2]
        # Use a small inflation radius for grid traversal; the larger obstacle margin
        # is still enforced separately by line-of-sight and success-threshold checks.
        clearance_cells = 1
        row, col = cell
        for rr in range(max(0, row - clearance_cells), min(rows, row + clearance_cells + 1)):
            for cc in range(max(0, col - clearance_cells), min(cols, col + clearance_cells + 1)):
                if occupancy[rr, cc] > 0:
                    return False
        return True

    def _robot_xy_to_grid(self, occupancy: np.ndarray, forward_m: float, lateral_m: float) -> tuple[int, int]:
        rows, cols = occupancy.shape[:2]
        col = int(round(forward_m / self.config.occupancy_resolution_m))
        row = int(round(lateral_m / self.config.occupancy_resolution_m)) + rows // 2
        return (int(np.clip(row, 0, rows - 1)), int(np.clip(col, 0, cols - 1)))

    def _grid_to_robot_xy(self, occupancy: np.ndarray, cell: tuple[int, int]) -> tuple[float, float]:
        rows, _ = occupancy.shape[:2]
        row, col = cell
        forward_m = col * self.config.occupancy_resolution_m
        lateral_m = (row - rows // 2) * self.config.occupancy_resolution_m
        return (forward_m, lateral_m)

    def _reconstruct_path(
        self,
        came_from: dict[tuple[int, int], tuple[int, int] | None],
        current: tuple[int, int],
    ) -> list[tuple[int, int]]:
        path = [current]
        while came_from[current] is not None:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _pose_near_xy(self, pose_xy_yaw: tuple[float, float, float], xy: tuple[float, float]) -> bool:
        pose_x, pose_y, _ = pose_xy_yaw
        return math.hypot(pose_x - xy[0], pose_y - xy[1]) <= max(0.12, self.config.action_forward_m * 0.5)

    def _closest_reachable_xy(self, obs: Observation) -> tuple[float, float]:
        pose_x, pose_y, pose_yaw = obs.pose_xy_yaw
        backoff = max(0.0, (self.state.last_detection.distance_m or 0.0) - self.config.success_distance_m)
        reach = max(0.0, min(backoff, self.config.action_forward_m))
        return (
            pose_x + math.cos(pose_yaw) * reach,
            pose_y + math.sin(pose_yaw) * reach,
        )

    def _robot_point_to_world(
        self,
        pose_x: float,
        pose_y: float,
        pose_yaw: float,
        robot_point: tuple[float, float, float],
    ) -> tuple[float, float]:
        forward, lateral, _ = robot_point
        wx = pose_x + forward * math.cos(pose_yaw) - lateral * math.sin(pose_yaw)
        wy = pose_y + forward * math.sin(pose_yaw) + lateral * math.cos(pose_yaw)
        return (float(wx), float(wy))
