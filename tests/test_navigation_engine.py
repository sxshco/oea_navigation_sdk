import numpy as np

from navigation_mcp.bridge import SimulatedRobotBridge
from navigation_mcp.models import Observation
from navigation_mcp.navigator import NavigationEngine


def test_search_rotates_and_reports_not_found():
    bridge = SimulatedRobotBridge()
    engine = NavigationEngine(bridge)
    engine.set_target("cup")
    result = engine.run_until_done(timeout_s=0.1)
    assert result["phase"] == "not_found"
    assert result["search_turns_completed"] == 4


def test_blocked_when_obstacle_intersects_path():
    engine = NavigationEngine(SimulatedRobotBridge())
    engine.set_target("cup", detection_hint={"rgb_range": [[180, 0, 0], [255, 60, 60]]})
    occupancy = np.zeros((12, 12), dtype=np.uint8)
    occupancy[6, 5:10] = 1
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    rgb[52:68, 72:88, 0] = 220
    depth = np.full((120, 160), np.nan, dtype=np.float32)
    depth[52:68, 72:88] = 1.5
    engine.load_observation(
        Observation(rgb=rgb, depth_m=depth, occupancy=occupancy, pose_xy_yaw=(0.0, 0.0, 0.0))
    )
    result = engine.step()
    assert result["phase"] == "tracking"
    assert result["message"] == "tracking target via obstacle detour"


def test_blocked_only_when_target_vicinity_fully_enclosed():
    engine = NavigationEngine(SimulatedRobotBridge())
    engine.set_target(
        "cup",
        success_distance_m=0.2,
        detection_hint={"rgb_range": [[180, 0, 0], [255, 60, 60]]},
    )
    occupancy = np.zeros((12, 12), dtype=np.uint8)
    occupancy[4:9, 8] = 1
    occupancy[4:9, 11] = 1
    occupancy[4, 8:12] = 1
    occupancy[8, 8:12] = 1
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    rgb[52:68, 72:88, 0] = 220
    depth = np.full((120, 160), np.nan, dtype=np.float32)
    depth[52:68, 72:88] = 1.0
    engine.load_observation(
        Observation(rgb=rgb, depth_m=depth, occupancy=occupancy, pose_xy_yaw=(0.0, 0.0, 0.0))
    )
    first = engine.step()
    assert first["phase"] == "tracking"
    assert first["message"] == "approaching closest reachable position before blocked assessment"
    assert first["closest_reachable_distance_m"] is not None

    engine.load_observation(
        Observation(rgb=rgb, depth_m=depth, occupancy=occupancy, pose_xy_yaw=(0.6, 0.0, 0.0))
    )
    second = engine.step()
    assert second["phase"] == "blocked"
    assert second["closest_reachable_distance_m"] is not None


def test_success_when_inside_threshold():
    engine = NavigationEngine(SimulatedRobotBridge())
    engine.set_target(
        "cup",
        success_distance_m=0.8,
        detection_hint={"rgb_range": [[180, 0, 0], [255, 60, 60]]},
    )
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    rgb[52:68, 72:88, 0] = 220
    depth = np.full((120, 160), np.nan, dtype=np.float32)
    depth[52:68, 72:88] = 0.55
    engine.load_observation(
        Observation(rgb=rgb, depth_m=depth, occupancy=np.zeros((12, 12), dtype=np.uint8))
    )
    result = engine.step()
    assert result["phase"] == "success"


def test_within_success_radius_aligns_before_success():
    engine = NavigationEngine(SimulatedRobotBridge())
    engine.config.success_heading_deg = 5.0
    engine.set_target(
        "cup",
        success_distance_m=0.8,
        detection_hint={"rgb_range": [[180, 0, 0], [255, 60, 60]]},
    )
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    rgb[52:68, 132:148, 0] = 220
    depth = np.full((120, 160), np.nan, dtype=np.float32)
    depth[52:68, 132:148] = 0.55
    engine.load_observation(
        Observation(rgb=rgb, depth_m=depth, occupancy=np.zeros((12, 12), dtype=np.uint8))
    )
    first = engine.step()
    assert first["phase"] == "tracking"
    assert first["message"] == "within success distance, aligning to face target"

    rgb2 = np.zeros((120, 160, 3), dtype=np.uint8)
    rgb2[52:68, 72:88, 0] = 220
    depth2 = np.full((120, 160), np.nan, dtype=np.float32)
    depth2[52:68, 72:88] = 0.55
    engine.load_observation(
        Observation(rgb=rgb2, depth_m=depth2, occupancy=np.zeros((12, 12), dtype=np.uint8))
    )
    second = engine.step()
    assert second["phase"] == "success"
