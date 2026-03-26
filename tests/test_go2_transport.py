import numpy as np

from navigation_mcp.bridge import ActionCommand, Go2BridgeConfig, Go2MoveBridge, MotionCommandServer
from navigation_mcp.models import NavigationConfig, Observation, TargetHint
from navigation_mcp.perception import TargetDetector


def test_monocular_distance_fallback_without_depth():
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    rgb[40:80, 60:100, 0] = 220
    detector = TargetDetector(NavigationConfig())
    obs = Observation(rgb=rgb, depth_m=None)
    hint = TargetHint(label="cup", rgb_range=((180, 0, 0), (255, 60, 60)), min_pixels=50)
    detection = detector.detect(obs, hint)
    assert detection.found is True
    assert detection.distance_m is not None
    assert detection.position_robot_m is not None
    assert detection.metadata["distance_source"] == "monocular_area"


def test_motion_command_server_sends_move_binary_flag():
    server = MotionCommandServer("127.0.0.1", 8000)
    server.websocket_ready.set()
    server.initialized_ready.set()
    server.start_error = None
    captured = {}

    def fake_send_now(api_id, parameter, binary=None):
        captured["api_id"] = api_id
        captured["parameter"] = parameter
        captured["binary"] = binary

    server._send_now = fake_send_now  # type: ignore[method-assign]
    result = server.send_atomic(ActionCommand(kind="forward", value=0.25), forward_speed_x=0.55, turn_speed_z=0.95)

    assert result["ok"] is True
    assert captured["api_id"] == 1008
    assert captured["binary"] == [127]
    assert captured["parameter"]["controller"] == "closed_loop"
    assert captured["parameter"]["target_distance_m"] == 0.25


def test_motion_command_server_sends_rotation_binary_flag():
    server = MotionCommandServer("127.0.0.1", 8000)
    server.websocket_ready.set()
    server.initialized_ready.set()
    server.start_error = None
    captured = {}

    def fake_send_now(api_id, parameter, binary=None):
        captured["api_id"] = api_id
        captured["parameter"] = parameter
        captured["binary"] = binary

    server._send_now = fake_send_now  # type: ignore[method-assign]
    result = server.send_atomic(ActionCommand(kind="turn_left", value=30.0), forward_speed_x=0.55, turn_speed_z=0.95)

    assert result["ok"] is True
    assert captured["api_id"] == 1008
    assert captured["binary"] == [127]
    assert captured["parameter"]["target_yaw_deg"] == 30.0


def test_motion_command_server_surfaces_robot_rejection():
    server = MotionCommandServer("127.0.0.1", 8000)
    server.websocket_ready.set()
    server.initialized_ready.set()
    server.start_error = None

    def fake_send_now(api_id, parameter, binary=None):
        server._record_robot_status(
            '{"event":"command_rejected","command_id":1,"reason":"no_subscriber_on_/api/sport/request"}'
        )

    server._send_now = fake_send_now  # type: ignore[method-assign]
    result = server.send_atomic(ActionCommand(kind="forward", value=0.25), forward_speed_x=0.55, turn_speed_z=0.95)

    assert result["ok"] is False
    assert result["reason"] == "no_subscriber_on_/api/sport/request"


def test_start_remote_services_orders_livox_before_robot_servers(monkeypatch):
    bridge = object.__new__(Go2MoveBridge)
    bridge.config = Go2BridgeConfig(
        host_bind="127.0.0.1",
        remote_project_dir="/tmp/project",
        remote_livox_setup="source /opt/livox/setup.sh",
        remote_livox_launch="ros2 launch livox_ros_driver2 rviz_MID360_launch.py",
        remote_data_script="move/data_server.py",
        remote_motion_script="move/atom_server.py",
        remote_startup_delay_s=0.0,
    )
    bridge._remote_processes = []

    calls = []

    class DummyProc:
        def __init__(self, pid):
            self.pid = pid

        def isalive(self):
            return True

        def sendintr(self):
            return None

        def close(self, force=False):
            return None

    def fake_spawn(command):
        calls.append(command)
        return DummyProc(len(calls))

    monkeypatch.setattr(bridge, "_spawn_interactive_process", fake_spawn)
    monkeypatch.setattr(bridge, "_wait_for_initial_observation", lambda timeout_s: {"ready": False})
    monkeypatch.setattr(bridge, "_probe_remote_runtime", lambda: {"sport_topic_info": "mock"})
    result = bridge.start_remote_services()

    assert result["ok"] is True
    assert len(calls) == 3
    assert "rviz_MID360_launch.py" in " ".join(calls[0])
    assert "data_server.py" in " ".join(calls[1])
    assert "atom_server.py" in " ".join(calls[2])
