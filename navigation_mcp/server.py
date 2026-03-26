from __future__ import annotations

import base64
import io
import json
import sys
import traceback
from typing import Any

import numpy as np
from PIL import Image

from .bridge import Go2BridgeConfig, Go2MoveBridge, SimulatedRobotBridge
from .models import Observation
from .navigator import NavigationEngine


class MCPServer:
    def __init__(self) -> None:
        self.engines = {
            "sim": NavigationEngine(SimulatedRobotBridge()),
            "go2": NavigationEngine(Go2MoveBridge(Go2BridgeConfig())),
        }
        self.default_backend = "sim"

    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        method = request.get("method")
        req_id = request.get("id")
        try:
            if method == "initialize":
                return self._ok(
                    req_id,
                    {
                        "protocolVersion": "2025-03-26",
                        "serverInfo": {"name": "go2-navigation-mcp", "version": "0.1.0"},
                        "capabilities": {"tools": {}},
                    },
                )
            if method == "notifications/initialized":
                return None
            if method == "tools/list":
                return self._ok(req_id, {"tools": self._tools()})
            if method == "tools/call":
                params = request.get("params", {})
                tool_name = params["name"]
                arguments = params.get("arguments", {})
                result = self._call_tool(tool_name, arguments)
                return self._ok(req_id, {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]})
            return self._err(req_id, -32601, f"method not found: {method}")
        except Exception as exc:
            return self._err(req_id, -32000, f"{exc}\n{traceback.format_exc()}")

    def _tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "configure_go2_backend",
                "description": "Configure host bind, ports, SSH target, and remote workspace for the real Go2 backend.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "host_bind": {"type": "string"},
                        "video_port": {"type": "number"},
                        "state_port": {"type": "number"},
                        "occupancy_port": {"type": "number"},
                        "depth_port": {"type": "number"},
                        "motion_port": {"type": "number"},
                        "ssh_host": {"type": "string"},
                        "ssh_user": {"type": "string"},
                        "ssh_password": {"type": "string"},
                        "remote_project_dir": {"type": "string"},
                        "remote_python": {"type": "string"},
                        "remote_setup": {"type": "string"},
                        "remote_ros_choice": {"type": "string"},
                        "remote_sudo_password": {"type": "string"},
                        "auto_start_remote": {"type": "boolean"},
                        "remote_livox_setup": {"type": "string"},
                        "remote_livox_launch": {"type": "string"},
                        "remote_data_script": {"type": "string"},
                        "remote_motion_script": {"type": "string"},
                        "remote_data_command": {"type": "string"},
                        "remote_motion_command": {"type": "string"},
                        "remote_data_video_backend": {"type": "string", "enum": ["auto", "realsense", "opencv"]},
                        "remote_data_video_index": {"type": "number"},
                        "remote_motion_backend": {"type": "string", "enum": ["auto", "sdk2", "ros2_topic"]},
                        "remote_motion_sdk_python_path": {"type": "string"},
                        "remote_motion_network_interface": {"type": "string"},
                        "remote_motion_require_subscriber": {"type": "boolean"},
                        "remote_sync_before_start": {"type": "boolean"},
                        "remote_startup_delay_s": {"type": "number"},
                        "remote_observation_wait_timeout_s": {"type": "number"},
                    },
                },
            },
            {
                "name": "start_go2_services",
                "description": "Start robot-side MID360, data_server.py, and atom_server.py over SSH. Host-side MCP listeners are started before this.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "stop_go2_services",
                "description": "Stop the SSH-launched robot-side data/control services.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "set_navigation_target",
                "description": "Set the target object, success distance/heading thresholds, and optional detection hint.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "backend": {"type": "string", "enum": ["sim", "go2"]},
                        "target_label": {"type": "string"},
                        "success_distance_m": {"type": "number"},
                        "success_heading_deg": {"type": "number"},
                        "detection_hint": {"type": "object"},
                    },
                    "required": ["target_label"],
                },
            },
            {
                "name": "step_navigation",
                "description": "Run one navigation cycle: observe, detect, search/replan, and issue one action.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"backend": {"type": "string", "enum": ["sim", "go2"]}},
                },
            },
            {
                "name": "run_navigation",
                "description": "Keep stepping until success, blocked, not found, cancelled, or timeout.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "backend": {"type": "string", "enum": ["sim", "go2"]},
                        "timeout_s": {"type": "number"},
                        "step_delay_s": {"type": "number"},
                    },
                },
            },
            {
                "name": "cancel_navigation",
                "description": "Cancel the active navigation task.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"backend": {"type": "string", "enum": ["sim", "go2"]}},
                },
            },
            {
                "name": "get_navigation_status",
                "description": "Return the current state of the navigation state machine.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"backend": {"type": "string", "enum": ["sim", "go2"]}},
                },
            },
            {
                "name": "load_observation",
                "description": "Inject a manual observation for offline testing or custom sensor plumbing.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "backend": {"type": "string", "enum": ["sim", "go2"]},
                        "rgb_base64_png": {"type": "string"},
                        "depth_m": {"type": "array"},
                        "occupancy": {"type": "array"},
                        "pose_xy_yaw": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                    },
                },
            },
        ]

    def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        engine = self.engines[arguments.get("backend", self.default_backend)]
        if tool_name == "configure_go2_backend":
            go2_bridge = self.engines["go2"].bridge
            return go2_bridge.configure(**arguments)
        if tool_name == "start_go2_services":
            go2_bridge = self.engines["go2"].bridge
            return go2_bridge.start_remote_services()
        if tool_name == "stop_go2_services":
            go2_bridge = self.engines["go2"].bridge
            return go2_bridge.stop_remote_services()
        if tool_name == "set_navigation_target":
            return engine.set_target(
                target_label=arguments["target_label"],
                success_distance_m=arguments.get("success_distance_m"),
                success_heading_deg=arguments.get("success_heading_deg"),
                detection_hint=arguments.get("detection_hint"),
            )
        if tool_name == "step_navigation":
            return engine.step()
        if tool_name == "run_navigation":
            return engine.run_until_done(
                timeout_s=float(arguments.get("timeout_s", 30.0)),
                step_delay_s=float(arguments.get("step_delay_s", 0.0)),
            )
        if tool_name == "cancel_navigation":
            return engine.cancel()
        if tool_name == "get_navigation_status":
            return engine.get_status()
        if tool_name == "load_observation":
            obs = Observation(
                rgb=self._decode_rgb(arguments.get("rgb_base64_png")),
                depth_m=self._decode_array(arguments.get("depth_m")),
                occupancy=self._decode_array(arguments.get("occupancy"), dtype=np.uint8),
                pose_xy_yaw=tuple(arguments.get("pose_xy_yaw", [0.0, 0.0, 0.0])),
            )
            return engine.load_observation(obs)
        raise ValueError(f"unknown tool: {tool_name}")

    def _decode_rgb(self, payload: str | None) -> np.ndarray | None:
        if not payload:
            return None
        raw = base64.b64decode(payload)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return np.array(img)

    def _decode_array(self, value: Any, dtype: Any = np.float32) -> np.ndarray | None:
        if value is None:
            return None
        return np.array(value, dtype=dtype)

    @staticmethod
    def _ok(req_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    @staticmethod
    def _err(req_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def main() -> None:
    server = MCPServer()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        response = server.handle(request)
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
