from __future__ import annotations

import abc
import asyncio
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import shlex
import socket
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pexpect
from PIL import Image

from .models import Observation


VIDEO_PORT = 5220
AUDIO_PORT = 5221
STATE_PORT = 5222
OCCUPANCY_PORT = 5223
DEPTH_PORT = 5224
MOTION_PORT = 8000


@dataclass
class ActionCommand:
    kind: str
    value: float = 0.0


@dataclass
class Go2BridgeConfig:
    host_bind: str = "0.0.0.0"
    video_port: int = VIDEO_PORT
    state_port: int = STATE_PORT
    occupancy_port: int = OCCUPANCY_PORT
    depth_port: int = DEPTH_PORT
    motion_port: int = MOTION_PORT
    ssh_host: str = "192.168.86.26"
    ssh_user: str = "unitree"
    ssh_password: str = ""
    ssh_options: tuple[str, ...] = ("-o", "StrictHostKeyChecking=accept-new")
    remote_project_dir: str = "/home/unitree/navigation_sdk"
    remote_python: str = "python3"
    remote_setup: str = ""
    remote_ros_choice: str = "1"
    remote_sudo_password: str = ""
    auto_start_remote: bool = False
    remote_livox_setup: str = ""
    remote_livox_launch: str = ""
    remote_data_script: str = "move/data_server.py"
    remote_motion_script: str = "move/atom_server.py"
    remote_data_command: str = ""
    remote_motion_command: str = ""
    remote_data_video_backend: str = "auto"
    remote_data_video_index: int = 4
    remote_motion_backend: str = "auto"
    remote_motion_sdk_python_path: str = "/home/unitree/unitree_sdk2_python"
    remote_motion_network_interface: str = ""
    remote_motion_require_subscriber: bool = True
    remote_sync_before_start: bool = True
    remote_sync_paths: tuple[str, ...] = ("move", "navigation_mcp", "pyproject.toml", "requirements.txt")
    remote_sync_excludes: tuple[str, ...] = ("__pycache__", "*.pyc", ".pytest_cache", ".venv", "sam3_main")
    remote_startup_delay_s: float = 2.0
    remote_observation_wait_timeout_s: float = 20.0
    forward_speed_x: float = 0.55
    turn_speed_z: float = 0.95
    motion_confirm_timeout_s: float = 8.0
    motion_confirm_translation_m: float = 0.05
    motion_confirm_rotation_deg: float = 8.0


class VideoStateOccupancyReceiver:
    def __init__(self, config: Go2BridgeConfig):
        self.config = config
        self.running = False
        self.latest_rgb: np.ndarray | None = None
        self.latest_depth: np.ndarray | None = None
        self.latest_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.latest_occupancy: np.ndarray | None = None
        self.latest_timestamp = 0.0
        self.latest_video_timestamp = 0.0
        self.latest_state_timestamp = 0.0
        self.latest_occupancy_timestamp = 0.0
        self.latest_depth_timestamp = 0.0
        self.lock = threading.Lock()
        self.threads: list[threading.Thread] = []

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.threads = [
            threading.Thread(target=self._receive_video, daemon=True),
            threading.Thread(target=self._receive_state, daemon=True),
            threading.Thread(target=self._receive_occupancy, daemon=True),
            threading.Thread(target=self._receive_depth, daemon=True),
        ]
        for thread in self.threads:
            thread.start()

    def stop(self) -> None:
        self.running = False

    def get_latest(self) -> Observation:
        with self.lock:
            return Observation(
                rgb=None if self.latest_rgb is None else self.latest_rgb.copy(),
                depth_m=None if self.latest_depth is None else self.latest_depth.copy(),
                occupancy=None if self.latest_occupancy is None else self.latest_occupancy.copy(),
                pose_xy_yaw=self.latest_pose,
                timestamp=self.latest_timestamp,
            )

    def _receive_video(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.config.host_bind, self.config.video_port))
        sock.settimeout(1.0)
        buffers: dict[int, dict[int, bytes]] = defaultdict(dict)
        totals: dict[int, int] = {}
        while self.running:
            try:
                packet, _ = sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            if len(packet) < 10:
                continue
            frame_id, total, idx, payload_len = struct.unpack("!IHHH", packet[:10])
            payload = packet[10:]
            if len(payload) != payload_len:
                continue
            buffers[frame_id][idx] = payload
            totals[frame_id] = total
            if len(buffers[frame_id]) == total:
                data = b"".join(buffers[frame_id][i] for i in range(total))
                try:
                    img = np.array(Image.open(__import__("io").BytesIO(data)).convert("RGB"))
                except Exception:
                    del buffers[frame_id]
                    totals.pop(frame_id, None)
                    continue
                with self.lock:
                    self.latest_rgb = img
                    self.latest_timestamp = time.time()
                    self.latest_video_timestamp = self.latest_timestamp
                del buffers[frame_id]
                totals.pop(frame_id, None)
        sock.close()

    def _receive_state(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.config.host_bind, self.config.state_port))
        server.listen(1)
        server.settimeout(1.0)
        while self.running:
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with conn:
                conn.settimeout(1.0)
                while self.running:
                    try:
                        header = self._recv_exact(conn, 4)
                    except (socket.timeout, ConnectionError, OSError):
                        break
                    payload_len = struct.unpack("!I", header)[0]
                    if payload_len == 0:
                        continue
                    try:
                        payload = self._recv_exact(conn, payload_len)
                    except (socket.timeout, ConnectionError, OSError):
                        break
                    x, y, yaw = struct.unpack("!ddd", payload)
                    with self.lock:
                        self.latest_pose = (float(x), float(y), float(yaw))
                        self.latest_timestamp = time.time()
                        self.latest_state_timestamp = self.latest_timestamp
        server.close()

    def _receive_occupancy(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.config.host_bind, self.config.occupancy_port))
        server.listen(1)
        server.settimeout(1.0)
        while self.running:
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with conn:
                conn.settimeout(1.0)
                while self.running:
                    try:
                        header = self._recv_exact(conn, 4)
                    except (socket.timeout, ConnectionError, OSError):
                        break
                    payload_len = struct.unpack("!I", header)[0]
                    if payload_len == 0:
                        continue
                    try:
                        payload = self._recv_exact(conn, payload_len)
                    except (socket.timeout, ConnectionError, OSError):
                        break
                    if len(payload) < 8:
                        continue
                    rows, cols = struct.unpack("!II", payload[:8])
                    grid = np.frombuffer(payload[8:], dtype=np.uint8).reshape(rows, cols)
                    with self.lock:
                        self.latest_occupancy = grid.copy()
                        self.latest_timestamp = time.time()
                        self.latest_occupancy_timestamp = self.latest_timestamp
        server.close()

    def _receive_depth(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.config.host_bind, self.config.depth_port))
        server.listen(1)
        server.settimeout(1.0)
        while self.running:
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with conn:
                conn.settimeout(1.0)
                while self.running:
                    try:
                        header = self._recv_exact(conn, 4)
                    except (socket.timeout, ConnectionError, OSError):
                        break
                    payload_len = struct.unpack("!I", header)[0]
                    if payload_len == 0:
                        continue
                    try:
                        payload = self._recv_exact(conn, payload_len)
                    except (socket.timeout, ConnectionError, OSError):
                        break
                    if len(payload) < 8:
                        continue
                    rows, cols = struct.unpack("!II", payload[:8])
                    png = payload[8:]
                    try:
                        image = Image.open(__import__("io").BytesIO(png))
                        depth_mm = np.array(image, dtype=np.uint16)
                    except Exception:
                        continue
                    if depth_mm.shape[:2] != (rows, cols):
                        continue
                    depth_m = depth_mm.astype(np.float32) * 0.001
                    depth_m[depth_mm == 0] = np.nan
                    with self.lock:
                        self.latest_depth = depth_m
                        self.latest_timestamp = time.time()
                        self.latest_depth_timestamp = self.latest_timestamp
        server.close()

    @staticmethod
    def _recv_exact(conn: socket.socket, size: int) -> bytes:
        buf = bytearray()
        while len(buf) < size:
            chunk = conn.recv(size - len(buf))
            if not chunk:
                raise ConnectionError("connection closed")
            buf.extend(chunk)
        return bytes(buf)


class MotionCommandServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.running = False
        self.thread: threading.Thread | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.websocket = None
        self.websocket_ready = threading.Event()
        self.initialized_ready = threading.Event()
        self.command_lock = threading.Lock()
        self.status_lock = threading.Condition()
        self.status_seq = 0
        self.last_status: dict[str, Any] | None = None
        self.next_command_id = 1
        self.session_initialized = False
        self.start_error: str | None = None

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.is_connected():
            try:
                self.send_shutdown_sequence()
            except Exception:
                pass
        self.running = False
        self.initialized_ready.clear()
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.loop.stop)

    def is_connected(self) -> bool:
        return self.websocket is not None and self.websocket_ready.is_set()

    def send_atomic(self, action: ActionCommand, forward_speed_x: float, turn_speed_z: float) -> dict[str, Any]:
        with self.command_lock:
            if self.start_error is not None:
                return {"ok": False, "reason": self.start_error}
            if not self.websocket_ready.wait(timeout=5):
                return {"ok": False, "reason": "robot motion websocket not connected"}
            if not self.initialized_ready.wait(timeout=8):
                return {"ok": False, "reason": "robot motion websocket connected but initialization not finished"}

            command_id = self.next_command_id
            self.next_command_id += 1
            before_seq = self.status_seq

            if action.kind == "stop":
                self._send_now(1003, {"command_id": command_id})
                status = self._wait_for_command_status(before_seq, command_id, timeout_s=2.0)
                return self._status_to_result(status, fallback={"ok": True, "kind": "stop", "command_id": command_id})

            if action.kind == "forward":
                move = {
                    "command_id": command_id,
                    "x": forward_speed_x,
                    "y": 0.0,
                    "z": -0.01,
                    "controller": "closed_loop",
                    "target_distance_m": float(action.value),
                }
            elif action.kind == "turn_left":
                move = {
                    "command_id": command_id,
                    "x": 0.0,
                    "y": 0.0,
                    "z": abs(turn_speed_z),
                    "controller": "closed_loop",
                    "target_yaw_deg": float(action.value),
                }
            elif action.kind == "turn_right":
                move = {
                    "command_id": command_id,
                    "x": 0.0,
                    "y": 0.0,
                    "z": -abs(turn_speed_z),
                    "controller": "closed_loop",
                    "target_yaw_deg": float(action.value),
                }
            else:
                return {"ok": False, "reason": f"unsupported action {action.kind}"}

            self._send_now(1008, move, binary=[127])
            status = self._wait_for_command_status(before_seq, command_id, timeout_s=2.5)
            status_result = self._status_to_result(status, fallback=None)
            if status_result is not None and not status_result.get("ok", False):
                return status_result
            return {
                "ok": True,
                "kind": action.kind,
                "command_id": command_id,
                "closed_loop": True,
                "parameter": move,
                "binary": [127],
            }

    def _run(self) -> None:
        try:
            import websockets
            from websockets.exceptions import ConnectionClosed
        except ImportError as exc:
            self.start_error = f"websockets import failed: {exc}"
            return

        async def handler(websocket):
            self.websocket = websocket
            self.websocket_ready.set()
            try:
                if not self.session_initialized:
                    await self._send_init_sequence()
                    self.session_initialized = True
                else:
                    self.initialized_ready.set()
                async for raw_message in websocket:
                    self._record_robot_status(raw_message)
            except ConnectionClosed:
                # Robot-side service shutdown/restart may close the socket without a formal close frame.
                pass
            finally:
                self.websocket = None
                self.websocket_ready.clear()
                self.initialized_ready.clear()

        async def main() -> None:
            async with websockets.serve(handler, self.host, self.port):
                while self.running:
                    await asyncio.sleep(0.1)

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(main())
        except RuntimeError:
            pass

    async def _send_init_sequence(self) -> None:
        last_status: dict[str, Any] | None = None
        for api_id, delay_s in ((1004, 2.0), (1002, 1.0)):
            for _ in range(4):
                before_seq = self.status_seq
                await self._send_async(api_id, {})
                status = await asyncio.to_thread(self._wait_for_command_status, before_seq, 0, 2.5)
                last_status = status
                if status is None or status.get("event") != "command_rejected":
                    break
                await asyncio.sleep(1.0)
            await asyncio.sleep(delay_s)
        if last_status is None or last_status.get("event") != "command_rejected":
            self.initialized_ready.set()
        else:
            self.start_error = f"robot initialization rejected: {last_status.get('reason')}"

    def send_shutdown_sequence(self) -> None:
        with self.command_lock:
            if not self.websocket_ready.wait(timeout=2):
                return
            self._send_now(1003, {})
            time.sleep(0.6)
            self._send_now(1005, {})
            time.sleep(2.5)
            self._send_now(1001, {})
            self.session_initialized = False

    def _send_now(self, api_id: int, parameter: dict[str, Any], binary: list[int] | None = None) -> None:
        if self.loop is None:
            raise RuntimeError("motion server loop is not running")
        future = asyncio.run_coroutine_threadsafe(
            self._send_async(api_id, parameter, binary=binary),
            self.loop,
        )
        future.result(timeout=5)

    async def _send_async(self, api_id: int, parameter: dict[str, Any], binary: list[int] | None = None) -> None:
        if self.websocket is None:
            raise RuntimeError("robot motion websocket is not connected")
        payload: dict[str, Any] = {"api_id": api_id, "parameter": parameter}
        if binary is not None:
            payload["binary"] = binary
        await self.websocket.send(json.dumps(payload))

    def _record_robot_status(self, raw_message: str) -> None:
        try:
            payload = json.loads(raw_message)
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        with self.status_lock:
            self.status_seq += 1
            self.last_status = payload
            self.status_lock.notify_all()

    def _wait_for_command_status(
        self,
        before_seq: int,
        command_id: int,
        timeout_s: float,
    ) -> dict[str, Any] | None:
        deadline = time.time() + timeout_s
        with self.status_lock:
            while time.time() < deadline:
                status = self.last_status
                if self.status_seq > before_seq and isinstance(status, dict):
                    status_command_id = status.get("command_id")
                    if status_command_id == command_id or (
                        command_id == 0 and status_command_id is None
                    ) or (
                        command_id == 0 and status.get("event") == "controller_ready"
                    ):
                        return status
                remaining = max(0.0, deadline - time.time())
                if remaining <= 0:
                    break
                self.status_lock.wait(timeout=remaining)
        return None

    @staticmethod
    def _status_to_result(status: dict[str, Any] | None, fallback: dict[str, Any] | None) -> dict[str, Any] | None:
        if status is None:
            return fallback
        event = status.get("event")
        if event == "command_rejected":
            return {
                "ok": False,
                "reason": status.get("reason", "command_rejected"),
                "command_id": status.get("command_id"),
                "robot_status": status,
            }
        if event in {"motion_started", "command_forwarded", "stop_ack"}:
            return {
                "ok": True,
                "command_id": status.get("command_id"),
                "robot_status": status,
                **({} if fallback is None else fallback),
            }
        return fallback


class RobotBridge(abc.ABC):
    @abc.abstractmethod
    def get_observation(self) -> Observation:
        raise NotImplementedError

    @abc.abstractmethod
    def execute(self, command: ActionCommand) -> dict[str, Any]:
        raise NotImplementedError

    def stop(self) -> dict[str, Any]:
        return self.execute(ActionCommand(kind="stop", value=0.0))


class Go2MoveBridge(RobotBridge):
    def __init__(self, config: Go2BridgeConfig | None = None) -> None:
        self.config = config or Go2BridgeConfig()
        self.receiver = VideoStateOccupancyReceiver(self.config)
        self.receiver.start()
        self.motion_server = MotionCommandServer(self.config.host_bind, self.config.motion_port)
        self.motion_server.start()
        self._remote_processes: list[pexpect.spawn] = []
        if self.config.auto_start_remote:
            self.start_remote_services()

    def configure(self, **kwargs: Any) -> dict[str, Any]:
        for key, value in kwargs.items():
            if value is None or not hasattr(self.config, key):
                continue
            setattr(self.config, key, value)
        return self.describe()

    def describe(self) -> dict[str, Any]:
        latest = self.receiver.get_latest()
        return {
            "host_bind": self.config.host_bind,
            "video_port": self.config.video_port,
            "state_port": self.config.state_port,
            "occupancy_port": self.config.occupancy_port,
            "depth_port": self.config.depth_port,
            "motion_port": self.config.motion_port,
            "ssh_host": self.config.ssh_host,
            "ssh_user": self.config.ssh_user,
            "remote_project_dir": self.config.remote_project_dir,
            "remote_livox_launch": self.config.remote_livox_launch,
            "remote_sync_before_start": self.config.remote_sync_before_start,
            "remote_data_video_backend": self.config.remote_data_video_backend,
            "remote_data_video_index": self.config.remote_data_video_index,
            "remote_motion_sdk_python_path": self.config.remote_motion_sdk_python_path,
            "remote_motion_network_interface": self.config.remote_motion_network_interface,
            "remote_motion_backend": self.config.remote_motion_backend,
            "remote_motion_require_subscriber": self.config.remote_motion_require_subscriber,
            "motion_connected": self.motion_server.is_connected(),
            "robot_motion_status": self.motion_server.last_status,
            "latest_observation_timestamp": latest.timestamp,
            "latest_video_timestamp": self.receiver.latest_video_timestamp,
            "latest_state_timestamp": self.receiver.latest_state_timestamp,
            "latest_occupancy_timestamp": self.receiver.latest_occupancy_timestamp,
            "latest_depth_timestamp": self.receiver.latest_depth_timestamp,
        }

    def start_remote_services(self) -> dict[str, Any]:
        self.stop_remote_services()
        if self.config.remote_sync_before_start:
            self._sync_remote_workspace()
        commands = []
        if self.config.remote_livox_launch:
            commands.append(self._build_remote_livox_command())
            time.sleep(max(0.0, float(self.config.remote_startup_delay_s)))
        commands.extend(
            [
                self._build_remote_data_server_command(),
                self._build_remote_motion_server_command(),
            ]
        )
        self._remote_processes = [self._spawn_interactive_process(command) for command in commands]
        observation_wait = self._wait_for_initial_observation(timeout_s=float(self.config.remote_observation_wait_timeout_s))
        health = self._probe_remote_runtime()
        return {
            "ok": True,
            "pids": [proc.pid for proc in self._remote_processes],
            "commands": commands,
            "observation_wait": observation_wait,
            "health": health,
        }

    def stop_remote_services(self) -> dict[str, Any]:
        stopped = []
        for proc in self._remote_processes:
            try:
                if proc.isalive():
                    proc.sendintr()
                    time.sleep(0.5)
                if proc.isalive():
                    proc.close(force=True)
            finally:
                stopped.append(proc.pid)
        self._remote_processes = []
        return {"ok": True, "stopped_pids": stopped}

    def get_observation(self) -> Observation:
        obs = self.receiver.get_latest()
        if obs.timestamp <= 0.0:
            raise RuntimeError("no live RGB/state observation received from robot data_server yet")
        return obs

    def execute(self, command: ActionCommand) -> dict[str, Any]:
        before = self.receiver.get_latest()
        send_result = self.motion_server.send_atomic(
            command,
            forward_speed_x=self.config.forward_speed_x,
            turn_speed_z=self.config.turn_speed_z,
        )
        if not send_result.get("ok"):
            return send_result
        motion_check = self._confirm_motion(command, before)
        return {**send_result, **motion_check}

    def _confirm_motion(self, command: ActionCommand, before: Observation) -> dict[str, Any]:
        if command.kind == "stop":
            return {"motion_confirmed": True, "motion_reason": "stop_not_verified"}

        deadline = time.time() + float(self.config.motion_confirm_timeout_s)
        while time.time() < deadline:
            current = self.receiver.get_latest()
            if current.timestamp <= before.timestamp:
                time.sleep(0.2)
                continue
            delta_x = current.pose_xy_yaw[0] - before.pose_xy_yaw[0]
            delta_y = current.pose_xy_yaw[1] - before.pose_xy_yaw[1]
            delta_yaw = self._angle_diff(current.pose_xy_yaw[2], before.pose_xy_yaw[2])
            translation = math.hypot(delta_x, delta_y)
            rotation_deg = abs(math.degrees(delta_yaw))

            if command.kind == "forward" and translation >= max(
                self.config.motion_confirm_translation_m,
                min(abs(command.value) * 0.3, abs(command.value)),
            ):
                return {
                    "motion_confirmed": True,
                    "motion_reason": "odometry_translation_observed",
                    "odometry_delta_xy": [delta_x, delta_y],
                    "odometry_delta_yaw_rad": delta_yaw,
                }
            if command.kind in {"turn_left", "turn_right"} and rotation_deg >= max(
                self.config.motion_confirm_rotation_deg,
                min(abs(command.value) * 0.3, abs(command.value)),
            ):
                return {
                    "motion_confirmed": True,
                    "motion_reason": "odometry_rotation_observed",
                    "odometry_delta_xy": [delta_x, delta_y],
                    "odometry_delta_yaw_rad": delta_yaw,
                }
            time.sleep(0.2)

        return {
            "ok": False,
            "motion_confirmed": False,
            "motion_reason": "odometry_change_not_observed",
            "odometry_before": list(before.pose_xy_yaw),
            "odometry_after": list(self.receiver.get_latest().pose_xy_yaw),
        }

    @staticmethod
    def _angle_diff(current: float, previous: float) -> float:
        delta = current - previous
        while delta > math.pi:
            delta -= 2 * math.pi
        while delta < -math.pi:
            delta += 2 * math.pi
        return delta

    def _build_ssh_prefix(self) -> list[str]:
        return ["ssh", *self.config.ssh_options, "-tt", f"{self.config.ssh_user}@{self.config.ssh_host}"]

    def _remote_shell(self, body: str) -> list[str]:
        return self._build_ssh_prefix() + [f"bash -lc {shlex.quote(body)}"]

    def _sync_remote_workspace(self) -> None:
        remote_root = f"{self.config.ssh_user}@{self.config.ssh_host}:{self.config.remote_project_dir}/"
        mkdir_cmd = self._remote_shell(f"mkdir -p {shlex.quote(self.config.remote_project_dir)}")
        self._run_interactive_process(mkdir_cmd)

        local_root = Path(__file__).resolve().parent.parent
        for relative_path in self.config.remote_sync_paths:
            source = local_root / relative_path
            if not source.exists():
                continue
            scp_cmd = ["scp", *self.config.ssh_options, "-r", str(source), remote_root]
            self._run_interactive_process(scp_cmd)

    def _spawn_interactive_process(self, command: list[str]) -> pexpect.spawn:
        child = pexpect.spawn(
            command[0],
            command[1:],
            encoding="utf-8",
            timeout=30,
        )
        self._drive_interactive_process(child, timeout_s=30.0, wait_for_exit=False, ready_pattern="__GO2_REMOTE_READY__")
        return child

    def _run_interactive_process(self, command: list[str]) -> None:
        child = pexpect.spawn(
            command[0],
            command[1:],
            encoding="utf-8",
            timeout=60,
        )
        try:
            self._drive_interactive_process(child, timeout_s=60.0, wait_for_exit=True, ready_pattern=None)
        finally:
            if child.isalive():
                child.close(force=True)

    def _capture_interactive_process(self, command: list[str], timeout_s: float = 60.0) -> str:
        child = pexpect.spawn(
            command[0],
            command[1:],
            encoding="utf-8",
            timeout=60,
        )
        try:
            self._drive_interactive_process(child, timeout_s=timeout_s, wait_for_exit=True, ready_pattern=None)
            transcript = (child.before or "") + ("" if child.after in {None, pexpect.EOF, pexpect.TIMEOUT} else child.after)
            return transcript.strip()
        finally:
            if child.isalive():
                child.close(force=True)

    def _drive_interactive_process(
        self,
        process: pexpect.spawn,
        timeout_s: float,
        wait_for_exit: bool,
        ready_pattern: str | None,
    ) -> None:
        deadline = time.time() + timeout_s
        transcript = ""
        sent_password = False
        sent_ros_choice = False
        sent_sudo_password = False
        accepted_host = False
        quiet_cycles = 0
        ready_seen = ready_pattern is None

        while True:
            if wait_for_exit and not process.isalive():
                if process.exitstatus not in {0, None}:
                    raise RuntimeError(f"remote command failed: {transcript[-400:]}")
                return

            if not wait_for_exit and not process.isalive():
                raise RuntimeError(f"remote startup exited early: {transcript[-400:]}")

            now = time.time()
            if now > deadline:
                if wait_for_exit:
                    raise TimeoutError(f"remote command timed out: {transcript[-400:]}")
                return

            patterns = [
                "Are you sure you want to continue connecting",
                "[Pp]assword:",
                "ros:foxy\\(1\\) noetic\\(2\\) \\?",
                "\\[sudo\\] password for",
            ]
            if ready_pattern:
                patterns.append(ready_pattern)
            patterns.extend([pexpect.EOF, pexpect.TIMEOUT])

            idx = process.expect(patterns, timeout=0.5)
            before = process.before or ""
            after = "" if process.after in {pexpect.EOF, pexpect.TIMEOUT, None} else process.after
            transcript = (transcript + before + after)[-4000:]
            if idx == 0:
                if not accepted_host:
                    process.sendline("yes")
                    accepted_host = True
            elif idx == 1:
                lower = (process.before + process.after).lower()
                if "sudo" in lower and not sent_sudo_password:
                    password = self.config.remote_sudo_password or self.config.ssh_password
                    if not password:
                        raise RuntimeError("sudo password is required for remote ROS selection flow")
                    process.sendline(password)
                    sent_sudo_password = True
                elif not sent_password:
                    if not self.config.ssh_password:
                        raise RuntimeError("ssh password is required for interactive remote login")
                    process.sendline(self.config.ssh_password)
                    sent_password = True
            elif idx == 2:
                if not sent_ros_choice:
                    process.sendline(self.config.remote_ros_choice)
                    sent_ros_choice = True
            elif idx == 3:
                password = self.config.remote_sudo_password or self.config.ssh_password
                if not password:
                    raise RuntimeError("sudo password is required for remote ROS selection flow")
                process.sendline(password)
                sent_sudo_password = True
            elif ready_pattern and idx == 4:
                ready_seen = True
                if not wait_for_exit:
                    return
            elif idx == len(patterns) - 2:
                if wait_for_exit:
                    if process.exitstatus not in {0, None}:
                        raise RuntimeError(f"remote command failed: {transcript[-400:]}")
                    return
                raise RuntimeError(f"remote startup exited early: {transcript[-400:]}")
            else:
                quiet_cycles += 1
                if not wait_for_exit and ready_seen:
                    return

    def _build_remote_data_server_command(self) -> list[str]:
        parts = []
        if self.config.remote_setup:
            parts.append(self.config.remote_setup)
        if self.config.remote_data_command:
            cmd = f"printf '__GO2_REMOTE_READY__\\n' && {self.config.remote_data_command}"
        else:
            script_path = f"{self.config.remote_project_dir}/{self.config.remote_data_script}"
            cmd = (
                f"cd {shlex.quote(self.config.remote_project_dir)} && "
                f"printf '__GO2_REMOTE_READY__\\n' && "
                f"{self.config.remote_python} {shlex.quote(script_path)} "
                f"--server-ip {shlex.quote(self._resolve_host_ip())} "
                f"--video-port {int(self.config.video_port)} "
                f"--state-port {int(self.config.state_port)} "
                f"--occupancy-port {int(self.config.occupancy_port)} "
                f"--depth-port {int(self.config.depth_port)} "
                f"--video-index {int(self.config.remote_data_video_index)} "
                f"--video-backend {shlex.quote(self.config.remote_data_video_backend)}"
            )
        parts.append(cmd)
        return self._remote_shell(" && ".join(parts))

    def _build_remote_livox_command(self) -> list[str]:
        parts = []
        if self.config.remote_setup:
            parts.append(self.config.remote_setup)
        if self.config.remote_livox_setup:
            parts.append(self.config.remote_livox_setup)
        parts.append(f"printf '__GO2_REMOTE_READY__\\n' && {self.config.remote_livox_launch}")
        return self._remote_shell(" && ".join(parts))

    def _build_remote_motion_server_command(self) -> list[str]:
        parts = []
        if self.config.remote_setup:
            parts.append(self.config.remote_setup)
        if self.config.remote_motion_command:
            cmd = f"printf '__GO2_REMOTE_READY__\\n' && {self.config.remote_motion_command}"
        else:
            script_path = f"{self.config.remote_project_dir}/{self.config.remote_motion_script}"
            cmd = (
                f"cd {shlex.quote(self.config.remote_project_dir)} && "
                f"printf '__GO2_REMOTE_READY__\\n' && "
                f"{self.config.remote_python} {shlex.quote(script_path)} "
                f"--server-ip {shlex.quote(self._resolve_host_ip())} "
                f"--server-port {int(self.config.motion_port)} "
                f"--motion-backend {shlex.quote(self.config.remote_motion_backend)} "
                f"--sdk-python-path {shlex.quote(self.config.remote_motion_sdk_python_path)}"
            )
            if self.config.remote_motion_network_interface:
                cmd += f" --network-interface {shlex.quote(self.config.remote_motion_network_interface)}"
            if self.config.remote_motion_require_subscriber:
                cmd += " --require-subscriber"
        parts.append(cmd)
        return self._remote_shell(" && ".join(parts))

    def _probe_remote_runtime(self) -> dict[str, Any]:
        checks: dict[str, Any] = {}
        topic_probe = (
            "set -o pipefail; "
            "if command -v ros2 >/dev/null 2>&1; then "
            "ros2 topic info /api/sport/request 2>&1 || true; "
            "else echo 'ros2_not_found'; fi"
        )
        process_probe = (
            "ps -ef | grep -E 'move/(atom_server|data_server)\\.py|ros2 run http_control (atom_server|data_server)' "
            "| grep -v grep || true"
        )
        if self.config.remote_setup:
            topic_probe = f"{self.config.remote_setup} && {topic_probe}"
            process_probe = f"{self.config.remote_setup} && {process_probe}"
        try:
            checks["processes"] = self._capture_interactive_process(self._remote_shell(process_probe), timeout_s=30.0)
        except Exception as exc:
            checks["processes_error"] = str(exc)
        try:
            checks["sport_topic_info"] = self._capture_interactive_process(self._remote_shell(topic_probe), timeout_s=30.0)
        except Exception as exc:
            checks["sport_topic_info_error"] = str(exc)
        return checks

    def _wait_for_initial_observation(self, timeout_s: float) -> dict[str, Any]:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            latest = self.receiver.get_latest()
            if latest.timestamp > 0.0:
                return {
                    "ready": True,
                    "latest_observation_timestamp": latest.timestamp,
                    "latest_video_timestamp": self.receiver.latest_video_timestamp,
                    "latest_state_timestamp": self.receiver.latest_state_timestamp,
                    "latest_occupancy_timestamp": self.receiver.latest_occupancy_timestamp,
                    "latest_depth_timestamp": self.receiver.latest_depth_timestamp,
                }
            time.sleep(0.5)
        return {
            "ready": False,
            "latest_observation_timestamp": self.receiver.latest_timestamp,
            "latest_video_timestamp": self.receiver.latest_video_timestamp,
            "latest_state_timestamp": self.receiver.latest_state_timestamp,
            "latest_occupancy_timestamp": self.receiver.latest_occupancy_timestamp,
            "latest_depth_timestamp": self.receiver.latest_depth_timestamp,
            "timeout_s": timeout_s,
        }

    def _resolve_host_ip(self) -> str:
        if self.config.host_bind not in {"0.0.0.0", "::"}:
            return self.config.host_bind
        env_ip = os.environ.get("GO2_HOST_IP")
        if env_ip:
            return env_ip
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]


class SimulatedRobotBridge(RobotBridge):
    def __init__(self) -> None:
        self.pose = np.array([0.0, 0.0, 0.0], dtype=float)
        self.target_world = np.array([2.0, 0.0], dtype=float)
        self.obstacle_cells = {(7, 4), (7, 5), (7, 6)}
        self.resolution = 0.25
        self.grid_size = (12, 12)

    def get_observation(self) -> Observation:
        occupancy = np.zeros(self.grid_size, dtype=np.uint8)
        for x, y in self.obstacle_cells:
            if 0 <= y < self.grid_size[0] and 0 <= x < self.grid_size[1]:
                occupancy[y, x] = 1
        rgb = np.zeros((120, 160, 3), dtype=np.uint8)
        rel = self.target_world - self.pose[:2]
        yaw = self.pose[2]
        rot = np.array(
            [
                [math.cos(-yaw), -math.sin(-yaw)],
                [math.sin(-yaw), math.cos(-yaw)],
            ]
        )
        rel_robot = rot @ rel
        forward, lateral = rel_robot[0], rel_robot[1]
        visible = forward > 0.1 and abs(math.degrees(math.atan2(lateral, forward))) < 35
        if visible:
            col = int(80 + np.clip(-lateral * 30, -50, 50))
            row = 60
            rgb[max(0, row - 8):row + 8, max(0, col - 8):col + 8, 0] = 220
        return Observation(
            rgb=rgb,
            depth_m=None,
            occupancy=occupancy,
            pose_xy_yaw=(float(self.pose[0]), float(self.pose[1]), float(self.pose[2])),
            timestamp=time.time(),
        )

    def execute(self, command: ActionCommand) -> dict[str, Any]:
        if command.kind == "forward":
            next_pose = self.pose.copy()
            next_pose[0] += math.cos(self.pose[2]) * command.value
            next_pose[1] += math.sin(self.pose[2]) * command.value
            if self._collides(next_pose[:2]):
                return {"ok": False, "reason": "collision_predicted"}
            self.pose = next_pose
        elif command.kind == "turn_right":
            self.pose[2] -= math.radians(command.value)
        elif command.kind == "turn_left":
            self.pose[2] += math.radians(command.value)
        elif command.kind == "stop":
            pass
        else:
            return {"ok": False, "reason": "unknown_command"}
        return {"ok": True, "pose": self.pose.tolist()}

    def _collides(self, xy: np.ndarray) -> bool:
        gx = int(round(xy[0] / self.resolution)) + 5
        gy = int(round(xy[1] / self.resolution)) + 5
        return (gx, gy) in self.obstacle_cells
