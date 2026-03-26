#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.publisher import Publisher
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2, PointField
import websockets


SERVER_HOST_IP = "192.168.86.16"
SERVER_PORT = 8000
CONTROL_HZ = 12.0
SDK_PYTHON_PATH = "/home/unitree/unitree_sdk2_python"

DEFAULT_FORWARD_SPEED_X = 0.55
DEFAULT_SIDE_SPEED_Y = 0.32
DEFAULT_TURN_SPEED_Z = 0.95
DEFAULT_FORWARD_DISTANCE_M = 0.25
DEFAULT_LATERAL_DISTANCE_M = 0.20
DEFAULT_TURN_ANGLE_DEG = 30.0

FORWARD_TOLERANCE_M = 0.05
LATERAL_TOLERANCE_M = 0.06
YAW_TOLERANCE_DEG = 4.0


class ApiId(IntEnum):
    DAMP = 1001
    BALANCESTAND = 1002
    STOPMOVE = 1003
    STANDUP = 1004
    STANDDOWN = 1005
    RECOVERYSTAND = 1006
    MOVE = 1008
    SIT = 1009


@dataclass
class MotionGoal:
    kind: str
    parameter: dict[str, Any]
    target_displacement_body: np.ndarray
    target_yaw_rad: float
    forward_speed_x: float
    side_speed_y: float
    turn_speed_z: float
    start_pose: Odometry
    started_at: float


class MotionController:
    def can_accept_motion(self) -> tuple[bool, str | None]:
        return True, None

    def call(self, api_id: int, parameter: dict[str, Any]) -> int:
        raise NotImplementedError

    def health_snapshot(self) -> dict[str, Any]:
        return {}


class Go2SportClientAdapter(MotionController):
    def __init__(self, sdk_python_path: str, network_interface: str = "") -> None:
        sdk_root = Path(sdk_python_path).expanduser()
        if sdk_root.exists():
            sdk_root_str = str(sdk_root)
            if sdk_root_str not in sys.path:
                sys.path.insert(0, sdk_root_str)

        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.go2.sport.sport_client import SportClient
        except ImportError as exc:
            raise RuntimeError(
                f"failed to import unitree_sdk2_python from {sdk_root}: {exc}"
            ) from exc

        if network_interface:
            ChannelFactoryInitialize(0, network_interface)
        else:
            ChannelFactoryInitialize(0)

        self._sport_client = SportClient()
        self._sport_client.SetTimeout(10.0)
        self._sport_client.Init()
        self._command_lock = threading.Lock()

    def call(self, api_id: int, parameter: dict[str, Any]) -> int:
        with self._command_lock:
            if api_id == int(ApiId.DAMP):
                return int(self._sport_client.Damp())
            if api_id == int(ApiId.BALANCESTAND):
                return int(self._sport_client.BalanceStand())
            if api_id == int(ApiId.STOPMOVE):
                return int(self._sport_client.StopMove())
            if api_id == int(ApiId.STANDUP):
                return int(self._sport_client.StandUp())
            if api_id == int(ApiId.STANDDOWN):
                return int(self._sport_client.StandDown())
            if api_id == int(ApiId.RECOVERYSTAND):
                return int(self._sport_client.RecoveryStand())
            if api_id == int(ApiId.SIT):
                return int(self._sport_client.Sit())
            if api_id == int(ApiId.MOVE):
                return int(
                    self._sport_client.Move(
                        float(parameter.get("x", 0.0)),
                        float(parameter.get("y", 0.0)),
                        float(parameter.get("z", 0.0)),
                    )
                )
        raise ValueError(f"unsupported api_id for sdk control: {api_id}")

    def health_snapshot(self) -> dict[str, Any]:
        return {"backend": "sdk2", "sdk_python_path": True}


class Ros2TopicMotionController(MotionController):
    def __init__(self, node: Node, require_subscriber: bool) -> None:
        try:
            from unitree_api.msg import Request, RequestHeader, RequestIdentity
        except ImportError as exc:
            raise RuntimeError(f"failed to import unitree_api ROS2 messages: {exc}") from exc

        self._request_cls = Request
        self._header_cls = RequestHeader
        self._identity_cls = RequestIdentity
        self._publisher: Publisher = node.create_publisher(Request, "/api/sport/request", 10)
        self._node = node
        self._require_subscriber = require_subscriber
        self._publish_lock = threading.Lock()

    def _wait_for_subscriber(self, timeout_s: float = 8.0) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if int(self._publisher.get_subscription_count()) > 0:
                return True
            time.sleep(0.05)
        return int(self._publisher.get_subscription_count()) > 0

    def can_accept_motion(self) -> tuple[bool, str | None]:
        if not self._require_subscriber:
            return True, None
        self._wait_for_subscriber(timeout_s=8.0)
        subs = int(self._publisher.get_subscription_count())
        if subs > 0:
            return True, None
        return False, "no_subscriber_on_/api/sport/request"

    def call(self, api_id: int, parameter: dict[str, Any]) -> int:
        ok, reason = self.can_accept_motion()
        if not ok:
            raise RuntimeError(reason or "ros2_motion_controller_unavailable")

        request = self._request_cls()
        request.header = self._header_cls(identity=self._identity_cls(api_id=int(api_id)))
        request.parameter = json.dumps(parameter)
        with self._publish_lock:
            self._publisher.publish(request)
            # 某些可行的 ROS2 实现会在 publish 后显式驱动一次事件循环。
            # 当前节点本身在 rclpy.spin 中运行，这里仅保留一个很短的刷新窗口。
            time.sleep(0.05)
        return 0

    def health_snapshot(self) -> dict[str, Any]:
        return {
            "backend": "ros2_topic",
            "topic": "/api/sport/request",
            "subscriber_count": int(self._publisher.get_subscription_count()),
            "require_subscriber": self._require_subscriber,
        }


class Go2MotionNode(Node):
    def __init__(
        self,
        server_host: str,
        server_port: int,
        motion_backend: str,
        sdk_python_path: str,
        network_interface: str,
        require_subscriber: bool,
    ) -> None:
        super().__init__("go2_motion_node")
        self.server_url = f"ws://{server_host}:{server_port}"
        self.running = True
        self.websocket = None
        self.ws_loop: asyncio.AbstractEventLoop | None = None
        self.motion_backend_name, self.motion_controller = self._build_motion_controller(
            motion_backend=motion_backend,
            sdk_python_path=sdk_python_path,
            network_interface=network_interface,
            require_subscriber=require_subscriber,
        )

        self.create_subscription(PointCloud2, "/livox/lidar", self.lidar_callback, 10)
        self.create_subscription(Odometry, "/utlidar/robot_odom", self.odometry_callback, 10)
        self.get_logger().info(f"动作控制后端: {self.motion_backend_name}")
        self.get_logger().info("里程计与点云仍用于闭环位姿估计和本地避障。")

        self.latest_cloud: PointCloud2 | None = None
        self.current_pose: Odometry | None = None
        self.total_drift_2d = np.zeros(2, dtype=float)
        self.started_motion_once = False

        t_raw = np.array(
            [
                [0.9743701, 0.0, -0.2249511, 0.1870],
                [0.0, 1.0, 0.0, 0.0],
                [0.2249511, 0.0, 0.9743701, 0.0803],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.t_lidar_calib = np.linalg.inv(t_raw)

        self.motion_lock = threading.Lock()
        self.active_motion: MotionGoal | None = None

        self.ws_thread = threading.Thread(target=self._run_websocket_client, daemon=True)
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.ws_thread.start()
        self.control_thread.start()

    def destroy_node(self) -> bool:
        self.running = False
        try:
            self._send_robot_command(int(ApiId.STOPMOVE), {})
        except Exception:
            pass
        return super().destroy_node()

    def _build_motion_controller(
        self,
        motion_backend: str,
        sdk_python_path: str,
        network_interface: str,
        require_subscriber: bool,
    ) -> tuple[str, MotionController]:
        if motion_backend == "sdk2":
            return "sdk2", Go2SportClientAdapter(sdk_python_path=sdk_python_path, network_interface=network_interface)
        if motion_backend == "ros2_topic":
            return "ros2_topic", Ros2TopicMotionController(self, require_subscriber=require_subscriber)
        if motion_backend != "auto":
            raise RuntimeError(f"unsupported motion backend: {motion_backend}")

        sdk_error: Exception | None = None
        try:
            return "sdk2", Go2SportClientAdapter(sdk_python_path=sdk_python_path, network_interface=network_interface)
        except Exception as exc:
            sdk_error = exc
            self.get_logger().warn(f"sdk2 后端初始化失败，回退到 ros2_topic: {exc}")
        try:
            return "ros2_topic", Ros2TopicMotionController(self, require_subscriber=require_subscriber)
        except Exception as exc:
            raise RuntimeError(f"auto backend failed; sdk2={sdk_error}; ros2_topic={exc}") from exc

    def lidar_callback(self, msg: PointCloud2) -> None:
        self.latest_cloud = msg

    def odometry_callback(self, msg: Odometry) -> None:
        self.current_pose = msg

    def _run_websocket_client(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.ws_loop = loop
        loop.run_until_complete(self._connect_and_listen())

    async def _connect_and_listen(self) -> None:
        while self.running:
            try:
                async with websockets.connect(self.server_url, ping_interval=10) as websocket:
                    self.websocket = websocket
                    self.get_logger().info("已连接主机动作控制端。")
                    self._send_status(
                        {
                            "event": "controller_ready",
                            "backend": self.motion_backend_name,
                            "health": self.motion_controller.health_snapshot(),
                        }
                    )
                    async for message in websocket:
                        await self._handle_message(message)
            except Exception as exc:
                self.get_logger().error(f"动作 websocket 连接失败: {exc}，5 秒后重试。")
                await asyncio.sleep(5.0)
            finally:
                self.websocket = None

    async def _handle_message(self, message: str) -> None:
        try:
            payload = json.loads(message)
            api_id = payload.get("api_id")
            parameter = payload.get("parameter", {})
            binary = payload.get("binary")
            if api_id is not None:
                self.process_command(int(api_id), parameter, binary=binary)
        except Exception as exc:
            self.get_logger().error(f"处理动作消息失败: {exc}")

    def process_command(self, api_id: int, parameter: dict[str, Any], binary: list[int] | None = None) -> None:
        del binary
        command_id = parameter.get("command_id")
        if api_id == int(ApiId.MOVE):
            ok, reason = self.motion_controller.can_accept_motion()
            if not ok:
                self._send_status(
                    {
                        "event": "command_rejected",
                        "command_id": command_id,
                        "reason": reason or "motion_controller_unavailable",
                        "health": self.motion_controller.health_snapshot(),
                    }
                )
                return
            motion = self._build_motion_goal(parameter)
            if motion is None:
                self._send_status(
                    {
                        "event": "command_rejected",
                        "command_id": command_id,
                        "reason": "motion_goal_not_created",
                    }
                )
                return
            self._set_active_motion(motion)
            self.started_motion_once = True
            self.get_logger().info(
                f"启动闭环动作: kind={motion.kind}, target_disp={motion.target_displacement_body.round(3)}, "
                f"target_yaw_deg={np.degrees(motion.target_yaw_rad):.2f}"
            )
            self._send_status(
                {
                    "event": "motion_started",
                    "command_id": command_id,
                    "kind": motion.kind,
                    "target_displacement_body": motion.target_displacement_body.tolist(),
                    "target_yaw_rad": motion.target_yaw_rad,
                }
            )
            return

        if api_id == int(ApiId.STOPMOVE):
            self._cancel_active_motion("external_stop")
            self._send_status({"event": "stop_ack", "command_id": command_id})
            return

        if api_id == int(ApiId.BALANCESTAND):
            self.total_drift_2d[:] = 0.0
        if api_id in {int(ApiId.STANDDOWN), int(ApiId.DAMP)} and self.started_motion_once:
            drift = np.linalg.norm(self.total_drift_2d)
            self.get_logger().info(f"累计平面漂移 {self.total_drift_2d.round(3)} m, norm={drift:.3f} m")

        try:
            self._send_robot_command(api_id, parameter)
        except Exception as exc:
            self._send_status(
                {
                    "event": "command_rejected",
                    "command_id": command_id,
                    "reason": str(exc),
                    "health": self.motion_controller.health_snapshot(),
                }
            )
            return
        self._send_status({"event": "command_forwarded", "command_id": command_id, "api_id": api_id})

    def _control_loop(self) -> None:
        period = 1.0 / CONTROL_HZ
        while self.running:
            time.sleep(period)
            motion = self._get_active_motion()
            if motion is None:
                continue
            try:
                self._step_active_motion(motion)
            except Exception as exc:
                self.get_logger().error(f"闭环动作异常: {exc}")
                self._cancel_active_motion("controller_exception")

    def _get_active_motion(self) -> MotionGoal | None:
        with self.motion_lock:
            return self.active_motion

    def _set_active_motion(self, motion: MotionGoal | None) -> None:
        with self.motion_lock:
            self.active_motion = motion

    def _cancel_active_motion(self, reason: str) -> None:
        motion = self._get_active_motion()
        if motion is None:
            self._send_robot_command(int(ApiId.STOPMOVE), {})
            return
        self._finish_motion(motion, reason=reason, send_stop=True)

    def _finish_motion(self, motion: MotionGoal, reason: str, send_stop: bool) -> None:
        if send_stop:
            self._send_robot_command(int(ApiId.STOPMOVE), {})
        if self.current_pose is not None:
            actual_body_disp, actual_yaw = self._motion_progress(motion.start_pose, self.current_pose)
            pos_error_body = actual_body_disp - motion.target_displacement_body
            yaw_error_rad = actual_yaw - motion.target_yaw_rad
            if motion.kind != "rotation":
                error_world = self._rotation_from_pose(motion.start_pose).apply(pos_error_body)
                self.total_drift_2d += error_world[0:2]
            self.get_logger().info(
                f"动作结束[{reason}] 实际位移={actual_body_disp[0:2].round(3)}m "
                f"位移误差={np.linalg.norm(pos_error_body[0:2]):.3f}m "
                f"实际转角={np.degrees(actual_yaw):.2f}° "
                f"转角误差={np.degrees(yaw_error_rad):.2f}°"
            )
            self._send_status(
                {
                    "event": "motion_finished",
                    "command_id": motion.parameter.get("command_id"),
                    "reason": reason,
                    "actual_body_disp": actual_body_disp.tolist(),
                    "actual_yaw_rad": actual_yaw,
                    "pos_error_body": pos_error_body.tolist(),
                    "yaw_error_rad": yaw_error_rad,
                }
            )
        self._set_active_motion(None)

    def _step_active_motion(self, motion: MotionGoal) -> None:
        pose = self.current_pose
        if pose is None:
            self.get_logger().warn("尚未收到里程计，等待闭环动作启动。")
            return

        actual_body_disp, actual_yaw = self._motion_progress(motion.start_pose, pose)
        remaining_disp = motion.target_displacement_body - actual_body_disp
        remaining_yaw = motion.target_yaw_rad - actual_yaw

        if motion.kind == "rotation":
            if abs(np.degrees(remaining_yaw)) <= YAW_TOLERANCE_DEG:
                self._finish_motion(motion, reason="rotation_target_reached", send_stop=True)
                return
            cmd = {"x": 0.0, "y": 0.0, "z": float(np.sign(remaining_yaw) * abs(motion.turn_speed_z))}
            self._send_robot_command(int(ApiId.MOVE), cmd)
            return

        forward_remaining = float(remaining_disp[0])
        lateral_remaining = float(remaining_disp[1])
        if abs(forward_remaining) <= FORWARD_TOLERANCE_M and abs(lateral_remaining) <= LATERAL_TOLERANCE_M:
            self._finish_motion(motion, reason="translation_target_reached", send_stop=True)
            return

        cmd = self._compute_safe_translation_command(motion, forward_remaining, lateral_remaining)
        if cmd is None:
            self._send_robot_command(int(ApiId.STOPMOVE), {})
            return
        self._send_robot_command(int(ApiId.MOVE), cmd)

    def _compute_safe_translation_command(
        self,
        motion: MotionGoal,
        forward_remaining: float,
        lateral_remaining: float,
    ) -> dict[str, float] | None:
        xyz = self.transform_cloud(self.latest_cloud) if self.latest_cloud is not None else None
        if xyz is None:
            return {"x": float(np.sign(forward_remaining) * motion.forward_speed_x), "y": 0.0, "z": 0.0}

        front_clear = self.region_clear(xyz, (0.0, 0.55), (-0.22, 0.22))
        left_clear = self.region_clear(xyz, (0.0, 0.45), (0.18, 0.42))
        right_clear = self.region_clear(xyz, (0.0, 0.45), (-0.42, -0.18))
        forward_sign = 1.0 if forward_remaining >= 0.0 else -1.0

        if front_clear:
            lateral_cmd = 0.0
            if abs(lateral_remaining) > LATERAL_TOLERANCE_M:
                lateral_cmd = float(np.clip(lateral_remaining * 1.8, -motion.side_speed_y, motion.side_speed_y))
            return {"x": forward_sign * motion.forward_speed_x, "y": lateral_cmd, "z": 0.0}

        if left_clear and (lateral_remaining >= 0.0 or not right_clear):
            return {"x": 0.12 * forward_sign, "y": motion.side_speed_y, "z": 0.0}
        if right_clear:
            return {"x": 0.12 * forward_sign, "y": -motion.side_speed_y, "z": 0.0}
        return None

    def _build_motion_goal(self, parameter: dict[str, Any]) -> MotionGoal | None:
        if self.current_pose is None:
            self.get_logger().error("尚未收到里程计，无法启动动作。")
            return None

        x_vel = float(parameter.get("x", 0.0))
        y_vel = float(parameter.get("y", 0.0))
        z_vel = float(parameter.get("z", 0.0))

        if abs(z_vel) > 1e-3 and abs(x_vel) < 1e-3 and abs(y_vel) < 1e-3:
            target_yaw_deg = float(parameter.get("target_yaw_deg", DEFAULT_TURN_ANGLE_DEG))
            target_yaw_rad = np.deg2rad(abs(target_yaw_deg)) * (1.0 if z_vel > 0 else -1.0)
            return MotionGoal(
                kind="rotation",
                parameter=dict(parameter),
                target_displacement_body=np.zeros(3, dtype=float),
                target_yaw_rad=float(target_yaw_rad),
                forward_speed_x=0.0,
                side_speed_y=0.0,
                turn_speed_z=max(0.1, abs(z_vel)),
                start_pose=self.current_pose,
                started_at=time.time(),
            )

        if abs(x_vel) < 1e-3 and abs(y_vel) < 1e-3:
            self.get_logger().warn("收到零速度 MOVE，已忽略。")
            return None

        target_forward = float(parameter.get("target_distance_m", DEFAULT_FORWARD_DISTANCE_M))
        target_lateral = float(parameter.get("target_lateral_distance_m", DEFAULT_LATERAL_DISTANCE_M))
        target_disp = np.array(
            [
                0.0 if abs(x_vel) < 1e-3 else np.sign(x_vel) * abs(target_forward),
                0.0 if abs(y_vel) < 1e-3 else np.sign(y_vel) * abs(target_lateral),
                0.0,
            ],
            dtype=float,
        )
        return MotionGoal(
            kind="translation",
            parameter=dict(parameter),
            target_displacement_body=target_disp,
            target_yaw_rad=0.0,
            forward_speed_x=max(0.08, abs(x_vel) or DEFAULT_FORWARD_SPEED_X),
            side_speed_y=max(0.08, abs(y_vel) or DEFAULT_SIDE_SPEED_Y),
            turn_speed_z=max(0.1, abs(z_vel) or DEFAULT_TURN_SPEED_Z),
            start_pose=self.current_pose,
            started_at=time.time(),
        )

    def _motion_progress(self, start_pose: Odometry, end_pose: Odometry) -> tuple[np.ndarray, float]:
        rot_start = self._rotation_from_pose(start_pose)
        rot_end = self._rotation_from_pose(end_pose)
        pos_start = self._position_from_pose(start_pose)
        pos_end = self._position_from_pose(end_pose)
        world_displacement = pos_end - pos_start
        actual_body_disp = rot_start.inv().apply(world_displacement)
        relative_rotation = rot_end * rot_start.inv()
        actual_yaw = relative_rotation.as_euler("xyz", degrees=False)[2]
        return actual_body_disp, float(actual_yaw)

    @staticmethod
    def _rotation_from_pose(pose_msg: Odometry) -> Rotation:
        q = pose_msg.pose.pose.orientation
        return Rotation.from_quat([q.x, q.y, q.z, q.w])

    @staticmethod
    def _position_from_pose(pose_msg: Odometry) -> np.ndarray:
        p = pose_msg.pose.pose.position
        return np.array([p.x, p.y, p.z], dtype=float)

    def transform_cloud(self, msg: PointCloud2 | None) -> np.ndarray | None:
        if msg is None:
            return None
        try:
            dtype_list = [(field.name, np.float32) for field in msg.fields if field.datatype == PointField.FLOAT32]
            if not dtype_list:
                return None
            arr = np.frombuffer(msg.data, dtype=np.dtype(dtype_list))
            if "x" not in arr.dtype.names:
                return None
            mask = np.isfinite(arr["x"]) & np.isfinite(arr["y"]) & np.isfinite(arr["z"])
            arr = arr[mask]
            xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=-1)
            xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1)
            return (self.t_lidar_calib @ xyz_h.T).T[:, :3]
        except Exception as exc:
            self.get_logger().error(f"点云转换失败: {exc}")
            return None

    @staticmethod
    def region_clear(
        xyz: np.ndarray | None,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        z_range: tuple[float, float] = (-0.3, 0.25),
        threshold: int = 20,
    ) -> bool:
        if xyz is None or len(xyz) == 0:
            return False
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        mask = (
            (x > x_range[0])
            & (x < x_range[1])
            & (y > y_range[0])
            & (y < y_range[1])
            & (z > z_range[0])
            & (z < z_range[1])
        )
        return int(np.sum(mask)) < threshold

    def _send_robot_command(self, api_id: int, parameter: dict[str, Any]) -> None:
        result = self.motion_controller.call(api_id, parameter)
        if result != 0:
            raise RuntimeError(f"sdk api {api_id} returned non-zero code {result}")

    def _send_status(self, payload: dict[str, Any]) -> None:
        if self.websocket is None or self.ws_loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self.websocket.send(json.dumps(payload, ensure_ascii=False)),
                self.ws_loop,
            ).result(timeout=1.0)
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-ip", default=SERVER_HOST_IP)
    parser.add_argument("--server-port", type=int, default=SERVER_PORT)
    parser.add_argument("--motion-backend", choices=("auto", "sdk2", "ros2_topic"), default="auto")
    parser.add_argument("--sdk-python-path", default=SDK_PYTHON_PATH)
    parser.add_argument("--network-interface", default="")
    parser.add_argument("--require-subscriber", action="store_true")
    return parser.parse_args()


def main(args=None) -> None:
    cli = parse_args()
    rclpy.init(args=args)
    node = Go2MotionNode(
        server_host=cli.server_ip,
        server_port=cli.server_port,
        motion_backend=cli.motion_backend,
        sdk_python_path=cli.sdk_python_path,
        network_interface=cli.network_interface,
        require_subscriber=cli.require_subscriber,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("接收到中断信号，动作服务退出。")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
