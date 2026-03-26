#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import socket
import struct
import threading
import time
from typing import Protocol

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2, PointField


SERVER_HOST_IP = "192.168.86.16"
VIDEO_PORT = 5220
STATE_PORT = 5222
OCCUPANCY_PORT = 5223
DEPTH_PORT = 5224
RECONNECT_DELAY_S = 5.0
VIDEO_INDEX = 4
VIDEO_FPS = 15.0
VIDEO_BACKEND = "auto"
DEPTH_FPS = 4.0
SELF_OCCUPANCY_CLEAR_FORWARD_M = 0.9
SELF_OCCUPANCY_CLEAR_LATERAL_M = 0.2

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


class FrameSource(Protocol):
    def read(self) -> tuple[bool, np.ndarray | None]:
        ...

    def latest_depth_m(self) -> np.ndarray | None:
        ...

    def release(self) -> None:
        ...


class OpenCVCameraSource:
    def __init__(self, video_index: int) -> None:
        self.cap = cv2.VideoCapture(video_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头索引 {video_index}")

    def read(self) -> tuple[bool, np.ndarray | None]:
        return self.cap.read()

    def latest_depth_m(self) -> np.ndarray | None:
        return None

    def release(self) -> None:
        self.cap.release()


class RealSenseColorSource:
    _STREAM_CANDIDATES = [
        (848, 480, 30),
        (640, 480, 30),
        (640, 480, 15),
        (480, 270, 15),
        (424, 240, 15),
    ]

    def __init__(self) -> None:
        if rs is None:
            raise RuntimeError("pyrealsense2 未安装，无法使用 realsense 视频源")
        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color)
        self.profile = None
        self.depth_scale = 0.001
        self._latest_depth_m: np.ndarray | None = None
        last_error: Exception | None = None
        for width, height, fps in self._STREAM_CANDIDATES:
            config = rs.config()
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            try:
                self.profile = self.pipeline.start(config)
                test_frames = self.pipeline.wait_for_frames(timeout_ms=3000)
                if test_frames:
                    self.width = width
                    self.height = height
                    self.fps = fps
                    self.depth_scale = (
                        self.profile.get_device().first_depth_sensor().get_depth_scale()
                    )
                    return
            except Exception as exc:
                last_error = exc
                try:
                    self.pipeline.stop()
                except Exception:
                    pass
                self.pipeline = rs.pipeline()
        raise RuntimeError(f"RealSense 管线启动失败: {last_error}")

    def read(self) -> tuple[bool, np.ndarray | None]:
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=2000)
        except Exception:
            return False, None
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame:
            return False, None
        if depth_frame:
            depth_raw = np.asanyarray(depth_frame.get_data())
            self._latest_depth_m = depth_raw.astype(np.float32) * float(self.depth_scale)
        return True, np.asanyarray(color_frame.get_data())

    def latest_depth_m(self) -> np.ndarray | None:
        return None if self._latest_depth_m is None else self._latest_depth_m.copy()

    def release(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass


class NavigationDataServer(Node):
    def __init__(
        self,
        server_host: str,
        video_port: int,
        state_port: int,
        occupancy_port: int,
        depth_port: int,
        video_index: int,
        video_backend: str,
    ) -> None:
        super().__init__("navigation_data_server")
        self.server_host = server_host
        self.video_port = video_port
        self.state_port = state_port
        self.occupancy_port = occupancy_port
        self.depth_port = depth_port
        self.video_index = video_index
        self.video_backend = video_backend
        self.running = True

        self.state_sock: socket.socket | None = None
        self.occupancy_sock: socket.socket | None = None
        self.depth_sock: socket.socket | None = None
        self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.latest_occupancy: np.ndarray | None = None
        self.latest_depth: np.ndarray | None = None
        self.initial_pos: np.ndarray | None = None
        self.initial_rot: Rotation | None = None
        self.last_odom_at = 0.0
        self.last_lidar_at = 0.0
        self.last_frame_at = 0.0

        self.create_subscription(Odometry, "/utlidar/robot_odom", self.odometry_callback, 10)
        self.create_subscription(PointCloud2, "/livox/lidar", self.lidar_callback, 10)
        self.get_logger().info("订阅里程计 /utlidar/robot_odom 和雷达 /livox/lidar。")
        self.create_timer(5.0, self._log_health)

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

        self.state_thread = threading.Thread(target=self._state_network_loop, daemon=True)
        self.occupancy_thread = threading.Thread(target=self._occupancy_network_loop, daemon=True)
        self.depth_thread = threading.Thread(target=self._depth_network_loop, daemon=True)
        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.state_thread.start()
        self.occupancy_thread.start()
        self.depth_thread.start()
        self.video_thread.start()

    def destroy_node(self) -> None:
        self.get_logger().info("关闭数据服务网络线程...")
        self.running = False
        if self.state_sock is not None:
            self.state_sock.close()
        if self.occupancy_sock is not None:
            self.occupancy_sock.close()
        if self.depth_sock is not None:
            self.depth_sock.close()
        self.video_socket.close()
        self.state_thread.join(timeout=2.0)
        self.occupancy_thread.join(timeout=2.0)
        self.depth_thread.join(timeout=2.0)
        self.video_thread.join(timeout=2.0)
        super().destroy_node()

    def odometry_callback(self, msg: Odometry) -> None:
        self.last_odom_at = time.time()
        if self.state_sock is None:
            return

        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        current_pos = np.array([pos.x, pos.y, pos.z], dtype=float)
        current_rot = Rotation.from_quat([ori.x, ori.y, ori.z, ori.w])

        if self.initial_pos is None or self.initial_rot is None:
            self.initial_pos = current_pos
            self.initial_rot = current_rot
            self.get_logger().info("已记录导航相对坐标初始位姿。")

        relative_rot = current_rot * self.initial_rot.inv()
        relative_pos = self.initial_rot.inv().apply(current_pos - self.initial_pos)
        relative_yaw = relative_rot.as_euler("xyz", degrees=False)[2]

        payload = struct.pack("!ddd", float(relative_pos[0]), float(relative_pos[1]), float(relative_yaw))
        message = struct.pack("!I", len(payload)) + payload
        try:
            self.state_sock.sendall(message)
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.get_logger().warn("状态连接断开，等待主机端重连。")
            self.state_sock.close()
            self.state_sock = None

    def lidar_callback(self, msg: PointCloud2) -> None:
        self.last_lidar_at = time.time()
        xyz = self.transform_cloud(msg)
        if xyz is None:
            return
        self.latest_occupancy = self.build_occupancy(xyz)

    def _state_network_loop(self) -> None:
        while self.running:
            try:
                self.get_logger().info(f"连接主机状态端口 {(self.server_host, self.state_port)}...")
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((self.server_host, self.state_port))
                self.state_sock = sock
                self.get_logger().info("状态上行已连接。")
                while self.running and self.state_sock is not None:
                    time.sleep(1.0)
                    self.state_sock.sendall(struct.pack("!I", 0))
            except Exception as exc:
                self.get_logger().warn(f"状态上行异常: {exc}，{RECONNECT_DELAY_S:.0f}s 后重试。")
                if self.state_sock is not None:
                    self.state_sock.close()
                self.state_sock = None
                time.sleep(RECONNECT_DELAY_S)

    def _occupancy_network_loop(self) -> None:
        while self.running:
            try:
                self.get_logger().info(f"连接主机占据图端口 {(self.server_host, self.occupancy_port)}...")
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((self.server_host, self.occupancy_port))
                self.occupancy_sock = sock
                self.get_logger().info("占据图上行已连接。")
                while self.running and self.occupancy_sock is not None:
                    occupancy = self.latest_occupancy
                    if occupancy is None:
                        time.sleep(0.2)
                        continue
                    rows, cols = occupancy.shape
                    payload = struct.pack("!II", rows, cols) + occupancy.tobytes()
                    message = struct.pack("!I", len(payload)) + payload
                    self.occupancy_sock.sendall(message)
                    time.sleep(0.2)
            except Exception as exc:
                self.get_logger().warn(f"占据图上行异常: {exc}，{RECONNECT_DELAY_S:.0f}s 后重试。")
                if self.occupancy_sock is not None:
                    self.occupancy_sock.close()
                self.occupancy_sock = None
                time.sleep(RECONNECT_DELAY_S)

    def _depth_network_loop(self) -> None:
        while self.running:
            try:
                self.get_logger().info(f"连接主机深度端口 {(self.server_host, self.depth_port)}...")
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((self.server_host, self.depth_port))
                self.depth_sock = sock
                self.get_logger().info("深度上行已连接。")
                while self.running and self.depth_sock is not None:
                    depth = self.latest_depth
                    if depth is None:
                        time.sleep(1.0 / DEPTH_FPS)
                        continue
                    depth_mm = np.nan_to_num(depth * 1000.0, nan=0.0, posinf=0.0, neginf=0.0).astype(np.uint16)
                    ok, encoded = cv2.imencode(".png", depth_mm)
                    if not ok:
                        time.sleep(1.0 / DEPTH_FPS)
                        continue
                    rows, cols = depth_mm.shape[:2]
                    payload = struct.pack("!II", rows, cols) + encoded.tobytes()
                    message = struct.pack("!I", len(payload)) + payload
                    self.depth_sock.sendall(message)
                    time.sleep(1.0 / DEPTH_FPS)
            except Exception as exc:
                self.get_logger().warn(f"深度上行异常: {exc}，{RECONNECT_DELAY_S:.0f}s 后重试。")
                if self.depth_sock is not None:
                    self.depth_sock.close()
                self.depth_sock = None
                time.sleep(RECONNECT_DELAY_S)

    def _video_loop(self) -> None:
        source = self._create_frame_source()
        if source is None:
            return
        self.get_logger().info(f"视频上行将发送到 {(self.server_host, self.video_port)}。")
        frame_interval = 1.0 / VIDEO_FPS
        try:
            while self.running:
                started_at = time.time()
                ok, frame = source.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                self.last_frame_at = time.time()
                self.latest_depth = source.latest_depth_m()
                ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                if not ok:
                    continue
                data = encoded.tobytes()
                frame_id = int(time.time() * 1000) & 0xFFFFFFFF
                max_size = 60000
                total = max(1, math.ceil(len(data) / max_size))
                for idx in range(total):
                    packet = data[idx * max_size:(idx + 1) * max_size]
                    header = struct.pack("!IHHH", frame_id, total, idx, len(packet))
                    self.video_socket.sendto(header + packet, (self.server_host, self.video_port))
                elapsed = time.time() - started_at
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
        finally:
            source.release()

    def _create_frame_source(self) -> FrameSource | None:
        backend = self.video_backend.lower()
        if backend in {"auto", "realsense"}:
            try:
                source = RealSenseColorSource()
                self.get_logger().info("视频源: RealSense 对齐彩色流。")
                return source
            except Exception as exc:
                if backend == "realsense":
                    self.get_logger().error(f"RealSense 视频源初始化失败: {exc}")
                    return None
                self.get_logger().warn(f"RealSense 视频源不可用，回退到 OpenCV: {exc}")
        try:
            source = OpenCVCameraSource(self.video_index)
            self.get_logger().info(f"视频源: OpenCV VideoCapture(index={self.video_index})。")
            return source
        except Exception as exc:
            self.get_logger().error(f"OpenCV 视频源初始化失败: {exc}")
            return None

    def _log_health(self) -> None:
        now = time.time()
        def age_str(ts: float) -> str:
            if ts <= 0.0:
                return "never"
            return f"{now - ts:.1f}s ago"

        self.get_logger().info(
            "data_server health | "
            f"video_backend={self.video_backend} "
            f"state_connected={self.state_sock is not None} "
            f"occupancy_connected={self.occupancy_sock is not None} "
            f"depth_connected={self.depth_sock is not None} "
            f"last_odom={age_str(self.last_odom_at)} "
            f"last_lidar={age_str(self.last_lidar_at)} "
            f"last_frame={age_str(self.last_frame_at)}"
        )

    def transform_cloud(self, msg: PointCloud2) -> np.ndarray | None:
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
            self.get_logger().warn(f"点云转换失败: {exc}")
            return None

    @staticmethod
    def build_occupancy(
        xyz: np.ndarray,
        resolution_m: float = 0.10,
        forward_m: float = 6.0,
        lateral_m: float = 3.0,
    ) -> np.ndarray:
        rows = int(round((2 * lateral_m) / resolution_m)) + 1
        cols = int(round(forward_m / resolution_m)) + 1
        occupancy = np.zeros((rows, cols), dtype=np.uint8)
        if xyz.size == 0:
            return occupancy
        mask = (
            (xyz[:, 0] >= 0.0)
            & (xyz[:, 0] <= forward_m)
            & (np.abs(xyz[:, 1]) <= lateral_m)
            & (xyz[:, 2] >= -0.3)
            & (xyz[:, 2] <= 0.5)
        )
        pts = xyz[mask]
        if pts.size == 0:
            return occupancy
        col = np.clip(np.round(pts[:, 0] / resolution_m).astype(int), 0, cols - 1)
        row = np.clip(np.round((pts[:, 1] + lateral_m) / resolution_m).astype(int), 0, rows - 1)
        occupancy[row, col] = 1
        # Clear a narrow near-body shadow corridor caused by the robot chassis / sensor mount.
        clear_cols = min(cols, int(round(SELF_OCCUPANCY_CLEAR_FORWARD_M / resolution_m)) + 1)
        clear_half_rows = int(round(SELF_OCCUPANCY_CLEAR_LATERAL_M / resolution_m))
        center_row = rows // 2
        occupancy[
            max(0, center_row - clear_half_rows):min(rows, center_row + clear_half_rows + 1),
            :clear_cols,
        ] = 0
        return occupancy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-ip", default=SERVER_HOST_IP)
    parser.add_argument("--video-port", type=int, default=VIDEO_PORT)
    parser.add_argument("--state-port", type=int, default=STATE_PORT)
    parser.add_argument("--occupancy-port", type=int, default=OCCUPANCY_PORT)
    parser.add_argument("--depth-port", type=int, default=DEPTH_PORT)
    parser.add_argument("--video-index", type=int, default=VIDEO_INDEX)
    parser.add_argument("--video-backend", choices=("auto", "realsense", "opencv"), default=VIDEO_BACKEND)
    return parser.parse_args()


def main(args=None) -> None:
    cli = parse_args()
    rclpy.init(args=args)
    node = NavigationDataServer(
        server_host=cli.server_ip,
        video_port=cli.video_port,
        state_port=cli.state_port,
        occupancy_port=cli.occupancy_port,
        depth_port=cli.depth_port,
        video_index=cli.video_index,
        video_backend=cli.video_backend,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("接收到中断信号，数据服务退出。")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
