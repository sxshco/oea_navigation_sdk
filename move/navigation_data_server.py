#!/usr/bin/env python3
import argparse
import base64
import json
import math
import socket
import struct
import threading
import time

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image, PointCloud2, PointField


class NavigationObservationSender(Node):
    def __init__(
        self,
        server_ip: str,
        port: int,
        color_topic: str,
        depth_topic: str,
        lidar_topic: str,
        odom_topic: str,
        publish_hz: float,
    ):
        super().__init__("navigation_observation_sender")
        self.server_ip = server_ip
        self.port = port
        self.publish_interval = 1.0 / max(0.5, publish_hz)
        self.running = True
        self.sock: socket.socket | None = None

        self.lock = threading.Lock()
        self.latest_rgb: np.ndarray | None = None
        self.latest_depth: np.ndarray | None = None
        self.latest_occupancy: np.ndarray | None = None
        self.latest_pose = (0.0, 0.0, 0.0)
        self.initial_pos = None
        self.initial_rot = None

        self.create_subscription(Image, color_topic, self.color_callback, 10)
        self.create_subscription(Image, depth_topic, self.depth_callback, 10)
        self.create_subscription(PointCloud2, lidar_topic, self.lidar_callback, 10)
        self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)

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

        self.sender_thread = threading.Thread(target=self.send_loop, daemon=True)
        self.sender_thread.start()

    def destroy_node(self):
        self.running = False
        if self.sock is not None:
            self.sock.close()
        self.sender_thread.join(timeout=2.0)
        super().destroy_node()

    def color_callback(self, msg: Image):
        frame = self.image_msg_to_array(msg)
        if frame is None:
            return
        with self.lock:
            self.latest_rgb = frame

    def depth_callback(self, msg: Image):
        depth = self.depth_msg_to_meters(msg)
        if depth is None:
            return
        with self.lock:
            self.latest_depth = depth

    def odom_callback(self, msg: Odometry):
        current_pos_vec = msg.pose.pose.position
        current_q_vec = msg.pose.pose.orientation
        current_pos = np.array([current_pos_vec.x, current_pos_vec.y, current_pos_vec.z])
        current_rot = Rotation.from_quat(
            [current_q_vec.x, current_q_vec.y, current_q_vec.z, current_q_vec.w]
        )
        if self.initial_pos is None or self.initial_rot is None:
            self.initial_pos = current_pos
            self.initial_rot = current_rot
        relative_rot = current_rot * self.initial_rot.inv()
        relative_pos = self.initial_rot.inv().apply(current_pos - self.initial_pos)
        with self.lock:
            self.latest_pose = (
                float(relative_pos[0]),
                float(relative_pos[1]),
                float(relative_rot.as_euler("xyz", degrees=False)[2]),
            )

    def lidar_callback(self, msg: PointCloud2):
        xyz = self.pointcloud_to_body(msg)
        if xyz is None:
            return
        occupancy = self.build_occupancy(xyz)
        with self.lock:
            self.latest_occupancy = occupancy

    def send_loop(self):
        while self.running:
            try:
                if self.sock is None:
                    self.sock = socket.create_connection((self.server_ip, self.port), timeout=5)
                    self.get_logger().info(f"connected to host {(self.server_ip, self.port)}")
                payload = self.build_payload()
                if payload is None:
                    time.sleep(self.publish_interval)
                    continue
                data = json.dumps(payload).encode("utf-8")
                self.sock.sendall(struct.pack("!I", len(data)) + data)
                time.sleep(self.publish_interval)
            except Exception as exc:
                self.get_logger().warn(f"stream send failed: {exc}")
                if self.sock is not None:
                    self.sock.close()
                    self.sock = None
                time.sleep(2.0)

    def build_payload(self):
        with self.lock:
            rgb = None if self.latest_rgb is None else self.latest_rgb.copy()
            depth = None if self.latest_depth is None else self.latest_depth.copy()
            occupancy = None if self.latest_occupancy is None else self.latest_occupancy.copy()
            pose = self.latest_pose
        if rgb is None and depth is None and occupancy is None:
            return None
        return {
            "timestamp": time.time(),
            "pose_xy_yaw": pose,
            "rgb_jpeg_base64": None if rgb is None else self.encode_rgb(rgb),
            "depth_png_base64": None if depth is None else self.encode_depth(depth),
            "depth_scale": 0.001,
            "occupancy_base64": None if occupancy is None else base64.b64encode(occupancy.tobytes()).decode("ascii"),
            "occupancy_shape": None if occupancy is None else list(occupancy.shape),
        }

    @staticmethod
    def encode_rgb(rgb: np.ndarray) -> str:
        if rgb.ndim != 3:
            raise ValueError("rgb frame must be HWC")
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ok:
            raise RuntimeError("failed to encode rgb")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    @staticmethod
    def encode_depth(depth_m: np.ndarray) -> str:
        depth_mm = np.nan_to_num(depth_m * 1000.0, nan=0.0, posinf=0.0, neginf=0.0).astype(np.uint16)
        ok, encoded = cv2.imencode(".png", depth_mm)
        if not ok:
            raise RuntimeError("failed to encode depth")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    @staticmethod
    def image_msg_to_array(msg: Image) -> np.ndarray | None:
        if msg.encoding not in {"rgb8", "bgr8"}:
            return None
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if msg.encoding == "bgr8":
            arr = arr[:, :, ::-1]
        return arr.copy()

    @staticmethod
    def depth_msg_to_meters(msg: Image) -> np.ndarray | None:
        if msg.encoding == "16UC1":
            arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            return arr.astype(np.float32) * 0.001
        if msg.encoding == "32FC1":
            return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
        return None

    def pointcloud_to_body(self, msg: PointCloud2) -> np.ndarray | None:
        dtype_list = [(f.name, np.float32) for f in msg.fields if f.datatype == PointField.FLOAT32]
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
        return occupancy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-ip", required=True)
    parser.add_argument("--port", type=int, default=5230)
    parser.add_argument("--color-topic", default="/camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--lidar-topic", default="/livox/lidar")
    parser.add_argument("--odom-topic", default="/utlidar/robot_odom")
    parser.add_argument("--publish-hz", type=float, default=4.0)
    return parser.parse_args()


def main(args=None):
    cli = parse_args()
    rclpy.init(args=args)
    node = NavigationObservationSender(
        server_ip=cli.server_ip,
        port=cli.port,
        color_topic=cli.color_topic,
        depth_topic=cli.depth_topic,
        lidar_topic=cli.lidar_topic,
        odom_topic=cli.odom_topic,
        publish_hz=cli.publish_hz,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
