#!/usr/bin/env python3
"""
Publish robot-frame tag measurements from reprocessed AprilTag detections.

Computes tag positions directly from the detected pixel corners and camera intrinsics
using cv2.solvePnP.

The static base_link-camera transform is required (published by
reprocess_bag_from_original_detections.sh or the legacy reprocessing scripts)
so the base-link-frame PointCloud is identical in format to the original node.

Subscribes:
  /detections_reprocessed   (apriltag_msgs/AprilTagDetectionArray)
  /camera/camera_info       (sensor_msgs/CameraInfo)  - once at startup
  /tf_static                (tf2_msgs/TFMessage)       - once at startup

Publishes:
  /tag_measurements_base    (sensor_msgs/PointCloud, frame_id=base_link)
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import time

import cv2
import numpy as np
import rclpy
from apriltag_msgs.msg import AprilTagDetectionArray
from geometry_msgs.msg import Point32
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, ChannelFloat32, PointCloud
from tf2_msgs.msg import TFMessage


MEASUREMENT_TOPIC = "/tag_measurements_base"

# 3-D corners of a square tag (tag-frame, z = 0 plane).
# Order matches the apriltag library's p[0..3] convention, which is what
# apriltag_ros feeds to its own solvePnP call internally.
_HALF = 1.0  # placeholder - scaled at runtime by tag_size / 2
_OBJECT_POINTS_UNIT = np.array(
    [
        [-_HALF, -_HALF, 0.0],
        [+_HALF, -_HALF, 0.0],
        [+_HALF, +_HALF, 0.0],
        [-_HALF, +_HALF, 0.0],
    ],
    dtype=np.float64,
)


def _quat_to_rot(x: float, y: float, z: float, w: float) -> np.ndarray:
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


@dataclass
class _Pending:
    msg: AprilTagDetectionArray
    created_at: float


class TagMeasurementNode(Node):
    def __init__(self) -> None:
        super().__init__("tag_measurement_node")

        self.declare_parameter("tag_size", 0.077)
        tag_size = (
            self.get_parameter("tag_size").get_parameter_value().double_value
        )
        half = tag_size / 2.0
        self._obj_pts: np.ndarray = _OBJECT_POINTS_UNIT * half  # (4, 3)

        # Filled on first callback; immutable after that.
        self._R_bc: np.ndarray | None = None   # rotation    base_link <- camera
        self._t_bc: np.ndarray | None = None   # translation base_link <- camera
        self._K: np.ndarray | None = None      # 3x3 camera matrix
        self._D: np.ndarray | None = None      # distortion coefficients

        # Pending detections buffered until startup data arrives.
        self._pending: OrderedDict[tuple, _Pending] = OrderedDict()
        self._pending_timeout_s = 5.0
        self._max_pending = 200

        # QoS profiles
        latched = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # Publishers / subscribers
        self._pub = self.create_publisher(PointCloud, MEASUREMENT_TOPIC, 10)

        self.create_subscription(
            AprilTagDetectionArray,
            "/detections_reprocessed",
            self._detection_cb,
            10,
        )
        # camera_info: the bag publishes with VOLATILE; use default (VOLATILE)
        # QoS so the subscription is compatible.
        self.create_subscription(
            CameraInfo,
            "/camera/camera_info",
            self._camera_info_cb,
            10,
        )
        # tf_static is TRANSIENT_LOCAL we receive the latest even if it was
        # published before this node started.
        self.create_subscription(
            TFMessage,
            "/tf_static",
            self._tf_static_cb,
            latched,
        )

        # Retry pending detections every 50 ms.
        self.create_timer(0.05, self._flush_pending)

    # Startup data callbacks

    def _tf_static_cb(self, msg: TFMessage) -> None:
        if self._R_bc is not None:
            return
        for tr in msg.transforms:
            if tr.header.frame_id == "base_link" and tr.child_frame_id == "camera":
                r = tr.transform.rotation
                t = tr.transform.translation
                self._R_bc = _quat_to_rot(r.x, r.y, r.z, r.w)
                self._t_bc = np.array([t.x, t.y, t.z])
                self.get_logger().info("Static TF base_link->camera received.")
                break

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        if self._K is not None:
            return
        self._K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self._D = np.array(msg.d, dtype=np.float64)
        self.get_logger().info(
            f"Camera intrinsics received: "
            f"fx={self._K[0, 0]:.1f}  fy={self._K[1, 1]:.1f}  "
            f"cx={self._K[0, 2]:.1f}  cy={self._K[1, 2]:.1f}  "
            f"distortion_coeffs={len(self._D)}"
        )

    # Detection pipeline

    def _detection_cb(self, msg: AprilTagDetectionArray) -> None:
        if not msg.detections:
            return
        key = (
            msg.header.stamp.sec,
            msg.header.stamp.nanosec,
            tuple(d.id for d in msg.detections),
        )
        if key not in self._pending:
            self._pending[key] = _Pending(msg=msg, created_at=time.monotonic())
            # Evict oldest if buffer overflows.
            while len(self._pending) > self._max_pending:
                old_key, old = self._pending.popitem(last=False)
                self.get_logger().warn(
                    f"Dropping stale detection "
                    f"stamp={old.msg.header.stamp.sec}."
                    f"{old.msg.header.stamp.nanosec:09d} "
                    f"ids={list(old_key[2])} reason=pending_overflow"
                )
        self._publish_if_ready(key)

    def _flush_pending(self) -> None:
        for key in list(self._pending):
            self._publish_if_ready(key)

    def _ready(self) -> bool:
        return self._R_bc is not None and self._K is not None

    def _publish_if_ready(self, key: tuple) -> None:
        entry = self._pending.get(key)
        if entry is None:
            return

        age_s = time.monotonic() - entry.created_at

        if not self._ready():
            if age_s > self._pending_timeout_s:
                reason = (
                    "missing_camera_info"
                    if self._K is None
                    else "missing_base_camera_tf"
                )
                self.get_logger().warn(
                    f"Dropping detection "
                    f"stamp={entry.msg.header.stamp.sec}."
                    f"{entry.msg.header.stamp.nanosec:09d} "
                    f"ids={[d.id for d in entry.msg.detections]} "
                    f"reason={reason}"
                )
                self._pending.pop(key, None)
            return

        msg = entry.msg
        ids: list[float] = []
        points: list[Point32] = []
        margins: list[float] = []
        hammings: list[float] = []
        goodnesses: list[float] = []
        pnp_failed: list[int] = []

        for det in msg.detections:
            p_cam = self._solve_pose(det)
            if p_cam is None:
                pnp_failed.append(det.id)
                continue

            p_base = self._R_bc @ p_cam + self._t_bc
            ids.append(float(det.id))
            points.append(Point32(x=float(p_base[0]), y=float(p_base[1]), z=float(p_base[2])))
            margins.append(float(det.decision_margin))
            hammings.append(float(det.hamming))
            goodnesses.append(float(det.goodness))

        if pnp_failed:
            self.get_logger().warn(
                f"solvePnP failed for tag(s) {pnp_failed} at "
                f"stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}"
            )

        self._pending.pop(key, None)

        if not ids:
            return

        pc = PointCloud()
        pc.header = msg.header
        pc.header.frame_id = "base_link"
        pc.points = points
        pc.channels = [
            ChannelFloat32(name="id",              values=ids),
            ChannelFloat32(name="decision_margin", values=margins),
            ChannelFloat32(name="hamming",         values=hammings),
            ChannelFloat32(name="goodness",        values=goodnesses),
        ]
        self._pub.publish(pc)

    def _solve_pose(self, det) -> np.ndarray | None:
        """
        Return the tag-centre position in the CAMERA frame as a (3,) array,
        or None if solvePnP fails.
        """
        img_pts = np.array([[c.x, c.y] for c in det.corners], dtype=np.float64)
        attempted: list[str] = []

        for flag_name in (
            "SOLVEPNP_IPPE_SQUARE",
            "SOLVEPNP_IPPE",
            "SOLVEPNP_ITERATIVE",
        ):
            ok, _rvec, tvec = cv2.solvePnP(
                self._obj_pts,
                img_pts,
                self._K,
                self._D,
                flags=getattr(cv2, flag_name),
            )
            if not ok:
                attempted.append(f"{flag_name}=fail")
                continue

            t = tvec.flatten()
            attempted.append(f"{flag_name}=z{t[2]:.3f}")
            if t[2] > 0.0:
                return t

        self.get_logger().warn(
            f"solvePnP returned no valid front-facing pose for tag {det.id} "
            f"({', '.join(attempted)})"
        )
        return None


def main() -> None:
    rclpy.init()
    node = TagMeasurementNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
