#!/usr/bin/env python3
"""
Relay an AprilTag detection topic to another topic.

Default use:
  /detections -> /detections_reprocessed

This is useful when a bag already contains a richer original detection stream
than the local reprocessing pass can recover from the recorded images.
"""

from __future__ import annotations

import rclpy
from apriltag_msgs.msg import AprilTagDetectionArray
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


class DetectionRelayNode(Node):
    def __init__(self) -> None:
        super().__init__("detection_relay_node")

        self.declare_parameter("input_topic", "/detections")
        self.declare_parameter("output_topic", "/detections_reprocessed")

        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value

        self._pub = self.create_publisher(AprilTagDetectionArray, output_topic, 10)
        self._sub = self.create_subscription(
            AprilTagDetectionArray,
            input_topic,
            self._callback,
            10,
        )

        self.get_logger().info(f"Relaying {input_topic} -> {output_topic}")

    def _callback(self, msg: AprilTagDetectionArray) -> None:
        self._pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = DetectionRelayNode()
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
