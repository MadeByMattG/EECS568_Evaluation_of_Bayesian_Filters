import yaml

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from world.world2d import *
from utils.Landmark import *

class marker_publisher:
    def __init__(self, world, node):

        self.node_ = node

        # load params
        with open("config/settings.yaml", 'r') as stream:
            param = yaml.safe_load(stream)

        marker_topic = param['landmark_topic']

        self.frame_id = param['marker_frame_id']
        self.world_dim = param['world_dimension']

        self.world = world

        qos = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )
        self.pub = self.node_.create_publisher(MarkerArray, marker_topic, qos)

        # Publish static landmark layout immediately and keep refreshing
        self._timer = self.node_.create_timer(1.0, self._publish_static)
        self._publish_static()


    def _publish_static(self):
        self.publish_landmarks([])

    def publish_landmarks(self, observed_tag_ids):
        # observed_tag_ids: list of 1-indexed tag IDs seen in this correction step.
        # A cube lights up if any of its face tags were observed.

        markerArray = MarkerArray()
        observed_set = set(observed_tag_ids)

        for cube_id, pos, tag_ids in self.world.getCubes():
            observed = any(t in observed_set for t in tag_ids)

            marker = Marker()
            marker.id = cube_id
            marker.header.stamp = self.node_.get_clock().now().to_msg()
            marker.header.frame_id = self.frame_id
            marker.type = marker.CUBE
            marker.action = marker.ADD
            marker.scale.x = 0.10
            marker.scale.y = 0.10
            marker.scale.z = 0.10
            marker.color.a = 1.0
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = float(pos[0])
            marker.pose.position.y = float(pos[1])
            marker.pose.position.z = 0.0

            if observed:  # maize
                marker.color.r = 255.0 / 255.0
                marker.color.g = 203.0 / 255.0
                marker.color.b = 5.0 / 255.0
            else:  # blue
                marker.color.r = 0.0
                marker.color.g = 39.0 / 255.0
                marker.color.b = 76.0 / 255.0

            markerArray.markers.append(marker)

        self.pub.publish(markerArray)


