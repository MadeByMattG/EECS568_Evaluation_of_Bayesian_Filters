
import sys
from tkinter.ttk import Labelframe
sys.path.append('.')

import yaml

from scipy.spatial.transform import Rotation
from scipy.linalg import expm, logm

from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Point
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from system.RobotState import *

class path_publisher:
    def __init__(self, node):

        self.node_ = node

        # load params
        with open("config/settings.yaml", 'r') as stream:
            param = yaml.safe_load(stream)

        pose_topic = param['pose_topic']
        path_topic = param['path_topic']
        gt_path_topic = param['gt_path_topic']
        gt_heading_topic = param.get('gt_heading_topic', 'groundtruth/heading')
        command_path_topic = param['command_path_topic']
        ellipse_topic = param['ellipse_topic']
        self.gt_heading_length = float(param.get('gt_heading_length', 0.18))
        self.gt_heading_shaft_diameter = float(param.get('gt_heading_shaft_diameter', 0.03))
        self.gt_heading_head_diameter = float(param.get('gt_heading_head_diameter', 0.06))
        self.gt_heading_head_length = float(param.get('gt_heading_head_length', 0.06))

        self.path_frame = param['path_frame_id']

        self.filter_name = param['filter_name']
        self.world_dim = param['world_dimension']

        self.path = Path()
        self.path.header.frame_id = self.path_frame

        self.gt_path = Path()
        self.gt_path.header.frame_id = self.path_frame

        self.cmd_path = Path()
        self.cmd_path.header.frame_id = self.path_frame

        self.pose_pub = self.node_.create_publisher(PoseWithCovarianceStamped, pose_topic, 100)
        self.path_pub = self.node_.create_publisher(Path, path_topic, 10)
        self.gt_path_pub = self.node_.create_publisher(Path, gt_path_topic, 10)
        self.gt_heading_pub = self.node_.create_publisher(Marker, gt_heading_topic, 10)
        self.cmd_path_pub = self.node_.create_publisher(Path, command_path_topic, 10)
        self.ellipse_pub = self.node_.create_publisher(Marker, ellipse_topic, 10)

    def _covariance_sqrt(self, cov):
        cov = np.array(cov, dtype=float)
        cov = 0.5 * (cov + cov.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 0.0, None)
        if np.max(eigvals) <= 1e-15:
            return None
        return eigvecs @ np.diag(np.sqrt(eigvals))

    def _planar_covariance(self, state):
        if self.filter_name == "InEKF":
            try:
                cov = state.getCartesianCovariance()
            except Exception:
                cov = state.getCovariance()
        else:
            cov = state.getCovariance()
        cov = np.array(cov, dtype=float)
        cov = 0.5 * (cov + cov.T)
        return cov

    def publish_pose(self, state):

        position = state.getPosition()
        orientation = state.getOrientation()

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = state.getTime()
        msg.header.frame_id = self.path_frame
        msg.pose.pose.position.x = float(position[0])
        msg.pose.pose.position.y = float(position[1])

        if self.world_dim == 2:
            msg.pose.pose.position.z = 0.0
            rot = Rotation.from_euler('z', float(orientation), degrees=False)
            quat = rot.as_quat()    # (x, y, z, w)
            msg.pose.pose.orientation.x = float(quat[0])
            msg.pose.pose.orientation.y = float(quat[1])
            msg.pose.pose.orientation.z = float(quat[2])
            msg.pose.pose.orientation.w = float(quat[3])

            cov = np.zeros((6,6))
            planar_cov = self._planar_covariance(state)
            cov[0:2,0:2] = planar_cov[0:2,0:2]
            if planar_cov.shape[0] >= 3:
                cov[5,5] = planar_cov[2,2]

            if self.filter_name == "InEKF":
                ellipse_line_msg = self.make_ellipse(state)
                if ellipse_line_msg is not None:
                    self.ellipse_pub.publish(ellipse_line_msg)

            # We wish to visualize 3 sigma contour
            msg.pose.covariance = np.reshape(9*cov,(-1,)).tolist()


        self.pose_pub.publish(msg)


    def make_ellipse(self,state):
        G1 = np.array([[0,0,1],
                    [0,0,0],
                    [0,0,0]])

        G2 = np.array([[0,0,0],
                        [0,0,1],
                        [0,0,0]])

        G3 = np.array([[0,-1,0],
                        [1,0,0],
                        [0,0,0]])

        phi = np.arange(-np.pi, np.pi, 0.01)
        circle = np.array([np.cos(phi), np.sin(phi), np.zeros(np.size(phi))]).T

        scale = np.sqrt(7.815)

        mean = state.getState()
        mean_matrix = np.eye(3)
        R = np.array([[np.cos(mean[2]),-np.sin(mean[2])],[np.sin(mean[2]),np.cos(mean[2])]])

        mean_matrix[0:2,0:2] = R
        mean_matrix[0,2] = mean[0]
        mean_matrix[1,2] = mean[1]

        ellipse_line_msg = Marker()
        ellipse_line_msg.type = Marker.LINE_STRIP
        ellipse_line_msg.id = 99
        ellipse_line_msg.header.stamp = state.getTime()
        ellipse_line_msg.header.frame_id = self.path_frame
        ellipse_line_msg.action = Marker.ADD
        ellipse_line_msg.scale.x = 0.02
        ellipse_line_msg.scale.y = 0.02
        ellipse_line_msg.scale.z = 0.02
        ellipse_line_msg.color.a = 0.6
        ellipse_line_msg.color.r = 239/255.0
        ellipse_line_msg.color.g = 41/255.0
        ellipse_line_msg.color.b = 41/255.0

        cov = self._planar_covariance(state)
        L = self._covariance_sqrt(cov)
        if L is None:
            return None
        for j in range(np.shape(circle)[0]):
            ell_se2_vec = scale * L @ circle[j,:].reshape(-1,1)
            temp = expm(G1 * ell_se2_vec[0] + G2 * ell_se2_vec[1] + G3 * ell_se2_vec[2]) @ mean_matrix
            ellipse_point = Point()
            ellipse_point.x = float(temp[0,2])
            ellipse_point.y = float(temp[1,2])
            ellipse_point.z = 0.0
            ellipse_line_msg.points.append(ellipse_point)

        return ellipse_line_msg


    def publish_state_path(self, state):

        position = state.getPosition()
        orientation = state.getOrientation()

        pose = PoseStamped()
        pose.header.stamp = state.getTime()
        pose.header.frame_id = self.path_frame
        pose.pose.position.x = float(position[0])
        pose.pose.position.y = float(position[1])

        if self.world_dim == 2:
            pose.pose.position.z = 0.0
            rot = Rotation.from_euler('z', float(orientation), degrees=False)
            quat = rot.as_quat()    # (x, y, z, w)
            pose.pose.orientation.x = float(quat[0])
            pose.pose.orientation.y = float(quat[1])
            pose.pose.orientation.z = float(quat[2])
            pose.pose.orientation.w = float(quat[3])

        self.path.poses.append(pose)

        self.path_pub.publish(self.path)

    def _make_heading_marker(self, data, stamp):

        start = Point()
        start.x = float(data[0])
        start.y = float(data[1])
        start.z = 0.0

        end = Point()
        end.x = float(data[0] + self.gt_heading_length * np.cos(float(data[2])))
        end.y = float(data[1] + self.gt_heading_length * np.sin(float(data[2])))
        end.z = 0.0

        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.path_frame
        marker.ns = "groundtruth_heading"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x = self.gt_heading_shaft_diameter
        marker.scale.y = self.gt_heading_head_diameter
        marker.scale.z = self.gt_heading_head_length
        marker.color.a = 0.95
        marker.color.r = 0.0
        marker.color.g = 84 / 255.0
        marker.color.b = 1.0
        marker.points = [start, end]
        return marker

    def publish_gt_path(self, data, stamp=None):

        if stamp is None:
            stamp = self.node_.get_clock().now().to_msg()

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.path_frame
        pose.pose.position.x = float(data[0])
        pose.pose.position.y = float(data[1])
        pose.pose.position.z = 0.0
        rot = Rotation.from_euler('z', float(data[2]), degrees=False)
        quat = rot.as_quat()
        pose.pose.orientation.x = float(quat[0])
        pose.pose.orientation.y = float(quat[1])
        pose.pose.orientation.z = float(quat[2])
        pose.pose.orientation.w = float(quat[3])

        self.gt_path.poses.append(pose)
        self.gt_path_pub.publish(self.gt_path)
        self.gt_heading_pub.publish(self._make_heading_marker(data, stamp))

    def publish_command_path(self, data):

        pose = PoseStamped()
        pose.header.stamp = self.node_.get_clock().now().to_msg()
        pose.header.frame_id = self.path_frame
        pose.pose.position.x = float(data[0])
        pose.pose.position.y = float(data[1])
        pose.pose.position.z = 0.0
        rot = Rotation.from_euler('z', float(data[2]), degrees=False)
        quat = rot.as_quat()
        pose.pose.orientation.x = float(quat[0])
        pose.pose.orientation.y = float(quat[1])
        pose.pose.orientation.z = float(quat[2])
        pose.pose.orientation.w = float(quat[3])

        self.cmd_path.poses.append(pose)
        self.cmd_path_pub.publish(self.cmd_path)

def main():

    state = RobotState()
    path_pub = path_publisher()

    path_pub.make_ellipse()

    pass

if __name__ == '__main__':
    main()
