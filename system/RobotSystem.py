import sys
sys.path.append('.')
import os
import yaml
import numpy as np

import rclpy
from rclpy.node import Node
import tf2_ros

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import rclpy.duration
from apriltag_msgs.msg import AprilTagDetectionArray
from sensor_msgs.msg import PointCloud

from comm.path_publisher import path_publisher
from comm.marker_publisher import marker_publisher
from system.evaluation_pipeline import EvaluationPipeline
from system.measurement_pipeline import MeasurementPipeline
from system.runtime_monitor import RuntimeMonitor
from utils.filter_initialization import filter_initialization
from utils.system_initialization import system_initialization
from utils.utils import (
    mahalanobis,
    wrap2Pi,
)


class RobotSystem(Node):

    def __init__(self, world=None):
        super().__init__('robot_state_estimator')
        self.runtime = RuntimeMonitor(self)

        # Suppress TF warnings
        import rclpy.logging
        rclpy.logging.set_logger_level('tf2_ros', rclpy.logging.LoggingSeverity.ERROR)
        rclpy.logging.set_logger_level('tf2_ros_py', rclpy.logging.LoggingSeverity.ERROR)

        # load params
        with open("config/settings.yaml", 'r') as stream:
            param = yaml.safe_load(stream)

        self.filter_name = param['filter_name']
        self._bag_name = os.getenv("ROB530_BAG_NAME", "").strip()
        self.auto_shutdown_on_idle = bool(
            param.get('auto_shutdown_on_idle', bool(self._bag_name))
        )
        self.idle_shutdown_wall_s = float(param.get('idle_shutdown_wall_s', 5.0))
        base_alphas_sqrt = np.array(param['alphas_sqrt'], dtype=float)
        pf_alphas_sqrt = np.array(
            param.get('pf_alphas_sqrt', param['alphas_sqrt']),
            dtype=float,
        )
        ukf_alphas_sqrt = np.array(
            param.get('ukf_alphas_sqrt', param['alphas_sqrt']),
            dtype=float,
        )
        if self.filter_name == 'UKF':
            self.motion_alphas_sqrt = ukf_alphas_sqrt
            self.motion_alphas_key = 'ukf_alphas_sqrt'
        elif self.filter_name == 'PF':
            self.motion_alphas_sqrt = pf_alphas_sqrt
            self.motion_alphas_key = 'pf_alphas_sqrt'
        else:
            self.motion_alphas_sqrt = base_alphas_sqrt
            self.motion_alphas_key = 'alphas_sqrt'
        alphas = self.motion_alphas_sqrt**2

        base_bearing_std_deg = float(param.get('bearing_std_deg', param['beta']))
        base_range_std_m = float(param.get('range_std_m', 0.15))
        ukf_bearing_std_deg = float(
            param.get('ukf_bearing_std_deg', base_bearing_std_deg)
        )
        ukf_range_std_m = float(
            param.get('ukf_range_std_m', base_range_std_m)
        )
        if self.filter_name == 'UKF':
            self.bearing_std_deg = ukf_bearing_std_deg
            self.range_std_m = ukf_range_std_m
            self.measurement_noise_key = 'ukf_measurement_noise'
        else:
            self.bearing_std_deg = base_bearing_std_deg
            self.range_std_m = base_range_std_m
            self.measurement_noise_key = 'measurement_noise'
        beta = np.deg2rad(self.bearing_std_deg)
        self.tag_body_position_std_m = float(param.get('tag_body_position_std_m', 0.25))
        self.bearing_innovation_max_deg = float(param.get('bearing_innovation_max_deg', 60.0))
        self.range_innovation_max_m = float(param.get('range_innovation_max_m', 1.5))
        self.min_decision_margin = float(param.get('min_decision_margin', 100.0))
        self.max_hamming = int(param.get('max_hamming', 0))
        self.quality_margin_ref = float(param.get('quality_margin_ref', 130.0))
        self.quality_std_scale_min = float(param.get('quality_std_scale_min', 0.75))
        self.quality_std_scale_max = float(param.get('quality_std_scale_max', 1.50))
        self.no_measurement_warning_s = float(param.get('no_measurement_warning_s', 2.0))
        self.no_correction_warning_s = float(param.get('no_correction_warning_s', 2.0))
        self.odom_gap_warning_s = float(param.get('odom_gap_warning_s', 0.2))
        self.odom_max_substep_s = float(param.get('odom_max_substep_s', 0.25))
        self.eval_gt_sync_tolerance_s = float(
            param.get('evaluation_gt_sync_tolerance_s', 0.05)
        )
        self.eval_gt_buffer_size = int(
            param.get('evaluation_gt_buffer_size', 512)
        )
        self.verbose_runtime_logging = bool(param.get('verbose_runtime_logging', False))
        self.log_measurement_gaps = bool(param.get('log_measurement_gaps', False))
        self.log_pf_resampling = bool(param.get('log_pf_resampling', True))

        init_state_mean = np.array(param['initial_state_mean'])
        init_state_cov = np.diag(param['initial_state_variance'])**2
        self.filter_params = {
            'pf_num_particles': int(param.get('pf_num_particles', 500)),
            'pf_initial_state_std': np.array(
                param.get('pf_initial_state_std', param['initial_state_variance']),
                dtype=float,
            ),
            'pf_resample_threshold_ratio': float(
                param.get('pf_resample_threshold_ratio', 0.2)
            ),
            'ukf_kappa_g': float(param.get('ukf_kappa_g', 2.0)),
            'ukf_sigma_point_jitter': float(
                param.get('ukf_sigma_point_jitter', 1.0e-9)
            ),
        }
        self.inekf_process_noise_std = np.array(
            param.get('inekf_process_noise_std', [0.05, 0.10, 0.10]),
            dtype=float,
        )

        self.system_ = system_initialization(
            alphas,
            beta,
            tag_body_position_std_m=self.tag_body_position_std_m,
            range_std_m=self.range_std_m,
            inekf_process_noise_std=self.inekf_process_noise_std,
        )
        self.Lie2Cart = param['Lie2Cart']
        self.odom_topic = param.get('odom_topic', '/odom')
        self.measurement_topic = param.get('measurement_topic', '')
        self.detection_topic = param.get('detection_topic', '/detections_reprocessed')
        self.mocap_frame_rotation = np.deg2rad(param.get('mocap_frame_rotation_deg', 0.0))
        self.mocap_translation = np.array([
            float(param.get('mocap_translation_x_m', 0.0)),
            float(param.get('mocap_translation_y_m', 0.0)),
        ], dtype=float)
        self.mocap_yaw_offset = np.deg2rad(param.get('mocap_yaw_offset_deg', 0.0))
        self.mocap_lever_arm = np.array([
            float(param.get('mocap_lever_arm_x_m', 0.0)),
            float(param.get('mocap_lever_arm_y_m', 0.0)),
        ], dtype=float)
        self.world_innovation_max_m = float(
            param.get(
                'world_innovation_max_m',
                param.get('inekf_world_innovation_max_m', 0.75),
            )
        )

        if world is not None:
            self.world = world
            self.landmarks = self.world.getLandmarksInWorld()
        else:
            self.get_logger().error("Please provide a world with landmarks!")

        if self.filter_name is not None:
            self.get_logger().info(f"Initializing {self.filter_name}")
            self.filter_ = filter_initialization(
                self.system_,
                init_state_mean,
                init_state_cov,
                self.filter_name,
                filter_params=self.filter_params,
            )
            self.state_ = self.filter_.getState()
            self.get_logger().info(
                "Tag correction settings: "
                f"body_std={self.tag_body_position_std_m:.2f} m, "
                f"bearing_gate={self.bearing_innovation_max_deg:.1f} deg, "
                f"range_gate={self.range_innovation_max_m:.2f} m"
            )
            if hasattr(self.filter_, 'Q'):
                self.get_logger().info(
                    "Legacy measurement noise settings: "
                    f"source={self.measurement_noise_key}, "
                    f"bearing_std={self.bearing_std_deg:.1f} deg, "
                    f"range_std={self.range_std_m:.2f} m"
                )
            self.get_logger().info(
                "Tag quality settings: "
                f"min_margin={self.min_decision_margin:.1f}, "
                f"max_hamming={self.max_hamming}, "
                f"margin_ref={self.quality_margin_ref:.1f}, "
                f"std_scale=[{self.quality_std_scale_min:.2f}, {self.quality_std_scale_max:.2f}]"
            )
            self.get_logger().info(
                "Motion alpha settings: "
                f"source={self.motion_alphas_key} values={self.motion_alphas_sqrt.tolist()}"
            )
            self.get_logger().info(
                "World-innovation gate: "
                f"{self.world_innovation_max_m:.2f} m"
            )
            if self.filter_name == 'InEKF':
                self.get_logger().info(
                    "InEKF process std: "
                    f"{self.inekf_process_noise_std.tolist()}"
                )
            elif self.filter_name == 'UKF':
                self.get_logger().info(
                    "UKF sigma-point settings: "
                    f"kappa={self.filter_params['ukf_kappa_g']:.2f}, "
                    f"jitter={self.filter_params['ukf_sigma_point_jitter']:.1e}"
                )
            elif self.filter_name == 'PF':
                self.get_logger().info(
                    "PF settings: "
                    f"particles={self.filter_params['pf_num_particles']}, "
                    f"resample_ratio={self.filter_params['pf_resample_threshold_ratio']:.2f}, "
                    f"init_std={self.filter_params['pf_initial_state_std'].tolist()}"
                )
        else:
            self.get_logger().error("Please specify a filter name!")

        self._last_odom_stamp = None
        self._last_correction_stamp = {}  # tag_id -> detection stamp (sec) for de-duping
        self._last_detection_key = None
        self._last_measurement_stamp_sec = None
        self._last_accepted_correction_sec = None
        self._base_filter_V = (
            np.array(getattr(self.filter_, 'V', self.system_.V), copy=True)
            if hasattr(self.filter_, 'V') or hasattr(self.system_, 'V')
            else None
        )
        self._base_filter_Q = (
            np.array(getattr(self.filter_, 'Q', self.system_.Q), copy=True)
            if hasattr(self.filter_, 'Q') or hasattr(self.system_, 'Q')
            else None
        )

        # filter state held between callbacks
        state = np.asarray(self.state_.getState(), dtype=float).reshape(3)
        covariance = np.asarray(self.state_.getCovariance(), dtype=float).reshape(3, 3)
        self.X = np.copy(state)
        self.P = np.copy(covariance)
        self.particles = None
        self.particle_weight = None
        self.mu = None
        self.step = 0
        self.initialized = False
        self._ukf_sigma_points = (None, None)  # UKF: sigma points from last prediction
        if self.filter_name == 'PF':
            self.particles = np.array(self.filter_.particles, copy=True)
            self.particle_weight = np.array(self.filter_.particle_weight, copy=True)
        elif self.filter_name == 'InEKF':
            self.mu = np.array(self.filter_.mu, copy=True)

        self.evaluation = EvaluationPipeline(self)
        self.measurement = MeasurementPipeline(self)

        # publishers
        self.pub = path_publisher(self)
        self.landmark_visualizer = marker_publisher(self.world, self)

        # subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self._prediction_callback,
            10
        )
        self.get_logger().info(f"Using odometry from {self.odom_topic}")
        if self.measurement_topic:
            self.tag_measurement_sub = self.create_subscription(
                PointCloud,
                self.measurement_topic,
                self._measurement_callback,
                10
            )
            self.get_logger().info(f"Using tag measurements from {self.measurement_topic}")
            self.tf_buffer = None
            self.tf_listener = None
        else:
            # TF2 for transforming tag detections from camera frame to base_link
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
            self.tag_sub = self.create_subscription(
                AprilTagDetectionArray,
                self.detection_topic,
                self._correction_callback,
                10
            )
            self.get_logger().info(f"Using tag detections from {self.detection_topic}")

        mocap_topic = param.get('mocap_topic', '')
        if mocap_topic:
            self.mocap_sub = self.create_subscription(
                PoseStamped,
                mocap_topic,
                self._mocap_callback,
                10
            )
            self.get_logger().info(f"Evaluation enabled: subscribing to {mocap_topic}")
        else:
            self.get_logger().info("Evaluation disabled: no mocap_topic set in settings.yaml")
        self._idle_shutdown_timer = None
        if self.auto_shutdown_on_idle:
            self._idle_shutdown_timer = self.create_timer(0.5, self._idle_shutdown_check)
            self.get_logger().info(
                f"Auto shutdown enabled for bag run after {self.idle_shutdown_wall_s:.1f}s of input inactivity"
            )

    def _mark_runtime_activity(self):
        self.runtime.mark_runtime_activity()

    def _idle_shutdown_check(self):
        self.runtime.idle_shutdown_check()

    def _mocap_callback(self, msg):
        self.evaluation.mocap_callback(msg)

    def _ground_truth_for_stamp(self, stamp_sec):
        return self.evaluation.ground_truth_for_stamp(stamp_sec)

    # Prediction callback occurs whenever a new odometry msg is received
    def _prediction_callback(self, msg):
        self._mark_runtime_activity()
        stamp = msg.header.stamp
        stamp_sec = stamp.sec + stamp.nanosec * 1e-9
        if self._last_odom_stamp is None:
            self._last_odom_stamp = stamp
            return
        dt = (stamp.sec - self._last_odom_stamp.sec) + \
             (stamp.nanosec - self._last_odom_stamp.nanosec) * 1e-9
        self._last_odom_stamp = stamp
        if dt <= 0:
            return
        if dt > self.odom_gap_warning_s:
            self.get_logger().warn(
                f"Large odom gap dt={dt:.3f}s at stamp={stamp.sec}.{stamp.nanosec:09d}; "
                f"integrating in <= {self.odom_max_substep_s:.3f}s substeps"
            )

        remaining = dt
        while remaining > 1e-9:
            step_dt = min(remaining, self.odom_max_substep_s)
            self._apply_prediction_step(msg, step_dt)
            remaining -= step_dt

        self.initialized = True
        self.state_ = self.filter_.getState()

        self.pub.publish_pose(self.state_)
        self.pub.publish_state_path(self.state_)

        gt_eval, gt_sync_dt = self._ground_truth_for_stamp(stamp_sec)
        if gt_eval is not None:
            result = mahalanobis(self.state_, gt_eval.copy(), self.filter_name, self.Lie2Cart)
            self._results.append(result)
            self._gt_history.append(gt_eval.copy())
            est_pose = np.array([
                gt_eval[0] - result[1],
                gt_eval[1] - result[2],
                wrap2Pi(gt_eval[2] - result[3]),
            ])
            self._est_history.append(est_pose)
            self._eval_gt_sync_dt.append(float(gt_sync_dt))
        elif self._latest_gt is not None:
            self._eval_skipped_unsynced_gt += 1

    def _apply_prediction_step(self, msg, dt):
        u = np.array([
            msg.twist.twist.linear.x * dt,
            msg.twist.twist.angular.z * dt,
            0.0
        ])

        if self.filter_name == 'EKF':
            self.X, self.P = self.filter_.prediction(u, self.X, self.P, self.step)

        elif self.filter_name == 'UKF':
            Y_sigma, w, self.X, self.P = self.filter_.prediction(u, self.X, self.P, self.step)
            self._ukf_sigma_points = (Y_sigma, w)

        elif self.filter_name == 'PF':
            self.particles = self.filter_.prediction(u, self.particles, self.step)

        elif self.filter_name == 'InEKF':
            self.mu, self.P = self.filter_.prediction(u, self.P, self.mu, self.step)

        self.step += 1

    def _current_pose_estimate(self):
        return self.measurement.current_pose_estimate()

    def _format_pose(self, pose):
        return self.measurement.format_pose(pose)

    def _channel_map(self, msg):
        return self.measurement.channel_map(msg)

    def _measurement_quality(self, channel_map, idx):
        return self.measurement.measurement_quality(channel_map, idx)

    def _detection_quality(self, detection):
        return self.measurement.detection_quality(detection)

    def _quality_log_suffix(self, quality):
        return self.measurement.quality_log_suffix(quality)

    def _quality_is_consistent(self, det_id, quality):
        return self.measurement.quality_is_consistent(det_id, quality)

    def _measurement_std_scale(self, quality):
        return self.measurement.measurement_std_scale(quality)

    def _set_filter_measurement_covariance(self, std_scale):
        return self.measurement.set_filter_measurement_covariance(std_scale)

    def _restore_filter_measurement_covariance(self, original):
        self.measurement.restore_filter_measurement_covariance(original)

    def _maybe_log_measurement_gap(self, stamp_sec, ids):
        self.measurement.maybe_log_measurement_gap(stamp_sec, ids)

    def _measurement_is_consistent(self, det_id, observation, Y):
        return self.measurement.measurement_is_consistent(det_id, observation, Y)

    def _apply_tag_observation(self, det_id, stamp, x_b, y_b, z_b=0.0, quality=None):
        return self.measurement.apply_tag_observation(det_id, stamp, x_b, y_b, z_b, quality)

    def _measurement_callback(self, msg):
        self.measurement.measurement_callback(msg)

    # Correction callback occurs whenever a new tag detection msg is received.
    # Each detected tag is applied as a sequential update so any number of
    # detections (>= 1) is supported.
    def _correction_callback(self, msg):
        self.measurement.correction_callback(msg)

    def _estimate_planar_alignment(self, source_xy, target_xy):
        return self.evaluation.estimate_planar_alignment(source_xy, target_xy)

    def _suggest_mocap_calibration(self, gt_array, est_array):
        return self.evaluation.suggest_mocap_calibration(gt_array, est_array)

    def _log_result_message(self, level, message):
        self.runtime.log_result_message(level, message)

    def _runtime_metrics(self):
        return self.runtime.runtime_metrics()

    def plot_results(self):
        self.evaluation.plot_results()

    def _sanitize_artifact_name(self, value):
        return self.evaluation.sanitize_artifact_name(value)
