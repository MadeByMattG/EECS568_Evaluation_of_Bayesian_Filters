import numpy as np
import rclpy

from utils.utils import wrap2Pi


class MeasurementPipeline:
    def __init__(self, host):
        self.host = host

    def current_pose_estimate(self):
        state = self.host.filter_.getState().getState()
        if state is None:
            return None
        return np.array(state, dtype=float).reshape(3)

    def format_pose(self, pose):
        if pose is None:
            return "(uninitialized)"
        return (
            f"(x={pose[0]:.3f}, y={pose[1]:.3f}, "
            f"theta={np.rad2deg(pose[2]):.1f}deg)"
        )

    def channel_map(self, msg):
        return {channel.name: list(channel.values) for channel in getattr(msg, 'channels', [])}

    def measurement_quality(self, channel_map, idx):
        def _value(name, default=None):
            values = channel_map.get(name, [])
            if idx >= len(values):
                return default
            return values[idx]

        decision_margin = _value('decision_margin')
        hamming = _value('hamming')
        goodness = _value('goodness')
        return {
            'decision_margin': None if decision_margin is None else float(decision_margin),
            'hamming': None if hamming is None else int(round(hamming)),
            'goodness': None if goodness is None else float(goodness),
        }

    def detection_quality(self, detection):
        return {
            'decision_margin': float(detection.decision_margin),
            'hamming': int(detection.hamming),
            'goodness': float(detection.goodness),
        }

    def quality_log_suffix(self, quality):
        if quality is None:
            return ""

        parts = []
        decision_margin = quality.get('decision_margin')
        hamming = quality.get('hamming')
        goodness = quality.get('goodness')
        if decision_margin is not None:
            parts.append(f"margin={decision_margin:.1f}")
        if hamming is not None:
            parts.append(f"hamming={hamming}")
        if goodness is not None and goodness > 0.0:
            parts.append(f"goodness={goodness:.3f}")
        if not parts:
            return ""
        return " " + " ".join(parts)

    def quality_is_consistent(self, det_id, quality):
        if quality is None:
            return True

        decision_margin = quality.get('decision_margin')
        hamming = quality.get('hamming')

        if hamming is not None and hamming > self.host.max_hamming:
            self.host.get_logger().warn(
                f"Rejecting tag {det_id}: hamming={hamming} exceeds {self.host.max_hamming}"
            )
            return False

        if decision_margin is not None and decision_margin < self.host.min_decision_margin:
            self.host.get_logger().warn(
                f"Rejecting tag {det_id}: decision margin {decision_margin:.1f} "
                f"below {self.host.min_decision_margin:.1f}"
            )
            return False

        return True

    def measurement_std_scale(self, quality):
        if quality is None:
            return 1.0

        decision_margin = quality.get('decision_margin')
        if decision_margin is None or decision_margin <= 1e-6:
            return 1.0

        scale = self.host.quality_margin_ref / decision_margin
        return float(np.clip(scale, self.host.quality_std_scale_min, self.host.quality_std_scale_max))

    def set_filter_measurement_covariance(self, std_scale):
        updated = {}

        if hasattr(self.host.filter_, 'V') and self.host._base_filter_V is not None:
            updated['V'] = np.array(self.host.filter_.V, copy=True)
            self.host.filter_.V = np.array(self.host._base_filter_V, copy=True) * (std_scale ** 2)

        if hasattr(self.host.filter_, 'Q') and self.host._base_filter_Q is not None:
            updated['Q'] = np.array(self.host.filter_.Q, copy=True)
            self.host.filter_.Q = np.array(self.host._base_filter_Q, copy=True) * (std_scale ** 2)

        return updated if updated else None

    def restore_filter_measurement_covariance(self, original):
        if original is None:
            return
        if 'V' in original and hasattr(self.host.filter_, 'V'):
            self.host.filter_.V = original['V']
        if 'Q' in original and hasattr(self.host.filter_, 'Q'):
            self.host.filter_.Q = original['Q']

    def maybe_log_measurement_gap(self, stamp_sec, ids):
        if not self.host.log_measurement_gaps:
            self.host._last_measurement_stamp_sec = stamp_sec
            return

        if self.host._last_measurement_stamp_sec is not None:
            gap = stamp_sec - self.host._last_measurement_stamp_sec
            if gap >= self.host.no_measurement_warning_s:
                self.host.get_logger().warn(
                    f"No tag measurements for {gap:.2f}s before "
                    f"stamp={int(stamp_sec)}.{int(round((stamp_sec % 1.0) * 1e9)):09d} "
                    f"ids={ids}"
                )

        if self.host._last_accepted_correction_sec is not None:
            gap = stamp_sec - self.host._last_accepted_correction_sec
            if gap >= self.host.no_correction_warning_s:
                self.host.get_logger().warn(
                    f"No accepted tag corrections for {gap:.2f}s before "
                    f"stamp={int(stamp_sec)}.{int(round((stamp_sec % 1.0) * 1e9)):09d} "
                    f"ids={ids}"
                )

        self.host._last_measurement_stamp_sec = stamp_sec

    def measurement_is_consistent(self, det_id, observation, Y):
        pose = self.current_pose_estimate()
        if pose is None:
            return True

        landmark = self.host.landmarks.getLandmark(int(observation[2]))
        z_hat = self.host.system_.hfun(
            landmark.getPosition()[0],
            landmark.getPosition()[1],
            pose
        )

        bearing_error = wrap2Pi(observation[0] - z_hat[0])
        range_error = observation[1] - z_hat[1]

        if abs(bearing_error) > np.deg2rad(self.host.bearing_innovation_max_deg):
            self.host.get_logger().warn(
                f"Rejecting tag {det_id}: "
                f"bearing innovation {np.rad2deg(bearing_error):.1f} deg"
            )
            return False

        if abs(range_error) > self.host.range_innovation_max_m:
            self.host.get_logger().warn(
                f"Rejecting tag {det_id}: "
                f"range innovation {range_error:.2f} m"
            )
            return False

        if self.host.world_innovation_max_m > 0.0:
            theta = pose[2]
            rot = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)],
            ])
            predicted_world = pose[:2] + rot @ np.asarray(Y[:2], dtype=float)
            landmark_world = np.array([
                landmark.getPosition()[0],
                landmark.getPosition()[1],
            ])
            innovation = np.linalg.norm(predicted_world - landmark_world)
            if self.host.verbose_runtime_logging:
                self.host.get_logger().info(
                    f"Tag {det_id}: body=({Y[0]:.3f},{Y[1]:.3f}) "
                    f"predicted_world=({predicted_world[0]:.3f},{predicted_world[1]:.3f}) "
                    f"lm=({landmark_world[0]:.3f},{landmark_world[1]:.3f}) "
                    f"innovation={innovation:.3f}m "
                    f"robot=({pose[0]:.3f},{pose[1]:.3f})"
                )
            if innovation > self.host.world_innovation_max_m:
                self.host.get_logger().warn(
                    f"Rejecting tag {det_id}: world innovation {innovation:.2f} m"
                )
                return False

        return True

    def apply_tag_observation(self, det_id, stamp, x_b, y_b, z_b=0.0, quality=None):
        det_stamp_sec = stamp.sec + stamp.nanosec * 1e-9
        pre_pose = self.current_pose_estimate()

        if x_b <= 0:
            if self.host.verbose_runtime_logging:
                self.host.get_logger().info(
                    f"SKIP tag {det_id} stamp={stamp.sec}.{stamp.nanosec:09d} "
                    f"reason=behind_robot x_b={x_b:.3f} y_b={y_b:.3f} z_b={z_b:.3f}"
                    f"{self.quality_log_suffix(quality)}"
                )
            return None

        if self.host._last_correction_stamp.get(det_id) == det_stamp_sec:
            if self.host.verbose_runtime_logging:
                self.host.get_logger().info(
                    f"SKIP tag {det_id} stamp={stamp.sec}.{stamp.nanosec:09d} "
                    f"reason=duplicate_tag_stamp{self.quality_log_suffix(quality)}"
                )
            return None

        if not self.quality_is_consistent(det_id, quality):
            return None

        observation = np.array([
            np.arctan2(y_b, x_b),
            np.sqrt(x_b**2 + y_b**2),
            float(det_id + 1)
        ])
        Y = np.array([x_b, y_b, 1.0])

        if self.host.verbose_runtime_logging:
            self.host.get_logger().info(
                f"CANDIDATE tag {det_id} stamp={stamp.sec}.{stamp.nanosec:09d} "
                f"body=({x_b:.3f},{y_b:.3f},{z_b:.3f}) bearing={np.rad2deg(observation[0]):.2f}deg "
                f"range={observation[1]:.3f} pre={self.format_pose(pre_pose)}"
                f"{self.quality_log_suffix(quality)}"
            )

        if not self.measurement_is_consistent(det_id, observation, Y):
            return None

        std_scale = self.measurement_std_scale(quality)
        original_filter_V = self.set_filter_measurement_covariance(std_scale)

        try:
            if self.host.filter_name == 'EKF':
                self.host.X, self.host.P = self.host.filter_.correction(
                    observation, self.host.landmarks, self.host.X, self.host.P)

            elif self.host.filter_name == 'UKF':
                Y_sigma, w = self.host._ukf_sigma_points
                self.host.X, self.host.P = self.host.filter_.correction(
                    observation, self.host.landmarks, Y_sigma, w, self.host.X, self.host.P)

            elif self.host.filter_name == 'PF':
                self.host.X, self.host.P, self.host.particles, self.host.particle_weight = (
                    self.host.filter_.correction(
                        observation,
                        self.host.landmarks,
                        self.host.particles,
                        self.host.particle_weight,
                        self.host.step,
                    )
                )
                if self.host.log_pf_resampling and getattr(self.host.filter_, 'resampled_last_step', False):
                    threshold = self.host.filter_.resample_threshold_ratio * self.host.filter_.n
                    self.host.get_logger().info(
                        f"PF resampled after tag {det_id} stamp={stamp.sec}.{stamp.nanosec:09d} "
                        f"Neff={self.host.filter_.last_neff:.1f} threshold={threshold:.1f}"
                    )

            elif self.host.filter_name == 'InEKF':
                self.host.X, self.host.P, self.host.mu = self.host.filter_.correction(
                    Y, observation, self.host.landmarks, self.host.mu, self.host.P)
        finally:
            self.restore_filter_measurement_covariance(original_filter_V)

        post_pose = self.current_pose_estimate()
        if self.host.verbose_runtime_logging:
            self.host.get_logger().info(
                f"APPLY tag {det_id} stamp={stamp.sec}.{stamp.nanosec:09d} "
                f"pre={self.format_pose(pre_pose)} post={self.format_pose(post_pose)} "
                f"std_scale={std_scale:.2f}{self.quality_log_suffix(quality)}"
            )
        self.host._last_correction_stamp[det_id] = det_stamp_sec
        self.host._last_accepted_correction_sec = det_stamp_sec
        return det_id + 1

    def measurement_callback(self, msg):
        self.host.runtime.mark_runtime_activity()
        if not self.host.initialized:
            return

        channel_map = self.channel_map(msg)
        ids = [int(round(value)) for value in channel_map.get('id', [])]
        if len(ids) < 1 or len(msg.points) < 1:
            return
        det_stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        measurement_key = (
            msg.header.stamp.sec,
            msg.header.stamp.nanosec,
            tuple(ids),
        )
        if measurement_key == self.host._last_detection_key:
            if self.host.verbose_runtime_logging:
                self.host.get_logger().info(
                    f"SKIP measurement msg stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d} "
                    f"ids={ids} reason=duplicate_msg"
                )
            return
        self.host._last_detection_key = measurement_key
        self.maybe_log_measurement_gap(det_stamp_sec, ids)

        if self.host.verbose_runtime_logging:
            self.host.get_logger().info(
                f"MEASURED stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d} ids={ids}"
            )

        measurements = []
        for idx, point in enumerate(msg.points):
            if idx >= len(ids):
                break
            quality = self.measurement_quality(channel_map, idx)
            measurements.append((ids[idx], point, quality))

        measurements.sort(
            key=lambda item: (
                -(item[2].get('decision_margin') if item[2].get('decision_margin') is not None else 0.0),
                item[2].get('hamming') if item[2].get('hamming') is not None else 0,
            )
        )

        observed_ids = []
        for det_id, point, quality in measurements:
            observed_id = self.apply_tag_observation(
                det_id,
                msg.header.stamp,
                point.x,
                point.y,
                point.z,
                quality=quality,
            )
            if observed_id is not None:
                observed_ids.append(observed_id)

        if not observed_ids:
            return

        self.host.state_ = self.host.filter_.getState()
        self.host.pub.publish_pose(self.host.state_)
        self.host.pub.publish_state_path(self.host.state_)
        self.host.landmark_visualizer.publish_landmarks(observed_ids)

    def correction_callback(self, msg):
        self.host.runtime.mark_runtime_activity()
        if not self.host.initialized:
            return

        if len(msg.detections) < 1:
            return
        det_stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        observed_ids = []
        detection_key = (
            msg.header.stamp.sec,
            msg.header.stamp.nanosec,
            tuple(det.id for det in msg.detections)
        )
        if detection_key == self.host._last_detection_key:
            if self.host.verbose_runtime_logging:
                self.host.get_logger().info(
                    f"SKIP detection msg stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d} "
                    f"ids={[det.id for det in msg.detections]} reason=duplicate_msg"
                )
            return
        self.host._last_detection_key = detection_key
        self.maybe_log_measurement_gap(det_stamp_sec, [det.id for det in msg.detections])

        if self.host.verbose_runtime_logging:
            self.host.get_logger().info(
                f"DETECTED stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d} "
                f"ids={[det.id for det in msg.detections]}"
            )

        det_time = rclpy.time.Time.from_msg(msg.header.stamp)
        for det in msg.detections:
            try:
                tag_frame = f'tag_{det.id}'
                tf_stamped = self.host.tf_buffer.lookup_transform(
                    'base_link', tag_frame, det_time,
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
            except Exception as exc:
                if self.host.verbose_runtime_logging:
                    self.host.get_logger().info(
                        f"SKIP tag {det.id} stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d} "
                        f"reason=tf_lookup_failed detail={exc}"
                    )
                continue

            observed_id = self.apply_tag_observation(
                det.id,
                msg.header.stamp,
                tf_stamped.transform.translation.x,
                tf_stamped.transform.translation.y,
                tf_stamped.transform.translation.z,
                quality=self.detection_quality(det),
            )
            if observed_id is not None:
                observed_ids.append(observed_id)

        if not observed_ids:
            return

        self.host.state_ = self.host.filter_.getState()
        self.host.pub.publish_pose(self.host.state_)
        self.host.pub.publish_state_path(self.host.state_)
        self.host.landmark_visualizer.publish_landmarks(observed_ids)
