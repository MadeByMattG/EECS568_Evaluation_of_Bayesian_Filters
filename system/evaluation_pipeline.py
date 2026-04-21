from collections import deque
import json
import os
import pathlib
import re

import numpy as np
from scipy.spatial.transform import Rotation

from utils.utils import (
    mahalanobis,
    plot_error_with_options,
    summarize_results,
    wrap2Pi,
)


class EvaluationPipeline:
    def __init__(self, host):
        self.host = host
        self.host._results = []
        self.host._gt_history = []
        self.host._est_history = []
        self.host._latest_gt = None
        self.host._mocap_history = deque(maxlen=max(self.host.eval_gt_buffer_size, 2))
        self.host._eval_gt_sync_dt = []
        self.host._eval_skipped_unsynced_gt = 0

    def mocap_callback(self, msg):
        self.host.runtime.mark_runtime_activity()
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        xy = np.array([msg.pose.position.x, msg.pose.position.y], dtype=float)
        if abs(self.host.mocap_frame_rotation) > 1e-12:
            c = np.cos(self.host.mocap_frame_rotation)
            s = np.sin(self.host.mocap_frame_rotation)
            xy = np.array([
                c * xy[0] - s * xy[1],
                s * xy[0] + c * xy[1],
            ])
        xy = xy + self.host.mocap_translation

        quat = [
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ]
        theta = Rotation.from_quat(quat).as_euler('xyz')[2]
        theta = wrap2Pi(theta + self.host.mocap_frame_rotation)
        theta = wrap2Pi(theta + self.host.mocap_yaw_offset)
        if np.linalg.norm(self.host.mocap_lever_arm) > 1e-12:
            c = np.cos(theta)
            s = np.sin(theta)
            lever_world = np.array([
                c * self.host.mocap_lever_arm[0] - s * self.host.mocap_lever_arm[1],
                s * self.host.mocap_lever_arm[0] + c * self.host.mocap_lever_arm[1],
            ])
            xy = xy - lever_world
        self.host._latest_gt = np.array([xy[0], xy[1], theta])
        if self.host._mocap_history and stamp_sec < self.host._mocap_history[-1][0] - 1e-9:
            self.host._mocap_history.clear()
        self.host._mocap_history.append((stamp_sec, self.host._latest_gt.copy()))
        self.host.get_logger().debug(
            f"GT: x={xy[0]:.3f} y={xy[1]:.3f} theta={np.rad2deg(theta):.1f}deg"
        )
        self.host.pub.publish_gt_path(self.host._latest_gt, msg.header.stamp)

    def ground_truth_for_stamp(self, stamp_sec):
        if not self.host._mocap_history:
            return None, None

        history = list(self.host._mocap_history)
        tolerance = self.host.eval_gt_sync_tolerance_s

        if len(history) == 1:
            sample_t, sample_gt = history[0]
            dt = abs(stamp_sec - sample_t)
            if dt <= tolerance:
                return sample_gt.copy(), dt
            return None, None

        first_t, first_gt = history[0]
        if stamp_sec <= first_t:
            dt = abs(stamp_sec - first_t)
            if dt <= tolerance:
                return first_gt.copy(), dt
            return None, None

        prev_t, prev_gt = history[0]
        for next_t, next_gt in history[1:]:
            if stamp_sec > next_t:
                prev_t, prev_gt = next_t, next_gt
                continue

            if abs(stamp_sec - prev_t) <= 1e-9:
                return prev_gt.copy(), 0.0
            if abs(stamp_sec - next_t) <= 1e-9:
                return next_gt.copy(), 0.0

            left_gap = stamp_sec - prev_t
            right_gap = next_t - stamp_sec
            if left_gap < -1e-9 or right_gap < -1e-9:
                return None, None
            if max(left_gap, right_gap) > tolerance:
                return None, None

            total_gap = next_t - prev_t
            if total_gap <= 1e-12:
                return prev_gt.copy(), max(left_gap, right_gap, 0.0)

            alpha = np.clip(left_gap / total_gap, 0.0, 1.0)
            xy = (1.0 - alpha) * prev_gt[:2] + alpha * next_gt[:2]
            dtheta = wrap2Pi(next_gt[2] - prev_gt[2])
            theta = wrap2Pi(prev_gt[2] + alpha * dtheta)
            return np.array([xy[0], xy[1], theta]), max(left_gap, right_gap)

        last_t, last_gt = history[-1]
        dt = abs(stamp_sec - last_t)
        if dt <= tolerance:
            return last_gt.copy(), dt
        return None, None

    def estimate_planar_alignment(self, source_xy, target_xy):
        if source_xy.shape != target_xy.shape:
            raise ValueError("source and target trajectories must have matching shapes")
        if source_xy.ndim != 2 or source_xy.shape[1] != 2:
            raise ValueError("planar alignment expects Nx2 arrays")
        if source_xy.shape[0] < 2:
            raise ValueError("need at least two poses to estimate a planar alignment")

        source_centroid = np.mean(source_xy, axis=0)
        target_centroid = np.mean(target_xy, axis=0)
        source_centered = source_xy - source_centroid
        target_centered = target_xy - target_centroid

        covariance = source_centered.T @ target_centered
        U, _, Vt = np.linalg.svd(covariance)
        rotation = Vt.T @ U.T
        if np.linalg.det(rotation) < 0.0:
            Vt[-1, :] *= -1.0
            rotation = Vt.T @ U.T

        translation = target_centroid - rotation @ source_centroid
        rotation_rad = np.arctan2(rotation[1, 0], rotation[0, 0])
        return rotation_rad, translation

    def suggest_mocap_calibration(self, gt_array, est_array):
        delta_rotation, delta_translation = self.estimate_planar_alignment(
            gt_array[:, :2],
            est_array[:, :2],
        )
        c = np.cos(delta_rotation)
        s = np.sin(delta_rotation)
        suggested_translation = np.array([
            c * self.host.mocap_translation[0] - s * self.host.mocap_translation[1],
            s * self.host.mocap_translation[0] + c * self.host.mocap_translation[1],
        ]) + delta_translation
        suggested_rotation = wrap2Pi(self.host.mocap_frame_rotation + delta_rotation)

        return {
            "mocap_current_frame_rotation_deg": float(np.rad2deg(self.host.mocap_frame_rotation)),
            "mocap_current_translation_x_m": float(self.host.mocap_translation[0]),
            "mocap_current_translation_y_m": float(self.host.mocap_translation[1]),
            "mocap_calibration_delta_rotation_deg": float(np.rad2deg(delta_rotation)),
            "mocap_calibration_delta_x_m": float(delta_translation[0]),
            "mocap_calibration_delta_y_m": float(delta_translation[1]),
            "mocap_suggested_frame_rotation_deg": float(np.rad2deg(suggested_rotation)),
            "mocap_suggested_translation_x_m": float(suggested_translation[0]),
            "mocap_suggested_translation_y_m": float(suggested_translation[1]),
        }

    def sanitize_artifact_name(self, value):
        value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
        value = value.strip("._-")
        return value or "eval_run"

    def plot_results(self):
        if not self.host._results:
            self.host.runtime.log_result_message(
                "info",
                "No evaluation data collected - mocap may not have published or could not be synchronized to odometry."
            )
            return
        results_array = np.array(self.host._results)
        gt_array = np.array(self.host._gt_history)
        est_array = np.array(self.host._est_history)
        metrics = summarize_results(results_array, gt_array)
        metrics.update(self.host.runtime.runtime_metrics())
        metrics["gt_eval_sync_tolerance_s"] = float(self.host.eval_gt_sync_tolerance_s)
        metrics["gt_eval_skipped_unsynced_count"] = int(self.host._eval_skipped_unsynced_gt)
        if self.host._eval_gt_sync_dt:
            sync_dt = np.array(self.host._eval_gt_sync_dt, dtype=float)
            metrics["gt_eval_sync_dt_mean_s"] = float(np.mean(sync_dt))
            metrics["gt_eval_sync_dt_max_s"] = float(np.max(sync_dt))

        calibration_metrics = None
        if est_array.shape == gt_array.shape and est_array.shape[0] >= 2:
            try:
                calibration_metrics = self.suggest_mocap_calibration(gt_array, est_array)
                metrics.update(calibration_metrics)
                self.host.runtime.log_result_message("info", "Suggested fixed mocap calibration for settings.yaml:")
                self.host.runtime.log_result_message(
                    "info",
                    f"  mocap_frame_rotation_deg: {metrics['mocap_suggested_frame_rotation_deg']:.6f}"
                )
                self.host.runtime.log_result_message(
                    "info",
                    f"  mocap_translation_x_m: {metrics['mocap_suggested_translation_x_m']:.6f}"
                )
                self.host.runtime.log_result_message(
                    "info",
                    f"  mocap_translation_y_m: {metrics['mocap_suggested_translation_y_m']:.6f}"
                )
            except Exception as exc:
                self.host.runtime.log_result_message(
                    "warn",
                    f"Could not compute mocap calibration suggestion: {exc}"
                )

        try:
            from evo.core import metrics as evo_metrics
            from evo.core.trajectory import PosePath3D, align_trajectory

            def _to_poses_se3(xyt):
                poses = []
                for x, y, theta in xyt:
                    c, s = np.cos(theta), np.sin(theta)
                    poses.append(np.array([
                        [c, -s, 0.0, x],
                        [s,  c, 0.0, y],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]))
                return poses

            traj_est = PosePath3D(poses_se3=_to_poses_se3(est_array))
            traj_ref = PosePath3D(poses_se3=_to_poses_se3(gt_array))
            traj_est_aligned = align_trajectory(traj_est, traj_ref, correct_scale=False)

            ape = evo_metrics.APE(evo_metrics.PoseRelation.translation_part)
            ape.process_data((traj_ref, traj_est_aligned))
            ape_stats = ape.get_all_statistics()
            metrics['evo_aligned_pos_rmse_m'] = float(ape_stats['rmse'])
            metrics['evo_aligned_pos_mean_m'] = float(ape_stats['mean'])
            metrics['evo_aligned_pos_max_m'] = float(ape_stats['max'])
            self.host.runtime.log_result_message(
                "info",
                f"evo aligned APE: rmse={ape_stats['rmse']:.3f} m  "
                f"mean={ape_stats['mean']:.3f} m  max={ape_stats['max']:.3f} m"
            )
        except ImportError:
            self.host.runtime.log_result_message(
                "warn",
                "evo not installed; skipping aligned metrics (pip install evo)"
            )
        except Exception as exc:
            self.host.runtime.log_result_message("warn", f"evo alignment failed: {exc}")

        self.host.runtime.log_result_message(
            "info",
            "Evaluation summary: "
            f"samples={metrics['num_samples']} "
            f"pos_rmse={metrics['pos_rmse_m']:.3f} m "
            f"theta_rmse={metrics['theta_rmse_deg']:.2f} deg "
            f"chi2_pass_rate={100.0 * metrics['chi2_pass_rate_95_3dof']:.1f}%"
        )
        self.host.runtime.log_result_message(
            "info",
            "Consistency split: "
            f"pos_chi2_pass={100.0 * metrics['pos_chi2_pass_rate_95_2dof']:.1f}% "
            f"theta_chi2_pass={100.0 * metrics['theta_chi2_pass_rate_95_1dof']:.1f}% "
            f"x_3sigma={100.0 * metrics['x_within_3sigma_rate']:.1f}% "
            f"y_3sigma={100.0 * metrics['y_within_3sigma_rate']:.1f}% "
            f"theta_3sigma={100.0 * metrics['theta_within_3sigma_rate']:.1f}%"
        )
        self.host.runtime.log_result_message(
            "info",
            "Evaluation extremes: "
            f"pos_max={metrics['pos_max_abs_m']:.3f} m "
            f"theta_max={metrics['theta_max_abs_deg']:.2f} deg "
            f"chi2_max={metrics['chi2_max']:.2f}"
        )
        self.host.runtime.log_result_message(
            "info",
            "Runtime: "
            f"wall={metrics['runtime_wall_seconds']:.2f}s "
            f"cpu={metrics['runtime_cpu_total_seconds']:.2f}s "
            f"avg_cpu={metrics['runtime_cpu_avg_percent']:.1f}% "
            f"max_rss={metrics['runtime_max_rss_kb']} KB"
        )
        if self.host._eval_gt_sync_dt:
            self.host.runtime.log_result_message(
                "info",
                "GT sync: "
                f"mean_dt={metrics['gt_eval_sync_dt_mean_s'] * 1e3:.1f} ms "
                f"max_dt={metrics['gt_eval_sync_dt_max_s'] * 1e3:.1f} ms "
                f"skipped={metrics['gt_eval_skipped_unsynced_count']}"
            )

        output_prefix = os.getenv("ROB530_EVAL_PREFIX", "").strip()
        if not output_prefix:
            bag_name = os.getenv("ROB530_BAG_NAME", "").strip()
            bag_label = pathlib.Path(bag_name).name.rstrip("/") if bag_name else "unknown_bag"
            run_label = self.sanitize_artifact_name(f"{bag_label}_{self.host.filter_name}")
            eval_root = pathlib.Path(
                os.getenv("ROB530_EVAL_DIR", "eval_outputs").strip() or "eval_outputs"
            )
            output_prefix = str(eval_root / run_label / run_label)

        no_show_env = os.getenv("ROB530_NO_SHOW", "").strip()
        no_show = True if no_show_env == "" else no_show_env.lower() in {"1", "true", "yes"}

        if output_prefix:
            prefix_path = pathlib.Path(output_prefix)
            prefix_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path = prefix_path.with_name(prefix_path.name + "_metrics.json")
            with metrics_path.open("w") as f:
                json.dump(metrics, f, indent=2, sort_keys=True)
            self.host.runtime.log_result_message("info", f"Saved evaluation metrics to {metrics_path}")
            summary_path = prefix_path.with_suffix(".txt")
            with summary_path.open("w") as f:
                f.write(f"bag={os.getenv('ROB530_BAG_NAME', '').strip() or 'unknown_bag'}\n")
                f.write(f"filter={self.host.filter_name}\n")
                f.write(
                    "Evaluation summary: "
                    f"samples={metrics['num_samples']} "
                    f"pos_rmse={metrics['pos_rmse_m']:.3f} m "
                    f"theta_rmse={metrics['theta_rmse_deg']:.2f} deg "
                    f"chi2_pass_rate={100.0 * metrics['chi2_pass_rate_95_3dof']:.1f}%\n"
                )
                f.write(
                    "Consistency split: "
                    f"pos_chi2_pass={100.0 * metrics['pos_chi2_pass_rate_95_2dof']:.1f}% "
                    f"theta_chi2_pass={100.0 * metrics['theta_chi2_pass_rate_95_1dof']:.1f}% "
                    f"x_3sigma={100.0 * metrics['x_within_3sigma_rate']:.1f}% "
                    f"y_3sigma={100.0 * metrics['y_within_3sigma_rate']:.1f}% "
                    f"theta_3sigma={100.0 * metrics['theta_within_3sigma_rate']:.1f}%\n"
                )
                f.write(
                    "Evaluation extremes: "
                    f"pos_max={metrics['pos_max_abs_m']:.3f} m "
                    f"theta_max={metrics['theta_max_abs_deg']:.2f} deg "
                    f"chi2_max={metrics['chi2_max']:.2f}\n\n"
                )
                f.write(
                    "Runtime: "
                    f"wall={metrics['runtime_wall_seconds']:.2f}s "
                    f"cpu={metrics['runtime_cpu_total_seconds']:.2f}s "
                    f"avg_cpu={metrics['runtime_cpu_avg_percent']:.1f}% "
                    f"max_rss={metrics['runtime_max_rss_kb']} KB\n\n"
                )
                if calibration_metrics is not None:
                    f.write("Suggested mocap calibration for settings.yaml:\n")
                    f.write(
                        f"mocap_frame_rotation_deg: {metrics['mocap_suggested_frame_rotation_deg']:.6f}\n"
                    )
                    f.write(
                        f"mocap_translation_x_m: {metrics['mocap_suggested_translation_x_m']:.6f}\n"
                    )
                    f.write(
                        f"mocap_translation_y_m: {metrics['mocap_suggested_translation_y_m']:.6f}\n\n"
                    )
                for key in sorted(metrics):
                    f.write(f"{key}: {metrics[key]}\n")
            self.host.runtime.log_result_message("info", f"Saved evaluation summary to {summary_path}")

        plot_error_with_options(
            results_array,
            gt_array,
            output_prefix=output_prefix if output_prefix else None,
            show=not no_show,
        )
