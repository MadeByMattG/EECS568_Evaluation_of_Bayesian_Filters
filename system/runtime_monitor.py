import resource
import time

import rclpy


class RuntimeMonitor:
    def __init__(self, host):
        self.host = host
        self._wall_start_time = time.perf_counter()
        self._cpu_start_usage = resource.getrusage(resource.RUSAGE_SELF)
        self._last_runtime_activity_wall_time = time.monotonic()
        self._received_runtime_activity = False
        self._shutdown_requested = False

    def mark_runtime_activity(self):
        self._last_runtime_activity_wall_time = time.monotonic()
        self._received_runtime_activity = True

    def idle_shutdown_check(self):
        if not self.host.auto_shutdown_on_idle or self._shutdown_requested:
            return
        if not self._received_runtime_activity:
            return
        idle_s = time.monotonic() - self._last_runtime_activity_wall_time
        if idle_s < self.host.idle_shutdown_wall_s:
            return
        self._shutdown_requested = True
        if self.host._idle_shutdown_timer is not None:
            self.host._idle_shutdown_timer.cancel()
        self.log_result_message(
            "info",
            f"No incoming messages for {idle_s:.1f}s after bag playback; shutting down."
        )
        if rclpy.ok():
            rclpy.shutdown()

    def log_result_message(self, level, message):
        try:
            if rclpy.ok():
                getattr(self.host.get_logger(), level)(message)
                return
        except Exception:
            pass
        print(f"[{level.upper()}] [robot_state_estimator]: {message}")

    def runtime_metrics(self):
        usage = resource.getrusage(resource.RUSAGE_SELF)
        wall_seconds = max(time.perf_counter() - self._wall_start_time, 0.0)
        user_seconds = max(usage.ru_utime - self._cpu_start_usage.ru_utime, 0.0)
        system_seconds = max(usage.ru_stime - self._cpu_start_usage.ru_stime, 0.0)
        cpu_seconds = user_seconds + system_seconds
        avg_cpu_percent = 100.0 * cpu_seconds / wall_seconds if wall_seconds > 1e-9 else 0.0
        return {
            "runtime_wall_seconds": float(wall_seconds),
            "runtime_cpu_user_seconds": float(user_seconds),
            "runtime_cpu_system_seconds": float(system_seconds),
            "runtime_cpu_total_seconds": float(cpu_seconds),
            "runtime_cpu_avg_percent": float(avg_cpu_percent),
            "runtime_max_rss_kb": int(usage.ru_maxrss),
        }
