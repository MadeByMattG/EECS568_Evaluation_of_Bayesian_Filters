from rclpy.clock import Clock
from filter.EKF import EKF

class EKF_ROS(EKF):
    def __init__(self, system, init):
        super().__init__(system, init)

    def prediction(self, u , X, P, step):
        # EKF propagation (prediction) step
        X_pred, P_pred = super().prediction(u, X, P, step)
        self.state_.setTime(Clock().now().to_msg())
        return X_pred, P_pred

    def correction(self, z, landmarks, X, P):
        # Add ROS-related correction method here
        X, P = super().correction(z, landmarks, X, P)
        self.state_.setTime(Clock().now().to_msg())
        return X, P
