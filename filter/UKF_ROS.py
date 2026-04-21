from rclpy.clock import Clock
from filter.UKF import UKF

class UKF_ROS(UKF):
    def __init__(self, system, init):
        super().__init__(system, init)

    def prediction(self, u, X, P , step):
        Y, w, X_pred, P_pred = super().prediction(u, X, P , step)
        self.state_.setTime(Clock().now().to_msg())
        return Y, w, X_pred, P_pred

    def correction(self, z, landmarks, Y, w, X, P):
        X, P = super().correction(z, landmarks, Y, w, X, P)
        self.state_.setTime(Clock().now().to_msg())
        return X, P