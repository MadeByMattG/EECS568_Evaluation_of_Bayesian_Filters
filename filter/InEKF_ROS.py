import numpy as np
from rclpy.clock import Clock
from filter.InEKF import InEKF

class InEKF_ROS(InEKF):
    def __init__(self, system, init):
        super().__init__(system, init)

    def prediction(self, u, Sigma, mu, step):
        mu_pred, sigma_pred = super().prediction(u, Sigma, mu, step)
        self.state_.setTime(Clock().now().to_msg())
        X = np.array([mu_pred[0, 2], mu_pred[1, 2], np.arctan2(mu_pred[1, 0], mu_pred[0, 0])])
        self.state_.setState(X)
        self.state_.setCovariance(sigma_pred)
        return mu_pred, sigma_pred

    def propagation(self, u, adjX, mu, Sigma , W):
        mu_pred, sigma_pred = super().propagation(u, adjX, mu, Sigma, W)
        return mu_pred, sigma_pred

    def correction(self, Y, z, landmarks, mu_pred, sigma_pred):
        X, Sigma, mu = super().correction(Y, z, landmarks, mu_pred, sigma_pred)
        self.state_.setTime(Clock().now().to_msg())
        return X, Sigma, mu