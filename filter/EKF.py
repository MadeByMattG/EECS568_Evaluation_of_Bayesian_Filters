import numpy as np
from scipy.linalg import block_diag
from copy import deepcopy, copy

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi

class EKF:

    def __init__(self, system, init):
        # EKF Construct an instance of this class
        # Inputs:
        #   system: system and noise models
        #   init:   initial state mean and covariance
        self.gfun = system.gfun  # motion model
        self.hfun = system.hfun  # measurement model
        self.Gfun = init.Gfun  # Jocabian of motion model
        self.Vfun = init.Vfun  # Jocabian of motion model
        self.Hfun = init.Hfun  # Jocabian of measurement model
        self.M = system.M # motion noise covariance
        self.Q = system.Q # measurement noise covariance

        self.state_ = RobotState()

        # init state
        self.state_.setState(init.mu)
        self.state_.setCovariance(init.Sigma)


    ## Do prediction and set state in RobotState()
    def prediction(self, u, X, P, step):
        if step == 0:
            X = self.state_.getState()
            P = self.state_.getCovariance()

        G = self.Gfun(X, u)
        V = self.Vfun(X, u)
        X_pred = self.gfun(X, u)
        P_pred = G @ P @ G.T + V @ self.M(u) @ V.T

        self.state_.setState(X_pred)
        self.state_.setCovariance(P_pred)
        return np.copy(X_pred), np.copy(P_pred)


    def correction(self, z, landmarks, X, P):
        # EKF correction step
        #
        # Inputs:
        #   z:  measurement [bearing, range, landmark_id] for a single landmark
        X_predict = X
        P_predict = P

        landmark = landmarks.getLandmark(z[2].astype(int))

        z_hat = self.hfun(landmark.getPosition()[0], landmark.getPosition()[1], X_predict)

        # evaluate measurement Jacobian at current operating point
        H = self.Hfun(landmark.getPosition()[0], landmark.getPosition()[1], X_predict, z_hat)

        S = H @ P_predict @ H.T + self.Q  # innovation covariance

        K = P_predict @ H.T @ np.linalg.inv(S)  # Kalman gain

        # correct the predicted state statistics
        diff = [
                wrap2Pi(z[0] - z_hat[0]),
                z[1] - z_hat[1]]
        X = X_predict + K @ diff
        X[2] = wrap2Pi(X[2])
        U = np.eye(np.shape(X)[0]) - K @ H
        P = U @ P_predict @ U.T + K @ self.Q @ K.T

        self.state_.setState(X)
        self.state_.setCovariance(P)
        return np.copy(X), np.copy(P)


    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state
