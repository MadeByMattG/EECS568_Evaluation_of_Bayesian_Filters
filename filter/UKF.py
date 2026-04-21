
from scipy.linalg import block_diag
from copy import deepcopy, copy
import numpy as np

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi


class UKF:
    # UKF construct an instance of this class
    #
    # Inputs:
    #   system: system and noise models
    #   init:   initial state mean and covariance

    def __init__(self, system, init):

        self.gfun = system.gfun  # motion model
        self.hfun = system.hfun  # measurement model
        self.M = system.M # motion noise covariance
        self.Q = system.Q # measurement noise covariance

        self.kappa_g = init.kappa_g
        self.sigma_point_jitter = getattr(init, 'sigma_point_jitter', 1.0e-9)

        self.state_ = RobotState()
        self.state_.setState(init.mu)
        self.state_.setCovariance(init.Sigma)
        self.n = 6


    def prediction(self, u, X, P , step):
        # prior belief
        if step == 0:
            mean = np.asarray(self.state_.getState(), dtype=float).reshape(3)
            sigma = np.asarray(self.state_.getCovariance(), dtype=float).reshape(3, 3)
        else:
            mean = np.asarray(X, dtype=float).reshape(3)
            sigma = np.asarray(P, dtype=float).reshape(3, 3)
        mean_aug = np.concatenate((mean, np.zeros(3))).reshape(-1, 1)
        sigma_aug = np.zeros((6,6))
        sigma_aug[:3,:3] = sigma
        sigma_aug[3:,3:] = self.M(u)
        self.sigma_point(mean_aug, sigma_aug, self.kappa_g)
        X_pred = np.zeros(3)
        self.Y = np.zeros((3, 2*self.n+1))
        for j in range(self.Y.shape[1]):
            self.Y[:,j] = self.gfun(self.X[:3,j], u+self.X[3:,j])
            self.Y[2, j] = wrap2Pi(self.Y[2, j])

        X_pred[:2] = self.Y[:2, :] @ self.w
        X_pred[2] = self._circular_mean(self.Y[2, :], self.w)

        X_pred_col = X_pred.reshape(-1, 1)
        state_residual = self.Y - X_pred_col
        state_residual[2, :] = self._wrap_residual(state_residual[2, :])
        P_pred = state_residual @ np.diag(self.w) @ state_residual.T
        P_pred = 0.5 * (P_pred + P_pred.T)

        self.state_.setState(X_pred)
        self.state_.setCovariance(P_pred)
        return np.copy(self.Y), np.copy(self.w), np.copy(X_pred), np.copy(P_pred)

    def correction(self, z, landmarks, Y, w, X, P):
        # Inputs:
        #   z: measurement [bearing, range, landmark_id] for a single landmark

        X_predict = np.asarray(X, dtype=float).reshape(3)
        P_predict = np.asarray(P, dtype=float).reshape(3, 3)
        self.Y = Y
        self.w = w

        landmark = landmarks.getLandmark(z[2].astype(int))

        Z = np.zeros((2, self.Y.shape[1]))
        z_hat = np.zeros(2)
        for j in range(Z.shape[1]):
            Z[:, j] = self.hfun(landmark.getPosition()[0], landmark.getPosition()[1], self.Y[:, j])
            Z[0, j] = wrap2Pi(Z[0, j])

        z_hat[0] = self._circular_mean(Z[0, :], self.w)
        z_hat[1] = Z[1, :] @ self.w

        measurement_residual = Z - z_hat.reshape(-1, 1)
        measurement_residual[0, :] = self._wrap_residual(measurement_residual[0, :])
        state_residual = self.Y - X_predict.reshape(-1,1)
        state_residual[2, :] = self._wrap_residual(state_residual[2, :])

        S = measurement_residual @ np.diag(self.w) @ measurement_residual.T + self.Q
        Sigma_xz = state_residual @ np.diag(self.w) @ measurement_residual.T
        K = Sigma_xz @ np.linalg.inv(S)
        diff = np.array([
                wrap2Pi(z[0] - z_hat[0]),
                z[1] - z_hat[1]])
        X = X_predict + K @ diff
        X[2] = wrap2Pi(X[2])
        P = P_predict - K @ S @ K.T
        P = 0.5 * (P + P.T)
        X = X.reshape(3)

        self.state_.setState(X)
        self.state_.setCovariance(P)
        return np.copy(X), np.copy(P)

    def sigma_point(self, mean, cov, kappa):
        self.n = len(mean) # dim of state
        L = np.sqrt(self.n + kappa) * self._covariance_sqrt(cov)
        Y = mean.repeat(len(mean), axis=1)
        self.X = np.hstack((mean, Y+L, Y-L))
        self.w = np.zeros([2 * self.n + 1, 1])
        self.w[0] = kappa / (self.n + kappa)
        self.w[1:] = 1 / (2 * (self.n + kappa))
        self.w = self.w.reshape(-1)

    def _covariance_sqrt(self, cov):
        cov = np.array(cov, dtype=float)
        cov = 0.5 * (cov + cov.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 0.0, None)
        eigvals = np.maximum(eigvals, self.sigma_point_jitter)
        return eigvecs @ np.diag(np.sqrt(eigvals))

    def _circular_mean(self, angles, weights):
        return np.arctan2(
            np.sum(weights * np.sin(angles)),
            np.sum(weights * np.cos(angles)),
        )

    def _wrap_residual(self, residual):
        return np.arctan2(np.sin(residual), np.cos(residual))

    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state
