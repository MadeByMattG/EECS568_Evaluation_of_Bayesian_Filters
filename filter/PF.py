
from statistics import mean
from scipy.linalg import block_diag
from copy import deepcopy, copy
import numpy as np

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi

from scipy.stats import multivariate_normal
from numpy.random import default_rng
# rng = default_rng()
rng = default_rng(seed=3)  #set seed to get deterministic random number generator

class PF:
    # PF construct an instance of this class
    #
    # Inputs:
    #   system: system and noise models
    #   init:   initial state mean and covariance

    def __init__(self, system, init):
        self.gfun = system.gfun  # motion model
        self.hfun = system.hfun  # measurement model
        self.M = system.M # motion noise covariance
        self.Q = system.Q # measurement noise covariance

        # PF parameters
        self.n = init.n
        self.Sigma = init.Sigma
        self.particles = init.particles
        self.particle_weight = init.particle_weight
        self.resample_threshold_ratio = getattr(init, 'resample_threshold_ratio', 0.2)
        self.last_neff = float(self.n)
        self.resampled_last_step = False


        self.state_ = RobotState()
        self.state_.setState(init.mu)
        self.state_.setCovariance(init.Sigma)


    def prediction(self, u, particles, step):
        if step != 0:
            self.particles = particles
        noise_transform = self._motion_noise_transform(u)
        for j in range(self.n):
            if noise_transform is None:
                noise = np.zeros((3, 1))
            else:
                noise = noise_transform @ rng.standard_normal((3, 1))
            sample_action = noise + u.reshape(-1,1)
            self.particles[:,j] = self.gfun(self.particles[:,j],sample_action.reshape(3))
        self.mean_variance()
        return self.particles

    def _motion_noise_transform(self, u):
        """
        Return a square-root factor for the action noise covariance.

        The PF motion model occasionally receives zero or near-zero odometry
        increments, especially now that long /odom gaps are integrated in
        substeps. In those cases M(u) can be positive semidefinite with exact
        zeros on the diagonal, and a raw Cholesky factorization fails even
        though the covariance is valid. Clamp tiny negative eigenvalues from
        roundoff and return None for the all-zero case.
        """
        cov = np.array(self.M(u), dtype=float)
        cov = 0.5 * (cov + cov.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 0.0, None)
        if np.max(eigvals) <= 1e-15:
            return None
        return eigvecs @ np.diag(np.sqrt(eigvals))


    def correction(self, z, landmarks, particles, particle_weight, step):
        # Inputs:
        #   z: measurement [bearing, range, landmark_id] for a single landmark
        landmark = landmarks.getLandmark(z[2].astype(int))

        self.particles = particles
        if step != 0:
            self.particle_weight = particle_weight
        weight = np.zeros(self.n)
        for j in range(self.n):
            z_hat = self.hfun(landmark.getPosition()[0], landmark.getPosition()[1], self.particles[:, j])
            diff = np.array([
                wrap2Pi(z[0] - z_hat[0]),
                z[1] - z_hat[1]])
            weight[j] = multivariate_normal.pdf(diff, np.zeros(2), cov=self.Q)
        self.particle_weight = np.multiply(self.particle_weight, weight)
        weight_sum = np.sum(self.particle_weight)
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            self.particle_weight = np.full(self.n, 1.0 / self.n)
        else:
            self.particle_weight = self.particle_weight / weight_sum
        self.last_neff = 1 / np.sum(np.power(self.particle_weight, 2))
        self.resampled_last_step = False
        if self.last_neff < self.resample_threshold_ratio * self.n:
            self.resample()
            self.resampled_last_step = True

        X, P = self.mean_variance()
        return np.copy(X), np.copy(P), np.copy(self.particles), np.copy(self.particle_weight)


    def resample(self):
        new_samples = np.zeros_like(self.particles)
        new_weight = np.full_like(self.particle_weight, 1 / self.n, dtype=float)
        W = np.cumsum(self.particle_weight)
        r = rng.random() / self.n
        count = 0
        for j in range(self.n):
            u = r + j/self.n
            while count < self.n - 1 and u > W[count]:
                count += 1
            new_samples[:,j] = self.particles[:,count]
        self.particles = new_samples
        self.particle_weight = new_weight


    def mean_variance(self):
        weights = np.asarray(self.particle_weight, dtype=float).reshape(-1)
        weight_sum = np.sum(weights)
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            weights = np.full(self.n, 1.0 / self.n)
        else:
            weights = weights / weight_sum

        X = np.zeros(3)
        X[:2] = self.particles[:2, :] @ weights
        sin_sum = np.sum(weights * np.sin(self.particles[2, :]))
        cos_sum = np.sum(weights * np.cos(self.particles[2, :]))
        X[2] = np.arctan2(sin_sum, cos_sum)

        zero_mean = self.particles - X.reshape(-1, 1)
        zero_mean[2, :] = np.arctan2(np.sin(zero_mean[2, :]), np.cos(zero_mean[2, :]))
        P = zero_mean @ np.diag(weights) @ zero_mean.T
        P = 0.5 * (P + P.T)
        self.state_.setState(X)
        self.state_.setCovariance(P)
        return np.copy(X), np.copy(P)

    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state
