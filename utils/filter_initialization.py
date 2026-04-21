
import numpy as np
from functools import partial
from numpy.random import default_rng
# rng = default_rng()
rng = default_rng(seed=3)  #set seed to get deterministic random number generator

from copy import deepcopy, copy
from utils.utils import wrap2Pi

class myStruct:
    pass

init = myStruct()

def Gfun(mu,u):
    if abs(u[1]) < 1e-6:  # straight-line motion: avoid division by zero
        output = np.array([[1, 0, -u[0]*np.sin(mu[2])],
                           [0, 1,  u[0]*np.cos(mu[2])],
                           [0, 0,  1]])
    else:
        output = np.array([[1,0,(u[0]*np.cos(mu[2]+u[1]))/u[1]-(u[0]*np.cos(mu[2]))/u[1]],\
                            [0,1,(u[0]*np.sin(mu[2]+u[1]))/u[1]-(u[0]*np.sin(mu[2]))/u[1]],\
                            [0,0,1]])
    return output

def Vfun(mu,u):
    if abs(u[1]) < 1e-6:  # straight-line motion: avoid division by zero
        output = np.array([[np.cos(mu[2]), 0, 0],
                           [np.sin(mu[2]), 0, 0],
                           [0,             0, 1]])
    else:
        output = np.array([[np.sin(mu[2]+u[1])/u[1]-np.sin(mu[2])/u[1], (u[0]*np.cos(mu[2]+u[1]))/u[1]-(u[0]*np.sin(mu[2]+u[1]))/(u[1]**2)+(u[0]*np.sin(mu[2]))/(u[1]**2), 0],\
                           [np.cos(mu[2])/u[1]-np.cos(mu[2]+u[1])/u[1], (u[0]*np.cos(mu[2]+u[1]))/(u[1]**2)+(u[0]*np.sin(mu[2]+u[1]))/u[1]-(u[0]*np.cos(mu[2]))/(u[1]**2), 0],\
                           [0,                                          1,                                                                                             1]])
    return output

def Hfun(landmark_x, landmark_y, mu_pred, z_hat):
    output = np.array([
        [(landmark_y-mu_pred[1])/(z_hat[1]**2),   -(landmark_x-mu_pred[0])/(z_hat[1]**2),-1],\
        [-(landmark_x-mu_pred[0])/z_hat[1],       -(landmark_y-mu_pred[1])/z_hat[1],0]])
    return output


def _covariance_sqrt(cov):
    cov = np.array(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    return eigvecs @ np.diag(np.sqrt(eigvals))


def filter_initialization(sys, initialStateMean, initialStateCov, filter_name, filter_params=None):
    filter_params = filter_params or {}

    if filter_name == 'EKF':
        init.mu = initialStateMean
        init.Sigma = initialStateCov
        init.Gfun = Gfun
        init.Vfun = Vfun
        init.Hfun = Hfun
        from filter.EKF_ROS import EKF_ROS
        filter = EKF_ROS(sys, init)

    if filter_name == 'UKF':
        init.mu = initialStateMean
        init.Sigma = initialStateCov
        init.kappa_g = float(filter_params.get('ukf_kappa_g', 2.0))
        init.sigma_point_jitter = float(filter_params.get('ukf_sigma_point_jitter', 1.0e-9))
        from filter.UKF_ROS import UKF_ROS
        filter = UKF_ROS(sys, init)

    if filter_name == 'PF':
        init.mu = initialStateMean
        init.n = int(filter_params.get('pf_num_particles', 500))
        init.resample_threshold_ratio = float(
            filter_params.get('pf_resample_threshold_ratio', 0.2)
        )
        pf_initial_state_std = filter_params.get('pf_initial_state_std')
        if pf_initial_state_std is None:
            init.Sigma = np.array(initialStateCov, dtype=float)
        else:
            pf_initial_state_std = np.asarray(pf_initial_state_std, dtype=float).reshape(3)
            init.Sigma = np.diag(pf_initial_state_std**2)
        init.particles = np.zeros((3, init.n))
        init.particle_weight = np.full(init.n, 1 / init.n)
        L = _covariance_sqrt(init.Sigma)
        for i in range(init.n):
            init.particles[:,i] = L @ rng.standard_normal((3,1)).reshape(3) + init.mu
            init.particles[2, i] = wrap2Pi(init.particles[2, i])
        from filter.PF_ROS import PF_ROS
        filter = PF_ROS(sys, init)


    if filter_name == "InEKF":
        theta0 = initialStateMean[2]
        init.mu = np.array([[np.cos(theta0), -np.sin(theta0), initialStateMean[0]],
                            [np.sin(theta0),  np.cos(theta0), initialStateMean[1]],
                            [0,               0,              1]])
        init.Sigma = initialStateCov; # note: the covariance here is wrt lie algebra (not wrt Cartesian coordinate (x,y,theta))
        from filter.InEKF_ROS import InEKF_ROS
        filter = InEKF_ROS(sys, init)

    if filter_name == 'test':
        init.mu = initialStateMean
        init.Sigma = initialStateCov
        from filter.DummyFilter import DummyFilter
        filter = DummyFilter(sys, init)

    return filter
