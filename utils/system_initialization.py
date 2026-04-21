
import numpy as np
from functools import partial

from utils.utils import wrap2Pi

class myStruct:
    pass

def gfun(mu, u):
    output = np.zeros(3)
    if abs(u[1]) < 1e-6:  # straight-line motion: avoid division by zero
        output[0] = mu[0] + u[0] * np.cos(mu[2])
        output[1] = mu[1] + u[0] * np.sin(mu[2])
        output[2] = mu[2] + u[2]
    else:
        output[0] = mu[0] + (-u[0] / u[1] * np.sin(mu[2]) + u[0] / u[1] * np.sin(mu[2] + u[1]))
        output[1] = mu[1] + ( u[0] / u[1] * np.cos(mu[2]) - u[0] / u[1] * np.cos(mu[2] + u[1]))
        output[2] = mu[2] + u[1] + u[2]
    return output

def hfun(landmark_x, landmark_y, mu_pred):
    output = np.zeros(2)
    output[0] = wrap2Pi(np.arctan2(landmark_y - mu_pred[1], landmark_x - mu_pred[0]) - mu_pred[2])
    output[1] = np.sqrt((landmark_y - mu_pred[1])**2 + (landmark_x - mu_pred[0])**2)
    return output

def M(u, alphas):
    output = np.array([[alphas[0]*u[0]**2+alphas[1]*u[1]**2,0,0], \
        [0,alphas[2]*u[0]**2+alphas[3]*u[1]**2,0],\
            [0,0,alphas[4]*u[0]**2+alphas[5]*u[1]**2]])
    return output

def system_initialization(
    alphas,
    beta,
    tag_body_position_std_m=0.25,
    range_std_m=0.15,
    inekf_process_noise_std=None,
):
    sys = myStruct()

    sys.gfun = gfun
    sys.hfun = hfun
    sys.M = partial(M, alphas=alphas)
    sys.Q = np.array([[beta**2, 0], [0, range_std_m**2]])
    if inekf_process_noise_std is None:
        inekf_process_noise_std = [0.05, 0.10, 0.10]
    inekf_process_noise_std = np.asarray(inekf_process_noise_std, dtype=float).reshape(3)
    sys.W = np.diag(inekf_process_noise_std**2)
    sys.V = np.diag([
        tag_body_position_std_m**2,
        tag_body_position_std_m**2,
    ])

    return sys
