import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, logm

from system.RobotState import *

def wrap2Pi(input):
    phases =  (( -input + np.pi) % (2.0 * np.pi ) - np.pi) * -1.0

    return phases

def func(x):
    G1 = np.array([[0,0,1],
                    [0,0,0],
                    [0,0,0]])

    G2 = np.array([[0,0,0],
                    [0,0,1],
                    [0,0,0]])

    G3 = np.array([[0,-1,0],
                    [1,0,0],
                    [0,0,0]])

    X = expm(x[0]*G1+x[1]*G2+x[2]*G3)
    y = np.zeros(3)

    y[0] = X[0,2]
    y[1] = X[1,2]
    y[2] = np.arctan2(X[1,0],X[0,0])
    return y


def unscented_propagate(mean, cov, kappa):
    n = np.size(mean)

    x_in = np.copy(mean)
    P_in = np.copy(cov)

    # sigma points
    L = np.sqrt(n + kappa) * np.linalg.cholesky(P_in)
    Y_temp = np.tile(x_in, (n, 1)).T
    X = np.copy(x_in.reshape(-1, 1))
    X = np.hstack((X, Y_temp + L))
    X = np.hstack((X, Y_temp - L))

    w = np.zeros((2 * n + 1, 1))
    w[0] = kappa / (n + kappa)
    w[1:] = 1 / (2 * (n + kappa))

    new_mean = np.zeros((n, 1))
    new_cov = np.zeros((n, n))
    Y = np.zeros((n, 2 * n + 1))
    for j in range(2 * n + 1):
        Y[:,j] = func(X[:,j])
        new_mean[:,0] = new_mean[:,0] + w[j] * Y[:,j]

    diff = Y - new_mean
    for j in range(np.shape(diff)[1]):
        diff[2,j] = wrap2Pi(diff[2,j])

    w_mat = np.zeros((np.shape(w)[0],np.shape(w)[0]))
    np.fill_diagonal(w_mat,w)
    new_cov = diff @ w_mat @ diff.T
    cov_xy = (X - x_in.reshape(-1,1)) @ w_mat @ diff.T

    return new_cov


def lieToCartesian(mean, cov):
    mean_matrix = np.eye(3)
    R = np.array([[np.cos(mean[2]),-np.sin(mean[2])],[np.sin(mean[2]),np.cos(mean[2])]])
    mean_matrix[0:2,0:2] = R
    mean_matrix[0,2] = mean[0]
    mean_matrix[1,2] = mean[1]

    kappa = 2
    X = logm(mean_matrix)
    x = np.zeros(3)
    x[0] = X[0,2]
    x[1] = X[1,2]
    x[2] = X[1,0]

    mu_cart = np.array([mean[0], mean[1], mean[2]])
    Sigma_cart = unscented_propagate(x, cov, kappa)

    return mu_cart, Sigma_cart

def mahalanobis(state, ground_truth, filter_name, Lie2Cart):
    # Output format (9D vector) : 1. Full-state chi-square (3 DOF)
    #                             2-4. difference between groundtruth and filter estimation in x,y,theta
    #                             5-7. 3*sigma (square root of variance) value of x,y, theta
    #                             8. Position-only chi-square (2 DOF)
    #                             9. Heading-only chi-square (1 DOF)

    def _stabilize_covariance(covariance):
        covariance = np.asarray(covariance, dtype=float)
        covariance = 0.5 * (covariance + covariance.T)
        diag_idx = np.diag_indices_from(covariance)
        covariance[diag_idx] = np.maximum(covariance[diag_idx], 1.0e-12)
        return covariance

    def _safe_mahalanobis(diff, covariance):
        covariance = _stabilize_covariance(covariance)
        eye = np.eye(covariance.shape[0])
        for jitter in (0.0, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6):
            try:
                solved = np.linalg.solve(covariance + jitter * eye, diff)
                return max(float(diff.T @ solved), 0.0), covariance + jitter * eye
            except np.linalg.LinAlgError:
                continue
        solved = np.linalg.pinv(covariance) @ diff
        return max(float(diff.T @ solved), 0.0), covariance

    ground_truth = np.asarray(ground_truth, dtype=float).copy()
    results = np.zeros((9,1))
    if filter_name != "InEKF":
        mu = state.getState()
        Sigma = state.getCovariance()
        ground_truth[2]=wrap2Pi(ground_truth[2])
        diff = ground_truth - mu
        diff[2] = wrap2Pi(diff[2])
        chi2, Sigma = _safe_mahalanobis(diff, Sigma)
        chi2_pos, Sigma_pos = _safe_mahalanobis(diff[:2], Sigma[:2, :2])
        chi2_theta, Sigma_theta = _safe_mahalanobis(
            np.array([diff[2]], dtype=float),
            np.array([[Sigma[2, 2]]], dtype=float),
        )
        results[0,0] = chi2
        results[1:4,0] = diff
        results[4] = 3*np.sqrt(Sigma[0,0])
        results[5] = 3*np.sqrt(Sigma[1,1])
        results[6] = 3*np.sqrt(Sigma[2,2])
        results[7] = chi2_pos
        results[8] = chi2_theta


        return results.reshape(-1)
    else:
        mu = state.getState()
        if Lie2Cart:
            Sigma_cart = state.getCartesianCovariance()
            mu_cart = state.getCartesianState()
        else:
            mu_cart = np.zeros((np.shape(mu)))
            Sigma_cart = np.eye(np.shape(mu)[0])
        ground_truth[2]=wrap2Pi(ground_truth[2])
        diff = ground_truth - mu_cart
        diff[2] = wrap2Pi(diff[2])
        chi2, Sigma_cart = _safe_mahalanobis(diff, Sigma_cart)
        chi2_pos, Sigma_pos = _safe_mahalanobis(diff[:2], Sigma_cart[:2, :2])
        chi2_theta, Sigma_theta = _safe_mahalanobis(
            np.array([diff[2]], dtype=float),
            np.array([[Sigma_cart[2, 2]]], dtype=float),
        )
        results[0,0] = chi2
        results[1:4,0] = diff
        results[4] = 3*np.sqrt(Sigma_cart[0,0])
        results[5] = 3*np.sqrt(Sigma_cart[1,1])
        results[6] = 3*np.sqrt(Sigma_cart[2,2])
        results[7] = chi2_pos
        results[8] = chi2_theta
        return results.reshape(-1)


def plot_error(results, gt):
    return plot_error_with_options(results, gt, output_prefix=None, show=True)


def summarize_results(results, gt=None):
    metrics = {}

    if results.size == 0:
        metrics["num_samples"] = 0
        return metrics

    chi2 = results[:, 0]
    dx = results[:, 1]
    dy = results[:, 2]
    dtheta = wrap2Pi(results[:, 3])
    sigma_x = results[:, 4]
    sigma_y = results[:, 5]
    sigma_theta = results[:, 6]

    pos_err = np.sqrt(dx**2 + dy**2)
    theta_err_deg = np.rad2deg(dtheta)

    chi2_thresh = 7.81
    chi2_pos = results[:, 7] if results.shape[1] >= 8 else (dx / np.maximum(sigma_x / 3.0, 1.0e-12))**2 + (dy / np.maximum(sigma_y / 3.0, 1.0e-12))**2
    chi2_theta = results[:, 8] if results.shape[1] >= 9 else (dtheta / np.maximum(sigma_theta / 3.0, 1.0e-12))**2
    chi2_pos_thresh = 5.991464547107979
    chi2_theta_thresh = 3.841458820694124
    metrics["num_samples"] = int(results.shape[0])
    metrics["chi2_mean"] = float(np.mean(chi2))
    metrics["chi2_median"] = float(np.median(chi2))
    metrics["chi2_max"] = float(np.max(chi2))
    metrics["chi2_pass_rate_95_3dof"] = float(np.mean(chi2 <= chi2_thresh))
    metrics["pos_chi2_mean"] = float(np.mean(chi2_pos))
    metrics["pos_chi2_median"] = float(np.median(chi2_pos))
    metrics["pos_chi2_max"] = float(np.max(chi2_pos))
    metrics["pos_chi2_pass_rate_95_2dof"] = float(np.mean(chi2_pos <= chi2_pos_thresh))
    metrics["theta_chi2_mean"] = float(np.mean(chi2_theta))
    metrics["theta_chi2_median"] = float(np.median(chi2_theta))
    metrics["theta_chi2_max"] = float(np.max(chi2_theta))
    metrics["theta_chi2_pass_rate_95_1dof"] = float(np.mean(chi2_theta <= chi2_theta_thresh))

    metrics["x_rmse_m"] = float(np.sqrt(np.mean(dx**2)))
    metrics["y_rmse_m"] = float(np.sqrt(np.mean(dy**2)))
    metrics["pos_rmse_m"] = float(np.sqrt(np.mean(pos_err**2)))
    metrics["theta_rmse_deg"] = float(np.sqrt(np.mean(theta_err_deg**2)))

    metrics["x_max_abs_m"] = float(np.max(np.abs(dx)))
    metrics["y_max_abs_m"] = float(np.max(np.abs(dy)))
    metrics["pos_max_abs_m"] = float(np.max(pos_err))
    metrics["theta_max_abs_deg"] = float(np.max(np.abs(theta_err_deg)))

    metrics["x_within_3sigma_rate"] = float(np.mean(np.abs(dx) <= sigma_x))
    metrics["y_within_3sigma_rate"] = float(np.mean(np.abs(dy) <= sigma_y))
    metrics["theta_within_3sigma_rate"] = float(np.mean(np.abs(dtheta) <= sigma_theta))

    metrics["final_x_error_m"] = float(dx[-1])
    metrics["final_y_error_m"] = float(dy[-1])
    metrics["final_pos_error_m"] = float(pos_err[-1])
    metrics["final_theta_error_deg"] = float(theta_err_deg[-1])

    return metrics


def plot_error_with_options(results, gt, output_prefix=None, show=True):
    num_data_range = range(np.shape(results)[0])

    gt_x = gt[:,0]
    gt_y = gt[:,1]

    plot2 = plt.figure(2)
    plt.plot(num_data_range,results[:,0])
    plt.plot(num_data_range, 7.81*np.ones(np.shape(results)[0]))
    plt.title("Chi-square Statistics")
    plt.legend(["Chi-square Statistics", "p = 0.05 in 3 DOF"])
    plt.xlabel("Iterations")

    plot3,  (ax1, ax2, ax3) = plt.subplots(3,1)
    ax1.set_title("Deviation from Ground Truth with 3rd Sigma Contour")
    ax1.plot(num_data_range, results[:,1])
    ax1.set_ylabel("X")
    ax1.plot(num_data_range,results[:,4],'r')
    ax1.plot(num_data_range,-1*results[:,4],'r')

    ax1.legend(["Deviation from Ground Truth","3rd Sigma Contour"])
    ax2.plot(num_data_range,results[:,2])
    ax2.plot(num_data_range,results[:,5],'r')
    ax2.plot(num_data_range,-1*results[:,5],'r')
    ax2.set_ylabel("Y")

    ax3.plot(num_data_range,results[:,3])
    ax3.plot(num_data_range,results[:,6],'r')
    ax3.plot(num_data_range,-1*results[:,6],'r')
    ax3.set_ylabel("theta")
    ax3.set_xlabel("Iterations")

    if output_prefix:
        plot2.savefig(f"{output_prefix}_chi2.png", dpi=150, bbox_inches="tight")
        plot3.savefig(f"{output_prefix}_error.png", dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(plot2)
        plt.close(plot3)


def main():

    i = -7
    j = wrap2Pi(i)
    print(j)


if __name__ == '__main__':
    main()
