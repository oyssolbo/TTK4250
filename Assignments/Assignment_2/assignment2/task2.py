from operator import matmul
import numpy as np
import solution
from numpy import ndarray
from scipy.stats import norm


def condition_mean(x: ndarray, P: ndarray,
                   z: ndarray, R: ndarray, H: ndarray) -> ndarray:
    """compute conditional mean

    Args:
        x (ndarray): initial state
        P (ndarray): initial state covariance
        z (ndarray): measurement
        R (ndarray): measurement covariance
        H (ndarray): measurement matrix i.e. z = H @ x + error

    Returns:
        cond_mean (ndarray): conditioned mean (state)
    """
    PH = np.matmul(P, np.transpose(H)) # HP = (PH)^T
    W = np.matmul(PH, np.linalg.inv(np.matmul(H, PH) + R))
    z_HX = z - np.matmul(H, x)

    return x - np.matmul(W, z_HX)


def condition_cov(P: ndarray, R: ndarray, H: ndarray) -> ndarray:
    """compute conditional covariance

    Args:
        P (ndarray): covariance of state estimate
        R (ndarray): covariance of measurement
        H (ndarray): measurement matrix

    Returns:
        ndarray: the conditioned covariance
    """
    PH = np.matmul(P, np.transpose(H)) # HP = (PH)^T
    W = np.matmul(PH, np.linalg.inv(np.matmul(H, PH) + R))
    W_HP = np.matmul(W, np.transpose(PH))

    return P - W_HP


def get_task_2f(x_bar: ndarray, P: ndarray,
                z_c: ndarray, R_c: ndarray, H_c: ndarray,
                z_r: ndarray, R_r: ndarray, H_r: ndarray
                ):
    """get state estimates after receiving measurement c or measurement r

    Args:
        x_bar (ndarray): initial state estimate
        P (ndarray): covariance of x_bar
        z_c (ndarray): measurement c
        R_c (ndarray): covariance of measurement c
        H_c (ndarray): measurement matrix i.e. z_c = H_c @ x + error
        z_r (ndarray): measurement r
        R_r (ndarray): covariance of measurement r
        H_r (ndarray): measurement matrix i.e. z_r + H_c @ x + error

    Returns:
        x_bar_c (ndarray): state estimate after measurement c
        P_c (ndarray): covariance of x_bar_c
        x_bar_r (ndarray): state estimate after measurement r
        P_r (ndarray): covariance of x_bar_r
    """
    # This function acts as a method for creating the a posteriori
    # estimate of the kalman filter
    z = np.array([[z_c], [z_r]])
    H = np.concatenate((H_c, H_r), axis=0)
    R = np.concatenate((R_c, R_r), axis=0)

    PH = np.matmul(P, np.transpose(H))
    W = np.matmul(np.transpose(PH), np.transpose(H)) + R
    KF = np.matmul(PH, np.linalg.inv(W))

    x_k = x_bar + np.matmul(KF, (z - np.matmul(H, x_bar)))
    P_k = np.matmul((np.eye(len(KF)) - KF), P)

    x_bar_c = x_k[0,:]
    P_c = P_k[0,:]
    x_bar_r = x_k[1,:]
    P_r = P_k[1,:]

    return x_bar_c, P_c, x_bar_r, P_r


def get_task_2g(x_bar_c: ndarray, P_c: ndarray,
                x_bar_r: ndarray, P_r: ndarray,
                z_c: ndarray, R_c: ndarray, H_c: ndarray,
                z_r: ndarray, R_r: ndarray, H_r: ndarray):
    """get state estimates after receiving measurement c and measurement r

    Args:
        x_bar_c (ndarray): state estimate after receiving measurement c
        P_c (ndarray): covariance of x_bar_c
        x_bar_r (ndarray): state estimate after receiving measurement r
        P_r (ndarray): covariance of x_bar_r
        z_c (ndarray): measurement c
        R_c (ndarray): covariance of measurement c
        H_c (ndarray): measurement matrix i.e. z_c = H_c @ x + error
        z_r (ndarray): measurement r
        R_r (ndarray): covariance of measurement r
        H_r (ndarray): measurement matrix i.e. z_r = H_r @ x + error

    Returns:
        x_bar_cr (ndarray): state estimate after receiving z_c then z_r
        P_cr (ndarray): covariance of x_bar_cr
        x_bar_rc (ndarray): state estimate after receiving z_r then z_c
        P_rc (ndarray): covariance of x_bar_rc
    """

    # TODO replace this with your own code
    x_bar_cr, P_cr, x_bar_rc, P_rc = solution.task2.get_task_2g(
        x_bar_c, P_c, x_bar_r, P_r, z_c, R_c, H_c, z_r, R_r, H_r)

    return x_bar_cr, P_cr, x_bar_rc, P_rc


def get_task_2h(x_bar_rc: ndarray, P_rc: ndarray):
    """get the probability that the boat is above the line

    Args:
        x_bar_rc (ndarray): state
        P_rc (ndarray): covariance

    Returns:
        prob_above_line: the probability that the boat is above the line
    """

    # TODO replace this with your own code
    prob_above_line = solution.task2.get_task_2h(x_bar_rc, P_rc)

    return prob_above_line
