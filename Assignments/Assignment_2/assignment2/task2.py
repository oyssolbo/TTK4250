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

    return x + np.matmul(W, z_HX)


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
    x_bar_c = condition_mean(x_bar, P, z_c, R_c, H_c)
    P_c = condition_cov(P, R_c, H_c)

    x_bar_r = condition_mean(x_bar, P, z_r, R_r, H_r)
    P_r = condition_cov(P, R_r, H_r)

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
    # In theory, these values should be identical
    x_bar_cr = condition_mean(x_bar_c, P_c, z_r, R_r, H_r)
    P_cr = condition_cov(P_c, R_r, H_r)

    x_bar_rc = condition_mean(x_bar_r, P_r, z_c, R_c, H_c)
    P_rc = condition_cov(P_r, R_c, H_c)

    return x_bar_cr, P_cr, x_bar_rc, P_rc


def get_task_2h(x_bar_rc: ndarray, P_rc: ndarray):
    """get the probability that the boat is above the line

    Args:
        x_bar_rc (ndarray): state
        P_rc (ndarray): covariance

    Returns:
        prob_above_line: the probability that the boat is above the line
    """

    """
    Explenation on how I thought about the process:
    We have a transformation where we would like to check
    x2 - x1 > 5

    which is equivalent to 

    [-1 1] x > 5

    Could just check to find a Y that satisfies
    Y = [-1 1] X

    which due to X being a gaussian gives that Y is a gaussian satisfying
    Y ~ N(y; [-1, 1] E[x_hat], [-1, 1] Cov[x_hat] [-1, 1]^T)

    However I cannot find a way to 
    """
    # TODO replace this with your own code
    prob_above_line = solution.task2.get_task_2h(x_bar_rc, P_rc)

    return prob_above_line


"""
Comments for task 2f)
For task 2f), the mean and the covariance of the camera and the radar is the 
most important. When studying the results from the camera, we can clearly see 
that the new measurement "moves" the estimated position and its covariance to
a place that is plausible for both the predicted and the measured state. The
latter observation is to see that the covariance of the new estimate
includes part of the covcariance from the camera and the original estimate.
This means that it will not assume that the new data is totally correct, however 
it will use the new data in combination with the old data/estimate to get a new
and hopefully better system estimate. 

A similar observation could be made for the radar, where an estimate is closer
to the initial prediciton. That means that the new estimate is inside of the
covariance for the original estimate and the covariance of the radar measurment. 



Comments for task 2g)

For task 2g), you can see that the estimate for x|z_rc is equivalent to the 
estimate for x|z_cr. This means that the order of the data has no effect on the
end result, such that the data could be fed into a KF (or other bayesian filter)
when it is available. As long as you either has a time-stamp of the data or can
guarantee that it is not too old to show an outdated system state, it could be
used directly in the filter.





Comments for task 2h)





Comments for task 2b)
After reading through the task more thoroughly, it is clear that we expect the
estimate to be given as a gaussian on the form N(x; x_bar, P)

That means that I at least thought correctly, even though I did think a bit
too complicated with the Brownian movement etc.


"""


