# %% Imports
from typing import Collection
import scipy
from scipy import stats
from matplotlib import pyplot as plt
import numpy as np
import numpy.linalg as nla
from numpy import ndarray
from tqdm import tqdm

from utils.dataloader import load_data
from utils.plotting import (plot_trajectory_with_measurements,
                            plot_ekf_trajectory,
                            plot_NIS_NEES_data)

from ekf import EKF
from utils.gaussparams import MultiVarGaussian
from measurementmodels import CartesianPosition2D
from dynamicmodels import WhitenoiseAcceleration2D
from analysis import get_ANIS, get_ANEES


def run_ekf(sigma_a: float, sigma_z: float,
            z_data: Collection[ndarray], Ts: float, N_data: int):
    """
    This function will estimate the initial state and covariance from
    the measurements and iterate the kalman filter through the data.

    Args:
        sigma_a (float): std of acceleration
        sigma_z (float): std of measurements
        
        Ts (float): the time step between each measurement
        N_data (int): the number of measurements

    Returns:
        state_pred_gauss_data (list[MultiVarGaussian]):
            list of all state predictions
        measurement_gauss_data (list[MultiVarGaussian]):
            list of all measurement pdfs
        state_upd_gauss_data (list[MultiVarGaussian]):
            list of all updated states
    """
    # create the model and estimator object
    dynmod = WhitenoiseAcceleration2D(sigma_a)
    measmod = CartesianPosition2D(sigma_z)
    ekf_filter = EKF(dynmod, measmod)

    # Optimal init for model
    mean = np.array([*z_data[1], *(z_data[1] - z_data[0]) / Ts])
    cov11 = sigma_z ** 2 * np.eye(2)
    cov12 = sigma_z ** 2 * np.eye(2) / Ts
    cov22 = (2 * sigma_z ** 2 / Ts ** 2 + sigma_a ** 2 * Ts / 3) * np.eye(2)
    cov = np.block([[cov11, cov12], [cov12.T, cov22]])
    init_ekfstate = MultiVarGaussian(mean, cov)

    # estimate
    x_upd_gauss = init_ekfstate
    x_pred_gauss_data = []
    z_pred_gauss_data = []
    x_upd_gauss_data = []
    NIS_data = []
    for z_k in z_data[2:]:

        (x_pred_gauss,
         z_pred_gauss,
         x_upd_gauss) = ekf_filter.step_with_info(x_upd_gauss, z_k, Ts)

        x_pred_gauss_data.append(x_pred_gauss)
        z_pred_gauss_data.append(z_pred_gauss)
        x_upd_gauss_data.append(x_upd_gauss)

    return x_pred_gauss_data, z_pred_gauss_data, x_upd_gauss_data


def show_ekf_output(sigma_a: float, sigma_z: float,
                    x_gt_data: Collection[ndarray],
                    z_data: Collection[ndarray], Ts: float, N_data: int):
    """
    Run the Kalman filter, find RMSE and show the trajectory
    """

    (x_pred_gauss,
     z_pred_gauss,
     x_upd_gauss) = run_ekf(sigma_a, sigma_z, z_data, Ts, N_data)

    x_hat_data = np.array([upd.mean[:2] for upd in x_upd_gauss])

    diff_pred_data = np.array([pred.mean - x_gt[:4] for pred, x_gt
                               in zip(x_pred_gauss, x_gt_data)])

    diff_upd_data = np.array([upd.mean - x_gt[:4]for upd, x_gt
                              in zip(x_upd_gauss, x_gt_data)])

    RMSE_pred = np.sqrt(
        np.mean(np.sum(diff_pred_data.reshape(-1, 2, 2)**2, axis=-1), axis=0))
    RMSE_upd = np.sqrt(
        np.mean(np.sum(diff_upd_data.reshape(-1, 2, 2)**2, axis=-1), axis=0))

    plot_ekf_trajectory(x_gt_data, x_hat_data,
                        RMSE_pred, RMSE_upd, sigma_a, sigma_z)
# %% Task 5 b and c


def try_multiple_alphas(x_gt_data: Collection[ndarray],
                        z_data: Collection[ndarray],
                        Ts: float, N_data: int):
    """
    Run the Kalman filter with multiple different sigma values,
    the result from each run is used to create a mesh plot of the NIS and NEES
    values for the different configurations
    """
    # % parameters for the parameter grid
    n_vals = 20
    sigma_a_low = 0.5
    sigma_a_high = 10
    sigma_z_low = 0.3
    sigma_z_high = 12

    # % set the grid on logscale(not mandatory)
    sigma_a_list = np.geomspace(sigma_a_low, sigma_a_high, n_vals)
    sigma_z_list = np.geomspace(sigma_z_low, sigma_z_high, n_vals)

    ANIS_data = np.empty((n_vals, n_vals))
    ANEES_pred_data = np.empty((n_vals, n_vals))
    ANEES_upd_data = np.empty((n_vals, n_vals))

    # tqdm is used to show progress bars
    for i, sigma_a in tqdm(enumerate(sigma_a_list), "sigma_a", n_vals, None):
        for j, sigma_z in tqdm(enumerate(sigma_z_list),
                               "sigma_z", n_vals, None):

            (x_pred_gauss_data,
             z_pred_gauss_data,
             x_upd_gauss_data) = run_ekf(sigma_a, sigma_z, z_data,
                                         Ts, N_data)

            # dont use the first 2 values of x_gt_data or a_data
            # as they are used for initialzation

            ANIS_data[i, j] = get_ANIS(z_pred_gauss_data, z_data[2:])

            ANEES_pred_data[i, j] = get_ANEES(x_pred_gauss_data,
                                              x_gt_data[2:, :4])

            ANEES_upd_data[i, j] = get_ANEES(x_upd_gauss_data,
                                             x_gt_data[2:, :4])

    confprob = 0.9
    CINIS = np.array(stats.chi2.interval(confprob, 2 * N_data)) / N_data
    CINEES = np.array(stats.chi2.interval(confprob, 4 * N_data)) / N_data
    plot_NIS_NEES_data(sigma_a_low, sigma_a_high, sigma_a_list,
                       sigma_z_low, sigma_z_high, sigma_z_list,
                       ANIS_data, CINIS,
                       ANEES_pred_data, ANEES_upd_data, CINEES)


if __name__ == '__main__':
    usePregen = False  # choose between own generated data and pregenerated
    x_gt_data, z_data, Ts, N_data = load_data(usePregen)
    plot_trajectory_with_measurements(x_gt_data, z_data)

    # set parameters
    sigma_a = 2.6
    sigma_z = 2.3

    show_ekf_output(sigma_a, sigma_z, x_gt_data, z_data, Ts, N_data)

    # print("Trying multiple alpha combos")
    # try_multiple_alphas(x_gt_data, z_data, Ts, N_data)

    # if input("Try multiple alpha combos? (y/n): ") == 'y':
    #     try_multiple_alphas(x_gt_data, z_data, Ts, N_data)

    plt.tight_layout()
    plt.show()

# %% Comments for task 5a)
    """
    It is known from the task that 
    sigma_a = 0.25
    sigma_w = pi / 15 ~= 0.0439

    where sigma_a controls the Q-matrix
    and sigma_z controls the R-matrix

    Default parameters:
    sigma_a = 2.6
    sigma_z = 3.1

    gives RMSE = [3.08, 3.72, 2.9, 3.12]

    Parameters tried and their corresponding RMSE:
        1. 
            sigma_a = 0.5
            sigma_z = 3.1

            RMSE = [5.33, 6.38, 4.81, 5.95]

            From the plots, we could see that the estimated position is following
            the ground truth to a certain degree, however it is overconfident. It doesn't
            trust the measurements enough, such that it deviates from the path 

        2. 
            sigma_a = 5
            sigma_z = 3.1

            RMSE = [3.23, 3.77, 3.03, 3.27]

            The estimated position is closer to the ground truth, howver one could see some
            oscillations around the ground truth. This is due to the EKF trusting the 
            measurements more than is required. Thus one should predict that by increasing the
            sigma_a variable even more, the EKF will follow/trust each measurement as the whole
            truth. For a low std sigma_z, this is not really a problem, due to the measurements
            being close enough to the ground truth. However, if the measurements are further 
            away from the ground truth, this could have a catastrophic event for the EKF

        3. 
            sigma_a = 500
            sigma_z = 3.1

            RMSE = [9.95, 65, 4.74, 65]

            This is only used for testing my theory in 2)

        4. 
            sigma_a = 500
            sigma_z = 31

            RMSE = [4.98, 15.17, 3.91, 15.10]

            Stupid me! Somehow I thought that sigma_z creates the variance of the measurements,
            and did not take into account that:
                -the measurements are prerecorded/premade
                -only the relationship between the Q and R matrix has any efefct on the KF
                    (assuming that the values are within a threshold that does not cause any
                    numerical problems in the computer)
            Therefore, by increasing sigma_z, it had the same effect as for decreasing sigma_a
            Bit ashamed of myself right now. i have been a bad kybber for forgetting such trivial 
            stuff... 

        5.  
            More "serious" tuning from now on

            We can see that the turn-rate never exceeds 0.8
            That means that sigma_a around 0.8 could be a reasonable starting point

            sigma_a = 0.8
            sigma_z = 3.1

            RMSE = [4.07, 5.45, 3.67, 4.96]

            One can see that the estimated trajectory is a bit slow compared to the ground truth.
            This could be resolved by increasing the values for Q or decreasing the values in R.
            By looking at the plot of the measurements compared to the ground truth, it looks like
            most are within 5 m. Thus, I think we safely could reduce sigma_z down to around 2.5-2.6

        6.
            sigma_a = 0.8
            sigma_z = 2.5

            RMSE = [3.69, 5.04, 3.33, 4.51]

            Follows better compared to 5. However, due to the turning, I think increasing the value
            for Q would allow the EKF to perform better. We are trusting a bit too much on the model,
            where we should trust the measurements more

        7.
            sigma_a = 1.6
            sigma_z = 2.5

            RMSE = [3.11, 3.96, 2.90, 3.36]

            Follows the ground truth nicely, however starts becoming a bit "noisy". Trying to increase Q
            such that it will follow the ground truth better

        8.
            sigma_a = 2
            sigma_z = 2.5

            RMSE = [3.08, 3.75, 2.89, 3.15]

            Could try to increase the Q matrix a bit more such that it follows the turn better. Another potential 
            is to reduce the R matrix a tad, such that it will reduce the deviations from the ground truth

        9.
            sigma_a = 2.4
            sigma_z = 2.5

            RMSE = [3.09, 3.65, 2.91, 3.06]

            By just looking at the RMSE values from 8 and 9, the value 0 and 2 was increased, while value
            1 and 3 was decreased. This means that we are getting closer towards the "optimum". Will
            try to decrease R a bit

        10.
            sigma_a = 2.4
            sigma_z = 2.3

            RMSE = [3.10, 3.63, 2.93, 3.04]

            Relatively happy with this tuning. Could pherhaps try to increase Q a bit more

        11.
            sigma_a = 2.6
            sigma_z = 2.3

            RMSE = [3.12, 3.62, 2.94, 3.05]

        I think it is good enough rn.
    """

# %% Comments for task 5b)
    """
    sigma_a = ~2.6, sigma_z ~= 2.3
        ANEES: 
            ANEES_pred = ~1
            ANEES_upd = ~0.6

        ANIS:
            ANIS = ~0.2

    Just by looking at the values for ANEES and ANIS, this indicates that the KF is too
    conservative. It is not the worst that could happen, since an overconfident filter is
    far worse than a conservative filter

    The tuning I did in a) was by comparing the estimated response to the ground truth. If
    I only had the ANEES and ANIS to adjust the values, I had not chosen the parameters that
    I chose in a). 
    
    To get a filter that is more confident / less conservative, it means to trust the model
    more compared to the measurements. That means to either increase the covariance-matrix R 
    (sigma_z) or decrease the covariance-matrix Q (sigma_a)

    A CT model is a model where the system experiences a coordinate turn. Assuming that air
    vocabulary is used, this means that the turn has a constant radius R and angular velocity
    \omega such that the centripetal force is equal to the horizontal lift-force. Similar
    for other movement, where we expect a constant turn.

    For both NIS and NEES, the estimates would require a more conservative KF to be able to
    follow the turn / angular velocity. This is due to the CV system assumes a more linear
    movement, while the CT assumes a constant turn. Thus, using a filter that relies more on
    the measurements is required to reduce the error during CT. However, the results will be
    worse during CV, since the filter has already been tuned to allow a larger change in the
    angular velocity

    NIS is useful when the ground truth is not available. It describes the Mahalanobis distance
    between the measurements and the predicted measurements. By using NIS, I assume that we
    compare our trust in the model with the measurements. However due to the model assuming CV,
    the estimates will be slightly off. Thus, the NIS will always be lagging a bit behind, 
    unless we make the filter overlyt conservative. It would make the estimates better during
    turning, but worse during linear movement.
    
    Off this reason, I think it would be better to use NEES if we have a "ground truth" to
    compare with. This allows us to be more independent from the model and modelling errors,
    such that we could tune based on the actual response. It will be more similar to the 
    tuning performed in a) where we compare the estimated state to the actual state. Note
    that by "ground truth", I do not mean the actual true state. For example you could tune
    the KF for a radar or a camera, where GNSS-position of the ships are used as the ground 
    truth. The GNSS has some variance itself, but will generally be accurate enough for us 
    to use it as the "true" value
    """

# %% Comments for task 5c)
    """
    Assuming that the new data is from the same population as before, one should expect similar
    results. if it is from another population, the tuning will likely be way off. This is due to
    that the tuning of the EKF assumes the noise of the process and the measurements. By changing
    either the system or the measurments, the covariances will be different  

    Even without tuning the system after the results in b), the predicted response follows the
    curve relatively well, except right before the jump in measurements. We can clearly see that
    the filter does not rely enough on the measurments (as indicated in b)). Thus it is a bit 
    overfitted to the data in a), however that was expected. By relying more on the measurements,
    the estimated states would be better
    """

