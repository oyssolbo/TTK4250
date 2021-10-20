import numpy as np
from numpy import ndarray
from typing import Sequence, Optional
from quaternion import RotationQuaterion

from datatypes.measurements import GnssMeasurement
from datatypes.eskf_states import NominalState, ErrorStateGauss
from datatypes.multivargaussian import MultiVarGaussStamped

import solution


def get_NIS(z_gnss: GnssMeasurement,
            z_gnss_pred_gauss: MultiVarGaussStamped,
            marginal_idxs: Optional[Sequence[int]] = None
            ) -> float:
    """
    Calculate NIS

    Args:
        z_gnss (GnssMeasurement): gnss measurement
        z_gnss_pred_gauss (MultiVarGaussStamped): predicted gnss measurement
        marginal_idxs (Optional[Sequence[int]]): Sequence of marginal indexes.
            For example used for calculating NIS in only xy direction.  

    Returns:
        NIS (float): NIS value
    """
    # NIS according to p. 69 is the normalized innovations squared
    # In other words, it is given as 
    # epsilon = (mu)^T S^-1 (mu)

    # Initializing in case of marginalizing
    z_gnss_pred_marg = z_gnss_pred_gauss
    z_gnss_marg = z_gnss.pos

    # Taking the marginalized indeces into account
    if marginal_idxs != None:
        z_gnss_pred_marg = z_gnss_pred_marg.marginalize(marginal_idxs)
        z_gnss_marg = z_gnss_marg[marginal_idxs]
    
    mu = z_gnss_marg - z_gnss_pred_marg.mean
    S = z_gnss_pred_marg.cov

    # Calculating NIS
    NIS = mu.T @ np.linalg.inv(S) @ mu

    # NIS = solution.nis_nees.get_NIS(z_gnss, z_gnss_pred_gauss, marginal_idxs)
    return NIS


def get_error(x_true: NominalState,
              x_nom: NominalState,
              ) -> 'ndarray[15]':
    """
    Finds the error (difference) between True state and 
    nominal state. See (Table 10.1).


    Returns:
        error (ndarray[15]): difference between x_true and x_nom. 
    """
    # Error = True - Nomial
    e_pos = x_true.pos - x_nom.pos
    e_vel = x_true.vel - x_nom.vel
    q_nom_inv = x_nom.ori.conjugate()
    q_nom_inv_norm = RotationQuaterion(q_nom_inv.real_part, q_nom_inv.vec_part) # Guaranteeing that it is normalized
    e_ori = q_nom_inv_norm @ x_true.ori
    e_ori = e_ori.as_euler()                                                    # Converting to euler angles
    e_accm_bias = x_true.accm_bias - x_nom.accm_bias
    e_gyro_bias = x_true.gyro_bias - x_nom.gyro_bias
    
    # Creating error-array
    error = np.concatenate([e_pos, e_vel, e_ori, e_accm_bias, e_gyro_bias])
    
    # error = solution.nis_nees.get_error(x_true, x_nom)
    return error


def get_NEES(error: 'ndarray[15]',
             x_err: ErrorStateGauss,
             marginal_idxs: Optional[Sequence[int]] = None
             ) -> float:
    """
    Calculate NEES

    Args:
        error (ndarray[15]): errors between x_true and x_nom (from get_error)
        x_err (ErrorStateGauss): estimated error
        marginal_idxs (Optional[Sequence[int]]): Sequence of marginal indexes.
            For example used for calculating NEES for only the position. 

    Returns:
        NEES (float): NEES value
    """
    # NEES according to p. 69 is the normalized estimation error squared
    # In other words, it is given as 
    # epsilon = (x_hat - x)^T P^-1 (x_hat - x)

    # Assuming that one should find the NEES given by the errors.
    # This gives that
    # epsilon = (e_hat - e_t)^T P^-1 (e_hat - e_t)

    # Initializing in case of marginalizing
    e_hat = x_err
    e_t = error

    # Possibly marginalizing the indeces
    if marginal_idxs != None:
        e_hat = e_hat.marginalize(marginal_idxs)
        e_t = e_t[marginal_idxs]
    
    e_diff = e_hat.mean - e_t
    P = e_hat.cov 

    # Calculating NEES
    NEES = e_diff.T @ np.linalg.inv(P) @ e_diff

    # NEES = solution.nis_nees.get_NEES(error, x_err, marginal_idxs)
    return NEES


def get_time_pairs(unique_data, data):
    """
    Match data from two different time series based on timestamps
    """
    gt_dict = dict(([x.ts, x] for x in unique_data))
    pairs = [(gt_dict[x.ts], x) for x in data if x.ts in gt_dict]
    times = [pair[0].ts for pair in pairs]
    return times, pairs
