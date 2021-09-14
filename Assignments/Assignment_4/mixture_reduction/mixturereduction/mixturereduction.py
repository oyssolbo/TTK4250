import numpy as np
from numpy import ndarray

import solution


def mixture_moments(weights: ndarray,
                    means: ndarray,
                    covs: ndarray,
                    ) -> tuple[ndarray, ndarray]:
    """
    Calculate the first two moments of a Gaussian mixture.

    Args:
        weights: shape = (N,)
        means: shape = (N, n)
        covs: shape = (N, n, n)

    Returns:
        mean: shape = (n,)
        cov: shape = (n, n)
    """
    mean = weights.T @ means

    # Internal covariance
    # I assume that we are talking about \Sigma_{i=1}^{M} w_i P_i
    # cov_internal = weights.T @ covs
    # cov_internal = np.zeros(len(weights))
    # for i in range(len(weights)):
    #     cov_internal += weights[i] * covs[i]
    cov_internal = np.average(covs, axis=0, weights=weights)

    # Spread of means, aka. external covariance
    # [diffs] = 1 x N
    # [cov_external] = 1 x N - n x n 
    # diffs = weights.T @ means @ means
    diffs = means - mean[np.newaxis]
    # cov_external = diffs - mean.T @ mean
    cov_external = np.average(diffs[:, :, np.newaxis] * diffs[:, np.newaxis, :], axis=0, weights=weights)

    # total covariance
    cov = cov_internal + cov_external

    # mean, cov = solution.mixturereduction.mixture_moments(weights, means, covs)
    return mean.T, cov
