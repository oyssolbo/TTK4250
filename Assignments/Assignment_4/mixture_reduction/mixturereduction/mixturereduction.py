import numpy as np
from numpy import ndarray
from numpy.core.numeric import zeros_like

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
    cov_internal = np.average(covs, axis=0, weights=weights)

    # External covariance
    # Assuming that we are talking about P_tilde
    mean_diff = np.array(means - mean[np.newaxis])

    # I couldn't quite figure out how to find the cov_external
    # cov_external = np.average(mean_diff @ mean_diff.T, axis=0, weights=weights) - mean @ mean.T
    cov_external = 0

    # Total covariance
    cov = cov_internal + cov_external

    mean, cov = solution.mixturereduction.mixture_moments(weights, means, covs)
    return mean.T, cov
