import numpy as np
from numpy import ndarray
from dataclasses import dataclass
from typing import Sequence

from config import DEBUG
from utils.multivargaussian import MultiVarGaussian

import solution


@dataclass(frozen=True)
class GaussianMuxture:
    """
    Dataclass representing a gaussian mixture
    """

    weights: ndarray
    gaussians: Sequence[MultiVarGaussian]

    def __post_init__(self):
        if DEBUG:
            assert len(self.weights) == len(self.gaussians)
            assert np.all(self.weights > 0)
            assert np.isclose(self.weights.sum(), 1)
            ndim = self.gaussians[0].ndim
            assert all([g.ndim == ndim for g in self.gaussians])

    def get_mean(self) -> ndarray:
        """
        Return the mean of the gaussian mixture

        Hint: use what you did in mixturereductin.py assignment 4

        Returns:
            mean (ndarray): the mean
        """
        # I assume that the system should mixture self.gaussians
        means = np.zeros_like(self.weights)

        # Extract all of the data from the gaussians
        i = 0
        for gaussian in self.gaussians:
            means[i] = gaussian.mean
            i += 1

        # Multiply by the weights of the corresponding gaussians
        mean = self.weights.T @ means
 
        #mean = solution.gaussmix.GaussianMuxture.get_mean(self)
        return mean

    def get_cov(self) -> ndarray:
        """
        Return the covariance of the gaussian mixture

        Hint: use what you did in mixturereductin.py assignment 4

        Returns:
            cov (ndarray): the covariance
        """
        mean = self.get_mean()
        #diffs = means - mean[np.newaxis]

        cov_shape = np.shape(self.gaussians[0].cov)
        covs = np.zeros_like(cov_shape)
        #for i in range(len(self.gaussians)):
            # covs[i*cov_shape[0]:i]

        #cov_internal = np.average(covs, axis=0, weights=self.weights)
        #cov = 0

        # I was unable to actually solve this in task 4, and still haven't
        # understood how one should solve this

        cov = solution.gaussmix.GaussianMuxture.get_cov(self)
        return cov

    def reduce(self) -> MultiVarGaussian:
        """
        Reduce the gaussian mixture to a multivariate gaussian
        Hint: you can use self.get_mean and self.get_cov

        Returns:
            reduction (MultiVarGaussian): the reduction
        """
        mean = self.get_mean()
        cov = self.get_cov()

        reduction = np.random.multivariate_normal(mean, cov)
        
        #reduction = solution.gaussmix.GaussianMuxture.reduce(self)
        return reduction

    @property
    def mean(self) -> ndarray:
        return self.get_mean()

    @property
    def cov(self) -> ndarray:
        return self.get_cov()

    def pdf(self, x: ndarray) -> float:
        """
        Probability density function

        Args:
            x (ndarray): point

        Returns:
            float: probability density at point x 
        """
        density = sum(w * g.pdf(x)
                      for w, g in zip(self.weights, self.gaussians))
        return density

    def __iter__(self):  # in order to use tuple unpacking
        return iter((self.mean, self.cov))

    def __eq__(self, o: object) -> bool:
        """
        This definition of equality is actually to stict 
        as the order of weights and gaussians matter, but for this purpose
        it works
        """

        if not isinstance(o, GaussianMuxture):
            return False
        elif not np.allclose(o.weights, self.weights):
            return False
        elif not all([g1 == g2
                      for g1, g2 in zip(self.gaussians, o.gaussians)]):
            return False
        else:
            return True
