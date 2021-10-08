# %% Imports
from dynamicmodels import WhitenoiseAcceleration2D
from typing import Any, Dict
from dataclasses import dataclass
import numpy as np
from numpy import ndarray

# import solution
# %% Measurement models interface declaration


@dataclass
class MeasurementModel:
    def h(self, x: ndarray, **kwargs) -> ndarray:
        """
        Calculate the noise free measurement location at x in sensor_state.
        Args:
            x (ndarray): state
        """
        raise NotImplementedError

    def H(self, x: ndarray, **kwargs) -> ndarray:
        """
        Calculate the measurement Jacobian matrix at x in sensor_state.
        Args:
            x (ndarray): state
        """
        raise NotImplementedError

    def R(self, x: ndarray, **kwargs) -> ndarray:
        """
        Calculate the measurement covariance matrix at x in sensor_state.
        Args:
            x (ndarray): state
        """
        raise NotImplementedError


@dataclass
class CartesianPosition2D(MeasurementModel):
    sigma_z: float

    def h(self, x: ndarray) -> ndarray:
        """
        Calculate the noise free measurement location at x in sensor_state.
        """
        
        # H = [eye(2), 0 * eye(2)]
        H = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ]
        )
        x_h = H @ x        
        # x_h = solution.measurementmodels.CartesianPosition2D.h(self, x)
        return x_h

    def H(self, x: ndarray) -> ndarray:
        """
        Calculate the measurement Jacobian matrix at x in sensor_state.
        """

        H = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ]
        )
        #H = solution.measurementmodels.CartesianPosition2D.H(self, x)
        return H

    def R(self, x: ndarray) -> ndarray:
        """
        Calculate the measurement covariance matrix at x in sensor_state.
        """
        # R = zigma_r * eye(2)
        R = np.array(
            [
                [1, 0], 
                [0, 1]
            ]
        ) * self.sigma_z**2
        #R = solution.measurementmodels.CartesianPosition2D.R(self, x)
        return R



































































