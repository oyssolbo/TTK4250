#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic models to be used with eg. EKF.

@author: Lars-Christian Tokle, lars-christian.n.tokle@ntnu.no
"""
# %%
from dataclasses import dataclass

import numpy as np
from numpy import ndarray
import solution


@dataclass
class DynamicModel:
    """
    Parent class for dynamic models.

    A model includes the discrete prediction equation f, its Jacobian F, and
    the process noise covariance Q.
    """

    def f(self, x: ndarray, Ts: float, **kwargs) -> ndarray:
        """
        Calculate the zero noise Ts time units transition from x.

          Args:
              x (ndarray): state
              Ts (float): time step

          Returns:
              x_kp1 (ndarray): x_k+1, the next state
          """
        raise NotImplementedError

    def F(self, x: ndarray, Ts: float, **kwargs) -> ndarray:
        """
        Calculate the transition function jacobian for Ts time units at x.
        Args:
            x (ndarray): state
            Ts (float): time step

        Returns:
            F (ndarray): Discrete ransition function jacobian,
                         for linear systems: x_k+1 = F @ x_k
        """
        raise NotImplementedError

    def Q(self, x: ndarray, Ts: float, **kwargs) -> ndarray:
        """
        Calculate the Ts time units transition Covariance.
        Args:
            x (ndarray): state
            Ts (float): time step

        Returns:
            Q (ndarray): covariance matrix
        """
        raise NotImplementedError


@dataclass
class WhitenoiseAcceleration2D(DynamicModel):
    """
    A white noise acceleration model, also known as constan velocity.
    States are position and speed.
    """

    # noise standard deviation
    sigma_a: float

    def f(self, x: ndarray, Ts: float,) -> ndarray:
        """
        Calculate the zero noise Ts time units transition from x.
        See DynamicModel for variable documentation
        """
        F = np.array([[1, 0, Ts, 0], 
                    [0, 1, 0, Ts],
                    [0, 0, 1, 0], 
                    [0, 0, 0, 1]]
                )
        Q = np.array([[1/3 * Ts**3, 0, 1/2 * Ts**2, 0], 
                    [0, 1/3 * Ts**3, 0, 1/2 * Ts**2],
                    [1/2 * Ts**2, 0, Ts, 0], 
                    [0, 1/2 * Ts**2, 0, Ts]]
                ) * self.sigma_a**2
        
        mean = np.array([0, 0, 0, 0])

        v = np.random.multivariate_normal(mean=mean, cov=Q)
        
        x_kp1 = F * x + v

        if x_kp1 != solution.dynamicmodels.WhitenoiseAcceleration2D.f(self, x, Ts):
            print("Incorrect solution for x_kp1")
            x_kp1 = solution.dynamicmodels.WhitenoiseAcceleration2D.f(self, x, Ts)
        return x_kp1

    def F(self, x: ndarray, Ts: float,) -> ndarray:
        """
        Calculate the transition function jacobian for Ts time units at x.
        See DynamicModel for variable documentation
        """

        F = np.array([[1, 0, Ts, 0], 
                    [0, 1, 0, Ts],
                    [0, 0, 1, 0], 
                    [0, 0, 0, 1]]
                )

        if F.T != solution.dynamicmodels.WhitenoiseAcceleration2D.F(self, x, Ts):
            print("Incorrect solution of F")
            F.T = solution.dynamicmodels.WhitenoiseAcceleration2D.F(self, x, Ts)
        return F.T

    def Q(self, x: ndarray, Ts: float,) -> ndarray:
        """
        Calculate the Ts time units transition Covariance.
        See(4.64) in the book.
        See DynamicModel for variable documentation
        """
        # From 4.64 we know that 
        Q = np.array([[1/3 * Ts**3, 0, 1/2 * Ts**2, 0], 
                    [0, 1/3 * Ts**3, 0, 1/2 * Ts**2],
                    [1/2 * Ts**2, 0, Ts, 0], 
                    [0, 1/2 * Ts**2, 0, Ts]]
                ) * self.sigma_a**2
        
        if Q != solution.dynamicmodels.WhitenoiseAcceleration2D.Q(self, x, Ts):
            print("Incorrect solution for Q")
            Q = solution.dynamicmodels.WhitenoiseAcceleration2D.Q(self, x, Ts)

        return Q



















































































