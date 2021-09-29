import numpy as np
from numpy import ndarray, zeros
from numpy.core.numeric import zeros_like
from scipy.stats import chi2
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

from gaussmix import GaussianMuxture
from utils.multivargaussian import MultiVarGaussian
from utils.ekf import EKF

import solution

from math import sqrt


@dataclass
class PDAF:

    ekf: EKF
    clutter_density: float
    detection_prob: float
    gate_percentile: float
    gate_size_sq: float = field(init=False)

    def __post_init__(self):
        self.gate_size_sq = chi2.ppf(self.gate_percentile,
                                     self.ekf.sensor_model.ndim)

    def predict_state(
                    self, 
                    state_upd_prev_gauss: MultiVarGaussian, 
                    Ts: float
                    ) -> MultiVarGaussian:
        """
        Prediction step
        Hint: use self.ekf

        Args:
            state_upd_prev_gauss (MultiVarGaussian): previous update gaussian
            Ts (float): timestep

        Returns:
            state_pred_gauss (MultiVarGaussian): predicted state gaussian
        """
        state_pred_gauss = self.ekf.predict_state(
                state_upd_prev_gauss=state_upd_prev_gauss,
                Ts=Ts)
        
        # state_pred_gauss = solution.pdaf.PDAF.predict_state(
            #     self, state_upd_prev_gauss, Ts)
        return state_pred_gauss

    def predict_measurement(
                            self, 
                            state_pred_gauss: MultiVarGaussian
                            ) -> MultiVarGaussian:
        """
        Measurement prediction step
        Hint: use self.ekf

        Args:
            state_pred_gauss (MultiVarGaussian): predicted state gaussian

        Returns:
            z_pred_gauss (MultiVarGaussian): predicted measurement gaussian
        """

        z_pred_gauss = self.ekf.predict_measurement(state_pred_gauss=state_pred_gauss)

        # z_pred_gauss = solution.pdaf.PDAF.predict_measurement(
        #     self, state_pred_gauss)
        return z_pred_gauss

    def gate(
            self,
            z_pred_gauss: MultiVarGaussian,
            measurements: Sequence[ndarray]
            ) -> ndarray:
        """
        Gate the incoming measurements. That is remove the measurements 
        that have a mahalanobis distance higher than a certain threshold. 

        Hint: use z_pred_gauss.mahalanobis_distance_sq and self.gate_size_sq

        Args:
            z_pred_gauss (MultiVarGaussian): predicted measurement gaussian 
            measurements (Sequence[ndarray]): sequence of measurements

        Returns:
            gated_measurements (ndarray[:,2]): array of accepted measurements
        """
        gated_measurements = np.full_like(measurements[0], fill_value=None)
        is_initialized = False

        for m in measurements:
            if z_pred_gauss.mahalanobis_distance_sq(m) > self.gate_size_sq:
                continue
            if is_initialized:
                np.append(gated_measurements, m)
                continue     
            gated_measurements[:] = m
            is_initialized = True

        # gated_measurements = solution.pdaf.PDAF.gate(
        #     self, z_pred_gauss, measurements)
        return gated_measurements

    def get_association_prob(
                            self, 
                            z_pred_gauss: MultiVarGaussian,
                            gated_measurements: ndarray
                            ) -> ndarray:
        """
        Finds the association probabilities.

        associations_probs[0]: prob that no association is correct
        associations_probs[1]: prob that gated_measurements[0] is correct
        associations_probs[2]: prob that gated_measurements[1] is correct
        ...

        the sum of associations_probs should be 1

        Args:
            z_pred_gauss (MultiVarGaussian): predicted measurement gaussian 
            gated_measurements (ndarray[:,2]): array of accepted measurements

        Returns:
            associations_probs (ndarray[:]): the association probabilities
        """
        P_D = self.detection_prob
        P_G = self.gate_percentile
        V = 1 / self.clutter_density   

        m = len(gated_measurements)
        shape = np.shape(gated_measurements)
        associations_probs = zeros_like(gated_measurements)
        np.append(associations_probs, zeros(shape))

        # Calculating for i == 0
        # associations_probs[0] = (1-P_D)*m*poi_m / poi_m_1
        associations_probs[0] = m/V * (1 - P_D*P_G)

        # Calculating for i > 0
        for i in range(1, m):  
            mah_dist = sqrt(z_pred_gauss.mahalanobis_distance_sq(gated_measurements[i]))
            I_i = 1 - mah_dist / (1 + mah_dist) 
            associations_probs[i] = P_G*P_D*I_i

        if associations_probs.sum() == 0:
            # Unable to normalize 
            assert 0
        associations_probs /= associations_probs.sum()

        # associations_probs = solution.pdaf.PDAF.get_association_prob(
        #     self, z_pred_gauss, gated_measurements)
        return associations_probs

    def get_cond_update_gaussians(
                            self, 
                            state_pred_gauss: MultiVarGaussian,
                            z_pred_gauss: MultiVarGaussian,
                            gated_measurements: ndarray
                            ) -> Sequence[MultiVarGaussian]:
        """
        Get the conditional updated state gaussians 
        for every association hypothesis

        update_gaussians[0]: update given that no measurement is correct
        update_gaussians[1]: update given that gated_measurements[0] is correct
        update_gaussians[2]: update given that gated_measurements[1] is correct
        ...


        Args:
            state_pred_gauss (MultiVarGaussian): predicted state gaussian
            z_pred_gauss (MultiVarGaussian): predicted measurement gaussian
            gated_measurements (ndarray[:,2]): array of accepted measurements

        Returns:
            Sequence[MultiVarGaussian]: The sequence of conditional updates
        """
        # # Since the gated measurements have been truncated, this must also
        # # be done to f_z to maintain scaling
        # P_G = self.gate_percentile

        # # Extract previous states
        # state_pred_mean = state_pred_gauss.mean     # x_hat_k|k-1
        # state_pred_cov = state_pred_gauss.cov       # P_k|k-1

        # z_pred_mean = z_pred_gauss.mean / P_G       # z_hat_k|k-1 = Hx_hat_k|k-1
        # z_pred_cov = z_pred_gauss.cov / (P_G**2)    # S_k = HP_k|k-1H^T + R

        # H = z_pred_mean @ state_pred_mean.T

        # # Calculate the kalman gain (assuming that the same one will be used for 
        # # all measurements)
        # W = state_pred_cov @ H.T @ np.linalg.inv(z_pred_cov)

        # Allocating memory
        n = len(gated_measurements)
        update_gaussians = np.arange(n+1, dtype=MultiVarGaussian)

        # Get the associated probabilities
        associated_probs = self.get_association_prob(z_pred_gauss, gated_measurements)

        # Iterate over the associated probabilities and calculate the posteriori state given
        # that the associated measurement is correct 
        for i in range(len(associated_probs)):
            update_gaussians[i] = associated_probs[i] * state_pred_gauss

        # update_gaussians = solution.pdaf.PDAF.get_cond_update_gaussians(
        #     self, state_pred_gauss, z_pred_gauss, gated_measurements)
        return update_gaussians

    def update(
            self, 
            state_pred_gauss: MultiVarGaussian,
            z_pred_gauss: MultiVarGaussian,
            measurements: Sequence[ndarray]
            ) ->MultiVarGaussian:
        """
        Perform the update step of the PDA filter

        Args:
            state_pred_gauss (MultiVarGaussian): predicted state gaussian
            z_pred_gauss (MultiVarGaussian): predicted measurement gaussian
            measurements (Sequence[ndarray]): sequence of measurements

        Returns:
            state_upd_gauss (MultiVarGaussian): updated state gaussian
        """
        # Gate the measurements
        gated_measurements = self.gate(z_pred_gauss=z_pred_gauss, measurements=measurements)

        # Get the conditional associated states
        cond_states = self.get_cond_update_gaussians(
                state_pred_gauss=state_pred_gauss, 
                z_pred_gauss=z_pred_gauss,
                gated_measurements=gated_measurements)

        gaussian_mixture_class = GaussianMuxture(weights=gated_measurements, gaussians=cond_states)
        state_upd_gauss = gaussian_mixture_class.reduce()
        
        # state_upd_gauss = solution.pdaf.PDAF.update(
        #     self, state_pred_gauss, z_pred_gauss, measurements)
        return state_upd_gauss

    def step_with_info(
                    self,
                    state_upd_prev_gauss: MultiVarGaussian,
                    measurements: Sequence[ndarray],
                    Ts: float
                    ) -> Tuple[MultiVarGaussian,
                                MultiVarGaussian,
                                MultiVarGaussian]:
        """
        Perform a full step and return usefull info

        Hint: you should not need to write any new code here, 
        just use the methods you have implemented

        Args:
            state_upd_prev_gauss (MultiVarGaussian): previous updated gaussian
            measurements (Sequence[ndarray]): sequence of measurements
            Ts (float): timestep

        Returns:
            state_pred_gauss (MultiVarGaussian): predicted state gaussian
            z_pred_gauss (MultiVarGaussian): predicted measurement gaussian
            state_upd_gauss (MultiVarGaussian): updated state gaussian
        """

        state_pred_gauss = self.predict_state(state_upd_prev_gauss=state_upd_prev_gauss, Ts=Ts)
        z_pred_gauss = self.predict_measurement(state_pred_gauss=state_pred_gauss)
        state_upd_gauss = self.update(state_pred_gauss=state_pred_gauss, z_pred_gauss=z_pred_gauss, measurements=measurements)

        # state_pred_gauss, z_pred_gauss, state_upd_gauss = solution.pdaf.PDAF.step_with_info(
        #     self, state_upd_prev_gauss, measurements, Ts)
        return state_pred_gauss, z_pred_gauss, state_upd_gauss

    def step(self, state_upd_prev_gauss, measurements, Ts):
        _, _, state_upd_gauss = self.step_with_info(state_upd_prev_gauss,
                                                    measurements,
                                                    Ts)
        return state_upd_gauss
