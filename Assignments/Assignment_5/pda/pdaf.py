import numpy as np
from numpy import array, ndarray, zeros
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
        gated_measurements = np.empty_like([measurements[0]])
        is_initialized = False

        for m in measurements:
            if z_pred_gauss.mahalanobis_distance_sq(m) > self.gate_size_sq:
                continue
            if is_initialized:
                gated_measurements = np.append(gated_measurements, np.array(m))
                continue     
            gated_measurements[:] = np.array(m)
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
        V = 1 / self.clutter_density   

        m = len(gated_measurements)
        associations_probs = np.zeros((m,1))
        associations_probs = np.append(associations_probs, np.zeros((1,1)))

        # Calculating for i == 0
        associations_probs[0] = m/V * (1 - P_D)

        # Calculating for i > 0
        for i in range(0, m):  
            I_i = z_pred_gauss.pdf(gated_measurements[i])
            associations_probs[i-1] = P_D*I_i

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
        # Allocating memory
        n = len(gated_measurements)
        update_gaussians = np.arange(n+1, dtype=MultiVarGaussian)

        # Get the associated probabilities
        associated_probs = self.get_association_prob(z_pred_gauss, gated_measurements)

        # Iterate over the associated probabilities and calculate the posteriori state given
        # that the associated measurement is correct 
        for i in range(len(associated_probs)):
            gaussian_mixture_class = GaussianMuxture(
                    weights=np.array([associated_probs[i]]), 
                    gaussians=np.array([state_pred_gauss]))
            update_gaussians[i] = gaussian_mixture_class.reduce()

        # Something is incorrect with this code, since it will not reduce properly
        # for all 

        update_gaussians = solution.pdaf.PDAF.get_cond_update_gaussians(
            self, state_pred_gauss, z_pred_gauss, gated_measurements)
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
        gated_measurements = self.gate(
                z_pred_gauss=z_pred_gauss, 
                measurements=measurements)

        # Get the associated probabilities
        associated_probs = self.get_association_prob(
                z_pred_gauss=z_pred_gauss,
                gated_measurements=gated_measurements)

        # Get the conditional associated states
        cond_states = self.get_cond_update_gaussians(
                state_pred_gauss=state_pred_gauss, 
                z_pred_gauss=z_pred_gauss,
                gated_measurements=gated_measurements)

        gaussian_mixture_class = GaussianMuxture(
                weights=associated_probs, 
                gaussians=cond_states)
        state_upd_gauss = gaussian_mixture_class.reduce()
        
        # state_upd_gauss is here just an array of float64
        # and not a gaussian... Just fuck python! C++ all the way (except for plotting)

        state_upd_gauss = solution.pdaf.PDAF.update(
            self, state_pred_gauss, z_pred_gauss, measurements)
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
