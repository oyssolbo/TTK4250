import numpy as np
from numpy import ndarray
import scipy
from dataclasses import dataclass, field
from typing import Tuple
from functools import cache

from datatypes.multivargaussian import MultiVarGaussStamped
from datatypes.measurements import (ImuMeasurement,
                                    CorrectedImuMeasurement,
                                    GnssMeasurement)
from datatypes.eskf_states import NominalState, ErrorStateGauss
from utils.indexing import block_3x3

from solution.quaternion import RotationQuaterion
from solution.cross_matrix import get_cross_matrix

import solution


@dataclass
class ESKF():

    accm_std: float
    accm_bias_std: float
    accm_bias_p: float

    gyro_std: float
    gyro_bias_std: float
    gyro_bias_p: float

    gnss_std_ne: float
    gnss_std_d: float

    accm_correction: 'ndarray[3,3]'
    gyro_correction: 'ndarray[3,3]'
    lever_arm: 'ndarray[3]'

    do_approximations: bool
    use_gnss_accuracy: bool = False

    Q_err: 'ndarray[12,12]' = field(init=False, repr=False)
    g: 'ndarray[3]' = np.array([0, 0, 9.82])

    def __post_init__(self):

        self.Q_err = scipy.linalg.block_diag(
            self.accm_std ** 2 * self.accm_correction @ self.accm_correction.T,
            self.gyro_std ** 2 * self.gyro_correction @ self.gyro_correction.T,
            self.accm_bias_std ** 2 * np.eye(3),
            self.gyro_bias_std ** 2 * np.eye(3),
        )
        self.gnss_cov = np.diag([self.gnss_std_ne]*2 + [self.gnss_std_d])**2

    def correct_z_imu(self,
                      x_nom_prev: NominalState,
                      z_imu: ImuMeasurement,
                      ) -> CorrectedImuMeasurement:
        solution.used["eskf.ESKF.correct_z_imu"] = True
        """Correct IMU measurement so it gives a measurmenet of acceleration 
        and angular velocity in body.

        Hint: self.accm_correction and self.gyro_correction translates 
        measurements from IMU frame (probably not correct name) to body frame

        Args:
            x_nom_prev (NominalState): previous nominal state
            z_imu (ImuMeasurement): raw IMU measurement

        Returns:
            CorrectedImuMeasurement: corrected IMU measurement
        """
        acc_est = self.accm_correction@(z_imu.acc - x_nom_prev.accm_bias)
        avel_est = self.gyro_correction@(z_imu.avel - x_nom_prev.gyro_bias)
        z_corr = CorrectedImuMeasurement(z_imu.ts, acc_est, avel_est)
        return z_corr

    def predict_nominal(self,
                        x_nom_prev: NominalState,
                        z_corr: CorrectedImuMeasurement,
                        ) -> NominalState:
        solution.used["eskf.ESKF.predict_nominal"] = True
        """Predict the nominal state, given a corrected IMU measurement

        Hint: Discrete time prediction of equation (10.58)
        See the assignment description for more hints 

        Args:
            x_nom_prev (NominalState): previous nominal state
            z_corr (CorrectedImuMeasurement): corrected IMU measuremnt

        Returns:
            x_nom_pred (NominalState): predicted nominal state
        """
        dt = z_corr.ts - x_nom_prev.ts
        pos_pred = (x_nom_prev.pos + dt*x_nom_prev.vel
                    + 0.5*(dt**2)*(x_nom_prev.ori.R @ z_corr.acc + self.g)
                    )
        vel_pred = (x_nom_prev.vel + dt *
                    (x_nom_prev.ori.R @ z_corr.acc + self.g))

        avel_norm = np.linalg.norm(z_corr.avel)
        rot_axis = z_corr.avel / avel_norm  # n in (10.28)
        angle_div_2 = 0.5*dt*avel_norm  # alpha in (10.28) divided by 2
        delta_rot = RotationQuaterion(np.cos(angle_div_2),
                                      np.sin(angle_div_2)*rot_axis)
        ori_pred = x_nom_prev.ori @ delta_rot

        acc_bias_pred = np.exp(-dt*self.accm_bias_p) * x_nom_prev.accm_bias
        gyro_bias_pred = np.exp(-dt*self.gyro_bias_p) * x_nom_prev.gyro_bias

        x_nom_pred = NominalState(pos_pred, vel_pred, ori_pred,
                                  acc_bias_pred, gyro_bias_pred,
                                  z_corr.ts)
        return x_nom_pred

    def get_error_A_continous(self,
                              x_nom_prev: NominalState,
                              z_corr: CorrectedImuMeasurement,
                              ) -> 'ndarray[15,15]':
        solution.used["eskf.ESKF.get_error_A_continous"] = True
        """Get the transition matrix, A, in (10.68)

        Hint: The S matrices can be created using get_cross_matrix

        You can use block_3x3 to simplify indexing if you want to.
        The first I element in A can be set as A[block_3x3(0, 1)] = np.eye(3)

        Args:
            x_nom_prev (NominalState): previous nominal state
            z_corr (CorrectedImuMeasurement): corrected IMU measurement
        Returns:
            A (ndarray[15,15]): A
        """
        A = np.zeros((15, 15))

        Rq = x_nom_prev.ori.as_rotmat()
        S_acc = get_cross_matrix(z_corr.acc)
        S_omega = get_cross_matrix(z_corr.avel)

        A[block_3x3(0, 1)] = np.eye(3)
        A[block_3x3(1, 2)] = - Rq @ S_acc
        A[block_3x3(1, 3)] = -Rq@self.accm_correction
        A[block_3x3(2, 2)] = -S_omega
        A[block_3x3(2, 4)] = -self.gyro_correction
        A[block_3x3(3, 3)] = -self.accm_bias_p*np.eye(3)
        A[block_3x3(4, 4)] = -self.gyro_bias_p*np.eye(3)
        return A

    def get_error_GQGT_continous(self,
                                 x_nom_prev: NominalState
                                 ) -> 'ndarray[15, 12]':
        solution.used["eskf.ESKF.get_error_GQGT_continous"] = True
        """The noise covariance matrix, GQGT, in (10.68)

        From (Theorem 3.2.2) we can see that (10.68) can be written as 
        d/dt x_err = A@x_err + G@n == A@x_err + m
        where m is gaussian with mean 0 and covariance G @ Q @ G.T. Thats why
        we need GQGT.

        Hint: you can use block_3x3 to simplify indexing if you want to.
        The first I element in G can be set as G[block_3x3(2, 1)] = -np.eye(3)

        Args:
            x_nom_prev (NominalState): previous nominal state
        Returns:
            GQGT (ndarray[15, 15]): G @ Q @ G.T
        """
        if True:
            G = np.zeros((15, 12))
        Rq = x_nom_prev.ori.as_rotmat()
        G[block_3x3(1, 0)] = -Rq
        G[block_3x3(2, 1)] = -np.eye(3)
        G[block_3x3(3, 2)] = np.eye(3)
        G[block_3x3(4, 3)] = np.eye(3)

        GQGT = G @ self.Q_err @ G.T

        return GQGT

    def get_van_loan_matrix(self, V: 'ndarray[30, 30]'):
        """Use this funciton in get_discrete_error_diff to get the van loan 
        matrix. See (4.63)

        All the tests are ran with do_approximations=False

        Args:
            V (ndarray[30, 30]): [description]

        Returns:
            VanLoanMatrix (ndarray[30, 30]): VanLoanMatrix
        """
        if self.do_approximations:
            # second order approcimation of matrix exponential which is faster
            VanLoanMatrix = np.eye(*V.shape) + V + (V@V) / 2
        else:
            VanLoanMatrix = scipy.linalg.expm(V)
        return VanLoanMatrix

    def get_discrete_error_diff(self,
                                x_nom_prev: NominalState,
                                z_corr: CorrectedImuMeasurement,
                                ) -> Tuple['ndarray[15, 15]',
                                           'ndarray[15, 15]']:
        solution.used["eskf.ESKF.get_discrete_error_diff"] = True
        """Get the discrete equivalents of A and GQGT in (4.63)

        Hint: you should use get_van_loan_matrix to get the van loan matrix

        See (4.5 Discretization) and (4.63) for more information. 
        Or see "Discretization of process noise" in 
        https://en.wikipedia.org/wiki/Discretization

        Args:
            x_nom_prev (NominalState): previous nominal state
            z_corr (CorrectedImuMeasurement): corrected IMU measurement

        Returns:
            Ad (ndarray[15, 15]): discrede transition matrix
            GQGTd (ndarray[15, 15]): discrete noise covariance matrix
        """
        if True:
            A = self.get_error_A_continous(x_nom_prev, z_corr)
            GQGT = self.get_error_GQGT_continous(x_nom_prev)

            dt = z_corr.ts - x_nom_prev.ts
        V = dt * np.block([[-A, GQGT],
                           [np.zeros_like(A), A.T]])
        VanLoanMatrix = self.get_van_loan_matrix(V)

        Ad = VanLoanMatrix[15:, 15:].T
        GQGTd = Ad @ VanLoanMatrix[:15, 15:]

        return Ad, GQGTd

    def predict_x_err(self,
                      x_nom_prev: NominalState,
                      x_err_prev_gauss: ErrorStateGauss,
                      z_corr: CorrectedImuMeasurement,
                      ) -> ErrorStateGauss:
        solution.used["eskf.ESKF.predict_x_err"] = True
        """Predict the error state

        Hint: This is doing a discrete step of (10.68) where x_err 
        is a multivariate gaussian.

        Args:
            x_nom_prev (NominalState): previous nominal state
            x_err_prev_gauss (ErrorStateGauss): previous error state gaussian
            z_corr (CorrectedImuMeasurement): corrected IMU measuremnt

        Returns:
            x_err_pred (ErrorStateGauss): predicted error state
        """
        Ad, GQGTd = self.get_discrete_error_diff(
            x_nom_prev, z_corr)

        P_pred = Ad @ x_err_prev_gauss.cov @ Ad.T + GQGTd
        x_err_pred = ErrorStateGauss(x_err_prev_gauss.mean, P_pred, z_corr.ts)
        return x_err_pred

    def predict_from_imu(self,
                         x_nom_prev: NominalState,
                         x_err_gauss: ErrorStateGauss,
                         z_imu: ImuMeasurement,
                         ) -> Tuple[NominalState, ErrorStateGauss]:
        solution.used["eskf.ESKF.predict_from_imu"] = True
        """Method called every time an IMU measurement is received

        Args:
            x_nom_prev (NominalState): previous nominal state
            x_err_gauss (ErrorStateGauss): previous error state gaussian
            z_imu (ImuMeasurement): raw IMU measurement

        Returns:
            x_nom_pred (NominalState): predicted nominal state
            x_err_pred (ErrorStateGauss): predicted error state
        """
        z_corr = self.correct_z_imu(x_nom_prev, z_imu)
        x_nom_pred = self.predict_nominal(x_nom_prev, z_corr)
        x_err_pred = self.predict_x_err(x_nom_prev, x_err_gauss,
                                        z_corr)
        return x_nom_pred, x_err_pred

    def get_gnss_measurment_jac(self, x_nom: NominalState) -> 'ndarray[3,15]':
        solution.used["eskf.ESKF.get_gnss_measurment_jac"] = True
        """Get the measurement jacobian, H.

        Hint: the gnss antenna has a relative position to the center given by
        self.lever_arm. How will the gnss measurement change if the drone is 
        rotated differently? Use get_cross_matrix and some other stuff :) 

        Returns:
            H (ndarray[3, 15]): [description]
        """
        H = np.eye(3, 15)
        H[:, 6:9] = x_nom.ori.as_rotmat() @ (-get_cross_matrix(self.lever_arm))
        return H

    def get_gnss_cov(self, z_gnss: GnssMeasurement) -> 'ndarray[3,3]':
        """Use this function in predict_gnss_measurement to get R. 
        Get gnss covariance estimate based on gnss estimated accuracy. 

        All the test data has self.use_gnss_accuracy=False, so this does not 
        affect the tests.

        There is no given solution to this function, feel free to play around!

        Returns:
            gnss_cov (ndarray[3,3]): the estimated gnss covariance
        """
        if self.use_gnss_accuracy and z_gnss.accuracy is not None:
            # play around with this part, the suggested way is not optimal
            gnss_cov = (z_gnss.accuracy/3)**2 * self.gnss_cov

        else:
            # dont change this part
            gnss_cov = self.gnss_cov
        return gnss_cov

    def predict_gnss_measurement(self,
                                 x_nom: NominalState,
                                 x_err: ErrorStateGauss,
                                 z_gnss: GnssMeasurement,
                                 ) -> MultiVarGaussStamped:
        solution.used["eskf.ESKF.predict_gnss_measurement"] = True
        """Predict the gnss measurement

        Hint: z_gnss is only used in get_gnss_cov and to get timestamp for 
        the predicted measurement

        Args:
            x_nom (NominalState): previous nominal state
            x_err (ErrorStateGauss): previous error state gaussian
            z_gnss (GnssMeasurement): gnss measurement

        Returns:
            z_gnss_pred_gauss (MultiVarGaussStamped): gnss prediction gaussian
        """
        z_pred = x_nom.pos + x_nom.ori.as_rotmat() @ self.lever_arm

        H = self.get_gnss_measurment_jac(x_nom)
        R = self.get_gnss_cov(z_gnss)
        S = H @ x_err.cov @ H.T + R

        z_gnss_pred_gauss = MultiVarGaussStamped(z_pred, S, z_gnss.ts)

        return z_gnss_pred_gauss

    def get_x_err_upd(self,
                      x_nom: NominalState,
                      x_err: ErrorStateGauss,
                      z_gnss_pred_gauss: MultiVarGaussStamped,
                      z_gnss: GnssMeasurement
                      ) -> ErrorStateGauss:
        solution.used["eskf.ESKF.get_x_err_upd"] = True
        """Update the error state from a gnss measurement

        Hint: see (10.75)
        Due to numerical error its recomended use the robust calculation of 
        posterior covariance.

        I_WH = np.eye(*P.shape) - W @ H
        P_upd = (I_WH @ P @ I_WH.T + W @ R @ W.T)

        Args:
            x_nom (NominalState): previous nominal state
            x_err (ErrorStateGauss): previous error state gaussian
            z_gnss_pred_gauss (MultiVarGaussStamped): gnss prediction gaussian
            z_gnss (GnssMeasurement): gnss measurement

        Returns:
            x_err_upd_gauss (ErrorStateGauss): updated error state gaussian
        """
        z_pred, S = z_gnss_pred_gauss
        innovation = z_gnss.pos - z_pred

        H = self.get_gnss_measurment_jac(x_nom)
        P = x_err.cov
        W = x_err.cov @ np.linalg.solve(S, H).T
        R = self.get_gnss_cov(z_gnss)

        x_err_upd = W @ innovation

        # x_err_cov_upd = P - W @ H @ P
        I_WH = np.eye(*P.shape) - W @ H
        x_err_cov_upd = (I_WH @ P @ I_WH.T + W @ R @ W.T)

        x_err_upd_gauss = ErrorStateGauss(x_err_upd, x_err_cov_upd, z_gnss.ts)
        return x_err_upd_gauss

    def inject(self,
               x_nom_prev: NominalState,
               x_err_upd: ErrorStateGauss
               ) -> Tuple[NominalState, ErrorStateGauss]:
        solution.used["eskf.ESKF.inject"] = True
        """Perform the injection step

        Hint: see (10.85) and (10.72) on how to inject into nominal state.
        See (10.86) on how to find error state after injection

        Args:
            x_nom_prev (NominalState): previous nominal state
            x_err_upd (ErrorStateGauss): updated error state gaussian

        Returns:
            x_nom_inj (NominalState): nominal state after injection
            x_err_inj (ErrorStateGauss): error state gaussian after injection
        """
        pos_inj = x_nom_prev.pos + x_err_upd.pos
        vel_inj = x_nom_prev.vel + x_err_upd.vel
        ori_inj = x_nom_prev.ori @ RotationQuaterion(1, 0.5*x_err_upd.avec)
        accm_bias_inj = x_nom_prev.accm_bias + x_err_upd.accm_bias
        gyro_bias_inj = x_nom_prev.gyro_bias + x_err_upd.gyro_bias

        x_nom_inj = NominalState(pos_inj, vel_inj, ori_inj,
                                 accm_bias_inj, gyro_bias_inj,
                                 x_nom_prev.ts)

        G_inj = scipy.linalg.block_diag(
            np.eye(6),
            np.eye(3) - get_cross_matrix(0.5 * x_err_upd.avec),
            np.eye(6))
        P_inj = G_inj @ x_err_upd.cov @ G_inj.T
        x_err_inj = ErrorStateGauss(np.zeros(15), P_inj, x_err_upd.ts)

        return x_nom_inj, x_err_inj

    def update_from_gnss(self,
                         x_nom_prev: NominalState,
                         x_err_prev: NominalState,
                         z_gnss: GnssMeasurement,
                         ) -> Tuple[NominalState,
                                    ErrorStateGauss,
                                    MultiVarGaussStamped]:
        solution.used["eskf.ESKF.update_from_gnss"] = True
        """Method called every time an gnss measurement is received.


        Args:
            x_nom_prev (NominalState): [description]
            x_nom_prev (NominalState): [description]
            z_gnss (GnssMeasurement): gnss measurement

        Returns:
            x_nom_inj (NominalState): previous nominal state 
            x_err_inj (ErrorStateGauss): previous error state
            z_gnss_pred_gauss (MultiVarGaussStamped): predicted gnss 
                measurement, used for NIS calculations.
        """
        z_gnss_pred_gauss = self.predict_gnss_measurement(x_nom_prev,
                                                          x_err_prev,
                                                          z_gnss)
        x_err_upd_gauss = self.get_x_err_upd(x_nom_prev,
                                             x_err_prev,
                                             z_gnss_pred_gauss,
                                             z_gnss)
        x_nom_inj, x_err_inj = self.inject(x_nom_prev, x_err_upd_gauss)

        return x_nom_inj, x_err_inj, z_gnss_pred_gauss
