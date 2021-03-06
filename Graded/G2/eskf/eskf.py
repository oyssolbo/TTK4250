import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
from numpy import ndarray
import scipy
from dataclasses import dataclass, field
from typing import Tuple
from functools import cache
from math import cos, sin, isnan
# from tests.test_eskf import Test_ESKF_predict_nominal

from datatypes.multivargaussian import MultiVarGaussStamped
from datatypes.measurements import (ImuMeasurement,
                                    CorrectedImuMeasurement,
                                    GnssMeasurement)
from datatypes.eskf_states import NominalState, ErrorStateGauss
from utils.indexing import block_3x3

from quaternion import RotationQuaterion
from cross_matrix import get_cross_matrix

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

        # 6c)
        # self.accm_correction = np.eye(3)
        # self.gyro_correction = np.eye(3)

        # 7b)
        # self.accm_correction = np.around(self.accm_correction, 1)
        # self.gyro_correction = np.around(self.gyro_correction, 1)

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
        """
        Correct IMU measurement so it gives a measurment of acceleration 
        and angular velocity in body.

        Hint: self.accm_correction and self.gyro_correction translates 
        measurements from IMU frame (probably not correct name) to body frame

        Args:
            x_nom_prev (NominalState): previous nominal state
            z_imu (ImuMeasurement): raw IMU measurement

        Returns:
            CorrectedImuMeasurement: corrected IMU measurement
        """
        lin_acc_m = z_imu.acc - x_nom_prev.accm_bias
        ang_vel_m = z_imu.avel - x_nom_prev.gyro_bias

        # Rotate the raw measurements from m-frame to b-frame
        lin_acc_b = self.accm_correction@lin_acc_m
        ang_vel_b = self.gyro_correction@ang_vel_m
        z_corr = CorrectedImuMeasurement(z_imu.ts, lin_acc_b, ang_vel_b)

        # z_corr = solution.eskf.ESKF.correct_z_imu(self, x_nom_prev, z_imu)
        return z_corr

    def predict_nominal(self,
                        x_nom_prev: NominalState,
                        z_corr: CorrectedImuMeasurement,
                        ) -> NominalState:
        """
        Predict the nominal state, given a corrected IMU measurement

        Hint: Discrete time prediction of equation (10.58)
        See the assignment description for more hints 

        Args:
            x_nom_prev (NominalState): previous nominal state
            z_corr (CorrectedImuMeasurement): corrected IMU measuremnt

        Returns:
            x_nom_pred (NominalState): predicted nominal state
        """
        # Should only be required to check x_nom_prev, as z_corr should always
        # have a timestamp
        if x_nom_prev.ts == None:
            x_nom_prev.ts = 0
        
        # Preventing errors from a noninitialized rotation
        # if isnan(x_nom_prev.ori.real_part) or isnan(np.sum(x_nom_prev.ori.vec_part)):
        #     x_nom_prev.ori = RotationQuaterion(1, np.array([0, 0, 0]))

        # Using equation 10.58 to predict the nominal state

        # Time difference
        dt = z_corr.ts - x_nom_prev.ts 
        if dt == 0:
            return x_nom_prev

        # Measurements
        omega = z_corr.avel
        lin_acc = x_nom_prev.ori.R @ z_corr.acc + self.g    # Remember that it is the corrected IMU measurements!
        
        # Quaternion-dynamics
        kappa = dt*omega
        kappa_norm = np.linalg.norm(kappa, 2)

        # Differential equations
        pos_pred_dot = x_nom_prev.vel + 1/2.0*dt*lin_acc  # Beware dt^1
        vel_pred_dot = lin_acc
        quad_pred_dot = RotationQuaterion(cos(kappa_norm/2.0), sin(kappa_norm/2.0) * kappa.T/kappa_norm)

        # Euler integration
        pos_pred = x_nom_prev.pos + dt*pos_pred_dot
        vel_pred = x_nom_prev.vel + dt*vel_pred_dot
        quad_pred = x_nom_prev.ori @ quad_pred_dot

        # Predicted bias in accm and gyro must use discrete integration
        accm_pred = x_nom_prev.accm_bias*np.exp(-self.accm_bias_p*dt)
        gyro_pred = x_nom_prev.gyro_bias*np.exp(-self.gyro_bias_p*dt)

        x_nom_pred = NominalState(pos_pred, vel_pred, quad_pred, accm_pred, gyro_pred, z_corr.ts)

        # x_nom_pred = solution.eskf.ESKF.predict_nominal(self, x_nom_prev, z_corr)
        return x_nom_pred

    def get_error_A_continous(self,
                              x_nom_prev: NominalState,
                              z_corr: CorrectedImuMeasurement,
                              ) -> 'ndarray[15,15]':
        """
        Get the transition matrix, A, in (10.68)

        Hint: The S matrices can be created using get_cross_matrix. In the book
        a perfect IMU is expected (thus many I matrices). Here we have 
        to use the correction matrices, self.accm_correction and 
        self.gyro_correction, instead of som of the I matrices.  

        You can use block_3x3 to simplify indexing if you want to.
        The first I element in A can be set as A[block_3x3(0, 1)] = np.eye(3)

        Args:
            x_nom_prev (NominalState): previous nominal state
            z_corr (CorrectedImuMeasurement): corrected IMU measurement
        Returns:
            A (ndarray[15,15]): A
        """
        # Remember that we have the corrected measurements z_corr
        # Also remember that there is a difference between the measurement-frame and 
        # body frame. This means that we must correct using self.accm_correction and
        # self.gyro_correction. The accm_correction must be rotated
        A = np.zeros((15,15))
        A[block_3x3(0, 1)] = np.eye(3)
        A[block_3x3(1, 2)] = -x_nom_prev.ori.R @ get_cross_matrix(z_corr.acc)
        A[block_3x3(2, 2)] = -get_cross_matrix(z_corr.avel)
        A[block_3x3(1, 3)] = -x_nom_prev.ori.R @ self.accm_correction
        A[block_3x3(3, 3)] = -self.accm_bias_p*np.eye(3)
        A[block_3x3(2, 4)] = -self.gyro_correction               
        A[block_3x3(4, 4)] = -self.gyro_bias_p*np.eye(3) 

        # A = solution.eskf.ESKF.get_error_A_continous(self, x_nom_prev, z_corr)
        return A

    def get_error_GQGT_continous(self,
                                 x_nom_prev: NominalState
                                 ) -> 'ndarray[15, 12]':
        """
        The noise covariance matrix, GQGT, in (10.68)

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
        G = np.zeros((15, 12))
        G[block_3x3(1, 0)] = -x_nom_prev.ori.R
        G[block_3x3(2, 1)] = -np.eye(3)
        G[block_3x3(3, 2)] = np.eye(3)
        G[block_3x3(4, 3)] = np.eye(3)

        GQGT = G @ self.Q_err @ G.T

        # GQGT = solution.eskf.ESKF.get_error_GQGT_continous(self, x_nom_prev)
        return GQGT

    def get_van_loan_matrix(self, V: 'ndarray[30, 30]'):
        """
        Use this funciton in get_discrete_error_diff to get the van loan 
        matrix. See (4.63)

        All the tests are ran with do_approximations=False

        Args:
            V (ndarray[30, 30]): [description]

        Returns:
            VanLoanMatrix (ndarray[30, 30]): VanLoanMatrix
        """
        if self.do_approximations:
            # Second order approcimation of matrix exponential which is faster
            VanLoanMatrix = np.eye(*V.shape) + V + (V@V) / 2
        else:
            VanLoanMatrix = scipy.linalg.expm(V)
        return VanLoanMatrix

    def get_discrete_error_diff(self,
                                x_nom_prev: NominalState,
                                z_corr: CorrectedImuMeasurement,
                                ) -> Tuple['ndarray[15, 15]',
                                           'ndarray[15, 15]']:
        """
        Get the discrete equivalents of A and GQGT in (4.63)

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
        # Got comment from vitass that it is better to use dt instead of ts
        dt = z_corr.ts - x_nom_prev.ts

        # Getting matrices continous in time
        Ac = self.get_error_A_continous(x_nom_prev, z_corr)
        GQGTc = self.get_error_GQGT_continous(x_nom_prev)

        # Using Van Loans formula to extract the desired matrices
        V = np.zeros((30, 30))
        V[0:15, 0:15] = -dt*Ac
        V[0:15, 15:30] = dt*GQGTc
        V[15:30, 15:30] = dt*Ac.T 

        VL = self.get_van_loan_matrix(V)

        Ad = VL[15:, 15:].T 
        GQGTd = Ad@VL[:15, 15:]
        
        # Ad, GQGTd = solution.eskf.ESKF.get_discrete_error_diff(
        #     self, x_nom_prev, z_corr)
        return Ad, GQGTd

    def predict_x_err(self,
                      x_nom_prev: NominalState,
                      x_err_prev_gauss: ErrorStateGauss,
                      z_corr: CorrectedImuMeasurement,
                      ) -> ErrorStateGauss:
        """
        Predict the error state

        Hint: This is doing a discrete step of (10.68) where x_err 
        is a multivariate gaussian.

        Args:
            x_nom_prev (NominalState): previous nominal state
            x_err_prev_gauss (ErrorStateGauss): previous error state gaussian
            z_corr (CorrectedImuMeasurement): corrected IMU measuremnt

        Returns:
            x_err_pred (ErrorStateGauss): predicted error state
        """
        Ad, GQGd = self.get_discrete_error_diff(x_nom_prev, z_corr)

        P = Ad @ x_err_prev_gauss.cov @ Ad.T + GQGd 
        x_err_pred = ErrorStateGauss(x_err_prev_gauss.mean, P, z_corr.ts)

        # x_err_pred = solution.eskf.ESKF.predict_x_err(
        #     self, x_nom_prev, x_err_prev_gauss, z_corr)
        return x_err_pred

    def predict_from_imu(self,
                         x_nom_prev: NominalState,
                         x_err_gauss: ErrorStateGauss,
                         z_imu: ImuMeasurement,
                         ) -> Tuple[NominalState, ErrorStateGauss]:
        """
        Method called every time an IMU measurement is received

        Args:
            x_nom_prev (NominalState): previous nominal state
            x_err_gauss (ErrorStateGauss): previous error state gaussian
            z_imu (ImuMeasurement): raw IMU measurement

        Returns:
            x_nom_pred (NominalState): predicted nominal state
            x_err_pred (ErrorStateGauss): predicted error state
        """
        # if np.isnan(np.sum(x_nom_prev.ori.vec_part)) or np.isnan(x_nom_prev.ori.real_part):
        #     x_nom_prev.ori = RotationQuaterion(1, np.zeros(3))

        # Correcting the measurement
        z_corr = self.correct_z_imu(x_nom_prev, z_imu)

        # Predict nominal state
        x_nom_pred = self.predict_nominal(x_nom_prev, z_corr)

        # Predict error state
        x_err_pred = self.predict_x_err(x_nom_prev, x_err_gauss, z_corr)

        # x_nom_pred, x_err_pred = solution.eskf.ESKF.predict_from_imu(
        #     self, x_nom_prev, x_err_gauss, z_imu)
        return x_nom_pred, x_err_pred

    def get_gnss_measurment_jac(self, x_nom: NominalState) -> 'ndarray[3,15]':
        """
        Get the measurement jacobian, H.

        Hint: the gnss antenna has a relative position to the center given by
        self.lever_arm. How will the gnss measurement change if the drone is 
        rotated differently? Use get_cross_matrix and some other stuff :) 

        Returns:
            H (ndarray[3, 15]): [description]
        """
        # Unsure if the jacobian should include the measurements optained from other
        # sensors or just the GNSS-receiver

        # Assuming that the GNSS will only measure the position, and not
        # SOG, COG, pseudoranges etc. 
        # Also assuming that the measurement-model is fine with euler angles

        # Here we have that the measurement-model will include 
        # z = h(xt) + w
        # where the linearized version gives that
        # z = H xt + w

        # We assume that only a GNSS-receiver is used. This will give three 
        # measurements that are dependent on the position and orientation of
        # the UAV. Thus, we get that H = [eye(3) zeros(3) H_CM(3) zeros(3) zeros(3)]
        # where H_CM(3) gives the change in coordinates from the arm between CO
        # and CM 
        H_CM = x_nom.ori.R @ (-get_cross_matrix(self.lever_arm))

        H = np.zeros((3, 15))
        H[block_3x3(0,0)] = np.eye(3)
        H[block_3x3(0, 2)] = H_CM

        # H = solution.eskf.ESKF.get_gnss_measurment_jac(self, x_nom)
        return H

    def get_gnss_cov(self, z_gnss: GnssMeasurement) -> 'ndarray[3,3]':
        """
        Use this function in predict_gnss_measurement to get R. 
        Get gnss covariance estimate based on gnss estimated accuracy. 

        All the test data has self.use_gnss_accuracy=False, so this does not 
        affect the tests.

        There is no given solution to this function, feel free to play around!

        Returns:
            gnss_cov (ndarray[3,3]): the estimated gnss covariance
        """
        if self.use_gnss_accuracy and z_gnss.accuracy is not None:
            # Play around with this part, the suggested way is not optimal
            gnss_cov = (z_gnss.accuracy/3)**2 * self.gnss_cov

        else:
            # Don't change this part
            gnss_cov = self.gnss_cov
        return gnss_cov

    def predict_gnss_measurement(self,
                                 x_nom: NominalState,
                                 x_err: ErrorStateGauss,
                                 z_gnss: GnssMeasurement,
                                 ) -> MultiVarGaussStamped:
        """
        Predict the gnss measurement

        Hint: z_gnss is only used in get_gnss_cov and to get timestamp for 
        the predicted measurement

        Args:
            x_nom (NominalState): previous nominal state
            x_err (ErrorStateGauss): previous error state gaussian
            z_gnss (GnssMeasurement): gnss measurement

        Returns:
            z_gnss_pred_gauss (MultiVarGaussStamped): gnss prediction gaussian
        """
        # For estimating the next measurement, it is required to estimate the
        # expected value (position) and the covariance of the estimate 
        H_gnss = self.get_gnss_measurment_jac(x_nom)
        R = self.get_gnss_cov(z_gnss)
        P = x_err.cov

        # Using algorithm 1 on p. 56
        # Expected value
        z_hat = x_nom.pos + x_nom.ori.R @ self.lever_arm
        
        # Innovation covariance 
        S = H_gnss @ P @ H_gnss.T + R

        # Probability distribution
        z_gnss_pred_gauss = MultiVarGaussStamped(z_hat, S, z_gnss.ts)

        # z_gnss_pred_gauss = solution.eskf.ESKF.predict_gnss_measurement(
        #     self, x_nom, x_err, z_gnss)
        return z_gnss_pred_gauss

    def get_x_err_upd(self,
                      x_nom: NominalState,
                      x_err: ErrorStateGauss,
                      z_gnss_pred_gauss: MultiVarGaussStamped,
                      z_gnss: GnssMeasurement
                      ) -> ErrorStateGauss:
        """
        Update the error state from a gnss measurement

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
        # Using equation 10.75
        H_gnss = self.get_gnss_measurment_jac(x_nom)
        R = self.get_gnss_cov(z_gnss)
        P = x_err.cov
        S = z_gnss_pred_gauss.cov
        # Debugging: 
        # gnss_cov_eigs = np.linalg.eigvals(R)
        # prev_cov_eigs = np.linalg.eigvals(P)
        # assert np.all(prev_cov_eigs >= 0)
        # assert np.all(gnss_cov_eigs >= 0)

        # W = P @ H_gnss.T @ np.linalg.inv(S) #H_gnss @ P @ H_gnss.T + R)
        W = P @ np.linalg.solve(S, H_gnss).T  # More efficient to use np.linalg.solve instead of np.linalg.inv

        delta_x_hat = W @ (z_gnss.pos - z_gnss_pred_gauss.mean)
        WH_gnss = W @ H_gnss
        # P = (np.eye(np.shape(WH_gnss)[0]) - WH_gnss) @ P  # Failed due to isPSD()
        I_WH = np.eye(*P.shape) - WH_gnss
        P_upd = (I_WH @ P @ I_WH.T) + (W @ R @ W.T)
        # upd_cov_eigs = np.linalg.eigvals(P_upd)

        # This is triggered sometimes... Makes no fucking sence!
        # This could only be negative semidefinite or indefinite if x_err.cov indefinite
        # or negative semidefinite, which it is not!
        # Edit: Triggered due to numerical inaccuracies
        # assert np.all(upd_cov_eigs >= 0)
        
        x_err_upd_gauss = ErrorStateGauss(delta_x_hat, P_upd, z_gnss.ts)

        # x_err_upd_gauss = solution.eskf.ESKF.get_x_err_upd(
        #     self, x_nom, x_err, z_gnss_pred_gauss, z_gnss)
        return x_err_upd_gauss

    def inject(self,
               x_nom_prev: NominalState,
               x_err_upd: ErrorStateGauss
               ) -> Tuple[NominalState, ErrorStateGauss]:
        """
        Perform the injection step

        Hint: see (10.85) and (10.72) on how to inject into nominal state.
        See (10.86) on how to find error state after injection

        Args:
            x_nom_prev (NominalState): previous nominal state
            x_err_upd (ErrorStateGauss): updated error state gaussian

        Returns:
            x_nom_inj (NominalState): nominal state after injection
            x_err_inj (ErrorStateGauss): error state gaussian after injection
        """
        # Using equation 10.72 to inject the nominal state

        # This is incorrect... Learned by trial and error that += not correct for 
        # arrays... (Python being pyton)
        # x_nom_inj = x_nom_prev
        # x_nom_inj.pos += x_err_upd.pos
        # x_nom_inj.vel += x_err_upd.vel
        # x_nom_inj.ori = x_nom_inj.ori.multiply(RotationQuaterion(1, 1/2.0*x_err_upd.avec))
        # x_nom_inj.accm_bias += x_err_upd.accm_bias
        # x_nom_inj.gyro_bias += x_err_upd.gyro_bias

        pos = x_nom_prev.pos + x_err_upd.pos
        vel = x_nom_prev.vel + x_err_upd.vel
        ori = x_nom_prev.ori.multiply(RotationQuaterion(1, 1/2.0*x_err_upd.avec))
        accm_bias = x_nom_prev.accm_bias + x_err_upd.accm_bias
        gyro_bias = x_nom_prev.gyro_bias + x_err_upd.gyro_bias
        
        x_nom_inj = NominalState(pos, vel, ori, accm_bias, gyro_bias, x_nom_prev.ts)

        # Using equation 10.86 to reset the error state
        I_S_delta_theta = np.eye(3) - get_cross_matrix(1/2.0*x_err_upd.avec)
        G = np.eye(15)
        G[6:9, 6:9] = I_S_delta_theta

        P = x_err_upd.cov
        x_error_inj_cov = G @ P @ G.T

        # Debugging
        # assert np.all(np.linalg.eigvals(x_error_inj_cov) >= 0)

        x_error_inj_mean = np.zeros_like(x_err_upd.mean)
        x_err_inj = ErrorStateGauss(x_error_inj_mean, x_error_inj_cov, x_err_upd.ts)

        # x_nom_inj, x_err_inj = solution.eskf.ESKF.inject(
        #     self, x_nom_prev, x_err_upd)
        return x_nom_inj, x_err_inj

    def update_from_gnss(self,
                         x_nom_prev: NominalState,
                         x_err_prev: NominalState,
                         z_gnss: GnssMeasurement,
                         ) -> Tuple[NominalState,
                                    ErrorStateGauss,
                                    MultiVarGaussStamped]:
        """
        Method called every time an gnss measurement is received.


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
        # Updating the system whenever a new GNSS-measurement is received 
        # if np.isnan(np.sum(x_nom_prev.ori.vec_part)) or np.isnan(x_nom_prev.ori.real_part):
        #     x_nom_prev.ori = RotationQuaterion(1, np.zeros(3))

        # Predict the GNSS-measurement
        z_gnss_pred_gauss = self.predict_gnss_measurement(x_nom_prev, x_err_prev, z_gnss)
    
        # Update the error
        x_err_upd = self.get_x_err_upd(x_nom_prev, x_err_prev, z_gnss_pred_gauss, z_gnss)

        # Calculate the injected states
        x_nom_inj, x_err_inj = self.inject(x_nom_prev, x_err_upd)

        # x_nom_inj, x_err_inj, z_gnss_pred_gauss = solution.eskf.ESKF.update_from_gnss(
        #     self, x_nom_prev, x_err_prev, z_gnss)
        return x_nom_inj, x_err_inj, z_gnss_pred_gauss
