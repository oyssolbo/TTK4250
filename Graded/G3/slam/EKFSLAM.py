from typing import Tuple
import numpy as np
from numpy import ndarray
from dataclasses import dataclass, field
from scipy.linalg import block_diag
import scipy.linalg as la
from utils import rotmat2d
from JCBB import JCBB
import utils
import solution


@dataclass
class EKFSLAM:
    Q: ndarray
    R: ndarray
    do_asso: bool
    alphas: 'ndarray[2]' = field(default=np.array([0.001, 0.0001]))
    sensor_offset: 'ndarray[2]' = field(default=np.zeros(2))

    def f(self, 
            x: np.ndarray, 
            u: np.ndarray) -> np.ndarray:
        """
        Add the odometry u to the robot state x.

        Parameters
        ----------
        x : np.ndarray, shape=(3,)
            the robot state
        u : np.ndarray, shape=(3,)
            the odometry

        Returns
        -------
        np.ndarray, shape = (3,)
            the predicted state
        """
        # Extracting the variables. Assuming that x = [x, y, psi] and
        # u = [u, v, phi]
        x_prev = x[0]
        y_prev = x[1]
        psi_prev = x[2]

        u_k = u[0]
        v_k = u[1]
        phi_k = u[2]

        # Using eq. 11.7
        x_pred = x_prev + u_k*np.cos(psi_prev) - v_k*np.sin(psi_prev)
        y_pred = y_prev + u_k*np.sin(psi_prev) + v_k*np.cos(psi_prev)
        psi_pred = utils.wrapToPi(psi_prev + phi_k)

        xpred = np.array([x_pred, y_pred, psi_pred])

        # xpred = solution.EKFSLAM.EKFSLAM.f(self, x, u)
        return xpred

    def Fx(self, 
            x: np.ndarray, 
            u: np.ndarray) -> np.ndarray:
        """
        Calculate the Jacobian of f with respect to x.

        Parameters
        ----------
        x : np.ndarray, shape=(3,)
            the robot state
        u : np.ndarray, shape=(3,)
            the odometry

        Returns
        -------
        np.ndarray
            The Jacobian of f wrt. x.
        """
        # Using equation 11.13
        u_k = u[0]
        v_k = u[1]
        psi_k = x[2]

        Fx_13 = -u_k*np.sin(psi_k) - v_k*np.cos(psi_k)
        Fx_23 = u_k*np.cos(psi_k) - v_k*np.sin(psi_k)

        Fx = np.eye(3)
        Fx[0][2] = Fx_13
        Fx[1][2] = Fx_23

        # Fx = solution.EKFSLAM.EKFSLAM.Fx(self, x, u)
        return Fx

    def Fu(self, 
            x: np.ndarray, 
            u: np.ndarray) -> np.ndarray:
        """
        Calculate the Jacobian of f with respect to u.

        Parameters
        ----------
        x : np.ndarray, shape=(3,)
            the robot state
        u : np.ndarray, shape=(3,)
            the odometry

        Returns
        -------
        np.ndarray
            The Jacobian of f wrt. u.
        """
        # Using eq. 11.14
        # What is the point of having u as an input here??
        # It is not used in this equation, unless one would have to predict
        # the next iteration, but that does not make sence either
        psi_k = x[2]

        rotmat = utils.rotmat2d(psi_k)

        Fu = np.zeros((3,3))
        Fu[:2, :2] = rotmat
        Fu[2][2] = 1

        # Fu = solution.EKFSLAM.EKFSLAM.Fu(self, x, u)
        return Fu

    def predict(
            self, 
            eta: np.ndarray, 
            P: np.ndarray, 
            z_odo: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict the robot state using the z_odo as odometry the corresponding state&map covariance.

        Parameters
        ----------
        eta : np.ndarray, shape=(3 + 2*#landmarks,)
            the robot state and map concatenated
        P : np.ndarray, shape=(3 + 2*#landmarks,)*2
            the covariance of eta
        z_odo : np.ndarray, shape=(3,)
            the measured odometry

        Returns
        -------
        Tuple[np.ndarray, np.ndarray], shapes= (3 + 2*#landmarks,), (3 + 2*#landmarks,)*2
            predicted mean and covariance of eta.
        """
        # etapred, P = solution.EKFSLAM.EKFSLAM.predict(self, eta, P, z_odo)
        # return etapred, P

        # Check input matrix
        assert np.allclose(P, P.T), "EKFSLAM.predict: not symmetric P input"
        assert np.all(
            np.linalg.eigvals(P) >= 0
        ), "EKFSLAM.predict: non-positive eigen values in P input"
        assert (
            eta.shape * 2 == P.shape
        ), "EKFSLAM.predict: input eta and P shape do not match"
        etapred = np.empty_like(eta)

        x = eta[:3]
        etapred[:3] = self.f(x, z_odo)
        etapred[3:] = eta[3:]   # Must assume that the landmarks are identical to previously

        Fx = self.Fx(x, z_odo)
        Fu = self.Fu(x, z_odo)  # What the fuck is the point of Fu?

        # Evaluate covariance prediction in place to save computation
        # Only robot state changes, so only rows and colums of robot state needs changing
        # Cov matrix layout:
        # [[P_xx, P_xm],
        # [P_mx, P_mm]]
        P[:3, :3] = Fx @ (P[:3, :3]) @ Fx.T + Fu @ self.Q @ Fu.T
        P[:3, 3:] = Fx @ (P[:3, 3:])
        P[3:, :3] = (P[:3, 3:]).T

        assert np.allclose(P, P.T), "EKFSLAM.predict: Not symmetric P"
        assert np.all(
            np.linalg.eigvals(P) > 0
        ), "EKFSLAM.predict: Non-positive eigen values"
        assert (
            etapred.shape * 2 == P.shape
        ), "EKFSLAM.predict: Calculated shapes does not match"

        return etapred, P

    def h(self, eta: np.ndarray) -> np.ndarray:
        """
        Predict all the landmark positions in sensor frame.

        Parameters
        ----------
        eta : np.ndarray, shape=(3 + 2 * #landmarks,)
            The robot state and landmarks stacked.

        Returns
        -------
        np.ndarray, shape=(2 * #landmarks,)
            The landmarks in the sensor frame.
        """
        zpred = solution.EKFSLAM.EKFSLAM.h(self, eta)
        return zpred

        # Extract states and map
        x = eta[0:3]
        # Reshape map (2, #landmarks), m[:, j] is the jth landmark
        m = eta[3:].reshape((-1, 2)).T
        rho_k = (x[0:2].T)[:, None]

        Rot = rotmat2d(-x[2])

        # None as index ads an axis with size 1 at that position.
        # Numpy broadcasts size 1 dimensions to any size when needed
        delta_m = (m - rho_k - (Rot @ self.sensor_offset)[:, None])  # Relative position of landmark to sensor on robot in world frame

        # Predicted measurements in cartesian coordinates, beware sensor offset for VP. But 
        # how to counteract the sensor offset? Isn't that accounted for in delta_m?
        zpredcart = Rot @ delta_m #m + Rot.T @ delta_m # To get the measurments into world-frame

        zpred_r = ((zpredcart**2).sum(axis=0))**0.5
        zpred_theta = np.arctan2(zpredcart[1,:], zpredcart[0,:]) # Something buggy here
        # The two arrays above stacked on top of each other vertically like
        # [ranges;
        #  bearings]
        # into shape (2, #lmrk)
        zpred = np.vstack(zpred_r, zpred_theta) 

        # Stack measurements along one dimension, [range1 bearing1 range2 bearing2 ...]
        zpred = zpred.T.ravel()

        assert (
            zpred.ndim == 1 and zpred.shape[0] == eta.shape[0] - 3
        ), "SLAM.h: Wrong shape on zpred"

        return zpred

    def h_jac(self, eta: np.ndarray) -> np.ndarray:
        """
        Calculate the jacobian of h.

        Parameters
        ----------
        eta : np.ndarray, shape=(3 + 2 * #landmarks,)
            The robot state and landmarks stacked.

        Returns
        -------
        np.ndarray, shape=(2 * #landmarks, 3 + 2 * #landmarks)
            the jacobian of h wrt. eta.
        """
        H = solution.EKFSLAM.EKFSLAM.h_jac(self, eta)
        return H

        # Extract states and map
        x = eta[0:3]
        # Reshape map (2, #landmarks), m[j] is the jth landmark
        m = eta[3:].reshape((-1, 2)).T
        rho_k = (x[0:2].T)[:, None]

        numM = m.shape[1]

        Rot = rotmat2d(x[2]) 

        # Relative position of landmark to robot in world frame. m - rho that appears in (11.15) and (11.16)
        delta_m = (m - rho_k)

        # (2, #measurements), each measured position in cartesian coordinates like
        # [x coordinates;
        #  y coordinates]
        zc = (m - (Rot @ self.sensor_offset)[:, None])


        # How tf should one use the predicted measurements??
        # (2, #measurements), predicted measurements, like
        # [ranges;
        #  bearings]
        zpred = None  
        zr = None  # TODO, ranges

        Rpihalf = rotmat2d(np.pi / 2)

        # In what follows you can be clever and avoid making this for all the landmarks you _know_
        # you will not detect (the maximum range should be available from the data).
        # But keep it simple to begin with.

        # Allocate H and set submatrices as memory views into H
        # You may or may not want to do this like this
        H = np.zeros((2 * numM, 3 + 2 * numM))
        Hx = H[:, :3]  # Slice view, setting elements of Hx will set H as well
        Hm = H[:, 3:]  # Slice view, setting elements of Hm will set H as well

        # Proposed way is to go through landmarks one by one
        # Preallocate and update this for some speed gain if looping
        jac_z_cb = -np.eye(2, 3)
        for i in range(numM):  # But this whole loop can be vectorized
            ind = 2 * i  # Starting postion of the ith landmark into H
            # The inds slice for the ith landmark into H
            inds = slice(ind, ind + 2)

            # Using in-place substitution for jac_z_cb
            jac_z_cb[:,2] = -Rpihalf @ delta_m[:, i]
            
            # Using eq. 11.15 and hint from the assignment
            Hx[inds] = zc[:, i].T / (np.linalg.norm(zc[:, i], 2)) @ jac_z_cb
            Hx[inds] = zc[:, i].T @ Rpihalf.T / (np.linalg.norm(zc[:, i], 2)**2) @ jac_z_cb

            # Using eq. 11.16, one sees that Hm is identical to the first column in Hx, except negative
            Hm[inds, inds] = -Hx[inds, :2]

        # assert H for something 
        return H

    def add_landmarks(
        self, eta: np.ndarray, P: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate new landmarks, their covariances and add them to the state.

        Parameters
        ----------
        eta : np.ndarray, shape=(3 + 2*#landmarks,)
            the robot state and map concatenated
        P : np.ndarray, shape=(3 + 2*#landmarks,)*2
            the covariance of eta
        z : np.ndarray, shape(2 * #newlandmarks,)
            A set of measurements to create landmarks for

        Returns
        -------
        Tuple[np.ndarray, np.ndarray], shapes=(3 + 2*(#landmarks + #newlandmarks,), (3 + 2*(#landmarks + #newlandmarks,)*2
            eta with new landmarks appended, and its covariance
        """
        # etaadded, Padded = solution.EKFSLAM.EKFSLAM.add_landmarks(
        #     self, eta, P, z)
        # return etaadded, Padded

        n = P.shape[0]
        assert z.ndim == 1, "SLAM.add_landmarks: z must be a 1d array"

        numLmk = z.shape[0] // 2

        lmnew = np.empty_like(z)

        Gx = np.empty((numLmk * 2, 3))
        Rall = np.zeros((numLmk * 2, numLmk * 2))

        I2 = np.eye(2)  # Preallocate, used for Gx
        # For transforming landmark position into world frame
        sensor_offset_world = rotmat2d(eta[2]) @ self.sensor_offset
        sensor_offset_world_der = rotmat2d(eta[2] + np.pi / 2) @ self.sensor_offset  # Used in Gx

        for j in range(numLmk):
            ind = 2 * j
            inds = slice(ind, ind + 2)
            zj = z[inds]

            # Extracting the angle and the distance to measurement j 
            # Are zj in cartesian or polar coordinates here?
            # Makes more sence for the measurements to be in polar coordinates here
            zj_r = zj[0] # np.linalg.norm(zj, 2)
            zj_phi = zj[1] # np.arctan2(zj[1], zj[0])
            alpha = eta[2] + zj_phi

            rot = rotmat2d(alpha)  # Rotmat in Gz
            # Calculate position of new landmark in body frame
            cart_lm_pos = zj_r * np.array([np.cos(zj_phi), np.sin(zj_phi)])
            # Calculate position of new landmark in NED
            lmnew[inds] = rotmat2d(eta[2]) @ cart_lm_pos + sensor_offset_world + eta[:2].T

            Gx[inds, :2] = I2
            Gx[inds, 2] = zj_r * (np.array([-np.sin(alpha), np.cos(alpha)]).T) + sensor_offset_world_der

            Gz = rot @ np.diag([1, zj_r])

            # Gz * R * Gz^T, transform measurement covariance from polar to cartesian coordinates
            Rall[inds, inds] = Gz @ self.R @ Gz.T

        assert len(lmnew) % 2 == 0, "SLAM.add_landmark: lmnew not even length"
        # Append new landmarks to state vector
        # Couldn't this cause problems when the measurements are not unique, such that the measurements 
        # match some of the original measurements?
        etaadded = np.concatenate(eta.reshape((1, len(eta))), lmnew.reshape(1, len(lmnew))) # Just wtf Python!!!
        # Block diagonal of P_new, see problem text in 1g) in graded assignment 3
        Padded = np.block(
            [
                [P, np.zeros_like(P.shape[0], Rall.shape[1])]
                [np.zeros_like(P.shape[1], Rall.shape[0]), Gx @ P[:3, :3] @ Gx.T + Rall]
            ])
        Padded[n:, :n] = P[:, :3] @ Gx.T # Top right corner of P_new
        # Transpose of above. Should yield the same as calculation, but this enforces symmetry and should be cheaper
        Padded[:n, n:] = (Padded[n:, :n]).T

        assert (
            etaadded.shape * 2 == Padded.shape
        ), "EKFSLAM.add_landmarks: calculated eta and P has wrong shape"
        assert np.allclose(
            Padded, Padded.T
        ), "EKFSLAM.add_landmarks: Padded not symmetric"
        assert np.all(
            np.linalg.eigvals(Padded) >= 0
        ), "EKFSLAM.add_landmarks: Padded not PSD"

        return etaadded, Padded

    def associate(
        self, z: np.ndarray, zpred: np.ndarray, H: np.ndarray, S: np.ndarray,
    ):  # -> Tuple[*((np.ndarray,) * 5)]:
        """
        Associate landmarks and measurements, and extract correct matrices for these.

        Parameters
        ----------
        z : np.ndarray,
            The measurements all in one vector
        zpred : np.ndarray
            Predicted measurements in one vector
        H : np.ndarray
            The measurement Jacobian matrix related to zpred
        S : np.ndarray
            The innovation covariance related to zpred

        Returns
        -------
        Tuple[*((np.ndarray,) * 5)]
            The extracted measurements, the corresponding zpred, H, S and the associations.

        Note
        ----
        See the associations are calculated  using JCBB. See this function for documentation
        of the returned association and the association procedure.
        """
        if self.do_asso:
            # Associate
            a = JCBB(z, zpred, S, self.alphas[0], self.alphas[1])

            # Extract associated measurements
            zinds = np.empty_like(z, dtype=bool)
            zinds[::2] = a > -1  # -1 means no association
            zinds[1::2] = zinds[::2]
            zass = z[zinds]

            # Extract and rearange predicted measurements and cov
            zbarinds = np.empty_like(zass, dtype=int)
            zbarinds[::2] = 2 * a[a > -1]
            zbarinds[1::2] = 2 * a[a > -1] + 1

            zpredass = zpred[zbarinds]
            Sass = S[zbarinds][:, zbarinds]
            Hass = H[zbarinds]

            assert zpredass.shape == zass.shape
            assert Sass.shape == zpredass.shape * 2
            assert Hass.shape[0] == zpredass.shape[0]

            return zass, zpredass, Hass, Sass, a
        else:
            # Should one do something here?
            pass

    def update(
            self, 
            eta: np.ndarray, 
            P: np.ndarray, 
            z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """
        Update eta and P with z, associating landmarks and adding new ones.

        Parameters
        ----------
        eta : np.ndarray
            [description]
        P : np.ndarray
            [description]
        z : np.ndarray, shape=(#detections, 2)
            [description]

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float, np.ndarray]
            [description]
        """
        # etaupd, Pupd, NIS, a = solution.EKFSLAM.EKFSLAM.update(self, eta, P, z)
        # return etaupd, Pupd, NIS, a

        numLmk = (eta.size - 3) // 2
        assert (len(eta) - 3) % 2 == 0, "EKFSLAM.update: landmark lenght not even"

        if numLmk > 0:
            # Prediction and innovation covariance
            zpred = self.h(eta)
            H = self.h_jac(eta)

            # Here you can use simply np.kron (a bit slow) to form the big (very big in VP after a while) R,
            # or be smart with indexing and broadcasting (3d indexing into 2d mat) realizing you are adding the same R on all diagonals
            # Using eq. 11.45
            S = H @ P @ H.T + np.kron(np.eye(numLmk), self.R) # Find a more efficient way to implement this
            assert (
                S.shape == zpred.shape * 2
            ), "EKFSLAM.update: wrong shape on either S or zpred"
            z = z.ravel()  # 2D -> flat

            # Perform data association
            za, zpred, Ha, Sa, a = self.associate(z, zpred, H, S)

            # No association could be made, so skip update
            if za.shape[0] == 0:
                etaupd = eta
                Pupd = P
                NIS = 0  # Beware this one when analysing consistency.
            else:
                # Create the associated innovation
                v = za.ravel() - zpred  # za: 2D -> flat
                v[1::2] = utils.wrapToPi(v[1::2])

                # Kalman mean update
                S_cho_factors = la.cho_factor(Sa) # Optional, used in places for S^-1, see scipy.linalg.cho_factor and scipy.linalg.cho_solve
                W = P @ H.T @ S_cho_factors # Kalman gain, can use S_cho_factors
                # m_upd = eta[3:] + W @ v
                etaupd = eta + W @ v #np.concatenate(eta[:3], m_upd)  # Kalman update

                # Kalman cov update: use Joseph form for stability
                jo = -W @ Ha
                # Same as adding Identity mat
                jo[np.diag_indices(jo.shape[0])] += 1
                Pupd = jo @ P @ jo.T + W @ self.R @ W.T # Kalman update using Joseph form. This is the main workload on VP after speedups

                # calculate NIS, can use S_cho_factors
                NIS = v.T @ S_cho_factors @ v

                # When tested, remove for speed
                assert np.allclose(
                    Pupd, Pupd.T), "EKFSLAM.update: Pupd not symmetric"
                assert np.all(
                    np.linalg.eigvals(Pupd) > 0
                ), "EKFSLAM.update: Pupd not positive definite"

        else:  # All measurements are new landmarks,
            a = np.full(z.shape[0], -1)
            z = z.flatten()
            NIS = 0  # Beware this one when analysing consistency.
            etaupd = eta
            Pupd = P

        # Create new landmarks if any is available
        if self.do_asso:
            is_new_lmk = a == -1
            if np.any(is_new_lmk):
                z_new_inds = np.empty_like(z, dtype=bool)
                z_new_inds[::2] = is_new_lmk
                z_new_inds[1::2] = is_new_lmk
                z_new = z[z_new_inds]
                etaupd, Pupd = self.add_landmarks(eta, P, z_new)  # Add new landmarks.

        assert np.allclose(
            Pupd, Pupd.T), "EKFSLAM.update: Pupd must be symmetric"
        assert np.all(np.linalg.eigvals(Pupd) >=
                      0), "EKFSLAM.update: Pupd must be PSD"

        return etaupd, Pupd, NIS, a

    @classmethod
    def NEESes(cls, x: np.ndarray, P: np.ndarray, x_gt: np.ndarray,) -> np.ndarray:
        """
        Calculates the total NEES and the NEES for the substates
        Args:
            x (np.ndarray): The estimate
            P (np.ndarray): The state covariance
            x_gt (np.ndarray): The ground truth
        Raises:
            AssertionError: If any input is of the wrong shape, and if debug mode is on, certain numeric properties
        Returns:
            np.ndarray: NEES for [all, position, heading], shape (3,)
        """

        assert x.shape == (3,), f"EKFSLAM.NEES: x shape incorrect {x.shape}"
        assert P.shape == (3, 3), f"EKFSLAM.NEES: P shape incorrect {P.shape}"
        assert x_gt.shape == (
            3,), f"EKFSLAM.NEES: x_gt shape incorrect {x_gt.shape}"

        d_x = x - x_gt
        d_x[2] = utils.wrapToPi(d_x[2])
        assert (
            -np.pi <= d_x[2] <= np.pi
        ), "EKFSLAM.NEES: error heading must be between (-pi, pi)"

        d_p = d_x[0:2]
        P_p = P[0:2, 0:2]
        assert d_p.shape == (2,), "EKFSLAM.NEES: d_p must be 2 long"
        d_heading = d_x[2]  # Note: scalar
        assert np.ndim(
            d_heading) == 0, "EKFSLAM.NEES: d_heading must be scalar"
        P_heading = P[2, 2]  # Note: scalar
        assert np.ndim(
            P_heading) == 0, "EKFSLAM.NEES: P_heading must be scalar"

        # NB: Needs to handle both vectors and scalars! Additionally, must handle division by zero
        NEES_all = d_x @ (np.linalg.solve(P, d_x))
        NEES_pos = d_p @ (np.linalg.solve(P_p, d_p))
        try:
            NEES_heading = d_heading ** 2 / P_heading
        except ZeroDivisionError:
            NEES_heading = 1.0  # TODO: beware

        NEESes = np.array([NEES_all, NEES_pos, NEES_heading])
        NEESes[np.isnan(NEESes)] = 1.0  # We may divide by zero, # TODO: beware

        assert np.all(NEESes >= 0), "ESKF.NEES: one or more negative NEESes"
        return NEESes
