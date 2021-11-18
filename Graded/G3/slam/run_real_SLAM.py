# %% Imports
from scipy.io import loadmat
from scipy.stats import chi2
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError as e:
    print(e)
    print("Install tqdm for progress bar")

    # def tqdm as dummy
    def tqdm(*args, **kwargs):
        return args[0]


import numpy as np
from EKFSLAM import EKFSLAM
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from plotting import ellipse
from vp_utils import detectTrees, odometry, Car
from utils import rotmat2d

import anxs
import utils

# %% plot config check and style setup


# to see your plot config
print(f"Matplotlib backend: {matplotlib.get_backend()}")
print(f"Matplotlib config file: {matplotlib.matplotlib_fname()}")
print(f"Matplotlib config dir: {matplotlib.get_configdir()}")
plt.close("all")

# try to set separate window ploting
if "inline" in matplotlib.get_backend():
    print("Plotting is set to inline at the moment:", end=" ")

    if "ipykernel" in matplotlib.get_backend():
        print("Backend is ipykernel (IPython?)")
        print("Trying to set backend to separate window:", end=" ")
        import IPython

        IPython.get_ipython().run_line_magic("matplotlib", "")
    else:
        print("Unknown inline backend")

print("Continuing with this plotting backend", end="\n\n\n")


# Set styles
try:
    # Installed with "pip install SciencePLots" (https://github.com/garrettj403/SciencePlots.git)
    # Gives quite nice plots
    plt_styles = ["science", "grid", "bright", "no-latex"]
    plt.style.use(plt_styles)
    print(f"Oyplot using style set {plt_styles}")
except Exception as e:
    print(e)
    print("Setting grid and only grid and legend manually")
    plt.rcParams.update(
        {
            # Setgrid
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.color": "k",
            "grid.alpha": 0.5,
            "grid.linewidth": 0.5,
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.fancybox": True,
            "legend.numpoints": 1,
        }
    )


def main():
# %% Load data
    victoria_park_foler = Path(
        __file__).parents[1].joinpath("data/victoria_park")
    realSLAM_ws = {
        **loadmat(str(victoria_park_foler.joinpath("aa3_dr"))),
        **loadmat(str(victoria_park_foler.joinpath("aa3_lsr2"))),
        **loadmat(str(victoria_park_foler.joinpath("aa3_gpsx"))),
    }

    timeOdo = (realSLAM_ws["time"] / 1000).ravel()
    timeLsr = (realSLAM_ws["TLsr"] / 1000).ravel()
    timeGnss = (realSLAM_ws["timeGps"] / 1000).ravel()

    steering = realSLAM_ws["steering"].ravel()
    speed = realSLAM_ws["speed"].ravel()
    LASER = (
        realSLAM_ws["LASER"] / 100
    )  # Divide by 100 to be compatible with Python implementation of detectTrees
    La_m = realSLAM_ws["La_m"].ravel()
    Lo_m = realSLAM_ws["Lo_m"].ravel()

    K = timeOdo.size
    mK = timeLsr.size
    Kgnss = timeGnss.size

# %% Parameters

    L = 2.83  # Axel distance
    H = 0.76  # Center to wheel encoder
    a = 0.95  # Laser distance in front of first axel
    b = 0.5  # Laser distance to the left of center

    car = Car(L, H, a, b)

    # %% Tuning
    # Run 1 
    # sigmas = np.array([0.018, 0.018, 0.45 * np.pi / 180])  
    # CorrCoeff = np.array([[1, 0, 0], [0, 1, 0.9], [0, 0.9, 1]])
    # Q = np.diag(sigmas) @ CorrCoeff @ np.diag(sigmas)
    # R = np.diag([0.1, 0.9 * np.pi / 180]) ** 2  
    # JCBBalphas = np.array([1e-5, 1e-5]) 

    # Run 2.1
    # 83% inside NIS
    # sigmas = np.array([1e-4, 1.25e-5, 0.4 * np.pi / 180]) 
    # CorrCoeff = np.array([[1, 0, 0], [0, 1, 0.9], [0, 0.9, 1]])
    # Q = np.diag(sigmas) @ CorrCoeff @ np.diag(sigmas)
    # R = np.diag([0.03, 0.75 * np.pi / 180]) ** 2  
    # JCBBalphas = np.array([1e-2, 1e-2]) 
    
    # Run 2.2
    # 75% inside NIS
    # sigmas = np.array([1e-2, 1.25e-2, 0.5 * np.pi / 180])  
    # CorrCoeff = np.array([[1, 0, 0], [0, 1, 0.9], [0, 0.9, 1]])
    # Q = np.diag(sigmas) @ CorrCoeff @ np.diag(sigmas)
    # R = np.diag([0.04, 0.5 * np.pi / 180]) ** 2 
    # JCBBalphas = np.array([1e-4, 1e-4]) 


    # Run 3
    sigmas = np.array([1e-5, 1e-5, 1e-4 * np.pi / 180])  
    CorrCoeff = np.array([[1, 0, 0], [0, 1, 0.9], [0, 0.9, 1]])
    Q = np.diag(sigmas) @ CorrCoeff @ np.diag(sigmas)
    R = np.diag([2.5, 2.5 * np.pi / 180]) ** 2  
    JCBBalphas = np.array([1e-5, 1e-5]) 

    sensorOffset = np.array([car.a + car.L, car.b])
    doAsso = True

    slam = EKFSLAM(Q, R, do_asso=doAsso, alphas=JCBBalphas,
                   sensor_offset=sensorOffset)

    # For consistency testing
    alpha = 0.05
    confidence_prob = 1 - alpha

    xupd = np.zeros((mK, 3))
    a = [None] * mK
    NIS = np.zeros(mK)
    NISnorm = np.zeros(mK)
    CI = np.zeros((mK, 2))
    CInorm = np.zeros((mK, 2))

    # Initialize state
    # You might want to tweak these for a good reference
    eta = np.array([Lo_m[0], La_m[0], 36 * np.pi / 180])
    P = np.zeros((3, 3))

    mk_first = 1  # First seems to be a bit off in timing
    mk = mk_first
    t = timeOdo[0]
    k_gnss = 0

    # Used for calculating the error between the current pos and the gnss
    pos_err = np.empty((0, 2), np.float64)


# %%  Run
    N = 5000  # K

    err_times = np.zeros((N))

    doPlot = False
    lh_pose = None

    if doPlot:
        fig, ax = plt.subplots(num=1, clear=True)

        lh_pose = ax.plot(eta[0], eta[1], "k", lw=3)[0]
        sh_lmk = ax.scatter(np.nan, np.nan, c="r", marker="x")
        sh_Z = ax.scatter(np.nan, np.nan, c="b", marker=".")

    do_raw_prediction = True
    if do_raw_prediction:
        odos = np.zeros((K, 3))
        odox = np.zeros((K, 3))
        
        odox[0] = eta
        P_odo = P.copy()
        for k in range(min(N, K - 1)):
            odos[k + 1] = odometry(speed[k + 1], steering[k + 1], 0.025, car)
            odox[k + 1], _ = slam.predict(odox[k], P_odo, odos[k + 1])

    num_total_asso = 0
    for k in tqdm(range(N)):
        if mk < mK - 1 and timeLsr[mk] <= timeOdo[k + 1]:
            # Force P to symmetric: there are issues with long runs (>10000 steps)
            # seem like the prediction might be introducing some minor asymetries,
            # so best to force P symetric before update
            P = (P + P.T) / 2
            dt = timeLsr[mk] - t
            if dt < 0:  # Avoid assertions as they can be optimized away?
                raise ValueError("Negative time increment")

            # Reset time to this laser time for next post predict
            t = timeLsr[mk]
            odo = odometry(speed[k + 1], steering[k + 1], dt, car)
            eta, P = slam.predict(eta, P, odo)

            z = detectTrees(LASER[mk])
            eta, P, NIS[mk], a[mk] = slam.update(eta, P, z)  

            num_asso = np.count_nonzero(a[mk] > -1)

            if num_asso > 0:
                NISnorm[mk] = NIS[mk] / (2 * num_asso)
                CInorm[mk] = np.array(chi2.interval(confidence_prob, 2 * num_asso)) / (
                    2 * num_asso
                )
                num_total_asso += num_asso
            else:
                NISnorm[mk] = 1
                CInorm[mk].fill(1)

            xupd[mk] = eta[:3]

            if doPlot:
                sh_lmk.set_offsets(eta[3:].reshape(-1, 2))
                if len(z) > 0:
                    zinmap = (
                        rotmat2d(eta[2])
                        @ (
                            z[:, 0] *
                            np.array([np.cos(z[:, 1]), np.sin(z[:, 1])])
                            + slam.sensor_offset[:, None]
                        )
                        + eta[0:2, None]
                    )
                    sh_Z.set_offsets(zinmap.T)
                lh_pose.set_data(*xupd[mk_first:mk, :2].T)

                ax.set(
                    xlim=[-200, 200],
                    ylim=[-200, 200],
                    title=f"step {k}, laser scan {mk}, landmarks {len(eta[3:])//2},\nmeasurements {z.shape[0]}, num new = {np.sum(a[mk] == -1)}",
                )
                plt.draw()
                plt.pause(0.00001)

            if timeGnss[k_gnss] <= timeLsr[mk + 1] and k_gnss < Kgnss:
                # Comparing position to GNSS-measurement 
                gnss_measurement = np.array([Lo_m[k_gnss], La_m[k_gnss]])
                temp = gnss_measurement - xupd[mk, :2]
                pos_err = np.append(pos_err, (gnss_measurement - xupd[mk, :2]).reshape(1, 2), axis=0)
                err_times[k] = mk
                k_gnss += 1

            mk += 1

        if k < K - 1:
            dt = timeOdo[k + 1] - t
            t = timeOdo[k + 1]
            odo = odometry(speed[k + 1], steering[k + 1], dt, car)
            eta, P = slam.predict(eta, P, odo)

# %% Consistency

    # NIS
    insideCI = (CInorm[:mk, 0] <= NISnorm[:mk]) * \
        (NISnorm[:mk] <= CInorm[:mk, 1])

    _, ax3 = plt.subplots(num=3, clear=True)
    ax3.plot(CInorm[:mk, 0], "--")
    ax3.plot(CInorm[:mk, 1], "--")
    ax3.plot(NISnorm[:mk], lw=0.5)

    ax3.set_title(f"NIS, {insideCI.mean()*100:.2f}% inside CI")

    # ANIS
    # Calculated from forum-post, by using the total number of associations * 2 as the TDOF, with
    # the total number of associations as a normalizing factor
    if num_total_asso > 0:
        ci = 1 - alpha
        dof = 2 * num_total_asso
        CI_ANIS = anxs.anXs_bounds(ci, dof, num_total_asso)
        ANIS = anxs.anis(NIS, dof)

        print(f"ANIS-lower confidence interval: {CI_ANIS[0]}")
        print(f"ANIS-upper confidence interval: {CI_ANIS[1]}") 
        print(f"ANIS: {ANIS}")

# %% Slam

    if do_raw_prediction:
        _, ax5 = plt.subplots(num=5, clear=True)
        ax5.scatter(
            Lo_m[timeGnss < timeOdo[N - 1]],
            La_m[timeGnss < timeOdo[N - 1]],
            c="r",
            marker=".",
            label="GNSS",
        )
        ax5.plot(*odox[:N, :2].T, label="Odom")
        ax5.grid()
        ax5.set_title("GNSS vs odometry integration")
        ax5.legend()

# %% Error in position vs GNSS over time
    if do_raw_prediction:
        fig6, ax6 = plt.subplots(nrows=3, ncols=1, figsize=(7, 5), num=5, clear=True, sharex=True)

        times = [i for i in err_times if i > 0]
        e_pos_x = pos_err[:,0]
        e_pos_y = pos_err[:,1]
        e_pos = np.stack([e_pos_x.T, e_pos_y.T])
        pos_err_norm = np.linalg.norm(e_pos, axis=0)

        fig6.suptitle('Difference between estimated position and GNSS')

        # Plot the x and y errors as a function of time
        ax6[0].scatter(times, pos_err[:,0], s=1, label="x")
        ax6[1].scatter(times, pos_err[:,1], s=1, label="y")
        ax6[2].scatter(times, pos_err_norm, s=1)#scatter(np.array([i for i in range(len(pos_err_norm))]), np.sqrt((pos_err_norm**2).mean()), label="y")

        ax6[0].set_title(r"$\mathbf{\delta}x$ [$m$]")
        ax6[1].set_title(r"$\mathbf{\delta}y$ [$m$]")
        ax6[2].set_title("RMSE")

        # for i in range(3):
        #     ax6[i].legend(loc="upper right")
        fig6.tight_layout()

# %% Plot summary
    _, ax7 = plt.subplots(num=6, clear=True)
    ax7.scatter(*eta[3:].reshape(-1, 2).T, color="r", marker="x", label="Trees")
    # Include the GNSS-measurements if available for comparison
    if do_raw_prediction:
        ax7.scatter(
            Lo_m[timeGnss < timeOdo[N - 1]],
            La_m[timeGnss < timeOdo[N - 1]],
            c="g",
            marker=".",
            label="GNSS",
        )
    ax7.plot(*xupd[mk_first:mk, :2].T)
    ax7.set(
        title=f"Steps {k}, laser scans {mk-1}, landmarks {len(eta[3:])//2},\nmeasurements {z.shape[0]}, num new = {np.sum(a[mk] == -1)}"
    )
    ax7.legend()
    plt.show()

# %% Main

if __name__ == "__main__":
    main()
