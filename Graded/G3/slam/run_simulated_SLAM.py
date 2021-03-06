# %% Imports
from plotting import ellipse
from EKFSLAM import EKFSLAM
from typing import List, Optional

from scipy.io import loadmat
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.stats import chi2
import utils
from pathlib import Path

import anxs

try:
    from tqdm import tqdm
except ImportError as e:
    print(e)
    print("Install tqdm to have progress bar")

    # Def tqdm as dummy as it is not available
    def tqdm(*args, **kwargs):
        return args[0]

# %% Plot config check and style setup


# To see your plot config
print(f"Matplotlib backend: {matplotlib.get_backend()}")
print(f"Matplotlib config file: {matplotlib.matplotlib_fname()}")
print(f"Matplotlib config dir: {matplotlib.get_configdir()}")
plt.close("all")

# Try to set separate window ploting
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
    print(f"pyplot using style set {plt_styles}")
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
    datafile = Path(__file__).parents[1].joinpath("data/simulatedSLAM")
    simSLAM_ws = loadmat(str(datafile))

    # NB: this is a MATLAB cell, so needs1 to "double index" to get out the measurements of a time step k:
    #
    # ex:
    #
    # z_k = z[k][0] # z_k is a (2, m_k) matrix with columns equal to the measurements of time step k
    #
    ##
    z = [zk.T for zk in simSLAM_ws["z"].ravel()]

    landmarks = simSLAM_ws["landmarks"].T
    odometry = simSLAM_ws["odometry"].T
    poseGT = simSLAM_ws["poseGT"].T

    K = 350 #len(z)
    M = len(landmarks)

    # %% Tuning
    # Run 1 
    Q = np.diag([0.1, 0.1, 5.0 * np.pi / 180]) ** 2
    R = np.diag([0.3, 15.0 * np.pi / 180]) ** 2  
    JCBBalphas = np.array([1e-8, 1e-8]) 

    # Run 2
    # Q = np.diag([0.018, 0.018, 0.45 * np.pi/180]) ** 2
    # R = np.diag([0.1, 0.9 * np.pi/180]) ** 2  
    # JCBBalphas = np.array([1e-5, 1e-5]) 

    # Run 3
    Q = np.diag([0.03, 0.03, 1 * np.pi/180]) ** 2
    # R = np.diag([1e-3, 1e-3 * np.pi/180]) ** 2  
    # JCBBalphas = np.array([1e-8, 1e-3]) 

    # %% Initialize    
    doAsso = True
    slam = EKFSLAM(Q, R, do_asso=doAsso, alphas=JCBBalphas)

    # Allocate
    eta_pred: List[Optional[np.ndarray]] = [None] * K
    P_pred: List[Optional[np.ndarray]] = [None] * K
    eta_hat: List[Optional[np.ndarray]] = [None] * K
    P_hat: List[Optional[np.ndarray]] = [None] * K
    a: List[Optional[np.ndarray]] = [None] * K
    NIS = np.zeros(K)
    NISnorm = np.zeros(K)
    CI = np.zeros((K, 2))
    CInorm = np.zeros((K, 2))
    NEESes = np.zeros((K, 3))

    # For consistency testing
    alpha = 0.05

    # Init
    eta_pred[0] = poseGT[0]  # We start at the correct position for reference
    # We also say that we are 100% sure about that
    P_pred[0] = np.zeros((3, 3))

# %% Set up plotting
    # Plotting

    doAssoPlot = False
    playMovie = False
    if doAssoPlot:
        figAsso, axAsso = plt.subplots(num=1, clear=True)

# %% Run simulation
    N = K

    print("Starting sim (" + str(N) + " iterations)")

    num_total_asso = 0  # Used for calculating ANIS
    for k, z_k in tqdm(enumerate(z[:N]), total=N):
        # See top: need to do "double indexing" to get z at time step k
        # Transpose is to stack measurements rowwise
        # z_k = z[k][0].T

        eta_hat[k], P_hat[k], NIS[k], a[k] = slam.update(eta_pred[k], P_pred[k], z_k)

        if k < K - 1:
            eta_pred[k + 1], P_pred[k + 1] = slam.predict(eta_hat[k], P_hat[k], odometry[k])

        assert (
            eta_hat[k].shape[0] == P_hat[k].shape[0]
        ), "Dimensions of mean and covariance do not match"

        num_asso = np.count_nonzero(a[k] > -1)
        num_total_asso += num_asso

        CI[k] = chi2.interval(1 - alpha, 2 * num_asso)
        # CI[k] = tuple(ti/len(eta_hat[k]) for ti in CI[k])

        if num_asso > 0:
            NISnorm[k] = NIS[k] / (2 * num_asso)#* len(eta_hat[k]))
            CInorm[k] = CI[k] / (2 * num_asso)
        else:
            NISnorm[k] = 1
            CInorm[k].fill(1)

        NEESes[k] = slam.NEESes(eta_hat[k][:3], P_hat[k][:3, :3], poseGT[k]) 

        if doAssoPlot and k > 0:
            axAsso.clear()
            axAsso.grid()
            zpred = slam.h(eta_pred[k]).reshape(-1, 2)
            axAsso.scatter(z_k[:, 0], z_k[:, 1], label="z")
            axAsso.scatter(zpred[:, 0], zpred[:, 1], label="zpred")
            xcoords = np.block(
                [[z_k[a[k] > -1, 0]], [zpred[a[k][a[k] > -1], 0]]]).T
            ycoords = np.block(
                [[z_k[a[k] > -1, 1]], [zpred[a[k][a[k] > -1], 1]]]).T
            for x, y in zip(xcoords, ycoords):
                axAsso.plot(x, y, lw=3, c="r")
            axAsso.legend()
            axAsso.set_title(
                f"k = {k}, {np.count_nonzero(a[k] > -1)} associations")
            plt.draw()
            plt.pause(0.001)

    print("Sim complete")

    pose_est = np.array([x[:3] for x in eta_hat[:N]])
    lmk_est = [eta_hat_k[3:].reshape(-1, 2) for eta_hat_k in eta_hat]
    lmk_est_final = lmk_est[N - 1]

    np.set_printoptions(precision=4, linewidth=100)

# %% Plotting of results
    mins = np.amin(landmarks, axis=0)
    maxs = np.amax(landmarks, axis=0)

    ranges = maxs - mins
    offsets = ranges * 0.2

    mins -= offsets
    maxs += offsets

    fig2, ax2 = plt.subplots(num=2, clear=True)
    # Landmarks
    ax2.scatter(*landmarks.T, c="r", marker="^")
    ax2.scatter(*lmk_est_final.T, c="b", marker=".")
    # Draw covariance ellipsis of measurements
    for l, lmk_l in enumerate(lmk_est_final):
        idxs = slice(3 + 2 * l, 3 + 2 * l + 2)
        rI = P_hat[N - 1][idxs, idxs]
        el = ellipse(lmk_l, rI, 5, 200)
        ax2.plot(*el.T, "b")

    ax2.plot(*poseGT.T[:2], c="r", label="gt")
    ax2.plot(*pose_est.T[:2], c="g", label="est")
    ax2.plot(*ellipse(pose_est[-1, :2], P_hat[N - 1][:2, :2], 5, 200).T, c="g")
    ax2.set(title="Results", xlim=(mins[0], maxs[0]), ylim=(mins[1], maxs[1]))
    ax2.axis("equal")
    ax2.grid()

# %% Consistency

    # NIS
    insideCI = (CInorm[:N, 0] <= NISnorm[:N]) * (NISnorm[:N] <= CInorm[:N, 1])

    fig3, ax3 = plt.subplots(num=3, clear=True)
    ax3.plot(CInorm[:N, 0], '--')
    ax3.plot(CInorm[:N, 1], '--')
    ax3.plot(NISnorm[:N], lw=0.5)

    ax3.set_title(f'NIS, {insideCI.mean()*100}% inside CI')

    # NEES

    fig4, ax4 = plt.subplots(nrows=3, ncols=1, figsize=(
        7, 5), num=4, clear=True, sharex=True)
    tags = ['All', 'Pos', 'Heading']
    dfs = [3, 2, 1]

    for ax, tag, NEES, df in zip(ax4, tags, NEESes.T, dfs):
        CI_NEES = chi2.interval(1 - alpha, df)
        ax.plot(np.full(N, CI_NEES[0]), '--')
        ax.plot(np.full(N, CI_NEES[1]), '--')
        ax.plot(NEES[:N], lw=0.5)
        insideCI = (CI_NEES[0] <= NEES) * (NEES <= CI_NEES[1])
        ax.set_title(f'NEES {tag}: {insideCI.mean()*100}% inside CI')

        CI_ANEES = np.array(chi2.interval(1 - alpha, df*N)) / N
        print(f"CI ANEES {tag}: {CI_ANEES}")
        print(f"ANEES {tag}: {NEES.mean()}")
        # print(f"Self-made ANEES for {tag}: {anxs.anXs(NEES)}") # For testing -- they are equal!!

    fig4.tight_layout()

    # ANIS
    # Calculated from forum-post, by using the total number of associations * 2 as the TDOF, with
    # the total number of associations as a normalizing factor
    ci = 1 - alpha
    dof = 2 * num_total_asso
    CI_ANIS = anxs.anXs_bounds(ci, dof, num_total_asso)
    ANIS = anxs.anis(NIS, dof)

    print(f"ANIS-lower confidence interval: {CI_ANIS[0]}")
    print(f"ANIS-upper confidence interval: {CI_ANIS[1]}") 
    print(f"ANIS: {ANIS}")

# %% RMSE

    ylabels = ['m', 'deg']
    scalings = np.array([1, 180/np.pi])

    fig5, ax5 = plt.subplots(nrows=2, ncols=1, figsize=(
        7, 5), num=5, clear=True, sharex=True)

    pos_error = pose_est[:N, :2] - poseGT[:N, :2]
    pos_err_norm = np.linalg.norm(pos_error, axis=1)
    heading_err = np.abs(utils.wrapToPi(pose_est[:N, 2] - poseGT[:N, 2]))

    errs = np.vstack((pos_err_norm, heading_err))

    for ax, err, tag, ylabel, scaling in zip(ax5, errs, tags[1:], ylabels, scalings):
        ax.plot(err*scaling)
        ax.set_title(
            f"{tag}: RMSE {np.sqrt((err**2).mean())*scaling} {ylabel}")
        ax.set_ylabel(f"[{ylabel}]")
        ax.grid()

    fig5.tight_layout()

# %% Position error
    # According to how I understood the response/answer on BB, they are looking for
    # a plot showing the positional error over time

    times = np.arange(K)

    pos_error_label = ylabels[0]
    pos_error_scale = scalings[0]

    fig6, ax6 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5), num=6, clear=True, sharex=True)

    ax6.plot(times, pos_error, label=[f"${s}$" for s in "xy"])
    ax6.set_title(f"{tags[1]}-error [{pos_error_label}]")
    ax6.set_ylabel(r"$\mathbf{\delta \rho}$ [$m$]")
    ax6.set_xlabel("s")
    ax6.legend(loc="upper right")
    ax6.grid()
    fig6.tight_layout()

# %% Movie time

    if playMovie:
        try:
            print("Recording movie...")

            from celluloid import Camera

            pauseTime = 0.05
            fig_movie, ax_movie = plt.subplots(num=6, clear=True)

            camera = Camera(fig_movie)

            ax_movie.grid()
            ax_movie.set(xlim=(mins[0], maxs[0]), ylim=(mins[1], maxs[1]))
            camera.snap()

            for k in tqdm(range(N)):
                ax_movie.scatter(*landmarks.T, c="r", marker="^")
                ax_movie.plot(*poseGT[:k, :2].T, "r-")
                ax_movie.plot(*pose_est[:k, :2].T, "g-")
                ax_movie.scatter(*lmk_est[k].T, c="b", marker=".")

                if k > 0:
                    el = ellipse(pose_est[k, :2], P_hat[k][:2, :2], 5, 200)
                    ax_movie.plot(*el.T, "g")

                numLmk = lmk_est[k].shape[0]
                for l, lmk_l in enumerate(lmk_est[k]):
                    idxs = slice(3 + 2 * l, 3 + 2 * l + 2)
                    rI = P_hat[k][idxs, idxs]
                    el = ellipse(lmk_l, rI, 5, 200)
                    ax_movie.plot(*el.T, "b")

                camera.snap()
            animation = camera.animate(interval=100, blit=True, repeat=False)
            print("playing movie")

        except ImportError:
            print(
                "Install celluloid module, \n\n$ pip install celluloid\n\nto get fancy animation of EKFSLAM."
            )

    plt.show()
# %% Main


if __name__ == "__main__":
    main()
