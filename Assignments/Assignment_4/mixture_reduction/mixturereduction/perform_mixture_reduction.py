# %% imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mixturereduction import mixture_moments

import solution
# %% setup and show initial


"""
Task description for task 1b)

Find out which of the gaussians to mixture

    i)
        w = 1 / 3 * {1, 1, 1}
        u = {0, 2, 4.5}
        P = {1, 1, 1}

    ii)
        w = 1/6 * {1, 4, 1}
        u = {0, 2, 4.5}
        P = {1, 1, 1}

    iii)
        w = 1/3 * {1, 1, 1}
        u = {0, 2, 4.5}
        P = {1, 2.25, 2.25}

    iv)
        w = 1/3 * {1, 1, 1}
        u = {0, 0, 2.5}
        P = {1, 2.25, 2.25}          
    
"""

# Quick access to the parameters changing the variables
w_list = [1/3, 1/3, 1/3]
u_list = [0, 0, 2.5]
P_list = [1, 2.25, 2.25]


"""
Discussion about which gaussians to merge together

Initial (and general) discussion, is that merging the gaussians depend quite a lot
about the problem. merging two gaussians might be better in one case, however could
be totally infeasable in another case. 

We would like to maintain as much probability as possible, and ideally it is better
to have a too wide probabilistic model compared to a too narrow, as a too narrow
model will cause us to lose too much probability

    i)
        I would say that it is best to merge 0 and 1
        
        This is due to the fact that it follows the original probability model, such 
        that the probabilistic estimate would be proportional to the actual 
        probability.

        By mergin either 0 and 2 or 1 and 2, the combined model would be increasing
        or decreasing where the original model is decreasing or increasing. This will
        lead to a likely too large shift in the probability

    ii)
        This is a bit more tricky. All of the combinations will cover the probability
        of the original function, at least when one is inside a domain D.

        The combination of 0 and 2 is too narrow, which causes the function to lose
        a lot of probability outside when outside of the expected value. 

        Combining 0 and 1 gives a proibability model that follows the original with 
        a good margine. It gives a slightly bias to lower values, and thus loses some
        probability for higher values. 

        Combining 1 and 2 cause a slight bias towards higher values, however it generates 
        a probability density with a larger variance. 

        I am a bit unsure which I think is best, however it would either be combining 
        0 and 1 or 1 and 2. The first is due to taking the probability at around 4.5 
        into account, and generally following the original probability for higher values
        reasonable well. The latter due to having a generally higher variance.

        However it is also important to take the system into account. If this was my friend(s)
        I hadn't trusted them to arrive until a year had passed. In that regard, both are
        valid.

    iii)
        Combining 1 and 2 would be best. This is due to the combination following the
        original almost perfectly.

    iv)
        Similar as in iii)

"""


def get_task_parameters():
    """
    Get mixture parameters

    Returns:
        w: shape (3,) normalized, 3 scalars
        mus: shape (3, 1),  3 1d vectors
        sigmas: shape (3, 1, 1) 3 1x1 matrices
    """
    mus = np.array(u_list).reshape(3, 1)
    sigmas = np.array(P_list).reshape(3, 1, 1)
    w = np.array(w_list)
    w = w.ravel() / np.sum(w)
    return w, mus, sigmas


def main():
    w, mus, sigmas = get_task_parameters()

    tot_mean, tot_sigma2 = (
        elem.squeeze() for elem in mixture_moments(w, mus, sigmas ** 2)
    )
    plot_n_sigmas = 3
    x = tot_mean + plot_n_sigmas * \
        np.sqrt(tot_sigma2) * np.arange(-1, 1 + 1e-10, 5e-2)

    fig1, ax1 = plt.subplots(num=1, clear=True)
    pdf_comp_vals = np.array(
        [
            multivariate_normal.pdf(
                x, mean=mus[i].item(), cov=sigmas[i].item() ** 2)
            for i in range(len(mus))
        ]
    )
    pdf_mix_vals = np.average(pdf_comp_vals, axis=0, weights=w)

    for i in range(len(mus)):
        ax1.plot(x, pdf_comp_vals[i], label=f"comp {i}")
    ax1.legend()

# %% merge and show combinations
    fi2, ax2 = plt.subplots(num=2, clear=True)
    ax2.plot(x, pdf_mix_vals, label="original")
    k = 0
    wcomb = np.zeros_like(w)
    mucomb = np.zeros_like(w)

    sigma2comb = np.zeros_like(w)
    pdf_mix_comb_vals = np.zeros_like(pdf_comp_vals)
    for i in range(2):  # index of first to merge
        for j in range(i + 1, 3):  # index of second to merge
            # the index of the non merging (only works for 3 components)
            k_other = 2 - k

        # merge components
            wcomb[k] = w[i] + w[j]
            mucomb[k], sigma2comb[k] = mixture_moments(
                w[[i, j]] / wcomb[k], mus[[i, j]], sigmas[[i, j]] ** 2
            )

        # plot
            pdf_mix_comb_vals[k] = (
                wcomb[k] * multivariate_normal.pdf(x, mucomb[k], sigma2comb[k])
                + w[k_other] * pdf_comp_vals[k_other]
            )
            ax2.plot(x, pdf_mix_comb_vals[k], label=f"combining {i} {j}")
            k += 1

    ax2.legend()

    print(mucomb)
    print(sigma2comb)
    sigmacomb = np.sqrt(sigma2comb)
    plt.show()


# %% run

if __name__ == "__main__":
    main()
