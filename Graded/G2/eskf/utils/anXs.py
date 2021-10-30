import numpy as np
from numpy.lib.function_base import _unwrap_dispatcher

from scipy.stats import chi2
from numpy import ndarray
from numpy import random

def get_average(arr: 'ndarray[1]'):
    assert len(arr) > 0
    return np.sum(arr) / len(arr) 

def get_bounds(
        confidence_interval: float,
        num_dof: float,
        num_vals: float):
    assert confidence_interval < 100
    assert confidence_interval > 0
    assert num_vals > 0
    assert num_dof > 0

    bound_percentile = (100 - confidence_interval) / 2
    bound_percentile /= 100

    lower = chi2.ppf(bound_percentile, num_dof) / num_vals
    upper = chi2.ppf(1 - bound_percentile, num_dof) / num_vals

    return lower, upper

def get_anXs(
        nXs_arr: 'ndarray[1]',
        confidence_interval: float,
        num_dof: int):
    avg = get_average(nXs_arr)
    lower, upper = get_bounds(confidence_interval, num_dof, len(nXs_arr))
    return avg, lower, upper 


if __name__ == '__main__':
    rnd_arr = random.rand(1,322)
    avg, lower, upper = get_anXs(rnd_arr, 2, 90)
    print("Average: ", avg)
    print("Lower: ", lower)
    print("Upper: ", upper)







