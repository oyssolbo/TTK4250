import numpy as np
from numpy import ndarray

from scipy.stats import chi2

def anis(
        nis: 'ndarray[2]', 
        dof: int)->float:
    return nis.sum() / dof

def anees(nXs: 'ndarray[2]')->float:
    assert len(nXs) > 0
    return np.sum(np.ravel(nXs)) / (float((np.ravel(nXs)).size))

def anXs_bounds(
        confidence: float, 
        Nd: int,
        N)->tuple:
    # Using 4.67
    assert confidence > 0
    assert confidence < 1
    lower = chi2.ppf((1 - confidence) / 2.0, df=Nd)/float(N)
    upper = chi2.ppf(1 - (1 - confidence) / 2.0, df=Nd)/float(N)
    return lower, upper

if __name__ == '__main__':
    lower, upper = anXs_bounds(0.9, 3, 5)
    print(lower)
    print(upper)