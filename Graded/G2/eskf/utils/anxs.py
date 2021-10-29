import numpy as np
from numpy import ndarray

def anXs(nXs: 'ndarray[1]')->float:
    assert len(nXs) > 0
    return np.sum(nXs) / len(nXs)