import numpy as np
from numpy import ndarray

def anXs(nXs: 'ndarray[2]')->float:
    assert len(nXs) > 0
    return np.sum(np.ravel(nXs)) / (float((np.ravel(nXs)).size))