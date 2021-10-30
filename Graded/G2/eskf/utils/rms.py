import numpy as np
from numpy import ndarray

def rmse(
        pred_arr: 'ndarray[1]',
        true_arr: 'ndarray[1]'
    )->float:
    assert pred_arr.shape == true_arr.shape
    return np.sqrt(np.mean((pred_arr - true_arr)**2))

def rms(
        error_arr: 'ndarray[1]', 
        vector_size: int = 1
        )->'ndarray[1]': 
    assert vector_size > 0
    assert len(error_arr) > 0
    assert len(error_arr[0]) % vector_size == 0

    if vector_size == 1:
        return rmse(error_arr, np.zeros_like(error_arr))

    rms_arr = np.zeros_like(error_arr)

    for i in range(len(error_arr)):
        rms_arr[i] = rmse(
            error_arr[i], 
            np.zeros_like(error_arr[i])
        )
     
    return rms_arr
    
if __name__ == '__main__':
    a0 = np.array([0, 1, 2, 3])
    a1 = np.array([1, 2, 3, 5])
    print(rmse(a0, a1))

    # print(rms(a0, 2))
    # print()
