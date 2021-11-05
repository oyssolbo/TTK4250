import numpy as np
from quaternion import RotationQuaterion
from datatypes.eskf_params import ESKFTuningParams
from datatypes.eskf_states import NominalState, ErrorStateGauss


"""
Warning: These values may not equate the values delivered in the report!

There have been some experimentation afterwards, both in terms of variances and
initial values
"""

tuning_params_real = ESKFTuningParams(
    accm_std        = 0.02,
    accm_bias_std   = 0.001,
    accm_bias_p     = 0.001,

    gyro_std        = 0.0001, 
    gyro_bias_std   = 0.00025,
    gyro_bias_p     = 0.0015, 

    gnss_std_ne     = 0.25,
    gnss_std_d      = 0.75#, 
    # use_gnss_accuracy=True
    )  

x_nom_init_real = NominalState(
    np.array([0, 0, -5]),                        # Position
    np.array([20, 0, 0]),                        # Velocity
    RotationQuaterion.from_euler([-25, 5, 190]),    # Orientation # RotationQuaterion.from_euler([-50, 0, 180]) 
    np.zeros(3),                                # Accelerometer bias
    np.zeros(3),                                # Gyro bias
    ts=0)

init_std_real = np.repeat(
    repeats=3,                                  # Repeat each element 3 times
    a=[
        1,                                      # Position
        1,                                      # Velocity
        np.deg2rad(1),                          # Angle vector
        0.5,                                      # Accelerometer bias
        0.005                                       # Gyro bias
    ])

x_err_init_real = ErrorStateGauss(np.zeros(15), np.diag(init_std_real**2), 0)
