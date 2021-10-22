import numpy as np
from quaternion import RotationQuaterion
from datatypes.eskf_params import ESKFTuningParams
from datatypes.eskf_states import NominalState, ErrorStateGauss

tuning_params_real = ESKFTuningParams(
    accm_std=1,
    accm_bias_std=1,
    accm_bias_p=1,

    gyro_std=1,
    gyro_bias_std=1,
    gyro_bias_p=1,

    gnss_std_ne=1,
    gnss_std_d=1,

    use_gnss_accuracy=False)

x_nom_init_real = NominalState(
    np.array([0, 0, 0]),                        # Position
    np.array([0, 0, 0]),                        # Velocity
    RotationQuaterion.from_euler([0, 0, 0]),    # Orientation
    np.zeros(3),                                # Accelerometer bias
    np.zeros(3),                                # Gyro bias
    ts=0)

init_std_real = np.repeat(
    repeats=3,                                  # Repeat each element 3 times
    a=[
        1,                                      # Position
        1,                                      # Velocity
        np.deg2rad(1),                          # Angle vector
        1,                                      # Accelerometer bias
        1                                       # Gyro bias
    ])

x_err_init_real = ErrorStateGauss(np.zeros(15), np.diag(init_std_real**2), 0)
