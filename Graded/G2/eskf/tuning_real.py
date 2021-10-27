import numpy as np
from quaternion import RotationQuaterion
from datatypes.eskf_params import ESKFTuningParams
from datatypes.eskf_states import NominalState, ErrorStateGauss

tuning_params_real = ESKFTuningParams(
    accm_std        = 0.105 * 1e-3,
    accm_bias_std   = 4 * 1e-5,
    accm_bias_p     = 0.07 / 60.0,

    gyro_std        = 8.0 / 3600.0,
    gyro_bias_std   = np.deg2rad(0.3/3600),
    gyro_bias_p     = np.deg2rad(0.15 / 60),

    gnss_std_ne     = 5,
    gnss_std_d      = 10
    )  

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
        0.00183,                          # Angle vector
        0.01,                                      # Accelerometer bias
        0.01                                       # Gyro bias
    ])


# From the STIM300-datasheet:

# Accm:
# Bias instability: 0.4
# Velocity random walk: 0.8 / 60

# Gyro:
# Bias instability: 0.2

x_err_init_real = ErrorStateGauss(np.zeros(15), np.diag(init_std_real**2), 0)
