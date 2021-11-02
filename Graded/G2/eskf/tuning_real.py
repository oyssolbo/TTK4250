import numpy as np
from quaternion import RotationQuaterion
from datatypes.eskf_params import ESKFTuningParams
from datatypes.eskf_states import NominalState, ErrorStateGauss

tuning_params_real = ESKFTuningParams(
    # accm_std        = 1e-2, # 0.105 * 1e-3,
    # accm_bias_std   = 1e-5,
    # accm_bias_p     = 1e-8, #0.07 / 60.0,

    # gyro_std        = 1e-3, # 8.0 / 3600.0,
    # gyro_bias_std   = 5e-3, #np.deg2rad(0.3/3600),
    # gyro_bias_p     = 1e-8, #np.deg2rad(0.15 / 60),

    # gnss_std_ne     = 1.5,
    # gnss_std_d      = 3
    accm_std        = 0.0125,
    accm_bias_std   = 0.00075,
    accm_bias_p     = 0.0025,

    gyro_std        = 0.000075, 
    gyro_bias_std   = 0.00025,
    gyro_bias_p     = 0.001, 

    gnss_std_ne     = 0.275,
    gnss_std_d      = 0.65#, 
    # use_gnss_accuracy=True
    )  

x_nom_init_real = NominalState(
    np.array([0, 0, -5]),                        # Position
    np.array([20, 0, 0]),                        # Velocity
    RotationQuaterion.from_euler([-50, 0, 185]),    # Orientation
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


# From the STIM300-datasheet:

# Accm:
# Bias instability: 0.4
# Velocity random walk: 0.8 / 60

# Gyro:
# Bias instability: 0.2

x_err_init_real = ErrorStateGauss(np.zeros(15), np.diag(init_std_real**2), 0)
