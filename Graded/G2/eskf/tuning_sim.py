import numpy as np
from quaternion import RotationQuaterion
from datatypes.eskf_params import ESKFTuningParams
from datatypes.eskf_states import NominalState, ErrorStateGauss

tuning_params_sim = ESKFTuningParams(
    accm_std=0.3,
    accm_bias_std=1e-1,
    accm_bias_p=1e-4,

    gyro_std=0.5,
    gyro_bias_std=1e-1,
    gyro_bias_p=1e-4,

    gnss_std_ne=2,
    gnss_std_d=5
    )    

x_nom_init_sim = NominalState(
    np.array([0, 0, 0]),                        # Position
    np.array([0, 0, 0]),                        # Velocity
    RotationQuaterion.from_euler([0, 0, 0]),    # Orientation
    np.zeros(3),                                # Accelerometer bias
    np.zeros(3),                                # Gyro bias
    ts=0)

init_std_sim = np.repeat(
    repeats=3,                                  # Repeat each element 3 times
    a=[
        1,                                      # Position
        1,                                      # Velocity
        np.deg2rad(1),                          # Angle vector
        1,                                      # Accelerometer bias
        1                                       # Gyro bias
    ])
    
x_err_init_sim = ErrorStateGauss(np.zeros(15), np.diag(init_std_sim**2), 0)
