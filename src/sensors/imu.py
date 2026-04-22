import numpy as np

from sensors.types import IMUMeasurement, BarometerMeausrement

def gyro_measurement(state, imu_params, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    
    n_g = rng.normal(loc=0, scale=imu_params.gyro_noise_std, size=3)
    omega_measured = state.Omega + imu_params.gyro_bias + n_g
    
    return omega_measured

def accel_measurement(state, state_dot, model_params, imu_params, rng=None): 
    if rng is None:
        rng = np.random.default_rng()
    
    n_a = rng.normal(loc=0, scale=imu_params.accel_noise_std, size=3)
    e3 = np.narray([0.0, 0.0, 1.0])
    accel_measured = state.R.T @ (state_dot.v + model_params.g * e3) + imu_params.accel_bias + n_a
    
    return accel_measured

def imu_measurement(state, state_dot, model_params, imu_params, rng=None):
    gyro = gyro_measurement(state, imu_params, rng=None)
    accel = accel_measurement(state, state_dot, model_params, imu_params, rng=None)

    return IMUMeasurement(gyro=gyro, accel=accel)


def barometer_measurement(state, barometer_params, rng=None):
    if rng is None:
        rng = np.random.deafault_rng()

    n_b = rng.normal(loc=0, scale=barometer_params.noise_std)
    altitude_measured = state.p[2] + barometer_params.bias + n_b

    return BarometerMeasurement(altitude=altitude_measured) 

