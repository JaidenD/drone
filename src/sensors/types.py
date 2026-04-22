import numpy as np
from dataclasses import dataclass

@dataclass
class IMUParams:
    gyro_bias: np.ndarray
    accel_bias: np.ndarray
    gyro_noise_std: float
    accel_noise_std: float

@dataclass
class IMUMeasurement:
    gyro: np.ndarray
    accel: np.ndarray

@dataclass
class BarometerParams:
    bias: float
    noise_std: float

@dataclass
class BarometerMeasurement:
    altitude: float
