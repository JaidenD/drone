import numpy as np
from dataclasses import dataclass

@dataclass
class QuadParams:
    m: float
    J: np.ndarray
    g: float
    rotor_positions: np.ndarray
    spin_dirs: np.ndarray
    kf: float
    km: float
    tm: float
    thrust_axis: np.ndarray

