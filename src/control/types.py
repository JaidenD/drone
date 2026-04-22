import numpy as np
from dataclasses import dataclass

@dataclass
class BodyRateCommand:
    Omega_d: np.ndarray

@dataclass
class WrenchCommand:
    thrust: float
    moment: np.ndarray

    

