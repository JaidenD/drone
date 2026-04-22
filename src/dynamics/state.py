import numpy as np
from dataclasses import dataclass

@dataclass
class QuadState:
    p: np.ndarray # (3, ) position
    v: np.ndarray # (3, ) velocity
    R: np.ndarray # (3,3) attitude
    Omega: np.ndarray # (3, ) body frame angular velocity
    w: np.ndarray # (4, 3) rotor angular velocity
    
    def __add__(self, other):
        if not isinstance(other, QuadState):
            return NotImplemented
        return QuadState(
            p = self.p + other.p,
            v = self.v + other.v,
            R = self.R + other.R,
            Omega = self.Omega + other.Omega,
            w = self.w + other.w,
        )


    def __sub__(self, other):
        if not isinstance(other, QuadState):
            return NotImplemented
        return QuadState(
            p = self.p - other.p,
            v = self.v - other.v,
            R = self.R - other.R,
            Omega = self.Omega - other.Omega,
            w = self.w - other.w,
        )


    def __mul__(self, scalar):
        if not np.isscalar(scalar):
            return NotImplemented
        return QuadState(
            p = scalar * self.p,
            v = scalar * self.v,
            R = scalar * self.R,
            Omega = scalar * self.Omega,
            w = scalar * self.w,
        )

    def __rmul__(self, scalar):
        return self.__mul__(scalar)
