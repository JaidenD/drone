import numpy as np

class BodyRateController:
    def __init__(self, J, K_omega, moment_limits = None):
        self.J = J
        self.K_omega = K_omega # proportional
        self.moment_limits = moment_limits

    def compute_moment(self, Omega, Omega_d):
        e_Omega = Omega - Omega_d

        M_des = np.cross(Omega, self.J @ Omega) - self.K_omega @ e_Omega

        if self.moment_limits is not None:
            M_des = np.clip(M_des, -self.moment_limits, self.moment_limits)

        return M_des
