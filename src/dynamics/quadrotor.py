import numpy as np
from dataclasses import dataclass

from dynamics.state import QuadState
from dynamics.params import QuadParams
from dynamics.motors import w_dot
from dynamics.allocation import total_wrench

from geometry.SE3_util import hat

def state_dot(state, w_cmd, params): 
    p = state.p
    v = state.v
    R = state.R
    Omega = state.Omega
    w = state.w

    # kinematics
    p_dot = v
    R_dot = R @ hat(Omega)

    # actuator dynamics
    w_dot_ = w_dot(w, w_cmd, params.tm)

    # body wrench
    F_body, M_body = total_wrench(w, params.rotor_positions, params.spin_dirs, params.kf, params.km, params.thrust_axis)

    F_world = R @ F_body

    e3 = np.array([0.0, 0.0, 1.0])
    v_dot = -params.g * e3 + F_world / params.m

    # rotational dynamics
    Omega_dot = np.linalg.solve(params.J, M_body - np.cross(Omega, params.J @ Omega))

    return QuadState(
        p = p_dot,
        v = v_dot,
        R = R_dot,
        Omega = Omega_dot,
        w = w_dot_,
    )



