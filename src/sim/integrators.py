import numpy as np

from dynamics.state import QuadState
from dynamics.quadrotor import state_dot

from geometry.SE3_util import SO3_project

# RK4 integrator
def RK4_step(dynamics, state, control, params, dt = 1e-4):
    k1 = dynamics(state, control, params)

    x2 = state + dt / 2 * k1
    x2.R = SO3_project(x2.R)
    k2 = dynamics(x2, control, params)
    
    x3 = state + dt / 2 * k2
    x3.R = SO3_project(x3.R)
    k3 = dynamics(x3, control, params)
    
    x4 = state + dt * k3
    x4.R = SO3_project(x4.R)
    k4 = dynamics(x4, control, params)

    state_next = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    state_next.R = SO3_project(state_next.R)
    return state_next 

