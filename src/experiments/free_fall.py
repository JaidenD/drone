import numpy as np

from dynamics.params import QuadParams
from dynamics.state import QuadState
from dynamics.quadrotor import state_dot 

from sim.integrators import RK4_step

params = QuadParams(
    m=10.0,
    J=np.eye(3),
    g=9.81,
    rotor_positions=np.array([
        [ 1.0,  1.0, 0.0],
        [-1.0,  1.0, 0.0],
        [-1.0, -1.0, 0.0],
        [ 1.0, -1.0, 0.0],
    ]),
    spin_dirs=np.array([1, -1, 1, -1]),
    kf=1.0,
    km=1.0,
    tm=1.0,
    thrust_axis=np.array([0.0, 0.0, 1.0]),
)

state = QuadState(
    p = np.array([0.0, 0.0, 0.0]),
    v = np.array([0.0, 0.0, 0.0]),
    R = np.eye(3),
    Omega = np.array([0.0, 0.0, 0.0]),
    w = np.array([0.0, 0.0, 0.0, 0.0]),
)    


w_cmd = np.array([0.0, 0.0, 0.0, 0.0])

# steps
N = 1000
dt = 1e-3

ts = []
zs = []
vzs = []

for k in range(N):
    t = k * dt

    ts.append(t)
    zs.append(state.p[2])
    vzs.append(state.v[2])

    state = RK4_step(state_dot, state, w_cmd, params, dt)

print("final z:", state.p[2])
print("final vz:", state.v[2])
print("final R:\n", state.R)
print("final Omega:", state.Omega)
