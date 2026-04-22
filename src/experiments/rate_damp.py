import numpy as np
import matplotlib.pyplot as plt

from dynamics.params import QuadParams
from dynamics.state import QuadState
from dynamics.quadrotor import state_dot

from sim.integrators import RK4_step

from control.control_allocation import allocate_wrench
from control.rate_controller import BodyRateController
from control.types import WrenchCommand

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

wh = np.sqrt(params.m * params.g / (4 * params.kf))

state = QuadState(
    p = np.array([0.0, 0.0, 0.0]),
    v = np.array([0.0, 0.0, 0.0]),
    R = np.eye(3),
    Omega = np.array([0.2, 0.0, 0.0]),
    w = np.array([wh, wh, wh, wh]),
)    

controller = BodyRateController(J=params.J, K_omega = np.array([1.0, 1.0, 1.0]))

# steps
N = 1000
dt = 1e-3

positions = []
velocities = []
omegas = []
ts = []
zs = []
vzs = []

Omega_d = np.array([0.0, 0.0, 0.0])

for k in range(N):
    t = k * dt
    ts.append(t)

    positions.append(state.p.copy())
    velocities.append(state.v.copy())
    omegas.append(state.Omega.copy())
    
    M_des = controller.compute_moment(state.Omega, Omega_d)
    thrust_des = params.m * params.g
    wrench = WrenchCommand(thrust=thrust_des, moment=M_des)
    w_cmd = allocate_wrench(wrench, params)

    state = RK4_step(state_dot, state, w_cmd, params, dt)

positions = np.array(positions)
velocities = np.array(velocities)
omegas = np.array(omegas)

plt.figure()
plt.plot(ts, omegas[:, 0], label="Omega_x")
plt.plot(ts, omegas[:, 1], label="Omega_y")
plt.plot(ts, omegas[:, 2], label="Omega_z")
plt.xlabel("time [s]")
plt.ylabel("body rate [rad/s]")
plt.legend()
plt.title("Body-rate response")

plt.figure()
plt.plot(ts, positions[:, 2])
plt.xlabel("time [s]")
plt.ylabel("z [m]")
plt.title("Altitude")

plt.figure()
plt.plot(ts, velocities[:, 2])
plt.xlabel("time [s]")
plt.ylabel("vz [m/s]")
plt.title("Vertical velocity")

plt.show()
