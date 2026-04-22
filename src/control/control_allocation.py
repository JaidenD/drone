import numpy as np

def allocation_matrix(params):
    a = params.thrust_axis / np.linalg.norm(params.thrust_axis)
    A = np.zeros((4, 4))

    for i in range(4):
        r_i = params.rotor_positions[i]
        s_i = params.spin_dirs[i]

        moment_col = params.kf * np.cross(r_i, a) - s_i * params.km * a

        A[0, i] = params.kf
        A[1:, i] = moment_col

    return A


def allocate_wrench(wrench, params):
    A = allocation_matrix(params)
    w = np.concatenate(([wrench.thrust], wrench.moment))

    u = np.linalg.pinv(A) @ w      # u = omega^2
    u = np.clip(u, 0.0, None)

    return np.sqrt(u)

