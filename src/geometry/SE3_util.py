import numpy as np

def hat(omega):
    wx, wy, wz = omega
    return np.array([
        [0.0, -wz,  wy],
        [wz,  0.0, -wx],
        [-wy, wx,  0.0],
    ])

def SO3_project(R):
    U, _, V = np.linalg.svd(R)
    D = np.diag([1.0, 1.0, np.linalg.det(U @ V)])

    R_projected = U @ D @ V
    return R_projected
