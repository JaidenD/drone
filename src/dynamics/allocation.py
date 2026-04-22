import numpy as np

def rotor_force(w_i, kf, thrust_axis):
    return kf * w_i**2 * thrust_axis

def rotor_drag_moment(w_i, km, spin_dir, thrust_axis):
    return spin_dir * km * w_i**2 * thrust_axis


def rotor_arm_moment(r_i, F_i):
    return np.cross(r_i, F_i)

def rotor_wrench(w_i, r_i, spin_dir, kf, km, thrust_axis):
    F_i = rotor_force(w_i, kf, thrust_axis)
    M_drag_i = rotor_drag_moment(w_i, km, spin_dir, thrust_axis)
    M_arm_i = rotor_arm_moment(r_i, F_i)
    M_i = M_drag_i + M_arm_i

    return F_i, M_i

def total_wrench(w, rotor_positions, spin_dirs, kf, km, thrust_axis):
    F_total = np.zeros(3)
    M_total = np.zeros(3)

    for i in range(len(w)):
        F_i, M_i = rotor_wrench(
            w[i],
            rotor_positions[i],
            spin_dirs[i],
            kf, 
            km,
            thrust_axis,
        )
        F_total += F_i
        M_total += M_i

    return F_total, M_total

