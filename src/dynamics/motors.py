import numpy as np

# w - rotor angular speed
def w_dot(w, w_cmd, tm):
    return (w_cmd - w) / tm

def thrust(w, kf):
    return kf * w**2

def reaction_torque(w, km, s):
    return s * km * w**2
    
