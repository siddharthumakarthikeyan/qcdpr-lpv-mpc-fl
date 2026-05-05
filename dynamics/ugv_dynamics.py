"""
Authors
-------
Siddharth U
    Research Scholar, School of Mechanical Engineering,
    SASTRA Deemed to be University, Thanjavur, India.
    E-mail: siddharth@sastra.ac.in

Dr. Badri Narayanan R
    Senior Assistant Professor, School of Mechanical Engineering,
    SASTRA Deemed to be University, Thanjavur, India.

Copyright (C) 2025 Siddharth U and Badri Narayanan R,
SASTRA Deemed to be University, Thanjavur, India.
All rights reserved. Released for academic research use only.
"""
import numpy as np
from config.parameters import (
    UGV_MASS, UGV_IZZ, UGV_KD, G, DT
)


def continuous_dynamics(state: np.ndarray,
                        ctrl: np.ndarray,
                        F_cable: np.ndarray) -> np.ndarray:
    px, py, psi, vx, vy, omega = state
    v_cmd, om_cmd = ctrl

    tau_v  = 0.05
    tau_om = 0.05

    ax_cable = F_cable[0] / UGV_MASS
    ay_cable = F_cable[1] / UGV_MASS

    ax_cmd = (v_cmd * np.cos(psi) - vx) / tau_v + ax_cable - UGV_KD * vx / UGV_MASS
    ay_cmd = (v_cmd * np.sin(psi) - vy) / tau_v + ay_cable - UGV_KD * vy / UGV_MASS
    dom    = (om_cmd - omega) / tau_om

    return np.array([
        vx, vy,
        omega,
        ax_cmd,
        ay_cmd,
        dom
    ])


def step_rk4(state: np.ndarray,
             ctrl: np.ndarray,
             F_cable: np.ndarray,
             dt: float = DT) -> np.ndarray:
    k1 = continuous_dynamics(state,           ctrl, F_cable)
    k2 = continuous_dynamics(state + dt/2*k1, ctrl, F_cable)
    k3 = continuous_dynamics(state + dt/2*k2, ctrl, F_cable)
    k4 = continuous_dynamics(state + dt*k3,   ctrl, F_cable)
    new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    new_state[2] = np.arctan2(np.sin(new_state[2]), np.cos(new_state[2]))
    return new_state


def lpv_linearise(state: np.ndarray, ctrl: np.ndarray) -> tuple:
    psi = state[2]
    v   = ctrl[0]
    cp, sp = np.cos(psi), np.sin(psi)
    tau_v = 0.05

    A_c = np.zeros((6, 6))
    A_c[0, 3] = 1.0
    A_c[1, 4] = 1.0
    A_c[2, 5] = 1.0
    A_c[3, 2]  = v * (-sp) / tau_v
    A_c[4, 2]  = v * ( cp) / tau_v
    A_c[3, 3]  = -1.0 / tau_v - UGV_KD / UGV_MASS
    A_c[4, 4]  = -1.0 / tau_v - UGV_KD / UGV_MASS
    A_c[5, 5]  = -1.0 / 0.05

    B_c = np.zeros((6, 2))
    B_c[3, 0] = cp / tau_v
    B_c[4, 0] = sp / tau_v
    B_c[5, 1] = 1.0 / 0.05

    d = np.zeros(6)
    return A_c, B_c, d


def discretise(A_c, B_c, d, dt=DT):
    n = A_c.shape[0]
    A_d = np.eye(n) + A_c * dt
    B_d = B_c * dt
    d_d = d * dt
    return A_d, B_d, d_d


class UGVState:
    def __init__(self, position_xy: np.ndarray, yaw: float = 0.0):
        self.x = np.zeros(6)
        self.x[0:2] = position_xy
        self.x[2]   = yaw

    @property
    def position(self) -> np.ndarray:
        return self.x[0:2]

    @property
    def heading(self) -> float:
        return float(self.x[2])

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:5]
