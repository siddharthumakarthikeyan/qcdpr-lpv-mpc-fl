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
    PLATFORM_MASS, PLATFORM_IXX, PLATFORM_IYY, PLATFORM_IZZ, G, DT
)
from dynamics.uav_dynamics import rotation_matrix


I_PLAT = np.diag([PLATFORM_IXX, PLATFORM_IYY, PLATFORM_IZZ])
I_PLAT_INV = np.diag([1/PLATFORM_IXX, 1/PLATFORM_IYY, 1/PLATFORM_IZZ])


def compute_wrench(cable_forces: np.ndarray,
                   corner_positions: np.ndarray) -> tuple:
    F_net   = cable_forces.sum(axis=0)
    tau_net = np.zeros(3)
    for i in range(8):
        tau_net += np.cross(corner_positions[i], cable_forces[i])
    return F_net, tau_net


def continuous_dynamics(state: np.ndarray,
                        F_cable_net: np.ndarray,
                        tau_cable_net: np.ndarray,
                        mass: float = PLATFORM_MASS) -> np.ndarray:
    vx, vy, vz = state[3], state[4], state[5]
    phi, theta, psi = state[6], state[7], state[8]
    dp, dq, dr = state[9], state[10], state[11]

    ax = F_cable_net[0] / mass
    ay = F_cable_net[1] / mass
    az = F_cable_net[2] / mass - G

    R = rotation_matrix(phi, theta, psi)
    tau_body = R.T @ tau_cable_net

    omega_body = np.array([dp, dq, dr])
    gyro = np.cross(omega_body, I_PLAT @ omega_body)
    omega_dot = I_PLAT_INV @ (tau_body - gyro)

    return np.array([
        vx, vy, vz,
        ax, ay, az,
        dp, dq, dr,
        omega_dot[0], omega_dot[1], omega_dot[2]
    ])


def step_rk4(state: np.ndarray,
             F_cable_net: np.ndarray,
             tau_cable_net: np.ndarray,
             dt: float = DT,
             mass: float = PLATFORM_MASS) -> np.ndarray:
    def f(s): return continuous_dynamics(s, F_cable_net, tau_cable_net, mass)
    k1 = f(state)
    k2 = f(state + dt/2*k1)
    k3 = f(state + dt/2*k2)
    k4 = f(state + dt*k3)
    new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    new_state[6:9] = np.arctan2(np.sin(new_state[6:9]), np.cos(new_state[6:9]))
    return new_state


class PlatformState:
    def __init__(self, position: np.ndarray, yaw: float = 0.0):
        self.x = np.zeros(12)
        self.x[0:3] = position
        self.x[8]   = yaw

    @property
    def position(self) -> np.ndarray:
        return self.x[0:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:6].copy()

    @property
    def attitude(self) -> np.ndarray:
        return self.x[6:9].copy()

    @property
    def pose(self) -> np.ndarray:
        return np.concatenate([self.x[0:3], self.x[6:9]])
