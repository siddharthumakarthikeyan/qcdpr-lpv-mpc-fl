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
    UAV_MASS, UAV_IXX, UAV_IYY, UAV_IZZ, UAV_KD, G, DT
)


def rotation_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    cp, sp = np.cos(phi),   np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi),   np.sin(psi)
    return np.array([
        [cy*ct,  cy*st*sp - sy*cp,  cy*st*cp + sy*sp],
        [sy*ct,  sy*st*sp + cy*cp,  sy*st*cp - cy*sp],
        [-st,    ct*sp,             ct*cp            ]
    ])


def continuous_dynamics(state: np.ndarray,
                        ctrl: np.ndarray,
                        F_cable: np.ndarray) -> np.ndarray:
    px, py, pz, vx, vy, vz, phi, theta, psi, dphi, dtheta, dpsi = state
    T, tau_r, tau_p, tau_y = ctrl

    R = rotation_matrix(phi, theta, psi)

    thrust_body = np.array([0.0, 0.0, T])
    thrust_earth = R @ thrust_body

    ax = (thrust_earth[0] + F_cable[0] - UAV_KD * vx) / UAV_MASS
    ay = (thrust_earth[1] + F_cable[1] - UAV_KD * vy) / UAV_MASS
    az = (thrust_earth[2] + F_cable[2] - UAV_KD * vz) / UAV_MASS - G

    ddphi   = ((UAV_IYY - UAV_IZZ) / UAV_IXX) * dtheta * dpsi + tau_r  / UAV_IXX
    ddtheta = ((UAV_IZZ - UAV_IXX) / UAV_IYY) * dphi   * dpsi + tau_p  / UAV_IYY
    ddpsi   = ((UAV_IXX - UAV_IYY) / UAV_IZZ) * dphi   * dtheta + tau_y / UAV_IZZ

    return np.array([
        vx, vy, vz,
        ax, ay, az,
        dphi, dtheta, dpsi,
        ddphi, ddtheta, ddpsi
    ])


def step_rk4(state: np.ndarray,
             ctrl: np.ndarray,
             F_cable: np.ndarray,
             dt: float = DT) -> np.ndarray:
    k1 = continuous_dynamics(state,          ctrl, F_cable)
    k2 = continuous_dynamics(state + dt/2*k1, ctrl, F_cable)
    k3 = continuous_dynamics(state + dt/2*k2, ctrl, F_cable)
    k4 = continuous_dynamics(state + dt*k3,   ctrl, F_cable)
    new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    new_state[6:9] = np.arctan2(np.sin(new_state[6:9]), np.cos(new_state[6:9]))
    return new_state


def lpv_linearise(state: np.ndarray, ctrl: np.ndarray) -> tuple:
    phi, theta, psi = state[6], state[7], state[8]
    T = ctrl[0]
    m = UAV_MASS

    cp, sp = np.cos(phi),   np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi),   np.sin(psi)

    A_c = np.zeros((12, 12))
    A_c[0, 3] = 1.0
    A_c[1, 4] = 1.0
    A_c[2, 5] = 1.0
    A_c[3, 3] = -UAV_KD / m
    A_c[4, 4] = -UAV_KD / m
    A_c[5, 5] = -UAV_KD / m
    A_c[6,  9] = 1.0
    A_c[7, 10] = 1.0
    A_c[8, 11] = 1.0
    dpsi   = state[11]
    dtheta = state[10]
    dphi   = state[9]
    A_c[9,  10] = ((UAV_IYY - UAV_IZZ) / UAV_IXX) * dpsi
    A_c[9,  11] = ((UAV_IYY - UAV_IZZ) / UAV_IXX) * dtheta
    A_c[10,  9] = ((UAV_IZZ - UAV_IXX) / UAV_IYY) * dpsi
    A_c[10, 11] = ((UAV_IZZ - UAV_IXX) / UAV_IYY) * dphi
    A_c[11,  9] = ((UAV_IXX - UAV_IYY) / UAV_IZZ) * dtheta
    A_c[11, 10] = ((UAV_IXX - UAV_IYY) / UAV_IZZ) * dphi

    dRz_dphi   = np.array([-cy*st*sp + sy*cp, -sy*st*sp - cy*cp, -ct*sp]) * (T / m)
    dRz_dtheta = np.array([cy*ct*cp,           sy*ct*cp,          -st*cp]) * (T / m)
    dRz_dpsi   = np.array([-sy*st*cp + cy*sp,  cy*st*cp + sy*sp,  0.0   ]) * (T / m)

    A_c[3, 6] = dRz_dphi[0]
    A_c[3, 7] = dRz_dtheta[0]
    A_c[3, 8] = dRz_dpsi[0]
    A_c[4, 6] = dRz_dphi[1]
    A_c[4, 7] = dRz_dtheta[1]
    A_c[4, 8] = dRz_dpsi[1]
    A_c[5, 6] = dRz_dphi[2]
    A_c[5, 7] = dRz_dtheta[2]
    A_c[5, 8] = dRz_dpsi[2]

    B_c = np.zeros((12, 4))
    R = rotation_matrix(phi, theta, psi)
    B_c[3:6, 0] = R[:, 2] / m
    B_c[9,  1] = 1.0 / UAV_IXX
    B_c[10, 2] = 1.0 / UAV_IYY
    B_c[11, 3] = 1.0 / UAV_IZZ

    d = np.zeros(12)
    d[5] = -G

    return A_c, B_c, d


def discretise_lpv(A_c: np.ndarray,
                   B_c: np.ndarray,
                   d: np.ndarray,
                   dt: float = DT) -> tuple:
    n = A_c.shape[0]
    A_d = np.eye(n) + A_c * dt + 0.5 * (A_c @ A_c) * dt**2
    B_d = B_c * dt
    d_d = d * dt
    return A_d, B_d, d_d


class UAVState:
    def __init__(self, position: np.ndarray, yaw: float = 0.0):
        self.x = np.zeros(12)
        self.x[0:3] = position
        self.x[8]   = yaw
        self.u = np.array([UAV_MASS * G, 0.0, 0.0, 0.0])

    @property
    def position(self) -> np.ndarray:
        return self.x[0:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:6]

    @property
    def attitude(self) -> np.ndarray:
        return self.x[6:9]

    @property
    def ang_vel(self) -> np.ndarray:
        return self.x[9:12]
