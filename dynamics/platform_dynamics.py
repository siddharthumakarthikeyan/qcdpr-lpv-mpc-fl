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

# Aerodynamic drag coefficients (linear viscous model).  These dampen modes
# the cable wrench cannot reach \u2014 in particular yaw rotation, which has
# zero moment arm from the symmetric cable geometry and would otherwise
# integrate freely under any small disturbance.  Translational drag is
# small (cable forces dominate), but rotational drag is significant for
# realism: a real platform suspended by 8 cables with two-cable-per-corner
# wraparound has substantial yaw friction.
DRAG_TRANS = 0.5         # N\u00b7s/m  (per axis)
DRAG_ROLL_PITCH = 0.2    # N\u00b7m\u00b7s/rad
DRAG_YAW   = 1.5         # N\u00b7m\u00b7s/rad  (high \u2014 cable wrap-around friction)


def compute_wrench(cable_forces: np.ndarray,
                   corner_positions: np.ndarray) -> tuple:
    F_net   = cable_forces.sum(axis=0)
    tau_net = np.zeros(3)
    for i in range(8):
        tau_net += np.cross(corner_positions[i], cable_forces[i])
    return F_net, tau_net


def _euler_rate_from_body(phi: float, theta: float,
                           omega_body: np.ndarray) -> np.ndarray:
    """
    Map body angular velocity (p, q, r) to ZYX Euler-angle rates.
    Standard kinematic relation:
        [dphi  ]   [1  sin(phi)tan(theta)  cos(phi)tan(theta)] [p]
        [dtheta] = [0  cos(phi)            -sin(phi)         ] [q]
        [dpsi  ]   [0  sin(phi)/cos(theta) cos(phi)/cos(theta)] [r]
    """
    sp, cp = np.sin(phi), np.cos(phi)
    ct = np.cos(theta)
    # Avoid singularity at theta = +/- pi/2; clamp ct away from zero.
    if abs(ct) < 1e-3:
        ct = 1e-3 if ct >= 0 else -1e-3
    tt = np.sin(theta) / ct
    p, q, r = omega_body
    dphi   = p + sp*tt*q + cp*tt*r
    dtheta = cp*q - sp*r
    dpsi   = (sp/ct)*q + (cp/ct)*r
    return np.array([dphi, dtheta, dpsi])


def continuous_dynamics(state: np.ndarray,
                        F_cable_net: np.ndarray,
                        tau_cable_net: np.ndarray,
                        mass: float = PLATFORM_MASS) -> np.ndarray:
    """
    Rigid-body 6-DoF dynamics.
    State layout (12,):
        [px,py,pz, vx,vy,vz, phi,theta,psi, p,q,r]
    where (p,q,r) are BODY angular rates and (phi,theta,psi) are ZYX Euler
    angles describing world->body rotation (R = Rz(psi)*Ry(theta)*Rx(phi)).
    Forces and torques are in the WORLD frame.
    """
    vx, vy, vz       = state[3], state[4], state[5]
    phi, theta, psi  = state[6], state[7], state[8]
    p, q, r          = state[9], state[10], state[11]

    ax = (F_cable_net[0] - DRAG_TRANS * vx) / mass
    ay = (F_cable_net[1] - DRAG_TRANS * vy) / mass
    az = (F_cable_net[2] - DRAG_TRANS * vz) / mass - G

    R = rotation_matrix(phi, theta, psi)
    tau_body = R.T @ tau_cable_net

    # Body-frame aerodynamic angular drag (essential for yaw stability since
    # the cable geometry cannot apply yaw torque).
    drag_body = np.array([DRAG_ROLL_PITCH * p,
                           DRAG_ROLL_PITCH * q,
                           DRAG_YAW        * r])

    omega_body = np.array([p, q, r])
    gyro       = np.cross(omega_body, I_PLAT @ omega_body)
    omega_dot  = I_PLAT_INV @ (tau_body - gyro - drag_body)

    euler_dot = _euler_rate_from_body(phi, theta, omega_body)

    return np.array([
        vx, vy, vz,
        ax, ay, az,
        euler_dot[0], euler_dot[1], euler_dot[2],
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
