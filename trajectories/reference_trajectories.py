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
from config.parameters import PLATFORM_ZREF, T_SIM


def _smooth_step(t: float, t0: float, t1: float) -> float:
    if t <= t0:
        return 0.0
    if t >= t1:
        return 1.0
    s = (t - t0) / (t1 - t0)
    return s * s * (3 - 2 * s)


def _wrap(angle: float) -> float:
    return np.arctan2(np.sin(angle), np.cos(angle))


class QuadrilateralTrajectory:
    def __init__(self):
        self.z_ref = PLATFORM_ZREF
        self.side  = 2.0
        self.wp = np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [0.0, 0.0],
        ])
        self.T_move  = 17.0
        self.T_pause = 7.5
        self.T_leg   = self.T_move + self.T_pause
        self.yaw_ref = 0.0

    def __call__(self, t: float) -> tuple:
        leg   = int(t / self.T_leg) % 4
        t_leg = t % self.T_leg

        wp_start = self.wp[leg]
        wp_end   = self.wp[leg + 1]

        if t_leg <= self.T_move:
            tau     = t_leg / self.T_move
            s       = 10*tau**3 - 15*tau**4 + 6*tau**5
            sd      = (30*tau**2 - 60*tau**3 + 30*tau**4) / self.T_move
            sdd     = (60*tau - 180*tau**2 + 120*tau**3) / self.T_move**2
            pos     = wp_start + s   * (wp_end - wp_start)
            vel_xy  = sd  * (wp_end - wp_start)
            acc_xy  = sdd * (wp_end - wp_start)
            psi     = self.yaw_ref
            dpsi    = 0.0
            ddpsi   = 0.0
        else:
            pos      = wp_end.copy()
            vel_xy   = np.zeros(2)
            acc_xy   = np.zeros(2)
            psi      = self.yaw_ref
            dpsi     = 0.0
            ddpsi    = 0.0

        pose  = np.array([pos[0], pos[1], self.z_ref, 0.0, 0.0, _wrap(psi)])
        vel   = np.array([vel_xy[0], vel_xy[1], 0.0, 0.0, 0.0, dpsi])
        accel = np.array([acc_xy[0], acc_xy[1], 0.0, 0.0, 0.0, ddpsi])
        return pose, vel, accel


class CircularTrajectory:
    def __init__(self, radius: float = 1.5, period: float = 100.0):
        self.R   = radius
        self.T   = period
        self.cx  = 2.0
        self.cy  = 1.5
        self.z   = PLATFORM_ZREF
        self.omega = 2 * np.pi / period

    def __call__(self, t: float) -> tuple:
        w  = self.omega
        th = w * t

        px  = self.cx + self.R * np.cos(th)
        py  = self.cy + self.R * np.sin(th)
        vx  = -self.R * w * np.sin(th)
        vy  =  self.R * w * np.cos(th)
        ax  = -self.R * w**2 * np.cos(th)
        ay  = -self.R * w**2 * np.sin(th)

        # Yaw is held at zero: the cable wrench has only a small
        # tangential moment arm (set by CABLE_SKEW_ANGLE_DEG) so the
        # achievable yaw bandwidth is much lower than the bandwidth of
        # this trajectory's tangential motion.  Commanding
        # psi = th + pi/2 would force the controller to track a yaw
        # rotation it cannot physically deliver.
        psi   = 0.0
        dpsi  = 0.0
        ddpsi = 0.0

        pose  = np.array([px, py, self.z, 0.0, 0.0, _wrap(psi)])
        vel   = np.array([vx, vy, 0.0, 0.0, 0.0, dpsi])
        accel = np.array([ax, ay, 0.0, 0.0, 0.0, ddpsi])
        return pose, vel, accel


class SinusoidalTrajectory:
    def __init__(self,
                 vx_mean: float = 0.05,
                 amplitude: float = 1.0,
                 period_wave: float = 25.0):
        self.vx  = vx_mean
        self.A   = amplitude
        self.Tw  = period_wave
        self.ww  = 2 * np.pi / period_wave
        self.z   = PLATFORM_ZREF

    def __call__(self, t: float) -> tuple:
        w  = self.ww
        px = self.vx * t
        py = self.A * np.sin(w * t)
        vx = self.vx
        vy = self.A * w * np.cos(w * t)
        ax = 0.0
        ay = -self.A * w**2 * np.sin(w * t)

        # Yaw held at zero — see CircularTrajectory note above.
        psi   = 0.0
        dpsi  = 0.0
        ddpsi = 0.0

        pose  = np.array([px, py, self.z, 0.0, 0.0, _wrap(psi)])
        vel   = np.array([vx, vy, 0.0, 0.0, 0.0, dpsi])
        accel = np.array([ax, ay, 0.0, 0.0, 0.0, ddpsi])
        return pose, vel, accel


class SingleUAVTestTrajectory:
    def __init__(self):
        self.z_base = PLATFORM_ZREF + 1.0
        self.T1 = 30.0
        self.T2 = 35.0
        self.T3 = 35.0

    def __call__(self, t: float) -> tuple:
        if t < self.T1:
            frac  = t / self.T1
            angle = frac * np.pi / 2
            R     = 2.0
            px    = R * np.cos(angle)
            py    = R * np.sin(angle)
            pz    = self.z_base
            vx    = -R * (np.pi/2/self.T1) * np.sin(angle)
            vy    =  R * (np.pi/2/self.T1) * np.cos(angle)
            vz    = 0.0
            ax    = -R * (np.pi/2/self.T1)**2 * np.cos(angle)
            ay    = -R * (np.pi/2/self.T1)**2 * np.sin(angle)
            az    = 0.0
        elif t < self.T1 + self.T2:
            frac  = (t - self.T1) / self.T2
            px    = 0.0 + 3.0 * frac
            py    = 2.0
            pz    = self.z_base
            vx    = 3.0 / self.T2
            vy    = 0.0
            vz    = 0.0
            ax = ay = az = 0.0
        else:
            frac  = (t - self.T1 - self.T2) / self.T3
            w_h   = 2 * np.pi / self.T3
            r_h   = 0.3
            px    = 3.0 + r_h * np.cos(w_h * (t - self.T1 - self.T2))
            py    = 2.0 + r_h * np.sin(w_h * (t - self.T1 - self.T2))
            pz    = self.z_base + 1.5 * frac
            vx    = -r_h * w_h * np.sin(w_h * (t - self.T1 - self.T2))
            vy    =  r_h * w_h * np.cos(w_h * (t - self.T1 - self.T2))
            vz    = 1.5 / self.T3
            ax    = -r_h * w_h**2 * np.cos(w_h * (t - self.T1 - self.T2))
            ay    = -r_h * w_h**2 * np.sin(w_h * (t - self.T1 - self.T2))
            az    = 0.0

        pose  = np.array([px, py, pz, 0.0, 0.0, 0.0])
        vel   = np.array([vx, vy, vz, 0.0, 0.0, 0.0])
        accel = np.array([ax, ay, az, 0.0, 0.0, 0.0])
        return pose, vel, accel


TRAJECTORY_CLASSES = {
    'quadrilateral': QuadrilateralTrajectory,
    'circular':      CircularTrajectory,
    'sinusoidal':    SinusoidalTrajectory,
    'single_uav':    SingleUAVTestTrajectory,
}


def get_trajectory(name: str):
    cls = TRAJECTORY_CLASSES.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown trajectory '{name}'. "
                         f"Choose from: {list(TRAJECTORY_CLASSES)}")
    return cls()
