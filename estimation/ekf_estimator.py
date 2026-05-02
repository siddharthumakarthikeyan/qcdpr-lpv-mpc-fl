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
from __future__ import annotations
import numpy as np
from config.parameters import G, PLATFORM_MASS

M_MIN, M_MAX = 0.5,  15.0
DIST_MAX     = 30.0

_Q_DEFAULT = np.diag([
    1e-4,
    0.05,
    0.05,
    0.02,
])

_R_DEFAULT = np.diag([0.04, 0.04, 0.04])

_ALPHA_DEFAULT = 0.95

_DM_MAX_DEFAULT = 0.30


class PlatformEKF:
    def __init__(self,
                 m_init:               float      = PLATFORM_MASS,
                 estimate_disturbance: bool       = True,
                 alpha:                float      = _ALPHA_DEFAULT,
                 m_rate_limit:         float      = _DM_MAX_DEFAULT,
                 Q:                   np.ndarray = _Q_DEFAULT,
                 R:                   np.ndarray = _R_DEFAULT):

        self.estimate_disturbance = estimate_disturbance
        self.alpha        = float(np.clip(alpha, 0.0, 1.0))
        self.m_rate_limit = float(m_rate_limit)

        self.theta = np.array([m_init, 0.0, 0.0, 0.0])

        self.P = np.diag([0.25, 1.0, 1.0, 0.25])

        self.Q = Q.copy()
        self.R = R.copy()

        self._F = np.diag([1.0, self.alpha, self.alpha, self.alpha])

    def predict(self) -> None:
        self.theta[1:4] *= self.alpha
        self.P = self._F @ self.P @ self._F.T + self.Q

    def update(self,
               F_cable: np.ndarray,
               vel:     np.ndarray,
               acc:     np.ndarray) -> None:
        if not (np.all(np.isfinite(F_cable)) and np.all(np.isfinite(acc))):
            return

        m, dx, dy, dz = self.theta
        Fx, Fy, Fz   = F_cable

        hx = Fx / m
        hy = Fy / m
        hz = Fz / m - G
        h  = np.array([hx, hy, hz])

        nu = acc - h

        im = 1.0 / m
        H = np.array([
            [-hx     * im,  im,  0.0,  0.0],
            [-hy     * im,  0.0, im,   0.0],
            [-(hz+G) * im,  0.0, 0.0,  im ],
        ])

        if not self.estimate_disturbance:
            H[:, 1:4] = 0.0

        H_mass = H.copy()
        H_mass[:, 1:4] = 0.0

        S = H_mass @ self.P @ H_mass.T + self.R
        K = self.P @ H_mass.T @ np.linalg.inv(S)

        m_prev = self.theta[0]
        self.theta = self.theta + K @ nu

        self.theta[0] = np.clip(self.theta[0],
                                m_prev - self.m_rate_limit,
                                m_prev + self.m_rate_limit)
        self.theta[0] = np.clip(self.theta[0], M_MIN, M_MAX)

        if self.estimate_disturbance:
            m_hat = self.theta[0]
            h_corrected = np.array([F_cable[0] / m_hat,
                                    F_cable[1] / m_hat,
                                    F_cable[2] / m_hat - G])
            d_raw = m_hat * (acc - h_corrected)
            self.theta[1:4] = (self.alpha * self.theta[1:4]
                               + (1.0 - self.alpha) * d_raw)
            self.theta[1:4] = np.clip(self.theta[1:4], -DIST_MAX, DIST_MAX)
        else:
            self.theta[1:4] = 0.0

        I_KH = np.eye(4) - K @ H_mass
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

    @property
    def m_eff(self) -> float:
        return float(self.theta[0])

    @property
    def disturbance(self) -> np.ndarray:
        return self.theta[1:4].copy()

    @property
    def state_vector(self) -> np.ndarray:
        return self.theta.copy()

    def convergence_time(self, time_arr: np.ndarray,
                         m_hat_arr: np.ndarray,
                         true_mass: float,
                         tol: float = 0.05) -> float:
        band = tol * true_mass
        within = np.abs(m_hat_arr - true_mass) <= band
        idx    = np.argmax(within)
        return float(time_arr[idx]) if within[idx] else float('nan')
