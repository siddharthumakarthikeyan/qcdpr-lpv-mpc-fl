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
from scipy.optimize import minimize
from config.parameters import UGV_HORIZON, UGV_MAX_VEL, UGV_MAX_ANG_VEL, DT

_Q = np.diag([300.0, 300.0, 20.0])
_R = np.diag([0.001, 0.001])


class MPCUGVController:
    def __init__(self, horizon: int = UGV_HORIZON):
        self.N  = horizon
        self._u_warm = np.zeros((horizon, 2))
        self._need_extra_iters = False

    def _cost_and_grad(self,
                       U_flat: np.ndarray,
                       x0:     np.ndarray,
                       X_ref:  np.ndarray) -> tuple:
        N, dt = self.N, DT
        U = U_flat.reshape(N, 2)

        X    = np.empty((N + 1, 3))
        X[0] = x0
        cpsi = np.empty(N)
        spsi = np.empty(N)
        for k in range(N):
            v, w     = U[k, 0], U[k, 1]
            psi      = X[k, 2]
            cpsi[k]  = np.cos(psi)
            spsi[k]  = np.sin(psi)
            X[k+1, 0] = X[k, 0] + v * cpsi[k] * dt
            X[k+1, 1] = X[k, 1] + v * spsi[k] * dt
            X[k+1, 2] = psi + w * dt

        J = 0.0
        D = np.empty((N, 3))
        for k in range(N):
            e    = X[k+1] - X_ref[k]
            e[2] = np.arctan2(np.sin(e[2]), np.cos(e[2]))
            Qe   = _Q @ e
            D[k] = 2.0 * Qe
            J   += float(e @ Qe) + float(U[k] @ (_R @ U[k]))

        dJ_dU = np.empty((N, 2))
        lam   = np.zeros(3)
        for k in range(N - 1, -1, -1):
            v   = U[k, 0]
            mu  = lam + D[k]
            dJ_dU[k, 0] = cpsi[k]*dt*mu[0] + spsi[k]*dt*mu[1] + 2.0*_R[0,0]*U[k,0]
            dJ_dU[k, 1] = dt*mu[2]                              + 2.0*_R[1,1]*U[k,1]
            lam = np.array([
                mu[0],
                mu[1],
                -v*spsi[k]*dt*mu[0] + v*cpsi[k]*dt*mu[1] + mu[2],
            ])

        return J, dJ_dU.flatten()

    def compute(self,
                state:      np.ndarray,
                ref_pos_xy: np.ndarray,
                ref_vel_xy: np.ndarray,
                ref_acc_xy: np.ndarray | None = None,
                ref_psi:    float = 0.0,
                ref_dpsi:   float = 0.0) -> np.ndarray:
        x0    = state[:3].copy()
        v_ref = float(np.linalg.norm(ref_vel_xy))
        acc   = ref_acc_xy if ref_acc_xy is not None else np.zeros(2)

        pos_err      = ref_pos_xy - x0[:2]
        pos_err_norm = float(np.linalg.norm(pos_err))
        dir_to_tgt   = float(np.arctan2(pos_err[1], pos_err[0]))
        alpha        = float(np.clip(pos_err_norm / 0.25, 0.0, 1.0))

        X_ref = np.empty((self.N, 3))
        for k in range(self.N):
            t_k       = (k + 1) * DT
            vel_psi_k = ref_psi + ref_dpsi * t_k
            blend_psi = float(np.arctan2(
                alpha * np.sin(dir_to_tgt) + (1 - alpha) * np.sin(vel_psi_k),
                alpha * np.cos(dir_to_tgt) + (1 - alpha) * np.cos(vel_psi_k),
            ))
            X_ref[k, 0] = ref_pos_xy[0] + t_k*ref_vel_xy[0] + 0.5*acc[0]*t_k**2
            X_ref[k, 1] = ref_pos_xy[1] + t_k*ref_vel_xy[1] + 0.5*acc[1]*t_k**2
            X_ref[k, 2] = blend_psi

        self._need_extra_iters = False
        if pos_err_norm > 0.20 and abs(self._u_warm[0, 0]) < 0.02:
            he = float(np.arctan2(np.sin(dir_to_tgt - x0[2]),
                                   np.cos(dir_to_tgt - x0[2])))
            sign   = np.sign(he)
            n_turn = min(int(abs(he) / (UGV_MAX_ANG_VEL * DT)) + 2, self.N // 2)
            self._u_warm[:n_turn] = [0.0, sign * UGV_MAX_ANG_VEL]
            self._u_warm[n_turn:] = [UGV_MAX_VEL, 0.0]
            self._need_extra_iters = True

        bounds  = [(-UGV_MAX_VEL, UGV_MAX_VEL),
                   (-UGV_MAX_ANG_VEL, UGV_MAX_ANG_VEL)] * self.N
        maxiter = 200 if self._need_extra_iters else 60

        result = minimize(
            self._cost_and_grad,
            self._u_warm.flatten(),
            args=(x0, X_ref),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-6, 'gtol': 1e-5},
        )

        U_opt = result.x.reshape(self.N, 2)

        self._u_warm[:-1] = U_opt[1:]
        self._u_warm[-1]  = U_opt[-1]

        u0 = U_opt[0]
        return np.array([
            np.clip(u0[0], -UGV_MAX_VEL,    UGV_MAX_VEL),
            np.clip(u0[1], -UGV_MAX_ANG_VEL, UGV_MAX_ANG_VEL),
        ])
