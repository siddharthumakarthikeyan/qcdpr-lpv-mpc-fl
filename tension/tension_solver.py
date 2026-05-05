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
import osqp
import scipy.sparse as sp
from config.parameters import (
    CABLE_MIN_TENSION, CABLE_MAX_TENSION,
    PLATFORM_MASS, G,
    TENSION_MAX_ITER, TENSION_TOL, TENSION_ALPHA_MIN
)

MU_REG         = 1e-2
# Wrench tracking weights.  All six rows of the wrench equation are
# weighted equally; the rank-6 cable layout (CABLE_SKEW_ANGLE_DEG > 0
# with alternating sign across corners) makes the yaw row reachable so
# no special yaw weighting is required.
WRENCH_W_FORCE  = 5e4
WRENCH_W_TORQUE = 5e4
WRENCH_W_YAW    = 5e4
WRENCH_PENALTY = 5e4   # legacy alias
RATE_PENALTY   = 50.0
DELTA_TAU_MAX  = 8.0


class _TensionQPSolver:
    def __init__(self, n: int = 8):
        self.n = n

        P_init = sp.triu(np.ones((n, n)), format='csc')
        P_coo  = P_init.tocoo()
        self._P_row = P_coo.row
        self._P_col = P_coo.col

        A_box  = sp.eye(n, format='csc')
        l_init = np.full(n, CABLE_MIN_TENSION)
        u_init = np.full(n, CABLE_MAX_TENSION)

        self._solver = osqp.OSQP()
        self._solver.setup(
            P_init, np.zeros(n), A_box, l_init, u_init,
            warm_starting=True,
            verbose=False,
            eps_abs=1e-5, eps_rel=1e-5,
            max_iter=10000,
            adaptive_rho=True, polish=False
        )

    def solve(self, W: np.ndarray, F_ext: np.ndarray,
              tau0: np.ndarray,
              rate_limited: bool = True) -> np.ndarray | None:
        if not np.all(np.isfinite(F_ext)):
            return None

        n = self.n
        # Diagonal wrench weighting: forces, roll/pitch torques, and yaw
        # torque can each have their own weight (yaw is up-weighted to
        # compensate for the smaller cable moment arm in that direction).
        Wt_diag = np.array([WRENCH_W_FORCE]*3
                           + [WRENCH_W_TORQUE, WRENCH_W_TORQUE, WRENCH_W_YAW])
        Wt_W    = W.T @ (Wt_diag[:, None] * W)

        P = (1.0 + 2.0 * RATE_PENALTY) * np.eye(n) + 2.0 * Wt_W

        q = -2.0 * (W.T @ (Wt_diag * F_ext)) - 2.0 * RATE_PENALTY * tau0

        if rate_limited:
            l = np.maximum(CABLE_MIN_TENSION, tau0 - DELTA_TAU_MAX)
            u = np.minimum(CABLE_MAX_TENSION, tau0 + DELTA_TAU_MAX)
        else:
            l = np.full(n, CABLE_MIN_TENSION)
            u = np.full(n, CABLE_MAX_TENSION)

        Px_new = P[self._P_row, self._P_col]
        self._solver.update(Px=Px_new, q=q, l=l, u=u)

        result = self._solver.solve()
        if result.info.status in ('solved', 'solved_inaccurate'):
            val = result.x
            if val is not None and np.all(np.isfinite(val)):
                return np.clip(val, CABLE_MIN_TENSION, CABLE_MAX_TENSION)
        return None


_tension_qp = _TensionQPSolver(n=8)


def reset_tension_solver() -> None:
    global _tension_qp
    _tension_qp = _TensionQPSolver(n=8)


def build_wrench_matrix(robot_positions: np.ndarray,
                        corner_positions: np.ndarray,
                        platform_center: np.ndarray | None = None) -> np.ndarray:
    if platform_center is None:
        platform_center = corner_positions.mean(axis=0)

    n = robot_positions.shape[0]
    W = np.zeros((6, n))
    for i in range(n):
        diff = robot_positions[i] - corner_positions[i]
        norm = np.linalg.norm(diff)
        if norm < 1e-8:
            continue
        u   = diff / norm
        b_i = corner_positions[i] - platform_center
        W[0:3, i] = u
        W[3:6, i] = np.cross(b_i, u)
    return W


def solve_tensions(robot_positions: np.ndarray,
                   corner_positions: np.ndarray,
                   platform_mass:    float = PLATFORM_MASS,
                   platform_center:  np.ndarray | None = None,
                   tau0:             np.ndarray | None = None,
                   F_ext:            np.ndarray | None = None) -> np.ndarray:
    if platform_center is None:
        platform_center = corner_positions.mean(axis=0)
    if F_ext is None:
        F_ext = np.array([0.0, 0.0, platform_mass * G, 0.0, 0.0, 0.0])

    W = build_wrench_matrix(robot_positions, corner_positions, platform_center)
    n = robot_positions.shape[0]

    rate_limited = tau0 is not None
    if tau0 is None:
        tau0 = np.full(n, CABLE_MIN_TENSION)

    result = _tension_qp.solve(W, F_ext, tau0, rate_limited=rate_limited)
    if result is not None:
        return result

    return _newton_solve(W, F_ext, n, tau0)


def _newton_solve(W: np.ndarray,
                  F_ext: np.ndarray,
                  n: int,
                  tau0: np.ndarray | None) -> np.ndarray:
    if tau0 is None:
        tau = np.zeros(n)
        tau[:4] = PLATFORM_MASS * G / 4.0 / 0.845 + CABLE_MIN_TENSION
        tau[4:]  = CABLE_MIN_TENSION
    else:
        tau = tau0.copy()
    tau = np.clip(tau, CABLE_MIN_TENSION, CABLE_MAX_TENSION)

    WT = W.T
    H  = WT @ W + MU_REG * np.eye(n)

    for _ in range(TENSION_MAX_ITER):
        residual = W @ tau - F_ext
        grad     = WT @ residual + MU_REG * (tau - CABLE_MIN_TENSION)
        f_val    = (0.5 * (residual @ residual) +
                    0.5 * MU_REG * np.sum((tau - CABLE_MIN_TENSION)**2))

        if np.linalg.norm(grad) < TENSION_TOL:
            break

        try:
            p = np.linalg.solve(H, -grad)
        except np.linalg.LinAlgError:
            p = -grad

        alpha    = 1.0
        c_armijo = 1e-4
        while alpha > TENSION_ALPHA_MIN:
            tau_new = np.clip(tau + alpha * p,
                              CABLE_MIN_TENSION, CABLE_MAX_TENSION)
            res_new = W @ tau_new - F_ext
            f_new   = (0.5 * (res_new @ res_new) +
                       0.5 * MU_REG * np.sum((tau_new - CABLE_MIN_TENSION)**2))
            if f_new <= f_val + c_armijo * alpha * (grad @ p):
                break
            alpha *= 0.5

        tau = np.clip(tau + alpha * p,
                      CABLE_MIN_TENSION, CABLE_MAX_TENSION)
    return tau


def cable_forces_on_platform(tau:              np.ndarray,
                              robot_positions:  np.ndarray,
                              corner_positions: np.ndarray) -> np.ndarray:
    forces = np.zeros((8, 3))
    for i in range(8):
        diff = robot_positions[i] - corner_positions[i]
        norm = np.linalg.norm(diff)
        if norm < 1e-8:
            continue
        forces[i] = tau[i] * diff / norm
    return forces


def cable_reaction_on_robot(tau:              np.ndarray,
                             robot_positions:  np.ndarray,
                             corner_positions: np.ndarray) -> np.ndarray:
    return -cable_forces_on_platform(tau, robot_positions, corner_positions)
