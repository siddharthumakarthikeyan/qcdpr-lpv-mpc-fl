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
    UAV_MASS, G,
    UAV_HORIZON,
    FL_KP, FL_KD,
    UAV_MAX_THRUST, UAV_PHI_MAX, UAV_THETA_MAX,
    DT
)
from dynamics.uav_dynamics import lpv_linearise


TAU_MAX = 2.0
_EPS    = 1e-6
 # Position-controller acceleration limits.
MAX_ACC_XY = 8.0
MAX_ACC_Z  = 10.0
# Cable-acceleration feedforward limits.  The cable can pull a UAV with up
# to ~T_max/m_uav = 50/0.5 = 100 m/s^2; do NOT clip below that or the FL
# controller cannot cancel real cable load — cable forces propagate from
# the platform through the UAVs and we MUST compensate them in full.
MAX_CABLE_ACC_XY = 80.0
MAX_CABLE_ACC_Z  = 100.0


class FeedbackLinearisationPositionController:
    def __init__(self, kp: np.ndarray = FL_KP, kd: np.ndarray = FL_KD):
        self.kp = kp
        self.kd = kd

    def compute(self,
                ref_pos:    np.ndarray,
                ref_vel:    np.ndarray,
                ref_acc:    np.ndarray,
                state:      np.ndarray,
                psi_d:      float = 0.0,
                F_cable:    np.ndarray | None = None) -> tuple:
        pos = state[0:3]
        vel = state[3:6]

        e_pos = ref_pos - pos
        e_vel = ref_vel - vel
        a_cmd = ref_acc + self.kd * e_vel + self.kp * e_pos
        a_cmd[:2] = np.clip(a_cmd[:2], -MAX_ACC_XY, MAX_ACC_XY)
        a_cmd[2] = np.clip(a_cmd[2], -MAX_ACC_Z, MAX_ACC_Z)

        if F_cable is not None:
            cable_acc = F_cable / UAV_MASS
            cable_acc[:2] = np.clip(cable_acc[:2], -MAX_CABLE_ACC_XY, MAX_CABLE_ACC_XY)
            cable_acc[2] = np.clip(cable_acc[2], -MAX_CABLE_ACC_Z, MAX_CABLE_ACC_Z)
            a_eff = a_cmd - cable_acc
        else:
            a_eff = a_cmd
        a_eff[:2] = np.clip(a_eff[:2], -MAX_ACC_XY, MAX_ACC_XY)
        a_eff[2] = np.clip(a_eff[2], -MAX_ACC_Z, MAX_ACC_Z + MAX_CABLE_ACC_Z)

        az_total = a_eff[2] + G

        az_for_thrust = max(az_total, 0.2)

        cp_psi = np.cos(psi_d)
        sp_psi = np.sin(psi_d)
        attitude_denom = max(az_total, 0.5)
        theta_d = np.arctan2(
            a_eff[0] * cp_psi + a_eff[1] * sp_psi,
            attitude_denom + _EPS
        )
        theta_d = np.clip(theta_d, -UAV_THETA_MAX, UAV_THETA_MAX)

        phi_d = np.arctan2(
            a_eff[0] * sp_psi - a_eff[1] * cp_psi,
            attitude_denom + _EPS
        )
        phi_d   = np.clip(phi_d, -UAV_PHI_MAX, UAV_PHI_MAX)

        attitude_vertical = max(
            np.cos(phi_d) * np.cos(theta_d),
            np.cos(UAV_PHI_MAX) * np.cos(UAV_THETA_MAX),
        )
        T_d = UAV_MASS * az_for_thrust / attitude_vertical
        T_d = np.clip(T_d, 0.1, UAV_MAX_THRUST)

        return T_d, phi_d, theta_d


class LPVMPCAttitudeController:
    def __init__(self, horizon: int = UAV_HORIZON):
        self.N  = horizon
        self.nx = 6
        self.nu = 3
        self._u_mpc_prev = np.zeros(3)
        self._setup_solver()

    def _setup_solver(self):
        from scipy.linalg import solve_discrete_are
        from config.parameters import UAV_IXX, UAV_IYY, UAV_IZZ

        N, nx, nu = self.N, self.nx, self.nu
        nu_t = N * nu

        Q = np.diag([1200., 1200., 800., 90., 90., 56.])
        R = np.diag([0.10,  0.10,  0.16])
        P = np.diag([0.02,  0.02,  0.03])

        A_hov = np.eye(6)
        A_hov[0,3]=DT; A_hov[1,4]=DT; A_hov[2,5]=DT
        B_hov = np.zeros((6,3))
        B_hov[3,0]=DT/UAV_IXX; B_hov[4,1]=DT/UAV_IYY; B_hov[5,2]=DT/UAV_IZZ
        try:
            P_f = solve_discrete_are(A_hov, B_hov, Q, R)
        except Exception:
            P_f = 10.0 * Q

        Q_bar_T = np.zeros((N*nx, N*nx))
        for k in range(N-1):
            Q_bar_T[k*nx:(k+1)*nx, k*nx:(k+1)*nx] = Q
        Q_bar_T[(N-1)*nx:N*nx, (N-1)*nx:N*nx] = P_f

        self._Q_bar  = Q_bar_T
        self._R_bar  = np.kron(np.eye(N), R)
        self._P_att  = P

        D_P = np.zeros((nu_t, nu_t))
        for k in range(N):
            D_P[k*nu:(k+1)*nu, k*nu:(k+1)*nu] = np.eye(nu)
            if k > 0:
                D_P[k*nu:(k+1)*nu, (k-1)*nu:k*nu] = -np.eye(nu)
        P_bar = np.kron(np.eye(N), P)
        self._DP_T_Pbar_DP = D_P.T @ P_bar @ D_P

        self._H_fixed = self._R_bar + self._DP_T_Pbar_DP

        P_init = sp.triu(np.ones((nu_t, nu_t)), format='csc')
        P_coo  = P_init.tocoo()
        self._P_row = P_coo.row
        self._P_col = P_coo.col

        l_box = np.full(nu_t, -TAU_MAX)
        u_box = np.full(nu_t,  TAU_MAX)
        A_box = sp.eye(nu_t, format='csc')

        self._solver = osqp.OSQP()
        self._solver.setup(
            P_init, np.zeros(nu_t), A_box, l_box, u_box,
            warm_starting=True, verbose=False,
            eps_abs=1e-4, eps_rel=1e-4, max_iter=6000,
            adaptive_rho=True, polish=False
        )
        self._nu_t = nu_t

    def _lpv_attitude_matrices(self, full_state: np.ndarray,
                                full_ctrl: np.ndarray) -> tuple:
        state_sched = full_state.copy()
        state_sched[9:12] = np.clip(state_sched[9:12], -3.0, 3.0)

        A_c, B_c, _ = lpv_linearise(state_sched, full_ctrl)
        A_att_c = A_c[6:12, 6:12]
        B_att_c = B_c[6:12, 1:4]
        A_att_d = np.eye(6) + A_att_c * DT + 0.5 * (A_att_c @ A_att_c) * DT**2
        B_att_d = B_att_c * DT
        return A_att_d, B_att_d

    def _build_prediction(self, A_d: np.ndarray,
                           B_d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        N, nx, nu = self.N, self.nx, self.nu
        A_pows = [np.eye(nx)]
        for _ in range(N):
            A_pows.append(A_pows[-1] @ A_d)

        Phi   = np.zeros((N * nx, nx))
        Gamma = np.zeros((N * nx, N * nu))
        for k in range(1, N + 1):
            Phi[(k-1)*nx:k*nx, :] = A_pows[k]
            for j in range(k):
                Gamma[(k-1)*nx:k*nx, j*nu:(j+1)*nu] = A_pows[k-1-j] @ B_d
        return Phi, Gamma

    def compute(self,
                full_state:  np.ndarray,
                full_ctrl:   np.ndarray,
                ref_phi:     float,
                ref_theta:   float,
                ref_psi:     float) -> np.ndarray:
        A_d, B_d = self._lpv_attitude_matrices(full_state, full_ctrl)

        x0     = full_state[6:12]
        x_ref  = np.array([ref_phi, ref_theta, ref_psi, 0.0, 0.0, 0.0])
        u_prev = full_ctrl[1:4]

        Phi, Gamma = self._build_prediction(A_d, B_d)

        if not (np.all(np.isfinite(Phi)) and np.all(np.isfinite(Gamma))):
            return np.clip(self._u_mpc_prev, -TAU_MAX, TAU_MAX)

        X_ref_stacked = np.tile(x_ref, self.N)
        E0            = Phi @ x0 - X_ref_stacked

        GtQ  = Gamma.T @ self._Q_bar
        H    = GtQ @ Gamma + self._H_fixed
        P_qp = 2.0 * H

        q_qp           = 2.0 * (GtQ @ E0)
        q_qp[:self.nu] += -2.0 * (self._P_att @ u_prev)

        if not (np.all(np.isfinite(P_qp)) and np.all(np.isfinite(q_qp))):
            return np.clip(self._u_mpc_prev, -TAU_MAX, TAU_MAX)

        Px_new = P_qp[self._P_row, self._P_col]
        self._solver.update(Px=Px_new, q=q_qp)

        result = self._solver.solve()
        if result.info.status in ('solved', 'solved_inaccurate') and result.x is not None:
            u_opt = result.x[:self.nu]
            if np.all(np.isfinite(u_opt)):
                self._u_mpc_prev = u_opt.copy()
                return np.clip(u_opt, -TAU_MAX, TAU_MAX)

        return np.clip(self._u_mpc_prev, -TAU_MAX, TAU_MAX)


class LPVMPCFLController:
    def __init__(self):
        self.fl  = FeedbackLinearisationPositionController()
        self.mpc = LPVMPCAttitudeController()
        self._T_prev   = UAV_MASS * G
        self._tau_prev = np.zeros(3)

    def compute(self,
                state:     np.ndarray,
                ref_pos:   np.ndarray,
                ref_vel:   np.ndarray,
                ref_acc:   np.ndarray,
                ref_psi:   float = 0.0,
                F_cable:   np.ndarray | None = None) -> np.ndarray:
        T_d, phi_d, theta_d = self.fl.compute(
            ref_pos, ref_vel, ref_acc, state, ref_psi, F_cable
        )

        prev_ctrl = np.array([self._T_prev, *self._tau_prev])
        tau = self.mpc.compute(state, prev_ctrl, phi_d, theta_d, ref_psi)

        self._T_prev   = T_d
        self._tau_prev = tau

        return np.array([T_d, tau[0], tau[1], tau[2]])
