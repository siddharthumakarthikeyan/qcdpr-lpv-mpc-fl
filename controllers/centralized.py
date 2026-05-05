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
from config.parameters import NUM_UAVS, NUM_UGVS, DT
from kinematics.inverse_kinematics import (
    compute_robot_targets,
    compute_desired_uav_velocity,
)
from controllers.lpv_mpc_fl_uav import LPVMPCFLController
from controllers.mpc_ugv import MPCUGVController


class CentralizedController:
    def __init__(self):
        self.uav_controllers = [LPVMPCFLController() for _ in range(NUM_UAVS)]
        self.ugv_controllers = [MPCUGVController()   for _ in range(NUM_UGVS)]
        self._ik_yaw = None         # set on first step() call

    def step(self,
             uav_states:    list,
             ugv_states:    list,
             ref_pose:      np.ndarray,
             ref_velocity:  np.ndarray,
             ref_accel:     np.ndarray,
             uav_cable_forces: list | None = None,
             actual_platform_pose: np.ndarray | None = None) -> tuple:
        # Build the IK reference pose directly from the trajectory.  With
        # the unsymmetric (rank-6) cable layout (CABLE_SKEW_ANGLE_DEG > 0)
        # yaw is actively controllable through the cables, so we no
        # longer force psi = 0 in the IK frame.  Roll/pitch are passed
        # through as well \u2014 trajectories currently command zero
        # roll/pitch, but the pipeline now supports non-zero attitude
        # references.
        ik_pose = ref_pose.copy()

        uav_targets, ugv_targets, corners_top, corners_bot = \
            compute_robot_targets(ik_pose)

        ik_vel = ref_velocity.copy()

        # Finite-difference UGV targets so the velocity / acceleration
        # feedforward includes the yaw-rate-induced motion of the
        # rotating IK frame (essential for circular/sinusoidal where
        # dpsi != 0).  UAV velocity uses the analytic body-rate Jacobian
        # via compute_desired_uav_velocity().
        ref_pose_next = ik_pose.copy()
        ref_pose_next[:3] += ik_vel[:3] * DT
        ref_pose_next[3:6] += ik_vel[3:6] * DT
        _, ugv_targets_next, _, _ = compute_robot_targets(ref_pose_next)
        ugv_vel_targets = (ugv_targets_next - ugv_targets) / DT

        ref_pose_prev = ik_pose.copy()
        ref_pose_prev[:3] -= ik_vel[:3] * DT
        ref_pose_prev[3:6] -= ik_vel[3:6] * DT
        _, ugv_targets_prev, _, _ = compute_robot_targets(ref_pose_prev)
        ugv_vel_prev = (ugv_targets - ugv_targets_prev) / DT
        ugv_acc_targets = (ugv_vel_targets - ugv_vel_prev) / DT

        uav_vel_targets = compute_desired_uav_velocity(ik_vel, ik_pose)

        uav_controls = []
        # The UAV body yaw is left at zero \u2014 only the UAV *position*
        # tracks the rotating IK frame; rotating the UAV body itself
        # would only burn control effort with no benefit.
        ref_psi = 0.0
        for i in range(NUM_UAVS):
            ref_acc_i = np.array([ref_accel[0], ref_accel[1], ref_accel[2]])
            f_cable_ff = (uav_cable_forces[i]
                          if uav_cable_forces is not None else None)
            ctrl_i = self.uav_controllers[i].compute(
                state    = uav_states[i],
                ref_pos  = uav_targets[i],
                ref_vel  = uav_vel_targets[i],
                ref_acc  = ref_acc_i,
                ref_psi  = ref_psi,
                F_cable  = f_cable_ff,
            )
            uav_controls.append(ctrl_i)

        ugv_controls = []
        ugv_dpsi = float(ref_velocity[5])
        for i in range(NUM_UGVS):
            vel_i      = ugv_vel_targets[i]
            vel_norm   = float(np.linalg.norm(vel_i))
            pos_err_i  = ugv_targets[i] - ugv_states[i][:2]
            pos_err_norm = float(np.linalg.norm(pos_err_i))

            if vel_norm > 0.01:
                desired_heading = float(np.arctan2(vel_i[1], vel_i[0]))
            else:
                desired_heading = float(ugv_states[i][2])
            ctrl_i = self.ugv_controllers[i].compute(
                state      = ugv_states[i],
                ref_pos_xy = ugv_targets[i],
                ref_vel_xy = vel_i,
                ref_acc_xy = ugv_acc_targets[i],
                ref_psi    = desired_heading,
                ref_dpsi   = ugv_dpsi,
            )
            ugv_controls.append(ctrl_i)

        return (uav_controls, ugv_controls,
                uav_targets, ugv_targets,
                corners_top, corners_bot)
