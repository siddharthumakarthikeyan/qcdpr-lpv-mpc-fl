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

    def step(self,
             uav_states:    list,
             ugv_states:    list,
             ref_pose:      np.ndarray,
             ref_velocity:  np.ndarray,
             ref_accel:     np.ndarray,
             uav_cable_forces: list | None = None,
             actual_platform_pose: np.ndarray | None = None) -> tuple:
        uav_targets, ugv_targets, corners_top, corners_bot = \
            compute_robot_targets(ref_pose)

        ref_pose_next = ref_pose.copy()
        ref_pose_next[:6] += ref_velocity[:6] * DT
        _, ugv_targets_next, _, _ = compute_robot_targets(ref_pose_next)
        ugv_vel_targets = (ugv_targets_next - ugv_targets) / DT

        ref_pose_prev = ref_pose.copy()
        ref_pose_prev[:6] -= ref_velocity[:6] * DT
        _, ugv_targets_prev, _, _ = compute_robot_targets(ref_pose_prev)
        ugv_vel_prev = (ugv_targets - ugv_targets_prev) / DT
        ugv_acc_targets = (ugv_vel_targets - ugv_vel_prev) / DT

        uav_vel_targets = compute_desired_uav_velocity(ref_velocity, ref_pose)

        uav_controls = []
        ref_psi = ref_pose[5]
        for i in range(NUM_UAVS):
            ref_acc_i = np.array([ref_accel[0], ref_accel[1], ref_accel[2]])
            ctrl_i = self.uav_controllers[i].compute(
                state    = uav_states[i],
                ref_pos  = uav_targets[i],
                ref_vel  = uav_vel_targets[i],
                ref_acc  = ref_acc_i,
                ref_psi  = ref_psi,
                F_cable  = None,
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
