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
    CORNERS_TOP_BODY, CORNERS_BOT_BODY,
    CABLE_OUTWARD_OFFSET, CABLE_VERTICAL_UAV,
    PLATFORM_ZREF, NUM_UAVS, NUM_UGVS,
)
from dynamics.uav_dynamics import rotation_matrix


_OUTWARD_SIGNS = np.array([
    [ 1,  1],
    [-1,  1],
    [-1, -1],
    [ 1, -1],
], dtype=float)
_OUTWARD_UNIT = _OUTWARD_SIGNS / np.linalg.norm(_OUTWARD_SIGNS[0])


def platform_corners_world(platform_pose: np.ndarray) -> tuple:
    px, py, pz, phi, theta, psi = platform_pose
    R = rotation_matrix(phi, theta, psi)
    p_world = np.array([px, py, pz])

    corners_top = (R @ CORNERS_TOP_BODY.T).T + p_world
    corners_bot = (R @ CORNERS_BOT_BODY.T).T + p_world
    return corners_top, corners_bot


def compute_robot_targets(platform_pose: np.ndarray) -> tuple:
    corners_top, corners_bot = platform_corners_world(platform_pose)

    uav_targets = np.zeros((NUM_UAVS, 3))
    ugv_targets = np.zeros((NUM_UGVS, 2))

    for i in range(NUM_UAVS):
        out_body = np.array([_OUTWARD_SIGNS[i, 0],
                              _OUTWARD_SIGNS[i, 1],
                              0.0])
        phi, theta, psi = platform_pose[3], platform_pose[4], platform_pose[5]
        R = rotation_matrix(phi, theta, psi)
        out_world = R @ out_body
        out_world[2] = 0.0
        norm = np.linalg.norm(out_world)
        if norm > 1e-8:
            out_world /= norm

        cx, cy = corners_top[i, 0], corners_top[i, 1]
        uav_x = cx + CABLE_OUTWARD_OFFSET * out_world[0]
        uav_y = cy + CABLE_OUTWARD_OFFSET * out_world[1]
        uav_targets[i] = [uav_x, uav_y, CABLE_VERTICAL_UAV]

        bx, by = corners_bot[i, 0], corners_bot[i, 1]
        ugv_offset = 2.0 * CABLE_OUTWARD_OFFSET
        ugv_targets[i] = [
            bx + ugv_offset * out_world[0],
            by + ugv_offset * out_world[1],
        ]

    return uav_targets, ugv_targets, corners_top, corners_bot


def all_robot_positions_3d(uav_positions: np.ndarray,
                            ugv_positions: np.ndarray) -> np.ndarray:
    ugv_3d = np.column_stack([ugv_positions, np.zeros(len(ugv_positions))])
    return np.vstack([uav_positions, ugv_3d])


def all_corner_positions(corners_top: np.ndarray,
                          corners_bot: np.ndarray) -> np.ndarray:
    return np.vstack([corners_top, corners_bot])


def compute_desired_uav_velocity(platform_velocity: np.ndarray,
                                  platform_pose: np.ndarray) -> np.ndarray:
    vx, vy, vz = platform_velocity[0], platform_velocity[1], platform_velocity[2]
    omega_psi = platform_velocity[5]
    phi, theta, psi = platform_pose[3:6]
    R = rotation_matrix(phi, theta, psi)

    uav_velocities = np.zeros((4, 3))
    for i in range(4):
        r_body = CORNERS_TOP_BODY[i]
        omega_vec = np.array([platform_velocity[3],
                               platform_velocity[4],
                               platform_velocity[5]])
        v_corner_body = np.cross(omega_vec, r_body)
        v_corner_world = R @ v_corner_body + platform_velocity[:3]
        uav_velocities[i] = v_corner_world
    return uav_velocities
