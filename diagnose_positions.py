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

from simulator import run_simulation
from trajectories import get_trajectory


def main():
    dt = 0.05
    log = run_simulation(
        get_trajectory("quadrilateral"),
        t_sim=30.0,
        dt=dt,
        verbose=False,
    )

    uav_pos = np.array(log.uav_pos)
    uav_targets = np.array(log.uav_targets)
    ugv_pos = np.array(log.ugv_pos)
    ugv_targets = np.array(log.ugv_targets)
    platform_pos = np.array(log.platform_pos)
    ref_pos = np.array(log.ref_pos)
    tensions = np.array(log.tensions)

    uav_target_err = np.linalg.norm(uav_pos - uav_targets, axis=2)
    ugv_target_err = np.linalg.norm(ugv_pos - ugv_targets, axis=2)
    uav_speed = np.linalg.norm(np.diff(uav_pos, axis=0) / dt, axis=2)
    ugv_speed = np.linalg.norm(np.diff(ugv_pos, axis=0) / dt, axis=2)
    platform_err = np.linalg.norm(platform_pos - ref_pos, axis=1)

    print("steps", len(log.time), "uav_shape", uav_pos.shape)
    print("finite_uav", bool(np.isfinite(uav_pos).all()))
    print("finite_platform", bool(np.isfinite(platform_pos).all()))
    print("finite_tensions", bool(np.isfinite(tensions).all()))
    print("platform_err_max", float(np.max(platform_err)))
    print("platform_err_final", float(platform_err[-1]))
    print("uav_target_err_max_by_uav", np.max(uav_target_err, axis=0))
    print("uav_target_err_mean_by_uav", np.mean(uav_target_err, axis=0))
    print("ugv_target_err_max_by_ugv", np.max(ugv_target_err, axis=0))
    print("ugv_target_err_mean_by_ugv", np.mean(ugv_target_err, axis=0))
    print("uav_speed_max_by_uav", np.max(uav_speed, axis=0))
    print("ugv_speed_max_by_ugv", np.max(ugv_speed, axis=0))
    print("uav_xyz_min")
    print(np.min(uav_pos, axis=0))
    print("uav_xyz_max")
    print(np.max(uav_pos, axis=0))
    print("tension_minmax", float(np.min(tensions)), float(np.max(tensions)))
    print("final_uav_pos")
    print(uav_pos[-1])
    print("final_ugv_pos")
    print(ugv_pos[-1])
    print("final_uav_targets")
    print(uav_targets[-1])
    print("final_ugv_targets")
    print(ugv_targets[-1])


if __name__ == "__main__":
    main()
