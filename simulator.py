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
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from config.parameters import (
    NUM_UAVS, NUM_UGVS,
    PLATFORM_ZREF, DT, T_SIM,
    UAV_MASS, G, UAV_MAX_THRUST, UAV_PHI_MAX, UAV_THETA_MAX,
    PLATFORM_MASS, PLATFORM_IZZ,
    PLAT_KP, PLAT_KD, PLAT_KP_YAW, PLAT_KD_YAW,
)
from dynamics.uav_dynamics      import UAVState, step_rk4 as uav_step
from dynamics.ugv_dynamics      import UGVState, step_rk4 as ugv_step
from dynamics.platform_dynamics import PlatformState, step_rk4 as platform_step
from dynamics.uav_dynamics      import rotation_matrix
from kinematics.inverse_kinematics import (
    compute_robot_targets,
    all_robot_positions_3d,
    all_corner_positions,
    platform_corners_world,
)
from tension.tension_solver import (
    solve_tensions,
    cable_forces_on_platform,
    cable_reaction_on_robot,
    reset_tension_solver,
)
from controllers.centralized import CentralizedController
from estimation.ekf_estimator import PlatformEKF

_Monitor = None
def _get_monitor_class():
    global _Monitor
    if _Monitor is None:
        from visualization.monitor import RealTimeMonitor
        _Monitor = RealTimeMonitor
    return _Monitor


def payload_change_scenario(t_change: float = 30.0,
                             mass_before: float = PLATFORM_MASS,
                             mass_after:  float = PLATFORM_MASS + 2.0
                             ) -> Callable[[float], float]:
    def _fn(t: float) -> float:
        return mass_after if t >= t_change else mass_before
    return _fn


def nominal_scenario() -> Callable[[float], float]:
    def _fn(t: float) -> float:
        return PLATFORM_MASS
    return _fn


@dataclass
class SimLog:
    time:         List[float] = field(default_factory=list)
    platform_pos: List[np.ndarray] = field(default_factory=list)
    platform_vel: List[np.ndarray] = field(default_factory=list)
    platform_att: List[np.ndarray] = field(default_factory=list)
    ref_pos:      List[np.ndarray] = field(default_factory=list)
    uav_pos:      List[List[np.ndarray]] = field(default_factory=list)
    uav_att:      List[List[np.ndarray]] = field(default_factory=list)
    uav_ctrl:     List[List[np.ndarray]] = field(default_factory=list)
    uav_targets:  List[np.ndarray]       = field(default_factory=list)
    ugv_pos:      List[List[np.ndarray]] = field(default_factory=list)
    ugv_ctrl:     List[List[np.ndarray]] = field(default_factory=list)
    ugv_targets:  List[np.ndarray]       = field(default_factory=list)
    tensions:     List[np.ndarray] = field(default_factory=list)
    F_net_log:    List[np.ndarray] = field(default_factory=list)
    est_params:   List[np.ndarray] = field(default_factory=list)
    true_mass:    List[float]      = field(default_factory=list)
    label:        str              = ''
    step_times:   List[float] = field(default_factory=list)


def _initial_uav_positions(platform_pose: np.ndarray) -> np.ndarray:
    uav_targets, _, _, _ = compute_robot_targets(platform_pose)
    return uav_targets


def _initial_ugv_positions(platform_pose: np.ndarray) -> np.ndarray:
    _, ugv_targets, _, _ = compute_robot_targets(platform_pose)
    return ugv_targets


def run_simulation(trajectory,
                   t_sim:            float                       = T_SIM,
                   dt:               float                       = DT,
                   adaptive:         bool                        = False,
                   est_mode:         str                         = 'full',
                   true_mass_fn:     Optional[Callable[[float], float]] = None,
                   label:            str                         = '',
                   monitor=None,
                   monitor_interval: int                         = 20,
                   verbose:          bool                        = True) -> SimLog:
    if true_mass_fn is None:
        true_mass_fn = nominal_scenario()

    reset_tension_solver()

    t0_pose, t0_vel, t0_acc = trajectory(0.0)

    platform = PlatformState(t0_pose[:3], yaw=t0_pose[5])

    uav_init = _initial_uav_positions(t0_pose)
    ugv_init = _initial_ugv_positions(t0_pose)

    uavs = [UAVState(uav_init[i], yaw=t0_pose[5]) for i in range(NUM_UAVS)]
    ugv_init_yaw = (float(np.arctan2(t0_vel[1], t0_vel[0]))
                    if np.hypot(t0_vel[0], t0_vel[1]) > 0.01 else 0.0)
    ugvs = [UGVState(ugv_init[i], yaw=ugv_init_yaw) for i in range(NUM_UGVS)]

    for uav in uavs:
        uav.u = np.array([UAV_MASS * G, 0.0, 0.0, 0.0])

    controller = CentralizedController()
    log        = SimLog(label=label)

    estimate_disturbance = (est_mode == 'full')
    ekf: Optional[PlatformEKF] = (
        PlatformEKF(estimate_disturbance=estimate_disturbance)
        if adaptive else None
    )

    if not adaptive:
        mode_str = "NOMINAL"
    elif est_mode == 'mass_only':
        mode_str = "ADAPTIVE — mass only (ablation B)"
    else:
        mode_str = "ADAPTIVE — mass + disturbance (full EKF)"
    print(f"[Sim] Mode: {mode_str}")

    uav_pos_3d_init = np.array([u.position for u in uavs])
    ugv_pos_2d_init = np.array([g.position for g in ugvs])
    robot_pos_init  = all_robot_positions_3d(uav_pos_3d_init, ugv_pos_2d_init)
    ctop_init, cbot_init = platform_corners_world(t0_pose)
    corner_pos_init = all_corner_positions(ctop_init, cbot_init)
    tau_prev = solve_tensions(robot_pos_init, corner_pos_init,
                              platform_center=platform.position)
    cable_f_robots_init = cable_reaction_on_robot(tau_prev, robot_pos_init, corner_pos_init)
    uav_cable_forces_prev = [cable_f_robots_init[i].copy() for i in range(NUM_UAVS)]

    for i, uav in enumerate(uavs):
        f_c      = uav_cable_forces_prev[i]
        a_eff_z  = -f_c[2] / UAV_MASS
        a_eff_xy = np.array([-f_c[0], -f_c[1]]) / UAV_MASS
        az_total = a_eff_z + G
        T_init   = UAV_MASS * np.sqrt(a_eff_xy[0]**2 + a_eff_xy[1]**2 + az_total**2)
        uav.u    = np.array([np.clip(T_init, 0.1, UAV_MAX_THRUST), 0.0, 0.0, 0.0])

    vel_prev = platform.velocity.copy()

    n_steps = int(t_sim / dt)
    print(f"[Sim] Starting simulation: {n_steps} steps, dt={dt}s")

    for step in range(n_steps):
        t = step * dt
        t_wall_start = time.perf_counter()

        true_mass = true_mass_fn(t)

        ref_pose, ref_vel, ref_acc = trajectory(t)

        ctrl_mass = ekf.m_eff if adaptive else PLATFORM_MASS

        uav_tgt_nom, ugv_tgt_nom, ctop_nom, cbot_nom = compute_robot_targets(ref_pose)
        ugv_pos_2d_now = np.array([g.position for g in ugvs])
        robot_pos_nom  = all_robot_positions_3d(uav_tgt_nom, ugv_pos_2d_now)
        corner_pos_nom = all_corner_positions(ctop_nom, cbot_nom)
        tau_nom        = solve_tensions(robot_pos_nom, corner_pos_nom,
                                        platform_center=ref_pose[:3])
        cable_f_nom    = cable_reaction_on_robot(tau_nom, robot_pos_nom, corner_pos_nom)
        uav_cable_ff   = [cable_f_nom[i].copy() for i in range(NUM_UAVS)]

        uav_states = [u.x.copy() for u in uavs]
        ugv_states = [g.x.copy() for g in ugvs]

        (uav_ctrls, ugv_ctrls,
         uav_tgt, ugv_tgt,
         ctop, cbot) = controller.step(
             uav_states, ugv_states,
             ref_pose, ref_vel, ref_acc,
             uav_cable_forces      = uav_cable_ff,
             actual_platform_pose  = platform.pose,
         )

        plat_pose_actual           = platform.pose
        ctop_actual, cbot_actual   = platform_corners_world(plat_pose_actual)
        uav_pos_3d                 = np.array([u.position for u in uavs])
        ugv_pos_2d                 = np.array([g.position for g in ugvs])
        robot_pos                  = all_robot_positions_3d(uav_pos_3d, ugv_pos_2d)
        corner_pos                 = all_corner_positions(ctop_actual, cbot_actual)

        pos_err  = ref_pose[:3] - platform.position
        vel_err  = ref_vel[:3]  - platform.velocity
        F_track  = ctrl_mass * ref_acc[:3] + PLAT_KP * pos_err + PLAT_KD * vel_err
        F_track[2] += ctrl_mass * G

        yaw_err  = np.arctan2(np.sin(ref_pose[5] - plat_pose_actual[5]),
                               np.cos(ref_pose[5] - plat_pose_actual[5]))
        dyaw_err = ref_vel[5] - platform.x[11]
        tau_z_des = PLATFORM_IZZ * (ref_acc[5]
                                     + PLAT_KP_YAW * yaw_err
                                     + PLAT_KD_YAW * dyaw_err)

        F_ext_desired      = np.zeros(6)
        F_ext_desired[:3]  = F_track
        F_ext_desired[5]   = tau_z_des

        tau = solve_tensions(robot_pos, corner_pos,
                             platform_mass   = ctrl_mass,
                             platform_center = platform.position,
                             tau0            = tau_prev,
                             F_ext           = F_ext_desired)
        tau_prev = tau.copy()

        cable_f_plat   = cable_forces_on_platform(tau, robot_pos, corner_pos)
        cable_f_robots = cable_reaction_on_robot( tau, robot_pos, corner_pos)
        uav_cable_forces_prev = [cable_f_robots[i].copy() for i in range(NUM_UAVS)]

        F_net   = cable_f_plat.sum(axis=0)
        tau_net = np.zeros(3)
        plat_center = platform.position
        for ci in range(8):
            tau_net += np.cross(corner_pos[ci] - plat_center, cable_f_plat[ci])
        tau_net[:] = np.array([0.0, 0.0, tau_z_des])

        vel_before_step = platform.velocity.copy()
        next_platform_x = platform_step(platform.x, F_net, tau_net, mass=true_mass, dt=dt)
        if np.all(np.isfinite(next_platform_x)):
            platform.x = next_platform_x

        for i, uav in enumerate(uavs):
            uav.x = uav_step(uav.x, uav_ctrls[i], np.zeros(3), dt=dt)
            uav.u = uav_ctrls[i]

        for i, ugv in enumerate(ugvs):
            ugv.x = ugv_step(ugv.x, ugv_ctrls[i], np.zeros(2), dt=dt)

        if adaptive and ekf is not None:
            acc_measured = (platform.velocity - vel_before_step) / dt
            ekf.predict()
            ekf.update(F_cable   = F_net,
                       vel       = vel_before_step,
                       acc       = acc_measured)

        t_wall_end = time.perf_counter()

        if monitor is not None and step % monitor_interval == 0:
            uav_pos_arr = np.array([u.position for u in uavs])
            uav_att_arr = np.array([u.attitude  for u in uavs])
            ugv_pos_arr = np.array([g.position  for g in ugvs])
            ugv_hdg_arr = np.array([g.x[2]      for g in ugvs])
            monitor.update(
                t             = t,
                platform_pos  = platform.position,
                platform_att  = platform.attitude,
                ref_pos       = ref_pose[:3],
                uav_positions = uav_pos_arr,
                uav_attitudes = uav_att_arr,
                ugv_positions = ugv_pos_arr,
                ugv_headings  = ugv_hdg_arr,
                corners_top   = ctop_actual,
                corners_bot   = cbot_actual,
                tensions      = tau,
                step_time_ms  = 1000 * (t_wall_end - t_wall_start),
                true_mass     = true_mass,
                est_params    = ekf.state_vector if (adaptive and ekf) else None,
            )

        log.time.append(t)
        log.platform_pos.append(platform.position.copy())
        log.platform_vel.append(platform.velocity.copy())
        log.platform_att.append(platform.attitude.copy())
        log.ref_pos.append(ref_pose[:3].copy())
        log.uav_pos.append([u.position.copy() for u in uavs])
        log.uav_att.append([u.attitude.copy()  for u in uavs])
        log.uav_ctrl.append([c.copy() for c in uav_ctrls])
        log.uav_targets.append(uav_tgt.copy())
        log.ugv_pos.append([g.position.copy() for g in ugvs])
        log.ugv_ctrl.append([c.copy() for c in ugv_ctrls])
        log.ugv_targets.append(ugv_tgt.copy())
        log.tensions.append(tau.copy())
        log.F_net_log.append(F_net.copy())
        log.true_mass.append(true_mass)
        log.step_times.append(t_wall_end - t_wall_start)
        if adaptive and ekf is not None:
            log.est_params.append(ekf.state_vector)

        if verbose and (step % int(10 / dt) == 0):
            err = np.linalg.norm(platform.position - ref_pose[:3])
            m_str = f"  m_hat={ekf.m_eff:.2f}kg" if adaptive and ekf else ""
            print(f"  t={t:6.1f}s  pos_err={err:.4f}m  "
                  f"step_time={1000*(t_wall_end-t_wall_start):.1f}ms{m_str}")

    print("[Sim] Done.")
    return log


def compute_error_metrics(log: SimLog) -> dict:
    actual = np.array(log.platform_pos)
    ref    = np.array(log.ref_pos)
    err    = actual - ref

    metrics = {}
    for i, axis in enumerate(['x', 'y', 'z']):
        e = err[:, i]
        metrics[f'MSE_{axis}'] = float(np.mean(e**2))
        metrics[f'MAE_{axis}'] = float(np.mean(np.abs(e)))
        metrics[f'SSE_{axis}'] = float(np.sum(e**2))
    return metrics
