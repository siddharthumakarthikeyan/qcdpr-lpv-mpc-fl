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
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Patch
from pathlib import Path


SAVE_DIR = Path(__file__).parent.parent / "results"
SAVE_DIR.mkdir(exist_ok=True)

COLOR_REF  = '#1f77b4'
COLOR_LPV  = '#d62728'
COLOR_UAV  = ['#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
COLOR_UGV  = ['#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
COLOR_PLAT = '#1f77b4'
COLOR_CABLE_UAV = '#888888'
COLOR_CABLE_UGV = '#bbbb00'


def _save(fig, name: str, dpi: int = 150):
    path = SAVE_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved -> {path}")


def plot_3d_trajectory(log, title: str = "RCDPR Trajectory", name: str = "traj_3d"):
    actual = np.array(log.platform_pos)
    ref    = np.array(log.ref_pos)

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot(ref[:, 0],    ref[:, 1],    ref[:, 2],
            '--', color=COLOR_REF, lw=1.5, label='Reference')
    ax.plot(actual[:, 0], actual[:, 1], actual[:, 2],
            '-',  color=COLOR_LPV, lw=1.2, label='LPV-MPC-FL')
    ax.scatter(*ref[0],    color='green', s=60, zorder=5, label='Start')
    ax.scatter(*ref[-1],   color='red',   s=60, zorder=5, label='End')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()

    all_xy = np.concatenate([actual[:, :2], ref[:, :2]], axis=0)
    cx = (all_xy[:, 0].min() + all_xy[:, 0].max()) / 2
    cy = (all_xy[:, 1].min() + all_xy[:, 1].max()) / 2
    cz = float(np.mean(ref[:, 2]))
    xr = all_xy[:, 0].max() - all_xy[:, 0].min()
    yr = all_xy[:, 1].max() - all_xy[:, 1].min()
    half = max((xr + yr) / 2, 1.0) / 2 + 0.5
    half = np.ceil(half)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(max(0.0, cz - half), cz + half)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.zaxis.set_major_locator(plt.MultipleLocator(1.0))

    fig.tight_layout()
    _save(fig, name)


def plot_top_view(log, title: str = "Top View", name: str = "traj_top"):
    actual = np.array(log.platform_pos)
    ref    = np.array(log.ref_pos)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(ref[:, 0],    ref[:, 1],    '--', color=COLOR_REF, lw=1.5, label='Reference')
    ax.plot(actual[:, 0], actual[:, 1], '-',  color=COLOR_LPV, lw=1.2, label='LPV-MPC-FL')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title(title); ax.legend(); ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    all_xy = np.concatenate([actual[:, :2], ref[:, :2]], axis=0)
    xmin = np.floor(all_xy[:, 0].min()) - 0.2
    xmax = np.ceil( all_xy[:, 0].max()) + 0.2
    ymin = np.floor(all_xy[:, 1].min()) - 0.2
    ymax = np.ceil( all_xy[:, 1].max()) + 0.2
    xr, yr = xmax - xmin, ymax - ymin
    if xr > yr:
        pad = (xr - yr) / 2; ymin -= pad; ymax += pad
    else:
        pad = (yr - xr) / 2; xmin -= pad; xmax += pad
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))

    fig.tight_layout()
    _save(fig, name)


def plot_position_velocity(log, name: str = "pos_vel"):
    actual = np.array(log.platform_pos)
    ref    = np.array(log.ref_pos)
    vel    = np.array(log.platform_vel)
    t      = np.array(log.time)
    ref_v  = np.gradient(ref, t, axis=0) if len(t) > 1 else vel

    axes_labels = ['X', 'Y', 'Z']
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    for i, lbl in enumerate(axes_labels):
        axes[i, 0].plot(t, ref[:, i],    '--', color=COLOR_REF, lw=1.5, label='Ref')
        axes[i, 0].plot(t, actual[:, i], '-',  color=COLOR_LPV, lw=1.2, label='LPV-MPC-FL')
        axes[i, 0].set_ylabel(f'{lbl} pos (m)')
        axes[i, 0].legend(loc='best', fontsize=8)
        axes[i, 0].grid(True, alpha=0.3)

        axes[i, 1].plot(t, ref_v[:, i], '--', color=COLOR_REF, lw=1.5, label='Ref vel')
        axes[i, 1].plot(t, vel[:, i],   '-',  color=COLOR_LPV, lw=1.2, label='Actual vel')
        axes[i, 1].set_ylabel(f'{lbl} vel (m/s)')
        axes[i, 1].legend(loc='best', fontsize=8)
        axes[i, 1].grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Time (s)')
    fig.suptitle('Position and Velocity Tracking – LPV-MPC-FL')
    fig.tight_layout()
    _save(fig, name)


def plot_cable_tensions(log, name: str = "tensions"):
    tensions = np.array(log.tensions)
    t        = np.array(log.time)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for i in range(4):
        axes[0].plot(t, tensions[:, i], label=f'UAV {i+1}', color=COLOR_UAV[i], lw=1.2)
    axes[0].set_ylabel('Tension (N)')
    axes[0].set_title('UAV Cable Tensions')
    axes[0].legend(ncol=4); axes[0].grid(True, alpha=0.3)

    for i in range(4):
        axes[1].plot(t, tensions[:, 4+i], label=f'UGV {i+1}', color=COLOR_UGV[i], lw=1.2)
    axes[1].set_ylabel('Tension (N)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_title('UGV Cable Tensions')
    axes[1].legend(ncol=4); axes[1].grid(True, alpha=0.3)

    fig.suptitle('Cable Tension Time History')
    fig.tight_layout()
    _save(fig, name)


def plot_tracking_error(log, name: str = "tracking_error"):
    actual = np.array(log.platform_pos)
    ref    = np.array(log.ref_pos)
    t      = np.array(log.time)
    err    = np.linalg.norm(actual - ref, axis=1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, err, color=COLOR_LPV, lw=1.5, label='LPV-MPC-FL')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Position error (m)')
    ax.set_title('End-Effector Tracking Error')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, name)


def plot_iteration_time(log, name: str = "iter_time"):
    t     = np.array(log.time)
    times = 1000 * np.array(log.step_times)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, times, color=COLOR_LPV, lw=0.8, alpha=0.7)
    ax.axhline(np.mean(times), color='k', ls='--', lw=1.5,
               label=f'Mean = {np.mean(times):.1f} ms')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Iteration time (ms)')
    ax.set_title('Computational Time per Step (LPV-MPC-FL)')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, name)


def plot_workspace_tension_map(name: str = "workspace_tension"):
    from tension.tension_solver import solve_tensions, build_wrench_matrix
    from config.parameters import (PLATFORM_MASS, CABLE_OUTWARD_OFFSET,
                                    CABLE_VERTICAL_UAV, CORNERS_TOP_BODY,
                                    CORNERS_BOT_BODY, PLATFORM_ZREF)
    from dynamics.uav_dynamics import rotation_matrix
    from kinematics.inverse_kinematics import compute_robot_targets

    resolution = 15
    xs = np.linspace(-2.0, 2.0, resolution)
    ys = np.linspace(-2.0, 2.0, resolution)
    Z  = np.zeros((resolution, resolution))

    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            pose = np.array([x, y, PLATFORM_ZREF, 0.0, 0.0, 0.0])
            uav_tgt, ugv_tgt, ctop, cbot = compute_robot_targets(pose)
            ugv_3d = np.column_stack([ugv_tgt, np.zeros(4)])
            rp = np.vstack([uav_tgt, ugv_3d])
            cp = np.vstack([ctop,    cbot   ])
            tau = solve_tensions(rp, cp)
            Z[iy, ix] = tau[:4].max()

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(xs, ys, Z, levels=20, cmap='hot_r')
    ax.contour( xs, ys, Z, levels=[32.0], colors='cyan', linewidths=2,
                linestyles='--')
    cb = fig.colorbar(cf); cb.set_label('Max UAV tension (N)')
    ax.set_xlabel('Platform X (m)'); ax.set_ylabel('Platform Y (m)')
    ax.set_title('Max UAV Cable Tension vs Platform Position (static, φ=θ=ψ=0)')
    ax.set_aspect('equal')
    fig.tight_layout()
    _save(fig, name)


def plot_tension_vs_mass(name: str = "tension_vs_mass"):
    from tension.tension_solver import solve_tensions
    from kinematics.inverse_kinematics import compute_robot_targets
    from config.parameters import PLATFORM_ZREF

    masses    = np.linspace(1.0, 5.0, 20)
    uav_mean  = []
    ugv_mean  = []
    pose = np.array([0.0, 0.0, PLATFORM_ZREF, 0.0, 0.0, 0.0])
    uav_tgt, ugv_tgt, ctop, cbot = compute_robot_targets(pose)
    ugv_3d = np.column_stack([ugv_tgt, np.zeros(4)])
    rp = np.vstack([uav_tgt, ugv_3d])
    cp = np.vstack([ctop,    cbot   ])

    for m in masses:
        tau = solve_tensions(rp, cp, platform_mass=m)
        uav_mean.append(tau[:4].mean())
        ugv_mean.append(tau[4:].mean())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(masses, uav_mean, '-o', color='#d62728', label='UAV (mean)')
    ax.plot(masses, ugv_mean, '-s', color='#1f77b4', label='UGV (mean)')
    ax.axvline(3.0, color='k', ls='--', lw=1.2, label='Chosen mass (3 kg)')
    ax.set_xlabel('Platform mass (kg)'); ax.set_ylabel('Mean cable tension (N)')
    ax.set_title('Static Cable Tension vs Payload Mass')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, name)


def plot_ekf_results(log_base, log_adap, trajectory_name: str = "trajectory"):
    tag = trajectory_name.lower().replace(" ", "_")
    t   = np.array(log_adap.time)

    if log_adap.est_params:
        params     = np.array(log_adap.est_params)
        true_mass  = np.array(log_adap.true_mass)
        m_hat      = params[:, 0]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, true_mass, '--', color='#1f77b4', lw=2.0, label='True mass')
        ax.plot(t, m_hat,      '-', color='#d62728', lw=1.5, label='EKF estimate')
        ax.axhline(true_mass[0],  color='grey', ls=':', lw=1.0)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Platform mass (kg)')
        ax.set_title('EKF — Online Mass Estimation')
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save(fig, f"{tag}_ekf_mass")

    if log_adap.est_params:
        params = np.array(log_adap.est_params)
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        labels = ['Disturbance d_x (N)', 'Disturbance d_y (N)', 'Disturbance d_z (N)']
        for i, (ax, lbl) in enumerate(zip(axes, labels)):
            ax.plot(t, params[:, i + 1], color=COLOR_LPV, lw=1.2)
            ax.axhline(0, color='grey', ls='--', lw=0.8)
            ax.set_ylabel(lbl); ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle('EKF — Estimated Disturbance Forces')
        fig.tight_layout()
        _save(fig, f"{tag}_ekf_disturbance")

    err_adap = np.linalg.norm(
        np.array(log_adap.platform_pos) - np.array(log_adap.ref_pos), axis=1)

    fig, ax = plt.subplots(figsize=(10, 4))
    if log_base is not None:
        err_base = np.linalg.norm(
            np.array(log_base.platform_pos) - np.array(log_base.ref_pos), axis=1)
        t_base = np.array(log_base.time)
        ax.plot(t_base, err_base, '-', color='#1f77b4', lw=1.5,
                label='Baseline (fixed model)', alpha=0.8)
    ax.plot(t, err_adap, '-', color='#d62728', lw=1.5,
            label='Adaptive (EKF)', alpha=0.8)

    true_mass_arr = np.array(log_adap.true_mass)
    if true_mass_arr.max() != true_mass_arr.min():
        t_change = t[np.argmax(np.diff(true_mass_arr) != 0) + 1]
        ax.axvline(t_change, color='orange', ls='--', lw=1.5,
                   label=f'Payload change (t={t_change:.0f} s)')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position error (m)')
    ax.set_title('Tracking Error — Baseline vs Adaptive LPV-MPC-FL')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, f"{tag}_ekf_error_comparison")

    if log_adap.est_params:
        params    = np.array(log_adap.est_params)
        m_hat     = params[:, 0]
        true_mass_arr = np.array(log_adap.true_mass)
        true_final = true_mass_arr[-1]
        band       = 0.05 * true_final
        within     = np.abs(m_hat - true_final) <= band
        if within.any():
            idx_conv = int(np.argmax(within))
            t_conv   = t[idx_conv]
            ss_err   = float(np.mean(np.abs(m_hat[idx_conv:] - true_final)))
            print(f"[EKF] Convergence to +-5%: t = {t_conv:.1f} s")
            print(f"[EKF] Steady-state mass error: {ss_err:.4f} kg")
        if log_base is not None:
            e_base = np.linalg.norm(
                np.array(log_base.platform_pos) - np.array(log_base.ref_pos), axis=1)
            e_adap_arr = np.linalg.norm(
                np.array(log_adap.platform_pos) - np.array(log_adap.ref_pos), axis=1)
            reduction = (e_base.mean() - e_adap_arr.mean()) / e_base.mean() * 100
            print(f"[EKF] Mean error reduction (adaptive vs baseline): {reduction:.1f}%")

    print(f"[Plot] EKF figures saved to ./results/")


def plot_acceleration_comparison(log_base, log_adap,
                                  trajectory_name: str = "trajectory"):
    from config.parameters import G as GRAV, PLATFORM_MASS as M_NOM
    tag = trajectory_name.lower().replace(" ", "_")

    t   = np.array(log_adap.time)
    dt  = t[1] - t[0]
    vel = np.array(log_adap.platform_vel)
    F   = np.array(log_adap.F_net_log)

    acc_meas = np.gradient(vel[:, 2], dt)

    true_mass_arr = np.array(log_adap.true_mass)
    acc_nominal   = F[:, 2] / M_NOM - GRAV

    params    = np.array(log_adap.est_params)
    m_hat_arr = params[:, 0]
    d_z_arr   = params[:, 3]
    acc_ekf   = (F[:, 2] + d_z_arr) / m_hat_arr - GRAV

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, acc_meas,    color='k',       lw=1.2, alpha=0.6,
            label='Measured acceleration (truth)')
    ax.plot(t, acc_nominal, color='#1f77b4', lw=1.5, ls='--',
            label=f'Fixed-model prediction (m = {M_NOM} kg)')
    ax.plot(t, acc_ekf,     color='#d62728', lw=1.2,
            label='EKF-model prediction (m = m̂)')

    if true_mass_arr.max() != true_mass_arr.min():
        t_ch = t[np.argmax(np.diff(true_mass_arr) != 0) + 1]
        ax.axvline(t_ch, color='orange', ls='--', lw=1.5,
                   label=f'Payload change (t={t_ch:.0f} s)')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Z acceleration (m/s²)')
    ax.set_title('Why Adaptive Works — Wrong Model → Wrong Acceleration Prediction')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, f"{tag}_ekf_acceleration_why")


def plot_ablation(logs: list, trajectory_name: str = "trajectory"):
    tag    = trajectory_name.lower().replace(" ", "_")
    colors = ['#1f77b4', '#ff7f0e', '#d62728']

    fig, ax = plt.subplots(figsize=(12, 5))

    draw_order = list(zip(logs, colors))
    draw_order = draw_order[1:] + [draw_order[0]]
    alphas     = [0.75, 0.75, 1.0]

    for (log, color), alpha in zip(draw_order, alphas):
        t   = np.array(log.time)
        err = np.linalg.norm(
            np.array(log.platform_pos) - np.array(log.ref_pos), axis=1)
        lbl = log.label if log.label else '—'
        ax.plot(t, err, color=color, lw=1.5, alpha=alpha, label=lbl)

    for log in logs:
        true_mass_arr = np.array(log.true_mass)
        if true_mass_arr.max() != true_mass_arr.min():
            t_arr = np.array(log.time)
            t_ch  = t_arr[np.argmax(np.diff(true_mass_arr) != 0) + 1]
            ax.axvline(t_ch, color='orange', ls='--', lw=1.5,
                       label=f'Payload change (t={t_ch:.0f} s)')
            break

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position error (m)')
    ax.set_title('Ablation Study — Contribution of Each Estimation Component')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, f"{tag}_ekf_ablation")

    print(f"\n[Ablation] {'Case':<35} {'X MAE':>10} {'Y MAE':>10} {'Z MAE':>10}")
    print(f"           {'-'*65}")
    for log in logs:
        act = np.array(log.platform_pos)
        ref = np.array(log.ref_pos)
        err = np.abs(act - ref)
        print(f"           {log.label:<35} "
              f"{1000*err[:,0].mean():>8.1f}mm "
              f"{1000*err[:,1].mean():>8.1f}mm "
              f"{1000*err[:,2].mean():>8.1f}mm")


def plot_platform_error_axes(log, name: str = "platform_error_axes"):
    actual = np.array(log.platform_pos)
    ref    = np.array(log.ref_pos)
    t      = np.array(log.time)
    err    = actual - ref
    norm   = np.linalg.norm(err, axis=1)

    axes_lbl = ['X error (m)', 'Y error (m)', 'Z error (m)', 'Total norm (m)']
    data     = [err[:, 0], err[:, 1], err[:, 2], norm]
    colors   = ['#e377c2', '#17becf', '#bcbd22', COLOR_LPV]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for ax, d, lbl, col in zip(axes, data, axes_lbl, colors):
        ax.plot(t, d, color=col, lw=1.2)
        ax.axhline(0, color='k', lw=0.6, ls='--')
        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.3)
        mae = np.mean(np.abs(d))
        ax.set_title(f'MAE = {mae*1000:.2f} mm', fontsize=8, loc='right')
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Platform Tracking Error — Per Axis')
    fig.tight_layout()
    _save(fig, name)


def plot_uav_tracking(log, name: str = "uav_tracking"):
    uav_pos = np.array(log.uav_pos)
    uav_tgt = np.array(log.uav_targets)
    t       = np.array(log.time)

    from matplotlib.lines import Line2D
    axes_lbl = ['X (m)', 'Y (m)', 'Z (m)']
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    for j, (ax, lbl) in enumerate(zip(axes, axes_lbl)):
        for i in range(4):
            ax.plot(t, uav_pos[:, i, j], color=COLOR_UAV[i], lw=1.0, alpha=0.9)
            ax.plot(t, uav_tgt[:, i, j], '--', color=COLOR_UAV[i], lw=0.8, alpha=0.45)
        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.3)
        if j == 0:
            handles  = [Line2D([0], [0], color=c, label=f'UAV {i+1}')
                        for i, c in enumerate(COLOR_UAV)]
            handles += [Line2D([0], [0], color='k', ls='--', alpha=0.6, label='IK target')]
            ax.legend(handles=handles, fontsize=7, ncol=5)

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('UAV Position Tracking — Actual (solid) vs IK Target (dashed)')
    fig.tight_layout()
    _save(fig, name)


def plot_uav_errors(log, name: str = "uav_errors"):
    from config.parameters import UAV_PHI_MAX, UAV_THETA_MAX, UAV_VEL_MAX
    from matplotlib.lines import Line2D

    uav_pos = np.array(log.uav_pos)
    uav_tgt = np.array(log.uav_targets)
    uav_att = np.array(log.uav_att)
    t       = np.array(log.time)
    dt      = float(t[1] - t[0]) if len(t) > 1 else 0.05

    err   = np.linalg.norm(uav_pos - uav_tgt, axis=2)
    speed = np.linalg.norm(np.diff(uav_pos, axis=0) / dt, axis=2)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    for i in range(4):
        axes[0].plot(t, err[:, i], color=COLOR_UAV[i], lw=1.0, label=f'UAV {i+1}')
    axes[0].axhline(0.40, color='r', ls='--', lw=1.2, label='0.40 m bound')
    axes[0].set_ylabel('Position error (m)')
    axes[0].legend(fontsize=7, ncol=5)
    axes[0].grid(True, alpha=0.3)

    lim_deg = np.rad2deg(UAV_PHI_MAX)
    for i in range(4):
        axes[1].plot(t, np.rad2deg(uav_att[:, i, 0]),
                     color=COLOR_UAV[i], lw=0.9, alpha=0.85)
        axes[1].plot(t, np.rad2deg(uav_att[:, i, 1]),
                     color=COLOR_UAV[i], lw=0.9, alpha=0.85, ls=':')
    axes[1].axhline( lim_deg, color='r', ls='--', lw=1.2, label=f'±{lim_deg:.0f}° limit')
    axes[1].axhline(-lim_deg, color='r', ls='--', lw=1.2)
    handles = ([Line2D([0], [0], color=c, label=f'UAV {i+1}')
                for i, c in enumerate(COLOR_UAV)]
               + [Line2D([0], [0], color='k', ls='-',  label='φ (roll)'),
                  Line2D([0], [0], color='k', ls=':',  label='θ (pitch)')])
    axes[1].legend(handles=handles, fontsize=7, ncol=6)
    axes[1].set_ylabel('Attitude (deg)')
    axes[1].grid(True, alpha=0.3)

    for i in range(4):
        axes[2].plot(t[:-1], speed[:, i], color=COLOR_UAV[i], lw=0.9,
                     alpha=0.85, label=f'UAV {i+1}')
    axes[2].axhline(UAV_VEL_MAX, color='r', ls='--', lw=1.2,
                    label=f'{UAV_VEL_MAX} m/s limit')
    axes[2].set_ylabel('Speed (m/s)')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend(fontsize=7, ncol=5)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('UAV Tracking Errors, Attitude Constraints, and Speed')
    fig.tight_layout()
    _save(fig, name)


def plot_ugv_tracking(log, name: str = "ugv_tracking"):
    ugv_pos = np.array(log.ugv_pos)
    ugv_tgt = np.array(log.ugv_targets)
    t       = np.array(log.time)

    from matplotlib.lines import Line2D
    axes_lbl = ['X (m)', 'Y (m)']
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    for j, (ax, lbl) in enumerate(zip(axes, axes_lbl)):
        for i in range(4):
            ax.plot(t, ugv_pos[:, i, j], color=COLOR_UGV[i], lw=1.0, alpha=0.9)
            ax.plot(t, ugv_tgt[:, i, j], '--', color=COLOR_UGV[i], lw=0.8, alpha=0.45)
        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.3)
        if j == 0:
            handles  = [Line2D([0], [0], color=c, label=f'UGV {i+1}')
                        for i, c in enumerate(COLOR_UGV)]
            handles += [Line2D([0], [0], color='k', ls='--', alpha=0.6, label='IK target')]
            ax.legend(handles=handles, fontsize=7, ncol=5)

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('UGV Position Tracking — Actual (solid) vs IK Target (dashed)')
    fig.tight_layout()
    _save(fig, name)


def plot_ugv_errors(log, name: str = "ugv_errors"):
    from config.parameters import UGV_MAX_VEL

    ugv_pos = np.array(log.ugv_pos)
    ugv_tgt = np.array(log.ugv_targets)
    t       = np.array(log.time)
    dt      = float(t[1] - t[0]) if len(t) > 1 else 0.05

    err   = np.linalg.norm(ugv_pos - ugv_tgt[:, :, :2], axis=2)
    speed = np.linalg.norm(np.diff(ugv_pos, axis=0) / dt, axis=2)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    for i in range(4):
        axes[0].plot(t, err[:, i], color=COLOR_UGV[i], lw=1.0, label=f'UGV {i+1}')
    axes[0].axhline(0.05, color='r', ls='--', lw=1.2, label='0.05 m bound')
    axes[0].set_ylabel('Position error (m)')
    axes[0].legend(fontsize=7, ncol=5)
    axes[0].grid(True, alpha=0.3)

    for i in range(4):
        axes[1].plot(t[:-1], speed[:, i], color=COLOR_UGV[i], lw=0.9,
                     alpha=0.85, label=f'UGV {i+1}')
    axes[1].axhline(UGV_MAX_VEL, color='r', ls='--', lw=1.2,
                    label=f'{UGV_MAX_VEL} m/s limit')
    axes[1].set_ylabel('Speed (m/s)')
    axes[1].set_xlabel('Time (s)')
    axes[1].legend(fontsize=7, ncol=5)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('UGV Tracking Errors and Speed')
    fig.tight_layout()
    _save(fig, name)


def plot_all(log, trajectory_name: str = "trajectory",
             save_animation: bool = False,
             animation_speed: float = 5.0,
             animation_fmt: str = 'mp4'):
    tag = trajectory_name.lower().replace(" ", "_")
    print(f"[Plot] Generating figures for '{tag}' ...")

    plot_3d_trajectory(log,         title=f"3D Trajectory – {trajectory_name}",  name=f"{tag}_3d")
    plot_top_view(log,              title=f"Top View – {trajectory_name}",        name=f"{tag}_top")
    plot_position_velocity(log,                                                    name=f"{tag}_pos_vel")
    plot_platform_error_axes(log,                                                  name=f"{tag}_platform_errors")
    plot_cable_tensions(log,                                                       name=f"{tag}_tensions")
    plot_tracking_error(log,                                                       name=f"{tag}_error")
    plot_iteration_time(log,                                                       name=f"{tag}_iter")

    plot_uav_tracking(log,                                                         name=f"{tag}_uav_tracking")
    plot_uav_errors(log,                                                           name=f"{tag}_uav_errors")

    plot_ugv_tracking(log,                                                         name=f"{tag}_ugv_tracking")
    plot_ugv_errors(log,                                                           name=f"{tag}_ugv_errors")

    print(f"[Plot] {11} figures saved to ./results/")

    if save_animation:
        print(f"[Plot] Generating {animation_fmt.upper()} animation from log ...")
        animate(log,
                trajectory_name=trajectory_name,
                speed_up=animation_speed,
                save_gif=(animation_fmt.lower() == 'gif'),
                save_mp4=(animation_fmt.lower() == 'mp4'),
                show=False)


_TRAJ_COLORS = {
    'quadrilateral': '#1f77b4',
    'circular':      '#d62728',
    'sinusoidal':    '#2ca02c',
}
_TRAJ_MARKERS = {'quadrilateral': 's', 'circular': 'o', 'sinusoidal': '^'}


def plot_multi_trajectory_overlay(logs: list, names: list,
                                   name: str = "all_trajectories_3d"):
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(12, 9))
    ax  = fig.add_subplot(111, projection='3d')

    all_pts = []
    for log, traj_name in zip(logs, names):
        c      = _TRAJ_COLORS.get(traj_name.lower(), '#333333')
        actual = np.array(log.platform_pos)
        ref    = np.array(log.ref_pos)
        ax.plot(ref[:,0],    ref[:,1],    ref[:,2],    '--', color=c, lw=1.2, alpha=0.45)
        ax.plot(actual[:,0], actual[:,1], actual[:,2], '-',  color=c, lw=1.8,
                label=traj_name.capitalize())
        ax.scatter(*ref[0],  color=c, s=60, marker='o', zorder=6)
        ax.scatter(*ref[-1], color=c, s=60, marker='*', zorder=6)
        all_pts.append(np.vstack([actual, ref]))

    pts = np.vstack(all_pts)
    pad = 0.4
    cx  = (pts[:,0].min() + pts[:,0].max()) / 2
    cy  = (pts[:,1].min() + pts[:,1].max()) / 2
    cz  = float(np.mean(pts[:,2]))
    half = max((pts[:,0].max()-pts[:,0].min()),
               (pts[:,1].max()-pts[:,1].min())) / 2 + pad
    ax.set_xlim(cx-half, cx+half); ax.set_ylim(cy-half, cy+half)
    ax.set_zlim(0, cz+half)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title('All Trajectories — Platform Paths')

    style_handles = [
        Line2D([0],[0], color='k', ls='-',  lw=1.8, label='Actual path'),
        Line2D([0],[0], color='k', ls='--', lw=1.2, alpha=0.5, label='Reference'),
    ]
    traj_handles = [
        Line2D([0],[0], color=_TRAJ_COLORS.get(n.lower(),'k'), lw=2, label=n.capitalize())
        for n in names
    ]
    ax.legend(handles=traj_handles + style_handles, fontsize=8, loc='upper left')
    fig.tight_layout()
    _save(fig, name)


def plot_error_metrics_bar(metrics_list: list, traj_names: list,
                            name: str = "error_metrics_bar"):
    axes_lbl = ['X', 'Y', 'Z']
    n_traj   = len(traj_names)
    x        = np.arange(n_traj)
    width    = 0.22
    colors   = [_TRAJ_COLORS.get(n.lower(), '#888') for n in traj_names]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    for j, (ax, lbl) in enumerate(zip(axes, axes_lbl)):
        vals = [1000 * m[f'MAE_{lbl.lower()}'] for m in metrics_list]
        bars = ax.bar(x, vals, width=0.55, color=colors, edgecolor='k', linewidth=0.6)
        ax.set_xticks(x); ax.set_xticklabels([n.capitalize() for n in traj_names],
                                               fontsize=8, rotation=10)
        ax.set_ylabel('MAE (mm)' if j == 0 else '')
        ax.set_title(f'{lbl}-axis MAE')
        ax.grid(True, axis='y', alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=7)

    fig.suptitle('Platform Tracking MAE — Trajectory Comparison', fontsize=12)
    fig.tight_layout()
    _save(fig, name)


def plot_tension_statistics(logs: list, names: list,
                             name: str = "tension_statistics"):
    from config.parameters import CABLE_MIN_TENSION, CABLE_MAX_TENSION
    cable_lbl_uav = [f'U{i+1}' for i in range(4)]
    cable_lbl_ugv = [f'G{i+1}' for i in range(4)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for row, (ax, indices, lbls, title) in enumerate(zip(
            axes,
            [range(4), range(4, 8)],
            [cable_lbl_uav, cable_lbl_ugv],
            ['UAV Cables (upper)', 'UGV Cables (lower)'])):

        positions, all_data, colors_bp, tick_lbls = [], [], [], []
        pos = 0
        for log, traj_name in zip(logs, names):
            tau = np.array(log.tensions)
            c   = _TRAJ_COLORS.get(traj_name.lower(), '#888')
            for ci in indices:
                positions.append(pos); all_data.append(tau[:, ci]); colors_bp.append(c)
                tick_lbls.append(f'{lbls[ci - (4 if ci >= 4 else 0)]}\n({traj_name[:4]})')
                pos += 1
            pos += 0.6

        bp = ax.boxplot(all_data, positions=positions, widths=0.55,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color='k', lw=1.5))
        for patch, c in zip(bp['boxes'], colors_bp):
            patch.set_facecolor(c); patch.set_alpha(0.65)

        ax.axhline(CABLE_MIN_TENSION, color='b', ls=':', lw=1.2,
                   label=f'Min ({CABLE_MIN_TENSION} N)')
        ax.axhline(CABLE_MAX_TENSION, color='r', ls='--', lw=1.2,
                   label=f'Max ({CABLE_MAX_TENSION} N)')
        ax.set_xticks(positions); ax.set_xticklabels(tick_lbls, fontsize=6.5)
        ax.set_ylabel('Tension (N)'); ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(True, axis='y', alpha=0.3)

        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=_TRAJ_COLORS.get(n.lower(),'k'),
                                  label=n.capitalize()) for n in names]
                  + [plt.Line2D([0],[0], color='b', ls=':', label=f'Min {CABLE_MIN_TENSION} N'),
                     plt.Line2D([0],[0], color='r', ls='--', label=f'Max {CABLE_MAX_TENSION} N')],
                  fontsize=7, loc='upper right')

    fig.suptitle('Cable Tension Distributions — All Trajectories', fontsize=12)
    fig.tight_layout()
    _save(fig, name)


def plot_phase_portrait(logs: list, names: list,
                         name: str = "tracking_phase_portrait"):
    fig, axes = plt.subplots(1, len(logs), figsize=(5 * len(logs), 5))
    if len(logs) == 1:
        axes = [axes]

    for ax, log, traj_name in zip(axes, logs, names):
        actual = np.array(log.platform_pos)
        ref    = np.array(log.ref_pos)
        t      = np.array(log.time)
        ex     = (actual[:,0] - ref[:,0]) * 1000
        ey     = (actual[:,1] - ref[:,1]) * 1000

        sc = ax.scatter(ex, ey, c=t, cmap='plasma', s=4, alpha=0.7, zorder=3)
        plt.colorbar(sc, ax=ax, label='Time (s)', shrink=0.8)
        ax.axhline(0, color='k', lw=0.7, ls='--'); ax.axvline(0, color='k', lw=0.7, ls='--')

        rms_x = float(np.sqrt(np.mean(ex**2)))
        rms_y = float(np.sqrt(np.mean(ey**2)))
        theta_circ = np.linspace(0, 2*np.pi, 200)
        r = max(rms_x, rms_y)
        ax.plot(r*np.cos(theta_circ), r*np.sin(theta_circ), 'r--', lw=1.0,
                label=f'RMS circle ({r:.1f} mm)')

        ax.set_xlabel('X error (mm)'); ax.set_ylabel('Y error (mm)')
        ax.set_title(f'{traj_name.capitalize()}')
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3); ax.legend(fontsize=7)

    fig.suptitle('Platform Tracking Error Phase Portrait (X vs Y)', fontsize=12)
    fig.tight_layout()
    _save(fig, name)


def plot_control_effort(logs: list, names: list,
                         name: str = "control_effort"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    thrust_data  = [np.array(log.uav_ctrl)[:,:,0].ravel() for log in logs]
    vp = axes[0].violinplot(thrust_data, positions=range(len(logs)),
                             showmedians=True, showextrema=True)
    for body, c in zip(vp['bodies'],
                        [_TRAJ_COLORS.get(n.lower(),'#888') for n in names]):
        body.set_facecolor(c); body.set_alpha(0.65)
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels([n.capitalize() for n in names], fontsize=9)
    axes[0].set_ylabel('Thrust T (N)')
    axes[0].set_title('UAV Thrust Distribution per Trajectory')
    axes[0].axhline(0.5 * 9.81, color='grey', ls='--', lw=1.0, label='Hover thrust')
    axes[0].legend(fontsize=8); axes[0].grid(True, axis='y', alpha=0.3)

    n_traj = len(logs)
    x      = np.arange(4)
    width  = 0.25
    for i, (log, traj_name) in enumerate(zip(logs, names)):
        thrust = np.array(log.uav_ctrl)[:,:,0]
        means  = thrust.mean(axis=0)
        stds   = thrust.std(axis=0)
        c      = _TRAJ_COLORS.get(traj_name.lower(), '#888')
        offset = (i - n_traj/2 + 0.5) * width
        axes[1].bar(x + offset, means, width, yerr=stds, color=c, alpha=0.75,
                    edgecolor='k', lw=0.5, capsize=3, label=traj_name.capitalize())

    axes[1].set_xticks(x); axes[1].set_xticklabels([f'UAV {i+1}' for i in range(4)])
    axes[1].set_ylabel('Mean Thrust (N)')
    axes[1].set_title('Per-UAV Mean Thrust (error bar = std)')
    axes[1].legend(fontsize=8); axes[1].grid(True, axis='y', alpha=0.3)

    fig.suptitle('UAV Control Effort Across Trajectories', fontsize=12)
    fig.tight_layout()
    _save(fig, name)


def plot_cooperative_workload(logs: list, names: list,
                               name: str = "cooperative_workload"):
    fig, axes = plt.subplots(len(logs), 1, figsize=(12, 3.5 * len(logs) + 0.5))
    if len(logs) == 1:
        axes = [axes]

    for ax, log, traj_name in zip(axes, logs, names):
        tau    = np.array(log.tensions)
        t      = np.array(log.time)
        mean_u = tau[:, :4].mean(axis=1)
        mean_g = tau[:, 4:].mean(axis=1)
        ax.plot(t, mean_u, color='#2ca02c', lw=1.4, label='UAV cables (mean)')
        ax.plot(t, mean_g, color='#ff7f0e', lw=1.4, label='UGV cables (mean)')
        ax.fill_between(t, tau[:,:4].min(axis=1), tau[:,:4].max(axis=1),
                        color='#2ca02c', alpha=0.15)
        ax.fill_between(t, tau[:,4:].min(axis=1), tau[:,4:].max(axis=1),
                        color='#ff7f0e', alpha=0.15)
        ax.set_ylabel('Tension (N)'); ax.set_title(f'{traj_name.capitalize()}')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time (s)')

    fig.suptitle('Cooperative Cable Load — UAV (upper) vs UGV (lower)', fontsize=12)
    fig.tight_layout()
    _save(fig, name)


def plot_performance_summary_table(metrics_list: list, traj_names: list,
                                    name: str = "performance_summary_table"):
    col_headers = ['X MSE', 'Y MSE', 'Z MSE',
                   'X MAE', 'Y MAE', 'Z MAE',
                   'X SSE', 'Y SSE', 'Z SSE']
    rows = []
    for m in metrics_list:
        rows.append([
            f"{m['MSE_x']:.6f}", f"{m['MSE_y']:.6f}", f"{m['MSE_z']:.6f}",
            f"{m['MAE_x']:.6f}", f"{m['MAE_y']:.6f}", f"{m['MAE_z']:.6f}",
            f"{m['SSE_x']:.4f}",  f"{m['SSE_y']:.4f}",  f"{m['SSE_z']:.4f}",
        ])

    fig, ax = plt.subplots(figsize=(16, 3))
    ax.axis('off')
    tbl = ax.table(
        cellText=rows,
        rowLabels=[n.capitalize() for n in traj_names],
        colLabels=col_headers,
        cellLoc='center', rowLoc='center', loc='center',
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.scale(1, 2.0)

    for j in range(len(col_headers)):
        tbl[0, j].set_facecolor('#1f77b4'); tbl[0, j].set_text_props(color='white')
    for i in range(len(traj_names)):
        tbl[i+1, -1].set_facecolor(_TRAJ_COLORS.get(traj_names[i].lower(), '#eee'))
        tbl[i+1, -1].set_text_props(color='white')

    ax.set_title('Platform Tracking Error Metrics — Summary Table',
                 fontsize=11, pad=12)
    fig.tight_layout()
    _save(fig, name)


def plot_nominal_vs_adaptive(log_nom, log_adap, traj_name: str,
                              name: str = None):
    from config.parameters import G as GRAV, PLATFORM_MASS as M_NOM
    tag = (name or traj_name.lower().replace(' ', '_'))

    t_nom  = np.array(log_nom.time)
    t_adap = np.array(log_adap.time)

    err_nom  = np.linalg.norm(np.array(log_nom.platform_pos)  - np.array(log_nom.ref_pos),  axis=1)
    err_adap = np.linalg.norm(np.array(log_adap.platform_pos) - np.array(log_adap.ref_pos), axis=1)

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=False)

    ax = axes[0]
    ax.plot(t_nom,  err_nom,  color='#1f77b4', lw=1.5, label='Nominal (fixed model)', alpha=0.85)
    ax.plot(t_adap, err_adap, color='#d62728', lw=1.5, label='Adaptive (EKF)',       alpha=0.85)
    true_mass_arr = np.array(log_adap.true_mass)
    if true_mass_arr.max() != true_mass_arr.min():
        t_ch = t_adap[np.argmax(np.diff(true_mass_arr) != 0) + 1]
        ax.axvline(t_ch, color='orange', ls='--', lw=1.5, label=f'Payload change t={t_ch:.0f}s')
    ax.set_ylabel('Position error (m)'); ax.set_title('Tracking Error — Nominal vs Adaptive (EKF)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1]
    if log_adap.est_params:
        params    = np.array(log_adap.est_params)
        true_mass = np.array(log_adap.true_mass)
        ax.plot(t_adap, true_mass, '--', color='k',       lw=2.0, label='True mass')
        ax.plot(t_adap, params[:,0], '-', color='#d62728', lw=1.5, label='EKF estimate')
        ax.axhline(M_NOM, color='grey', ls=':', lw=1.0, label=f'Nominal ({M_NOM} kg)')
        if true_mass_arr.max() != true_mass_arr.min():
            ax.axvline(t_ch, color='orange', ls='--', lw=1.5)
    ax.set_ylabel('Mass (kg)'); ax.set_title('EKF Mass Estimation')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[2]
    if log_adap.est_params:
        params = np.array(log_adap.est_params)
        for i, (lbl, c) in enumerate(zip(['d_x','d_y','d_z'],['#e377c2','#7f7f7f','#bcbd22'])):
            ax.plot(t_adap, params[:,i+1], color=c, lw=1.2, label=lbl)
        ax.axhline(0, color='k', ls='--', lw=0.8)
        if true_mass_arr.max() != true_mass_arr.min():
            ax.axvline(t_ch, color='orange', ls='--', lw=1.5)
    ax.set_ylabel('Disturbance (N)'); ax.set_title('EKF Disturbance Estimation')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[3]
    if log_adap.est_params:
        vel    = np.array(log_adap.platform_vel)
        F      = np.array(log_adap.F_net_log)
        dt     = float(t_adap[1] - t_adap[0])
        acc_m  = np.gradient(vel[:,2], dt)
        acc_n  = F[:,2] / M_NOM - GRAV
        params = np.array(log_adap.est_params)
        acc_e  = (F[:,2] + params[:,3]) / params[:,0] - GRAV
        ax.plot(t_adap, acc_m, color='k',       lw=1.0, alpha=0.55, label='Measured')
        ax.plot(t_adap, acc_n, color='#1f77b4', lw=1.3, ls='--',    label=f'Fixed model ({M_NOM} kg)')
        ax.plot(t_adap, acc_e, color='#d62728', lw=1.2,             label='EKF-updated model')
        if true_mass_arr.max() != true_mass_arr.min():
            ax.axvline(t_ch, color='orange', ls='--', lw=1.5)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Z accel (m/s^2)')
    ax.set_title('Why EKF Helps — Correct Z-Acceleration Prediction')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    fig.suptitle(f'Nominal vs Adaptive (EKF) — {traj_name.capitalize()} + Payload Change',
                 fontsize=12)
    fig.tight_layout()
    _save(fig, f"{tag}_nominal_vs_adaptive")


def _platform_box_edges(center: np.ndarray,
                         corners_top: np.ndarray,
                         corners_bot: np.ndarray) -> list:
    segs = []
    for i in range(4):
        segs.append(np.vstack([corners_top[i], corners_top[(i+1) % 4]]))
    for i in range(4):
        segs.append(np.vstack([corners_bot[i], corners_bot[(i+1) % 4]]))
    for i in range(4):
        segs.append(np.vstack([corners_top[i], corners_bot[i]]))
    return segs


def animate(log,
            trajectory_name: str = "trajectory",
            speed_up: float = 5.0,
            save_gif: bool = False,
            save_mp4: bool = False,
            show: bool = True,
            step_skip: int = 2) -> None:
    from kinematics.inverse_kinematics import (
        platform_corners_world, all_robot_positions_3d
    )
    from config.parameters import DT

    try:
        import imageio_ffmpeg as _iff
        matplotlib.rcParams['animation.ffmpeg_path'] = _iff.get_ffmpeg_exe()
    except Exception:
        pass

    if show:
        try:
            matplotlib.use('TkAgg')
        except Exception:
            try:
                matplotlib.use('Qt5Agg')
            except Exception:
                matplotlib.use('Agg')
                show = False
    else:
        matplotlib.use('Agg')

    n_total = len(log.time)
    indices = list(range(0, n_total, step_skip))
    n_frames = len(indices)

    time_arr    = np.array(log.time)
    plat_pos    = np.array(log.platform_pos)
    plat_att    = np.array(log.platform_att)
    ref_pos     = np.array(log.ref_pos)
    uav_pos_all = np.array(log.uav_pos)
    ugv_pos_all = np.array(log.ugv_pos)
    tensions    = np.array(log.tensions)
    uav_tgt_all = np.array(log.uav_targets)
    ugv_tgt_all = np.array(log.ugv_targets)

    all_x = np.concatenate([
        plat_pos[:, 0], ref_pos[:, 0],
        uav_pos_all[:, :, 0].ravel(),
        ugv_pos_all[:, :, 0].ravel(),
        uav_tgt_all[:, :, 0].ravel(),
        ugv_tgt_all[:, :, 0].ravel(),
    ])
    all_y = np.concatenate([
        plat_pos[:, 1], ref_pos[:, 1],
        uav_pos_all[:, :, 1].ravel(),
        ugv_pos_all[:, :, 1].ravel(),
        uav_tgt_all[:, :, 1].ravel(),
        ugv_tgt_all[:, :, 1].ravel(),
    ])
    all_z = np.concatenate([plat_pos[:, 2], uav_pos_all[:, :, 2].ravel()])
    pad = 0.5
    xlim = (all_x.min() - pad, all_x.max() + pad)
    ylim = (all_y.min() - pad, all_y.max() + pad)
    zlim = (0.0, all_z.max() + 0.5)

    fig = plt.figure(figsize=(14, 7))
    fig.canvas.manager.set_window_title(f"RCDPR Animation – {trajectory_name}") \
        if hasattr(fig.canvas, 'manager') else None

    ax3d = fig.add_axes([0.02, 0.05, 0.62, 0.90], projection='3d')
    ax_t = fig.add_axes([0.70, 0.10, 0.27, 0.80])

    ax3d.set_xlim(*xlim); ax3d.set_ylim(*ylim); ax3d.set_zlim(*zlim)
    ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Z (m)')
    ax3d.set_title('RCDPR – LPV-MPC-FL')

    ax3d.plot(ref_pos[:, 0], ref_pos[:, 1], ref_pos[:, 2],
              '--', color=COLOR_REF, lw=1.0, alpha=0.5, label='Reference', zorder=1)

    gx = np.linspace(xlim[0], xlim[1], 3)
    gy = np.linspace(ylim[0], ylim[1], 3)
    GX, GY = np.meshgrid(gx, gy)
    ax3d.plot_surface(GX, GY, np.zeros_like(GX),
                      alpha=0.08, color='grey', zorder=0)

    trail_line, = ax3d.plot([], [], [], '-', color=COLOR_LPV, lw=1.2,
                             alpha=0.6, label='Platform', zorder=3)
    box_segs = [ax3d.plot([], [], [], '-', color=COLOR_PLAT, lw=2.0, zorder=4)[0]
                for _ in range(12)]
    plat_dot, = ax3d.plot([], [], [], 'o', color=COLOR_PLAT, ms=6, zorder=5)

    uav_dots  = [ax3d.plot([], [], [], 's', color=COLOR_UAV[i], ms=8,
                            label=f'UAV {i+1}', zorder=5)[0] for i in range(4)]
    uav_cables = [ax3d.plot([], [], [], '-', color=COLOR_CABLE_UAV, lw=1.0,
                             alpha=0.7, zorder=2)[0] for _ in range(4)]

    ugv_dots  = [ax3d.plot([], [], [], '^', color=COLOR_UGV[i], ms=8,
                            label=f'UGV {i+1}', zorder=5)[0] for i in range(4)]
    ugv_cables = [ax3d.plot([], [], [], '-', color=COLOR_CABLE_UGV, lw=1.0,
                             alpha=0.7, zorder=2)[0] for _ in range(4)]

    ax3d.legend(loc='upper left', fontsize=7, ncol=2)

    cable_labels = [f'U{i+1}' for i in range(4)] + [f'G{i+1}' for i in range(4)]
    bar_colors   = COLOR_UAV + COLOR_UGV
    bars = ax_t.bar(cable_labels, np.zeros(8), color=bar_colors, edgecolor='k', lw=0.5)
    ax_t.set_ylim(0, 35)
    ax_t.set_ylabel('Tension (N)')
    ax_t.set_title('Cable Tensions')
    ax_t.axhline(32.0, color='r', ls='--', lw=1.0, label='32 N limit')
    ax_t.legend(fontsize=7)
    ax_t.grid(True, axis='y', alpha=0.3)

    time_txt = ax3d.text2D(0.02, 0.96, '', transform=ax3d.transAxes,
                            fontsize=9, va='top')

    def _update(frame_idx):
        k = indices[frame_idx]
        pose = np.concatenate([plat_pos[k], plat_att[k]])

        from kinematics.inverse_kinematics import platform_corners_world
        ctop, cbot = platform_corners_world(pose)

        trail_line.set_data(plat_pos[:k+1, 0], plat_pos[:k+1, 1])
        trail_line.set_3d_properties(plat_pos[:k+1, 2])

        plat_dot.set_data([plat_pos[k, 0]], [plat_pos[k, 1]])
        plat_dot.set_3d_properties([plat_pos[k, 2]])

        edges = _platform_box_edges(plat_pos[k], ctop, cbot)
        for seg_line, seg in zip(box_segs, edges):
            seg_line.set_data([seg[0, 0], seg[1, 0]], [seg[0, 1], seg[1, 1]])
            seg_line.set_3d_properties([seg[0, 2], seg[1, 2]])

        uav_pos_k = uav_pos_all[k]
        for i in range(4):
            uav_dots[i].set_data([uav_pos_k[i, 0]], [uav_pos_k[i, 1]])
            uav_dots[i].set_3d_properties([uav_pos_k[i, 2]])
            xs = [uav_pos_k[i, 0], ctop[i, 0]]
            ys = [uav_pos_k[i, 1], ctop[i, 1]]
            zs = [uav_pos_k[i, 2], ctop[i, 2]]
            uav_cables[i].set_data(xs, ys)
            uav_cables[i].set_3d_properties(zs)

        ugv_pos_k = ugv_pos_all[k]
        for i in range(4):
            ugv_dots[i].set_data([ugv_pos_k[i, 0]], [ugv_pos_k[i, 1]])
            ugv_dots[i].set_3d_properties([0.0])
            xs = [ugv_pos_k[i, 0], cbot[i, 0]]
            ys = [ugv_pos_k[i, 1], cbot[i, 1]]
            zs = [0.0, cbot[i, 2]]
            ugv_cables[i].set_data(xs, ys)
            ugv_cables[i].set_3d_properties(zs)

        tau_k = tensions[k]
        for bar, h in zip(bars, tau_k):
            bar.set_height(h)

        time_txt.set_text(f't = {time_arr[k]:.1f} s')

        return (trail_line, plat_dot, time_txt,
                *box_segs, *uav_dots, *uav_cables,
                *ugv_dots, *ugv_cables,
                *[b for b in bars])

    interval_ms = int(DT * step_skip * 1000 / speed_up)
    anim = FuncAnimation(fig, _update, frames=n_frames,
                         interval=interval_ms, blit=False, repeat=True)

    if save_gif:
        gif_path = SAVE_DIR / f"{trajectory_name}_animation.gif"
        print(f"[Anim] Saving GIF -> {gif_path}  (this may take a minute) ...")
        writer = PillowWriter(fps=int(1000 / interval_ms))
        anim.save(str(gif_path), writer=writer, dpi=90)
        print(f"[Anim] GIF saved -> {gif_path}")

    if save_mp4:
        mp4_path = SAVE_DIR / f"{trajectory_name}_animation.mp4"
        print(f"[Anim] Saving MP4 -> {mp4_path} ...")
        try:
            writer = FFMpegWriter(fps=int(1000 / interval_ms), bitrate=1800)
            anim.save(str(mp4_path), writer=writer, dpi=120)
            print(f"[Anim] MP4 saved -> {mp4_path}")
        except Exception as e:
            print(f"[Anim] MP4 failed (ffmpeg not found?): {e}")
            print("[Anim] Try: pip install imageio[ffmpeg]")

    if show:
        plt.show()
    else:
        plt.close(fig)
