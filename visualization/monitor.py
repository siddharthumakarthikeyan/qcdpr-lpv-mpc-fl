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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from collections import deque

from config.parameters import (
    PLATFORM_MASS, CABLE_MAX_TENSION,
    PLATFORM_LX, PLATFORM_LY, PLATFORM_LZ,
)

_BG_DARK   = '#0d0d1a'
_BG_PANEL  = '#0d1117'
_BG_FIG    = '#111122'
_GRID      = '#1e1e3a'
_TEXT      = '#ccccdd'
_DIM_TEXT  = '#888899'
_REF       = '#3399ff'
_ERR_COL   = '#d62728'
_TRAIL_COL = '#ff9900'

UAV_COLORS = ['#00e676', '#ff9100', '#e040fb', '#40c4ff']
UGV_COLORS = ['#f48fb1', '#b0bec5', '#d4e157', '#4dd0e1']

TENSION_CMAP = matplotlib.colormaps.get_cmap('RdYlGn_r')


def _draw_drone(ax, center, yaw: float, color: str,
                arm: float = 0.13, prop_r: float = 0.045) -> None:
    cx, cy, cz = center
    a45 = yaw + np.pi / 4
    tips = [
        (cx + arm * np.cos(a45 + k * np.pi / 2),
         cy + arm * np.sin(a45 + k * np.pi / 2),
         cz)
        for k in range(4)
    ]
    for i, j in [(0, 2), (1, 3)]:
        ax.plot([tips[i][0], cx, tips[j][0]],
                [tips[i][1], cy, tips[j][1]],
                [tips[i][2], cz, tips[j][2]],
                color=color, lw=2.8, solid_capstyle='round', zorder=5)
    th = np.linspace(0, 2 * np.pi, 20)
    for tip in tips:
        ax.plot(tip[0] + prop_r * np.cos(th),
                tip[1] + prop_r * np.sin(th),
                np.full(20, tip[2]),
                color=color, lw=1.5, alpha=0.75, zorder=5)
    ax.scatter(*center, color=color, s=90, marker='o',
               edgecolors='white', linewidths=0.5, zorder=6)


def _draw_ugv(ax, pos2d, heading: float, color: str,
              bL: float = 0.22, bW: float = 0.16, bH: float = 0.07,
              wheel_r: float = 0.055, wheel_w: float = 0.03) -> None:
    px, py = pos2d
    ch, sh = np.cos(heading), np.sin(heading)

    def _rot(lx, ly):
        return px + lx * ch - ly * sh, py + lx * sh + ly * ch

    corners_local = [(bL/2, bW/2), (bL/2, -bW/2),
                     (-bL/2, -bW/2), (-bL/2, bW/2), (bL/2, bW/2)]
    xs = [_rot(c[0], c[1])[0] for c in corners_local]
    ys = [_rot(c[0], c[1])[1] for c in corners_local]
    ax.plot(xs, ys, np.full(len(xs), bH), color=color, lw=2.0, zorder=4)
    for lx, ly in [(bL/2, bW/2), (bL/2, -bW/2),
                   (-bL/2, -bW/2), (-bL/2, bW/2)]:
        wx, wy = _rot(lx, ly)
        ax.plot([wx, wx], [wy, wy], [0.0, bH], color=color, lw=1.5, zorder=4)
    fx, fy = _rot(bL/2, 0)
    ax.plot([px, fx], [py, fy], [bH, bH],
            color='white', lw=1.5, alpha=0.6, zorder=4)
    th = np.linspace(0, 2 * np.pi, 24)
    for side in [1, -1]:
        wc_x, wc_y = _rot(0, side * (bW / 2 + wheel_w / 2))
        wz = wheel_r * np.cos(th) + wheel_r
        wr = wheel_r * np.sin(th)
        wx_pts = wc_x + wr * ch
        wy_pts = wc_y + wr * sh
        ax.plot(wx_pts, wy_pts, wz, color='#dddddd', lw=1.8, zorder=4)


def _draw_platform(ax, corners_top, corners_bot) -> None:
    c_edge = 'saddlebrown'
    c_face = '#8b4513'

    def _edge(p, q):
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]],
                color=c_edge, lw=2.2, zorder=3)

    for i in range(4):
        j = (i + 1) % 4
        _edge(corners_top[i], corners_top[j])
        _edge(corners_bot[i], corners_bot[j])
        _edge(corners_top[i], corners_bot[i])

    top_verts = [list(corners_top)]
    poly = Poly3DCollection(top_verts, alpha=0.45,
                             facecolor=c_face, edgecolor=c_edge, linewidth=0)
    ax.add_collection3d(poly)


def _draw_cables(ax, uav_pos, ugv_z0, corners_top, corners_bot,
                 tensions) -> None:
    for i in range(4):
        t_norm = float(np.clip(tensions[i] / CABLE_MAX_TENSION, 0, 1))
        rgba = TENSION_CMAP(t_norm)
        ax.plot([uav_pos[i, 0], corners_top[i, 0]],
                [uav_pos[i, 1], corners_top[i, 1]],
                [uav_pos[i, 2], corners_top[i, 2]],
                color=rgba, lw=1.3, alpha=0.85, zorder=2)
    for i in range(4):
        t_norm = float(np.clip(tensions[4 + i] / CABLE_MAX_TENSION, 0, 1))
        rgba = TENSION_CMAP(t_norm)
        ax.plot([ugv_z0[i, 0], corners_bot[i, 0]],
                [ugv_z0[i, 1], corners_bot[i, 1]],
                [0.0,          corners_bot[i, 2]],
                color=rgba, lw=1.3, alpha=0.70, zorder=2)


class RealTimeMonitor:

    _HISTORY = 500

    def __init__(self, trajectory_name: str = '', t_sim: float = 100.0,
                 adaptive: bool = False):
        self.traj_name = trajectory_name
        self.t_sim     = t_sim
        self.adaptive  = adaptive

        self._t       = deque(maxlen=self._HISTORY)
        self._err     = deque(maxlen=self._HISTORY)
        self._tau     = [deque(maxlen=self._HISTORY) for _ in range(8)]
        self._m_hat   = deque(maxlen=self._HISTORY)
        self._m_true  = deque(maxlen=self._HISTORY)
        self._st      = deque(maxlen=self._HISTORY)

        self._trail_x = deque(maxlen=250)
        self._trail_y = deque(maxlen=250)
        self._trail_z = deque(maxlen=250)
        self._ref_pts: list = []

        try:
            matplotlib.use('TkAgg')
        except Exception:
            try:
                matplotlib.use('Qt5Agg')
            except Exception:
                pass

        self._build_layout()
        plt.ion()
        self.fig.show()

    def _build_layout(self) -> None:
        self.fig = plt.figure(figsize=(21, 11), facecolor=_BG_FIG)
        try:
            self.fig.canvas.manager.set_window_title(
                f'QCDPR Live Monitor  —  {self.traj_name}')
        except Exception:
            pass

        gs = gridspec.GridSpec(
            4, 2, figure=self.fig,
            left=0.01, right=0.99, bottom=0.03, top=0.96,
            width_ratios=[3.2, 1], hspace=0.38, wspace=0.05)

        self.ax3d = self.fig.add_subplot(gs[:, 0], projection='3d')
        self.ax3d.set_facecolor(_BG_DARK)
        for pane in [self.ax3d.xaxis.pane,
                     self.ax3d.yaxis.pane,
                     self.ax3d.zaxis.pane]:
            pane.fill = True
            pane.set_facecolor('#0c0c1a')
            pane.set_edgecolor(_GRID)
        self.ax3d.grid(True, color=_GRID, alpha=0.6)
        self.ax3d.set_title(
            f'LPV-MPC-FL  ·  {self.traj_name}  ·  QCDPR',
            color=_TEXT, fontsize=11, pad=8, fontweight='bold')
        for lbl in [self.ax3d.xaxis.label,
                    self.ax3d.yaxis.label,
                    self.ax3d.zaxis.label]:
            lbl.set_color(_DIM_TEXT)
            lbl.set_fontsize(8)
        self.ax3d.tick_params(colors='#666688', labelsize=7)
        self.ax3d.set_xlabel('X (m)'); self.ax3d.set_ylabel('Y (m)')
        self.ax3d.set_zlabel('Z (m)')

        def _right_ax(row, title):
            ax = self.fig.add_subplot(gs[row, 1], facecolor=_BG_PANEL)
            ax.set_title(title, color=_DIM_TEXT, fontsize=8, pad=3)
            ax.tick_params(colors=_DIM_TEXT, labelsize=7)
            for sp in ax.spines.values():
                sp.set_color(_GRID)
            return ax

        self.ax_err  = _right_ax(0, 'End-Effector Error (m)')
        self.ax_tau  = _right_ax(1, 'Cable Tensions (N)')
        self.ax_mass = _right_ax(2, 'EKF Mass Estimate (kg)')
        self.ax_info = _right_ax(3, 'System Status')

        self._ln_err, = self.ax_err.plot([], [], color=_ERR_COL, lw=1.8)
        self.ax_err.axhline(0, color=_GRID, lw=0.8)
        self.ax_err.set_xlim(0, self.t_sim)
        self.ax_err.set_ylim(0, 0.25)
        self.ax_err.set_xlabel('t (s)', color=_DIM_TEXT, fontsize=7)

        tau_clrs = [TENSION_CMAP(i / 8) for i in range(8)]
        self._ln_tau = [self.ax_tau.plot([], [], lw=1.2, color=c)[0]
                        for c in tau_clrs]
        self.ax_tau.set_xlim(0, self.t_sim)
        self.ax_tau.set_ylim(0, CABLE_MAX_TENSION * 1.05)
        self.ax_tau.axhline(CABLE_MAX_TENSION, color='#ff4444',
                             ls='--', lw=0.8, alpha=0.6)
        self.ax_tau.set_xlabel('t (s)', color=_DIM_TEXT, fontsize=7)

        self._ln_mass,      = self.ax_mass.plot([], [], color='#ff9100',
                                                 lw=1.8, label='EKF estimate')
        self._ln_true_mass, = self.ax_mass.plot([], [], '--', color=_REF,
                                                 lw=1.5, label='True mass')
        self.ax_mass.axhline(PLATFORM_MASS, color=_GRID, lw=0.8, ls=':')
        self.ax_mass.set_xlim(0, self.t_sim)
        self.ax_mass.set_ylim(PLATFORM_MASS * 0.8, PLATFORM_MASS * 2.2)
        self.ax_mass.legend(fontsize=6, facecolor=_BG_PANEL,
                            labelcolor=_DIM_TEXT, loc='lower right')
        self.ax_mass.set_xlabel('t (s)', color=_DIM_TEXT, fontsize=7)
        if not self.adaptive:
            self.ax_mass.set_title('Mass (EKF disabled)', color=_DIM_TEXT,
                                   fontsize=8, pad=3)

        self.ax_info.set_xticks([]); self.ax_info.set_yticks([])
        self._status_txt = self.ax_info.text(
            0.05, 0.88, '',
            transform=self.ax_info.transAxes,
            color=_TEXT, fontsize=8.5, va='top', fontfamily='monospace',
            linespacing=1.7)

    def update(self,
               t:              float,
               platform_pos:   np.ndarray,
               platform_att:   np.ndarray,
               ref_pos:        np.ndarray,
               uav_positions:  np.ndarray,
               uav_attitudes:  np.ndarray,
               ugv_positions:  np.ndarray,
               ugv_headings:   np.ndarray,
               corners_top:    np.ndarray,
               corners_bot:    np.ndarray,
               tensions:       np.ndarray,
               step_time_ms:   float,
               true_mass:      float        = PLATFORM_MASS,
               est_params:     np.ndarray | None = None) -> None:
        err = float(np.linalg.norm(platform_pos - ref_pos))

        self._t.append(t)
        self._err.append(err)
        for i in range(8):
            self._tau[i].append(float(tensions[i]))
        self._st.append(step_time_ms)
        self._trail_x.append(platform_pos[0])
        self._trail_y.append(platform_pos[1])
        self._trail_z.append(platform_pos[2])
        self._ref_pts.append(ref_pos.copy())

        m_hat = est_params[0] if est_params is not None else PLATFORM_MASS
        self._m_hat.append(m_hat)
        self._m_true.append(true_mass)

        t_arr = list(self._t)

        ax = self.ax3d
        ax.cla()
        ax.set_facecolor(_BG_DARK)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = True; pane.set_facecolor('#0c0c1a')
            pane.set_edgecolor(_GRID)
        ax.grid(True, color=_GRID, alpha=0.5)
        ax.set_xlabel('X (m)', color=_DIM_TEXT, fontsize=8)
        ax.set_ylabel('Y (m)', color=_DIM_TEXT, fontsize=8)
        ax.set_zlabel('Z (m)', color=_DIM_TEXT, fontsize=8)
        ax.tick_params(colors='#666688', labelsize=7)
        ax.set_title(
            f'LPV-MPC-FL  ·  {self.traj_name}  ·  t = {t:.1f} s',
            color=_TEXT, fontsize=11, pad=8, fontweight='bold')

        if len(self._ref_pts) > 1:
            rp = np.array(self._ref_pts)
            ax.plot(rp[:, 0], rp[:, 1], rp[:, 2],
                    '--', color=_REF, lw=1.2, alpha=0.5, label='Reference')

        if len(self._trail_x) > 1:
            ax.plot(list(self._trail_x), list(self._trail_y),
                    list(self._trail_z),
                    color=_TRAIL_COL, lw=1.8, alpha=0.6, label='Trajectory')

        all_x = (
            list(self._trail_x)
            + [p[0] for p in self._ref_pts[-20:]]
            + list(uav_positions[:, 0])
            + list(ugv_positions[:, 0])
        )
        all_y = (
            list(self._trail_y)
            + [p[1] for p in self._ref_pts[-20:]]
            + list(uav_positions[:, 1])
            + list(ugv_positions[:, 1])
        )
        if all_x and all_y:
            xlo, xhi = min(all_x) - 0.4, max(all_x) + 0.4
            ylo, yhi = min(all_y) - 0.4, max(all_y) + 0.4
            gx = np.array([xlo, xhi, xhi, xlo, xlo])
            gy = np.array([ylo, ylo, yhi, yhi, ylo])
            ax.plot(gx, gy, np.zeros(5), color=_GRID, lw=0.8, alpha=0.4)

        ugv_3d = np.column_stack([ugv_positions, np.zeros(len(ugv_positions))])
        for i in range(len(ugv_positions)):
            _draw_ugv(ax, ugv_positions[i], ugv_headings[i], UGV_COLORS[i])

        _draw_cables(ax, uav_positions, ugv_3d,
                     corners_top, corners_bot, tensions)

        _draw_platform(ax, corners_top, corners_bot)

        for i in range(len(uav_positions)):
            yaw = float(uav_attitudes[i, 2]) if uav_attitudes.ndim > 1 else 0.0
            _draw_drone(ax, uav_positions[i], yaw, UAV_COLORS[i])

        ax.scatter(*platform_pos, color='white', s=60, zorder=7,
                   edgecolors=_TRAIL_COL, linewidths=1.5)

        cx = float(np.mean([p[0] for p in self._ref_pts[-1:]] or [platform_pos[0]]))
        cy = float(np.mean([p[1] for p in self._ref_pts[-1:]] or [platform_pos[1]]))
        if len(self._ref_pts) > 1:
            rp = np.array(self._ref_pts)
            scene_x = np.concatenate([rp[:, 0], uav_positions[:, 0], ugv_positions[:, 0]])
            scene_y = np.concatenate([rp[:, 1], uav_positions[:, 1], ugv_positions[:, 1]])
            half_xy = max((scene_x.max() - scene_x.min()) / 2,
                          (scene_y.max() - scene_y.min()) / 2,
                          1.2) + 0.5
            cx = (scene_x.max() + scene_x.min()) / 2
            cy = (scene_y.max() + scene_y.min()) / 2
        else:
            half_xy = 2.0
        uav_z_max = float(uav_positions[:, 2].max())
        zhi = max(uav_z_max + 0.4, 2.5)
        ax.set_xlim(cx - half_xy, cx + half_xy)
        ax.set_ylim(cy - half_xy, cy + half_xy)
        ax.set_zlim(0.0, zhi)
        ax.set_box_aspect([1, 1, max(zhi / (2 * half_xy), 0.35)])

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc='upper left', fontsize=7,
                      facecolor='#1a1a2e', labelcolor=_TEXT,
                      framealpha=0.7, edgecolor=_GRID)

        self._ln_err.set_data(t_arr, list(self._err))
        self.ax_err.set_xlim(0, max(self.t_sim, t + 1))
        ymax_e = max(max(self._err, default=0.1) * 1.15, 0.05)
        self.ax_err.set_ylim(0, ymax_e)

        for i in range(8):
            self._ln_tau[i].set_data(t_arr, list(self._tau[i]))

        self._ln_mass.set_data(t_arr, list(self._m_hat))
        self._ln_true_mass.set_data(t_arr, list(self._m_true))
        ylo_m = min(PLATFORM_MASS * 0.85, min(self._m_hat, default=PLATFORM_MASS))
        yhi_m = max(PLATFORM_MASS * 1.8,  max(self._m_hat, default=PLATFORM_MASS))
        self.ax_mass.set_ylim(ylo_m, yhi_m * 1.1)
        self.ax_mass.set_xlim(0, max(self.t_sim, t + 1))

        mode = 'Adaptive (EKF)' if self.adaptive else 'Nominal'
        mean_st = float(np.mean(self._st)) if self._st else 0.0
        status = (
            f't      : {t:6.1f} / {self.t_sim:.0f} s\n'
            f'Error  : {err * 1000:6.1f} mm\n'
            f'Mass   : {m_hat:5.2f} kg  (true: {true_mass:.1f} kg)\n'
            f'Step   : {step_time_ms:5.1f} ms  (mean {mean_st:.1f})\n'
            f'Mode   : {mode}'
        )
        self._status_txt.set_text(status)
        self.ax_info.set_xlim(0, 1); self.ax_info.set_ylim(0, 1)

        try:
            self.fig.canvas.flush_events()
        except Exception:
            pass

    def close(self) -> None:
        try:
            plt.close(self.fig)
        except Exception:
            pass
