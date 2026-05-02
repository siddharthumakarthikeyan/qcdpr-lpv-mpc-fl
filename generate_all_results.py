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
import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from simulator import (run_simulation, compute_error_metrics,
                        payload_change_scenario, nominal_scenario)
from trajectories import get_trajectory
from config.parameters import PLATFORM_MASS
from visualization.plotter import (
    plot_all,
    plot_workspace_tension_map,
    plot_tension_vs_mass,
    plot_ekf_results,
    plot_acceleration_comparison,
    plot_ablation,
    plot_multi_trajectory_overlay,
    plot_error_metrics_bar,
    plot_tension_statistics,
    plot_phase_portrait,
    plot_control_effort,
    plot_cooperative_workload,
    plot_performance_summary_table,
    plot_nominal_vs_adaptive,
)

TRAJ_NAMES  = ['quadrilateral', 'circular', 'sinusoidal']
PAYLOAD_AFTER = PLATFORM_MASS + 2.0


def _banner(msg: str):
    line = '=' * 62
    print(f"\n{line}\n  {msg}\n{line}")


def _run(display: str, traj, t_sim, dt, **kwargs) -> object:
    print(f"  [{display}] simulating {t_sim}s ...", end='', flush=True)
    t0  = time.perf_counter()
    log = run_simulation(traj, t_sim=t_sim, dt=dt, verbose=False, **kwargs)
    print(f" done ({time.perf_counter()-t0:.1f}s)")
    return log


def phase1_nominal(t_sim, dt, save_video):
    _banner("PHASE 1 — Nominal simulations (all three trajectories)")
    logs, metrics = [], []
    for traj_name in TRAJ_NAMES:
        traj = get_trajectory(traj_name)
        log  = _run(traj_name, traj, t_sim, dt,
                    adaptive=False, true_mass_fn=nominal_scenario())
        m    = compute_error_metrics(log)
        print(f"  Platform MAE  X={1000*m['MAE_x']:.2f}mm  "
              f"Y={1000*m['MAE_y']:.2f}mm  Z={1000*m['MAE_z']:.2f}mm")
        plot_all(log, trajectory_name=traj_name,
                 save_animation=save_video, animation_fmt='mp4')
        logs.append(log); metrics.append(m)
    return logs, metrics


def phase2_cross_trajectory(logs, metrics):
    _banner("PHASE 2 — Cross-trajectory comparison plots")

    print("  multi-trajectory 3D overlay ...")
    plot_multi_trajectory_overlay(logs, TRAJ_NAMES)

    print("  error metrics bar chart ...")
    plot_error_metrics_bar(metrics, TRAJ_NAMES)

    print("  cable tension statistics ...")
    plot_tension_statistics(logs, TRAJ_NAMES)

    print("  tracking error phase portrait ...")
    plot_phase_portrait(logs, TRAJ_NAMES)

    print("  UAV control effort ...")
    plot_control_effort(logs, TRAJ_NAMES)

    print("  cooperative cable workload ...")
    plot_cooperative_workload(logs, TRAJ_NAMES)

    print("  performance summary table ...")
    plot_performance_summary_table(metrics, TRAJ_NAMES)

    print("  Phase 2 complete.")


def phase3_ablation(t_sim, dt):
    _banner("PHASE 3 — Ablation study (quadrilateral + payload change at t=30s)")
    traj    = get_trajectory('quadrilateral')
    mass_fn = payload_change_scenario(t_change=30.0,
                                       mass_before=PLATFORM_MASS,
                                       mass_after=PAYLOAD_AFTER)

    log_A = _run('No EKF (A)',    traj, t_sim, dt,
                 adaptive=False, true_mass_fn=mass_fn,
                 label='Case A: No adaptation')
    log_B = _run('Mass only (B)', traj, t_sim, dt,
                 adaptive=True, est_mode='mass_only', true_mass_fn=mass_fn,
                 label='Case B: Mass estimation only')
    log_C = _run('Full EKF (C)', traj, t_sim, dt,
                 adaptive=True, est_mode='full', true_mass_fn=mass_fn,
                 label='Case C: Full EKF (mass + disturbance)')

    print("  ablation plot ...")
    plot_ablation([log_A, log_B, log_C], trajectory_name='quadrilateral')

    print("  EKF mass/disturbance/error ...")
    plot_ekf_results(log_A, log_C, trajectory_name='quadrilateral')

    print("  acceleration comparison ...")
    plot_acceleration_comparison(log_A, log_C, trajectory_name='quadrilateral')

    print("  Phase 3 complete.")
    return log_A, log_B, log_C


def phase4_nominal_vs_adaptive(t_sim, dt):
    _banner("PHASE 4 — Nominal vs Adaptive (with / without EKF) — all trajectories")
    mass_fn = payload_change_scenario(t_change=30.0,
                                       mass_before=PLATFORM_MASS,
                                       mass_after=PAYLOAD_AFTER)

    for traj_name in TRAJ_NAMES:
        print(f"\n  {traj_name.upper()}:")
        traj    = get_trajectory(traj_name)
        log_nom = _run('nominal', traj, t_sim, dt,
                       adaptive=False, true_mass_fn=mass_fn)
        log_adp = _run('adaptive', traj, t_sim, dt,
                       adaptive=True, est_mode='full', true_mass_fn=mass_fn)

        print(f"  nominal_vs_adaptive plot ...")
        plot_nominal_vs_adaptive(log_nom, log_adp,
                                  traj_name=traj_name,
                                  name=f"{traj_name}_nominal_vs_adaptive")

        plot_ekf_results(log_nom, log_adp, trajectory_name=traj_name)
        plot_acceleration_comparison(log_nom, log_adp, trajectory_name=traj_name)

    print("  Phase 4 complete.")


def phase5_static_analysis():
    _banner("PHASE 5 — Static workspace analysis")
    print("  Workspace tension map (this may take ~30 s) ...")
    plot_workspace_tension_map()
    print("  Tension vs payload mass ...")
    plot_tension_vs_mass()
    print("  Phase 5 complete.")


def main():
    parser = argparse.ArgumentParser(description='Generate all RCDPR results')
    parser.add_argument('--no-video',   action='store_true',
                        help='Skip MP4 animation (faster run)')
    parser.add_argument('--fast',       action='store_true',
                        help='Use 40 s simulations instead of 100 s')
    parser.add_argument('--phase',      type=int, default=0,
                        help='Run only one phase (1-5); 0 = all')
    args = parser.parse_args()

    t_sim = 40.0 if args.fast else 100.0
    dt    = 0.05
    save_video = not args.no_video

    print(f"\nRCDPR Results Generator")
    print(f"  T_sim = {t_sim} s  |  dt = {dt} s  |  video = {save_video}")
    print(f"  Results will be saved to ./results/\n")

    t_total = time.perf_counter()

    if args.phase in (0, 1):
        logs, metrics = phase1_nominal(t_sim, dt, save_video)
    else:
        print("[INFO] Re-simulating for cross-trajectory data ...")
        logs, metrics = [], []
        for traj_name in TRAJ_NAMES:
            traj = get_trajectory(traj_name)
            log  = _run(traj_name, traj, t_sim, dt,
                        adaptive=False, true_mass_fn=nominal_scenario())
            logs.append(log); metrics.append(compute_error_metrics(log))

    if args.phase in (0, 2):
        phase2_cross_trajectory(logs, metrics)

    if args.phase in (0, 3):
        phase3_ablation(t_sim, dt)

    if args.phase in (0, 4):
        phase4_nominal_vs_adaptive(t_sim, dt)

    if args.phase in (0, 5):
        phase5_static_analysis()

    elapsed = time.perf_counter() - t_total
    _banner(f"ALL DONE — total time {elapsed/60:.1f} min")
    print(f"  Figures and videos saved to ./results/\n")


if __name__ == '__main__':
    main()
