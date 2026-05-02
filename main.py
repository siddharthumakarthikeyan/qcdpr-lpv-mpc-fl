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
import sys
import pickle
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.parameters    import T_SIM, DT, PLATFORM_MASS
from trajectories         import get_trajectory
from simulator            import (run_simulation, compute_error_metrics,
                                   payload_change_scenario, nominal_scenario)
from visualization.plotter import (
    plot_all,
    plot_workspace_tension_map,
    plot_tension_vs_mass,
    plot_ekf_results,
    plot_acceleration_comparison,
    plot_ablation,
    animate,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='RCDPR LPV-MPC-FL Simulation'
    )
    parser.add_argument(
        'trajectory',
        nargs='?',
        default='quadrilateral',
        choices=['quadrilateral', 'circular', 'sinusoidal',
                 'single_uav', 'plots_only'],
        help='Trajectory type to simulate (default: quadrilateral)'
    )
    parser.add_argument(
        '--t-sim', type=float, default=T_SIM,
        help=f'Simulation duration in seconds (default: {T_SIM})'
    )
    parser.add_argument(
        '--dt', type=float, default=DT,
        help=f'Time step in seconds (default: {DT})'
    )
    parser.add_argument(
        '--save-log', action='store_true',
        help='Save simulation log to results/<traj>_log.pkl'
    )
    parser.add_argument(
        '--no-plots', action='store_true',
        help='Skip static plot generation'
    )
    parser.add_argument(
        '--animate', action='store_true',
        help='Show interactive 3-D animation after simulation'
    )
    parser.add_argument(
        '--save-gif', action='store_true',
        help='Save animation as GIF to results/ (requires Pillow)'
    )
    parser.add_argument(
        '--save-mp4', action='store_true',
        help='Save animation as MP4 to results/ (requires ffmpeg)'
    )
    parser.add_argument(
        '--speed-up', type=float, default=5.0,
        help='Animation playback speed multiplier (default: 5x)'
    )
    parser.add_argument(
        '--save-animation', action='store_true',
        help='Save MP4 animation from log after all plots are generated'
    )
    parser.add_argument(
        '--static-analysis', action='store_true',
        help='Also run static workspace / mass analysis plots (slow)'
    )
    parser.add_argument(
        '--adaptive', action='store_true',
        help='Enable EKF online parameter estimation (adaptive mode)'
    )
    parser.add_argument(
        '--scenario',
        choices=['nominal', 'payload_change'],
        default='nominal',
        help='Simulation scenario (default: nominal). '
             'payload_change: true mass jumps at t=30 s to test EKF.'
    )
    parser.add_argument(
        '--payload-after', type=float, default=PLATFORM_MASS + 2.0,
        help='True platform mass after the payload change (kg, default: +2 kg)'
    )
    parser.add_argument(
        '--compare', action='store_true',
        help='Run both nominal and adaptive back-to-back and overlay results'
    )
    parser.add_argument(
        '--ablation', action='store_true',
        help='Run 3-case ablation: (A) no EKF, (B) mass only, (C) full EKF'
    )
    parser.add_argument(
        '--monitor', action='store_true',
        help='Show real-time 3-D dashboard while simulating'
    )
    parser.add_argument(
        '--monitor-interval', type=int, default=20,
        help='Update monitor every N simulation steps (default: 20)'
    )
    return parser.parse_args()


def print_metrics(metrics: dict, trajectory_name: str):
    print(f"\n{'='*60}")
    print(f"  Error metrics – {trajectory_name} trajectory")
    print(f"{'='*60}")
    print(f"  {'Axis':<8} {'MSE':>12} {'MAE':>12} {'SSE':>12}")
    print(f"  {'-'*48}")
    for axis in ['x', 'y', 'z']:
        mse = metrics[f'MSE_{axis}']
        mae = metrics[f'MAE_{axis}']
        sse = metrics[f'SSE_{axis}']
        print(f"  {axis.upper():<8} {mse:>12.6f} {mae:>12.6f} {sse:>12.4f}")
    print(f"{'='*60}\n")


def main():
    args = parse_args()

    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    if args.trajectory == 'plots_only' or args.static_analysis:
        print("[Main] Generating static workspace tension map …")
        plot_workspace_tension_map()
        print("[Main] Generating tension vs mass plot …")
        plot_tension_vs_mass()
        if args.trajectory == 'plots_only':
            return

    traj_name = args.trajectory
    print(f"[Main] Trajectory : {traj_name}")
    print(f"[Main] Duration   : {args.t_sim} s  |  dt = {args.dt} s")
    print(f"[Main] Steps      : {int(args.t_sim / args.dt)}")

    if args.scenario == 'payload_change':
        true_mass_fn = payload_change_scenario(
            t_change    = 30.0,
            mass_before = PLATFORM_MASS,
            mass_after  = args.payload_after,
        )
        print(f"[Main] Scenario    : payload change at t=30 s  "
              f"({PLATFORM_MASS} kg -> {args.payload_after} kg)")
    else:
        true_mass_fn = nominal_scenario()
        print(f"[Main] Scenario    : nominal (fixed mass {PLATFORM_MASS} kg)")

    print(f"[Main] Adaptive    : {'YES (EKF)' if args.adaptive else 'NO'}")

    traj = get_trajectory(traj_name)

    monitor = None
    if args.monitor:
        from visualization.monitor import RealTimeMonitor
        monitor = RealTimeMonitor(
            trajectory_name = traj_name,
            t_sim           = args.t_sim,
            adaptive        = args.adaptive,
        )
        print(f"[Monitor] Dashboard opened  (update every {args.monitor_interval} steps)")

    if args.ablation:
        print("\n[Main] === ABLATION STUDY ===")
        print("[Main] Case A — No adaptation (baseline)")
        log_A = run_simulation(traj, t_sim=args.t_sim, dt=args.dt,
                                adaptive=False, true_mass_fn=true_mass_fn,
                                label='Case A: No adaptation', verbose=True)
        print("\n[Main] Case B — Mass estimation only")
        log_B = run_simulation(traj, t_sim=args.t_sim, dt=args.dt,
                                adaptive=True, est_mode='mass_only',
                                true_mass_fn=true_mass_fn,
                                label='Case B: Mass estimation only', verbose=True)
        print("\n[Main] Case C — Full EKF (mass + disturbance)")
        log_C = run_simulation(traj, t_sim=args.t_sim, dt=args.dt,
                                adaptive=True, est_mode='full',
                                true_mass_fn=true_mass_fn,
                                label='Case C: Full EKF (mass + disturbance)', verbose=True)

        if not args.no_plots:
            plot_ablation([log_A, log_B, log_C], trajectory_name=traj_name)
            plot_ekf_results(log_A, log_C, trajectory_name=traj_name)
            plot_acceleration_comparison(log_A, log_C, trajectory_name=traj_name)

        log = log_C

    elif args.compare:
        print("\n[Main] --- Baseline (nominal, no EKF) ---")
        log_base = run_simulation(traj, t_sim=args.t_sim, dt=args.dt,
                                   adaptive=False, true_mass_fn=true_mass_fn,
                                   verbose=True)
        print("\n[Main] --- Adaptive (EKF) ---")
        log_adap = run_simulation(traj, t_sim=args.t_sim, dt=args.dt,
                                   adaptive=True, true_mass_fn=true_mass_fn,
                                   verbose=True)

        print("\n[Main] === Baseline Metrics ===")
        print_metrics(compute_error_metrics(log_base), traj_name + " [Baseline]")
        print("[Main] === Adaptive Metrics ===")
        print_metrics(compute_error_metrics(log_adap), traj_name + " [Adaptive]")

        if not args.no_plots:
            plot_ekf_results(log_base, log_adap, trajectory_name=traj_name)
            plot_acceleration_comparison(log_base, log_adap, trajectory_name=traj_name)
            _save_anim = args.save_animation or args.save_mp4 or args.save_gif
            _anim_fmt  = 'gif' if (args.save_gif and not args.save_mp4) else 'mp4'
            plot_all(log_adap, trajectory_name=traj_name + "_adaptive",
                     save_animation=_save_anim, animation_speed=args.speed_up,
                     animation_fmt=_anim_fmt)

        log = log_adap

    else:
        log = run_simulation(traj, t_sim=args.t_sim, dt=args.dt,
                              adaptive=args.adaptive,
                              true_mass_fn=true_mass_fn,
                              monitor=monitor,
                              monitor_interval=args.monitor_interval,
                              verbose=True)

        metrics = compute_error_metrics(log)
        print_metrics(metrics, traj_name)

        mean_iter_ms = 1000 * np.mean(log.step_times)
        print(f"  Mean iteration time : {mean_iter_ms:.2f} ms")
        print(f"  Max  iteration time : {1000*max(log.step_times):.2f} ms\n")

        if not args.no_plots:
            _save_anim = args.save_animation or args.save_mp4 or args.save_gif
            _anim_fmt  = 'gif' if (args.save_gif and not args.save_mp4) else 'mp4'
            plot_all(log, trajectory_name=traj_name,
                     save_animation=_save_anim,
                     animation_speed=args.speed_up,
                     animation_fmt=_anim_fmt)
            if args.adaptive and log.est_params:
                plot_ekf_results(None, log, trajectory_name=traj_name)

    if args.save_log:
        log_path = results_dir / f"{traj_name}_log.pkl"
        with open(log_path, 'wb') as f:
            pickle.dump(log, f)
        print(f"[Main] Log saved → {log_path}")

    if args.animate:
        animate(
            log,
            trajectory_name = traj_name,
            speed_up        = args.speed_up,
            save_gif        = False,
            save_mp4        = False,
            show            = True,
        )

    if monitor is not None:
        print("[Monitor] Simulation complete — close the dashboard window to exit.")
        import matplotlib.pyplot as plt
        plt.ioff()
        plt.show(block=True)
        monitor.close()

    print("[Main] Complete.")


if __name__ == '__main__':
    main()
