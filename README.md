# QCDPR-LPV-MPC-FL

**Real-Time Simulation of a Quadrotor-based Reconfigurable Cable-Driven Parallel Robot using LPV-MPC with Feedback Linearisation**

---

## Overview

This repository contains the full Python simulation for a **Quadrotor Cable-Driven Parallel Robot (QCDPR)** — a reconfigurable aerial manipulation system in which four UAVs (quadrotors) and four UGVs (differential-drive ground vehicles) cooperatively suspend and manoeuvre a rigid end-effector platform via eight cables.

The controller is a **Linear Parameter Varying – Model Predictive Control – Feedback Linearisation (LPV-MPC-FL)** cascade:

- **Outer loop (FL):** Feedback-linearising position controller generates desired thrust and attitude references for each UAV.
- **Inner loop (LPV-MPC):** Quasi-LPV attitude MPC tracks roll, pitch, and yaw by re-linearising the UAV dynamics at each time step.
- **UGV layer:** Independent NL-MPC for each ground vehicle tracks formation positions derived from the end-effector inverse kinematics.
- **Tension solver:** A rate-limited QP distributes cable tensions to satisfy wrench equilibrium at every control cycle.
- **EKF estimator:** Online Extended Kalman Filter estimates platform effective mass and external disturbances for adaptive control.

All MPC problems are solved via **direct OSQP calls on condensed (input-only) QPs**, eliminating CVXPY overhead and achieving mean solve times of **~2 ms per control cycle** — well below the 50 ms real-time budget.

---

## System Architecture

```
Trajectory Reference
        │
        ▼
 Centralized Controller
        │
        ├── Inverse Kinematics → UAV / UGV target positions
        │
        ├── FL Position Controller (per UAV) → T_d, φ_d, θ_d
        │
        ├── LPV-MPC Attitude Controller (per UAV) → torques
        │
        └── NL-MPC UGV Controller (per UGV) → velocity commands
                │
                ▼
        Rate-Limited Tension Solver (OSQP QP)
                │
                ▼
        Platform & Robot Dynamics (RK4 integration)
                │
                ▼
        EKF Parameter Estimator (online mass + disturbance)
```

---

## Key Results

### Nominal tracking (no payload change)

| Trajectory    | X MAE   | Y MAE  | Z MAE  |
|---------------|---------|--------|--------|
| Quadrilateral | 2.0 mm  | 2.0 mm | 0.0 mm |
| Circular      | 3.0 mm  | 3.0 mm | 0.0 mm |
| Sinusoidal    | 2.5 mm  | 8.0 mm | 0.1 mm |

### EKF ablation (quadrilateral + +2 kg payload step at t = 30 s)

| Case                        | X MAE  | Y MAE  | Z MAE   |
|-----------------------------|--------|--------|---------|
| A: No adaptation (baseline) | 2.0 mm | 2.0 mm | 45.7 mm |
| B: Mass estimation only     | 2.0 mm | 2.0 mm | 0.3 mm  |
| C: Full EKF (mass + dist.)  | 2.0 mm | 2.0 mm | 0.3 mm  |

- EKF convergence to ±5%: **1.0 s** after the mass step
- Steady-state mass error: **0.002 kg**
- Mean tracking error reduction vs baseline: **82–91%** across trajectories

---

## Repository Structure

```
simulation/
├── main.py                          # Entry point — single-trajectory runs
├── simulator.py                     # Main simulation loop (RK4 integration)
├── generate_all_results.py          # Bulk results generator (all 5 phases)
├── diagnose_positions.py            # Diagnostic script
│
├── config/
│   └── parameters.py                # All system parameters (mass, gains, horizons)
│
├── controllers/
│   ├── centralized.py               # Coordinates all UAV and UGV controllers
│   ├── lpv_mpc_fl_uav.py            # FL position controller + LPV-MPC attitude MPC
│   └── mpc_ugv.py                   # NL-MPC for differential-drive UGVs
│
├── dynamics/
│   ├── uav_dynamics.py              # Quadrotor rigid-body dynamics + LPV linearisation
│   ├── ugv_dynamics.py              # UGV kinematic/dynamic model + LPV linearisation
│   └── platform_dynamics.py        # Rigid end-effector platform dynamics
│
├── estimation/
│   └── ekf_estimator.py             # EKF for online mass + disturbance estimation
│
├── kinematics/
│   └── inverse_kinematics.py       # Platform pose → UAV / UGV target positions
│
├── tension/
│   └── tension_solver.py            # Rate-limited OSQP tension QP + Newton fallback
│
├── trajectories/
│   └── reference_trajectories.py   # Quadrilateral, circular, sinusoidal generators
│
├── visualization/
│   ├── plotter.py                   # All result plots + 3-D animation
│   └── monitor.py                   # Real-time live dashboard (--monitor flag)
│
└── results/                         # Generated PNG figures and videos (git-ignored)
```

---

## Requirements

- Python 3.10 or later

| Package      | Purpose                                |
|--------------|----------------------------------------|
| `numpy`      | Numerical computation                  |
| `scipy`      | Sparse matrices, optimisation          |
| `matplotlib` | Plotting and animation                 |
| `osqp`       | QP solver for MPC and tension problems |

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

All simulations are launched from inside the `simulation/` directory.

---

### `main.py` — Single trajectory run

#### Basic trajectory runs

```bash
python main.py                          # quadrilateral (default), 100 s
python main.py circular                 # circular trajectory
python main.py sinusoidal               # sinusoidal trajectory
python main.py quadrilateral            # explicit
```

#### Time and step control

```bash
python main.py circular --t-sim 200          # run for 200 s
python main.py quadrilateral --dt 0.02       # finer time step
python main.py sinusoidal --t-sim 50 --dt 0.02
```

#### Skip / save plots

```bash
python main.py circular --no-plots           # skip PNG generation (faster)
python main.py quadrilateral --save-log      # pickle SimLog to results/
```

#### Animation

```bash
python main.py circular --save-mp4           # save MP4 with plots (requires ffmpeg)
python main.py circular --save-gif           # save GIF with plots (requires Pillow)
python main.py circular --save-animation     # alias for --save-mp4
python main.py quadrilateral --animate       # interactive 3-D window after run
python main.py circular --save-mp4 --speed-up 10   # 10× playback speed
```

#### Adaptive mode (EKF)

```bash
python main.py quadrilateral --adaptive                          # EKF on, nominal mass (no change)
python main.py quadrilateral --adaptive --scenario payload_change  # EKF + +2 kg step at t=30 s
python main.py circular --adaptive --scenario payload_change --payload-after 6.0  # custom mass after
```

#### Compare nominal vs adaptive

```bash
python main.py quadrilateral --compare --scenario payload_change
python main.py circular      --compare --scenario payload_change
python main.py sinusoidal    --compare --scenario payload_change
```

#### Full 3-case ablation study

```bash
python main.py quadrilateral --ablation --scenario payload_change
python main.py circular      --ablation --scenario payload_change
python main.py sinusoidal    --ablation --scenario payload_change
```

#### Real-time live monitor dashboard

```bash
python main.py quadrilateral --monitor
python main.py circular      --monitor --monitor-interval 10   # update every 10 steps
python main.py quadrilateral --adaptive --monitor --scenario payload_change
```

#### Static workspace analysis only

```bash
python main.py plots_only                    # workspace tension map + tension vs mass
python main.py quadrilateral --static-analysis   # run sim + also generate static plots
```

#### Full combination examples

```bash
python main.py circular --adaptive --scenario payload_change --save-mp4 --t-sim 100
python main.py quadrilateral --ablation --scenario payload_change --no-plots
python main.py sinusoidal --compare --scenario payload_change --save-log
python main.py circular --adaptive --monitor --scenario payload_change --t-sim 60
```

---

### `generate_all_results.py` — Full paper results (all 5 phases)

#### Run everything

```bash
python generate_all_results.py                   # all phases, 100 s sims, with MP4
python generate_all_results.py --no-video        # all phases, skip MP4 (faster)
python generate_all_results.py --fast            # all phases, 40 s sims (quick check)
python generate_all_results.py --fast --no-video # fastest full run
```

#### Individual phases

```bash
python generate_all_results.py --phase 1                 # nominal simulations only
python generate_all_results.py --phase 2                 # cross-trajectory comparison only
python generate_all_results.py --phase 3                 # EKF ablation study only
python generate_all_results.py --phase 4                 # nominal vs adaptive comparison only
python generate_all_results.py --phase 5                 # static workspace analysis only
python generate_all_results.py --phase 1 --no-video      # Phase 1 without animation
python generate_all_results.py --phase 3 --fast          # ablation with 40 s sims
python generate_all_results.py --phase 4 --fast --no-video
```

**What each phase produces:**

| Phase | Description | Output files |
|-------|-------------|--------------|
| 1 | Nominal sims for all 3 trajectories | 11 PNGs + optional MP4 per trajectory |
| 2 | Cross-trajectory comparison plots | 7 PNGs (overlay, bar chart, box plots, phase portrait, control effort, workload, table) |
| 3 | EKF ablation (A/B/C cases, quadrilateral) | ablation error, EKF mass, EKF disturbance, acceleration comparison |
| 4 | Nominal vs adaptive for all 3 trajectories | 4 PNGs per trajectory |
| 5 | Static workspace analysis | workspace tension map, tension vs mass |

---

### Output files reference

All outputs are saved to `results/`. Key filenames:

| File pattern | Content |
|---|---|
| `<traj>_3d.png` | 3-D trajectory (equal-scale axes) |
| `<traj>_top.png` | Top-view XY trajectory |
| `<traj>_pos_vel.png` | Position and velocity tracking |
| `<traj>_platform_errors.png` | Per-axis tracking error |
| `<traj>_tensions.png` | UAV and UGV cable tensions |
| `<traj>_error.png` | Total position error over time |
| `<traj>_iter.png` | Computation time per step |
| `<traj>_uav_tracking.png` | UAV position vs IK targets |
| `<traj>_uav_errors.png` | UAV errors, attitude, speed |
| `<traj>_ugv_tracking.png` | UGV position vs IK targets |
| `<traj>_ugv_errors.png` | UGV errors and speed |
| `<traj>_ekf_ablation.png` | Ablation study error overlay |
| `<traj>_ekf_mass.png` | EKF mass estimate vs true mass |
| `<traj>_ekf_disturbance.png` | EKF disturbance estimates |
| `<traj>_ekf_error_comparison.png` | Baseline vs adaptive error |
| `<traj>_ekf_acceleration_why.png` | Z-acceleration prediction |
| `<traj>_nominal_vs_adaptive_*.png` | Nominal vs adaptive comparison |
| `all_trajectories_3d.png` | All three paths on one 3-D plot |
| `error_metrics_bar.png` | MAE bar chart across trajectories |
| `tension_statistics.png` | Cable tension box plots |
| `tracking_phase_portrait.png` | X-error vs Y-error phase portrait |
| `control_effort.png` | UAV thrust distributions |
| `cooperative_workload.png` | UAV vs UGV cable load sharing |
| `performance_summary_table.png` | Full error metrics table |
| `workspace_tension.png` | Max tension vs platform XY position |
| `tension_vs_mass.png` | Mean tension vs payload mass |
| `<traj>_animation.mp4` | 3-D animation (with --save-mp4) |
| `<traj>_animation.gif` | 3-D animation (with --save-gif) |

---

## Reproducing Paper Results

```bash
python generate_all_results.py --no-video
```

This runs all 5 phases (~7 min on a modern laptop) and reproduces every figure in the paper.

For individual trajectory figures only:

```bash
python main.py quadrilateral
python main.py circular
python main.py sinusoidal
```

---

## Tuning Key Parameters

All system and controller parameters are centralised in `config/parameters.py`.

| Parameter         | Default | Description                                  |
|-------------------|---------|----------------------------------------------|
| `UAV_HORIZON`     | 10      | MPC prediction horizon for UAV attitude      |
| `UGV_HORIZON`     | 15      | MPC prediction horizon for UGV               |
| `PLAT_KP`         | [200, 200, 300] | Platform PD proportional gains    |
| `PLAT_KD`         | [60, 60, 70]    | Platform PD derivative gains      |
| `FL_KP`           | [15, 15, 25]    | FL position controller gains      |
| `PLATFORM_MASS`   | 3.0 kg  | Nominal platform mass                        |
| `DT`              | 0.05 s  | Simulation time step                         |
| `T_SIM`           | 100.0 s | Default simulation duration                  |

EKF tuning is in `estimation/ekf_estimator.py`:

| Parameter        | Default | Description                          |
|------------------|---------|--------------------------------------|
| `_Q_DEFAULT`     | diag([1e-4, 0.05, 0.05, 0.02]) | Process noise |
| `_R_DEFAULT`     | diag([0.04, 0.04, 0.04]) | Measurement noise |
| `_ALPHA_DEFAULT` | 0.95    | AR(1) disturbance damping            |
| `_DM_MAX_DEFAULT`| 0.30 kg | Max mass change per EKF step         |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{siddharthu2025qcdpr,
  title   = {Real-Time Control of a Quadrotor-based Reconfigurable
             Cable-Driven Parallel Robot with LPV-MPC-FL},
  author  = {Siddharth U and Badri Narayanan R},
  journal = {Journal Name},
  year    = {2025}
}
```

---

## License

Copyright (C) 2025 Siddharth U and Badri Narayanan R, SASTRA Deemed to be University, Thanjavur, India. All rights reserved. Released for academic research use only.
