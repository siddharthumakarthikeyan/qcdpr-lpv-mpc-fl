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

G = 9.81

UAV_MASS        = 0.5
UAV_IXX         = 0.0075
UAV_IYY         = 0.0075
UAV_IZZ         = 0.013
UAV_ARM         = 0.23
UAV_KT          = 2.98e-6
UAV_KQ          = 1.14e-7
UAV_KD          = 0.02
UAV_MAX_THRUST  = 50.0
UAV_TENSION_LIMIT = 32.0
NUM_UAVS        = 4

UGV_MASS        = 1.2
UGV_IZZ         = 0.05
UGV_WHEEL_R     = 0.05
UGV_WHEEL_BASE  = 0.30
UGV_MAX_VEL     = 1.5
UGV_MAX_ANG_VEL = 2.5
UGV_KD          = 0.05
NUM_UGVS        = 4

PLATFORM_MASS   = 3.0
PLATFORM_IXX    = 0.12
PLATFORM_IYY    = 0.12
PLATFORM_IZZ    = 0.20
PLATFORM_LX     = 0.80
PLATFORM_LY     = 0.80
PLATFORM_LZ     = 0.10

_hx = PLATFORM_LX / 2
_hy = PLATFORM_LY / 2
_hz = PLATFORM_LZ / 2

CORNERS_TOP_BODY = np.array([
    [ _hx,  _hy,  _hz],
    [-_hx,  _hy,  _hz],
    [-_hx, -_hy,  _hz],
    [ _hx, -_hy,  _hz],
])

CORNERS_BOT_BODY = np.array([
    [ _hx,  _hy, -_hz],
    [-_hx,  _hy, -_hz],
    [-_hx, -_hy, -_hz],
    [ _hx, -_hy, -_hz],
])

PLATFORM_ZREF        = 1.00
CABLE_OUTWARD_OFFSET = 0.60
CABLE_VERTICAL_UAV   = (PLATFORM_ZREF + _hz + 0.95)
CABLE_MIN_TENSION    = 0.50
CABLE_MAX_TENSION    = 50.0

# ---------------------------------------------------------------------------
# Cable skew angle (azimuthal tilt of the cable anchor offset).
#
# With CABLE_SKEW_ANGLE_DEG = 0 the radial offset places each robot
# directly outward from its platform corner; the resulting wrench
# matrix W in R^{6x8} has rank 5 because every cable's horizontal force
# is collinear with the corner's position vector (zero yaw moment arm).
#
# We use a non-zero skew with sign alternating across the 4 corners
# (+,-,+,-) inside kinematics/inverse_kinematics.py.  This:
#   1. lifts the rank of W to 6 (yaw becomes actively controllable
#      through the cables), and
#   2. keeps zero net yaw torque at uniform tensions (no drift bias).
# Together these eliminate the previous "yaw makeup" (forcing psi_ref=0
# in the IK frame and stabilising yaw with artificial aerodynamic drag
# on the platform) and let the controller track the reference yaw
# directly through the physical cable wrench.
CABLE_SKEW_ANGLE_DEG = 8.0

NUM_CABLES  = 8

DT      = 0.05
T_SIM   = 100.0

UAV_HORIZON = 10
UGV_HORIZON = 15

UAV_Q = np.diag([
    15.0, 15.0, 20.0,
    2.0,  2.0,  3.0,
    8.0,  8.0,  5.0,
    0.5,  0.5,  0.5
])
UAV_R_MPC = np.diag([0.05, 0.5, 0.5, 1.0])
UAV_P_MPC = np.diag([0.01, 0.05, 0.05, 0.1])

UAV_PHI_MAX   = np.deg2rad(30)
UAV_THETA_MAX = np.deg2rad(30)
UAV_VEL_MAX   = 5.0

UGV_Q = np.diag([100.0, 100.0, 80.0, 0.5, 0.5, 0.1])
UGV_R_MPC = np.diag([0.02, 0.08])
UGV_P_MPC = np.diag([0.005, 0.02])

PLAT_KP = np.array([240.0, 240.0, 400.0])
PLAT_KD = np.array([85.0,  85.0,  110.0])
# Yaw closed-loop stabilisation through the cables (rank-6 layout, see
# CABLE_SKEW_ANGLE_DEG).  The cable yaw moment arm is small (set by
# sin(skew_angle)), so the achievable yaw-torque slew is limited by the
# DELTA_TAU_MAX rate constraint inside the tension QP.  We keep
# omega_n ~ 1.6 rad/s and zeta ~ 1.4 to stay safely inside that
# actuator bandwidth.
PLAT_KP_YAW = 2.5
PLAT_KD_YAW = 4.5
# Roll/pitch attitude stabilisation through cables.  Without aerodynamic
# drag the cables are the sole source of damping; gains are chosen for
# omega_n ~ 7 rad/s with zeta ~ 1.3 to suppress the slow tilt mode that
# is otherwise excited by trajectories with non-zero yaw rate.
PLAT_KP_RP = 50.0
PLAT_KD_RP = 18.0

FL_KP = np.array([15.0, 15.0, 25.0])
FL_KD = np.array([10.0, 10.0, 12.0])

# ---------------------------------------------------------------------------
# Platform aerodynamic drag.  The platform is a bluff body (0.8 x 0.8 x 0.1 m)
# moving slowly through still air, so a linear drag model F_drag = -C v is
# adequate.  A small but non-zero drag provides a guaranteed damping floor
# for the position and attitude loops -- without it, the only sources of
# damping are the cables themselves, which become proportionally weaker as
# the payload mass grows (mu_eff = c_cable / m).  The values below give
# tau_settle ~ 0.5 s in translation and ~0.3 s in rotation at the 3 kg
# nominal mass, well below the controller bandwidths.
PLAT_DRAG_TRANS    = 6.0   # N / (m/s)   per axis
PLAT_DRAG_ROLLPITCH = 1.2  # N.m / (rad/s)
PLAT_DRAG_YAW       = 0.8  # N.m / (rad/s)

TENSION_MAX_ITER  = 100
TENSION_TOL       = 1e-6
TENSION_ALPHA_MIN = 1e-4
