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

PLAT_KP = np.array([200.0, 200.0, 300.0])
PLAT_KD = np.array([60.0,  60.0,  70.0])
PLAT_KP_YAW = 12.0
PLAT_KD_YAW = 6.0

FL_KP = np.array([15.0, 15.0, 25.0])
FL_KD = np.array([10.0, 10.0, 12.0])

TENSION_MAX_ITER  = 100
TENSION_TOL       = 1e-6
TENSION_ALPHA_MIN = 1e-4
