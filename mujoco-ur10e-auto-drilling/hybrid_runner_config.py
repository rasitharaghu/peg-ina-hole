import numpy as np

XML_PATH = "scene.xml"

CONTROL_SITE_NAME = "attachment_site"
TIP_SITE_NAME = "peg_tip"

CUSTOM_HOME_QPOS = np.array([
    1.20,
   -2.00,
    2.10,
   -1.80,
   -1.57,
    0.60,
], dtype=float)

HOLE_TARGET_POS = np.array([-0.75, 0.024, 0.908], dtype=float)
PREAPPROACH_OFFSET = np.array([0.20, 0.00, 0.00], dtype=float)

# Desired tool/drill direction in world frame (flip sign if needed)
INSERTION_AXIS_WORLD = np.array([-1.0, 0.0, 0.0], dtype=float)
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=float)

# Pose control
DLS_DAMPING_6D = 0.01
POSE_GAIN = 0.10
POS_TOL = 0.003
ORI_TOL = 0.03
ORI_GAIN = 0.35

# Admittance-like insertion
MAX_INSERTION_TRAVEL = 0.010
INSERTION_STEP_NOMINAL = 0.0012
INSERTION_DAMPING = 0.15
INSERTION_GAIN = 0.08
INSERTION_LATERAL_COMPLIANCE = 0.20
INSERTION_AXIAL_SCALE = 1.0

# Simple drilling phase
DRILL_EXTRA_DEPTH = 0.006
DRILL_STEP_NOMINAL = 0.0006
DRILL_DAMPING = 0.20
DRILL_GAIN = 0.06

MAX_STEPS_PER_PHASE = 6000
DEFAULT_SLEEP = 0.08
HOLD_SECONDS_AFTER_DONE = 5.0
