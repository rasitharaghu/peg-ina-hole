import numpy as np

XML_PATH = "scene.xml"

# Separate control frame and tip frame
CONTROL_SITE_NAME = "attachment_site"
TIP_SITE_NAME = "peg_tip"

# Only used for initial debug start
HOME_KEY = "home"

CUSTOM_HOME_QPOS = np.array([
    1.20,
   -2.00,
    2.10,
   -1.80,
   -1.57,
    0.60,
], dtype=float)

# Hole position in world frame
HOLE_TARGET_POS = np.array([-0.75, 0.024, 0.908], dtype=float)

# Positive X offset means start away from the wall, then approach hole
PREAPPROACH_OFFSET = np.array([0.20, 0.00, 0.00], dtype=float)

# World insertion direction toward the wall/hole.
# Flip sign if needed.
INSERTION_AXIS_WORLD = np.array([-1.0, 0.0, 0.0], dtype=float)
MAX_INSERTION_TRAVEL = 0.010  # 10 mm

# Desired roll reference for frame construction
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=float)

# Pose control parameters
DLS_DAMPING_6D = 0.01
POSE_GAIN = 0.10
POS_TOL = 0.003
ORI_TOL = 0.03
ORI_GAIN = 0.35

MAX_STEPS_PER_PHASE = 6000
DEFAULT_SLEEP = 0.08
HOLD_SECONDS_AFTER_DONE = 5.0
