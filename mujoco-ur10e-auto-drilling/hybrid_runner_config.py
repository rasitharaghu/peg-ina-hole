import numpy as np

XML_PATH = "scene.xml"

# Use the actual tip site for control
EE_SITE_NAME = "peg_tip"

HOME_KEY = "home"
HOLE_KEY = "hole"

# Far visible home pose for debugging
CUSTOM_HOME_QPOS = np.array([
    1.20,
   -2.00,
    2.10,
   -1.80,
   -1.57,
    0.60,
], dtype=float)

# Hole target that worked in earlier runs
HOLE_TARGET_POS = np.array([-0.75, 0.024, 0.908], dtype=float)
PREAPPROACH_OFFSET = np.array([0.18, 0.06, 0.00], dtype=float)

# Safe insertion settings
INSERTION_AXIS_WORLD = np.array([1.0, 0.0, 0.0], dtype=float)
MAX_INSERTION_TRAVEL = 0.012   # 12 mm hard limit
INSERTION_GAIN = 0.10
INSERTION_DAMPING = 0.20
INSERTION_POS_TOL = 0.0015

# Pose approach settings
DLS_DAMPING_6D = 0.01
POSE_GAIN = 0.12
POS_TOL = 0.003
ORI_TOL = 0.03
ORI_GAIN = 0.25

MAX_STEPS_PER_PHASE = 6000
