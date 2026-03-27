import numpy as np

XML_PATH = "scene.xml"

EE_SITE_NAME = "peg_tip"

HOME_KEY = "home"
HOLE_KEY = "hole"

CUSTOM_HOME_QPOS = np.array([
    1.20,
   -2.00,
    2.10,
   -1.80,
   -1.57,
    0.60,
], dtype=float)

HOLE_TARGET_POS = np.array([-0.75, 0.024, 0.908], dtype=float)
PREAPPROACH_OFFSET = np.array([0.18, 0.06, 0.00], dtype=float)

INSERTION_DEPTH = 0.02
INSERTION_AXIS_WORLD = np.array([1.0, 0.0, 0.0], dtype=float)

DLS_DAMPING_6D = 0.01
POSE_GAIN = 0.12
POS_TOL = 0.003
ORI_TOL = 0.03
ORI_GAIN = 0.25

MAX_STEPS_PER_PHASE = 6000
