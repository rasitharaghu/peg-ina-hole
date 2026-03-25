import numpy as np

XML_PATH = "scene.xml"

# Use peg_tip for debugging so the controlled point is the actual drill/peg tip.
EE_SITE_NAME = "peg_tip"
PEG_TIP_SITE_NAME = "peg_tip"

FORCE_SENSOR_NAME = "wrist_force"
TORQUE_SENSOR_NAME = "wrist_torque"

HOME_KEY = "home"
HOLE_KEY = "hole"

# -------- Home pose selection --------
# Set to True to use the custom far home pose below instead of the XML keyframe "home".
USE_CUSTOM_HOME_QPOS = True

# A visibly far joint configuration for UR10e-style arm.
# You can change these 6 joint values and rerun.
CUSTOM_HOME_QPOS = np.array([
    0.0,     # shoulder_pan_joint
   -1.80,    # shoulder_lift_joint
    1.80,    # elbow_joint
   -1.50,    # wrist_1_joint
   -1.57,    # wrist_2_joint
    0.00,    # wrist_3_joint
], dtype=float)

# -------- Targets --------
HOLE_TARGET_POS = np.array([-0.75, 0.024, 0.908], dtype=float)
PREAPPROACH_OFFSET = np.array([0.08, 0.0, 0.0], dtype=float)

# -------- Motion / process params --------
INSERTION_DEPTH = -0.03
RETRACT_DISTANCE = 0.05
SERVO_DURATION = 4.0
INSERT_DURATION = 8.0
DRILL_DURATION = 3.0
RETRACT_DURATION = 2.5

DLS_DAMPING_6D = 0.01
DLS_DAMPING_3D = 1.0
SERVO_GAIN = 0.35
SERVO_POS_TOL = 0.003
SERVO_ORI_GAIN = 0.15

K_POS = np.array([200.0, 5.0, 5.0], dtype=float)
K_FORCE = np.array([0.02, 0.10, 0.10], dtype=float)
K_TORQUE = np.array([0.0, 0.01, 0.01], dtype=float)
INSERT_INTEGRATION_GAIN = 0.03
FORCE_LIMIT_N = 150.0
TORQUE_LIMIT_NM = 40.0

DRILL_FEED_SPEED = 0.0025
DRILL_HOLD_FORCE = 20.0
DRILL_FORCE_GAIN = 0.0004

HEADLESS_STEPS = 12000
