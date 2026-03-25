import numpy as np

XML_PATH = "scene.xml"

EE_SITE_NAME = "peg_tip"
PEG_TIP_SITE_NAME = "peg_tip"
FORCE_SENSOR_NAME = "wrist_force"
TORQUE_SENSOR_NAME = "wrist_torque"

HOME_KEY = "home"
HOLE_KEY = "hole"

# Very visible custom home pose
CUSTOM_HOME_QPOS = np.array([
    1.20,   # shoulder_pan_joint
   -2.00,   # shoulder_lift_joint
    2.10,   # elbow_joint
   -1.80,   # wrist_1_joint
   -1.57,   # wrist_2_joint
    0.60,   # wrist_3_joint
], dtype=float)

# Use your current hole target and a larger, clearer pre-approach offset
HOLE_TARGET_POS = np.array([-0.75, 0.024, 0.908], dtype=float)
PREAPPROACH_OFFSET = np.array([0.18, 0.06, 0.00], dtype=float)

DLS_DAMPING_3D = 1.0
SERVO_GAIN = 0.35
POS_TOL = 0.003

MAX_STEPS_PER_PHASE = 4000
HOLD_VIEWER_STEPS = 600
