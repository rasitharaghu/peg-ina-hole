import numpy as np

# Exact model / scene integration
XML_PATH = "scene.xml"
EE_SITE_NAME = "attachment_site"
PEG_TIP_SITE_NAME = "peg_tip"
FORCE_SENSOR_NAME = "wrist_force"
TORQUE_SENSOR_NAME = "wrist_torque"

# Keyframes from the provided UR10e model
HOME_KEY = "home"
HOLE_KEY = "hole"

# Hole target reused from the original move_to_hole.py
HOLE_TARGET_POS = np.array([-0.75, 0.024, 0.908], dtype=float)

# Motion parameters
PREAPPROACH_OFFSET = np.array([0.08, 0.0, 0.0], dtype=float)   # stand off in +X before insertion
INSERTION_DEPTH = -0.03                                         # same 30 mm insertion as original code
RETRACT_DISTANCE = 0.05
SERVO_DURATION = 4.0
INSERT_DURATION = 8.0
DRILL_DURATION = 3.0
RETRACT_DURATION = 2.5

# DLS / servo gains
DLS_DAMPING_6D = 0.01
DLS_DAMPING_3D = 1.0
SERVO_GAIN = 0.35
SERVO_POS_TOL = 0.003
SERVO_ORI_GAIN = 0.15

# Admittance-like contact gains
# X = drilling axis in this scene's original scripts
K_POS = np.array([200.0, 5.0, 5.0], dtype=float)
K_FORCE = np.array([0.02, 0.10, 0.10], dtype=float)
K_TORQUE = np.array([0.0, 0.01, 0.01], dtype=float)
INSERT_INTEGRATION_GAIN = 0.03
FORCE_LIMIT_N = 150.0
TORQUE_LIMIT_NM = 40.0

# Drill feed behavior
DRILL_FEED_SPEED = 0.0025      # m/s equivalent target drift on X
DRILL_HOLD_FORCE = 20.0        # desired compressive force magnitude on X (scene-specific tuning)
DRILL_FORCE_GAIN = 0.0004

HEADLESS_STEPS = 12000
