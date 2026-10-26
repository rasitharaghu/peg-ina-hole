# Data flow

Strategy 1: `GoCator feedback -> PID_X/PID_Z/PID_Ry -> FOH_X/Z/Ry -> FANUC adapter -> DPM`.

Strategy 2: `GoCator geometry FF + current residual feedback -> FF + PID -> FOH -> FANUC DPM`. No look-ahead compensation.

Strategy 3: `GoCator -> surface estimation + calibrated frame transform (upstream) -> spatial preview buffer -> match current dispenser TCP -> FF + residual PID -> FOH/rate/accel limiting -> FANUC DPM`.

The buffer solves actuator/sensor spatial separation; it is not primarily a noise filter. Timestamp/age/confidence are validity checks. Spatial location is the primary matching key.
