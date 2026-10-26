# SAMU FANUC X/Z/Ry control prototypes

Three deliberately separate ROS 2 C++ strategies derived from the Nov27th PID→FOH concept.

## Strategy 1 — decoupled PID + FOH
`strategy1_pid_foh.cpp`: three independent PID loops for X, Z, process Ry. Each PID target is reconstructed by its own FOH/rate/acceleration limiter, then sent as a six-value FANUC DPM packet.

## Strategy 2 — feedforward + PID + FOH, no sensor/tool separation
`strategy2_ff_pid_foh.cpp`: combines geometry feedforward and residual PID feedback immediately: `u_target = Kff*u_ff + u_pid`, then independent FOH blocks. This intentionally assumes the sensor information applies to the dispenser now.

## Strategy 3 — feedforward + PID + preview buffer
`strategy3_preview_buffer.cpp`: GoCator surface samples are buffered with their physical surface location. Current dispenser TCP is supplied in the SAME common/workpiece frame. The nearest valid historical sample is selected, then FF+PID and FOH are applied. This models the ahead-mounted GoCator.

`buffer.mode=spatial_buffer` is recommended. `fixed_delay` exists only for controlled comparison/debugging.

## Units and FANUC mapping
Control internals use SI: metres and radians. The adapter defaults to m→mm and rad→deg because the supplied FANUC KAREL DPM example exchanges six REAL offsets `(dx,dy,dz,dw,dp,dr)`. The config field `map_process_ry_to_fanuc_p` is intentionally configurable: process-frame Ry must NOT be assumed to equal FANUC P until the actual DPM frame/tool setup is verified.

`fanuc.dry_run=true` by default. Do not connect to the robot until signs, frames, limits, units, DPM schedule and safety behavior are verified.

## Strategy 3 frame requirement
The controller does not invent calibration. `/samu/gocator/surface_sample` locations and `/samu/dispenser/tcp_pose` must already be expressed in the same calibrated common/workpiece frame. The GoCator→TCP extrinsic/hand-eye-related transformation should normally be performed upstream by perception/spatial analysis. This keeps robot control independent of sensor calibration implementation.

Surface sample array layout:
`[px, py, pz, measured_x, measured_z, measured_ry, ff_x, ff_z, ff_ry, confidence]`

TCP array layout:
`[x, y, z, ry]`

## Dummy tests
Run `dummy_gocator.py` for synthetic feedback/feedforward/surface samples. For Strategy 3 also run `dummy_tcp_publisher.py`. `control_test.py` is a pure-Python offline timing/buffer sanity test and needs no ROS/FANUC.

## Important commissioning order
Start with dry-run logging. Then Z only, X only, Ry only, X+Z, finally X+Z+Ry. Confirm correction signs and DPM frame before enabling combined control. PID/limit values in configs are placeholders and require system identification/tuning on the real setup.
