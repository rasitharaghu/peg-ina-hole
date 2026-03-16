# Stable MuJoCo UR10e-Style Drilling Setup

This package is a **stable debug-first drilling setup** for MuJoCo.

Important:
- It uses a **UR10e-style simplified arm**, not an OEM-exact UR10e model.
- The main purpose is to give you a **stable scene, reachable target marker, position-servo control, and approach-only debug flow**.
- Once this works, you can replace the arm with a real UR10e MJCF/URDF model while keeping the same scene/control structure.

## What is improved
- vertical panel facing the robot
- visible red target marker on the robot-facing face
- position actuators instead of raw motor torques
- softer controller with joint-target updates
- approach-only mode first, then optional drilling mode
- panel placed closer and slightly forward for elbow clearance

## Run
```bash
python3 -m src.main --config config/drilling.yaml
```

## Benchmark
```bash
python3 -m src.benchmark --config config/benchmark.yaml
```
