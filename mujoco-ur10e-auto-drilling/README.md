# MuJoCo UR10e Automatic Drilling Unit

A starter simulation for a UR10e carrying an automatic drilling unit in MuJoCo.

## Features
- UR10e-like 6-DOF arm
- drilling spindle/tool TCP
- workpiece surface with drill targets
- hybrid controller:
  - XY position control
  - Z thrust-force regulation
  - orientation alignment to target normal
- drilling state machine:
  - move to approach
  - move to pre-drill
  - contact detect
  - thrust control
  - spindle on
  - dwell / drilling
  - retract
- benchmark mode across multiple holes
- KPI logging and plots

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run single simulation

```bash
python -m src.main --config config/drilling.yaml
```

## Run benchmark

```bash
python -m src.benchmark --config config/benchmark.yaml
```

## Notes
This repository is a starter baseline. The drilling process is simulated with a simplified contact / dwell / material-removal abstraction.
For production-grade research, extend the spindle dynamics, feed-rate logic, process forces, and exact robot geometry.
