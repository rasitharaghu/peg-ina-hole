# BT + MuJoCo drilling update for the provided UR10e scene

This package keeps the original folder structure and original files intact, and adds a separate control path:

- `run_bt_drilling.py`
- `bt.py`
- `tree_factory_bt.py`
- `actions_bt.py`
- `controllers.py`
- `mujoco_adapter.py`
- `config_bt.py`
- `adu_process.py`

## Exact model integration used
This update is wired to the exact names found in the provided model:

- end-effector site: `attachment_site`
- peg tip site: `peg_tip`
- force sensor: `wrist_force`
- torque sensor: `wrist_torque`
- keyframes: `home`, `hole`

## What is reused from the original code
- Jacobian-based DLS inverse kinematics
- direct MuJoCo joint integration through `mj_integratePos`
- original hole target and insertion depth idea
- contact-phase correction idea

## What is added
- behavior tree orchestration
- explicit phases: reset -> pre-approach -> approach -> admittance insert -> drill -> retract
- real MuJoCo force / torque sensor reads through `data.sensordata`
- simple ADU process abstraction (spindle/feed/depth)

## Important honesty
This remains a MuJoCo-oriented scaffold.  
It is **not real MoveIt2 runtime code** and **not production drilling control**.

The "MoveIt2 / position control" phase is represented here as a structured free-space approach controller inside the same MuJoCo setup so you can keep the same robot model and scene.

## Run
```bash
cd Ishrath/peg_ina_hole
python run_bt_drilling.py
```

Headless:
```bash
python run_bt_drilling.py --headless
```

## Force sensor integration
The update reads:
- `wrist_force`
- `wrist_torque`

via MuJoCo sensor addresses:
- `model.sensor_adr`
- `model.sensor_dim`
- `data.sensordata`

Bias is zeroed at the start of the insertion phase.

## Tuning notes
You will likely need to tune:
- `K_POS`
- `K_FORCE`
- `K_TORQUE`
- `INSERT_INTEGRATION_GAIN`
- `FORCE_LIMIT_N`
- `TORQUE_LIMIT_NM`

depending on contact stiffness and the wall / peg geometry.
