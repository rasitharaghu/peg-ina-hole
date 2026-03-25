# Simple sequential runner for MuJoCo UR10e drilling debug

This is a stripped-down debug path to verify visible motion only.

Sequence:
1. Set a clearly far custom home joint pose
2. Move to a pre-approach target
3. Move to the hole target
4. Hold position so you can visually inspect

## Files
- `simple_sequential_runner.py`
- `simple_runner_config.py`
- `simple_runner_mujoco_adapter.py`

## How to use
Copy these files into your existing project folder, alongside `scene.xml`.

Run in pure kinematic debug mode first:
```bash
python simple_sequential_runner.py --debug-forward
```

If that shows correct visible motion, try:
```bash
python simple_sequential_runner.py
```

## Notes
- This uses `peg_tip` as the controlled site.
- It restricts Jacobian and updates to the 6 UR10e joints.
- It also syncs actuator controls to commanded joint values.
- This is intentionally not using BT yet.
