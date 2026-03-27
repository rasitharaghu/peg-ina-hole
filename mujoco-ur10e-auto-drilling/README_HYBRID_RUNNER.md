# Hybrid runner: global approach + local insertion with depth limit

These files are **new files only**. They do **not** modify the original files.

## What this does
- uses a stable full-pose approach from a far custom home pose
- then switches to a local insertion phase that reuses the **same style of logic as the original insertion code**:
  - local Cartesian position error
  - Jacobian-based DLS joint update
  - direct MuJoCo state update
- adds a **hard insertion depth limit** so the tool tip does not go completely through the surface

## Files
- `hybrid_runner_config.py`
- `hybrid_runner_mujoco_adapter.py`
- `hybrid_insert_limited.py`
- `hybrid_runner.py`

## Run
Put these files in the same folder as `scene.xml`, then run:

```bash
python hybrid_runner.py
```

For slower visual debugging:
```bash
python hybrid_runner.py --sleep 0.03
```

## Important note
This is a hybrid debug runner:
- global motion = stable pose approach
- local insertion = adapted from the original insert-peg control idea
- depth stop = newly added hard stop
