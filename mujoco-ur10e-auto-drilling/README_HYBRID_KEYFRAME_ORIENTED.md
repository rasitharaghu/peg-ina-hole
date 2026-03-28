# Hybrid runner: hole-keyframe orientation + slow visualization

This update keeps the hybrid structure but fixes the main orientation issue:

- pre-approach and hole approach now use the **hole keyframe rotation**
- limited insertion is still used with a hard depth limit
- visualization speed is controlled with `--sleep`

## Run
```bash
python hybrid_runner.py --sleep 0.08
```

Try larger values for slower motion:
- `0.03` = mildly slow
- `0.08` = clearly visible
- `0.15` = very slow

## Files
- `hybrid_runner_config.py`
- `hybrid_runner_mujoco_adapter.py`
- `hybrid_insert_limited.py`
- `hybrid_runner.py`
