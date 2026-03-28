# Corrected stop conditions: tip-based success + actual insertion/drill starts

This update fixes the misleading success logic:

- approach phases now require:
  - control frame position tolerance
  - control frame orientation tolerance
  - **peg tip position tolerance**
- admittance insertion now starts from the **actual current peg tip**
- drill feed now starts from the **actual current peg tip after insertion**
- added clearer logging so viewer and terminal match better

## Files
- `hybrid_runner_config.py`
- `hybrid_runner_mujoco_adapter.py`
- `admittance_insert_controller.py`
- `simple_drill_controller.py`
- `hybrid_runner.py`

## Run
```bash
python hybrid_runner.py --sleep 0.08
```

Slower:
```bash
python hybrid_runner.py --sleep 0.15
```
