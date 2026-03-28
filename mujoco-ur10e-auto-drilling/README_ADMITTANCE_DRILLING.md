# Approach + Admittance Insertion + Simple Drilling Phase

This package adds the next control stage on top of the frame-based approach:

1. Far home
2. Pre-approach pose
3. Hole pose
4. **Admittance-style insertion**
5. **Simple drilling feed phase**
6. Hold viewer open

## Control structure
- Global motion: pose control using `attachment_site`
- Tip tracking: `peg_tip`
- Insertion: admittance-style compliance
- Drilling: simple position + force-style feed placeholder with depth limit

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

## Notes
This is still a simulation/debug controller, not a validated industrial drilling controller.
