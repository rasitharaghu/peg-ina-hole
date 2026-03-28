# Stronger insertion + drill parameters

This is the same corrected stop-condition package, but with stronger insertion
and drill parameters so the motion is visible and the peg seats more clearly.

## Main parameter changes
- MAX_INSERTION_TRAVEL: 0.020
- INSERTION_STEP_NOMINAL: 0.0030
- INSERTION_GAIN: 0.18
- INSERTION_DAMPING: 0.10
- INSERTION_LATERAL_COMPLIANCE: 0.10

- DRILL_EXTRA_DEPTH: 0.012
- DRILL_STEP_NOMINAL: 0.0015
- DRILL_GAIN: 0.12
- DRILL_DAMPING: 0.10

## Run
```bash
python hybrid_runner.py --sleep 0.15
```
