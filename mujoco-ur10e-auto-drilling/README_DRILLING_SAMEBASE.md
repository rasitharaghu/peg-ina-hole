# Same-base drilling adaptation

This keeps your colleague's original **part-with-hole scene** and **UR10e placement** exactly as-is.

What changes:
- `universal_robots/ur10e_drill.xml` replaces only the peg/gripper with a simple drilling unit.
- `scene_drilling_samebase.xml` is the same original scene except it includes the drill-tool robot XML.
- `drill_targets_samebase.json` reuses the same validated hole pose as the drilling target.
- `drilling_samebase_debug.py` runs:
  - approach
  - predrill
  - contact detect
  - force-guided feed
  - dwell
  - retract

Run:
```bash
python drilling_samebase_debug.py
```
