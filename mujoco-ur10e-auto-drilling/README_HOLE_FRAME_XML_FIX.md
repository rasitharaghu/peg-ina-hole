# Hole-frame XML fix package

This package adds an explicit hole frame to the scene and updates the runner to use it.

## New XML
- `scene_with_hole_frame.xml`
  - adds `hole_site` at hole center
  - adds `hole_axis_site` 30 mm along the insertion direction
  - adds visible RGB frame markers

## Updated control logic
- reads hole center from `hole_site`
- reads insertion axis from `hole_axis_site - hole_site`
- uses `attachment_site` for orientation control
- uses `peg_tip` for tip targeting
- uses stronger insertion/drill parameters
- requires peg tip error to converge before approach succeeds

## Run
```bash
python hybrid_runner.py --sleep 0.15
```

If the hole marker is not centered on the visible hole, edit the `pos` of `hole_frame` in `scene_with_hole_frame.xml`.
