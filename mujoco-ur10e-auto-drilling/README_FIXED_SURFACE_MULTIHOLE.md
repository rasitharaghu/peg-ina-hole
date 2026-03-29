Fixed version based on your feedback.

What is fixed:
- panel is now planar and vertical
- robot setup and table are preserved from the original scene
- hole frames are compact and readable
- after each hole, the robot retracts slightly and transitions smoothly to the next hole
- no reset to home between holes

Added/updated files:
- scene_fixed_surface_multihole.xml
- hole_catalog.py
- multihole_runner.py
- README_FIXED_SURFACE_MULTIHOLE.md

Run:
python multihole_runner.py --sleep 0.15


Updated hole spacing:
- hole_2 / hole_3 moved to ±0.08 m in Y
- hole_4 / hole_5 moved to ±0.08 m in Z
