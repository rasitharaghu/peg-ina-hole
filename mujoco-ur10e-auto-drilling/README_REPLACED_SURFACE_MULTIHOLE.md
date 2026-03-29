This version REPLACES the previous flat surface with a new panel surface and 5 associated hole frames.

Added:
- scene_replaced_surface_multihole_frames.xml
- hole_catalog.py
- multihole_runner.py
- README_REPLACED_SURFACE_MULTIHOLE.md

What changed relative to the previous multi-hole frame version:
- the old surface is not reused in the new scene
- a NEW panel surface is created
- 5 hole centers are defined directly on that new panel
- each hole has a compact, properly aligned XYZ frame
- the frames are small enough to be readable and not overlap heavily

Run:
python multihole_runner.py --hole hole_1 --sleep 0.15

Hide site frames if needed:
python multihole_runner.py --hole hole_1 --sleep 0.15 --hide-site-frames


Updated behavior:
- `multihole_runner.py` now drills ALL holes in sequence in one run:
  1. move to hole_1
  2. insert + drill
  3. move to hole_2
  4. insert + drill
  5. continue until hole_5

Run:
```bash
python multihole_runner.py --sleep 0.15
```
