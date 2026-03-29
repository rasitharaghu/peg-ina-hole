Multi-hole panel with visible hole frames (XYZ axes), aligned insertion axis, and multi-hole runner.

Added:
- scene_multihole_frames.xml
- hole_catalog.py
- multihole_runner.py

What is visible:
- 5 hole frames
- each hole frame has RGB axis markers
- site frames can also be shown from the viewer

Axis convention used in XML frames:
- Red axis = local X
- Green axis = local Y
- Blue axis = local Z
- hole center site uses zaxis="1 0 0", so the blue axis points along the insertion direction

Run:
python multihole_runner.py --hole hole_1 --sleep 0.15

Other holes:
python multihole_runner.py --hole hole_2 --sleep 0.15
python multihole_runner.py --hole hole_3 --sleep 0.15
python multihole_runner.py --hole hole_4 --sleep 0.15
python multihole_runner.py --hole hole_5 --sleep 0.15

Hide site frames if too cluttered:
python multihole_runner.py --hole hole_1 --sleep 0.15 --hide-site-frames
