# Fixed sequential multihole project

This version includes:
- perpendicular planar panel
- larger hole spacing
- medium-small visible hole frames
- built-in MuJoCo site frames OFF by default
- smooth retract/transition to next hole
- no reset to home between holes

Run:
```bash
python multihole_runner.py --sleep 0.15
```

Optional large viewer overlay:
```bash
python multihole_runner.py --sleep 0.15 --show-site-frames
```
