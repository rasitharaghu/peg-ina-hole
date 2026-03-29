This package leaves all original file names untouched.

New files only:
- common_control_new.py
- move_to_hole_new.py
- insert_peg_new.py
- drill_feed_new.py
- run_all_new.py

Key fix:
- desired hole target is treated as a PEG TIP target
- corresponding attachment_site target is computed internally using the home-frame tip offset

Recommended run:
python run_all_new.py --sleep 0.15
