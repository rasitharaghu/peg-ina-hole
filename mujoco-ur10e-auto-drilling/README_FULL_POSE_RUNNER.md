# Full-pose sequential runner for MuJoCo UR10e drilling debug

This version adds orientation control on top of the earlier simple sequential runner.

Sequence:
1. Set a clearly far custom home joint pose
2. Move to a pre-approach pose (position + orientation)
3. Move to the hole pose (position + orientation)
4. Hold the viewer open so you can inspect

Recommended first:
python full_pose_sequential_runner.py --debug-forward --sleep 0.03
