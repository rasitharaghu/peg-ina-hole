import numpy as np

class LimitedInsertionController:
    """
    Frame-based limited insertion controller.
    Controls the CONTROL SITE position so that the TIP reaches the desired insert target,
    while the runner keeps the same desired orientation.
    """
    def __init__(self, start_tip_pos, axis_world, max_travel):
        self.start_tip_pos = start_tip_pos.copy()
        self.axis_world = axis_world / np.linalg.norm(axis_world)
        self.max_travel = float(max_travel)

    def get_tip_target(self):
        return self.start_tip_pos + self.max_travel * self.axis_world
