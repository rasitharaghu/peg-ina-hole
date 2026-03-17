import numpy as np

class ForceFeedController:
    def __init__(self, desired_force=20.0, kp_force=0.00006, max_feed_step=0.0004, min_feed=-0.002, max_feed=0.015):
        self.desired_force = desired_force
        self.kp_force = kp_force
        self.max_feed_step = max_feed_step
        self.min_feed = min_feed
        self.max_feed = max_feed
        self.feed = 0.0

    def reset(self):
        self.feed = 0.0

    def update(self, measured_force):
        err = self.desired_force - measured_force
        delta = self.kp_force * err
        delta = float(np.clip(delta, -self.max_feed_step, self.max_feed_step))
        self.feed = float(np.clip(self.feed + delta, self.min_feed, self.max_feed))
        return self.feed
