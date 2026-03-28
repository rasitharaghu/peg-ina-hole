import numpy as np

class SimpleDrillController:
    def __init__(
        self,
        robot,
        start_tip_pos,
        axis_world,
        extra_depth,
        nominal_step,
        damping,
        gain,
        control_rot_target,
        tip_offset_local,
    ):
        self.robot = robot
        self.start_tip_pos = start_tip_pos.copy()
        self.axis_world = axis_world / np.linalg.norm(axis_world)
        self.extra_depth = float(extra_depth)
        self.nominal_step = float(nominal_step)
        self.damping = float(damping)
        self.gain = float(gain)
        self.control_rot_target = control_rot_target.copy()
        self.tip_offset_local = tip_offset_local.copy()
        self.feed_cmd = 0.0

    def step(self):
        current_tip = self.robot.get_tip_pos()
        current_control = self.robot.get_control_pos()

        self.feed_cmd = min(self.extra_depth, self.feed_cmd + self.nominal_step)
        target_tip = self.start_tip_pos + self.feed_cmd * self.axis_world
        target_control_pos = target_tip - self.control_rot_target @ self.tip_offset_local

        control_err = target_control_pos - current_control
        dq = self.robot.dls_pos_control(control_err, damping=self.damping)
        self.robot.integrate_dq(dq, self.gain)

        traveled = np.dot(current_tip - self.start_tip_pos, self.axis_world)
        done = traveled >= self.extra_depth * 0.98 or self.feed_cmd >= self.extra_depth
        return done, traveled, target_tip
