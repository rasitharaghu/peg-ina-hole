import numpy as np

class AdmittanceInsertionController:
    def __init__(
        self,
        robot,
        start_tip_pos,
        axis_world,
        max_travel,
        nominal_step,
        damping,
        gain,
        lateral_compliance,
        axial_scale,
        control_rot_target,
        tip_offset_local,
    ):
        self.robot = robot
        self.start_tip_pos = start_tip_pos.copy()
        self.axis_world = axis_world / np.linalg.norm(axis_world)
        self.max_travel = float(max_travel)
        self.nominal_step = float(nominal_step)
        self.damping = float(damping)
        self.gain = float(gain)
        self.lateral_compliance = float(lateral_compliance)
        self.axial_scale = float(axial_scale)
        self.control_rot_target = control_rot_target.copy()
        self.tip_offset_local = tip_offset_local.copy()

        self.travel_cmd = 0.0

    def step(self):
        current_tip = self.robot.get_tip_pos()
        current_control = self.robot.get_control_pos()

        self.travel_cmd = min(self.max_travel, self.travel_cmd + self.nominal_step)
        axial_target = self.start_tip_pos + self.travel_cmd * self.axis_world

        tip_err = axial_target - current_tip
        lateral_component = tip_err - np.dot(tip_err, self.axis_world) * self.axis_world
        corrected_tip_target = axial_target - self.lateral_compliance * lateral_component

        target_control_pos = corrected_tip_target - self.control_rot_target @ self.tip_offset_local
        control_err = target_control_pos - current_control

        dq = self.robot.dls_pos_control(self.axial_scale * control_err, damping=self.damping)
        self.robot.integrate_dq(dq, self.gain)

        traveled = np.dot(current_tip - self.start_tip_pos, self.axis_world)
        done = traveled >= self.max_travel * 0.98 or self.travel_cmd >= self.max_travel
        return done, traveled, np.linalg.norm(lateral_component), corrected_tip_target
