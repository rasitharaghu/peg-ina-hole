import numpy as np

class LimitedInsertionController:
    '''
    Local insertion controller that reuses the original insertion idea:
    - local Cartesian target
    - Jacobian-based DLS update
    - direct incremental joint updates

    New addition:
    - hard travel limit so the tool cannot go too deep
    '''
    def __init__(self, robot, axis_world, max_travel, damping, gain, pos_tol):
        self.robot = robot
        self.axis_world = axis_world / np.linalg.norm(axis_world)
        self.max_travel = float(max_travel)
        self.damping = float(damping)
        self.gain = float(gain)
        self.pos_tol = float(pos_tol)

        self.start_pos = robot.get_ee_pos().copy()
        self.target_pos = self.start_pos + self.max_travel * self.axis_world

    def step(self):
        current = self.robot.get_ee_pos()
        traveled = np.dot(current - self.start_pos, self.axis_world)

        # hard stop on insertion travel
        if traveled >= self.max_travel:
            print(f"[INSERT] hard travel limit reached: {traveled:.6f} m")
            return True

        pos_err = self.target_pos - current

        if np.linalg.norm(pos_err) < self.pos_tol:
            print(f"[INSERT] local insertion target reached, err={np.linalg.norm(pos_err):.6f}")
            return True

        dq = self.robot.dls_pos(pos_err, damping=self.damping)
        self.robot.integrate_dq(dq, self.gain)
        return False
