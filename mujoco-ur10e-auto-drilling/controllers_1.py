import time
import numpy as np

from bt import Status
from config_bt import (
    DLS_DAMPING_6D,
    DLS_DAMPING_3D,
    SERVO_GAIN,
    SERVO_POS_TOL,
    SERVO_ORI_GAIN,
    K_POS,
    K_FORCE,
    K_TORQUE,
    INSERT_INTEGRATION_GAIN,
    FORCE_LIMIT_N,
    TORQUE_LIMIT_NM,
    DRILL_DURATION,
    DRILL_FEED_SPEED,
    DRILL_HOLD_FORCE,
    DRILL_FORCE_GAIN,
)

def orientation_error_vec(r_des, r_cur):
    r_err = r_des @ r_cur.T
    return 0.5 * np.array([
        r_err[2, 1] - r_err[1, 2],
        r_err[0, 2] - r_err[2, 0],
        r_err[1, 0] - r_err[0, 1],
    ])

class CartesianServoController:
    def __init__(self, robot, target_pos, target_rot=None):
        self.robot = robot
        self.target_pos = np.array(target_pos, dtype=float)
        self.target_rot = target_rot

    def tick(self):
        pos = self.robot.get_ee_pos()
        pos_err = self.target_pos - pos

        print("[SERVO] pos:", np.round(pos, 4),
              "target:", np.round(self.target_pos, 4),
              "err_norm:", round(float(np.linalg.norm(pos_err)), 6))

        if self.target_rot is None:
            dq = self.robot.compute_dls_step(pos_err, full=False, damping=DLS_DAMPING_3D)
            print("[SERVO] dq:", np.round(dq, 6))
            self.robot.integrate_dq(dq, SERVO_GAIN)
            if np.linalg.norm(pos_err) < SERVO_POS_TOL:
                print("[SERVO] reached position target")
                return Status.SUCCESS
            return Status.RUNNING

        rot = self.robot.get_ee_rot()
        ori_err = orientation_error_vec(self.target_rot, rot)
        task = np.concatenate([pos_err, SERVO_ORI_GAIN * ori_err])
        dq = self.robot.compute_dls_step(task, full=True, damping=DLS_DAMPING_6D)
        print("[SERVO] ori_err_norm:", round(float(np.linalg.norm(ori_err)), 6),
              "dq:", np.round(dq, 6))
        self.robot.integrate_dq(dq, SERVO_GAIN)
        if np.linalg.norm(pos_err) < SERVO_POS_TOL and np.linalg.norm(ori_err) < 0.02:
            print("[SERVO] reached pose target")
            return Status.SUCCESS
        return Status.RUNNING

class AdmittanceInsertController:
    def __init__(self, robot, start_pos, insertion_depth, duration):
        self.robot = robot
        self.start_pos = np.array(start_pos, dtype=float)
        self.insertion_depth = float(insertion_depth)
        self.duration = float(duration)
        self.started = False
        self.t0 = None

    def tick(self):
        if not self.started:
            self.started = True
            self.t0 = time.time()
            self.robot.zero_ft_bias()

        elapsed = time.time() - self.t0
        s = min(elapsed / self.duration, 1.0)

        target = self.start_pos.copy()
        target[0] += s * self.insertion_depth

        pos = self.robot.get_ee_pos()
        pos_err = target - pos

        force = self.robot.get_force()
        torque = self.robot.get_torque()

        task = (K_POS * pos_err) - (K_FORCE * force)
        task[1] += -K_TORQUE[1] * torque[1]
        task[2] += -K_TORQUE[2] * torque[2]

        print("[INSERT] s:", round(float(s), 4),
              "pos_err:", np.round(pos_err, 5),
              "force:", np.round(force, 4),
              "torque:", np.round(torque, 4))

        dq = self.robot.compute_dls_step(task, full=False, damping=DLS_DAMPING_3D)
        self.robot.integrate_dq(dq, INSERT_INTEGRATION_GAIN)

        if self.robot.force_limit_exceeded(FORCE_LIMIT_N, TORQUE_LIMIT_NM):
            print("[INSERT] force/torque limit exceeded")
            return Status.FAILURE

        if s >= 1.0 and np.linalg.norm(pos_err) < 0.004:
            print("[INSERT] insertion target reached")
            return Status.SUCCESS
        return Status.RUNNING

class DrillFeedController:
    def __init__(self, robot, duration=DRILL_DURATION):
        self.robot = robot
        self.duration = duration
        self.t0 = None
        self.start_pos = None

    def tick(self):
        if self.t0 is None:
            self.t0 = time.time()
            self.start_pos = self.robot.get_ee_pos().copy()

        elapsed = time.time() - self.t0
        if elapsed >= self.duration:
            print("[DRILL] duration complete")
            return Status.SUCCESS

        force = self.robot.get_force()
        target = self.start_pos.copy()
        target[0] += -elapsed * DRILL_FEED_SPEED
        pos = self.robot.get_ee_pos()
        pos_err = target - pos

        task = np.zeros(3)
        task[0] = 80.0 * pos_err[0] + DRILL_FORCE_GAIN * (DRILL_HOLD_FORCE - abs(force[0]))
        task[1] = -0.08 * force[1]
        task[2] = -0.08 * force[2]

        print("[DRILL] elapsed:", round(float(elapsed), 4),
              "pos_err:", np.round(pos_err, 5),
              "force:", np.round(force, 4))

        dq = self.robot.compute_dls_step(task, full=False, damping=DLS_DAMPING_3D)
        self.robot.integrate_dq(dq, INSERT_INTEGRATION_GAIN)

        if self.robot.force_limit_exceeded(FORCE_LIMIT_N, TORQUE_LIMIT_NM):
            print("[DRILL] force/torque limit exceeded")
            return Status.FAILURE
        return Status.RUNNING
