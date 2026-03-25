from bt import Status
from controllers import CartesianServoController, AdmittanceInsertController, DrillFeedController
from config_bt import (
    HOLE_TARGET_POS,
    PREAPPROACH_OFFSET,
    INSERTION_DEPTH,
    INSERT_DURATION,
    RETRACT_DISTANCE,
    USE_CUSTOM_HOME_QPOS,
    CUSTOM_HOME_QPOS,
)
from adu_process import ADUStartAction, ADUStopAction


class ResetHome:
    def __init__(self, robot):
        self.robot = robot
        self.done = False

    def tick(self):
        if not self.done:
            if USE_CUSTOM_HOME_QPOS:
                self.robot.set_qpos(CUSTOM_HOME_QPOS)
                print("[BT] Reset to CUSTOM far home qpos:", CUSTOM_HOME_QPOS)
            else:
                self.robot.reset_to_home()
                print("[BT] Reset to home keyframe")
            self.done = True
        return Status.SUCCESS


class MoveToPreApproach:
    def __init__(self, robot):
        self.robot = robot
        self.controller = None

    def tick(self):
        if self.controller is None:
            pre = HOLE_TARGET_POS + PREAPPROACH_OFFSET
            self.controller = CartesianServoController(self.robot, pre, None)
            print("[BT] Move to pre-approach target:", pre)
        return self.controller.tick()


class CartesianApproach:
    def __init__(self, robot):
        self.robot = robot
        self.controller = None

    def tick(self):
        if self.controller is None:
            self.controller = CartesianServoController(self.robot, HOLE_TARGET_POS, None)
            print("[BT] Final Cartesian approach target:", HOLE_TARGET_POS)
        return self.controller.tick()


class AdmittanceInsert:
    def __init__(self, robot):
        self.robot = robot
        self.controller = None

    def tick(self):
        if self.controller is None:
            start = self.robot.get_ee_pos().copy()
            self.controller = AdmittanceInsertController(
                self.robot, start, INSERTION_DEPTH, INSERT_DURATION
            )
            print("[BT] Admittance insertion start:", start)
        return self.controller.tick()


class Drill:
    def __init__(self, robot, adu):
        self.robot = robot
        self.adu = adu
        self.start_action = ADUStartAction(adu)
        self.feed_controller = DrillFeedController(robot)
        self.stop_action = ADUStopAction(adu)
        self.phase = 0

    def tick(self):
        if self.phase == 0:
            self.start_action.tick()
            self.phase = 1
            return Status.RUNNING
        if self.phase == 1:
            status = self.feed_controller.tick()
            if status == Status.SUCCESS:
                self.adu.mark_depth_reached()
                self.phase = 2
                return Status.RUNNING
            if status == Status.FAILURE:
                return Status.FAILURE
            return Status.RUNNING
        self.stop_action.tick()
        print("[BT] Drill phase complete")
        return Status.SUCCESS


class Retract:
    def __init__(self, robot):
        self.robot = robot
        self.controller = None

    def tick(self):
        if self.controller is None:
            target = self.robot.get_ee_pos().copy()
            target[0] += RETRACT_DISTANCE
            self.controller = CartesianServoController(self.robot, target, None)
            print("[BT] Retract target:", target)
        return self.controller.tick()


class SafeAbortToHome:
    def __init__(self, robot):
        self.robot = robot
        self.done = False

    def tick(self):
        if not self.done:
            if USE_CUSTOM_HOME_QPOS:
                self.robot.set_qpos(CUSTOM_HOME_QPOS)
                print("[BT] Safe abort to CUSTOM far home qpos")
            else:
                self.robot.reset_to_home()
                print("[BT] Safe abort to home")
            self.done = True
        return Status.SUCCESS
