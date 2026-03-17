from dataclasses import dataclass
from enum import Enum, auto
import numpy as np

class DrillState(Enum):
    MOVE_APPROACH = auto()
    MOVE_PREDRILL = auto()
    SEEK_CONTACT = auto()
    FORCE_FEED = auto()
    DWELL = auto()
    RETRACT = auto()
    DONE = auto()

@dataclass
class DrillingContext:
    approach_offset: float = 0.12
    predrill_offset: float = 0.05
    retract_offset: float = 0.12
    contact_force_threshold: float = 6.0
    desired_drill_force: float = 20.0
    max_safe_force: float = 70.0
    target_feed_depth: float = 0.008
    dwell_time: float = 1.0

class DrillingStateMachine:
    def __init__(self, ctx, logger):
        self.ctx = ctx
        self.logger = logger
        self.state = DrillState.MOVE_APPROACH
        self.phase_start_time = None
        self.depth_start = None

    def update(self, target, tcp_pos, sim_time, measured_force, feed_offset):
        axis = np.array(target["normal"], dtype=float)
        axis = axis / (np.linalg.norm(axis) + 1e-12)

        approach_pos = target["position"] - self.ctx.approach_offset * axis
        predrill_pos = target["position"] - self.ctx.predrill_offset * axis
        retract_pos = target["position"] - self.ctx.retract_offset * axis

        if self.state == DrillState.MOVE_APPROACH:
            x_des = approach_pos
            if np.linalg.norm(tcp_pos - approach_pos) < 0.02:
                self.logger.info("Reached approach pose")
                self.state = DrillState.MOVE_PREDRILL

        elif self.state == DrillState.MOVE_PREDRILL:
            x_des = predrill_pos
            if np.linalg.norm(tcp_pos - predrill_pos) < 0.015:
                self.logger.info("Reached predrill pose")
                self.state = DrillState.SEEK_CONTACT

        elif self.state == DrillState.SEEK_CONTACT:
            x_des = predrill_pos + axis * feed_offset
            if measured_force >= self.ctx.contact_force_threshold:
                self.logger.info("Contact detected")
                self.state = DrillState.FORCE_FEED
                self.depth_start = float(np.dot(tcp_pos, axis))

        elif self.state == DrillState.FORCE_FEED:
            x_des = predrill_pos + axis * feed_offset
            current_depth = float(np.dot(tcp_pos, axis))
            drilled_depth = current_depth - (self.depth_start if self.depth_start is not None else current_depth)
            if measured_force >= self.ctx.max_safe_force:
                self.logger.warning("Force too high, retracting")
                self.state = DrillState.RETRACT
            elif drilled_depth >= self.ctx.target_feed_depth:
                self.logger.info("Reached target depth")
                self.state = DrillState.DWELL
                self.phase_start_time = sim_time

        elif self.state == DrillState.DWELL:
            x_des = predrill_pos + axis * feed_offset
            if self.phase_start_time is not None and (sim_time - self.phase_start_time) >= self.ctx.dwell_time:
                self.logger.info("Dwell complete")
                self.state = DrillState.RETRACT

        elif self.state == DrillState.RETRACT:
            x_des = retract_pos
            if np.linalg.norm(tcp_pos - retract_pos) < 0.02:
                self.logger.info("Retracted safely")
                self.state = DrillState.DONE
        else:
            x_des = retract_pos

        return self.state, x_des
