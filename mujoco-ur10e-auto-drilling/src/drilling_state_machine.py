from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
import numpy as np
from src.target_loader import DrillTarget

class DrillState(Enum):
    MOVE_APPROACH = auto()
    MOVE_PREDRILL = auto()
    DRILL = auto()
    DONE = auto()
    FAIL = auto()

@dataclass
class DrillingContext:
    approach_offset: float
    predrill_offset: float
    contact_force_threshold: float
    thrust_force_contact: float
    thrust_force_drill: float
    lateral_force_limit: float
    axial_force_limit: float
    retract_distance: float
    feed_depth_tolerance: float
    max_retries: int
    debug_approach_only: bool = True

class DrillingStateMachine:
    def __init__(self, ctx: DrillingContext, logger):
        self.ctx=ctx; self.logger=logger; self.state=DrillState.MOVE_APPROACH
        self.retry_count=0; self.peak_force_z=0.0; self.peak_force_xy=0.0; self.jam_events=0
    def update(self, target: DrillTarget, tcp_pos: np.ndarray, measured_force: np.ndarray, drill_complete: bool):
        desired_axis = np.array([1.0, 0.0, 0.0], dtype=float)
        approach_pos = target.position + np.array([-self.ctx.approach_offset, 0.0, 0.0], dtype=float)
        predrill_pos = target.position + np.array([-self.ctx.predrill_offset, 0.0, 0.0], dtype=float)
        x_des = tcp_pos.copy(); desired_force_z = 0.0; spindle_on = False
        if self.state == DrillState.MOVE_APPROACH:
            x_des = approach_pos
            if np.linalg.norm(tcp_pos - approach_pos) < 0.02:
                self.logger.info('Reached approach pose')
                self.state = DrillState.MOVE_PREDRILL
        elif self.state == DrillState.MOVE_PREDRILL:
            x_des = predrill_pos
            if np.linalg.norm(tcp_pos - predrill_pos) < 0.02:
                self.logger.info('Reached predrill pose')
                if self.ctx.debug_approach_only:
                    self.state = DrillState.DONE
                else:
                    self.state = DrillState.DRILL
        elif self.state == DrillState.DRILL:
            x_des = predrill_pos
            desired_force_z = self.ctx.thrust_force_contact
            spindle_on = True
            if drill_complete:
                self.state = DrillState.DONE
        return self.state, x_des, desired_axis, desired_force_z, spindle_on
