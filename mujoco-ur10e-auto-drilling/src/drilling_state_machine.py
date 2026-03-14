from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import numpy as np

from src.target_loader import DrillTarget
from src.utils import unit


class DrillState(Enum):
    MOVE_APPROACH = auto()
    MOVE_PREDRILL = auto()
    SEEK_CONTACT = auto()
    DRILL = auto()
    RETRACT = auto()
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


class DrillingStateMachine:
    def __init__(self, ctx: DrillingContext, logger):
        self.ctx = ctx
        self.logger = logger
        self.state = DrillState.MOVE_APPROACH
        self.retry_count = 0
        self.peak_force_z = 0.0
        self.peak_force_xy = 0.0
        self.contact_z = None
        self.jam_events = 0
        self._retract_start = None

    def update(
        self,
        target: DrillTarget,
        tcp_pos: np.ndarray,
        measured_force: np.ndarray,
        drill_complete: bool,
    ) -> tuple[DrillState, np.ndarray, np.ndarray, float, bool]:
        force_xy = float(np.linalg.norm(measured_force[:2]))
        force_z = abs(measured_force[2])
        self.peak_force_xy = max(self.peak_force_xy, force_xy)
        self.peak_force_z = max(self.peak_force_z, force_z)

        desired_axis = unit(-target.normal)
        x_des = tcp_pos.copy()
        desired_force_z = 0.0
        spindle_on = False

        approach_pos = target.position - target.normal * self.ctx.approach_offset
        predrill_pos = target.position - target.normal * self.ctx.predrill_offset

        if self.state == DrillState.MOVE_APPROACH:
            x_des = approach_pos
            if np.linalg.norm(tcp_pos - approach_pos) < 5e-3:
                self.state = DrillState.MOVE_PREDRILL

        elif self.state == DrillState.MOVE_PREDRILL:
            x_des = predrill_pos
            if np.linalg.norm(tcp_pos - predrill_pos) < 4e-3:
                self.state = DrillState.SEEK_CONTACT

        elif self.state == DrillState.SEEK_CONTACT:
            x_des = predrill_pos.copy()
            desired_force_z = self.ctx.thrust_force_contact
            if force_z > self.ctx.contact_force_threshold:
                self.contact_z = float(tcp_pos[2])
                self.state = DrillState.DRILL

        elif self.state == DrillState.DRILL:
            x_des = target.position.copy()
            desired_force_z = self.ctx.thrust_force_drill
            spindle_on = True

            if force_xy > self.ctx.lateral_force_limit or force_z > self.ctx.axial_force_limit:
                self.jam_events += 1
                self.state = DrillState.RETRACT
                self._retract_start = tcp_pos.copy()
                x_des = tcp_pos.copy()
                x_des[2] += self.ctx.retract_distance
                desired_force_z = 0.0
                spindle_on = False
            elif drill_complete:
                self.state = DrillState.RETRACT
                self._retract_start = tcp_pos.copy()
                x_des = tcp_pos.copy()
                x_des[2] += self.ctx.retract_distance
                desired_force_z = 0.0
                spindle_on = False

        elif self.state == DrillState.RETRACT:
            if self._retract_start is None:
                self._retract_start = tcp_pos.copy()
            x_des = self._retract_start.copy()
            x_des[2] += self.ctx.retract_distance
            desired_force_z = 0.0
            if np.linalg.norm(tcp_pos - x_des) < 5e-3:
                if self.jam_events > 0 and self.retry_count < self.ctx.max_retries:
                    self.retry_count += 1
                    self.state = DrillState.MOVE_APPROACH
                elif self.jam_events > 0 and self.retry_count >= self.ctx.max_retries:
                    self.state = DrillState.FAIL
                else:
                    self.state = DrillState.DONE

        return self.state, x_des, desired_axis, desired_force_z, spindle_on
