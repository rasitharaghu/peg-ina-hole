from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DrillUnitState:
    spindle_on: bool = False
    spindle_speed_rpm: float = 0.0
    drilling_complete: bool = False
    accumulated_dwell_s: float = 0.0


class AutomaticDrillingUnit:
    def __init__(self, nominal_rpm: float, dwell_time_s: float):
        self.nominal_rpm = nominal_rpm
        self.dwell_time_s = dwell_time_s
        self.state = DrillUnitState()

    def spindle_start(self) -> None:
        self.state.spindle_on = True
        self.state.spindle_speed_rpm = self.nominal_rpm

    def spindle_stop(self) -> None:
        self.state.spindle_on = False
        self.state.spindle_speed_rpm = 0.0

    def reset(self) -> None:
        self.state = DrillUnitState()

    def update(self, dt: float, in_cut: bool) -> None:
        if self.state.spindle_on and in_cut:
            self.state.accumulated_dwell_s += dt
            if self.state.accumulated_dwell_s >= self.dwell_time_s:
                self.state.drilling_complete = True
