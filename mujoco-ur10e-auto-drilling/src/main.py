from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer

from src.utils import load_yaml
from src.logging_utils import setup_logger
from src.target_loader import load_targets
from src.kinematics import MuJoCoKinematics
from src.hybrid_controller import HybridForcePositionController
from src.contact_wrench import ContactWrenchEstimator
from src.drill_unit import AutomaticDrillingUnit
from src.drilling_state_machine import DrillingContext, DrillingStateMachine, DrillState


def build_drilling_context(cfg: dict) -> DrillingContext:
    c = cfg["drilling"]
    return DrillingContext(
        approach_offset=float(c["approach_offset"]),
        predrill_offset=float(c["predrill_offset"]),
        contact_force_threshold=float(c["contact_force_threshold"]),
        thrust_force_contact=float(c["thrust_force_contact"]),
        thrust_force_drill=float(c["thrust_force_drill"]),
        lateral_force_limit=float(c["lateral_force_limit"]),
        axial_force_limit=float(c["axial_force_limit"]),
        retract_distance=float(c["retract_distance"]),
        feed_depth_tolerance=float(c["feed_depth_tolerance"]),
        max_retries=int(c["max_retries"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/drilling.yaml")
    parser.add_argument("--target-id", type=int, default=0)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    logger = setup_logger()
    targets = load_targets(cfg["paths"]["targets_file"])
    target = next(t for t in targets if t.target_id == args.target_id)

    model = mujoco.MjModel.from_xml_path(str(Path(cfg["scene"]["base_xml"])))
    data = mujoco.MjData(model)

    controller = HybridForcePositionController(**cfg["controller"])
    kin = MuJoCoKinematics(model, data, site_name="tcp")
    wrench_estimator = ContactWrenchEstimator(model, data, tcp_site_name="tcp")
    drill_unit = AutomaticDrillingUnit(
        nominal_rpm=float(cfg["drilling"]["spindle_speed_rpm"]),
        dwell_time_s=float(cfg["drilling"]["dwell_time_s"]),
    )
    sm = DrillingStateMachine(build_drilling_context(cfg), logger)

    home_q = np.array(cfg["robot"]["home_q"], dtype=float)
    data.qpos[:6] = home_q
    mujoco.mj_forward(model, data)

    viewer = None
    if cfg["simulation"]["render"]:
        viewer = mujoco.viewer.launch_passive(model, data)

    dt = float(cfg["simulation"]["dt"])
    max_steps = int(cfg["simulation"]["max_steps"])

    for step in range(max_steps):
        task = kin.site_state()
        wrench = wrench_estimator.estimate_world_wrench()
        force = wrench[:3]

        state, x_des, desired_axis, desired_force_z, spindle_on = sm.update(
            target=target,
            tcp_pos=task.position,
            measured_force=force,
            drill_complete=drill_unit.state.drilling_complete,
        )

        if spindle_on:
            drill_unit.spindle_start()
        else:
            drill_unit.spindle_stop()

        in_cut = state == DrillState.DRILL
        drill_unit.update(dt=dt, in_cut=in_cut)

        controller.desired_force_z = desired_force_z
        tau = controller.compute(
            model=model,
            data=data,
            task=task,
            x_des=x_des,
            desired_axis=desired_axis,
            measured_force=force,
            home_q=home_q,
        )

        data.ctrl[:] = np.clip(tau / float(cfg["controller"]["max_tau"]), -1.0, 1.0)
        mujoco.mj_step(model, data)

        if viewer is not None:
            viewer.sync()

        if state in {DrillState.DONE, DrillState.FAIL}:
            logger.info(
                "Finished target=%d state=%s spindle_time=%.3f s",
                target.target_id,
                state.name,
                drill_unit.state.accumulated_dwell_s,
            )
            break

    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
