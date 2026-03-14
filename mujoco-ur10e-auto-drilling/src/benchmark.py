from __future__ import annotations

import argparse
import numpy as np
import mujoco

from src.utils import load_yaml, unit
from src.logging_utils import setup_logger
from src.target_loader import load_targets, DrillTarget
from src.kinematics import MuJoCoKinematics
from src.hybrid_controller import HybridForcePositionController
from src.contact_wrench import ContactWrenchEstimator
from src.drill_unit import AutomaticDrillingUnit
from src.drilling_state_machine import DrillingContext, DrillingStateMachine, DrillState
from src.metrics import DrillMetrics, append_metrics_csv, write_summary_json
from src.plotting import plot_benchmark


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


def noisy_target(target: DrillTarget, pos_noise_mm: float, normal_noise_deg: float, rng: np.random.Generator) -> DrillTarget:
    pos_noise = rng.uniform(-pos_noise_mm, pos_noise_mm, size=3) * 1e-3
    n = target.normal.copy()
    tilt = rng.uniform(-normal_noise_deg, normal_noise_deg, size=2) * np.pi / 180.0
    rx = np.array([[1, 0, 0], [0, np.cos(tilt[0]), -np.sin(tilt[0])], [0, np.sin(tilt[0]), np.cos(tilt[0])]], dtype=float)
    ry = np.array([[np.cos(tilt[1]), 0, np.sin(tilt[1])], [0, 1, 0], [-np.sin(tilt[1]), 0, np.cos(tilt[1])]], dtype=float)
    n2 = unit(ry @ rx @ n)
    return DrillTarget(
        target_id=target.target_id,
        position=target.position + pos_noise,
        normal=n2,
        depth=target.depth,
        diameter=target.diameter,
    )


def run_target(cfg: dict, target: DrillTarget) -> DrillMetrics:
    logger = setup_logger()
    model = mujoco.MjModel.from_xml_path(cfg["scene"]["base_xml"])
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

        drill_unit.update(dt=dt, in_cut=(state == DrillState.DRILL))

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

        if state in {DrillState.DONE, DrillState.FAIL}:
            return DrillMetrics(
                target_id=target.target_id,
                success=1 if state == DrillState.DONE else 0,
                steps=step,
                retries=sm.retry_count,
                peak_force_z=sm.peak_force_z,
                peak_force_xy=sm.peak_force_xy,
                spindle_time_s=drill_unit.state.accumulated_dwell_s,
                jam_events=sm.jam_events,
            )

    return DrillMetrics(
        target_id=target.target_id,
        success=0,
        steps=max_steps,
        retries=sm.retry_count,
        peak_force_z=sm.peak_force_z,
        peak_force_xy=sm.peak_force_xy,
        spindle_time_s=drill_unit.state.accumulated_dwell_s,
        jam_events=sm.jam_events,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/benchmark.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    rng = np.random.default_rng(42)
    targets = load_targets(cfg["paths"]["targets_file"])
    rows = []

    for target in targets:
        noisy = noisy_target(
            target,
            pos_noise_mm=float(cfg["benchmark"]["target_position_noise_mm"]),
            normal_noise_deg=float(cfg["benchmark"]["target_normal_noise_deg"]),
            rng=rng,
        )
        row = run_target(cfg, noisy)
        rows.append(row)
        append_metrics_csv(cfg["benchmark"]["output_csv"], row)

    summary = {
        "n_targets": len(rows),
        "success_rate": float(sum(r.success for r in rows) / max(len(rows), 1)),
        "mean_steps": float(sum(r.steps for r in rows) / max(len(rows), 1)),
        "mean_peak_force_z": float(sum(r.peak_force_z for r in rows) / max(len(rows), 1)),
        "mean_peak_force_xy": float(sum(r.peak_force_xy for r in rows) / max(len(rows), 1)),
        "mean_spindle_time_s": float(sum(r.spindle_time_s for r in rows) / max(len(rows), 1)),
        "mean_jam_events": float(sum(r.jam_events for r in rows) / max(len(rows), 1)),
    }
    write_summary_json(cfg["benchmark"]["summary_json"], summary)
    plot_benchmark(cfg["benchmark"]["output_csv"], cfg["benchmark"]["plots_dir"])
    print(summary)


if __name__ == "__main__":
    main()
