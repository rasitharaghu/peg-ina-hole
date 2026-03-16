from __future__ import annotations
import argparse, numpy as np, mujoco
from src.utils import load_yaml
from src.logging_utils import setup_logger
from src.target_loader import load_targets
from src.kinematics import MuJoCoKinematics
from src.servo_controller import ServoController
from src.contact_wrench import ContactWrenchEstimator
from src.drilling_state_machine import DrillingContext, DrillingStateMachine, DrillState
from src.metrics import DrillMetrics, append_metrics_csv, write_summary_json
from src.plotting import plot_benchmark

class DummyDrillUnit:
    def __init__(self): self.drilling_complete=False; self.accumulated_dwell_s=0.0

def build_ctx(cfg: dict) -> DrillingContext:
    c=cfg['drilling']
    return DrillingContext(float(c['approach_offset']), float(c['predrill_offset']), float(c['contact_force_threshold']), float(c['thrust_force_contact']), float(c['thrust_force_drill']), float(c['lateral_force_limit']), float(c['axial_force_limit']), float(c['retract_distance']), float(c['feed_depth_tolerance']), int(c['max_retries']), bool(c.get('debug_approach_only', True)))

def run_one(cfg, target):
    logger=setup_logger(); model=mujoco.MjModel.from_xml_path(cfg['scene']['base_xml']); data=mujoco.MjData(model)
    controller=ServoController(**cfg['controller']); kin=MuJoCoKinematics(model, data, 'tcp'); wrench=ContactWrenchEstimator(model, data, 'tcp'); sm=DrillingStateMachine(build_ctx(cfg), logger); drill=DummyDrillUnit()
    home_q=np.array(cfg['robot']['home_q'], dtype=float); data.qpos[:6]=home_q; data.ctrl[:6]=home_q; mujoco.mj_forward(model,data); dt=float(cfg['simulation']['dt'])
    for step in range(int(cfg['simulation']['max_steps'])):
        task=kin.site_state(); force=wrench.estimate_world_wrench()[:3]
        state, x_des, desired_axis, desired_force_z, spindle_on = sm.update(target, task.position, force, drill.drilling_complete)
        q_des=controller.compute_q_des(task, data.qpos[:6].copy(), data.qvel[:6].copy(), x_des, desired_axis, force, home_q, dt)
        data.ctrl[:6]=q_des; mujoco.mj_step(model,data)
        if state in {DrillState.DONE, DrillState.FAIL}:
            return DrillMetrics(target.target_id, 1 if state==DrillState.DONE else 0, step, sm.retry_count, sm.peak_force_z, sm.peak_force_xy, 0.0, sm.jam_events)
    return DrillMetrics(target.target_id, 0, int(cfg['simulation']['max_steps']), sm.retry_count, sm.peak_force_z, sm.peak_force_xy, 0.0, sm.jam_events)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config', default='config/benchmark.yaml'); args=ap.parse_args(); cfg=load_yaml(args.config)
    rows=[]
    for target in load_targets(cfg['paths']['targets_file']):
        row=run_one(cfg,target); rows.append(row); append_metrics_csv(cfg['benchmark']['output_csv'], row)
    summary={'n_targets': len(rows), 'success_rate': float(sum(r.success for r in rows)/max(1,len(rows)))}
    write_summary_json(cfg['benchmark']['summary_json'], summary); plot_benchmark(cfg['benchmark']['output_csv'], cfg['benchmark']['plots_dir']); print(summary)

if __name__=='__main__': main()
