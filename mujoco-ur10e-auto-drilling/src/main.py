from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, mujoco, mujoco.viewer
from src.utils import load_yaml
from src.logging_utils import setup_logger
from src.target_loader import load_targets
from src.kinematics import MuJoCoKinematics
from src.servo_controller import ServoController
from src.contact_wrench import ContactWrenchEstimator
from src.drilling_state_machine import DrillingContext, DrillingStateMachine, DrillState

class DummyDrillUnit:
    def __init__(self): self.drilling_complete=False; self.accumulated_dwell_s=0.0

def build_ctx(cfg: dict) -> DrillingContext:
    c=cfg['drilling']
    return DrillingContext(float(c['approach_offset']), float(c['predrill_offset']), float(c['contact_force_threshold']), float(c['thrust_force_contact']), float(c['thrust_force_drill']), float(c['lateral_force_limit']), float(c['axial_force_limit']), float(c['retract_distance']), float(c['feed_depth_tolerance']), int(c['max_retries']), bool(c.get('debug_approach_only', True)))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config', default='config/drilling.yaml'); ap.add_argument('--target-id', type=int, default=0); args=ap.parse_args()
    cfg=load_yaml(args.config); logger=setup_logger(); targets=load_targets(cfg['paths']['targets_file']); target=next(t for t in targets if t.target_id==args.target_id)
    model=mujoco.MjModel.from_xml_path(str(Path(cfg['scene']['base_xml']))); data=mujoco.MjData(model)
    controller=ServoController(**cfg['controller']); kin=MuJoCoKinematics(model, data, 'tcp'); wrench=ContactWrenchEstimator(model, data, 'tcp'); sm=DrillingStateMachine(build_ctx(cfg), logger); drill=DummyDrillUnit()
    home_q=np.array(cfg['robot']['home_q'], dtype=float); data.qpos[:6]=home_q; data.ctrl[:6]=home_q; mujoco.mj_forward(model,data)
    viewer = mujoco.viewer.launch_passive(model, data) if cfg['simulation']['render'] else None
    dt=float(cfg['simulation']['dt'])
    for step in range(int(cfg['simulation']['max_steps'])):
        task=kin.site_state(); force=wrench.estimate_world_wrench()[:3]
        state, x_des, desired_axis, desired_force_z, spindle_on = sm.update(target, task.position, force, drill.drilling_complete)
        q_des=controller.compute_q_des(task, data.qpos[:6].copy(), data.qvel[:6].copy(), x_des, desired_axis, force, home_q, dt)
        data.ctrl[:6]=q_des; mujoco.mj_step(model,data)
        if viewer is not None: viewer.sync()
        if state in {DrillState.DONE, DrillState.FAIL}:
            logger.info('Finished state=%s step=%d', state.name, step)
            break
    if viewer is not None: viewer.close()

if __name__=='__main__': main()
