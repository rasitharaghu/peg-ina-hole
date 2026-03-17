import json
import time
import logging
import mujoco
import mujoco.viewer
import numpy as np

from simple_ik import SimpleIK
from drilling_samebase_state_machine import DrillingContext, DrillingStateMachine
from force_sensor_utils import world_force_from_site_sensor, project_force_along_axis
from force_feed_controller import ForceFeedController

XML_PATH = 'scene_drilling_samebase.xml'
SITE_NAME = 'drill_tcp'
MAX_Q_STEP = 0.01

def main():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
    logger = logging.getLogger("drill_samebase")

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    with open('drill_targets_samebase.json', 'r', encoding='utf-8') as f:
        target = json.load(f)["targets"][0]

    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    mujoco.mj_forward(model, data)

    site_id = model.site(SITE_NAME).id
    ik = SimpleIK(model, data, site_name=SITE_NAME)
    sm = DrillingStateMachine(DrillingContext(), logger=logger)
    ff = ForceFeedController(desired_force=DrillingContext().desired_drill_force)

    axis = np.array(target["normal"], dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            tcp_pos = data.site_xpos[site_id].copy()
            _, world_force = world_force_from_site_sensor(data, model, sensor_name='wrist_force', site_name='wrist_ft_sensor')
            measured_force = max(0.0, project_force_along_axis(world_force, axis))

            if sm.state.name in ['SEEK_CONTACT', 'FORCE_FEED', 'DWELL']:
                feed_offset = ff.update(measured_force)
            else:
                feed_offset = ff.feed

            state, x_des = sm.update(target=target, tcp_pos=tcp_pos, sim_time=data.time, measured_force=measured_force, feed_offset=feed_offset)

            q_current = data.qpos[:model.nu].copy()
            q_goal = ik.solve(x_des, q_current, max_iters=60, alpha=0.10)
            dq = np.clip(q_goal - q_current, -MAX_Q_STEP, MAX_Q_STEP)
            q_des = q_current + dq

            data.ctrl[:model.nu] = q_des
            mujoco.mj_step(model, data)
            viewer.sync()

            if state.name == 'DONE':
                logger.info("Finished same-base drilling debug")
                break

            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)

if __name__ == '__main__':
    main()
