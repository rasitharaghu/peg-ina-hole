import json
import logging
import time
import mujoco
import mujoco.viewer
import numpy as np

XML_PATH = 'scene_drilling_exact.xml'
DURATION_TO_HOLE = 6.0
DWELL_TIME = 1.0
DURATION_RETRACT = 4.0


def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def main():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
    logger = logging.getLogger("drilling_exact_jointspace")

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    home_id = model.key('home').id
    hole_id = model.key('hole').id

    home_q = model.key_qpos[home_id][:model.nu].copy()
    hole_q = model.key_qpos[hole_id][:model.nu].copy()

    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    phase = 'move_to_hole'
    phase_start = time.time()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            elapsed = time.time() - phase_start

            if phase == 'move_to_hole':
                s = smoothstep(elapsed / DURATION_TO_HOLE)
                q_des = home_q + s * (hole_q - home_q)
                data.ctrl[:model.nu] = q_des

                if s >= 1.0:
                    logger.info("Reached hole pose")
                    phase = 'dwell'
                    phase_start = time.time()

            elif phase == 'dwell':
                data.ctrl[:model.nu] = hole_q
                if elapsed >= DWELL_TIME:
                    logger.info("Dwell complete")
                    phase = 'retract'
                    phase_start = time.time()

            elif phase == 'retract':
                s = smoothstep(elapsed / DURATION_RETRACT)
                q_des = hole_q + s * (home_q - hole_q)
                data.ctrl[:model.nu] = q_des

                if s >= 1.0:
                    logger.info("Retracted to home")
                    logger.info("Finished exact-same-scene drilling debug")
                    break

            mujoco.mj_step(model, data)
            viewer.sync()

            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


if __name__ == '__main__':
    main()