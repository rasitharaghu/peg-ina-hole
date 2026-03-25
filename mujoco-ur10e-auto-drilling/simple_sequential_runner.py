import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer

from simple_runner_config import (
    XML_PATH,
    CUSTOM_HOME_QPOS,
    HOLE_TARGET_POS,
    PREAPPROACH_OFFSET,
    DLS_DAMPING_3D,
    SERVO_GAIN,
    POS_TOL,
    MAX_STEPS_PER_PHASE,
    HOLD_VIEWER_STEPS,
)
from simple_runner_mujoco_adapter import MujocoRobot

def step_to_target(robot, target_pos, label, debug_forward=False, viewer=None):
    print(f"\n[PHASE] {label}")
    print("[PHASE] target:", target_pos)

    for i in range(MAX_STEPS_PER_PHASE):
        pos = robot.get_ee_pos()
        err = target_pos - pos
        err_norm = float(np.linalg.norm(err))

        if i % 25 == 0:
            print(f"[{label}] step={i} pos={np.round(pos,4)} err_norm={err_norm:.6f}")

        if err_norm < POS_TOL:
            print(f"[{label}] reached target with err_norm={err_norm:.6f}")
            return True

        dq = robot.dls_pos(err, damping=DLS_DAMPING_3D)
        robot.integrate_dq(dq, SERVO_GAIN)

        if debug_forward:
            mujoco.mj_forward(robot.model, robot.data)
        else:
            robot.step_sim()

        if viewer is not None:
            viewer.sync()

    print(f"[{label}] failed to converge within {MAX_STEPS_PER_PHASE} steps")
    return False

def main(headless=False, debug_forward=False):
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    robot = MujocoRobot(model, data)

    preapproach = HOLE_TARGET_POS + PREAPPROACH_OFFSET

    print("[RUN] XML_PATH:", XML_PATH)
    print("[RUN] CUSTOM_HOME_QPOS:", CUSTOM_HOME_QPOS)
    print("[RUN] HOLE_TARGET_POS:", HOLE_TARGET_POS)
    print("[RUN] PREAPPROACH_TARGET:", preapproach)
    print("[RUN] debug_forward:", debug_forward)

    def mission(viewer=None):
        print("\n[PHASE] set custom far home")
        robot.set_qpos(CUSTOM_HOME_QPOS)
        if debug_forward:
            mujoco.mj_forward(model, data)
        else:
            robot.step_sim()
        if viewer is not None:
            viewer.sync()

        time.sleep(0.5)

        ok1 = step_to_target(robot, preapproach, "PREAPPROACH", debug_forward=debug_forward, viewer=viewer)
        if not ok1:
            return False

        time.sleep(0.5)

        ok2 = step_to_target(robot, HOLE_TARGET_POS, "HOLE_APPROACH", debug_forward=debug_forward, viewer=viewer)
        if not ok2:
            return False

        print("\n[RUN] sequence finished successfully")
        return True

    if headless:
        mission(None)
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mission(viewer)
        for _ in range(HOLD_VIEWER_STEPS):
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.01)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--debug-forward", action="store_true")
    args = parser.parse_args()
    main(headless=args.headless, debug_forward=args.debug_forward)
