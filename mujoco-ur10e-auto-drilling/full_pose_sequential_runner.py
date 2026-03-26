import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer

from full_pose_runner_config import (
    XML_PATH,
    CUSTOM_HOME_QPOS,
    HOLE_TARGET_POS,
    PREAPPROACH_OFFSET,
    DLS_DAMPING_6D,
    POSE_GAIN,
    POS_TOL,
    ORI_TOL,
    ORI_GAIN,
    MAX_STEPS_PER_PHASE,
    HOLD_VIEWER_STEPS,
)
from full_pose_runner_mujoco_adapter import MujocoRobot

def orientation_error_vec(r_des, r_cur):
    r_err = r_des @ r_cur.T
    return 0.5 * np.array([
        r_err[2, 1] - r_err[1, 2],
        r_err[0, 2] - r_err[2, 0],
        r_err[1, 0] - r_err[0, 1],
    ])

def step_to_pose(robot, target_pos, target_rot, label, debug_forward=False, viewer=None, sleep_s=0.0):
    print("\n[PHASE] " + label)
    print("[PHASE] target_pos:", np.round(target_pos, 4))

    for i in range(MAX_STEPS_PER_PHASE):
        pos = robot.get_ee_pos()
        rot = robot.get_ee_rot()

        pos_err = target_pos - pos
        ori_err = orientation_error_vec(target_rot, rot)

        pos_norm = float(np.linalg.norm(pos_err))
        ori_norm = float(np.linalg.norm(ori_err))

        if i % 25 == 0:
            print(f"[{label}] step={i} pos={np.round(pos,4)} pos_err={pos_norm:.6f} ori_err={ori_norm:.6f}")

        if pos_norm < POS_TOL and ori_norm < ORI_TOL:
            print(f"[{label}] reached target with pos_err={pos_norm:.6f}, ori_err={ori_norm:.6f}")
            return True

        task = np.concatenate([pos_err, ORI_GAIN * ori_err])
        dq = robot.dls_pose(task, damping=DLS_DAMPING_6D)
        robot.integrate_dq(dq, POSE_GAIN)

        if debug_forward:
            mujoco.mj_forward(robot.model, robot.data)
        else:
            robot.step_sim()

        if viewer is not None:
            viewer.sync()
            if sleep_s > 0:
                time.sleep(sleep_s)

    print(f"[{label}] failed to converge within {MAX_STEPS_PER_PHASE} steps")
    return False

def main(headless=False, debug_forward=False, sleep_s=0.0):
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    robot = MujocoRobot(model, data)

    hole_key_pos, hole_key_rot = robot.get_hole_key_pose()
    preapproach_pos = HOLE_TARGET_POS + PREAPPROACH_OFFSET
    preapproach_rot = hole_key_rot.copy()

    print("[RUN] XML_PATH:", XML_PATH)
    print("[RUN] CUSTOM_HOME_QPOS:", CUSTOM_HOME_QPOS)
    print("[RUN] HOLE_TARGET_POS:", HOLE_TARGET_POS)
    print("[RUN] HOLE_KEY_POS:", np.round(hole_key_pos, 4))
    print("[RUN] PREAPPROACH_POS:", np.round(preapproach_pos, 4))
    print("[RUN] debug_forward:", debug_forward)
    print("[RUN] sleep:", sleep_s)

    def mission(viewer=None):
        print("\n[PHASE] set custom far home")
        robot.set_qpos(CUSTOM_HOME_QPOS)
        if debug_forward:
            mujoco.mj_forward(model, data)
        else:
            robot.step_sim()
        if viewer is not None:
            viewer.sync()
            if sleep_s > 0:
                time.sleep(0.5)

        ok1 = step_to_pose(robot, preapproach_pos, preapproach_rot, "PREAPPROACH_POSE",
                           debug_forward=debug_forward, viewer=viewer, sleep_s=sleep_s)
        if not ok1:
            return False

        if viewer is not None and sleep_s > 0:
            time.sleep(0.5)

        ok2 = step_to_pose(robot, HOLE_TARGET_POS, hole_key_rot, "HOLE_POSE",
                           debug_forward=debug_forward, viewer=viewer, sleep_s=sleep_s)
        if not ok2:
            return False

        print("\n[RUN] full-pose sequence finished successfully")
        return True

    if headless:
        mission(None)
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mission(viewer)
        for _ in range(HOLD_VIEWER_STEPS):
            mujoco.mj_forward(model, data)
            viewer.sync()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                time.sleep(0.01)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--debug-forward", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0, help="Slow down visualization, e.g. 0.03")
    args = parser.parse_args()
    main(headless=args.headless, debug_forward=args.debug_forward, sleep_s=args.sleep)
