import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer

from hybrid_runner_config import (
    XML_PATH,
    CUSTOM_HOME_QPOS,
    HOLE_TARGET_POS,
    PREAPPROACH_OFFSET,
    INSERTION_AXIS_WORLD,
    MAX_INSERTION_TRAVEL,
    INSERTION_GAIN,
    INSERTION_DAMPING,
    INSERTION_POS_TOL,
    DLS_DAMPING_6D,
    POSE_GAIN,
    POS_TOL,
    ORI_TOL,
    ORI_GAIN,
    MAX_STEPS_PER_PHASE,
)
from hybrid_runner_mujoco_adapter import MujocoRobot
from hybrid_insert_limited import LimitedInsertionController

def orientation_error_vec(r_des, r_cur):
    r_err = r_des @ r_cur.T
    return 0.5 * np.array([
        r_err[2, 1] - r_err[1, 2],
        r_err[0, 2] - r_err[2, 0],
        r_err[1, 0] - r_err[0, 1],
    ])

def move_to_pose(robot, target_pos, target_rot, label, viewer, sleep_s):
    print(f"\n[PHASE] {label}")
    print("[TARGET POS]", np.round(target_pos, 4))

    for i in range(MAX_STEPS_PER_PHASE):
        pos = robot.get_ee_pos()
        rot = robot.get_ee_rot()

        pos_err = target_pos - pos
        ori_err = orientation_error_vec(target_rot, rot)

        if i % 50 == 0:
            print(f"[{label}] step={i} pos_err={np.linalg.norm(pos_err):.6f} ori_err={np.linalg.norm(ori_err):.6f}")

        if np.linalg.norm(pos_err) < POS_TOL and np.linalg.norm(ori_err) < ORI_TOL:
            print(f"[{label}] reached")
            return True

        task = np.concatenate([pos_err, ORI_GAIN * ori_err])
        dq = robot.dls_pose(task, damping=DLS_DAMPING_6D)
        robot.integrate_dq(dq, POSE_GAIN)

        mujoco.mj_forward(robot.model, robot.data)
        viewer.sync()
        time.sleep(sleep_s)

    print(f"[{label}] failed")
    return False

def main(sleep_s=0.03):
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    robot = MujocoRobot(model, data)

    hole_pos, hole_rot = robot.get_hole_pose()
    pre_pos = HOLE_TARGET_POS + PREAPPROACH_OFFSET

    print("[RUN] pre target:", pre_pos)
    print("[RUN] hole target:", HOLE_TARGET_POS)
    print("[RUN] insertion axis:", INSERTION_AXIS_WORLD)
    print("[RUN] max insertion travel:", MAX_INSERTION_TRAVEL)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        robot.set_qpos(CUSTOM_HOME_QPOS)
        viewer.sync()
        time.sleep(1.0)

        ok1 = move_to_pose(robot, pre_pos, hole_rot, "PREAPPROACH", viewer, sleep_s)
        if not ok1:
            time.sleep(5)
            return

        ok2 = move_to_pose(robot, HOLE_TARGET_POS, hole_rot, "HOLE_POSE", viewer, sleep_s)
        if not ok2:
            time.sleep(5)
            return

        print("\n[PHASE] LIMITED INSERTION")
        inserter = LimitedInsertionController(
            robot=robot,
            axis_world=np.array(INSERTION_AXIS_WORLD, dtype=float),
            max_travel=MAX_INSERTION_TRAVEL,
            damping=INSERTION_DAMPING,
            gain=INSERTION_GAIN,
            pos_tol=INSERTION_POS_TOL,
        )

        for i in range(MAX_STEPS_PER_PHASE):
            done = inserter.step()
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(sleep_s)
            if done:
                print("[LIMITED INSERTION] done")
                break

        time.sleep(5.0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleep", type=float, default=0.03, help="Slow visualization, default 0.03")
    args = parser.parse_args()
    main(sleep_s=args.sleep)
