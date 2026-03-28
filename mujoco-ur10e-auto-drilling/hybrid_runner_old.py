import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer

from hybrid_runner_config import (
    XML_PATH,
    CONTROL_SITE_NAME,
    TIP_SITE_NAME,
    HOLE_SITE_NAME,
    HOLE_AXIS_SITE_NAME,
    CUSTOM_HOME_QPOS,
    PREAPPROACH_OFFSET,
    WORLD_UP,
    DLS_DAMPING_6D,
    POSE_GAIN,
    POS_TOL,
    ORI_TOL,
    TIP_TOL,
    ORI_GAIN,
    MAX_INSERTION_TRAVEL,
    INSERTION_STEP_NOMINAL,
    INSERTION_DAMPING,
    INSERTION_GAIN,
    INSERTION_LATERAL_COMPLIANCE,
    INSERTION_AXIAL_SCALE,
    DRILL_EXTRA_DEPTH,
    DRILL_STEP_NOMINAL,
    DRILL_DAMPING,
    DRILL_GAIN,
    MAX_STEPS_PER_PHASE,
    DEFAULT_SLEEP,
)
from hybrid_runner_mujoco_adapter import MujocoRobot
from admittance_insert_controller import AdmittanceInsertionController
from simple_drill_controller import SimpleDrillController

def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Cannot normalize near-zero vector")
    return v / n

def orientation_error_vec(r_des, r_cur):
    r_err = r_des @ r_cur.T
    return 0.5 * np.array([
        r_err[2, 1] - r_err[1, 2],
        r_err[0, 2] - r_err[2, 0],
        r_err[1, 0] - r_err[0, 1],
    ])

def build_rotation_from_local_axis(local_tip_axis, desired_world_axis, world_up):
    z_l = normalize(local_tip_axis)
    ref_l = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(np.dot(ref_l, z_l)) > 0.95:
        ref_l = np.array([0.0, 1.0, 0.0], dtype=float)
    x_l = normalize(np.cross(ref_l, z_l))
    y_l = np.cross(z_l, x_l)
    L = np.column_stack((x_l, y_l, z_l))

    z_w = normalize(desired_world_axis)
    up = normalize(world_up)
    if abs(np.dot(up, z_w)) > 0.95:
        up = np.array([0.0, 1.0, 0.0], dtype=float)
    x_w = normalize(np.cross(up, z_w))
    y_w = np.cross(z_w, x_w)
    W = np.column_stack((x_w, y_w, z_w))

    return W @ L.T

def move_control_site_to_pose(robot, target_control_pos, target_control_rot, target_tip_pos, label, viewer, sleep_s):
    print(f"\n[PHASE] {label}")
    print("[TARGET CONTROL POS]", np.round(target_control_pos, 4))
    print("[TARGET TIP POS]", np.round(target_tip_pos, 4))

    for i in range(MAX_STEPS_PER_PHASE):
        control_pos = robot.get_control_pos()
        control_rot = robot.get_control_rot()
        tip_pos = robot.get_tip_pos()

        pos_err = target_control_pos - control_pos
        ori_err = orientation_error_vec(target_control_rot, control_rot)
        tip_err = target_tip_pos - tip_pos

        if i % 25 == 0:
            print(
                f"[{label}] step={i} "
                f"control_pos_err={np.linalg.norm(pos_err):.6f} "
                f"ori_err={np.linalg.norm(ori_err):.6f} "
                f"tip_err={np.linalg.norm(tip_err):.6f} "
                f"tip_pos={np.round(tip_pos,4)}"
            )

        if (
            np.linalg.norm(pos_err) < POS_TOL
            and np.linalg.norm(ori_err) < ORI_TOL
            and np.linalg.norm(tip_err) < TIP_TOL
        ):
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

def main(sleep_s=DEFAULT_SLEEP):
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    robot = MujocoRobot(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        robot.set_qpos(CUSTOM_HOME_QPOS)
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(1.0)

        hole_tip_target = robot.get_hole_pos()
        hole_axis_pos = robot.get_hole_axis_pos()
        desired_axis_world = normalize(hole_axis_pos - hole_tip_target)

        tip_offset_local = robot.get_tip_offset_local()
        target_control_rot = build_rotation_from_local_axis(
            local_tip_axis=tip_offset_local,
            desired_world_axis=desired_axis_world,
            world_up=np.array(WORLD_UP, dtype=float),
        )

        pre_tip_target = hole_tip_target + PREAPPROACH_OFFSET
        pre_control_target = pre_tip_target - target_control_rot @ tip_offset_local
        hole_control_target = hole_tip_target - target_control_rot @ tip_offset_local

        print("[RUN] XML_PATH:", XML_PATH)
        print("[RUN] control frame:", CONTROL_SITE_NAME)
        print("[RUN] tip frame:", TIP_SITE_NAME)
        print("[RUN] hole site:", HOLE_SITE_NAME)
        print("[RUN] hole axis site:", HOLE_AXIS_SITE_NAME)
        print("[RUN] hole tip target:", np.round(hole_tip_target, 4))
        print("[RUN] hole axis pos:", np.round(hole_axis_pos, 4))
        print("[RUN] desired insertion axis world:", np.round(desired_axis_world, 4))
        print("[RUN] tip_offset_local:", np.round(tip_offset_local, 6))
        print("[RUN] pre tip target:", np.round(pre_tip_target, 4))
        print("[RUN] sleep:", sleep_s)

        ok1 = move_control_site_to_pose(robot, pre_control_target, target_control_rot, pre_tip_target, "PREAPPROACH", viewer, sleep_s)
        if not ok1:
            print("[DONE] Failed in PREAPPROACH.")
            while viewer.is_running():
                mujoco.mj_forward(model, data); viewer.sync(); time.sleep(sleep_s)
            return

        ok2 = move_control_site_to_pose(robot, hole_control_target, target_control_rot, hole_tip_target, "HOLE_POSE", viewer, sleep_s)
        if not ok2:
            print("[DONE] Failed in HOLE_POSE.")
            while viewer.is_running():
                mujoco.mj_forward(model, data); viewer.sync(); time.sleep(sleep_s)
            return

        print("\n[PHASE] ADMITTANCE INSERTION")
        actual_tip_start = robot.get_tip_pos().copy()
        print("[ADMITTANCE] actual start tip:", np.round(actual_tip_start, 4))

        inserter = AdmittanceInsertionController(
            robot=robot,
            start_tip_pos=actual_tip_start,
            axis_world=desired_axis_world,
            max_travel=MAX_INSERTION_TRAVEL,
            nominal_step=INSERTION_STEP_NOMINAL,
            damping=INSERTION_DAMPING,
            gain=INSERTION_GAIN,
            lateral_compliance=INSERTION_LATERAL_COMPLIANCE,
            axial_scale=INSERTION_AXIAL_SCALE,
            control_rot_target=target_control_rot,
            tip_offset_local=tip_offset_local,
        )

        inserted_tip = actual_tip_start.copy()
        for i in range(MAX_STEPS_PER_PHASE):
            done, traveled, lateral_err, inserted_tip = inserter.step()
            if i % 25 == 0:
                print(f"[ADMITTANCE] step={i} traveled={traveled:.6f} lateral_err={lateral_err:.6f} current_tip={np.round(robot.get_tip_pos(),4)} target_tip={np.round(inserted_tip,4)}")
            mujoco.mj_forward(model, data); viewer.sync(); time.sleep(sleep_s)
            if done:
                print("[ADMITTANCE] done")
                break

        print("\n[PHASE] SIMPLE DRILL FEED")
        actual_inserted_tip = robot.get_tip_pos().copy()
        print("[DRILL] actual inserted tip start:", np.round(actual_inserted_tip, 4))

        driller = SimpleDrillController(
            robot=robot,
            start_tip_pos=actual_inserted_tip,
            axis_world=desired_axis_world,
            extra_depth=DRILL_EXTRA_DEPTH,
            nominal_step=DRILL_STEP_NOMINAL,
            damping=DRILL_DAMPING,
            gain=DRILL_GAIN,
            control_rot_target=target_control_rot,
            tip_offset_local=tip_offset_local,
        )

        for i in range(MAX_STEPS_PER_PHASE):
            done, traveled, drill_tip_target = driller.step()
            if i % 25 == 0:
                print(f"[DRILL] step={i} depth={traveled:.6f} current_tip={np.round(robot.get_tip_pos(),4)} target_tip={np.round(drill_tip_target,4)}")
            mujoco.mj_forward(model, data); viewer.sync(); time.sleep(sleep_s)
            if done:
                print("[DRILL] done")
                break

        print("[DONE] Task complete. Close viewer window to exit.")
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(sleep_s)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP, help="Slow visualization. Example: 0.08 or 0.15")
    args = parser.parse_args()
    main(sleep_s=args.sleep)
