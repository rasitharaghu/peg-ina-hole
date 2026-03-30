import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer
from hole_catalog import HOLES

XML_PATH = "scene_from_original_active_multihole.xml"
CONTROL_SITE = "attachment_site"
TIP_SITE = "peg_tip"

MOVE_DURATION = 2.8
MOVE_DAMPING = 0.01
MOVE_GAIN = 0.10
MOVE_POS_TOL = 0.002

INSERTION_DURATION = 1.1
INSERTION_DAMPING = 1.0
INSERTION_DT = 0.03
INSERTION_MAX_TRAVEL = 0.008
INSERTION_K = np.array([220.0, 8.0, 8.0], dtype=float)

DRILL_DURATION = 0.8
DRILL_DT = 0.02
DRILL_MAX_TRAVEL = 0.004
DRILL_K = np.array([260.0, 8.0, 8.0], dtype=float)

CLEARANCE_X = 0.055
CLEARANCE_Z = 0.035
TRANSITION_DURATION = 1.0
DEFAULT_SLEEP = 0.08

def get_site_vec(data, sid):
    return data.site_xpos[sid].copy()

def set_only_active_frame_visible(model, hole_name):
    for hole in HOLES:
        active = hole["name"] == hole_name
        for geom_name in hole["frame_geoms"]:
            gid = model.geom(geom_name).id
            rgba = model.geom_rgba[gid].copy()
            rgba[3] = 1.0 if active else 0.0
            model.geom_rgba[gid] = rgba

def smooth_move_site(model, data, viewer, control_sid, target_pos, duration, sleep_s):
    start_pos = get_site_vec(data, control_sid)
    t0 = time.time()
    while viewer.is_running():
        s = min((time.time() - t0) / duration, 1.0)
        alpha = s * s * (3 - 2 * s)
        current_target = start_pos + alpha * (target_pos - start_pos)
        current_pos = get_site_vec(data, control_sid)
        pos_error = current_target - current_pos
        error_6d = np.concatenate([pos_error, np.zeros(3)])
        jac = np.zeros((6, model.nv))
        mujoco.mj_jacSite(model, data, jac[:3], jac[3:], control_sid)
        dq = jac.T @ np.linalg.solve(jac @ jac.T + MOVE_DAMPING * np.eye(6), error_6d)
        mujoco.mj_integratePos(model, data.qpos, dq, MOVE_GAIN)
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(sleep_s)
        if s >= 1.0 and np.linalg.norm(target_pos - get_site_vec(data, control_sid)) < MOVE_POS_TOL:
            break

def hole_tip_to_control_target(data, control_sid, tip_sid, target_tip):
    current_control = get_site_vec(data, control_sid)
    current_rot = data.site_xmat[control_sid].reshape(3, 3).copy()
    tip_offset_local = current_rot.T @ (get_site_vec(data, tip_sid) - current_control)
    return target_tip - current_rot @ tip_offset_local

def move_tip_to_hole(model, data, viewer, control_sid, tip_sid, hole_center, sleep_s):
    target_control = hole_tip_to_control_target(data, control_sid, tip_sid, hole_center)
    print("[MOVE] desired peg tip:", np.round(hole_center, 4))
    print("[MOVE] control target:", np.round(target_control, 4))
    smooth_move_site(model, data, viewer, control_sid, target_control, MOVE_DURATION, sleep_s)
    print("[MOVE] final tip:", np.round(get_site_vec(data, tip_sid), 4))

def run_axial_phase(model, data, viewer, control_sid, depth, duration, damping, dt, max_travel, k_vec, sleep_s, label):
    start_site = get_site_vec(data, control_sid).copy()
    start_tip = data.site_xpos[model.site(TIP_SITE).id].copy()
    t0 = time.time()
    while viewer.is_running():
        s = min((time.time() - t0) / duration, 1.0)
        target_pos = start_site.copy()
        target_pos[0] += s * depth
        current_pos = get_site_vec(data, control_sid)
        pos_error = target_pos - current_pos
        f_task = k_vec * pos_error
        jac = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jac, None, control_sid)
        dq = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(3), f_task)
        mujoco.mj_integratePos(model, data.qpos, dq, dt)
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(sleep_s)
        traveled = abs(get_site_vec(data, control_sid)[0] - start_site[0])
        if traveled >= max_travel or s >= 1.0:
            break
    end_tip = data.site_xpos[model.site(TIP_SITE).id].copy()
    print(f"[{label}] start tip:", np.round(start_tip, 4))
    print(f"[{label}] end tip:", np.round(end_tip, 4))

def retract_liftoff_and_transition(model, data, viewer, control_sid, tip_sid, current_hole_center, next_hole_center, sleep_s):
    tip_retract = current_hole_center.copy()
    tip_retract[0] += CLEARANCE_X
    target_control = hole_tip_to_control_target(data, control_sid, tip_sid, tip_retract)
    print("[TRANSITION] retracting off the surface")
    smooth_move_site(model, data, viewer, control_sid, target_control, TRANSITION_DURATION * 0.7, sleep_s)

    tip_clear = next_hole_center.copy()
    tip_clear[0] += CLEARANCE_X
    tip_clear[2] += CLEARANCE_Z
    target_control = hole_tip_to_control_target(data, control_sid, tip_sid, tip_clear)
    print("[TRANSITION] moving in free space above the panel")
    smooth_move_site(model, data, viewer, control_sid, target_control, TRANSITION_DURATION, sleep_s)

    tip_pre = next_hole_center.copy()
    tip_pre[0] += CLEARANCE_X
    target_control = hole_tip_to_control_target(data, control_sid, tip_sid, tip_pre)
    print("[TRANSITION] descending to next-hole pre-approach")
    smooth_move_site(model, data, viewer, control_sid, target_control, TRANSITION_DURATION * 0.7, sleep_s)

def main(sleep=DEFAULT_SLEEP):
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    control_sid = model.site(CONTROL_SITE).id
    tip_sid = model.site(TIP_SITE).id
    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(1.0)

        for i, hole in enumerate(HOLES):
            set_only_active_frame_visible(model, hole["name"])
            mujoco.mj_forward(model, data)
            viewer.sync()

            center_sid = model.site(hole["center_site"]).id
            hole_center = get_site_vec(data, center_sid)

            print("\n==================================================")
            print("[RUN] selected hole:", hole["name"])
            print("[RUN] hole center:", np.round(hole_center, 4))

            move_tip_to_hole(model, data, viewer, control_sid, tip_sid, hole_center, sleep)

            print("[ADMITTANCE INSERT]")
            run_axial_phase(model, data, viewer, control_sid, depth=-0.008, duration=INSERTION_DURATION,
                            damping=INSERTION_DAMPING, dt=INSERTION_DT, max_travel=INSERTION_MAX_TRAVEL,
                            k_vec=INSERTION_K, sleep_s=sleep, label="INSERT")

            print("[DRILL]")
            run_axial_phase(model, data, viewer, control_sid, depth=-0.004, duration=DRILL_DURATION,
                            damping=1.0, dt=DRILL_DT, max_travel=DRILL_MAX_TRAVEL,
                            k_vec=DRILL_K, sleep_s=sleep, label="DRILL")

            if i < len(HOLES) - 1:
                next_center_sid = model.site(HOLES[i + 1]["center_site"]).id
                next_hole_center = get_site_vec(data, next_center_sid)
                retract_liftoff_and_transition(model, data, viewer, control_sid, tip_sid, hole_center, next_hole_center, sleep)

        for hole in HOLES:
            for geom_name in hole["frame_geoms"]:
                gid = model.geom(geom_name).id
                rgba = model.geom_rgba[gid].copy()
                rgba[3] = 0.0
                model.geom_rgba[gid] = rgba
        mujoco.mj_forward(model, data)

        print("\n[DONE] all holes processed. Close viewer to exit.")
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(sleep)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP)
    args = parser.parse_args()
    main(sleep=args.sleep)
