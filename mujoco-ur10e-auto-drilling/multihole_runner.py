import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer
from hole_catalog import HOLES

XML_PATH = "scene_multihole_frames.xml"
CONTROL_SITE = "attachment_site"
TIP_SITE = "peg_tip"

MOVE_DURATION = 4.0
MOVE_DAMPING = 0.01
MOVE_GAIN = 0.10
MOVE_POS_TOL = 0.002
INSERTION_DURATION = 2.5
INSERTION_DAMPING = 1.0
INSERTION_DT = 0.03
INSERTION_MAX_TRAVEL = 0.014
INSERTION_K = np.array([220.0, 8.0, 8.0], dtype=float)
DRILL_DURATION = 1.8
DRILL_DT = 0.02
DRILL_MAX_TRAVEL = 0.008
DRILL_K = np.array([260.0, 8.0, 8.0], dtype=float)
DEFAULT_SLEEP = 0.08

def get_site_vec(data, site_id):
    return data.site_xpos[site_id].copy()

def run_move_to_hole(model, data, viewer, site_id, tip_id, hole_center, sleep_s):
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    mujoco.mj_forward(model, data)
    start_pos = get_site_vec(data, site_id)
    home_rot = data.site_xmat[site_id].reshape(3, 3).copy()

    tip_offset_local = home_rot.T @ (get_site_vec(data, tip_id) - start_pos)
    attachment_target = hole_center - home_rot @ tip_offset_local

    print("[MOVE] desired peg_tip target:", np.round(hole_center, 4))
    print("[MOVE] computed attachment_site target:", np.round(attachment_target, 4))
    print("[MOVE] tip offset local:", np.round(tip_offset_local, 6))

    t0 = time.time()
    while viewer.is_running():
        elapsed = time.time() - t0
        s = min(elapsed / MOVE_DURATION, 1.0)
        alpha = s * s * (3 - 2 * s)
        current_target_pos = start_pos + alpha * (attachment_target - start_pos)

        current_pos = get_site_vec(data, site_id)
        pos_error = current_target_pos - current_pos
        error_6d = np.concatenate([pos_error, np.zeros(3)])

        jac = np.zeros((6, model.nv))
        mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
        dq = jac.T @ np.linalg.solve(jac @ jac.T + MOVE_DAMPING * np.eye(6), error_6d)

        mujoco.mj_integratePos(model, data.qpos, dq, MOVE_GAIN)
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(sleep_s)

        if s >= 1.0 and np.linalg.norm(attachment_target - get_site_vec(data, site_id)) < MOVE_POS_TOL:
            break

def run_axial_phase(model, data, viewer, site_id, axis_sign_depth, duration, damping, dt, max_travel, k_vec, sleep_s, label):
    start_site = get_site_vec(data, site_id).copy()
    start_tip = data.site_xpos[model.site(TIP_SITE).id].copy()
    t0 = time.time()
    while viewer.is_running():
        elapsed = time.time() - t0
        s = min(elapsed / duration, 1.0)
        target_pos = start_site.copy()
        target_pos[0] += s * axis_sign_depth

        current_pos = get_site_vec(data, site_id)
        pos_error = target_pos - current_pos
        f_task = k_vec * pos_error

        jac = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jac, None, site_id)
        dq = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(3), f_task)

        mujoco.mj_integratePos(model, data.qpos, dq, dt)
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(sleep_s)

        traveled = abs(get_site_vec(data, site_id)[0] - start_site[0])
        if traveled >= max_travel or s >= 1.0:
            break

    end_tip = data.site_xpos[model.site(TIP_SITE).id].copy()
    print(f"[{label}] start tip:", np.round(start_tip, 4))
    print(f"[{label}] end tip:", np.round(end_tip, 4))

def main(sleep=DEFAULT_SLEEP, hole_name="hole_1", show_site_frames=True):
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    site_id = model.site(CONTROL_SITE).id
    tip_id = model.site(TIP_SITE).id

    hole_cfg = next((h for h in HOLES if h["name"] == hole_name), None)
    if hole_cfg is None:
        raise ValueError(f"Unknown hole name: {hole_name}")

    center_id = model.site(hole_cfg["center_site"]).id
    axis_id = model.site(hole_cfg["axis_site"]).id

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Show site frames for visibility
        try:
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE if show_site_frames else mujoco.mjtFrame.mjFRAME_NONE
        except Exception:
            pass

        mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(1.0)

        hole_center = get_site_vec(data, center_id)
        hole_axis = get_site_vec(data, axis_id)
        print("[RUN] selected hole:", hole_name)
        print("[RUN] hole center:", np.round(hole_center, 4))
        print("[RUN] hole axis point:", np.round(hole_axis, 4))
        print("[RUN] site frames visible:", show_site_frames)

        run_move_to_hole(model, data, viewer, site_id, tip_id, hole_center, sleep)

        print("[ADMITTANCE INSERT]")
        run_axial_phase(model, data, viewer, site_id, axis_sign_depth=0.014, duration=INSERTION_DURATION,
                        damping=INSERTION_DAMPING, dt=INSERTION_DT, max_travel=INSERTION_MAX_TRAVEL,
                        k_vec=INSERTION_K, sleep_s=sleep, label="INSERT")

        print("[DRILL]")
        run_axial_phase(model, data, viewer, site_id, axis_sign_depth=0.008, duration=DRILL_DURATION,
                        damping=1.0, dt=DRILL_DT, max_travel=DRILL_MAX_TRAVEL,
                        k_vec=DRILL_K, sleep_s=sleep, label="DRILL")

        print("[DONE] close viewer to exit")
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(sleep)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP)
    parser.add_argument("--hole", type=str, default="hole_1",
                        choices=[h["name"] for h in HOLES])
    parser.add_argument("--hide-site-frames", action="store_true")
    args = parser.parse_args()
    main(sleep=args.sleep, hole_name=args.hole, show_site_frames=not args.hide_site_frames)
