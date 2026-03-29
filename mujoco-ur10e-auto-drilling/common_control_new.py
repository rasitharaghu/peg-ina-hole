import time
import numpy as np
import mujoco

XML_PATH = "scene.xml"
CONTROL_SITE = "attachment_site"
TIP_SITE = "peg_tip"
HOLE_TARGET_POS = np.array([-0.75, 0.024, 0.908], dtype=float)

MOVE_DURATION = 5.0
MOVE_DAMPING = 0.01
MOVE_GAIN = 0.10
MOVE_POS_TOL = 0.002

INSERTION_DEPTH = -0.018
INSERTION_DURATION = 4.0
INSERTION_K_POS = np.array([220.0, 8.0, 8.0], dtype=float)
INSERTION_DAMPING = 1.0
INSERTION_INTEGRATION_DT = 0.03
INSERTION_MAX_TRAVEL = 0.018

DRILL_DEPTH = -0.010
DRILL_DURATION = 2.5
DRILL_K_POS = np.array([260.0, 8.0, 8.0], dtype=float)
DRILL_DAMPING = 1.0
DRILL_INTEGRATION_DT = 0.02
DRILL_MAX_TRAVEL = 0.010

DEFAULT_SLEEP = 0.03

def get_ids(model):
    site_id = model.site(CONTROL_SITE).id
    tip_id = model.site(TIP_SITE).id
    return site_id, tip_id

def step_view(model, data, viewer=None, sleep_s=DEFAULT_SLEEP):
    mujoco.mj_forward(model, data)
    if viewer is not None:
        viewer.sync()
    if sleep_s > 0:
        time.sleep(sleep_s)

def move_to_hole_phase(model, data, viewer=None, sleep_s=DEFAULT_SLEEP):
    key_id = model.key('home').id
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    site_id, tip_id = get_ids(model)
    initial_start_pos = data.site_xpos[site_id].copy()
    home_rot = data.site_xmat[site_id].reshape(3, 3).copy()

    print("[MOVE TO HOLE NEW]")
    print("[MOVE] start attachment_site:", np.round(initial_start_pos, 4))
    print("[MOVE] target attachment_site:", np.round(HOLE_TARGET_POS, 4))

    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        t = min(elapsed / MOVE_DURATION, 1.0)
        alpha = t * t * (3 - 2 * t)
        current_target_pos = initial_start_pos + alpha * (HOLE_TARGET_POS - initial_start_pos)

        current_pos = data.site_xpos[site_id].copy()
        current_rot = data.site_xmat[site_id].reshape(3, 3).copy()
        pos_error = current_target_pos - current_pos

        rot_error_mat = home_rot @ current_rot.T
        quat_ref = np.zeros(4)
        quat_curr = np.zeros(4)
        rot_res = np.zeros(3)
        mujoco.mju_mat2Quat(quat_ref, np.eye(3).flatten())
        mujoco.mju_mat2Quat(quat_curr, rot_error_mat.flatten())
        error_6d = np.concatenate([pos_error, rot_res])

        jac = np.zeros((6, model.nv))
        mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
        jj_t = jac @ jac.T
        dq = jac.T @ np.linalg.solve(jj_t + MOVE_DAMPING * np.eye(6), error_6d)

        mujoco.mj_integratePos(model, data.qpos, dq, MOVE_GAIN)
        step_view(model, data, viewer, sleep_s)

        if t >= 1.0 and np.linalg.norm(HOLE_TARGET_POS - data.site_xpos[site_id]) < MOVE_POS_TOL:
            break

    print("[MOVE] final attachment_site:", np.round(data.site_xpos[site_id], 4))
    print("[MOVE] final peg_tip:", np.round(data.site_xpos[tip_id], 4))

def insertion_or_drill_phase(model, data, viewer=None, sleep_s=DEFAULT_SLEEP,
                             depth=-0.01, duration=3.0, k_pos=None, damping=1.0,
                             integration_dt=0.02, max_travel=0.01, label="INSERT"):
    if k_pos is None:
        k_pos = np.array([200.0, 5.0, 5.0], dtype=float)

    site_id, tip_id = get_ids(model)
    start_site = data.site_xpos[site_id].copy()
    start_tip = data.site_xpos[tip_id].copy()
    start_time = time.time()

    print(f"[{label}] start attachment_site:", np.round(start_site, 4))
    print(f"[{label}] start tip:", np.round(start_tip, 4))

    while True:
        elapsed = time.time() - start_time
        t = min(elapsed / duration, 1.0)

        target_pos = start_site.copy()
        target_pos[0] += t * depth

        current_pos = data.site_xpos[site_id].copy()
        pos_error = target_pos - current_pos
        f_task = k_pos * pos_error

        jac = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jac, None, site_id)
        jj_t = jac @ jac.T
        dq = jac.T @ np.linalg.solve(jj_t + damping * np.eye(3), f_task)

        mujoco.mj_integratePos(model, data.qpos, dq, integration_dt)
        step_view(model, data, viewer, sleep_s)

        traveled = abs(data.site_xpos[site_id][0] - start_site[0])
        if traveled >= max_travel or t >= 1.0:
            break

    print(f"[{label}] end attachment_site:", np.round(data.site_xpos[site_id], 4))
    print(f"[{label}] end tip:", np.round(data.site_xpos[tip_id], 4))
