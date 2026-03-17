import mujoco
import mujoco.viewer
import numpy as np
import time

# --- Configuration ---
xml_path = 'scene.xml'
site_name = 'attachment_site' 
insertion_depth = -0.03  # 30mm
duration = 10.0

# --- Admittance/Kinematic Gains ---
# High K_pos means the robot follows the target strictly.
# Low K_pos means the robot "gives way" more easily.
K_pos = np.array([200.0, 5, 5])  # Stiff X, Soft Y/Z
# How much the "spring" moves the joints per frame
integration_dt = 0.05 

def main():
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 1. Initialize to 'hole' entrance
    mujoco.mj_resetDataKeyframe(model, data, model.key('hole').id)
    mujoco.mj_forward(model, data)
    
    site_id = model.site(site_name).id
    start_pos = data.site_xpos[site_id].copy()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        time.sleep(30)
        while viewer.is_running():
            step_start = time.time()
            elapsed = time.time() - start_time
            
            # --- PHASE 1: VIRTUAL TARGET TRAJECTORY ---
            t = min(elapsed / duration, 1.0)
            target_pos = start_pos.copy()
            target_pos[0] += t * insertion_depth 

            # --- PHASE 2: CALCULATE ERROR ---
            current_pos = data.site_xpos[site_id]
            pos_error = target_pos - current_pos
            
            # Virtual Force (The 'Spring' pull)
            f_task = K_pos * pos_error

            # --- PHASE 3: KINEMATIC MAPPING ---
            jac = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jac, None, site_id)
            
            # Map the task-space spring pull to a joint-space nudge (dq)
            # We use DLS here to keep the joint movement stable
            jj_t = jac @ jac.T
            dq = jac.T @ np.linalg.solve(jj_t + 1 * np.eye(3), f_task)
            print(dq)

            # --- PHASE 4: UPDATE QPOS DIRECTLY ---
            # This is the "Teleportation" step. 
            # It moves qpos based on the dq we just calculated.
            mujoco.mj_integratePos(model, data.qpos, dq, integration_dt)
            # data.ctrl[:model.nu] += dq[:model.nu] * integration_dt

            # --- PHASE 5: REFRESH KINEMATICS ---
            # Since we changed qpos manually, we MUST call mj_forward
            # to update the peg_tip position for the next loop.
            mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)
            
            viewer.sync()
            
            slow_motion_factor = 1000 
            time.sleep(0.002 * slow_motion_factor)

if __name__ == "__main__":
    main()