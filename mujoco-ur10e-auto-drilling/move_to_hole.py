import mujoco
import mujoco.viewer
import numpy as np
import time

# --- Configuration ---
xml_path = 'scene.xml'
hole_target = np.array([-0.75, 0.024, 0.908]) 
site_name = 'attachment_site' 
duration = 5.0  
damping = 0.01  # Small damping is fine for kinematics

def main():
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 1. Initialize to Home
    key_id = model.key('home').id
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    # 2. Capture Initial State
    site_id = model.site(site_name).id
    initial_start_pos = data.site_xpos[site_id].copy()
    # Capture orientation to keep it locked
    home_rot = data.site_xmat[site_id].reshape(3, 3).copy()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        
        while viewer.is_running():
            step_start = time.time()
            elapsed = time.time() - start_time
            
            # --- PHASE 1: TRAJECTORY ---
            t = min(elapsed / duration, 1.0)
            alpha = t * t * (3 - 2 * t) # Smooth acceleration
            current_target_pos = initial_start_pos + alpha * (hole_target - initial_start_pos)

            # --- PHASE 2: DLS MATH ---
            current_pos = data.site_xpos[site_id]
            current_rot = data.site_xmat[site_id].reshape(3, 3)

            # Translation Error
            pos_error = current_target_pos - current_pos
            
            # Orientation Error (Keep it locked to home)
            rot_error_mat = home_rot @ current_rot.T
            quat_ref, quat_curr = np.zeros(4), np.zeros(4)
            res = np.zeros(3) 
            mujoco.mju_mat2Quat(quat_ref, np.eye(3).flatten())
            mujoco.mju_mat2Quat(quat_curr, rot_error_mat.flatten())
            # mujoco.mju_subQuat(res, quat_ref, quat_curr)
            
            error_6d = np.concatenate([pos_error, res])

            # Jacobian
            jac = np.zeros((6, model.nv))
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
            
            # DLS Solve
            jj_t = jac @ jac.T
            dq = jac.T @ np.linalg.solve(jj_t + damping * np.eye(6), error_6d)

            # --- PHASE 3: DIRECT POSITION UPDATE ---
            # We bypass actuators and move the 'bones' directly
            # 0.1 is the 'integration gain'—adjust for speed
            mujoco.mj_integratePos(model, data.qpos, dq, 0.1)

            # Update the robot state
            mujoco.mj_forward(model, data)
            
            # --- PHASE 4: VIEWING ---
            viewer.sync()

            # Keep it real-time
            time_until_next = model.opt.timestep - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)

if __name__ == "__main__":
    main()