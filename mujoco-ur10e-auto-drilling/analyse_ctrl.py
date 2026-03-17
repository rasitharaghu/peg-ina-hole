import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter

# --- Configuration ---
xml_path = 'scene.xml'
site_name = 'attachment_site' 
insertion_depth = -0.03  # 30mm
duration = 10.0
dt = 0.002

# Admittance Gains (Virtual Stiffness)
# X is stiff to push, Y/Z are softer to allow for misalignment correction
K_pos = np.array([200.0, 5.0, 5.0]) 
integration_dt = 0.05 

def main():
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 1. Initialize to 'hole' entrance keyframe
    mujoco.mj_resetDataKeyframe(model, data, model.key('hole').id)
    
    # --- ADD RANDOM NOISE TO JOINTS ---
    # 0.001 rad noise roughly translates to ~1-2mm at the end effector
    noise = [0.005,0.0]#np.random.uniform(-0.002, 0.002, 2)
    data.qpos[0:2] += noise
    
    # Sync kinematics to find the NEW noisy start position
    mujoco.mj_forward(model, data)
    
    site_id = model.site(site_name).id
    # Capture the "Ideal" trajectory start (where we WANT to be)
    # We define this manually based on your hole location
    ideal_start_pos = data.site_xpos[site_id].copy()#np.array([-0.75, 0.024, 0.908]) 

    # Data collection
    logs = {"time": [], "pos_act": [], "pos_tar": [], "force": []}

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        
        while viewer.is_running():
            step_start = time.time()
            elapsed = time.time() - start_time
            if elapsed > duration: break
            
            # --- PHASE 1: TARGET TRAJECTORY ---
            t_alpha = min(elapsed / duration, 1.0)
            target_pos = ideal_start_pos.copy()
            target_pos[0] += t_alpha * insertion_depth 

            # --- PHASE 2: CALCULATE VIRTUAL FORCE ---
            # This is the 'Admittance Law' F = K * (x_d - x)
            current_pos = data.site_xpos[site_id].copy()
            pos_error = target_pos - current_pos
            f_virtual = K_pos * pos_error

            # --- PHASE 3: KINEMATIC MAPPING ---
            jac = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jac, None, site_id)
            jj_t = jac @ jac.T
            # Solve for dq: the joint change required to exert that force
            dq = jac.T @ np.linalg.solve(jj_t + 1.0 * np.eye(3), f_virtual)

            # --- PHASE 4: KINEMATIC INTEGRATION (Teleportation) ---
            # We update qpos directly, bypassing physics/actuators
            mujoco.mj_integratePos(model, data.qpos, dq, integration_dt)

            # --- PHASE 5: REFRESH ---
            mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)
            
            # Log Data
            logs["time"].append(elapsed)
            logs["pos_act"].append(current_pos)
            logs["pos_tar"].append(target_pos.copy())
            logs["force"].append(f_virtual.copy())

            viewer.sync()
            
            # Slow motion for visibility
            time.sleep(max(0, 0.002 - (time.time() - step_start)))

    # --- PLOTTING ---
    t_plot = np.array(logs["time"][5:])
    p_act = np.array(logs["pos_act"][5:])
    p_tar = np.array(logs["pos_tar"][5:])
    f_log = np.array(logs["force"][5:])

    fig, axes = plt.subplots(4, 1, figsize=(10, 14))
    axis_labels = ['X', 'Y', 'Z']
    colors = ['tab:red', 'tab:green', 'tab:blue']

    for i in range(3):
        axes[i].plot(t_plot, p_tar[:, i], color='black', linestyle='--', alpha=0.6, label=f'Target {axis_labels[i]}')
        axes[i].plot(t_plot, p_act[:, i], color=colors[i], label=f'Actual {axis_labels[i]}')
        axes[i].set_ylabel(f"Pos {axis_labels[i]} [m]")
        axes[i].set_title(f"{axis_labels[i]}-Axis Trajectory Tracking")
        axes[i].legend(loc='upper right')
        axes[i].grid(True)

    # Virtual Force Plot (Admittance Law)
    # axes[3].plot(t_plot, f_log[:, 0], label='Fx (Pushing Force)', color='red')
    axes[3].plot(t_plot, f_log[:, 1], label='Fy (Alignment Force)', color='green')
    axes[3].plot(t_plot, f_log[:, 2], label='Fz (Alignment Force)', color='blue')
    axes[3].set_title("Virtual Forces (Based on Admittance Law)")
    axes[3].set_ylabel("Force [N]")
    axes[3].set_xlabel("Time [s]")
    axes[3].legend()
    axes[3].grid(True)

    # Apply same clean formatting to force plot
    force_formatter = ScalarFormatter(useOffset=False)
    force_formatter.set_scientific(False)
    axes[3].yaxis.set_major_formatter(force_formatter)
    axes[3].yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    
    axes[3].legend()
    axes[3].grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("kinematic_admittance_analysis.png")
    plt.show()

if __name__ == "__main__":
    main()