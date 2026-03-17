import mujoco
import mujoco.viewer
import time

def main():
    # Load the model from the XML file
    # Ensure scene.xml and the ur10e folder are in the same directory
    try:
        model = mujoco.MjModel.from_xml_path('scene.xml')
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Error loading XML: {e}")
        return

    # Look up the keyframe by name
    key_name = "home"
    key_id = model.key(key_name).id
    
    # Copy the qpos from the keyframe into the simulation state
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    # Forward kinematics to update body positions based on new joint angles
    mujoco.mj_forward(model, data)

    hole_cx, hole_cy = 0.524, 0.908
    # Launch the passive viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("MuJoCo Viewer started. Close the window to exit.")
        
        # Keep the simulation running
        while viewer.is_running():
            step_start = time.time()

            # Advance the simulation state
            mujoco.mj_step(model, data)

            # Sync the viewer with the updated data at ~60fps
            # This handles the internal timing so the visualization looks smooth
            viewer.sync()

            # Rudimentary time synchronization to real-time
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()