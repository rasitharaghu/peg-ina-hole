import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path('scene.xml')
data = mujoco.MjData(model)

# Set to a pose where gravity is very obvious (arm extended)
mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Update physics state
        mujoco.mj_forward(model, data)
        print(data.qfrc_gravcomp[:model.nu])
        
        # Apply ONLY gravity compensation
        # data.ctrl[:model.nu] = data.qfrc_gravcomp[:model.nu]
        
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.002)