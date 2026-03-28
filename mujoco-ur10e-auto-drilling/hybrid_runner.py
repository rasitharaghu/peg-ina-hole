
import numpy as np
import mujoco
import mujoco.viewer
import time

from admittance_controller import AdmittanceController
from drill_controller import DrillController

XML_PATH = "scene.xml"

HOLE_POS = np.array([-0.75, 0.024, 0.908])

SITE = "attachment_site"
TIP = "peg_tip"

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE)
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TIP)

    with mujoco.viewer.launch_passive(model, data) as viewer:

        mujoco.mj_resetDataKeyframe(model, data, 0)  # home
        mujoco.mj_forward(model, data)

        home_rot = data.site_xmat[site_id].reshape(3,3)

        print("[MOVE TO HOLE]")
        for i in range(4000):
            pos = data.site_xpos[site_id]
            err = HOLE_POS - pos

            if np.linalg.norm(err) < 0.002:
                break

            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

            J = jacp[:, :6]
            dq = J.T @ np.linalg.inv(J @ J.T + 0.01*np.eye(3)) @ err

            data.qpos[:6] += 0.1 * dq

            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.01)

        print("[ADMITTANCE INSERT]")
        insert = AdmittanceController(model, data, site_id, tip_id)
        insert.run(viewer)

        print("[DRILL]")
        drill = DrillController(model, data, site_id, tip_id)
        drill.run(viewer)

        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()
