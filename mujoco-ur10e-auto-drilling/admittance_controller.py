
import numpy as np
import mujoco
import time

class AdmittanceController:
    def __init__(self, model, data, site_id, tip_id):
        self.model = model
        self.data = data
        self.site_id = site_id
        self.tip_id = tip_id

    def run(self, viewer):
        axis = np.array([-1.0,0,0])
        start = self.data.site_xpos[self.tip_id].copy()

        for i in range(3000):
            tip = self.data.site_xpos[self.tip_id]
            target = start + 0.02 * axis

            err = target - tip

            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)

            dq = jacp[:,:6].T @ np.linalg.inv(jacp[:,:6]@jacp[:,:6].T + 0.01*np.eye(3)) @ err

            self.data.qpos[:6] += 0.15*dq

            mujoco.mj_forward(self.model, self.data)
            viewer.sync()
            time.sleep(0.01)
