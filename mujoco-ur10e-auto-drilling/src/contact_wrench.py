from __future__ import annotations

import numpy as np
import mujoco


class ContactWrenchEstimator:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, tcp_site_name: str = "tcp"):
        self.model = model
        self.data = data
        self.tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tcp_site_name)

    def estimate_world_wrench(self) -> np.ndarray:
        tcp_pos = self.data.site_xpos[self.tcp_site_id].copy()
        total_force = np.zeros(3, dtype=float)
        total_torque = np.zeros(3, dtype=float)

        c_array = np.zeros(6, dtype=float)
        rot = np.zeros((3, 3), dtype=float)

        for i in range(self.data.ncon):
            mujoco.mj_contactForce(self.model, self.data, i, c_array)
            con = self.data.contact[i]
            rot[:, :] = np.array(con.frame).reshape(3, 3)
            force_local = c_array[:3].copy()
            torque_local = c_array[3:].copy()
            force_world = rot @ force_local
            torque_world = rot @ torque_local
            contact_point = con.pos.copy()
            arm = contact_point - tcp_pos
            total_force += force_world
            total_torque += torque_world + np.cross(arm, force_world)

        return np.hstack([total_force, total_torque])
