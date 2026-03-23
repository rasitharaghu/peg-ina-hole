import mujoco
import numpy as np

from config_bt import (
    EE_SITE_NAME,
    PEG_TIP_SITE_NAME,
    FORCE_SENSOR_NAME,
    TORQUE_SENSOR_NAME,
    HOME_KEY,
    HOLE_KEY,
)

class MujocoRobot:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE_NAME)
        self.peg_tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, PEG_TIP_SITE_NAME)

        self.force_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, FORCE_SENSOR_NAME)
        self.torque_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, TORQUE_SENSOR_NAME)

        self.home_key_id = model.key(HOME_KEY).id
        self.hole_key_id = model.key(HOLE_KEY).id

        self.force_bias = np.zeros(3)
        self.torque_bias = np.zeros(3)

    def reset_to_home(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
        mujoco.mj_forward(self.model, self.data)

    def reset_to_hole(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.hole_key_id)
        mujoco.mj_forward(self.model, self.data)

    def get_qpos(self):
        return self.data.qpos[:self.model.nq].copy()

    def set_qpos(self, qpos):
        self.data.qpos[:len(qpos)] = qpos
        mujoco.mj_forward(self.model, self.data)

    def get_ee_pos(self):
        return self.data.site_xpos[self.ee_site_id].copy()

    def get_peg_tip_pos(self):
        return self.data.site_xpos[self.peg_tip_site_id].copy()

    def get_ee_rot(self):
        return self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()

    def get_home_ee_rot(self):
        q_backup = self.get_qpos()
        self.reset_to_home()
        rot = self.get_ee_rot()
        self.set_qpos(q_backup)
        return rot

    def jacobian_site(self, site_id=None, full=False):
        site_id = self.ee_site_id if site_id is None else site_id
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
        return np.vstack([jacp, jacr]) if full else jacp

    def compute_dls_step(self, task_vec, full=False, damping=0.01):
        J = self.jacobian_site(full=full)
        ident = np.eye(J.shape[0])
        dq = J.T @ np.linalg.solve(J @ J.T + damping * ident, task_vec)
        return dq

    def integrate_dq(self, dq, gain):
        mujoco.mj_integratePos(self.model, self.data.qpos, dq, gain)
        mujoco.mj_forward(self.model, self.data)

    def step_sim(self):
        mujoco.mj_step(self.model, self.data)

    def sensor_vec(self, sensor_id):
        adr = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]
        return self.data.sensordata[adr:adr+dim].copy()

    def get_force_raw(self):
        return self.sensor_vec(self.force_sensor_id)

    def get_torque_raw(self):
        return self.sensor_vec(self.torque_sensor_id)

    def zero_ft_bias(self):
        self.force_bias = self.get_force_raw()
        self.torque_bias = self.get_torque_raw()

    def get_force(self):
        return self.get_force_raw() - self.force_bias

    def get_torque(self):
        return self.get_torque_raw() - self.torque_bias

    def force_limit_exceeded(self, force_limit, torque_limit):
        f = np.linalg.norm(self.get_force())
        t = np.linalg.norm(self.get_torque())
        return f > force_limit or t > torque_limit
