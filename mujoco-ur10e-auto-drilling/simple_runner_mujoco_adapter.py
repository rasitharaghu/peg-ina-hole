import mujoco
import numpy as np
from simple_runner_config import (
    EE_SITE_NAME,
    PEG_TIP_SITE_NAME,
    FORCE_SENSOR_NAME,
    TORQUE_SENSOR_NAME,
    HOME_KEY,
    HOLE_KEY,
)

ROBOT_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

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

        self.qpos_ids = []
        self.dof_ids = []
        self.joint_ids = []
        for joint_name in ROBOT_JOINT_NAMES:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            self.joint_ids.append(jid)
            self.qpos_ids.append(model.jnt_qposadr[jid])
            self.dof_ids.append(model.jnt_dofadr[jid])

        self.qpos_ids = np.array(self.qpos_ids, dtype=int)
        self.dof_ids = np.array(self.dof_ids, dtype=int)
        self.joint_ids = np.array(self.joint_ids, dtype=int)

        self.actuator_ids = []
        for jid in self.joint_ids:
            found = None
            for a in range(model.nu):
                if model.actuator_trnid[a, 0] == jid:
                    found = a
                    break
            self.actuator_ids.append(found)

        print("[ADAPTER] joint ids:", self.joint_ids)
        print("[ADAPTER] qpos ids:", self.qpos_ids)
        print("[ADAPTER] dof ids:", self.dof_ids)
        print("[ADAPTER] actuator ids:", self.actuator_ids)

    def _sync_ctrl_to_current_q(self):
        q = self.get_qpos()
        for i, aid in enumerate(self.actuator_ids):
            if aid is not None:
                self.data.ctrl[aid] = q[i]

    def reset_to_home_key(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
        self.data.qvel[:] = 0.0
        self._sync_ctrl_to_current_q()
        mujoco.mj_forward(self.model, self.data)

    def set_qpos(self, qpos):
        self.data.qpos[self.qpos_ids] = qpos
        self.data.qvel[self.dof_ids] = 0.0
        self._sync_ctrl_to_current_q()
        mujoco.mj_forward(self.model, self.data)

    def get_qpos(self):
        return self.data.qpos[self.qpos_ids].copy()

    def get_ee_pos(self):
        return self.data.site_xpos[self.ee_site_id].copy()

    def jacobian_pos(self, site_id=None):
        site_id = self.ee_site_id if site_id is None else site_id
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
        return jacp[:, self.dof_ids]

    def dls_pos(self, pos_err, damping=1.0):
        J = self.jacobian_pos()
        eye = np.eye(J.shape[0])
        dq = J.T @ np.linalg.solve(J @ J.T + damping * eye, pos_err)
        return dq

    def integrate_dq(self, dq, gain):
        q = self.get_qpos()
        q_cmd = q + gain * dq
        self.data.qpos[self.qpos_ids] = q_cmd
        self.data.qvel[self.dof_ids] = 0.0
        for i, aid in enumerate(self.actuator_ids):
            if aid is not None:
                self.data.ctrl[aid] = q_cmd[i]
        mujoco.mj_forward(self.model, self.data)

    def step_sim(self):
        mujoco.mj_step(self.model, self.data)
