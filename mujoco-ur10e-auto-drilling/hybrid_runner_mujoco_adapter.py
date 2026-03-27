import mujoco
import numpy as np
from hybrid_runner_config import EE_SITE_NAME, HOLE_KEY

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
        self.hole_key_id = model.key(HOLE_KEY).id

        self.qpos_ids = []
        self.dof_ids = []
        self.joint_ids = []

        for j in ROBOT_JOINT_NAMES:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
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

    def _sync_ctrl(self):
        q = self.get_qpos()
        for i, aid in enumerate(self.actuator_ids):
            if aid is not None:
                self.data.ctrl[aid] = q[i]

    def set_qpos(self, q):
        self.data.qpos[self.qpos_ids] = q
        self.data.qvel[self.dof_ids] = 0.0
        self._sync_ctrl()
        mujoco.mj_forward(self.model, self.data)

    def get_qpos(self):
        return self.data.qpos[self.qpos_ids].copy()

    def get_ee_pos(self):
        return self.data.site_xpos[self.ee_site_id].copy()

    def get_ee_rot(self):
        return self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()

    def get_hole_pose(self):
        q_backup = self.data.qpos.copy()
        qvel_backup = self.data.qvel.copy()
        ctrl_backup = self.data.ctrl.copy()

        mujoco.mj_resetDataKeyframe(self.model, self.data, self.hole_key_id)
        ee_pos = self.get_ee_pos()
        ee_rot = self.get_ee_rot()

        self.data.qpos[:] = q_backup
        self.data.qvel[:] = qvel_backup
        self.data.ctrl[:] = ctrl_backup
        mujoco.mj_forward(self.model, self.data)
        return ee_pos.copy(), ee_rot.copy()

    def jacobian_pose(self):
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        return np.vstack([jacp[:, self.dof_ids], jacr[:, self.dof_ids]])

    def jacobian_pos(self):
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        return jacp[:, self.dof_ids]

    def dls_pose(self, task_vec, damping=0.01):
        J = self.jacobian_pose()
        eye = np.eye(J.shape[0])
        return J.T @ np.linalg.solve(J @ J.T + damping * eye, task_vec)

    def dls_pos(self, pos_err, damping=0.2):
        J = self.jacobian_pos()
        eye = np.eye(J.shape[0])
        return J.T @ np.linalg.solve(J @ J.T + damping * eye, pos_err)

    def integrate_dq(self, dq, gain):
        q = self.get_qpos()
        self.set_qpos(q + gain * dq)

    def step_sim(self):
        mujoco.mj_step(self.model, self.data)
