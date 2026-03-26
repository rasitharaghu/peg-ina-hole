import mujoco
import numpy as np
from full_pose_runner_config import *

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
        self.attachment_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ATTACHMENT_SITE_NAME)

        self.hole_key_id = model.key(HOLE_KEY).id

        self.qpos_ids = []
        self.dof_ids = []

        for j in ROBOT_JOINT_NAMES:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            self.qpos_ids.append(model.jnt_qposadr[jid])
            self.dof_ids.append(model.jnt_dofadr[jid])

        self.qpos_ids = np.array(self.qpos_ids)
        self.dof_ids = np.array(self.dof_ids)

    def set_qpos(self, q):
        self.data.qpos[self.qpos_ids] = q
        self.data.qvel[self.dof_ids] = 0
        mujoco.mj_forward(self.model, self.data)

    def get_qpos(self):
        return self.data.qpos[self.qpos_ids].copy()

    def get_ee_pos(self):
        return self.data.site_xpos[self.ee_site_id].copy()

    def get_ee_rot(self):
        return self.data.site_xmat[self.ee_site_id].reshape(3,3).copy()

    def get_attachment_pos(self):
        return self.data.site_xpos[self.attachment_site_id].copy()

    def get_peg_tip_pos(self):
        return self.data.site_xpos[self.peg_tip_site_id].copy()

    def get_hole_pose(self):
        backup = self.data.qpos.copy()
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.hole_key_id)

        ee_pos = self.get_ee_pos()
        ee_rot = self.get_ee_rot()
        attachment_pos = self.get_attachment_pos()
        peg_tip_pos = self.get_peg_tip_pos()

        self.data.qpos[:] = backup
        mujoco.mj_forward(self.model, self.data)

        return ee_pos, ee_rot, attachment_pos, peg_tip_pos

    def jacobian(self):
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        return np.vstack([jacp[:, self.dof_ids], jacr[:, self.dof_ids]])
