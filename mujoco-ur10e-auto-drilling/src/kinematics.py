from __future__ import annotations
from dataclasses import dataclass
import numpy as np, mujoco

@dataclass
class TaskState:
    position: np.ndarray
    rotation: np.ndarray
    jacp: np.ndarray
    jacr: np.ndarray

class MuJoCoKinematics:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, site_name: str='tcp'):
        self.model=model; self.data=data
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        self.nv = 6
    def site_state(self) -> TaskState:
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        pos = self.data.site_xpos[self.site_id].copy()
        rot = self.data.site_xmat[self.site_id].reshape(3,3).copy()
        return TaskState(pos, rot, jacp[:, :self.nv], jacr[:, :self.nv])
