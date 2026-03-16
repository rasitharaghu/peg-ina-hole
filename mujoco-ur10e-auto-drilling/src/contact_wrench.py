from __future__ import annotations
import numpy as np, mujoco

class ContactWrenchEstimator:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, tcp_site_name: str='tcp'):
        self.model=model; self.data=data
        self.tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tcp_site_name)
    def estimate_world_wrench(self) -> np.ndarray:
        return np.zeros(6, dtype=float)
