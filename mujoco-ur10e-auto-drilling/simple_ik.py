import mujoco
import numpy as np

class SimpleIK:
    def __init__(self, model, data, site_name='drill_tcp'):
        self.model = model
        self.data = data
        self.site_id = model.site(site_name).id
        self.nq = model.nu

    def tcp_position(self, q):
        q_backup = self.data.qpos.copy()
        self.data.qpos[:self.nq] = q
        mujoco.mj_forward(self.model, self.data)
        pos = self.data.site_xpos[self.site_id].copy()
        self.data.qpos[:] = q_backup
        mujoco.mj_forward(self.model, self.data)
        return pos

    def solve(self, target_xyz, q_init, max_iters=80, alpha=0.10):
        q = q_init.copy()
        eps = 1e-4
        for _ in range(max_iters):
            p = self.tcp_position(q)
            err = target_xyz - p
            if np.linalg.norm(err) < 5e-3:
                break

            J = np.zeros((3, self.nq))
            for i in range(self.nq):
                q2 = q.copy()
                q2[i] += eps
                p2 = self.tcp_position(q2)
                J[:, i] = (p2 - p) / eps

            A = J @ J.T + 1e-3 * np.eye(3)
            dq = alpha * (J.T @ np.linalg.solve(A, err))
            dq = np.clip(dq, -0.03, 0.03)
            q = q + dq
        return q
