from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from src.kinematics import TaskState
from src.utils import unit

@dataclass
class ServoController:
    kp_xy: float
    kd_xy: float
    kp_ori: float
    kd_ori: float
    kz_force: float
    force_damping: float
    kp_null: float
    max_joint_step: float
    damping_lambda: float
    max_z_velocity: float
    desired_force_z: float = 0.0

    def compute_q_des(self, task: TaskState, q: np.ndarray, qd: np.ndarray, x_des: np.ndarray, desired_axis: np.ndarray, measured_force: np.ndarray, home_q: np.ndarray, dt: float) -> np.ndarray:
        Jp = task.jacp[:, :6]
        Jr = task.jacr[:, :6]
        tcp_vel = Jp @ qd
        ang_vel = Jr @ qd
        pos_err = x_des - task.position
        v_xy_cmd = np.array([
            self.kp_xy * pos_err[0] - self.kd_xy * tcp_vel[0],
            self.kp_xy * pos_err[1] - self.kd_xy * tcp_vel[1],
        ], dtype=float)
        v_z_cmd = 0.0
        desired_axis = unit(desired_axis)
        current_axis = task.rotation[:, 2]
        ori_err = np.cross(current_axis, desired_axis)
        w_cmd = self.kp_ori * ori_err - self.kd_ori * ang_vel
        twist_cmd = np.hstack([v_xy_cmd[0], v_xy_cmd[1], v_z_cmd, w_cmd])
        J = np.vstack([Jp, Jr])
        A = J @ J.T + (self.damping_lambda ** 2) * np.eye(6)
        J_pinv = J.T @ np.linalg.inv(A)
        dq_cmd = J_pinv @ twist_cmd
        N = np.eye(6) - J_pinv @ J
        dq_null = self.kp_null * (home_q - q)
        dq_cmd = dq_cmd + N @ dq_null
        dq_cmd = np.clip(dq_cmd, -self.max_joint_step, self.max_joint_step)
        return q + dq_cmd * dt
