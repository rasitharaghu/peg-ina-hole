import numpy as np
import mujoco
import mujoco.viewer
import time
from full_pose_runner_config import *
from full_pose_runner_mujoco_adapter import MujocoRobot

def ori_err(Rd, Rc):
    Re = Rd @ Rc.T
    return 0.5*np.array([
        Re[2,1]-Re[1,2],
        Re[0,2]-Re[2,0],
        Re[1,0]-Re[0,1]
    ])

def move(robot, pos_t, rot_t, label, viewer):
    for i in range(MAX_STEPS_PER_PHASE):
        p = robot.get_ee_pos()
        R = robot.get_ee_rot()

        pe = pos_t - p
        oe = ori_err(rot_t, R)

        if np.linalg.norm(pe) < POS_TOL and np.linalg.norm(oe) < ORI_TOL:
            print(label, "reached")
            return

        J = robot.jacobian()
        task = np.concatenate([pe, ORI_GAIN*oe])

        dq = J.T @ np.linalg.inv(J@J.T + 0.01*np.eye(6)) @ task

        q = robot.get_qpos()
        robot.set_qpos(q + POSE_GAIN*dq)

        mujoco.mj_forward(robot.model, robot.data)
        viewer.sync()
        time.sleep(0.03)

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)
robot = MujocoRobot(model, data)

hole_p, hole_R, attach_p, tip_p = robot.get_hole_pose()

pre_p = HOLE_TARGET_POS + PREAPPROACH_OFFSET

insert_axis = tip_p - attach_p
insert_axis = insert_axis / np.linalg.norm(insert_axis)

insert_p = HOLE_TARGET_POS + INSERTION_DEPTH * insert_axis

print("INSERT AXIS:", insert_axis)
print("INSERT TARGET:", insert_p)

with mujoco.viewer.launch_passive(model, data) as viewer:
    robot.set_qpos(CUSTOM_HOME_QPOS)
    time.sleep(1)

    move(robot, pre_p, hole_R, "PRE", viewer)
    move(robot, HOLE_TARGET_POS, hole_R, "HOLE", viewer)
    move(robot, insert_p, hole_R, "INSERT", viewer)

    time.sleep(5)
