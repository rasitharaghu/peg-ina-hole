import argparse
import time
import mujoco
import mujoco.viewer

from bt import Status
from tree_factory_bt import create_tree
from mujoco_adapter import MujocoRobot
from config_bt import XML_PATH, HEADLESS_STEPS, USE_CUSTOM_HOME_QPOS, CUSTOM_HOME_QPOS


def main(headless=False, debug_forward=False):
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    robot = MujocoRobot(model, data)
    tree = create_tree(robot)

    print("[RUN] XML_PATH:", XML_PATH)
    print("[RUN] debug_forward:", debug_forward)
    print("[RUN] USE_CUSTOM_HOME_QPOS:", USE_CUSTOM_HOME_QPOS)
    if USE_CUSTOM_HOME_QPOS:
        print("[RUN] CUSTOM_HOME_QPOS:", CUSTOM_HOME_QPOS)

    if headless:
        for i in range(HEADLESS_STEPS):
            status = tree.tick()

            if debug_forward:
                mujoco.mj_forward(model, data)
            else:
                robot.step_sim()

            if status == Status.SUCCESS:
                print(f"[RUN] full mission finished with SUCCESS at step {i}")
                break
            elif status == Status.FAILURE:
                print(f"[RUN] full mission finished with FAILURE at step {i}")
                break
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        finished = False

        while viewer.is_running() and not finished:
            status = tree.tick()

            if debug_forward:
                mujoco.mj_forward(model, data)
            else:
                robot.step_sim()

            viewer.sync()

            if status == Status.SUCCESS:
                print("[RUN] full mission finished with SUCCESS")
                finished = True
            elif status == Status.FAILURE:
                print("[RUN] full mission finished with FAILURE")
                finished = True

        if finished:
            for _ in range(100):
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(0.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--debug-forward", action="store_true",
                        help="Use mj_forward instead of mj_step for pure kinematic debugging")
    args = parser.parse_args()
    main(headless=args.headless, debug_forward=args.debug_forward)
