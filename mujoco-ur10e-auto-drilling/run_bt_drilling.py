import argparse
import time
import mujoco
import mujoco.viewer

from bt import Status
from tree_factory_bt import create_tree
from mujoco_adapter import MujocoRobot
from config_bt import XML_PATH, HEADLESS_STEPS

def main(headless=False):
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    robot = MujocoRobot(model, data)
    tree = create_tree(robot)

    if headless:
        for i in range(HEADLESS_STEPS):
            status = tree.tick()
            robot.step_sim()
            if status in (Status.SUCCESS, Status.FAILURE):
                print(f"[RUN] finished with {status.name} at step {i}")
                break
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t0 = time.time()
            status = tree.tick()
            robot.step_sim()
            viewer.sync()
            if status in (Status.SUCCESS, Status.FAILURE):
                print(f"[RUN] finished with {status.name}")
                time.sleep(1.0)
                break
            dt = model.opt.timestep - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    main(headless=args.headless)
