import argparse
import mujoco
import mujoco.viewer
from common_control_new import XML_PATH, DEFAULT_SLEEP, move_to_hole_phase, step_view

def main(sleep=DEFAULT_SLEEP):
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        move_to_hole_phase(model, data, viewer, sleep)
        print("[DONE] close viewer to exit")
        while viewer.is_running():
            step_view(model, data, viewer, sleep)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP)
    args = parser.parse_args()
    main(sleep=args.sleep)
