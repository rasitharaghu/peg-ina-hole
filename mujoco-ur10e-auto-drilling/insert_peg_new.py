import argparse
import mujoco
import mujoco.viewer
from common_control_new import (
    XML_PATH, DEFAULT_SLEEP, insertion_or_drill_phase,
    INSERTION_DEPTH, INSERTION_DURATION, INSERTION_K_POS,
    INSERTION_DAMPING, INSERTION_INTEGRATION_DT, INSERTION_MAX_TRAVEL, step_view
)

def main(sleep=DEFAULT_SLEEP):
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('hole').id)
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("[ADMITTANCE INSERT NEW]")
        insertion_or_drill_phase(
            model, data, viewer=viewer, sleep_s=sleep,
            depth=INSERTION_DEPTH, duration=INSERTION_DURATION, k_pos=INSERTION_K_POS,
            damping=INSERTION_DAMPING, integration_dt=INSERTION_INTEGRATION_DT,
            max_travel=INSERTION_MAX_TRAVEL, label="INSERT"
        )
        print("[DONE] close viewer to exit")
        while viewer.is_running():
            step_view(model, data, viewer, sleep)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP)
    args = parser.parse_args()
    main(sleep=args.sleep)
