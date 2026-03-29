import argparse
import mujoco
import mujoco.viewer
from common_control_new import (
    XML_PATH, DEFAULT_SLEEP, insertion_or_drill_phase,
    DRILL_DEPTH, DRILL_DURATION, DRILL_K_POS,
    DRILL_DAMPING, DRILL_INTEGRATION_DT, DRILL_MAX_TRAVEL, step_view
)

def main(sleep=DEFAULT_SLEEP):
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('hole').id)
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("[DRILL NEW]")
        insertion_or_drill_phase(
            model, data, viewer=viewer, sleep_s=sleep,
            depth=DRILL_DEPTH, duration=DRILL_DURATION, k_pos=DRILL_K_POS,
            damping=DRILL_DAMPING, integration_dt=DRILL_INTEGRATION_DT,
            max_travel=DRILL_MAX_TRAVEL, label="DRILL"
        )
        print("[DONE] close viewer to exit")
        while viewer.is_running():
            step_view(model, data, viewer, sleep)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP)
    args = parser.parse_args()
    main(sleep=args.sleep)
