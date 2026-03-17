# UR10e Peg-in-Hole Insertion via Admittance Control

A collection of Python scripts simulating the peg-in-hole insertion problem using **Kinematic Admittance Control**. This repository demonstrates how to handle robotic misalignments and compliant interactions using the MuJoCo physics engine without the need for external vision systems.

## 📌 Project Description
This repository simulates a **Universal Robots UR10e** mounted on a table, equipped with a custom parallel-jaw gripper holding a cylindrical peg. The objective is to insert the peg into a hole in a rigid wall. 

The project focuses on **Admittance Control**, where virtual forces are calculated to guide the robot. By setting different gains ($K$) for different axes, the robot remains stiff in the insertion direction (X) but compliant in the alignment directions (Y and Z), allowing it to "self-correct" during the insertion process.

### Key Technical Specs:
* **Robot:** UR10e
* **Gripper:** Custom parallel jaw (no camera/external force sensor).
* **Control:** Kinematic Admittance (Task-space mapping to Joint-space via DLS).
* **Environment:** MuJoCo Physics Engine.

---

## 📂 File Descriptions

| File | Description |
| :--- | :--- |
| `universal_robots/` | Contains all assets, meshes, and textures for the UR10e robot. |
| `scene.xml` | The main simulation setup: UR10e, table, gripper, peg, and the wall with the hole. |
| `scene.py` | Simple utility script to launch the MuJoCo viewer to inspect the visual scene. |
| `move_to_hole.py` | Solves Inverse Kinematics (IK) to reach the "pre-insertion" goal position using Damped Least Squares (DLS). |
| `insert_peg.py` | The main execution script using kinematic admittance control to perform the insertion. |
| `analyse_ctrl.py` | Helper script that injects noise into initial positions and plots trajectory tracking (X, Y, Z) and virtual forces ($F_x, F_y, F_z$). |
| `gravity_comp.py` | Verification script to ensure gravity compensation is working correctly within the MuJoCo model. |
| `Peg-in-Hole.mp4` | Demo Video of peg insertion. |
| `kinematic_admittance_analysis` | Sample Performance analysis plot of insertion task for a given mis-aligned start position. |

---

## ⚙️ How to Run

### Prerequisites
1. **MuJoCo:** `pip install mujoco`
2. **Python Dependencies:** `pip install numpy matplotlib`

### Execution
You can run any script directly from your terminal. For example:

```bash
# To view the static scene
python scene.py

# To run the insertion with admittance control
python insert_peg.py

# To generate performance analysis plots
python analyse_ctrl.py
