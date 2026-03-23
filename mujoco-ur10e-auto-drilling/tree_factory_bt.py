from bt import BehaviorTree, Sequence, Selector
from actions_bt import (
    ResetHome,
    MoveToPreApproach,
    CartesianApproach,
    AdmittanceInsert,
    Drill,
    Retract,
    SafeAbortToHome,
)
from adu_process import ADUProcessController

def create_tree(robot):
    adu = ADUProcessController()
    normal_flow = Sequence([
        ResetHome(robot),
        MoveToPreApproach(robot),
        CartesianApproach(robot),
        AdmittanceInsert(robot),
        Drill(robot, adu),
        Retract(robot),
    ], name="normal_flow")

    root = Selector([
        normal_flow,
        SafeAbortToHome(robot),
    ], name="root")
    return BehaviorTree(root)
