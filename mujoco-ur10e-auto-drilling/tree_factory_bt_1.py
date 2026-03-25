from bt import BehaviorTree, Sequence
from actions_bt import (
    ResetHome,
    MoveToPreApproach,
    CartesianApproach,
    AdmittanceInsert,
    Drill,
    Retract,
)
from adu_process import ADUProcessController


def create_tree(robot):
    adu = ADUProcessController()

    root = Sequence([
        ResetHome(robot),
        MoveToPreApproach(robot),
        CartesianApproach(robot),
        AdmittanceInsert(robot),
        Drill(robot, adu),
        Retract(robot),
    ], name="drilling_sequence")

    return BehaviorTree(root)
