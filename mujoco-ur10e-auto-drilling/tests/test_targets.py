from src.target_loader import load_targets


def test_targets_load():
    targets = load_targets("assets/targets.json")
    assert len(targets) > 0
    assert targets[0].position.shape == (3,)
    assert targets[0].normal.shape == (3,)
