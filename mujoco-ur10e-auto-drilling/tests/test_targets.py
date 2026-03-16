from src.target_loader import load_targets

def test_load_targets():
    targets = load_targets('assets/targets.json')
    assert len(targets) == 1
