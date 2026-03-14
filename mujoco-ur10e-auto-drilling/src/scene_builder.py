from __future__ import annotations

from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
import numpy as np


def build_scene_with_panel_offset(
    base_scene_path: str | Path,
    workpiece_body_name: str,
    delta_xyz: np.ndarray,
) -> Path:
    tree = ET.parse(str(base_scene_path))
    root = tree.getroot()

    target_body = None
    for body in root.iter("body"):
        if body.attrib.get("name") == workpiece_body_name:
            target_body = body
            break

    if target_body is None:
        raise ValueError(f"Could not find body '{workpiece_body_name}'")

    old_pos = np.fromstring(target_body.attrib.get("pos", "0 0 0"), sep=" ")
    new_pos = old_pos + delta_xyz
    target_body.set("pos", f"{new_pos[0]} {new_pos[1]} {new_pos[2]}")

    tmp = tempfile.NamedTemporaryFile(prefix="drilling_trial_", suffix=".xml", delete=False)
    tree.write(tmp.name, encoding="utf-8", xml_declaration=True)
    return Path(tmp.name)
