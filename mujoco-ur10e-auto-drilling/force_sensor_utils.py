import numpy as np

def normalize(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def world_force_from_site_sensor(data, model, sensor_name='wrist_force', site_name='wrist_ft_sensor'):
    adr = model.sensor(sensor_name).adr
    local_force = np.array(data.sensordata[adr:adr+3], dtype=float)
    site_id = model.site(site_name).id
    R = data.site_xmat[site_id].reshape(3, 3).copy()
    world_force = R @ local_force
    return local_force, world_force

def project_force_along_axis(world_force, axis_world):
    axis = normalize(np.array(axis_world, dtype=float))
    return float(np.dot(world_force, axis))
