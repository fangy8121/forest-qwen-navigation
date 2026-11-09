"""OmniPerception MID-360 角度序列 + Isaac Lab Warp 理想化扫描。"""
import importlib.util
from pathlib import Path
import numpy as np
from .geometry import quat_matrix


def downward_cone(rays, half_angle_deg):
    """Body-frame unit directions spread evenly over a cap around straight down."""
    if rays < 1 or not 0 < half_angle_deg < 90:
        raise ValueError('downward cone needs rays >= 1 and 0 < half angle < 90 deg')
    index = np.arange(rays)+.5
    cos_tilt = 1.-index/rays*(1.-np.cos(np.radians(half_angle_deg)))
    sin_tilt = np.sqrt(1.-cos_tilt**2)
    azimuth = index*np.pi*(3.-np.sqrt(5.))
    return np.c_[sin_tilt*np.cos(azimuth), sin_tilt*np.sin(azimuth), -cos_tilt]


class Mid360:
    """MID-360 pattern scan, optionally with a downward sensing cone.

    ``downward`` (dict of ``rays``, ``half_angle_deg``, ``max_range``) emulates
    the deployment M300 RTK's downward vision/ToF: an upright MID-360 never
    observes the space under its own body (run 6244d06b). The cone rays share
    the LiDAR origin and frame, so every consumer treats them as LiDAR rays; a
    ray without a return reports exactly its own range, never the LiDAR's.
    """
    def __init__(self, root, mesh, device, max_range=15., downsample=5,
                 mount_position=(0,0,.15), mount_quaternion=(1,0,0,0), downward=None):
        file = Path(root).expanduser()/'LidarSensor/LidarSensor/sensor_pattern/sensor_lidar/genera_lidar_scan_pattern.py'
        spec = importlib.util.spec_from_file_location('omni_perception_scan_pattern', file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.pattern = module.LivoxGenerator('mid360')
        self.mesh, self.device = mesh, device
        self.max_range, self.downsample = max_range, downsample
        self.mount_position = np.asarray(mount_position)
        self.mount_quaternion = np.asarray(mount_quaternion)
        self.source = str(file)
        self.downward = None
        if downward is not None:
            down_range = float(downward['max_range'])
            if not 0 < down_range <= max_range:
                raise ValueError('downward range must be positive and within the LiDAR range')
            self.downward = (downward_cone(int(downward['rays']), float(downward['half_angle_deg'])), down_range)

    def reset(self):
        self.pattern.currStartIndex = 0

    def scan(self, timestamp, position, quaternion):
        import torch
        from scipy.spatial.transform import Rotation
        from isaaclab.utils.warp import raycast_mesh
        theta, phi = self.pattern.sample_ray_angles(downsample=self.downsample)
        directions = np.c_[np.cos(theta)*np.cos(phi), np.sin(theta)*np.cos(phi), np.sin(phi)]
        body = quat_matrix(quaternion)
        rotation = body @ quat_matrix(self.mount_quaternion)
        sensor_position = np.asarray(position) + body @ self.mount_position
        directions_world = directions @ rotation.T
        starts = torch.as_tensor(np.broadcast_to(sensor_position, directions_world.shape).copy(), dtype=torch.float32, device=self.device)
        dirs = torch.as_tensor(directions_world, dtype=torch.float32, device=self.device)
        # 本地 Isaac Lab 2.3.2 的距离返回实现要求 [B,N,3]，虽然文档写作 [N,3]。
        hits, distances, _, _ = raycast_mesh(starts[None], dirs[None], self.mesh, self.max_range, return_distance=True)
        hits, distances = hits[0].cpu().numpy(), distances[0].cpu().numpy()
        valid = np.isfinite(distances) & (distances >= self.pattern.laser_min_range) & (distances <= self.max_range)
        if self.downward is not None:
            cone, down_range = self.downward
            down_world = cone @ body.T
            down_starts = torch.as_tensor(np.broadcast_to(sensor_position, down_world.shape).copy(),
                                          dtype=torch.float32, device=self.device)
            down_hits, down_distances, _, _ = raycast_mesh(down_starts[None],
                torch.as_tensor(down_world, dtype=torch.float32, device=self.device)[None],
                self.mesh, down_range, return_distance=True)
            down_hits, down_distances = down_hits[0].cpu().numpy(), down_distances[0].cpu().numpy()
            down_valid = (np.isfinite(down_distances) & (down_distances >= self.pattern.laser_min_range)
                          & (down_distances <= down_range))
            # Free space is evidenced only up to the cone's own range.
            down_distances = np.where(down_valid, down_distances, down_range)
            directions_world = np.vstack([directions_world, down_world])
            hits, distances = np.vstack([hits, down_hits]), np.r_[distances, down_distances]
            valid = np.r_[valid, down_valid]
        # 未命中保留 inf 距离和 NaN 点，绝不伪造成最远距离障碍。
        points = np.full_like(hits, np.nan)
        points[valid] = (hits[valid]-sensor_position) @ rotation
        q = Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]]
        return dict(timestamp=np.array(timestamp), points_sensor=points, valid=valid,
                    directions_world=directions_world, distances=distances,
                    sensor_position=sensor_position, sensor_quaternion=q,
                    position=np.array(position), quaternion=np.array(quaternion), max_range=np.array(self.max_range))
