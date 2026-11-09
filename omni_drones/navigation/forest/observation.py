"""Causal 3D LiDAR history and equal-height views of the observed volume."""
from collections import deque
from threading import Lock
import numpy as np
from scipy.spatial import cKDTree
from .geometry import (BOUNDS, Z_BOUNDS, HEIGHT_LEVELS, GRID_RESOLUTION, REPRESENTATION,
                       INFLATION, VERTICAL_RADIUS, RADIUS, BODY_VERTICAL_RADIUS, BODY_HALF_HEIGHT, EVIDENCE_TOLERANCE,
                       observed_centers, observed_body_height, world_to_local, quat_matrix)


RAY_STEP = .15

def representation_spec():
    return dict(name=REPRESENTATION, bounds=list(BOUNDS), z_bounds=list(Z_BOUNDS),
                height_levels=list(HEIGHT_LEVELS), resolution=GRID_RESOLUTION,
                horizontal_radius=INFLATION, vertical_radius=VERTICAL_RADIUS,
                body_horizontal_radius=RADIUS, collision_body_vertical_radius=BODY_VERTICAL_RADIUS,
                physical_half_height=BODY_HALF_HEIGHT,
                lower_obstacle_inset=dict(z_max=min(HEIGHT_LEVELS)-VERTICAL_RADIUS,
                                          content='observed_returns_only', placement='right'),
                known_body_rule='reference tilt: physical_half_height + body_horizontal_radius * max_acceleration / 9.81',
                trajectory_sample_margin=.05, evidence_tolerance=EVIDENCE_TOLERANCE, ray_step=RAY_STEP)


def volume_bounds(center_clearance=None):
    """Keep hits outside the center's planning bounds when the body can touch them."""
    horizontal = INFLATION if center_clearance is None else center_clearance
    vertical = VERTICAL_RADIUS if center_clearance is None else center_clearance
    lower = np.array([BOUNDS[0]-horizontal, BOUNDS[2]-horizontal, Z_BOUNDS[0]-vertical])
    upper = np.array([BOUNDS[1]+horizontal, BOUNDS[3]+horizontal, Z_BOUNDS[1]+vertical])
    return np.floor(lower/GRID_RESOLUTION)*GRID_RESOLUTION, np.ceil(upper/GRID_RESOLUTION)*GRID_RESOLUTION


def lower_obstacles(points):
    """主高度带以下的观测障碍；保留其信息，不推断投影内的自由空间。"""
    points = np.asarray(points).reshape(-1, 3)
    return points[points[:, 2] < min(HEIGHT_LEVELS)-VERTICAL_RADIUS]


class History:
    def __init__(self, seconds=3.0, center_clearance=None, ray_traversal='sampled_v1'):
        self.seconds = seconds
        if ray_traversal not in ('sampled_v1', 'voxel_dda_v1'):
            raise ValueError('unknown free-evidence traversal')
        self.center_clearance, self.ray_traversal = center_clearance, ray_traversal
        self.reset()

    def reset(self):
        self.frames = deque()
        self.latest = -float('inf')

    def add(self, frame):
        t = float(frame['timestamp'])
        if t < self.latest:
            raise ValueError('nonmonotonic timestamp: reset history at episode boundary')
        self.latest = t
        self.frames.append(frame)
        while self.frames and float(self.frames[0]['timestamp']) < t-self.seconds-1e-8:
            self.frames.popleft()

    def snapshot(self, now, position, yaw, include_evidence=True):
        lower, upper = volume_bounds(self.center_clearance)
        shape = np.rint((upper-lower)/GRID_RESOLUTION).astype(int)+1
        seen = np.zeros(shape, dtype=bool)
        points = []
        for frame in self.frames:
            if not now-self.seconds-1e-8 <= float(frame['timestamp']) <= now:
                continue
            rotation = quat_matrix(frame['sensor_quaternion'])
            hit = frame['points_sensor'][frame['valid']]
            world = hit @ rotation.T + frame['sensor_position']
            local = world_to_local(world, position, yaw)
            points.append(local[((local >= lower) & (local <= upper)).all(axis=1)])
            if not include_evidence:
                continue
            distances = np.minimum(frame['distances'], float(frame['max_range']))
            origin = world_to_local(frame['sensor_position'], position, yaw)
            directions = world_to_local(frame['directions_world'], np.zeros(3), yaw)
            if self.ray_traversal == 'voxel_dda_v1':
                from .ray_evidence import trace_free_voxels
                trace_free_voxels(seen, origin, directions, distances, lower, GRID_RESOLUTION)
                continue
            steps = np.arange(0., float(frame['max_range']), RAY_STEP)
            # Bound temporary arrays; no-hit rays contribute free evidence up to max range.
            for start in range(0, len(directions), 512):
                along = origin+directions[start:start+512, None, :]*steps[None, :, None]
                mask = (steps[None, :] < distances[start:start+512, None])
                mask &= ((along >= lower) & (along <= upper)).all(axis=2)
                ijk = np.rint((along[mask]-lower)/GRID_RESOLUTION).astype(np.int32)
                seen[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
        points = np.concatenate(points) if points else np.empty((0, 3))
        if len(points):
            ijk = np.rint((points-lower)/GRID_RESOLUTION).astype(np.int32)
            seen[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = False
        evidence = np.argwhere(seen)*GRID_RESOLUTION+lower
        return points, evidence


def grid_observation(points, evidence, bounds=BOUNDS, resolution=GRID_RESOLUTION, height=0., max_acceleration=1.,
                     center_clearance=None):
    """One drone-center height: obstacle distance and evidence for the whole body."""
    xs = np.arange(bounds[0], bounds[1]+resolution/2, resolution)
    ys = np.arange(bounds[2], bounds[3]+resolution/2, resolution)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    centers = np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, height)]
    points = np.asarray(points)
    if center_clearance is not None:
        distance = cKDTree(points.reshape(-1, 3)).query(centers)[0].reshape(xx.shape)
    else:
        relevant = points[np.abs(points[:, 2]-height) <= VERTICAL_RADIUS] if len(points) else points
        distance = (cKDTree(relevant[:, :2]).query(centers[:, :2])[0].reshape(xx.shape)
                    if len(relevant) else np.full(xx.shape, np.inf))
    seen = observed_centers(evidence, centers, vertical=observed_body_height(max_acceleration)).reshape(xx.shape)
    return xs, ys, distance, seen


_render_lock = Lock()


def render_observation(points, evidence, goal, output, view='bev', path=None, corridor=None, max_acceleration=1.):
    with _render_lock:
        _render_observation(points, evidence, goal, output, view, path, corridor, max_acceleration)


def _render_observation(points, evidence, goal, output, view='bev', path=None, corridor=None, max_acceleration=1.):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    points, goal = np.asarray(points), np.asarray(goal)
    evidence = cKDTree(np.asarray(evidence).reshape(-1, 3))
    if view == 'bev':
        fig, panels = plt.subplots(1, len(HEIGHT_LEVELS)+1, figsize=(5*len(HEIGHT_LEVELS), 5),
                                  dpi=100, gridspec_kw={'width_ratios': [1]*len(HEIGHT_LEVELS)+[.4]})
        axes, inset = panels[:-1], panels[-1]
        for ax, height in zip(axes, HEIGHT_LEVELS):
            xs, ys, distance, seen = grid_observation(points, evidence, height=height, max_acceleration=max_acceleration)
            rgb = np.full((*seen.shape, 3), .58)
            rgb[seen] = [.90, .93, .95]
            rgb[distance <= INFLATION] = [.95, .62, .30]
            half = GRID_RESOLUTION/2
            ax.imshow(rgb.transpose(1, 0, 2), origin='lower',
                      extent=[xs[0]-half, xs[-1]+half, ys[0]-half, ys[-1]+half])
            hits = points[np.abs(points[:, 2]-height) <= VERTICAL_RADIUS]
            if len(hits):
                ax.scatter(hits[:, 0], hits[:, 1], s=1, c='black')
            ax.add_patch(plt.Circle((0, 0), .5, fill=False, color='blue'))
            ax.arrow(0, 0, .8, 0, width=.035, color='blue')
            ax.scatter(*goal[:2], marker='*', s=100, c='green')
            ax.annotate(f'goal z={goal[2]:+.2f}', goal[:2], fontsize=8, xytext=(3, 5), textcoords='offset points')
            if path is not None:
                route = np.asarray(path)
                ax.plot(route[:, 0], route[:, 1], 'r.--', alpha=.65)
                for point in route:
                    ax.annotate(f'{point[2]:+.1f}', point[:2], fontsize=7, color='darkred')
            if corridor is not None and len(corridor):
                border = np.vstack([corridor, corridor[0]])
                ax.plot(border[:, 0], border[:, 1], color='purple', linestyle='--', linewidth=1)
            ax.set(xlim=BOUNDS[:2], ylim=BOUNDS[2:], xlabel='x forward (m)',
                   title=f'Drone center z = {height:+g} m')
            ax.set_xticks(np.arange(BOUNDS[0], BOUNDS[1]+1, 2))
            ax.set_yticks(np.arange(BOUNDS[2], BOUNDS[3]+1, 2))
            ax.grid(alpha=.25)
            ax.set_aspect('equal')
        lower = lower_obstacles(points)
        inset.set_facecolor('.58')
        if len(lower):
            # 合并重复命中；颜色保留高度，避免把低处障碍当作首层碰撞区。
            _, indices = np.unique(np.rint(lower/GRID_RESOLUTION), axis=0, return_index=True)
            lower = lower[indices]
            dots = inset.scatter(lower[:, 0], lower[:, 1], c=lower[:, 2], s=2,
                                 cmap='viridis', vmin=volume_bounds()[0][2],
                                 vmax=min(HEIGHT_LEVELS)-VERTICAL_RADIUS)
            bar = fig.colorbar(dots, cax=inset.inset_axes([1.04, 0., .06, 1.]),
                               ticks=[volume_bounds()[0][2], min(HEIGHT_LEVELS)-VERTICAL_RADIUS])
            bar.ax.tick_params(labelsize=6)
        inset.set(xlim=BOUNDS[:2], ylim=BOUNDS[2:], aspect='equal', xticks=[], yticks=[])
        inset.set_title(f'Lower hits z < {min(HEIGHT_LEVELS)-VERTICAL_RADIUS:g} m', fontsize=7)
        axes[0].set_ylabel('y left (m)')
        fig.suptitle('Orange: blocked body envelope | Light: observed body volume | Gray: unknown', fontsize=11)
    elif view == 'oblique':
        fig = plt.figure(figsize=(7, 6), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        for height in HEIGHT_LEVELS:
            xs, ys, distance, seen = grid_observation(points, evidence, height=height, max_acceleration=max_acceleration)
            xx, yy = np.meshgrid(xs, ys, indexing='ij')
            colors = np.where((distance <= INFLATION).ravel(), 'orange', np.where(seen.ravel(), 'whitesmoke', 'gray'))
            ax.scatter(xx.ravel(), yy.ravel(), np.full(xx.size, height), c=colors, s=2, alpha=.25)
        if len(points):
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='black')
        ax.quiver(0, 0, 0, 1, 0, 0, color='blue')
        ax.scatter(*goal, c='green', marker='*')
        if path is not None:
            ax.plot(*np.asarray(path).T, 'r.-')
        if corridor is not None and len(corridor):
            border = np.vstack([corridor, corridor[0]])
            ax.plot(border[:, 0], border[:, 1], np.zeros(len(border)), color='purple', linestyle='--')
        ax.view_init(elev=35, azim=-55)
        ax.set(xlim=BOUNDS[:2], ylim=BOUNDS[2:], zlim=Z_BOUNDS,
               xlabel='x forward (m)', ylabel='y left (m)', zlabel='z up (m)')
    else:
        raise ValueError('view must be bev or oblique')
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


class SampleSelector:
    """按位姿/观测变化采样，同一森林的重复飞行共享去重记录；不读取教师答案。"""
    def __init__(self, min_distance=.8, min_yaw_degrees=15., min_change=.12, duplicate_change=.04):
        self.parameters = dict(min_distance=min_distance, min_yaw_degrees=min_yaw_degrees,
                               min_change=min_change, duplicate_change=duplicate_change)
        self.accepted = []
        self.reset_episode()

    def reset_episode(self):
        self.previous = None

    def describe(self, points, evidence, goal, position, yaw, velocity=None, acceleration=None, max_acceleration=1.):
        layers = []
        evidence = cKDTree(np.asarray(evidence).reshape(-1, 3))
        for height in HEIGHT_LEVELS:
            xs, ys, distance, seen = grid_observation(points, evidence, resolution=.4, height=height, max_acceleration=max_acceleration)
            layer = seen.astype(np.uint8)
            layer[distance <= INFLATION] = 2
            layer[distance <= .25] = 3
            layers.append(layer)
        # 低处小图也参与去重，防止下方障碍变化被五张主图掩盖。
        lower = lower_obstacles(points)
        xx, yy = np.meshgrid(xs, ys, indexing='ij')
        layer = np.zeros(xx.shape, dtype=np.uint8)
        if len(lower):
            distance, index = cKDTree(lower[:, :2]).query(np.c_[xx.ravel(), yy.ravel()])
            active = distance <= INFLATION
            layer.ravel()[active] = 1+np.clip(np.rint(
                (lower[index[active], 2]-volume_bounds()[0][2])/GRID_RESOLUTION), 0, 254).astype(np.uint8)
        layers.append(layer)
        return dict(signature=np.stack(layers),goal=np.asarray(goal).copy(),
                    position=np.asarray(position).copy(),yaw=float(yaw),
                    velocity=np.zeros(3) if velocity is None else np.asarray(velocity).copy(),
                    acceleration=np.zeros(3) if acceleration is None else np.asarray(acceleration).copy())

    @staticmethod
    def change(a, b):
        active = (a != 0) | (b != 0)
        return np.count_nonzero((a != b) & active)/max(1,np.count_nonzero(active))

    def reason(self, candidate):
        p = self.parameters
        if self.previous is not None:
            old = self.previous
            yaw_delta = abs(np.arctan2(np.sin(candidate['yaw']-old['yaw']),np.cos(candidate['yaw']-old['yaw'])))
            if (np.linalg.norm(candidate['position']-old['position']) < p['min_distance'] and
                yaw_delta < np.deg2rad(p['min_yaw_degrees']) and
                np.linalg.norm(candidate['goal']-old['goal']) < .5 and
                np.linalg.norm(candidate['velocity']-old['velocity']) < .2 and
                np.linalg.norm(candidate['acceleration']-old['acceleration']) < .3 and
                self.change(candidate['signature'],old['signature']) < p['min_change']):
                return 'insufficient_change'
        for old in self.accepted:
            if (np.linalg.norm(candidate['goal']-old['goal']) < .3 and
                np.linalg.norm(candidate['velocity']-old['velocity']) < .2 and
                np.linalg.norm(candidate['acceleration']-old['acceleration']) < .3 and
                self.change(candidate['signature'],old['signature']) < p['duplicate_change']):
                return 'duplicate_observation'
        return None

    def accept(self, candidate):
        self.previous = candidate
        self.accepted.append(candidate)
