"""世界坐标任务引导、按实际进度移动的子目标与执行反馈。"""
from dataclasses import dataclass
import numpy as np
from .geometry import world_to_local


@dataclass
class Guidance:
    route_id: str
    revision: int
    segment_id: str
    points: np.ndarray
    mode: str = 'FOLLOW'
    speed_limit: float = 1.
    corridor: np.ndarray | None = None
    source_time: float = 0.
    height_bounds: tuple[float, float] | None = None


@dataclass
class Feedback:
    route_id: str
    revision: int
    segment_id: str
    progress: float
    lateral_error: float
    height_error: float
    status: str
    reason: str
    observation_age: float
    no_progress_seconds: float


class GuidanceTracker:
    def __init__(self, guidance, lookahead=2., now=0.):
        self.lookahead = lookahead
        self.progress = 0.
        self.last_progress = now
        self.set(guidance, now)

    def set(self, guidance, now=0.):
        if hasattr(self, 'guidance'):
            old = self.guidance
            if guidance.route_id == old.route_id and guidance.revision <= old.revision:
                return False
            self.progress = 0.
        points = np.asarray(guidance.points, dtype=float)
        if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
                or not np.isfinite(points).all()):
            raise ValueError('guidance points must be a nonempty array of finite world XYZ')
        # A measured hold can have just one anchor. Remove consecutive repeated
        # anchors so a moving route never projects onto a zero-length segment.
        points = points[np.r_[True, np.any(np.diff(points, axis=0) != 0., axis=1)]]
        self.guidance = guidance
        self.points = points
        self.delta = np.diff(self.points, axis=0)
        self.lengths = np.linalg.norm(self.delta, axis=1)
        self.arc = np.r_[0., np.cumsum(self.lengths)]
        self.reported_progress = self.progress
        self.last_progress = now
        return True

    def update(self, position, yaw, now):
        position = np.asarray(position)
        if len(self.points) == 1:
            self.target = self.points[0].copy()
            self.error = position-self.target
            return world_to_local([self.target], position, yaw)[0]
        fraction = np.clip(np.sum((position-self.points[:-1])*self.delta, axis=1)
                           / self.lengths**2, 0., 1.)
        projected = self.points[:-1]+fraction[:, None]*self.delta
        arc = self.arc[:-1]+fraction*self.lengths
        distance = np.linalg.norm(projected-position, axis=1)
        distance[(arc < self.progress-0.2) | (arc > self.progress+self.lookahead)] = np.inf
        index = np.argmin(distance)
        self.progress = max(self.progress, float(arc[index])) if np.isfinite(distance[index]) else self.progress
        if self.progress-self.reported_progress >= .05:
            self.last_progress = now
            self.reported_progress = self.progress
        target_arc = min(self.arc[-1], self.progress+self.lookahead)
        self.target = np.array([np.interp(target_arc, self.arc, self.points[:, d]) for d in range(3)])
        center = np.array([np.interp(self.progress, self.arc, self.points[:, d]) for d in range(3)])
        self.error = position-center
        return world_to_local([self.target], position, yaw)[0]

    def feedback(self, now, status, reason, observation_age):
        g = self.guidance
        return Feedback(g.route_id, g.revision, g.segment_id, self.progress,
                        float(np.linalg.norm(self.error[:2])), float(self.error[2]),
                        status, reason, observation_age, now-self.last_progress)
