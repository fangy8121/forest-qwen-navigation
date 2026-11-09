"""Offline numeric adapter for Survey Planner's clamped uniform cubic splines.

Matches bspline_utils' control-point / interval convention using SciPy's
independent evaluator. No ROS runtime, optimizer or flight authorization.
"""
import numpy as np
from scipy.interpolate import BSpline


class CatkinSpline:
    schema = 'survey_planner_cubic_world_v1'

    def __init__(self, control_points, interval):
        points = np.asarray(control_points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4 or not np.isfinite(points).all():
            raise ValueError('Need at least four finite world-XYZ control points')
        if not np.isfinite(interval) or interval <= 0:
            raise ValueError('Knot interval must be finite and positive')
        self.control_points = points.copy()
        self.interval = float(interval)
        self.duration = (len(points)-3)*self.interval
        # Degree-three open uniform knot vector, matching bspline::set_interval.
        knots = np.clip((np.arange(len(points)+4)-3)*self.interval, 0., self.duration)
        self.curve = BSpline(knots, self.control_points, 3, extrapolate=False)
        self.derivatives = [self.curve.derivative(order) for order in (1, 2, 3)]

    def sample(self, t):
        t = np.asarray(t, dtype=float)
        if not np.isfinite(t).all() or (t < 0).any() or (t > self.duration).any():
            raise ValueError('Sampling time must lie within the spline duration')
        return self.curve(t), self.derivatives[0](t), self.derivatives[1](t)

    def derivative_bounds(self):
        """Conservative full-interval norm bounds from derivative convex hulls.

        These are nominal kinematic bounds, not tracking or collision guarantees.
        """
        result = {}
        for name, curve in zip(('speed', 'acceleration', 'jerk'), self.derivatives):
            active = len(curve.t)-curve.k-1
            result[name] = float(np.linalg.norm(curve.c[:active], axis=1).max())
        return result

    def as_dict(self):
        return dict(schema=self.schema, frame='world', position_unit='m', time_unit='s',
                    control_points=self.control_points.tolist(), interval=self.interval)

    @classmethod
    def from_dict(cls, value):
        if (value.get('schema') != cls.schema or value.get('frame') != 'world'
                or value.get('position_unit') != 'm' or value.get('time_unit') != 's'):
            raise ValueError('Spline schema, frame and units must be explicit')
        return cls(value['control_points'], value['interval'])
