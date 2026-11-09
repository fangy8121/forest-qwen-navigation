"""局部策略的因果观测、速度边界与任务奖励；不依赖 Isaac。"""
import numpy as np
from .geometry import local_to_world, world_to_local, yaw_from_quat
from .local import BODY_NUMERIC_TOLERANCE, VelocityTrajectory, certificate
from .trajectory import QuinticPath


def bound_velocity(action, limit):
    velocity = np.tanh(action)*limit
    return velocity/max(1., np.linalg.norm(velocity)/limit)


def supervise_velocity(safety, position, velocity, acceleration, raw, horizon=.2):
    masks = np.array([[1,1,1], [1,1,0], [1,0,1], [0,1,1], [1,0,0], [0,1,0], [0,0,1], [0,0,0]])
    candidates = np.vstack([masks*raw, .5*raw, .25*raw])
    order = np.argsort(np.linalg.norm(candidates-raw, axis=1), kind='stable')
    first_reason = ''
    for executed in candidates[order]:
        path = VelocityTrajectory(position, velocity, acceleration, executed, safety.flight)
        reason, envelope = certificate(safety, path, position, horizon)
        first_reason = first_reason or reason
        if not reason:
            return path, executed, first_reason, True, envelope
    path = QuinticPath.brake(position, velocity, acceleration, safety.flight.max_acceleration, safety.flight.max_jerk)
    envelope = path.remaining(0., position)
    certified = safety.clear(envelope) and safety.known(envelope, position, body_padding=BODY_NUMERIC_TOLERANCE)
    return path, np.zeros(3), first_reason, certified, envelope


def local_reward(before, after, error, raw, executed, previous, success, failed, weights):
    return (weights['progress']*(before-after) + weights['complete']*success - weights['failure']*failed
            - weights['time'] - weights['deviation']*np.linalg.norm(error)
            - weights['smoothness']*np.square(executed-previous).sum()
            - weights['shield']*np.linalg.norm(raw-executed))


class LocalObservation:
    def __init__(self, resolution=.4):
        lower, upper = np.array([-.8, -2.4, -.8]), np.array([3.2, 2.4, 1.2])
        shape = np.rint((upper-lower)/resolution).astype(int)+1
        self.centers = lower+np.indices(shape).reshape(3, -1).T*resolution
        self.size = 3*len(self.centers)+27

    def encode(self, safety, state, tracker, acceleration, executed, shield, reason, remaining):
        position, yaw = state[:3], yaw_from_quat(state[3:7])
        centers = local_to_world(self.centers, position, yaw)
        clear = safety.clear_points(centers)
        known = safety.known_points(centers, position, body_padding=BODY_NUMERIC_TOLERANCE)
        distance = np.minimum(safety.obstacles.query(centers)[0], 3.)/3.
        grid = np.stack([~clear, known, distance], axis=1).ravel()
        vectors = world_to_local(np.array([tracker.target, position+state[7:10], position+tracker.error,
                                           position+acceleration, position+executed]), position, yaw)
        target, velocity, error, acc, last = vectors
        features = np.r_[np.clip(target/4., -1., 1.), velocity/safety.flight.max_speed, state[3:7],
                         np.clip(error/2., -1., 1.), acc/safety.flight.max_acceleration,
                         last/safety.flight.max_speed, min(1., (tracker.arc[-1]-tracker.progress)/4.),
                         np.clip((np.array(safety.height_bounds)-position[2])/2., -1., 1.),
                         remaining, shield, [reason == r for r in ('UNOBSERVED', 'OBSTACLE', 'DYNAMICS')]]
        return np.r_[grid, features].astype(np.float32)
