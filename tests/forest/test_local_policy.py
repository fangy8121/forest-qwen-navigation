import numpy as np
from omni_drones.navigation.forest.guidance import Guidance, GuidanceTracker
from omni_drones.navigation.forest.local_policy import LocalObservation, bound_velocity, local_reward, supervise_velocity
from omni_drones.navigation.forest.local import VelocityTrajectory, certificate
from omni_drones.navigation.forest.planning import PathSafety
from omni_drones.navigation.forest.teacher import DEFAULT_FLIGHT


def test_diagonal_velocity_obeys_vector_limit():
    velocity = bound_velocity([100., -100., 100.], .9)
    assert np.linalg.norm(velocity) <= .9000001
    np.testing.assert_array_equal(np.sign(velocity), [1., -1., 1.])


def test_reward_favors_real_progress_over_hover_and_penalizes_shield():
    w = dict(progress=10., complete=10., failure=10., time=.005, deviation=.01, smoothness=.02, shield=.1)
    zero = np.zeros(3)
    hover = local_reward(4., 4., zero, zero, zero, zero, False, False, w)
    forward = local_reward(4., 3.9, zero, zero, zero, zero, False, False, w)
    rejected = local_reward(4., 4., zero, [1.,0,0], zero, zero, False, False, w)
    assert forward > 0 > hover > rejected
    assert local_reward(4., 4., zero, zero, zero, zero, False, True, w) < rejected


def test_observation_retains_unknown_and_changes_with_sensor_geometry():
    encoder = LocalObservation()
    tracker = GuidanceTracker(Guidance('task', 1, 'segment', np.array([[0.,0,0], [4.,0,0]])))
    tracker.update(np.zeros(3), 0., 0.)
    state = np.r_[np.zeros(3), [1.,0,0,0], np.zeros(6)]
    empty = PathSafety(np.empty((0,3)), [[0.,0,0]], None, DEFAULT_FLIGHT, height_bounds=(-1.,2.))
    blocked = PathSafety([[1.2,0,0]], [[0.,0,0]], None, DEFAULT_FLIGHT, height_bounds=(-1.,2.))
    inputs = (state, tracker, np.zeros(3), np.zeros(3), False, '', 1.)
    a, b = encoder.encode(empty, *inputs), encoder.encode(blocked, *inputs)
    assert a.shape == (encoder.size,) and np.isfinite(a).all()
    grid = a[:-27].reshape(-1,3)
    assert (grid[:,1] == 0).any()
    assert np.count_nonzero(a[:-27] != b[:-27]) > 0
    np.testing.assert_array_equal(a[-27:], b[-27:])


def test_supervision_preserves_forward_motion_when_descent_is_unobserved():
    x, y = np.meshgrid(np.arange(-2.,4.1,.1), np.arange(-2.,2.1,.1))
    evidence = np.c_[x.ravel(), y.ravel(), np.full(x.size,.08)]
    safety = PathSafety(np.empty((0,3)), evidence, None, DEFAULT_FLIGHT)
    raw, zero = np.array([.6,0.,-.6]), np.zeros(3)
    rejected = VelocityTrajectory(zero, zero, zero, raw, DEFAULT_FLIGHT)
    assert certificate(safety, rejected, zero)[0] == 'UNOBSERVED'
    path, executed, reason, certified, _ = supervise_velocity(safety, zero, zero, zero, raw)
    assert certified and reason == 'UNOBSERVED'
    assert executed[0] > 0 and executed[2] == 0
    assert certificate(safety, path, zero)[0] == ''
