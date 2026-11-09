"""A task can wait at its measured position without inventing a motion leg."""
import math

import numpy as np
import pytest

from omni_drones.navigation.forest.guidance import Guidance, GuidanceTracker


def guide(points, revision=1):
    return Guidance('online_task', revision, f'goal_{revision}', np.asarray(points))


@pytest.mark.parametrize('points', [
    [[2., 3., 4.]],
    [[2., 3., 4.], [2., 3., 4.], [2., 3., 4.]],
])
def test_stationary_anchor_has_finite_local_target_and_feedback(points):
    tracker = GuidanceTracker(guide(points), now=5.)
    local = tracker.update([1., 3., 3.5], math.pi/2, 7.)
    np.testing.assert_allclose(local, [0., -1., .5], atol=1e-12)
    np.testing.assert_array_equal(tracker.target, [2., 3., 4.])
    feedback = tracker.feedback(7., 'WAITING', '', .1)
    assert feedback.progress == 0.
    assert feedback.lateral_error == 1.
    assert feedback.height_error == -.5
    assert feedback.no_progress_seconds == 2.
    # Repeated controller ticks preserve a fixed world anchor and its clock.
    np.testing.assert_array_equal(tracker.update([2., 3., 4.], 0., 8.), np.zeros(3))
    assert tracker.last_progress == 5.


def test_repeated_route_anchors_preserve_moving_projection_and_lookahead():
    clean = [[0., 0., 2.], [3., 0., 2.], [3., 4., 2.]]
    repeated = [clean[0], clean[0], clean[1], clean[1], clean[2], clean[2]]
    a, b = GuidanceTracker(guide(clean)), GuidanceTracker(guide(repeated))
    for time, position in enumerate(([0., 0., 2.], [1., 0., 2.],
                                     [2.9, 0., 2.], [3., 1., 2.])):
        np.testing.assert_array_equal(a.update(position, 0., time), b.update(position, 0., time))
        assert a.progress == b.progress
        np.testing.assert_array_equal(a.error, b.error)


def test_stationary_and_moving_goals_can_replace_each_other():
    tracker = GuidanceTracker(guide([[0., 0., 2.]]))
    tracker.update([0., 0., 2.], 0., 1.)
    assert tracker.set(guide([[0., 0., 2.], [4., 0., 2.]], revision=2), now=1.)
    np.testing.assert_array_equal(tracker.update([1., 0., 2.], 0., 2.), [2., 0., 0.])
    assert tracker.progress == 1.
    assert tracker.set(guide([[1., 0., 2.]], revision=3), now=3.)
    np.testing.assert_array_equal(tracker.update([1., 0., 2.], 0., 4.), np.zeros(3))
    assert tracker.progress == 0.
    assert not tracker.set(guide([[9., 9., 9.]], revision=2), now=5.)
    np.testing.assert_array_equal(tracker.target, [1., 0., 2.])


@pytest.mark.parametrize('points', [[], [1., 2., 3.], [[1., 2.]],
                                     [[1., 2., np.nan]], [[1., 2., np.inf]]])
def test_empty_non_xyz_and_nonfinite_guidance_is_rejected(points):
    with pytest.raises(ValueError, match='finite world XYZ'):
        GuidanceTracker(guide(points))
