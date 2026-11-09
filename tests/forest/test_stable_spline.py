"""Future-reference handoffs preserve active execution until a candidate is usable."""
from types import SimpleNamespace

import numpy as np
import pytest

from omni_drones.navigation.forest.catkin_spline import CatkinSpline
from omni_drones.navigation.forest.local import VelocityTrajectory
from omni_drones.navigation.forest.observed_esdf import ObservedESDF
from omni_drones.navigation.forest.planning import PathSafety
from omni_drones.navigation.forest.spline_optimizer import SplineValidation, endpoint_controls
from omni_drones.navigation.forest.stable_spline import StableSplinePlanner, height_guided_route
from omni_drones.navigation.forest.teacher import DEFAULT_FLIGHT


FLIGHT = SimpleNamespace(max_speed=1., max_acceleration=1., max_jerk=4.)


def test_short_flat_astar_route_keeps_public_height_slope_and_future_anchor():
    route = np.array([[.1, 0., 2.], [.4, 0., 2.], [1.2, 0., 2.]])
    corrected = height_guided_route(route, [2., 0., 2.1], [0., 0., 2.], Safety())
    np.testing.assert_array_equal(corrected[0], route[0])
    assert .05 < corrected[-1, 2]-route[-1, 2] < .1
    assert np.all(np.diff(corrected[:, 2]) > 0)


def test_height_guidance_does_not_override_unknown_space_or_vertical_avoidance():
    route = np.array([[0., 0., 2.], [.4, 0., 2.], [1.2, 0., 2.]])
    safety = Safety()
    safety.known = lambda *args, **kwargs: False
    np.testing.assert_array_equal(height_guided_route(route, [2., 0., 2.2], route[0], safety), route)
    vertical = route.copy(); vertical[1, 2] += .4
    np.testing.assert_array_equal(height_guided_route(vertical, [2., 0., 2.2], route[0], Safety()), vertical)


def test_small_continuous_height_slope_is_not_quantized_back_to_a_flat_route():
    route = np.array([[0., 0., 2.], [.4, 0., 2.], [1.2, 0., 2.]])
    corrected = height_guided_route(route, [2., 0., 2.004], route[0], Safety())
    assert corrected[-1, 2] > 2.002
    np.testing.assert_array_equal(corrected[0], route[0])


class Safety:
    flight = FLIGHT

    def clear(self, points):
        return True

    def known(self, points, actual, body_padding=0.):
        return True


class RecordingOptimizer:
    def __init__(self, accepted=True):
        self.accepted = accepted
        self.calls = []

    def optimize(self, route, position, velocity, acceleration, field, safety,
                 *, nominal_speed, actual_position):
        self.calls.append(dict(route=np.asarray(route).copy(), start=np.array([position, velocity, acceleration]),
                               field=field, actual=np.asarray(actual_position).copy()))
        path = (CatkinSpline(endpoint_controls(position, velocity, acceleration, route[-1], 1.), 1.)
                if self.accepted else None)
        return SimpleNamespace(path=path, validation=SplineValidation(
            self.accepted, '' if self.accepted else 'UNOBSERVED_CURVE'), attempts=[], seconds=.001)


def setup_planner(*, accepted=True):
    command = VelocityTrajectory(np.zeros(3), [.1, 0., 0.], np.zeros(3), [.2, 0., 0.], FLIGHT)
    measured = np.zeros(13)
    measured[:3] = command.sample(.1)[0]+[-.025, .01, 0.]
    field = object()
    snapshot = dict(safety=Safety(), points=np.empty((0, 3)), evidence=np.empty((0, 3)),
                    distance_field=field, decision=dict(route=[[0., 0., 0.], [.8, 0., 0.], [1.4, 0., 0.]]),
                    state=measured, timestamp=.1, captured=10.)
    planner = StableSplinePlanner(FLIGHT, activation_lead=.4, candidate_max_age=.8, control_dt=.01)
    optimizer = RecordingOptimizer(accepted)
    planner.optimizer = optimizer
    return planner, command, snapshot, optimizer


def submit(planner, command, snapshot):
    return planner.submit(command, 0., .1, snapshot, 2, 3, [1.4, 0., .2], .5)


def finish_worker(planner):
    # Resolve the worker deterministically; simulated clocks remain explicit.
    planner._future.result(timeout=5.)


def poll(planner, command, now=.5, wall_time=10.4, generation=2, revision=3):
    return planner.poll(now, wall_time, generation, revision, command, 0.)


def test_future_boundary_is_exact_and_waiting_never_changes_active_reference():
    planner, command, snapshot, optimizer = setup_planner()
    original = [value.copy() for value in command.sample(np.array([.1, .5, .7]))]
    with planner:
        assert planner.can_submit and not planner.busy
        assert submit(planner, command, snapshot)
        assert planner.busy and not planner.can_submit
        assert not submit(planner, command, snapshot)
        finish_worker(planner)
        candidate, _ = poll(planner, command, now=.49, wall_time=10.39)
        assert candidate is None
        assert planner.busy
        candidate, event = poll(planner, command)
        assert event['reason'] == 'READY'
        assert candidate.request.switch_time == pytest.approx(.5)
        np.testing.assert_allclose(candidate.path.sample(0.), command.sample(.5), atol=1e-12)
        assert not planner.busy and planner.can_submit
    for expected, actual in zip(original, command.sample(np.array([.1, .5, .7]))):
        np.testing.assert_array_equal(actual, expected)
    assert optimizer.calls[0]['field'] is snapshot['distance_field']
    # Evidence exemption belongs to the actual measured body, not the future anchor.
    np.testing.assert_array_equal(optimizer.calls[0]['actual'], snapshot['state'][:3])
    assert np.linalg.norm(optimizer.calls[0]['actual']-optimizer.calls[0]['start'][0]) > .02
    # The fine-height near target is appended after the coarse-grid route.
    np.testing.assert_allclose(optimizer.calls[0]['route'][-1], [1.4, 0., .2])


@pytest.mark.parametrize('generation,revision', [(3, 3), (2, 4)])
def test_old_generation_or_goal_cannot_replace_active_command(generation, revision):
    planner, command, snapshot, _ = setup_planner()
    before = command.sample(.5)
    with planner:
        assert submit(planner, command, snapshot)
        finish_worker(planner)
        candidate, event = poll(planner, command, generation=generation, revision=revision)
        assert candidate is None and event['reason'] == 'SUPERSEDED'
        assert planner.can_submit
    np.testing.assert_array_equal(command.sample(.5), before)


def test_late_candidate_is_rejected_instead_of_restarting_its_clock():
    planner, command, snapshot, _ = setup_planner()
    before = command.sample(.51)
    with planner:
        assert submit(planner, command, snapshot)
        finish_worker(planner)
        candidate, event = poll(planner, command, now=.51, wall_time=10.41)
        assert candidate is None and event['reason'] == 'MISSED_SWITCH'
        assert planner.can_submit
    np.testing.assert_array_equal(command.sample(.51), before)


def test_candidate_wall_age_is_bounded_independently_of_the_promised_tick():
    planner, command, snapshot, _ = setup_planner()
    with planner:
        assert submit(planner, command, snapshot)
        finish_worker(planner)
        candidate, event = poll(planner, command, wall_time=10.95)
        assert candidate is None and event['reason'] == 'STALE_CANDIDATE'


def test_command_change_without_generation_change_still_fails_boundary_check():
    planner, command, snapshot, _ = setup_planner()
    changed = VelocityTrajectory([.1, 0., 0.], [.1, 0., 0.], np.zeros(3), [.2, 0., 0.], FLIGHT)
    with planner:
        assert submit(planner, command, snapshot)
        finish_worker(planner)
        candidate, event = poll(planner, changed)
        assert candidate is None and event['reason'] == 'START_STATE'


def test_failed_optimization_does_not_change_or_stop_existing_reference():
    planner, command, snapshot, _ = setup_planner(accepted=False)
    before = command.sample(.5)
    with planner:
        assert submit(planner, command, snapshot)
        finish_worker(planner)
        candidate, event = poll(planner, command)
        assert candidate is None and event['reason'] == 'NO_CANDIDATE'
        assert planner.can_submit
    np.testing.assert_array_equal(command.sample(.5), before)


def test_height_smoothing_rejection_retries_observed_route_with_same_boundary_and_validator():
    planner, command, snapshot, _ = setup_planner()

    class HeightRejectingOptimizer(RecordingOptimizer):
        def optimize(self, route, *args, **kwargs):
            self.accepted = np.ptp(np.asarray(route)[:, 2]) < 1e-8
            return super().optimize(route, *args, **kwargs)

    optimizer = HeightRejectingOptimizer()
    planner.optimizer = optimizer
    with planner:
        assert planner.submit(command, 0., .1, snapshot, 2, 3, [3., 0., .4], .5)
        finish_worker(planner)
        candidate, event = poll(planner, command)
        assert candidate is not None and candidate.validation.accepted
        trials = event['optimization']['route_trials']
        assert [t['route'] for t in trials] == ['height_guided', 'observed_astar']
        assert trials[0]['validation']['reason'] == 'UNOBSERVED_CURVE'
        assert trials[1]['validation']['accepted']
    assert len(optimizer.calls) == 2
    for call in optimizer.calls:
        np.testing.assert_array_equal(call['start'], command.sample(.5))
        assert call['field'] is snapshot['distance_field']
        np.testing.assert_array_equal(call['actual'], snapshot['state'][:3])
    assert np.ptp(optimizer.calls[0]['route'][:, 2]) > .1
    assert np.ptp(optimizer.calls[1]['route'][:, 2]) < 1e-8


@pytest.mark.parametrize('reason, held', [('UNOBSERVED_BRAKE', True), ('OBSTACLE_BRAKE', False)])
def test_unobserved_descent_is_retried_at_the_start_height(reason, held):
    # Run 4b639dae: from a stop, a 0.4 m A* step down 3.4 m ahead failed every
    # stop check for 20 s; the level route was observed.
    planner, command, snapshot, _ = setup_planner()
    snapshot['decision'] = dict(route=[[0., 0., 0.], [.8, 0., 0.], [1.4, 0., -.4]])

    class DescentRejectingOptimizer(RecordingOptimizer):
        def optimize(self, route, *args, **kwargs):
            self.accepted = np.ptp(np.asarray(route)[:, 2]) < 1e-8
            result = super().optimize(route, *args, **kwargs)
            if not self.accepted:
                result.validation.reason = reason
            return result

    optimizer = DescentRejectingOptimizer()
    planner.optimizer = optimizer
    with planner:
        assert planner.submit(command, 0., .1, snapshot, 2, 3, [1.4, 0., -.4], .5)
        finish_worker(planner)
        candidate, event = poll(planner, command)
        names = [t['route'] for t in event['optimization']['route_trials']]
        if held:
            assert candidate is not None and names == ['observed_astar', 'level_hold']
            assert np.ptp(optimizer.calls[1]['route'][:, 2]) < 1e-8
            np.testing.assert_allclose(optimizer.calls[1]['route'][-1, :2], [1.4, 0.])
        else:
            assert candidate is None and names == ['observed_astar']


def test_real_optimizer_reuses_observed_field_and_yields_a_bounded_handoff():
    grid = np.meshgrid(np.arange(-1.6, 4.01, .2), np.arange(-2., 2.01, .2),
                       np.arange(-.8, .81, .2), indexing='ij')
    evidence = np.stack(grid, axis=-1).reshape(-1, 3)
    field = ObservedESDF([], evidence)
    safety = PathSafety([], evidence, None, DEFAULT_FLIGHT)
    command = VelocityTrajectory(np.zeros(3), [.1, 0., 0.], np.zeros(3), [.1, 0., 0.], DEFAULT_FLIGHT)
    state = np.zeros(13)
    state[:3] = command.sample(.1)[0]
    snapshot = dict(safety=safety, points=np.empty((0, 3)), evidence=evidence, distance_field=field,
                    decision=dict(route=[state[:3], [.8, 0., 0.], [1.4, 0., 0.]]),
                    state=state, timestamp=.1, captured=10.)
    with StableSplinePlanner(DEFAULT_FLIGHT) as planner:
        assert planner.submit(command, 0., .1, snapshot, 2, 3, [1.4, 0., 0.], .5)
        finish_worker(planner)
        candidate, event = poll(planner, command)
        assert candidate is not None, event
        assert candidate.validation.accepted
        np.testing.assert_allclose(candidate.path.sample(0.), command.sample(.5), atol=1e-9)
        assert candidate.path.spline.derivative_bounds()['acceleration'] <= DEFAULT_FLIGHT.max_acceleration*1.001
