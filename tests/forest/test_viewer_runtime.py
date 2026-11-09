"""Deterministic wall-time render throttling without simulator imports."""
import json

import pytest

from omni_drones.navigation.forest.viewer_runtime import RenderCadence


class Clock:
    def __init__(self, now=0.):
        self.now, self.reads = now, 0

    def __call__(self):
        self.reads += 1
        return self.now


def test_default_twenty_hz_limits_viewer_but_every_physics_step_is_executed():
    clock = Clock()
    cadence = RenderCadence(clock=clock)
    physics_steps, render_times = [], []
    # Match the environment's structure: physics advances regardless of the
    # callback's result. Only its render Boolean differs between substeps.
    for step in range(100):
        clock.now = step*.01
        render = cadence(step % 2)
        physics_steps.append(step)
        if render:
            render_times.append(clock.now)
    assert physics_steps == list(range(100))
    assert render_times == pytest.approx([i*.05 for i in range(20)])
    assert cadence.render_count == 20 and cadence.skipped_count == 80
    stats = cadence.stats()
    assert stats['callback_calls'] == 100
    assert stats['observed_render_request_hz'] == pytest.approx(20.)
    assert stats['minimum_render_interval_seconds'] == pytest.approx(.05)


def test_ten_hz_is_configurable_and_independent_of_substep_number():
    clock = Clock(300.)
    cadence = RenderCadence(10., clock=clock)
    assert cadence.should_render(7)
    assert not cadence.should_render(0)
    clock.now += .099
    assert not cadence.should_render(7)
    clock.now += .001
    assert cadence.should_render(0)
    assert cadence.render_hz == 10.


def test_long_pause_requests_one_frame_and_does_not_burst_to_catch_up():
    clock = Clock()
    cadence = RenderCadence(20., clock=clock)
    assert cadence(0)
    clock.now = 100.
    assert cadence(0)
    for substep in range(20):
        assert not cadence(substep)
    clock.now = 100.049
    assert not cadence(0)
    clock.now = 100.05
    assert cadence(0)
    assert cadence.render_count == 3
    assert cadence.stats()['maximum_render_interval_seconds'] == 100.


def test_late_tick_rebases_deadline_instead_of_scheduling_an_early_next_frame():
    clock = Clock()
    cadence = RenderCadence(10., clock=clock)
    assert cadence()
    clock.now = .17
    assert cadence()
    clock.now = .20  # old phase boundary; only 30 ms after actual render
    assert not cadence()
    clock.now = .27
    assert cadence()


@pytest.mark.parametrize('hz', [0., -10., float('nan'), float('inf'), '20', None, True, 1e-320])
def test_invalid_rates_are_rejected(hz):
    with pytest.raises(ValueError, match='render_hz'):
        RenderCadence(hz)


@pytest.mark.parametrize('reading', [float('nan'), float('inf'), None, '1.0', True])
def test_invalid_clock_reading_does_not_change_statistics(reading):
    clock = Clock()
    cadence = RenderCadence(clock=clock)
    assert cadence()
    before = cadence.stats()
    clock.now = reading
    with pytest.raises(ValueError, match='clock'):
        cadence()
    assert cadence.stats() == before


def test_backward_clock_is_rejected_without_resetting_the_deadline():
    clock = Clock(10.)
    cadence = RenderCadence(clock=clock)
    assert cadence()
    clock.now = 9.
    with pytest.raises(ValueError, match='monotonic'):
        cadence()
    clock.now = 10.049
    assert not cadence()
    clock.now = 10.05
    assert cadence()


def test_statistics_are_bounded_clock_free_and_json_serializable():
    clock = Clock()
    cadence = RenderCadence(clock=clock)
    empty = cadence.stats()
    assert clock.reads == 0 and empty['render_requests'] == 0
    assert empty['observed_render_request_hz'] is None
    cadence()
    reads = clock.reads
    stats = cadence.stats()
    json.dumps(stats, allow_nan=False)
    assert clock.reads == reads
    stats['render_requests'] = 1000
    assert cadence.render_count == 1  # returned snapshot cannot mutate counters


def test_camera_callback_is_invoked_only_for_accepted_render_requests():
    clock = Clock()
    camera_updates = []
    cadence = RenderCadence(10., clock=clock, on_render=camera_updates.append)
    assert cadence(1)
    assert not cadence(2)
    clock.now = .1
    assert cadence(3)
    assert camera_updates == [1, 3]


def test_callback_errors_are_not_silently_hidden():
    def fail(substep):
        raise RuntimeError('viewer camera failed')

    cadence = RenderCadence(clock=Clock(), on_render=fail)
    with pytest.raises(RuntimeError, match='camera failed'):
        cadence(0)
    assert cadence.stats()['render_requests'] == 1


def test_clock_and_optional_render_hook_must_be_callable():
    with pytest.raises(TypeError, match='clock'):
        RenderCadence(clock=0.)
    with pytest.raises(TypeError, match='on_render'):
        RenderCadence(on_render=1)
