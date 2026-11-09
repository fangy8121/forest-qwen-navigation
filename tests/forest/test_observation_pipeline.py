from types import SimpleNamespace

import numpy as np
import pytest

from omni_drones.navigation.forest.observation_pipeline import ObservationPipeline, observation_stop_cause


class SlowExecutor:
    def __init__(self, clock):
        self.clock, self.requests = clock, []

    def submit(self, fn, *args, **kwargs):
        start = self.clock[0]
        self.requests.append((start, args, kwargs))
        return SimpleNamespace(done=lambda:self.clock[0] >= start+.11-1e-9,
            result=lambda:fn(*args, **kwargs))


def test_slow_worker_resumes_between_sensor_ticks_without_aging_the_capture():
    clock = [0.]
    executor = SlowExecutor(clock)
    pipeline = ObservationPipeline(executor, lambda frames, **context:dict(frames=frames, **context))
    latest, ages = None, []
    for step in range(101):
        clock[0] = step/100
        if step % 10 == 0:
            pipeline.push(step, captured=clock[0], source_time=clock[0])
        result = pipeline.poll()
        if result is not None:
            latest = result
        if latest is not None:
            ages.append(clock[0]-latest['captured'])
    # A 110 ms worker must consume the waiting 100 ms scan at 110 ms,
    # instead of idling until the 200 ms scan and crossing a 300 ms deadline.
    assert executor.requests[1][0] == pytest.approx(.11)
    assert executor.requests[1][2]['captured'] == .1
    assert max(ages) < .3
    submitted = [frame for _, args, _ in executor.requests for frame in args[0]]
    assert sorted(submitted+pipeline.frames) == list(range(0, 101, 10))
    assert len(submitted) == len(set(submitted))


def test_batched_frames_keep_the_latest_real_capture_and_context():
    clock = [0.]
    executor = SlowExecutor(clock)
    pipeline = ObservationPipeline(executor, lambda frames, **context:dict(frames=frames, **context))
    pipeline.push('first', captured=10., goal='old')
    pipeline.push('second', captured=10.1, goal='old')
    pipeline.push('third', captured=10.2, goal='new')
    clock[0] = .11
    assert pipeline.poll()['frames'] == ['first']
    _, args, context = executor.requests[1]
    assert args[0] == ['second', 'third']
    assert context == dict(captured=10.2, goal='new')


def test_braking_reserve_never_extends_the_observation_validity_limit():
    assert observation_stop_cause(.249, .3, .05) is None
    assert observation_stop_cause(.25, .3, .05) == 'OBSERVATION_DEADLINE'
    assert observation_stop_cause(.3, .3, .05) == 'OBSERVATION_DEADLINE'
    assert observation_stop_cause(.301, .3, .05) == 'STALE'
    assert observation_stop_cause(.3, .3) is None
    for value in (-.01, .3, np.nan):
        with pytest.raises(ValueError):
            observation_stop_cause(.1, .3, value)


def test_stale_observation_only_uncertifies_a_stop_that_is_still_moving():
    # Run 443cc8d3: one 0.337 s observation just after a measured recovery stop
    # (speed ~0) ended the mission; braking from rest sweeps ~mm.
    from omni_drones.navigation.forest.observation_pipeline import stale_stop_uncertified
    assert not stale_stop_uncertified(.2, .3, .4, .4, .05)            # fresh
    assert stale_stop_uncertified(.34, .3, .01, .01, None)             # no exemption configured
    assert not stale_stop_uncertified(.34, .3, .01, .02, .05)          # stale at rest
    assert stale_stop_uncertified(.34, .3, .2, .0, .05)                # stale while moving
    assert stale_stop_uncertified(.34, .3, .0, .2, .05)                # reference still moving
