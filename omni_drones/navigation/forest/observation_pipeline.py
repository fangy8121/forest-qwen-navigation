"""Keep one geometry job busy without waiting for the next sensor tick."""
import numpy as np


class ObservationPipeline:
    def __init__(self, executor, prepare):
        self.executor, self.prepare = executor, prepare
        self.future, self.frames, self.context = None, [], None

    def push(self, frame, **context):
        # Context belongs to the most recent real capture, not submission time.
        self.frames.append(frame)
        self.context = context
        self._submit()

    def _submit(self):
        if self.future is None and self.frames:
            frames, self.frames = self.frames, []
            self.future = self.executor.submit(self.prepare, frames, **self.context)

    def poll(self):
        if self.future is None or not self.future.done():
            return None
        result, self.future = self.future.result(), None
        self._submit()
        return result


def stale_stop_uncertified(age, max_age, actual_speed, reference_speed, rest_speed=None):
    """Whether a stale observation alone makes a latched stop uncertified.

    A stale map cannot certify the space a brake sweeps. At (near) rest the
    brake sweeps essentially nothing (0.05 m/s at 0.6 m/s^2 is ~2 mm), so the
    geometric check against the last observation stands (run 443cc8d3: one
    0.337 s observation right after a measured recovery stop ended the run).
    """
    if age <= max_age:
        return False
    if rest_speed is None:
        return True
    return not (actual_speed <= rest_speed and reference_speed <= rest_speed)


def observation_stop_cause(age, max_age, reserve=0.):
    """Reserve time to certify braking while the map is still fresh."""
    if not np.isfinite([age, max_age, reserve]).all() or not 0 <= reserve < max_age or age < 0:
        raise ValueError('finite observation age and 0 <= reserve < max_age required')
    if age > max_age:
        return 'STALE'
    if reserve and age >= max_age-reserve:
        return 'OBSERVATION_DEADLINE'
    return None
