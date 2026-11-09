"""Wall-time viewer cadence; no physics, simulator clock or sensor dependencies.

Pass a RenderCadence instance to IsaacEnv.enable_render. The environment still
executes every physics substep; this callback controls only its render Boolean.
Counters describe render requests, not completed GPU frames or a realtime claim.
"""
from __future__ import annotations

import math
from numbers import Real
import time


class RenderCadence:
    """Request at most one viewer render per wall-time period.

    The first call renders immediately. After a long pause, a single render is
    requested and the deadline is rebased on the current time; missed frames
    are never replayed. Repeated substeps at the same wall time cannot render
    more than once. No sleeps or simulation-time adjustments are performed.

    ``clock`` supplies monotonic seconds, allowing deterministic tests.
    ``on_render(substep)`` optionally updates a camera before accepted renders.
    It should be lightweight; failures propagate rather than hiding a GUI fault.
    """

    def __init__(self, render_hz=20., *, clock=time.monotonic, on_render=None):
        if (not isinstance(render_hz, Real) or isinstance(render_hz, bool)
                or not math.isfinite(render_hz) or render_hz <= 0):
            raise ValueError('render_hz must be finite and positive')
        if not callable(clock):
            raise TypeError('clock must be callable')
        if on_render is not None and not callable(on_render):
            raise TypeError('on_render must be callable or None')
        self._render_hz = float(render_hz)
        self._period = 1./self._render_hz
        if not math.isfinite(self._period) or self._period <= 0:
            raise ValueError('render_hz must have a finite positive wall-time period')
        self._clock, self.on_render = clock, on_render
        self._first_call = self._last_call = None
        self._first_render = self._last_render = self._next_render = None
        self._minimum_interval = self._maximum_interval = None
        self._calls = self._renders = 0

    @property
    def render_hz(self):
        return self._render_hz

    @property
    def render_count(self):
        return self._renders

    @property
    def skipped_count(self):
        return self._calls-self._renders

    def should_render(self, substep=0):
        now = self._clock()
        if (not isinstance(now, Real) or isinstance(now, bool) or not math.isfinite(now)):
            raise ValueError('render clock must return finite monotonic seconds')
        now = float(now)
        if self._last_call is not None and now < self._last_call:
            raise ValueError('render clock must be monotonic')
        if self._first_call is None:
            self._first_call = now
        self._last_call = now
        self._calls += 1
        # A tiny period-relative tolerance avoids losing exact boundary ticks
        # to floating-point addition; it is not a GUI/physics timing allowance.
        if self._next_render is not None and now+self._period*1e-9 < self._next_render:
            return False
        if self._last_render is not None:
            interval = now-self._last_render
            self._minimum_interval = (interval if self._minimum_interval is None
                                      else min(self._minimum_interval, interval))
            self._maximum_interval = (interval if self._maximum_interval is None
                                      else max(self._maximum_interval, interval))
        if self._first_render is None:
            self._first_render = now
        self._last_render = now
        self._next_render = now+self._period
        self._renders += 1
        if self.on_render is not None:
            self.on_render(substep)
        return True

    def __call__(self, substep=0):
        return self.should_render(substep)

    def stats(self):
        """Return a JSON-safe snapshot without reading or advancing the clock."""
        elapsed = (None if self._first_call is None else self._last_call-self._first_call)
        span = (None if self._first_render is None else self._last_render-self._first_render)
        return dict(clock='monotonic_wall_seconds', requested_render_hz=self._render_hz,
            period_seconds=self._period, callback_calls=self._calls,
            render_requests=self._renders, render_skips=self.skipped_count,
            observed_wall_seconds=elapsed,
            observed_render_request_hz=((self._renders-1)/span if span is not None and span > 0 else None),
            minimum_render_interval_seconds=self._minimum_interval,
            maximum_render_interval_seconds=self._maximum_interval,
            first_render_wall_time=self._first_render, last_render_wall_time=self._last_render,
            next_render_wall_time=self._next_render,
            scope='viewer render requests only; every physics step remains the caller responsibility')
