"""Optional cyclic-GC scheduling and telemetry for a soft real-time loop.

Reference counting remains active. Deferred cyclic collections run only at a
caller-certified measured rest; this does not change observation age checks.
This is not a hard real-time guarantee or a policy for an actual flight ECU.
"""
import gc
import math
import time


class RestGarbageCollector:
    def __init__(self, *, defer=False, defer_full=False, trace=False, interval=15., force_interval=None):
        if defer and defer_full:
            raise ValueError('choose either legacy all-generation or full-generation deferral')
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError('collection interval must be finite and positive')
        if force_interval is not None and (not math.isfinite(force_interval) or force_interval < interval):
            raise ValueError('force interval must be finite and at least the collection interval')
        self.defer, self.defer_full = bool(defer), bool(defer_full)
        self.trace, self.interval = bool(trace or defer or defer_full), float(interval)
        # Collect at any measured rest after ``interval``; brake only for GC
        # after ``force_interval`` (run f830a311: 26 dedicated stops, 91 s).
        self.force_interval = self.interval if force_interval is None else float(force_interval)
        self.events, self._starts = [], {}
        self.now, self.last_collection, self.phase = 0., 0., 'control'
        self.resume_after_wall = None

    def _callback(self, phase, info):
        generation = info['generation']
        if phase == 'start':
            self._starts[generation] = time.perf_counter()
        elif generation in self._starts:
            self.events.append(dict(simulation_time=self.now, generation=generation,
                wall_seconds=time.perf_counter()-self._starts.pop(generation), phase=self.phase,
                collected=info.get('collected'), uncollectable=info.get('uncollectable')))

    def __enter__(self):
        self.was_enabled = gc.isenabled()
        self.original_threshold = gc.get_threshold()
        if self.trace:
            gc.callbacks.append(self._callback)
        if self.defer and self.was_enabled:
            gc.disable()
        elif self.defer_full and self.was_enabled:
            # Python 3.11's automatic generation-2 trigger is its third
            # threshold. Young cycles still collect at the original cadence.
            gc.set_threshold(*self.original_threshold[:2], 2**31-1)
        return self

    def full_collection_due(self, now):
        return bool(self.defer_full and self.was_enabled and self.resume_after_wall is None
                    and now-self.last_collection >= self.force_interval)

    @property
    def waiting_for_observation(self):
        return self.resume_after_wall is not None

    def release_after_observation(self, *, captured, age, max_age, certified, measured_rest):
        """A completed full GC cannot turn an earlier capture into fresh evidence."""
        if not self.waiting_for_observation:
            return False
        if (not measured_rest or not certified or not math.isfinite(captured) or not math.isfinite(age)
                or not 0 <= age <= max_age or captured <= self.resume_after_wall):
            return False
        self.resume_after_wall = None
        return True

    def collect_at_rest(self, now, certified_rest):
        self.now = float(now)
        if not ((self.defer or self.defer_full) and self.was_enabled and certified_rest
                and not self.waiting_for_observation and now-self.last_collection >= self.interval):
            return None
        self.phase = 'measured_rest'
        began = time.perf_counter()
        try:
            count = gc.collect(2)
        finally:
            self.phase = 'control'
        self.last_collection = float(now)
        finished = time.perf_counter()
        if self.defer_full:
            self.resume_after_wall = finished
        return dict(wall_seconds=finished-began, collected=count, certified_rest=True,
                    gc_generation=2, gc_start_wall=began, gc_end_wall=finished)

    def __exit__(self, *args):
        if self.trace:
            gc.callbacks.remove(self._callback)
        gc.set_threshold(*self.original_threshold)
        if self.was_enabled:
            gc.enable()
        else:
            gc.disable()
