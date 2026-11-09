import gc
import pytest
from omni_drones.navigation.forest.runtime_gc import RestGarbageCollector


def test_cyclic_collection_is_deferred_until_measured_rest_and_traced(monkeypatch):
    was_enabled = gc.isenabled()
    gc.enable()
    calls = []
    original = gc.collect
    def counted(generation=2):
        calls.append(generation)
        return original(generation)
    monkeypatch.setattr(gc, 'collect', counted)
    try:
        with RestGarbageCollector(defer=True) as guard:
            assert not gc.isenabled()
            assert guard.collect_at_rest(20., False) is None and not calls
            assert guard.collect_at_rest(20., True)['certified_rest']
            assert calls == [2] and guard.events
            assert all(e['phase'] == 'measured_rest' for e in guard.events)
            assert guard.collect_at_rest(21., True) is None
        assert gc.isenabled()
    finally:
        if not was_enabled:
            gc.disable()


def test_exception_restores_original_gc_state_and_removes_callback():
    was_enabled = gc.isenabled()
    callbacks = list(gc.callbacks)
    try:
        gc.disable()
        with pytest.raises(RuntimeError):
            with RestGarbageCollector(defer=True) as guard:
                assert guard.collect_at_rest(30., True) is None
                raise RuntimeError('control failure')
        assert not gc.isenabled()
        assert gc.callbacks == callbacks
    finally:
        if was_enabled:
            gc.enable()


def test_trace_only_does_not_disable_gc():
    original = gc.isenabled()
    with RestGarbageCollector(trace=True, defer=False) as guard:
        assert gc.isenabled() == original
        assert guard.collect_at_rest(30., True) is None
    assert gc.isenabled() == original


def test_full_deferral_keeps_young_cycles_collectible_and_restores_thresholds():
    import weakref
    enabled, thresholds = gc.isenabled(), gc.get_threshold()
    class Cycle:
        def __init__(self):
            self.link = self
    try:
        gc.enable()
        gc.collect()
        gc.set_threshold(40, 4, 7)
        with RestGarbageCollector(defer_full=True, interval=30.) as guard:
            assert gc.isenabled()
            assert gc.get_threshold()[:2] == (40, 4)
            assert gc.get_threshold()[2] == 2**31-1
            refs = [weakref.ref(Cycle()) for _ in range(1000)]
            assert sum(ref() is not None for ref in refs) < 1000
            assert any(e['generation'] in (0, 1) for e in guard.events)
            assert not any(e['generation'] == 2 for e in guard.events)
            assert not guard.full_collection_due(29.99)
            assert guard.full_collection_due(30.)
            assert guard.collect_at_rest(30., False) is None
        assert gc.get_threshold() == (40, 4, 7)
    finally:
        gc.set_threshold(*thresholds)
        gc.enable() if enabled else gc.disable()


def test_full_collection_holds_until_a_new_fresh_certified_capture_at_measured_rest(monkeypatch):
    import omni_drones.navigation.forest.runtime_gc as module
    enabled = gc.isenabled()
    clock, calls = [100.], []
    def collect(generation):
        calls.append(generation)
        clock[0] += .4
        return 12
    monkeypatch.setattr(gc, 'collect', collect)
    monkeypatch.setattr(module.time, 'perf_counter', lambda: clock[0])
    try:
        gc.enable()
        with RestGarbageCollector(defer_full=True, interval=30.) as guard:
            result = guard.collect_at_rest(30., True)
            assert result['gc_end_wall'] == pytest.approx(100.4)
            assert result['gc_generation'] == 2 and 'generation' not in result
            assert dict(generation=17, **result)['gc_generation'] == 2
            assert calls == [2] and guard.waiting_for_observation
            assert not guard.full_collection_due(90.)
            assert guard.collect_at_rest(90., True) is None
            base = dict(captured=100.5, age=.1, max_age=.3, certified=True, measured_rest=True)
            for override in (dict(captured=100.39), dict(captured=100.4), dict(age=.301),
                    dict(certified=False), dict(measured_rest=False)):
                assert not guard.release_after_observation(**(base | override))
                assert guard.waiting_for_observation
            assert guard.release_after_observation(**base)
            assert not guard.waiting_for_observation
            assert not guard.full_collection_due(59.99)
            assert guard.full_collection_due(60.)
    finally:
        gc.enable() if enabled else gc.disable()


@pytest.mark.parametrize('enabled', [False, True])
def test_full_deferral_exception_restores_enabled_thresholds_and_callbacks(enabled):
    previous, thresholds, callbacks = gc.isenabled(), gc.get_threshold(), list(gc.callbacks)
    try:
        gc.enable() if enabled else gc.disable()
        with pytest.raises(RuntimeError):
            with RestGarbageCollector(defer_full=True) as guard:
                assert gc.isenabled() == enabled
                if not enabled:
                    assert not guard.full_collection_due(90.)
                    assert guard.collect_at_rest(90., True) is None
                raise RuntimeError('fault')
        assert gc.isenabled() == enabled
        assert gc.get_threshold() == thresholds
        assert gc.callbacks == callbacks
    finally:
        gc.enable() if previous else gc.disable()


def test_force_interval_collects_at_any_rest_but_brakes_only_when_overdue():
    with RestGarbageCollector(defer_full=True, interval=30., force_interval=180.) as guard:
        assert not guard.full_collection_due(100.)
        assert guard.collect_at_rest(100., True) is not None
        guard.resume_after_wall = None
        assert not guard.full_collection_due(279.) and guard.full_collection_due(280.)
    with pytest.raises(ValueError):
        RestGarbageCollector(defer_full=True, interval=30., force_interval=10.)
