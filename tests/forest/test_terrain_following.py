import json
import numpy as np
import pytest
from omni_drones.navigation.forest.terrain_following import (
    GroundSurface, HeightReference, lift_path, parse_xy, audit_path)


def plane(slope=.1, hole=False, time=10.):
    x, y = np.meshgrid(np.arange(-3, 7, .08), np.arange(-3, 3, .08))
    p = np.c_[x.ravel(), y.ravel(), -2.5+slope*x.ravel()]
    if hole:
        p = p[np.linalg.norm(p[:, :2]-[3, 0], axis=1) > 1.8]
    return GroundSurface(p, time, 10.)


def test_sloped_surface_and_turning_xy_lift():
    g = plane()
    q = g.query([[0, 0], [1, 0], [2, 1]])
    assert q.valid.all()
    assert np.allclose(q.z, [-2.5, -2.4, -2.3], atol=.015)
    ref = HeightReference(g)
    assert ref.mode == 'terrain_following'
    path = lift_path([[2, 0], [2, 1], [1, 1]], ref)
    assert np.allclose(path[:, 2], .1*path[:, 0], atol=.015)
    assert np.array_equal(path[0], [0, 0, 0])


def test_holes_edges_stale_and_future_ground_stay_unknown():
    assert not plane(hole=True).query([[3, 0]]).valid[0]
    assert not plane().query([[10, 0]]).valid[0]
    assert not plane(time=-100.).query([[0, 0]]).valid[0]
    assert not plane(time=11.).query([[0, 0]]).valid[0]
    with pytest.raises(ValueError, match='ground support'):
        lift_path([[3, 0]], HeightReference(plane(hole=True)))


def test_missing_or_steep_ground_uses_explicit_constant_height_fallback():
    for g in [GroundSurface([], [], 10.), plane(slope=1.)]:
        ref = HeightReference(g)
        assert ref.mode == 'hold_current_altitude_ground_unreliable'
        assert (lift_path([[2, 1]], ref)[:, 2] == 0).all()


def test_overhead_sheet_is_not_an_acceptable_initial_ground_anchor():
    x, y = np.meshgrid(np.arange(-2, 3, .1), np.arange(-2, 3, .1))
    g = GroundSurface(np.c_[x.ravel(), y.ravel(), np.ones(x.size)], 10., 10.)
    assert HeightReference(g).mode == 'hold_current_altitude_ground_unreliable'


def test_full_connections_checked_after_height_lift():
    class Safety:
        def clear(self, p):
            return not ((p[:, 0] > .9) & (p[:, 0] < 1.1)).any()
        def known(self, p, origin):
            return True
    result, _ = audit_path([[2, 0]], HeightReference(plane()), Safety(), [0, 0, 0], dynamics=False)
    assert not result['accepted'] and not result['obstacle_clear']


@pytest.mark.parametrize('value', [
    {'positions_local_xy_m': [[True, 0]]*6},
    {'positions_local_xy_m': [[float('nan'), 0]]*6},
    {'positions_local_xy_m': [[0, 0]]*6},
    {'positions_local_xy_m': [[1, 0, 0]]*6},
])
def test_bad_coordinates_rejected(value):
    with pytest.raises(ValueError):
        parse_xy(json.dumps(value))


def test_hold_and_non_monotonic_coordinates_allowed():
    assert parse_xy('{"status":"hold","positions_local_xy_m":[]}').shape == (0, 2)
    assert parse_xy(json.dumps({'positions_local_xy_m': [[1, 0], [2, 0], [2, 1], [1, 2], [0, 2], [-1, 2]]})).shape == (6, 2)


def test_sloped_turn_is_reconstructed_and_can_brake_in_observed_space():
    from types import SimpleNamespace
    from omni_drones.navigation.forest.planning import PathSafety
    ground = plane()
    x, y, z = np.meshgrid(np.arange(-2, 7, .2), np.arange(-3, 3, .2), np.arange(-1.4, 1.61, .2))
    evidence = np.c_[x.ravel(), y.ravel(), z.ravel()]
    flight = SimpleNamespace(max_speed=1., max_acceleration=1., max_jerk=4., max_yaw_rate=1.,
        max_yaw_acceleration=2., corner_distance=1.5, max_path_deviation=.3, safety_profile='reference_sphere_v1')
    safety = PathSafety(ground.vertices, evidence, None, flight)
    original_guard = safety.guard
    checked_times = []
    def guard_at_rollout_pose(curve, t, actual, **kwargs):
        assert np.allclose(actual, curve.sample(t)[0])
        checked_times.append(t)
        return original_guard(curve, t, actual, **kwargs)
    safety.guard = guard_at_rollout_pose
    result, curve = audit_path([[1, 0], [2, .3], [2.5, .8], [2.5, 1.4]],
                               HeightReference(ground), safety, [0, 0, 0])
    assert result['accepted'] and result['braking_safe']
    assert result['height_error_m'] < .02
    assert result['max_vz_mps'] <= .6 and result['max_az_mps2'] <= .6
    assert np.linalg.norm(curve.sample(curve.duration)[1]) < 1e-6
    assert max(checked_times) == pytest.approx(curve.duration)
