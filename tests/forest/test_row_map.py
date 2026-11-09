"""Measured rows must survive slopes/rotation and never invent observed ends."""
import numpy as np
import pytest
from omni_drones.navigation.forest.row_map import extract_stems, fit_rows, RowMap


def stems_xy(rotation=0.):
    xy = np.array([[x, y] for y in (0., 5., 10.) for x in (0., 4., 8., 12., 30., 34., 38., 42.)])
    c, s = np.cos(rotation), np.sin(rotation)
    xy = xy @ np.array([[c,s],[-s,c]])
    return [dict(xy=p.tolist()) for p in xy]


@pytest.mark.parametrize('angle', [0., .55, 1.7])
def test_rotated_rows_and_unobserved_gaps(angle):
    rows = fit_rows(stems_xy(angle))
    assert len(rows) == 3
    assert all(len(r['spans']) == 2 for r in rows)
    assert all(r['endpoints'] == 'observed_support_limits_not_confirmed_row_ends' for r in rows)
    assert all(abs(np.dot(r['direction'], [np.cos(angle),np.sin(angle)])) > .999 for r in rows)


@pytest.mark.parametrize('angle', [0., .6])
def test_short_partial_grid_cannot_promote_cross_column_axis(angle):
    rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    partial = np.array([[x, y] for y in (0., 5., 10., 15.) for x in (0., 2., 4., 6.)])@rotation
    assert fit_rows([dict(xy=p.tolist()) for p in partial]) == []
    extended = np.array([[x, y] for y in (0., 5., 10., 15.) for x in (0., 2., 4., 6., 8., 10.)])@rotation
    rows = fit_rows([dict(xy=p.tolist()) for p in extended])
    assert len(rows) == 4
    assert all(abs(np.dot(r['direction'], [np.cos(angle), np.sin(angle)])) > .999 for r in rows)


@pytest.mark.parametrize('angle', [0., .65])
def test_sparse_observed_prefix_vetoes_long_diagonal_column_alias(angle):
    # Full05 measured stem prefix: long cross-columns are tempting, while the
    # actual rows are still too short for permanent promotion.
    prefix = np.array([[21.58, 23.38], [22.78, 18.38], [24.08, 7.78], [25.88, 13.98],
        [26.08, 19.08], [26.58, 8.68], [27.38, 3.48], [27.58, 19.08], [27.98, 13.98],
        [28.78, 3.58], [28.98, 19.18], [29.28, 8.38], [30.58, 14.28], [31.68, 8.58], [33.38, 3.58]])
    rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    assert fit_rows([dict(xy=p.tolist()) for p in prefix@rotation]) == []
    # A synthetic continuation provides sufficient along-row support without
    # changing the same spacing/density and permanent-promotion requirements.
    extended = np.vstack([prefix, [[35.68, 3.08], [35.08, 8.78], [34.68, 14.], [35.48, 19.]]])@rotation
    rows = fit_rows([dict(xy=p.tolist()) for p in extended])
    assert len(rows) == 4
    assert all(abs(np.dot(r['direction'], [np.cos(angle), np.sin(angle)])) > .98 for r in rows)


def test_explained_support_wins_over_dropping_a_jittered_row_end():
    xy = [[27.38, 3.48], [28.78, 3.58], [33.38, 3.58], [35.68, 2.98],
          [24.18, 7.78], [26.68, 8.68], [29.38, 8.38], [31.78, 8.78], [35.08, 8.78],
          [25.88, 13.98], [28.08, 13.98], [30.58, 14.28], [22.78, 18.38], [25.98, 18.98]]
    rows = fit_rows([dict(xy=p) for p in xy])
    assert len(rows) == 2
    assert all(abs(r['direction'][0]) > .98 for r in rows)


def test_short_off_row_cluster_cannot_suppress_strong_neighbor_rows():
    xy = [[x, y] for y in np.arange(0., 31., 5.) for x in np.arange(0., 41., 2.5)]
    xy += [[x, 12.4] for x in (0., 2., 4.)]
    rows = fit_rows([dict(xy=p) for p in xy])
    assert len(rows) == 7
    assert all(abs(r['direction'][0]) > .999 for r in rows)


def test_sloping_ground_and_horizontal_branches_do_not_become_stems():
    x,y = np.meshgrid(np.arange(-4.,16.,.4),np.arange(-4.,10.,.4))
    z = .04*x+.03*y
    ground = np.c_[x.ravel(),y.ravel(),z.ravel()]
    points = [ground]
    centers = [(0.,0.),(5.,0.),(10.,0.),(0.,5.),(5.,5.),(10.,5.)]
    for cx,cy in centers:
        a,h = np.meshgrid(np.linspace(0,2*np.pi,20,endpoint=False),np.arange(.3,3.,.15))
        points.append(np.c_[(cx+.3*np.cos(a)).ravel(), (cy+.3*np.sin(a)).ravel(),
                            (h+.04*cx+.03*cy).ravel()])
    # A low horizontal branch lies inside the selected height band but has no
    # multi-height support at its XY position.
    branch = np.c_[np.arange(1.,9.,.1),np.full(80,2.5),np.full(80,1.5)]
    points.append(branch)
    detected, _ = extract_stems(np.vstack(points))
    assert len(detected) == len(centers)
    for center in centers:
        assert min(np.linalg.norm(np.array(s['xy'])-center) for s in detected) < .3
    assert not extract_stems(ground)[0]


def test_row_map_prefix_is_immutable_and_ignores_teacher_fields():
    def frame(t, poison=False):
        return dict(timestamp=t, position=np.array([t,0.,2.]), points_sensor=np.array([[1.,0.,0.]]),
            valid=np.array([True]), sensor_position=np.array([t,0.,2.]),
            sensor_quaternion=np.array([1.,0.,0.,0.]),
            phase='TURN' if poison else 'INSPECT', task_goal=np.array([999.,999.,999.]))
    a,b = RowMap(),RowMap()
    a.add(frame(0.)); b.add(frame(0.,True))
    saved = a.snapshot()
    assert saved[0] == b.snapshot()[0]
    a.add(frame(10.))
    assert saved[0]['timestamp_s'] == 0.
    np.testing.assert_array_equal(saved[1], [[0.,0.,0.]])
    with pytest.raises(ValueError): a.add(frame(1.))


def test_sampling_cannot_change_estimator_state():
    a,b = RowMap(),RowMap()
    for i in range(11):
        f = dict(timestamp=float(i),position=np.array([float(i),0.,2.]),
            points_sensor=np.array([[1.,0.,0.]]),valid=np.array([True]),
            sensor_position=np.array([float(i),0.,2.]),sensor_quaternion=np.array([1.,0.,0.,0.]))
        a.add(f); b.add(f); a.snapshot()
    assert a.snapshot()[0] == b.snapshot()[0]
