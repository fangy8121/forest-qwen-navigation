"""Terrain and persistent structure must remain causal and supported."""
from types import SimpleNamespace
import json

import numpy as np
import pytest
from scipy.spatial import cKDTree

from omni_drones.navigation.forest.observed_terrain import ObservedTerrain, query_terrain
from omni_drones.navigation.forest.online_structure import _Accumulator
from omni_drones.navigation.forest.row_map import RowMap
from omni_drones.navigation.forest.terrain_following import supported_triangles


def plane(x0=-3., x1=4., y0=-3., y1=4., spacing=.2):
    x, y = np.meshgrid(np.arange(x0, x1, spacing), np.arange(y0, y1, spacing))
    return np.c_[x.ravel(), y.ravel(), (.08*x+.04*y+2.).ravel()]


def test_observed_slope_is_interpolated_and_never_extrapolated():
    terrain = ObservedTerrain()
    terrain.add(plane(), 1., [0., 0., 4.5])
    z, valid = query_terrain(terrain.snapshot(), [[0., 0.], [1., 1.], [8., 0.]])
    assert valid.tolist() == [True, True, False]
    np.testing.assert_allclose(z[:2], [2., 2.12], atol=.07)
    assert np.isnan(z[2])


def test_missing_stale_future_and_sparse_support_are_unknown():
    terrain = ObservedTerrain()
    terrain.add(plane(), 10., [0., 0., 4.5])
    snapshot = terrain.snapshot()
    assert not query_terrain(snapshot, [[0., 0.]], now=20., max_age=5.)[1].any()
    assert not query_terrain(snapshot, [[0., 0.]], now=5.)[1].any()
    assert not query_terrain({}, [[0., 0.]])[1].any()
    sparse = dict(vertices=[[0., 0., 0.], [0., 10., 0.], [10., 0., 0.]], timestamps=[0., 0., 0.])
    assert not query_terrain(sparse, [[1., 1.]])[1].any()


def test_degenerate_or_nonfinite_triangle_is_rejected_individually():
    vertices = np.array([[0., 0., 0.], [1., 0., .1], [2., 0., .2],
                         [0., 1., .2], [1., 1e-12, .1]])
    transform = np.zeros((4, 3, 2))
    transform[-1] = np.nan
    tri = SimpleNamespace(simplices=np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 3]]),
                          transform=transform)
    assert supported_triangles(vertices, tri, 3., .7).tolist() == [False, True, False, False]


def test_nearly_collinear_ground_vertices_do_not_abort_supported_queries():
    vertices = np.array([[0., 0., 2.], [1., 0., 2.1], [2., 0., 2.2],
                         [0., 1e-13, 2.], [1., 1e-13, 2.1], [2., 1e-13, 2.2],
                         [0., 1., 2.2], [1., 1., 2.3], [2., 1., 2.4]])
    terrain = dict(vertices=vertices.tolist(), timestamps=[1.]*len(vertices))
    z, valid = query_terrain(terrain, [[.5, .5], [1.5, .5], [3., .5]])
    assert valid.tolist() == [True, True, False]
    np.testing.assert_allclose(z[:2], [2.15, 2.25], atol=1e-6)


def test_structure_worker_preserves_traceback_for_failed_frame(tmp_path, monkeypatch):
    import omni_drones.navigation.forest.online_structure as module
    def fail(frame):
        raise np.linalg.LinAlgError('singular fixture')
    monkeypatch.setattr(module, '_WORKER_ACCUMULATOR', SimpleNamespace(_add=fail, output=tmp_path))
    with pytest.raises(np.linalg.LinAlgError):
        module._accumulate_frame(dict(timestamp=np.array(12.5)))
    diagnostic = json.loads((tmp_path/'structure_error.json').read_text())
    assert diagnostic['timestamp'] == 12.5
    assert 'LinAlgError: singular fixture' in diagnostic['traceback']


def test_sensor_blind_annulus_requires_surrounding_planar_support():
    angle = np.linspace(0., 2*np.pi, 48, endpoint=False)
    xy = np.c_[2.*np.cos(angle), 2.*np.sin(angle)]
    vertices = np.c_[xy, 2.+xy@np.array([.08, .04])]
    terrain = dict(vertices=vertices.tolist(), timestamps=[1.]*len(vertices), max_edge=1.8)
    z, valid = query_terrain(terrain, [[0., 0.], [4., 0.]])
    assert valid.tolist() == [True, False]
    assert abs(z[0]-2.) < 1e-6
    # A cliff across the blind area cannot be treated as a smooth reference.
    vertices[vertices[:, 0] > 0., 2] += 2.
    terrain['vertices'] = vertices.tolist()
    assert not query_terrain(terrain, [[0., 0.]])[1].any()
    terrain['vertices'] = vertices[:18].tolist()
    terrain['timestamps'] = [1.]*18
    assert not query_terrain(terrain, [[0., 0.]])[1].any()


def test_local_bracketing_survives_movement_on_ground_bending_farther_away():
    angle = np.linspace(0., 2*np.pi, 16, endpoint=False)
    inner = np.c_[1.5*np.cos(angle), 1.5*np.sin(angle), np.full(16, 2.)]
    outer = np.c_[2.5*np.cos(angle), 2.5*np.sin(angle), np.full(16, 2.65)]
    terrain = dict(vertices=np.vstack([inner, outer]).tolist(), timestamps=[1.]*32)
    # The nearest32 cannot fit one plane across the bending outer ground, but
    # the nearer measured ring brackets both body positions without guessing.
    z, valid = query_terrain(terrain, [[0., 0.], [.1, 0.]])
    assert valid.all()
    np.testing.assert_allclose(z, 2., atol=1e-6)


def test_rolling_raw_window_retains_historical_terrain_with_original_time():
    terrain = ObservedTerrain(radius=8.)
    terrain.add(plane(), 1., [0., 0., 4.5])
    terrain.add(plane(47., 54.), 40., [50., 0., 8.5])
    snapshot = terrain.snapshot()
    assert query_terrain(snapshot, [[0., 0.], [50., 0.]])[1].all()
    assert not query_terrain(snapshot, [[0., 0.]], now=40., max_age=5.)[1].any()
    assert query_terrain(snapshot, [[50., 0.]], now=40., max_age=5.)[1].all()
    assert all(k[0]*.2 > 40. for k in terrain.points)


def test_terrain_snapshots_share_immutable_storage_until_an_actual_update():
    terrain = ObservedTerrain(update_seconds=1.)
    terrain.add(plane(), 1., [0., 0., 4.5])
    first = terrain.snapshot()
    second = terrain.snapshot()
    assert first is not second
    assert first['vertices'] is second['vertices']
    assert first['timestamps'] is second['timestamps']
    with pytest.raises(TypeError):
        first['vertices'][0][2] = 100.
    first['vertices'] = ()
    first['version'] = -1
    assert terrain.snapshot() == second
    terrain.add(plane(), 1.1, [0., 0., 4.5])
    assert terrain.snapshot()['vertices'] is second['vertices']
    terrain.add(plane(), 2., [0., 0., 4.5])
    latest = terrain.snapshot()
    assert latest['version'] == second['version']+1
    assert latest['vertices'] is not second['vertices']
    assert max(second['timestamps']) == 1.
    assert max(latest['timestamps']) == 2.
    assert query_terrain(second, [[0., 0.]])[1].all()
    assert query_terrain(latest, [[0., 0.]])[1].all()


def frame(now, position, hits=None):
    p = np.empty((0, 3)) if hits is None else np.asarray(hits)
    return dict(timestamp=np.array(now), position=np.asarray(position, float),
        points_sensor=p-np.asarray(position), valid=np.ones(len(p), bool),
        distances=np.linalg.norm(p-position, axis=1), sensor_position=np.asarray(position, float),
        sensor_quaternion=np.array([1., 0., 0., 0.]), max_range=np.array(15.),
        directions_world=(p-position)/np.maximum(np.linalg.norm(p-position, axis=1, keepdims=True), 1e-9))


def test_rows_survive_loss_of_recent_returns_and_extend_same_identity(monkeypatch):
    row = dict(direction=[1., 0.], normal=[0., 1.], offset_m=1.,
        spans=[[[0., 1.], [10., 1.]]], support=5, residual_m=.1)
    import omni_drones.navigation.forest.row_map as module
    monkeypatch.setattr(module, 'extract_stems', lambda points:([], {}))
    neighbor = dict(row, offset_m=6., spans=[[[0., 6.], [10., 6.]]])
    observed = [row, neighbor]
    monkeypatch.setattr(module, 'fit_rows', lambda stems, **kwargs:observed)
    memory = RowMap(update_seconds=1., local_radius=10.)
    memory.add(frame(0., [0., 0., 2.]))
    memory.add(frame(1., [0., 0., 2.]))
    observed = []
    memory.add(frame(2., [50., 0., 2.]))
    assert memory.snapshot()[0]['rows'][0]['id'] == 'R01'
    observed = [dict(row, spans=[[[8., 1.], [20., 1.]]]), neighbor]
    memory.add(frame(3., [15., 0., 2.]))
    rows = memory.snapshot()[0]['rows']
    assert len(rows) == 2 and rows[0]['id'] == 'R01'
    np.testing.assert_allclose(rows[0]['spans'], [[[0., 1.], [20., 1.]]])


def test_single_early_diagonal_cannot_poison_later_parallel_rows(monkeypatch):
    import omni_drones.navigation.forest.row_map as module
    monkeypatch.setattr(module, 'extract_stems', lambda points:([], {}))
    diagonal = dict(direction=[.6, -.8], normal=[.8, .6], offset_m=0.,
        spans=[[[0., 0.], [6., -8.]]], support=4, residual_m=.1)
    observed = [diagonal]
    monkeypatch.setattr(module, 'fit_rows', lambda stems, **kwargs:observed)
    memory = RowMap(update_seconds=1.)
    memory.add(frame(0., [0., 0., 2.]))
    assert not memory.tracks
    observed = [dict(direction=[1., 0.], normal=[0., 1.], offset_m=y,
        spans=[[[0., y], [12., y]]], support=5, residual_m=.1) for y in (0., 5.)]
    memory.add(frame(1., [1., 0., 2.]))
    assert not memory.tracks  # One coherent update remains tentative.
    memory.add(frame(2., [2., 0., 2.]))
    rows = memory.snapshot()[0]['rows']
    assert len(rows) == 2 and set(memory.tracks) == {'R01', 'R02'}
    assert all(np.dot(row['direction'], [1., 0.]) > .99 for row in rows)


def test_jittered_observed_first_stem_extends_extent_without_tilting_fit(monkeypatch):
    import omni_drones.navigation.forest.row_map as module
    # These coordinates come from the full02 observed stem/row snapshot.
    stem = dict(xy=[24.1836, 7.7784], radius_m=.3606, support_columns=7)
    rows = [dict(direction=[1., 0.], normal=[0., 1.], offset_m=y,
        spans=[[[26.66, y], [35.68, y]]], support=4, residual_m=.2,
        support_xy=[[x, y] for x in (26.66, 29., 32., 35.68)]) for y in (3.62, 8.88)]
    monkeypatch.setattr(module, 'extract_stems', lambda points:([stem], {}))
    monkeypatch.setattr(module, 'fit_rows', lambda stems, **kwargs:rows)
    memory = RowMap(update_seconds=1.)
    memory.add(frame(0., [0., 0., 2.]))
    memory.add(frame(1., [0., 0., 2.]))
    row = memory.tracks['R02']
    assert abs(row['spans'][0][0][0]-24.1836) < 1e-6
    assert row['offset_m'] == 8.88 and row['direction'] == [1., 0.]
    assert stem['xy'] in row['extent_support_xy']


def test_union_tail_is_required_but_not_claimed_as_bilateral(tmp_path):
    accumulator = _Accumulator(tmp_path)
    rows = [dict(id='R01', direction=[1., 0.], normal=[0., 1.], offset_m=0., spans=[[[0., 0.], [10., 0.]]]),
            dict(id='R02', direction=[1., 0.], normal=[0., 1.], offset_m=5., spans=[[[2., 5.], [14., 5.]]])]
    accumulator.row_map = SimpleNamespace(add=lambda frame:None, voxels=set(), origin=np.zeros(3),
        snapshot=lambda:(dict(rows=rows), None))
    result = accumulator._add(frame(0., [0., 2.5, 2.]))
    assert result['corridors'][0]['span'] == [0., 14.]
    assert result['corridors'][0]['bilateral_span'] == [2., 10.]
    assert result['corridors'][0]['inspection_half_width_m'] == 1.75
    assert not result['corridors'][0]['ends']['low']['confirmed']


def test_no_return_cannot_confirm_empty_space_or_boundary(tmp_path):
    accumulator = _Accumulator(tmp_path)
    terrain = ObservedTerrain()
    terrain.add(plane(), 0., [0., 0., 4.5])
    accumulator.views[(0, 0, 0)] = dict(position=[0., 0., 4.5])
    accumulator.views[(1, 0, 0)] = dict(position=[2., 0., 4.5])
    evidence = accumulator._evidence(np.array([[0., 0.], [1., 0.]]),
        cKDTree(np.empty((0, 3))), frame(1., [0., 0., 4.5]), np.zeros(2), 'scan', terrain.snapshot())
    assert evidence['terrain_supported_fraction'] == 1.
    assert evidence['observed_fraction'] == 0. and not evidence['confirmed']


def test_tentative_stems_are_exported_in_world_frame_before_row_confirmation(tmp_path):
    accumulator = _Accumulator(tmp_path)
    stem = dict(xy=[1., 2.], radius_m=.3, support_columns=4, confidence='candidate')
    accumulator.row_map = SimpleNamespace(add=lambda frame:None, voxels=set(), origin=np.array([10., 20., 2.]),
        snapshot=lambda:(dict(rows=[], stems=[stem], height_band_xy=[[1., 2.], [1.01, 2.01]]), None))
    result = accumulator._add(frame(0., [10., 20., 2.]))
    assert result['stem_candidates'] == [dict(stem, xy=[11., 22.])]
    assert result['rows'] == []
    np.testing.assert_allclose(result['structure_points_xy'], [[11.2, 22.]])


def test_positive_stem_vetoes_false_end_even_with_full_sampled_free_fraction(tmp_path, monkeypatch):
    import omni_drones.navigation.forest.online_structure as module
    accumulator = _Accumulator(tmp_path)
    rows = [dict(id=rid, direction=[1., 0.], normal=[0., 1.], offset_m=y,
        spans=[[[26.66, y], [35.68, y]]]) for rid, y in [('R01', 3.62), ('R02', 8.88)]]
    stem = dict(xy=[24.1836, 7.7784], radius_m=.3606, support_columns=7)
    accumulator.row_map = SimpleNamespace(add=lambda frame:None, voxels=set(), origin=np.zeros(3),
        snapshot=lambda:(dict(rows=rows, stems=[stem]), None))
    accumulator.terrain = SimpleNamespace(add=lambda *args:None, snapshot=lambda:dict(vertices=()))
    monkeypatch.setattr(module, 'query_terrain', lambda terrain, xy:(np.zeros(len(xy)), np.ones(len(xy), bool)))
    probes = np.array([[26.66-s, off+w, z] for off in (3.62, 8.88)
        for s in np.arange(1., 5.51, .8) for w in (-.4, 0., .4) for z in (.8, 1.6)])
    accumulator.free.update(map(tuple, np.rint(probes/.4).astype(int).tolist()))
    accumulator.views[(1, 0, 0)] = dict(position=[24., 6., 2.5])
    accumulator.views[(2, 0, 0)] = dict(position=[26., 6., 2.5])
    value = accumulator._add(frame(1., [25., 6., 2.5]))
    end = value['corridors'][0]['ends']['low']
    assert end['observed_fraction'] == 1. and end['diverse_views']
    assert not end['confirmed']
    assert end['positive_stem_counterexamples'][0]['xy'] == stem['xy']
    assert end['positive_stem_counterexamples'][0]['reported_at_scan_id'] == 'scan_00001'


def test_wide_missing_row_gap_is_an_unresolved_frontier_not_identity_failure(tmp_path):
    accumulator = _Accumulator(tmp_path)
    rows = [dict(id=rid, direction=[1., 0.], normal=[0., 1.], offset_m=y,
        spans=[[[20., y], [40., y]]]) for rid, y in [('R01', 3.8), ('R03', 14.4), ('R04', 19.5)]]
    accumulator.row_map = SimpleNamespace(add=lambda frame:None, voxels=set(), origin=np.zeros(3),
        snapshot=lambda:(dict(rows=rows), None))
    first = accumulator._add(frame(0., [25., 8., 2.]))
    assert not first['identity_ambiguous']
    assert len(first['corridors']) == 1
    assert first['corridors'][0]['bordering_rows'] == ['R03', 'R04']
    assert len(first['unresolved_row_gaps']) == 1
    assert first['unresolved_row_gaps'][0]['bordering_rows'] == ['R01', 'R03']
    assert first['unresolved_row_gaps'][0]['status'] == 'UNRESOLVED_MISSING_ROW_OR_OPEN_SPACE'
    rows.insert(1, dict(id='R02', direction=[1., 0.], normal=[0., 1.], offset_m=8.7,
        spans=[[[20., 8.7], [40., 8.7]]]))
    second = accumulator._add(frame(1., [26., 8., 2.]))
    assert not second['identity_ambiguous']
    assert second['unresolved_row_gaps'] == [] and len(second['corridors']) == 3


def test_public_work_area_excludes_outside_probes_without_free_space_credit(tmp_path, monkeypatch):
    import omni_drones.navigation.forest.online_structure as module
    (tmp_path/'bounded').mkdir()
    (tmp_path/'legacy').mkdir()
    bounded = _Accumulator(tmp_path/'bounded', work_area_bounds=[0., 2., -1., 1.])
    legacy = _Accumulator(tmp_path/'legacy', work_area_bounds=None)
    for a in (bounded, legacy):
        a.views = {(0, 0):dict(position=[0., 0., 2.]), (1, 0):dict(position=[1., 0., 2.])}
    def ground(terrain, xy):
        return np.zeros(len(xy)), np.ones(len(xy), bool)
    monkeypatch.setattr(module, 'query_terrain', ground)
    probes = np.array([[.5, 0.], [1.5, 0.], [2.5, 0.], [3.5, 0.]])
    free = cKDTree(np.array([[x, 0., z] for x in (.5, 1.5) for z in (.8, 1.6)]))
    args = (probes, free, frame(1., [0., 0., 2.]), np.array([1., 0.]), 'scan', {})
    clipped, original = bounded._evidence(*args), legacy._evidence(*args)
    assert clipped['confirmed'] and clipped['observed_fraction'] == 1.
    assert clipped['required_probe_count'] == 2 and clipped['original_probe_count'] == 4
    assert clipped['outside_public_work_area'] == dict(classification='OUTSIDE_PUBLIC_WORK_AREA',
        count=2, xy=[[2.5, 0.], [3.5, 0.]], free_space_credit=0)
    assert clipped['public_work_area']['coverage_scope'] == 'BOUND_PUBLIC_WORK_AREA'
    assert clipped['proof_kind'] == 'OBSERVED_ABSENCE_TO_PUBLIC_WORK_AREA_BOUNDARY'
    assert not original['confirmed'] and original['observed_fraction'] == .5
    assert original['outside_public_work_area'] is None and original['public_work_area'] is None
    # Lost evidence inside the job remains a failure even when all excluded
    # positions are advertised as out of scope.
    missing = cKDTree(np.array([[.5, 0., .8], [.5, 0., 1.6]]))
    partial = bounded._evidence(probes, missing, args[2], args[3], 'scan', {})
    assert partial['observed_fraction'] == .5 and not partial['confirmed']
    empty = bounded._evidence(probes[2:], free, args[2], args[3], 'scan', {})
    assert not empty['confirmed'] and empty['observed_fraction'] == 0.
    assert empty['unresolved_reason'] == 'NO_IN_DOMAIN_PROBES'
    monkeypatch.setattr(module, 'query_terrain', lambda terrain, xy:(np.zeros(len(xy)), xy[:, 0] < 1.))
    unsupported = bounded._evidence(*args)
    assert unsupported['terrain_supported_fraction'] == .5 and not unsupported['confirmed']


def test_public_work_area_keeps_outside_stems_visible_and_bounds_public(tmp_path):
    accumulator = _Accumulator(tmp_path, work_area_bounds=[0., 2., -1., 1.])
    stem = dict(xy=[3., 0.], radius_m=.3, support_columns=4)
    accumulator.row_map = SimpleNamespace(add=lambda frame:None, voxels=set(), origin=np.zeros(3),
        snapshot=lambda:(dict(rows=[], stems=[stem]), None))
    result = accumulator._add(frame(0., [1., 0., 2.]))
    assert result['stem_candidates'] == result['outside_public_work_area_stem_candidates'] == [stem]
    assert result['coverage_scope'] == 'BOUND_PUBLIC_WORK_AREA'
    assert result['public_work_area'] == dict(xy_bounds=[0., 2., -1., 1.],
        coverage_scope='BOUND_PUBLIC_WORK_AREA', source='operator supplied XY task rectangle')


@pytest.mark.parametrize('bounds', [[0., 0., 0., 1.], [1., 0., 0., 1.], [0., 1., float('nan'), 1.], [0., 1.]])
def test_public_work_area_requires_finite_ordered_xy_rectangle(tmp_path, bounds):
    with pytest.raises(ValueError, match='work_area_bounds'):
        _Accumulator(tmp_path, work_area_bounds=bounds)


def test_public_work_area_is_visible_before_first_scan(tmp_path):
    from omni_drones.navigation.forest.online_structure import OnlineStructure
    structure = OnlineStructure(tmp_path, execution='inline', work_area_bounds=[0., 2., -1., 1.])
    assert structure.snapshot()['public_work_area']['xy_bounds'] == [0., 2., -1., 1.]
    assert structure.snapshot()['coverage_scope'] == 'BOUND_PUBLIC_WORK_AREA'
    structure.close()


def test_precomputed_terrain_index_matches_the_rebuilt_query_and_stays_out_of_logs():
    # Run c49e9da0: rebuilding arrays and triangulation on the control thread for
    # each guidance lift grew to ~0.06 s per goal activation.
    terrain = ObservedTerrain()
    ground = plane()
    ground[::7, 2] += .05   # uneven enough that triangle choice matters
    terrain.add(ground, 1., [0., 0., 4.5])
    snapshot = terrain.snapshot()
    assert '_index' in snapshot
    plain = {k:v for k, v in snapshot.items() if k != '_index'}
    xy = np.random.default_rng(3).uniform(-4., 5., (200, 2))
    for now, max_age in ((None, None), (1.5, 5.)):
        indexed, rebuilt = query_terrain(snapshot, xy, now, max_age), query_terrain(plain, xy, now, max_age)
        np.testing.assert_array_equal(indexed[1], rebuilt[1])
        np.testing.assert_array_equal(indexed[0], rebuilt[0])
    # The worker's per-frame JSON event keeps scalar metadata only.
    event = {k:v for k, v in snapshot.items() if k not in ('vertices', 'timestamps') and not k.startswith('_')}
    json.dumps(event)


def test_known_terrain_supports_row_end_probes_without_lidar_ground(tmp_path):
    xs = ys = np.arange(-5., 6.)
    truth = dict(xs=xs, ys=ys, heights=np.full((len(xs), len(ys)), 2.))
    accumulator = _Accumulator(tmp_path, terrain_truth=truth)
    accumulator.views[(0, 0, 0)] = dict(position=[0., 0., 4.5])
    accumulator.views[(1, 0, 0)] = dict(position=[2., 0., 4.5])
    probes = np.array([[0., 0.], [1., 0.]])
    free = cKDTree(np.c_[np.r_[probes, probes], np.r_[2.8, 2.8, 3.6, 3.6]])
    evidence = accumulator._evidence(probes, free, frame(1., [0., 0., 4.5]), np.zeros(2), 'scan', {})
    assert evidence['terrain_supported_fraction'] == 1. and evidence['confirmed']


def _row(rid, y, x0, x1, support):
    xs = np.linspace(x0, x1, support)
    return dict(id=rid, direction=[1., 0.], normal=[0., 1.], offset_m=y, support=support,
                spans=[[[x0, y], [x1, y]]], support_xy=[[x, y] for x in xs])


def _structure(tmp_path, rows):
    accumulator = _Accumulator(tmp_path)
    accumulator.row_map = SimpleNamespace(add=lambda frame:None, voxels=set(), origin=np.zeros(3),
        snapshot=lambda:(dict(rows=rows), None))
    return accumulator._add(frame(0., [50., 25., 4.]))


@pytest.mark.parametrize('later_support', [4, 30])
def test_split_track_of_one_row_is_merged_under_the_older_id(tmp_path, later_support):
    # Run 8e03d728: a sparse row became R06 and R07 0.2 m apart; the structure
    # stayed ambiguous while the vehicle hovered and the run failed.
    rows = [_row('R03', 22.92, 24.5, 90.4, 60), _row('R06', 27.66, 23.5, 86.8, 12),
            _row('R07', 27.88, 71.3, 89.5, later_support)]
    result = _structure(tmp_path, rows)
    assert not result['identity_ambiguous']
    assert [r['id'] for r in result['rows']] == ['R03', 'R06'] and result['merged_rows'] == [['R06', 'R07']]
    assert [c['id'] for c in result['corridors']] == ['R03_R06']
    merged = result['rows'][1]
    assert merged['support'] == 12+later_support
    assert 27.66 <= merged['offset_m'] <= 27.88
    assert max(np.asarray(merged['spans']).reshape(-1, 2)[:, 0]) == pytest.approx(89.5)


def test_a_flipped_row_family_resets_identities_instead_of_staying_ambiguous(tmp_path):
    # Run 5d434c21: the first fit took tree columns (y direction) for rows; when
    # the true x-direction family appeared, the vanished corridor kept the
    # structure ambiguous for the rest of the run.
    columns = [dict(id=rid, direction=[0., 1.], normal=[-1., 0.], offset_m=-x, support=8,
                    spans=[[[x, 0.], [x, 12.]]], support_xy=[[x, y] for y in np.linspace(0., 12., 8)])
               for rid, x in [('R01', 20.), ('R02', 26.)]]
    rows = [_row('R03', 3., 20., 40., 10), _row('R04', 8., 20., 40., 10), _row('R05', 13., 20., 40., 10)]
    current = dict(rows=columns)
    accumulator = _Accumulator(tmp_path)
    accumulator.row_map = SimpleNamespace(add=lambda frame:None, voxels=set(), origin=np.zeros(3),
        snapshot=lambda:(dict(rows=current['rows']), None))
    first = accumulator._add(frame(0., [22., 5., 4.]))
    assert [c['id'] for c in first['corridors']] == ['R02_R01'] and not first['family_reset']
    current['rows'] = rows
    second = accumulator._add(frame(1., [22., 5., 4.]))
    assert second['family_reset'] and second['family_resets'] == 1
    assert not second['identity_ambiguous'] and second['missing_previously_observed_corridor_ids'] == []
    assert [c['id'] for c in second['corridors']] == ['R03_R04', 'R04_R05']
    # A later vanished corridor of the same family still makes it ambiguous.
    current['rows'] = rows[:2]
    third = accumulator._add(frame(2., [22., 5., 4.]))
    assert not third['family_reset'] and third['identity_ambiguous']
    assert third['missing_previously_observed_corridor_ids'] == ['R04_R05']


@pytest.mark.parametrize('returns', [False, True])
def test_a_vanished_corridor_is_ambiguous_only_for_the_settle_window(tmp_path, returns):
    # val/000 run ef65ee46: an aisle seen 6.5-9 s kept the structure ambiguous
    # (only HOLD offered) until observation_no_progress at 43 s.
    rows = [_row('R03', 3., 20., 40., 10), _row('R04', 8., 20., 40., 10), _row('R05', 13., 20., 40., 10)]
    current = dict(rows=rows)
    accumulator = _Accumulator(tmp_path)
    accumulator.row_map = SimpleNamespace(add=lambda frame:None, voxels=set(), origin=np.zeros(3),
        snapshot=lambda:(dict(rows=current['rows']), None))
    assert not accumulator._add(frame(0., [22., 5., 4.]))['identity_ambiguous']
    current['rows'] = rows[:2]
    for now in (1., 5., 10.9):
        held = accumulator._add(frame(now, [22., 5., 4.]))
        assert held['identity_ambiguous'] and held['missing_previously_observed_corridor_ids'] == ['R04_R05']
    if returns:
        current['rows'] = rows
        back = accumulator._add(frame(10.95, [22., 5., 4.]))
        assert not back['identity_ambiguous']
        current['rows'] = rows[:2]
        # Missing again: a new window, not the old one.
        assert accumulator._add(frame(11.5, [22., 5., 4.]))['identity_ambiguous']
        assert accumulator._add(frame(20.9, [22., 5., 4.]))['identity_ambiguous']
        settled = accumulator._add(frame(21.5, [22., 5., 4.]))
    else:
        settled = accumulator._add(frame(11., [22., 5., 4.]))
    assert not settled['identity_ambiguous'] and settled['forgotten_corridor_ids'] == ['R04_R05']
    assert settled['missing_previously_observed_corridor_ids'] == []
    assert [c['id'] for c in settled['corridors']] == ['R03_R04']
    # It is forgotten once; a later return is a newly observed corridor.
    current['rows'] = rows
    again = accumulator._add(frame(12. if not returns else 22., [22., 5., 4.]))
    assert not again['identity_ambiguous'] and again['forgotten_corridor_ids'] == []


def test_parallel_tracks_beyond_the_merge_band_but_under_row_spacing_stay_ambiguous(tmp_path):
    rows = [_row('R01', 20., 20., 80., 30), _row('R02', 21.8, 20., 80., 30)]
    result = _structure(tmp_path, rows)
    assert result['identity_ambiguous'] and result['merged_rows'] == []
    assert [r['id'] for r in result['rows']] == ['R01', 'R02']


def test_scattered_off_family_track_is_rejected_not_ambiguous(tmp_path):
    accumulator = _Accumulator(tmp_path)
    rows = [dict(id='R01', direction=[1., 0.], normal=[0., 1.], offset_m=0., support=12,
                 spans=[[[0., 0.], [20., 0.]]], support_xy=[[x, 0.] for x in range(0, 21, 2)]),
            dict(id='R02', direction=[1., 0.], normal=[0., 1.], offset_m=5., support=12,
                 spans=[[[0., 5.], [20., 5.]]], support_xy=[[x, 5.] for x in range(0, 21, 2)]),
            # Diagonal stale track: 4 stems spread 15 m across the aisles.
            dict(id='R03', direction=[.5, .866], normal=[-.866, .5], offset_m=2., support=4,
                 spans=[[[8., 1.], [16., 15.]]], support_xy=[[8., 1.], [9., 3.], [12., 8.], [16., 15.]])]
    accumulator.row_map = SimpleNamespace(add=lambda frame:None, voxels=set(), origin=np.zeros(3),
        snapshot=lambda:(dict(rows=rows), None))
    result = accumulator._add(frame(0., [0., 2.5, 2.]))
    assert [r['id'] for r in result['rejected_rows']] == ['R03']
    assert not result['identity_ambiguous'] and [c['id'] for c in result['corridors']] == ['R01_R02']
    np.testing.assert_allclose(result['rows'][0]['direction'], [1., 0.])


def test_one_stray_stem_does_not_remove_a_well_supported_row(tmp_path):
    # Run c1c3147f: R02 (114 stems, 1.9 m lateral span) gained one stem 0.6 m
    # outside it and was rejected for good; its two aisles vanished and the
    # resulting R01-R03 gap blocked completion until coverage_stalled.
    accumulator = _Accumulator(tmp_path)
    x = np.arange(0., 60., .5)
    rows = [dict(id=rid, direction=[1., 0.], normal=[0., 1.], offset_m=y, support=len(x),
                 spans=[[[0., y], [59.5, y]]],
                 support_xy=np.c_[x, y+.6*np.sin(x)].tolist()+([[30.25, y+2.]] if rid == 'R02' else []))
            for rid, y in [('R01', 0.), ('R02', 5.), ('R03', 10.)]]
    assert np.ptp(np.asarray(rows[1]['support_xy'])[:, 1]) > 2.5
    accumulator.row_map = SimpleNamespace(add=lambda frame:None, voxels=set(), origin=np.zeros(3),
        snapshot=lambda:(dict(rows=rows), None))
    result = accumulator._add(frame(0., [30., 2.5, 2.]))
    assert result['rejected_rows'] == [] and result['unresolved_row_gaps'] == []
    assert [c['id'] for c in result['corridors']] == ['R01_R02', 'R02_R03']
