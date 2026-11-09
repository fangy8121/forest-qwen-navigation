"""Moving map keeps world evidence, forgets evicted cells and stays bounded."""
import numpy as np
import pytest

from omni_drones.navigation.forest.geometry import GRID_RESOLUTION
from omni_drones.navigation.forest.local import LocalMap


def scan(now, position=(0., 0., 0.), directions=None, distances=None, valid=None):
    directions = (np.array([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.],
                            [0., -1., 0.], [0., 0., 1.], [0., 0., -1.]])
                  if directions is None else np.asarray(directions, dtype=float))
    distances = np.full(len(directions), 1.4) if distances is None else np.asarray(distances, dtype=float)
    valid = np.ones(len(directions), bool) if valid is None else np.asarray(valid, dtype=bool)
    return dict(timestamp=float(now), sensor_quaternion=np.array([1., 0., 0., 0.]),
                sensor_position=np.asarray(position, dtype=float),
                points_sensor=directions*distances[:, None], valid=valid,
                directions_world=directions, distances=distances, max_range=15.)


def stamps(memory, array):
    cells = np.argwhere(np.isfinite(array))
    world_cells = np.rint(memory.origin/GRID_RESOLUTION).astype(np.int64)+cells
    return {tuple(world): float(array[tuple(local)]) for world, local in zip(world_cells, cells)}


def world_cells(points):
    return set(map(tuple, np.rint(np.asarray(points)/GRID_RESOLUTION).astype(np.int64)))


@pytest.mark.parametrize('traversal', ['sampled_v1', 'voxel_dda_v1'])
@pytest.mark.parametrize('shift', [(0.6, -.4, .2), (-.4, .6, -.2)])
def test_recenter_copies_only_overlap_without_inventing_observations(traversal, shift):
    memory = LocalMap([[0., 0., 0.], [80., 80., 20.]], rolling=True,
                      window_size=(6., 6., 4.), ray_traversal=traversal)
    memory.add(scan(.2))
    before_seen, before_occupied = stamps(memory, memory.seen), stamps(memory, memory.occupied)
    assert memory.recenter(shift)
    origin = np.rint(memory.origin/GRID_RESOLUTION).astype(np.int64)
    shape = np.asarray(memory.seen.shape)

    def retained(values):
        return {key: value for key, value in values.items()
                if ((np.asarray(key)-origin >= 0) & (np.asarray(key)-origin < shape)).all()}

    assert stamps(memory, memory.seen) == retained(before_seen)
    assert stamps(memory, memory.occupied) == retained(before_occupied)
    np.testing.assert_allclose(memory.origin/GRID_RESOLUTION,
                               np.rint(memory.origin/GRID_RESOLUTION), atol=1e-12)
    assert memory.recenter_count == 1
    assert not memory.recenter(shift)
    assert memory.recenter_count == 1


@pytest.mark.parametrize('traversal', ['sampled_v1', 'voxel_dda_v1'])
def test_more_than_eighty_metres_and_vertical_movement_keep_memory_bounded(traversal):
    memory = LocalMap([[0., 0., 2.], [1000., 1000., 1000.]], rolling=True,
                      window_size=(6., 6., 4.), ray_traversal=traversal)
    shape = memory.seen.shape
    nbytes = memory.seen.nbytes+memory.occupied.nbytes
    positions = [(0., 0., 2.), (20., -20., 5.), (80., 70., 12.),
                 (100., 80., 20.), (-80., -100., -12.)]
    for index, position in enumerate(positions):
        memory.add(scan(index*.1, position))
        hits, evidence = memory.snapshot(index*.1, position)
        assert memory.seen.shape == shape
        assert memory.seen.nbytes+memory.occupied.nbytes == nbytes
        assert len(hits) == 6 and len(evidence) > 6
        np.testing.assert_allclose(hits.mean(axis=0), position, atol=1e-12)
        assert np.linalg.norm(evidence-np.asarray(position), axis=1).max() < 1.5
    # Travelling back without a new measurement cannot revive evicted evidence.
    memory.recenter(positions[0])
    hits, evidence = memory.snapshot(.4, positions[0])
    assert not len(hits) and not len(evidence)


def test_crossing_negative_world_cells_never_changes_lattice_phase():
    memory = LocalMap([[.19, .19, .19]], rolling=True, window_size=(2., 2., 2.))
    for position in ((.21, .41, .61), (-.01, -.21, -.41), (-.2, -.4, -.6), (.01, .01, .01)):
        memory.recenter(position)
        expected = np.floor(np.asarray(position)/GRID_RESOLUTION).astype(int)-np.asarray(memory.seen.shape)//2
        np.testing.assert_allclose(memory.origin, expected*GRID_RESOLUTION, atol=1e-12)


def test_half_voxel_hit_quantization_is_independent_of_moving_origin():
    memory = LocalMap([[0., 0., 0.]], rolling=True, window_size=(4., 4., 4.))
    points = np.array([[.1, -.3, .5], [-.1, .3, -.5]])
    expected = np.rint(points/GRID_RESOLUTION).astype(np.int64)
    for position in ((0., 0., 0.), (.2, -.2, .2), (-.4, .6, -.2)):
        memory.recenter(position)
        actual = memory._cell_indices(points)+np.rint(memory.origin/GRID_RESOLUTION).astype(np.int64)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('rolling', [False, True])
def test_snapshot_outside_any_side_of_map_is_empty(rolling):
    memory = LocalMap([[0., 0., 0.], [2., 0., 0.]], rolling=rolling)
    memory.add(scan(0.))
    for axis in range(3):
        for sign in (-1., 1.):
            position = np.zeros(3); position[axis] = 100.*sign
            hits, evidence = memory.snapshot(0., position)
            assert hits.shape == (0, 3) and evidence.shape == (0, 3)


@pytest.mark.parametrize('traversal', ['sampled_v1', 'voxel_dda_v1'])
def test_occupied_timestamps_survive_shift_and_expire_without_becoming_permanent(traversal):
    memory = LocalMap([[0., 0., 0.]], seconds=1., rolling=True,
                      window_size=(6., 6., 4.), ray_traversal=traversal)
    memory.add(scan(0., directions=[[1., 0., 0.]], distances=[1.]))
    memory.recenter([.4, .2, .2])
    hit_cell = (5, 0, 0)
    assert hit_cell in stamps(memory, memory.occupied)
    # A fresh ray through the old hit does not override an unexpired occupied cell.
    memory.add(scan(.5, directions=[[1., 0., 0.]], distances=[2.], valid=[False]))
    assert hit_cell not in world_cells(memory.snapshot(.5, [0., 0., 0.])[1])
    # Once the old occupied evidence expires, the still-fresh later ray supports free space.
    assert hit_cell in world_cells(memory.snapshot(1.01, [0., 0., 0.])[1])
    assert not len(memory.snapshot(1.51, [0., 0., 0.])[1])
    assert not len(memory.snapshot(1.51, [0., 0., 0.])[0])


@pytest.mark.parametrize('rolling', [False, True])
def test_past_snapshot_never_authorizes_future_rays_or_hits(rolling):
    memory = LocalMap([[0., 0., 0.]], rolling=rolling, ray_traversal='voxel_dda_v1')
    memory.add(scan(0., directions=[[1., 0., 0.]], distances=[1.4]))
    memory.add(scan(1., directions=[[0., 1., 0.]], distances=[1.4]))
    hits, evidence = memory.snapshot(.5, [0., 0., 0.])
    np.testing.assert_allclose(hits, [[1.4, 0., 0.]])
    assert len(evidence)
    assert np.abs(evidence[:, 1:]).max() < 1e-12
    assert len(memory.snapshot(1., [0., 0., 0.])[1]) > len(evidence)


@pytest.mark.parametrize('traversal', ['sampled_v1', 'voxel_dda_v1'])
def test_same_scans_replay_exactly_and_agree_with_fixed_world_map(traversal):
    rolling = LocalMap([[0., 0., 2.]], rolling=True, window_size=(12., 12., 8.), ray_traversal=traversal)
    replay = LocalMap([[-300., 80., -70.], [900., 600., 40.]], rolling=True,
                      window_size=(12., 12., 8.), ray_traversal=traversal)
    fixed = LocalMap([[0., 0., 2.], [3., 0., 3.]], ray_traversal=traversal)
    for index in range(16):
        # Avoid exact voxel-boundary ties; fixed legacy maps retain their
        # historical floating-point tie convention, while rolling maps use a
        # stable global convention tested independently above.
        position = [.037+index*.137, .187*np.sin(index), 2.037+index*.047]
        frame = scan(index*.1, position)
        for memory in (rolling, replay, fixed):
            memory.add(frame)
        hit, evidence = rolling.snapshot(index*.1, position)
        replay_hit, replay_evidence = replay.snapshot(index*.1, position)
        fixed_hit, fixed_evidence = fixed.snapshot(index*.1, position)
        np.testing.assert_array_equal(rolling.origin, replay.origin)
        np.testing.assert_array_equal(hit, replay_hit)
        np.testing.assert_array_equal(evidence, replay_evidence)
        np.testing.assert_array_equal(hit, fixed_hit)
        assert world_cells(evidence) == world_cells(fixed_evidence)


def test_invalid_window_and_nonmonotonic_scans_are_rejected():
    for size in ((0., 2., 2.), (2., 2.), (2., np.inf, 2.), (.1, 2., 2.)):
        with pytest.raises(ValueError):
            LocalMap([[0., 0., 0.]], rolling=True, window_size=size)
    memory = LocalMap([[0., 0., 0.]], rolling=True)
    memory.add(scan(1.))
    with pytest.raises(ValueError, match='monotonic'):
        memory.add(scan(.9, [80., 0., 0.]))
    np.testing.assert_allclose(memory.origin, [-12., -10., -5.])
    with pytest.raises(ValueError):
        memory.recenter([np.nan, 0., 0.])


def test_static_memory_keeps_free_space_and_hits_beyond_the_raw_scan_window():
    # Run 198e22c3: an upright MID-360 sees the lower body volume only from
    # more than 2 m away, so a 3 s window had already dropped it.
    kwargs = dict(rolling=True, window_size=(8., 8., 4.), ray_traversal='voxel_dda_v1')
    short, long = LocalMap([[0., 0., 0.]], 3., **kwargs), LocalMap([[0., 0., 0.]], 3., memory_seconds=30., **kwargs)
    for memory in (short, long):
        memory.add(scan(0.))
    _, forgotten = short.snapshot(10., np.zeros(3))
    points, remembered = long.snapshot(10., np.zeros(3))
    assert not len(forgotten) and len(remembered)
    # Remembered returns stay obstacles at their exact positions, never free.
    hits = scan(0.)['points_sensor']
    assert world_cells(points) == world_cells(hits)
    assert not world_cells(remembered) & world_cells(hits)
    assert not len(long.snapshot(31., np.zeros(3))[1])
    with pytest.raises(ValueError, match='memory'):
        LocalMap([[0., 0., 0.]], 3., memory_seconds=1., **kwargs)


def test_takeoff_column_is_observed_but_never_over_a_recorded_hit():
    memory = LocalMap([[0., 0., 0.]], 3., rolling=True, window_size=(8., 8., 6.),
                      ray_traversal='voxel_dda_v1', memory_seconds=30.)
    memory.add(scan(0., directions=[[0., 0., -1.]], distances=[.8]))   # one hit 0.8 m below
    assert memory.mark_observed_column(np.zeros(3), 0., depth=1., radius=.6) > 0
    _, evidence = memory.snapshot(.1, np.zeros(3))
    column = evidence[(np.linalg.norm(evidence[:, :2], axis=1) <= .6) & (evidence[:, 2] <= 0.)]
    assert column[:, 2].min() <= -.9 and np.linalg.norm(column[:, :2], axis=1).max() >= .4
    assert (0, 0, -4) not in world_cells(evidence)


def test_evidence_range_sets_how_far_free_space_is_recorded_along_rays():
    # Run 976f816b: with a 6 m cap an upright LiDAR could never pre-observe the
    # body volume of a 0.5 m descent ahead.
    kwargs = dict(rolling=True, window_size=(24., 8., 4.), ray_traversal='voxel_dda_v1')
    frame = scan(0., directions=[[1., 0., 0.]], distances=[40.], valid=[False])   # no return
    reach = {}
    for limit in (6., 11.):
        memory = LocalMap([[0., 0., 0.]], 3., evidence_range=limit, **kwargs)
        memory.add(frame)
        # Snapshots crop to +-5 m around the query point; look further along the ray.
        _, evidence = memory.snapshot(.1, np.array([8., 0., 0.]))
        reach[limit] = evidence[:, 0].max()
    assert reach[6.] <= 6.1 and reach[11.] >= 10.8
    with pytest.raises(ValueError, match='evidence_range'):
        LocalMap([[0., 0., 0.]], 3., evidence_range=0., **kwargs)
