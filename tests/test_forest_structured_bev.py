"""Geometry tests for the shared fixed-altitude upper-layer representation."""
import numpy as np
import pytest
from scipy.spatial import cKDTree

from omni_drones.navigation.forest.geometry import INFLATION, observed_body_height
from omni_drones.navigation.forest.structured_bev import (
    contiguous_intervals, extract_structured_bev, summarize_cross_sections)


def evidence_plane(bounds=(-1., 1., -2., 2.), height=0., step=.1):
    x = np.arange(bounds[0]-.7, bounds[1]+.71, step)
    y = np.arange(bounds[2]-.7, bounds[3]+.71, step)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    return np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, height)]


def test_all_intervals_retain_positions_and_grid_boundary_widths():
    ys = np.arange(-2., 2.01, .5)
    gaps = contiguous_intervals(ys, [True, True, False, True, False,
                                    True, True, True, False], (-2., 2.), .5)
    assert [(gap['index_start'], gap['index_end']) for gap in gaps] == [(0, 1), (3, 3), (5, 7)]
    np.testing.assert_allclose([(gap['y_min'], gap['y_max']) for gap in gaps],
                               [(-2., -1.25), (-.75, -.25), (.25, 1.75)])
    np.testing.assert_allclose([gap['width'] for gap in gaps], [.75, .5, 1.5])
    assert not contiguous_intervals(ys, np.zeros(len(ys), bool), (-2., 2.), .5)


def test_unknown_breaks_a_gap_instead_of_becoming_free():
    xs, ys = np.array([0., 1.]), np.arange(-1., 1.01, .5)
    observed = np.array([[True, True, False, True, True], [True]*5])
    free = observed.copy()
    sections, widths, centers, edges, ratios, coverage, unknown = summarize_cross_sections(
        xs, ys, free, observed, ~observed, (0., 1., -1., 1.), .5)
    assert len(sections[0]['intervals']) == 2
    assert sections[0]['center'] is None
    assert widths[0, 1] == 0 and np.isnan(centers[0, 1])
    assert np.isnan(edges[0, 1]).all()
    assert ratios[0, 0] < 1 and unknown[0, 0] > 0
    assert sections[1]['center_connects_left'] and sections[1]['center_connects_right']
    assert ratios[1, 0] == coverage[1, 0] == 1


def test_side_summaries_clip_center_gap_and_keep_original_interval():
    xs, ys = np.array([0.]), np.arange(-2., 2.01, .2)
    free = np.ones((1, len(ys)), bool)
    sections, widths, _, _, *_ = summarize_cross_sections(
        xs, ys, free, free, ~free, (0., 1., -2., 2.), .2)
    assert len(sections[0]['intervals']) == 1
    np.testing.assert_allclose(widths[0], [4., 4., 2., 2.])
    assert sections[0]['left']['y_min'] == 0
    assert sections[0]['right']['y_max'] == 0


def test_raw_single_hit_blocks_even_when_density_is_below_threshold_and_unobserved():
    bev = extract_structured_bev([[0., 0., 0.]], np.empty((0, 3)), bounds=(-1., 1., -1., 1.))
    ix, iy = np.argmin(abs(bev.xs)), np.argmin(abs(bev.ys))
    assert bev.hit_count.sum() == 1
    assert bev.occupancy_density.max() == .2
    assert bev.blocked[ix, iy] and not bev.observed[ix, iy]
    assert not bev.free.any()
    assert not bev.unknown[ix, iy]
    assert bev.unknown.any()


def test_body_center_gap_width_does_not_subtract_inflation_twice():
    bounds = (-1., 1., -2., 2.)
    x = np.arange(-2., 2.01, .1)
    walls = np.r_[np.c_[x, np.full(len(x), -1.5), np.zeros(len(x))],
                  np.c_[x, np.full(len(x), 1.5), np.zeros(len(x))]]
    bev = extract_structured_bev(walls, evidence_plane(bounds), bounds=bounds)
    ix = np.argmin(abs(bev.xs))
    width = bev.sections[ix]['center']['width']
    geometric_center_span = 3.-2*INFLATION
    assert abs(width-geometric_center_span) <= bev.resolution+1e-10
    assert width > 1.5  # A second body subtraction would yield less than .6 m.
    assert bev.sections[ix]['center']['y_min'] < 0 < bev.sections[ix]['center']['y_max']


def test_selected_height_ignores_other_height_collision_but_not_evidence():
    bounds = (-1., 1., -1., 1.)
    plane = evidence_plane(bounds)
    zero = extract_structured_bev([[0., 0., 1.]], plane, bounds=bounds)
    high = extract_structured_bev([[0., 0., 1.]], plane, bounds=bounds, height=1.)
    assert not zero.blocked.any() and zero.free.all()
    assert high.blocked.any() and not high.observed.any()
    assert high.hit_count.sum() == 1 and zero.hit_count.sum() == 0


def test_corridor_masks_unknown_and_free_without_erasing_raw_hits():
    bounds = (-1., 1., -1., 1.)
    corridor = np.array([[-1., -.4], [1., -.4], [1., .4], [-1., .4]])
    bev = extract_structured_bev([[0., .9, 0.]], evidence_plane(bounds),
                                 bounds=bounds, corridor=corridor)
    assert not bev.free[~bev.allowed].any()
    assert not bev.unknown[~bev.allowed].any()
    assert bev.blocked[~bev.allowed].any()
    assert bev.sections[np.argmin(abs(bev.xs))]['widest']['width'] <= 1.+1e-10


def test_actual_body_support_is_sample_union_not_circle_clearing_and_preserves_hits():
    # Sparse body-unrelated evidence gives the executor a nonempty ray tree.
    evidence = np.array([[3., 3., 0.]])
    strict = extract_structured_bev(np.empty((0, 3)), evidence, bounds=(-1., 1., -1., 1.))
    supported = extract_structured_bev(np.empty((0, 3)), evidence,
                                      bounds=(-1., 1., -1., 1.), actual_start=[0., 0., 0.])
    ix, iy = np.argmin(abs(supported.xs)), np.argmin(abs(supported.ys))
    assert not strict.observed.any() and not supported.ray_observed.any()
    assert supported.observed[ix, iy] and supported.actual_body_supported[ix, iy]
    # A shifted center whose center lies inside the actual body cannot borrow
    # an unobserved leading surface outside that body.
    assert not supported.observed[np.argmin(abs(supported.xs-.4)), iy]
    assert supported.observed.sum() < 5
    occupied = extract_structured_bev([[0., 0., 0.]], evidence,
                                     bounds=(-1., 1., -1., 1.), actual_start=[0., 0., 0.])
    assert occupied.blocked[ix, iy] and not occupied.free[ix, iy]
    assert supported.spec()['actual_start'] == [0., 0., 0.]


def test_array_and_tree_evidence_have_identical_masks():
    bounds = (-1., 1., -1., 1.)
    points, evidence = [[.6, -.6, 0.]], evidence_plane(bounds)
    a = extract_structured_bev(points, evidence, bounds=bounds, actual_start=[0., 0., 0.])
    b = extract_structured_bev(points, cKDTree(evidence), bounds=bounds, actual_start=[0., 0., 0.])
    for name in ('free', 'unknown', 'blocked', 'observed', 'ray_observed', 'actual_body_supported'):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    np.testing.assert_allclose(a.gap_widths, b.gap_widths)


@pytest.mark.parametrize('kwargs', [dict(resolution=0), dict(bounds=(0., 0., -1., 1.)),
                                  dict(height=float('nan')), dict(actual_start=[0., 0.])])
def test_invalid_geometry_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        extract_structured_bev(np.empty((0, 3)), np.empty((0, 3)), **kwargs)
