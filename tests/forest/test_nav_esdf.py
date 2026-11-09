import numpy as np

from forest_generator.nav_esdf import NavigationESDF


def box(center, size):
    c, h = np.asarray(center, float), np.asarray(size, float)/2
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])*h+c
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return corners, faces


def flat(points):
    return np.zeros(len(np.asarray(points)))


def field(vertices, faces, **kwargs):
    return NavigationESDF(vertices, faces, flat, [0., 10., -3., 3.], height=2.5, resolution=.2,
        horizontal=1., vertical=.75, agl_band=(2., 3.), **kwargs)


def test_branch_inside_height_band_needs_horizontal_clearance():
    # A thin horizontal branch at 2.5 m reaching y=0.5 across x=4..6.
    vertices, faces = box([5., 1.5, 2.5], [2., 2., .1])
    esdf = field(vertices, faces)
    assert not esdf.clear(np.array([[5., 1.2, 2.5]]))
    assert esdf.clear(np.array([[5., -1., 2.5]]))
    route = esdf.route(np.array([.5, 0., 2.5]), np.array([9.5, 0., 2.5]), [])
    dense = np.concatenate([np.linspace(a, b, 20) for a, b in zip(route[:-1], route[1:])])
    near = dense[np.abs(dense[:, 0]-5.) < 1.]
    assert (near[:, 1] < -.5+1e-6).all()


def test_surface_far_above_band_does_not_block():
    # 1.2 m above the top of the vertical band (3.0+0.75): outside the cylinder.
    vertices, faces = box([5., 0., 5.], [2., 6., .1])
    esdf = field(vertices, faces)
    assert esdf.clear(np.array([[5., 0., 2.5]]))


def test_height_outside_check_band_is_not_clear():
    esdf = field(*box([9.5, 2.8, .3], [.1, .1, .1]), agl_check=(1.75, 3.25))
    assert esdf.clear(np.array([[5., 0., 1.8]])) and not esdf.clear(np.array([[5., 0., 1.7]]))
    assert not esdf.free[:, :, np.round((1.8-esdf.origin[2])/.2).astype(int)].any()
