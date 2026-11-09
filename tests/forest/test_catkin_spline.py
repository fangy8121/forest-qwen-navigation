import numpy as np
import pytest
from omni_drones.navigation.forest.catkin_spline import CatkinSpline


def test_cubic_bezier_line_has_exact_endpoints_and_analytic_velocity():
    curve=CatkinSpline([[0,0,0],[1,0,0],[2,0,0],[3,0,0]],2.)
    t=np.linspace(0,2,11);p,v,a=curve.sample(t)
    np.testing.assert_allclose(p[:,0],1.5*t,atol=1e-12)
    np.testing.assert_allclose(v,np.tile([1.5,0,0],(11,1)),atol=1e-12)
    np.testing.assert_allclose(a,0,atol=1e-12)
    assert curve.derivative_bounds()==dict(speed=1.5,acceleration=0.,jerk=0.)
    restored=CatkinSpline.from_dict(curve.as_dict())
    np.testing.assert_array_equal(restored.sample(t)[0],p)


def test_derivatives_and_full_interval_bounds_for_curved_multispan_spline():
    curve=CatkinSpline([[0,0,2],[.1,.2,2],[1,1,2.2],[2,-1,2.4],[3,0,2.2],[3.1,0,2]],.5)
    t=np.linspace(.001,curve.duration-.001,1001);p,v,a=curve.sample(t)
    eps=1e-5
    np.testing.assert_allclose((curve.sample(t+eps)[0]-curve.sample(t-eps)[0])/(2*eps),v,atol=1e-7)
    np.testing.assert_allclose((curve.sample(t+eps)[1]-curve.sample(t-eps)[1])/(2*eps),a,atol=1e-6)
    bounds=curve.derivative_bounds()
    for name,values in [('speed',v),('acceleration',a),('jerk',curve.derivatives[2](t))]:
        assert np.linalg.norm(values,axis=1).max()<=bounds[name]+1e-9


def test_rejects_ambiguous_units_invalid_interval_and_extrapolation():
    points=np.zeros((4,3))
    for interval in (0,-1,np.nan):
        with pytest.raises(ValueError):CatkinSpline(points,interval)
    curve=CatkinSpline(points,1.)
    with pytest.raises(ValueError):curve.sample(-.01)
    with pytest.raises(ValueError):curve.sample(1.01)
    value=curve.as_dict();value['frame']='body'
    with pytest.raises(ValueError):CatkinSpline.from_dict(value)
