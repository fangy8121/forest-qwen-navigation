"""Independent Python local cubic-spline optimization and offline validation.

Survey Planner supplies the algorithmic reference, not copied Ceres code.
Endpoint states are eliminated exactly; soft costs never authorize execution.
The returned candidate must pass the separate full-curve/braking validator.
"""
from dataclasses import dataclass, field
import time
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import least_squares, minimize
from .catkin_spline import CatkinSpline
from .geometry import INFLATION, VERTICAL_RADIUS, center_clearance
from .local import BODY_NUMERIC_TOLERANCE
from .trajectory import QuinticPath

# Weight on vertical deviation from the (body-observed) route; xyz tracking is 2.
HEIGHT_WEIGHT = 20.


@dataclass
class SplineValidation:
    accepted: bool
    reason: str
    bounds: dict = field(default_factory=dict)
    curve_samples: int = 0
    brake_checks: int = 0
    failed_time: float | None = None


@dataclass
class SplineOptimization:
    path: CatkinSpline | None
    validation: SplineValidation
    attempts: list
    seconds: float


def endpoint_controls(position, velocity, acceleration, end, interval, count=10):
    """Clamped cubic controls with exact initial p/v/a and final p/v/a=(end,0,0)."""
    state = np.asarray([position, velocity, acceleration, end], dtype=float)
    if (state.shape != (4, 3) or not np.isfinite(state).all() or count < 7
            or not np.isfinite(interval) or interval <= 0):
        raise ValueError('Need finite XYZ endpoint state, positive interval and >= 7 controls')
    p, v, a, end = state
    controls = np.linspace(p, end, count)
    controls[0] = p
    controls[1] = p+interval*v/3
    # The second knot span is twice the first in the endpoint derivative curve.
    controls[2] = controls[1]+(2*interval/3)*(v+.5*interval*a)
    controls[-3:] = end
    return controls


def stop_profile(length, start_speed, cruise_speed, acceleration):
    """Arc length over normalized time for reach-cruise-stop along one route.

    Returns (duration, fraction): ``fraction(tau)`` is the travelled share of
    ``length`` at ``tau*duration``. Speed goes from ``start_speed`` to at most
    ``cruise_speed`` and back to rest exactly at the route end, all at
    ``acceleration``; a start too fast to stop that way brakes harder.
    """
    if length <= 0 or cruise_speed <= 0 or acceleration <= 0:
        raise ValueError('Need a positive length, cruise speed and acceleration')
    v0 = min(max(float(start_speed), 0.), cruise_speed)
    a = float(acceleration)
    if v0*v0/(2*a) >= length:
        # Cannot stop at this rate: one uniform deceleration to the end.
        brake = v0*v0/(2*length)
        times, speeds = np.array([0., v0/brake]), np.array([v0, 0.])
    else:
        peak = min(cruise_speed, np.sqrt(a*length+v0*v0/2))
        rise, fall = (peak-v0)/a, peak/a
        cruise = (length-(peak*peak-v0*v0)/(2*a)-peak*peak/(2*a))/peak
        times = np.cumsum([0., rise, max(cruise, 0.), fall])
        speeds = np.array([v0, peak, peak, 0.])
    distance = np.r_[0., np.cumsum(np.diff(times)*(speeds[:-1]+speeds[1:])/2)]
    duration = float(times[-1])

    def fraction(tau):
        t = np.clip(np.asarray(tau, float), 0., 1.)*duration
        index = np.clip(np.searchsorted(times, t, side='right')-1, 0, len(times)-2)
        dt = t-times[index]
        slope = (speeds[index+1]-speeds[index])/np.maximum(times[index+1]-times[index], 1e-12)
        s = distance[index]+speeds[index]*dt+.5*slope*dt*dt
        return np.clip(s/distance[-1], 0., 1.)
    return duration, fraction


def validate_spline(path, safety, actual, *, expected_start=None, expected_end=None,
                    spacing=.04, brake_period=.2):
    """Check nominal curve, body evidence, and complete stops along the curve.

    Kinematic bounds use derivative control hulls over the full interval.
    Collision sampling includes padding bounded by the global speed and sample
    period. Ray/body coverage retains the existing discrete-evidence contract;
    neither this nor a nominal brake is a tracking-error safety proof.
    """
    if spacing <= 0 or brake_period <= 0:
        raise ValueError('Validation spacings must be positive')
    bounds = path.derivative_bounds()
    for name, limit in (('speed', safety.flight.max_speed),
                        ('acceleration', safety.flight.max_acceleration),
                        ('jerk', safety.flight.max_jerk)):
        if bounds[name] > limit*(1+1e-7):
            return SplineValidation(False, 'DYNAMICS_'+name.upper(), bounds)
    if expected_start is not None:
        if not np.allclose(path.sample(0.), expected_start, atol=1e-8, rtol=0.):
            return SplineValidation(False, 'START_STATE', bounds)
    end = path.sample(path.duration)
    if ((expected_end is not None and not np.allclose(end[0], expected_end, atol=1e-8, rtol=0.))
            or np.linalg.norm(end[1])+np.linalg.norm(end[2]) > 1e-8):
        return SplineValidation(False, 'END_STATE', bounds)
    count = max(2, int(np.ceil(path.duration*max(bounds['speed'], 1e-6)/spacing))+1)
    times = np.linspace(0., path.duration, count)
    positions = path.sample(times)[0]
    result = SplineValidation(False, '', bounds, count)
    pad = .05+.5*bounds['speed']*path.duration/(count-1)
    # A curve may start inside the planning reserve but only move away from it.
    if not safety.clear_leaving(positions, pad, actual):
        result.reason = 'OBSTACLE_OR_PUBLIC_BOUNDS'
        return result
    if not safety.known(positions, actual, body_padding=BODY_NUMERIC_TOLERANCE):
        result.reason = 'UNOBSERVED_CURVE'
        return result
    # Height above the ground: routes kept to the band still overshot 0.2 m when
    # a climb was handed on with its upward speed (run 04376dbe: 3.58 m against
    # the task's 3.5 m). The curve and every stop along it stay in the band.
    band = getattr(safety, 'height_band', None)
    if band is not None and not band.leaving(positions):
        result.reason = 'HEIGHT_BAND_CURVE'
        return result
    # Include the actual-state connector; a shifted start must not bypass it.
    connector = np.linspace(actual, positions[0],
                            max(2, int(np.ceil(np.linalg.norm(actual-positions[0])/spacing))+1))
    if (not safety.clear(connector, start=actual)
            or not safety.known(connector, actual, body_padding=BODY_NUMERIC_TOLERANCE)):
        result.reason = 'START_CONNECTOR'
        return result
    for t in np.linspace(0., path.duration, max(2, int(np.ceil(path.duration/brake_period))+1)):
        p, v, a = path.sample(t)
        try:
            brake = QuinticPath.brake(p, v, a, safety.flight.max_acceleration, safety.flight.max_jerk)
        except ValueError:
            result.reason, result.failed_time = 'BRAKE_DYNAMICS', float(t)
            return result
        envelope = brake.remaining(0., p)
        result.brake_checks += 1
        if not safety.clear(envelope, start=envelope[0]):
            result.reason, result.failed_time = 'OBSTACLE_BRAKE', float(t)
            return result
        if not safety.known(envelope, actual, body_padding=BODY_NUMERIC_TOLERANCE):
            result.reason, result.failed_time = 'UNOBSERVED_BRAKE', float(t)
            return result
        if band is not None and not band.leaving(envelope):
            result.reason, result.failed_time = 'HEIGHT_BAND_BRAKE', float(t)
            return result
    result.accepted, result.reason = True, ''
    return result


class PythonSplineOptimizer:
    """Small analytic-Jacobian least-squares problem plus bounded retiming.

    Route fitting and smoothness share costs with observed-obstacle clearance,
    center evidence and derivative-hull limits. Initial state is rebuilt on
    every retiming attempt; simply stretching a moving spline changes its
    starting velocity and acceleration and is therefore forbidden here.
    """
    def __init__(self, flight, *, controls=10, max_evaluations=35, retimings=3,
                 enforce_feasible_controls=False, profile_acceleration=None):
        if controls < 7 or max_evaluations < 1 or retimings < 1:
            raise ValueError('Invalid optimizer budget')
        if profile_acceleration is not None and not 0 < profile_acceleration <= flight.max_acceleration:
            raise ValueError('profile_acceleration must be within (0, max_acceleration]')
        self.flight, self.count = flight, controls
        self.max_evaluations, self.retimings = max_evaluations, retimings
        self.enforce_feasible_controls = bool(enforce_feasible_controls)
        # None keeps the uniform 1.5x-length timing. A value times the route
        # as reach-cruise-stop at that rate, so only its last metres brake.
        self.profile_acceleration = None if profile_acceleration is None else float(profile_acceleration)

    def optimize(self, route, position, velocity, acceleration, esdf, safety, *, nominal_speed=.5,
                 actual_position=None):
        began = time.perf_counter()
        route = np.asarray(route, dtype=float)
        state = np.asarray([position, velocity, acceleration], dtype=float)
        if (route.ndim != 2 or route.shape[1] != 3 or len(route) < 2
                or not np.isfinite(route).all() or state.shape != (3, 3)
                or not np.isfinite(state).all() or not np.allclose(route[0], position, atol=1e-6)
                or not np.isfinite(nominal_speed) or nominal_speed <= 0):
            raise ValueError('Need a finite route starting at the supplied state and positive speed')
        lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
        keep = np.r_[True, np.diff(lengths) > 1e-8]
        route, lengths = route[keep], lengths[keep]
        if lengths[-1] < 1e-4:
            return SplineOptimization(None, SplineValidation(False, 'NO_ROUTE'), [], time.perf_counter()-began)
        for name, values, limit in (('VELOCITY', velocity, self.flight.max_speed),
                                    ('ACCELERATION', acceleration, self.flight.max_acceleration)):
            if np.linalg.norm(values) > limit*(1+1e-7):
                return SplineOptimization(None, SplineValidation(False, 'INITIAL_'+name), [], time.perf_counter()-began)
        n = self.count
        # Unit-time basis permits precomputation of all position and derivative Jacobians.
        template = CatkinSpline(np.zeros((n, 3)), 1.)
        basis = BSpline(template.curve.t, np.eye(n), 3, extrapolate=False)
        u = np.linspace(0., n-3, 49)
        b = basis(u)
        derivative_maps = []
        for order in (1, 2, 3):
            curve = basis.derivative(order)
            derivative_maps.append(curve.c[:len(curve.t)-curve.k-1])
        cruise_speed = min(nominal_speed, self.flight.max_speed)
        if self.profile_acceleration is None:
            share = u/(n-3)
            profile_duration = 1.5*lengths[-1]/cruise_speed
        else:
            profile_duration, fraction = stop_profile(lengths[-1], np.linalg.norm(velocity), cruise_speed,
                                                      self.profile_acceleration)
            share = fraction(u/(n-3))
        # Retiming stretches this same normalized reference uniformly.
        reference = np.column_stack([np.interp(share*lengths[-1], lengths, route[:, d]) for d in range(3)])
        free = np.arange(3, n-3)
        spatial_jac = np.einsum('ij,ab->iajb', b[:, free], np.eye(3)).reshape(len(u), 3, -1)
        derivative_jacs = [np.einsum('ij,ab->iajb', d[:, free], np.eye(3)).reshape(len(d), 3, -1)
                           for d in derivative_maps]
        clearance = center_clearance(self.flight)
        # A spherical soft cost is intentionally conservative; final validation
        # uses the configured cylinder/sphere geometry, not this approximation.
        clearance = np.hypot(INFLATION, VERTICAL_RADIUS)+.05 if clearance is None else clearance+.05
        clearance += getattr(safety, 'clearance_reserve', 0.)
        duration = max(1., profile_duration, 2*np.linalg.norm(velocity)/self.flight.max_acceleration)
        attempts, last_path = [], None
        validation = SplineValidation(False, 'OPTIMIZATION_FAILED')
        for attempt in range(self.retimings):
            interval = duration/(n-3)
            fixed = endpoint_controls(position, velocity, acceleration, route[-1], interval, n)
            if ((fixed < esdf.origin) | (fixed > esdf.upper)).any():
                validation = SplineValidation(False, 'ENDPOINT_OUTSIDE_LOCAL_MAP')
                break
            initial = np.linalg.lstsq(b[:, free], reference-b@np.where(
                np.isin(np.arange(n), free)[:, None], 0., fixed), rcond=None)[0]
            lower, upper = np.broadcast_to(esdf.origin, initial.shape), np.broadcast_to(esdf.upper, initial.shape)
            initial = np.clip(initial, lower+1e-8, upper-1e-8)

            def evaluate(value, jacobian=False):
                controls = fixed.copy()
                controls[free] = value.reshape(-1, 3)
                positions = b@controls
                residuals = [2.*(positions-reference).ravel()]
                jacobians = [2.*spatial_jac.reshape(-1, len(value))]
                for order, (d, jac, weight, limit) in enumerate(zip(
                        derivative_maps, derivative_jacs, (.03, .05, .01),
                        (self.flight.max_speed, self.flight.max_acceleration, self.flight.max_jerk)), 1):
                    values, dj = d@controls/interval**order, jac/interval**order
                    norms = np.linalg.norm(values, axis=1)
                    # Reserve room for finite soft-cost convergence. The actual
                    # limit is still independently enforced by validate_spline.
                    target_limit = .9*limit
                    residuals += [weight*values.ravel(), 10.*np.maximum(0., norms-target_limit)]
                    norm_jac = np.einsum('ij,ijk->ik', values/np.maximum(norms[:, None], 1e-12), dj)
                    jacobians += [weight*dj.reshape(-1, len(value)), 10.*(norms > target_limit)[:, None]*norm_jac]
                # Hold the observed route height. The spherical clearance cost
                # wants ~1.3 m from a branch overhead that the cylinder check
                # clears at 0.55 m. It lowered a stopped body below its
                # observed layer (run cc69e994: 6.5 cm, then every curve
                # UNOBSERVED_CURVE); a branch 0.9 m overhead lowered it 0.6 m.
                residuals.append(HEIGHT_WEIGHT*(positions[:, 2]-reference[:, 2]))
                jacobians.append(HEIGHT_WEIGHT*spatial_jac[:, 2, :])
                query = esdf.query(positions)
                active = query.distance < clearance
                residuals.append(6.*np.maximum(0., clearance-query.distance))
                jacobians.append(-6.*active[:, None]*np.einsum('ij,ijk->ik', query.gradient, spatial_jac))
                distance, gradient = esdf.evidence_distance(positions)
                residuals.append(2.*np.maximum(0., distance-.15))
                jacobians.append(2.*(distance > .15)[:, None]*np.einsum('ij,ijk->ik', gradient, spatial_jac))
                return np.vstack(jacobians) if jacobian else np.concatenate(residuals)

            solved = least_squares(evaluate, initial.ravel(), jac=lambda x: evaluate(x, True),
                                   bounds=(lower.ravel(), upper.ravel()), max_nfev=self.max_evaluations,
                                   ftol=1e-5, xtol=1e-5, gtol=1e-5)
            refinement = None
            if self.enforce_feasible_controls:
                # Soft derivative residuals can converge just outside limits.
                # Refine with actual convex derivative-hull constraints, keeping
                # the exact eliminated endpoint p/v/a and all geometry costs.
                limits = (self.flight.max_speed, self.flight.max_acceleration, self.flight.max_jerk)

                def hull(value, jacobian=False):
                    c = fixed.copy(); c[free] = value.reshape(-1, 3)
                    values, gradients = [], []
                    for order, (d, dj, limit) in enumerate(zip(derivative_maps, derivative_jacs, limits), 1):
                        derivative, derivative_jac = d@c/interval**order, dj/interval**order
                        values.append(1.-np.sum(derivative**2, axis=1)/limit**2)
                        gradients.append(-2.*np.einsum('ij,ijk->ik', derivative, derivative_jac)/limit**2)
                    return np.vstack(gradients) if jacobian else np.concatenate(values)

                def objective(value):
                    residual = evaluate(value)
                    return .5*float(residual@residual)

                refined = minimize(objective, solved.x, jac=lambda x: evaluate(x, True).T@evaluate(x),
                    method='SLSQP', bounds=list(zip(lower.ravel(), upper.ravel())),
                    constraints=dict(type='ineq', fun=hull, jac=lambda x: hull(x, True)),
                    options=dict(maxiter=60, ftol=1e-9))
                # The independent hull and whole-curve validator below still
                # decides acceptance, regardless of either solver's success.
                if np.isfinite(refined.x).all():
                    solved.x = refined.x
                    solved.cost = objective(refined.x)
                refinement = dict(converged=bool(refined.success), evaluations=int(refined.nfev),
                                  minimum_constraint=float(hull(solved.x).min()))
            controls = fixed.copy()
            controls[free] = solved.x.reshape(-1, 3)
            last_path = CatkinSpline(controls, interval)
            bounds = last_path.derivative_bounds()
            validation = None
            for name, limit in (('speed', self.flight.max_speed),
                                ('acceleration', self.flight.max_acceleration), ('jerk', self.flight.max_jerk)):
                if bounds[name] > limit*(1+1e-7):
                    validation = SplineValidation(False, 'DYNAMICS_'+name.upper(), bounds)
                    break
            if validation is None:
                validation = validate_spline(last_path, safety, position if actual_position is None else actual_position,
                                             expected_start=state, expected_end=route[-1])
            attempts.append(dict(interval=interval, optimizer_converged=bool(solved.success),
                                 evaluations=int(solved.nfev), cost=float(solved.cost),
                                 reason=validation.reason, bounds=validation.bounds,
                                 constraint_refinement=refinement))
            if validation.accepted or not validation.reason.startswith('DYNAMICS_'):
                break
            bounds = validation.bounds
            ratio = max(bounds['speed']/self.flight.max_speed,
                        np.sqrt(bounds['acceleration']/self.flight.max_acceleration),
                        np.cbrt(bounds['jerk']/self.flight.max_jerk))
            duration *= max(1.2, 1.05*ratio)
        return SplineOptimization(last_path, validation, attempts, time.perf_counter()-began)
