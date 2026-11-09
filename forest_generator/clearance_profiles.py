"""Generation clearance derived from the navigation envelope, not a separate constant.

A map is only useful if the online stack can fly what the reference flies. The
local planner admits a coarse cell when its centre clears the body inflation,
the planning reserve and half a grid cell (run a0a40f7f stalled at a task
obstacle whose bypass met the old 0.75 m sphere but not that 1.0 m rule).
``clearance_m`` replaces ``REFERENCE_CLEARANCE`` in every generation check
(route search, smoothing, audits, task placement); ``horizontal_m`` and
``vertical_m`` define the final envelope veto in the planner's own cylinder
geometry.
"""
from omni_drones.navigation.forest.geometry import BODY_VERTICAL_RADIUS, MARGIN, RADIUS, REFERENCE_CLEARANCE

PLANNING_RESERVE = .15  # cfg/forest_coverage.yaml lower.planning_clearance_reserve
LOCAL_GRID = .4  # LocalPlanner default resolution; free cells need half a cell more
DESIRED_AGL = 2.5  # cfg/forest_coverage.yaml coverage.desired_agl
AGL_TOLERANCE = .5  # online heights stayed within 2.31-2.74 m (runs 443cc8d3, 6244d06b)
AGL_CHECK_TOLERANCE = .75  # local A* reaches about +-0.8 m around its guidance height


def navigation_envelope(body_radius, vertical_radius=BODY_VERTICAL_RADIUS):
    padding = MARGIN+PLANNING_RESERVE+LOCAL_GRID/2
    return dict(horizontal_m=round(body_radius+padding, 6), vertical_m=round(vertical_radius+padding, 6))


def clearance_profile(name):
    if name == 'legacy':
        return dict(name=name, clearance_m=REFERENCE_CLEARANCE, horizontal_m=None, vertical_m=None, agl_band_m=None, agl_check_m=None,
                    derivation='historical 0.75 m reference sphere; no navigation envelope veto')
    if name != 'sim':
        raise ValueError(f'unknown clearance profile: {name}')
    envelope = navigation_envelope(RADIUS)
    return dict(name=name, clearance_m=envelope['horizontal_m'], body_radius_m=RADIUS, **envelope,
                agl_band_m=[DESIRED_AGL-AGL_TOLERANCE, DESIRED_AGL+AGL_TOLERANCE],
                # Smoothed curves; within the online evaluation's 1.5-3.5 m.
                agl_check_m=[DESIRED_AGL-AGL_CHECK_TOLERANCE, DESIRED_AGL+AGL_CHECK_TOLERANCE],
                derivation=f'body {RADIUS} + margin {MARGIN} + planning reserve {PLANNING_RESERVE}'
                           f' + half local grid {LOCAL_GRID/2}')


PROFILE_NAMES = ('legacy', 'sim')
