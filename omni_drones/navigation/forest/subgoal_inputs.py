"""Small observation-only candidate set and one compact pilot BEV.

These are geometric movement anchors, not inferred forest corridors.  They use
only current world-frame hits/ray evidence and public task bounds.  A candidate
is not a trajectory or a stopping certificate: the executing lower layer must
recheck fresh geometry, dynamics and its complete stopping envelope.
"""
from pathlib import Path
import math

import numpy as np

from .geometry import local_to_world, world_to_local
from .planning import PathSafety


SCHEMA = 'observed_subgoal_candidates_v1'
BODY_PADDING = 1e-5  # Match the lower executor's numerical tolerance, in metres.


def _vector(value, name):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f'{name} requires a finite world XYZ vector')
    return vector


def _bounds_polygon(bounds):
    if bounds is None:
        return None
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape == (4,):
        xmin, xmax, ymin, ymax = bounds
        if not xmin < xmax or not ymin < ymax:
            raise ValueError('operating_bounds must be ordered xmin,xmax,ymin,ymax')
        bounds = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    if bounds.ndim != 2 or bounds.shape[1] != 2 or len(bounds) < 3 or not np.isfinite(bounds).all():
        raise ValueError('operating_bounds must be a finite XY polygon or four bounds')
    return bounds


def _safety(snapshot, position, yaw, flight, operating_bounds, height_bounds):
    points = np.asarray(snapshot['points'], dtype=float).reshape(-1, 3)
    evidence = np.asarray(snapshot['evidence'], dtype=float).reshape(-1, 3)
    if not np.isfinite(points).all() or not np.isfinite(evidence).all():
        raise ValueError('observation hits and ray evidence must be finite')
    heights = None if height_bounds is None else np.asarray(height_bounds, dtype=float)
    if heights is not None and (heights.shape != (2,) or not np.isfinite(heights).all()
                                or not heights[0] < heights[1]):
        raise ValueError('height_bounds require two ordered world Z bounds')
    result = PathSafety(points, evidence, _bounds_polygon(operating_bounds), flight,
                        height_bounds=height_bounds)
    result.frame_position, result.frame_yaw = position, yaw
    return result


def _dense_segment(position, target, spacing=.04):
    length = float(np.linalg.norm(target-position))
    return np.linspace(position, target, max(2, int(math.ceil(length/spacing))+1))


def validate_candidate(snapshot, position, yaw, candidate, flight, operating_bounds=None,
                       height_bounds=None):
    """Recheck one selected movement anchor against current observed geometry.

    Malformed/nonfinite candidate or observation data is rejected with ``False``.
    HOLD is never a movement authorization.  This deliberately does not rebuild
    other candidates or require that a geometric slot still be offered; the
    supervisor must check its task/candidate identity, versions and observation
    age separately.  Full stopping and dynamics checks remain with the executor.
    """
    if not isinstance(candidate, dict):
        return False
    if (str(candidate.get('id', '')).upper() == 'HOLD'
            or str(candidate.get('candidate_id', '')).upper() == 'HOLD'
            or str(candidate.get('kind', '')).lower() == 'hold'
            or candidate.get('movement_authorized', True) is False):
        return False
    try:
        position = _vector(position, 'position')
        target = _vector(candidate['target_world'], 'candidate target')
        if not math.isfinite(yaw):
            return False
        safety = _safety(snapshot, position, yaw, flight, operating_bounds, height_bounds)
        # Reject an outside endpoint before allocating a potentially long line.
        if not safety.clear_points(target[None]).all():
            return False
        segment = _dense_segment(position, target)
        return bool(safety.clear(segment)
                    and safety.known(segment, position, body_padding=BODY_PADDING))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def build_candidates(snapshot, position, yaw, home_world, flight, operating_bounds=None,
                     height_bounds=None, visited_anchors=(), step_distance=2., anchor_yaw=0.):
    """Return task-independent candidates using observed straight connections.

    ``snapshot['points']`` and ``snapshot['evidence']`` are world XYZ arrays.
    ``visited_anchors`` may contain prior actual world XYZ positions or objects
    with ``id``/``target_world``.  Past visits do not bypass current observation
    checks.  Directional anchors are fixed around home in the ``anchor_yaw``
    takeoff-heading frame.  IDs and world targets therefore remain stable after
    translation/rotation; the caller must still bind observation versions before
    asynchronous inference.  This is a public short-range pilot anchor set.

    The always-present HOLD is a request to remain stopped, not a safety claim.
    Anchors closer than 0.3 m to the current position, including HOME, are omitted.
    Actual arrival/completion history belongs to the task supervisor.
    """
    position, home = _vector(position, 'position'), _vector(home_world, 'home_world')
    if (not math.isfinite(yaw) or not math.isfinite(anchor_yaw)
            or not math.isfinite(step_distance) or not .3 <= step_distance <= 3.):
        raise ValueError('finite yaw and step_distance in [0.3,3] metres are required')
    safety = _safety(snapshot, position, yaw, flight, operating_bounds, height_bounds)
    timestamp = float(snapshot['timestamp'])
    if not math.isfinite(timestamp):
        raise ValueError('snapshot timestamp must be finite')

    def record(identifier, kind, target, label, movement_authorized):
        relative = world_to_local([target], position, yaw)[0]
        return dict(id=identifier, candidate_id=identifier, kind=kind, label=label,
                    target_world=np.round(target, 6).tolist(),
                    relative_body_xyz_m=np.round(relative, 6).tolist(),
                    relative_home_xyz_m=np.round(world_to_local([target], home, anchor_yaw)[0], 6).tolist(),
                    distance_m=round(float(np.linalg.norm(target-position)), 6),
                    evidence_timestamp=timestamp,
                    geometry='dense_current_observation_straight_connection' if movement_authorized else 'no_motion_request',
                    movement_authorized=movement_authorized,
                    execution_certificate=False)

    candidates = [record('HOLD', 'hold', position, 'Hold current position / observe', False)]
    used = []

    def append(identifier, kind, target, label):
        target = _vector(target, 'candidate target')
        if (np.linalg.norm(target-position) < .3
                or any(np.linalg.norm(target-old) < .2 for old in used)):
            return
        segment = _dense_segment(position, target)
        if safety.clear(segment) and safety.known(segment, position, body_padding=BODY_PADDING):
            candidates.append(record(identifier, kind, target, label, True))
            used.append(target)

    append('HOME', 'home', home, 'Recorded public takeoff/home position')
    directions = (
        ('FORWARD', (1., 0.), 'Forward'),
        ('LEFT', (0., 1.), 'Left'),
        ('RIGHT', (0., -1.), 'Right'),
        ('BACKWARD', (-1., 0.), 'Backward'),
        ('FORWARD_LEFT', (2**-.5, 2**-.5), 'Forward-left'),
        ('FORWARD_RIGHT', (2**-.5, -2**-.5), 'Forward-right'),
    )
    for identifier, (dx, dy), label in directions:
        target = local_to_world([[step_distance*dx, step_distance*dy, 0.]], home, anchor_yaw)[0]
        append(identifier, 'movement', target, label+' relative to the recorded takeoff heading')
    # Bound the model input and preserve the latest physically recorded anchors.
    for i, anchor in enumerate(list(visited_anchors)[-3:]):
        if isinstance(anchor, dict):
            identifier = 'VISIT_'+str(anchor.get('id', i))
            target = anchor['target_world']
        else:
            identifier, target = f'VISIT_{i}', anchor
        append(identifier, 'visited', target, 'Previously visited anchor; freshly rechecked')
    if len({c['id'] for c in candidates}) != len(candidates):
        raise ValueError('candidate IDs must be unique')
    return candidates


def render_candidate_bev(snapshot, position, yaw, home_world, candidates, output,
                         flight, operating_bounds=None, height_bounds=None, trail=(),
                         anchor_yaw=0.):
    """Write one 860x600 world-aligned centre-height BEV; return small metadata.

    Cells show nominal centre clearance/observation at the current world Z only.
    They do not certify other heights or continuous trajectories.  The caption
    makes current heading and world axes explicit, including after a turn.
    """
    from PIL import Image, ImageDraw, ImageFont

    position, home = _vector(position, 'position'), _vector(home_world, 'home_world')
    if not math.isfinite(yaw) or not math.isfinite(anchor_yaw):
        raise ValueError('yaw and anchor_yaw must be finite')
    safety = _safety(snapshot, position, yaw, flight, operating_bounds, height_bounds)
    xs = np.arange(-4.5, 4.5001, .2)+position[0]
    ys = np.arange(-4., 4.0001, .2)+position[1]
    xx, yy = np.meshgrid(xs, ys, indexing='xy')
    centres = np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, position[2])]
    clear = safety.clear_points(centres)
    known = safety.known_points(centres, position, body_padding=BODY_PADDING)
    pixels = np.full((len(ys), len(xs), 3), [170, 178, 185], dtype=np.uint8)
    pixels[(clear & known).reshape(len(ys), len(xs))] = [234, 244, 235]
    pixels[(~clear).reshape(len(ys), len(xs))] = [232, 157, 74]
    image = Image.new('RGB', (860, 600), 'white')
    draw = ImageDraw.Draw(image)
    fonts = {}
    for size in (12, 14, 16, 20):
        try:
            fonts[size] = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)
        except OSError:
            fonts[size] = ImageFont.load_default()
    draw.text((24, 12), 'OBSERVED SUBGOALS | world XY from home', fill='#1c2c38', font=fonts[20])
    map_box = (28, 57, 548, 521)
    raster = Image.fromarray(pixels[::-1]).resize((520, 464), resample=Image.Resampling.NEAREST)
    image.paste(raster, map_box[:2])
    draw.rectangle(map_box, outline='#354b59', width=2)

    def pixel(point):
        x = (float(point[0])-xs[0])/(xs[-1]-xs[0])
        y = (float(point[1])-ys[0])/(ys[-1]-ys[0])
        return (map_box[0]+x*(map_box[2]-map_box[0]), map_box[3]-y*(map_box[3]-map_box[1]))

    def in_map(point):
        return xs[0] <= point[0] <= xs[-1] and ys[0] <= point[1] <= ys[-1]

    for x in np.arange(math.ceil(xs[0]), xs[-1]+1e-8):
        a, b = pixel([x, ys[0]]), pixel([x, ys[-1]])
        draw.line([a, b], fill='#bbc5c5', width=1)
        draw.text((a[0]-8, map_box[3]+5), f'{x-home[0]:g}', fill='#344550', font=fonts[12])
    for y in np.arange(math.ceil(ys[0]), ys[-1]+1e-8):
        a, b = pixel([xs[0], y]), pixel([xs[-1], y])
        draw.line([a, b], fill='#bbc5c5', width=1)
        draw.text((2, a[1]-7), f'{y-home[1]:g}', fill='#344550', font=fonts[12])
    recent = np.asarray(trail, float).reshape(-1, 3)
    for a, b in zip(recent[:-1], recent[1:]):
        if in_map(a) and in_map(b):
            draw.line([pixel(a), pixel(b)], fill='#2380aa', width=3)
    if in_map(home):
        x, y = pixel(home)
        draw.rectangle((x-6, y-6, x+6, y+6), fill='#214e2d', outline='white', width=2)
        draw.text((x+8, y+7), 'HOME', fill='#164523', font=fonts[12])

    here = pixel(position)
    head = pixel(position+np.array([math.cos(yaw), math.sin(yaw), 0.])*.8)
    draw.line([here, head], fill='#c2382b', width=4)
    draw.ellipse((here[0]-5, here[1]-5, here[0]+5, here[1]+5), fill='#c2382b')
    tip = np.asarray(head); start = np.asarray(here)
    unit = (tip-start)/np.linalg.norm(tip-start)
    side = np.array([-unit[1], unit[0]])
    draw.polygon([tuple(tip), tuple(tip-12*unit+5*side), tuple(tip-12*unit-5*side)], fill='#c2382b')

    draw.text((572, 59), 'Candidate IDs', fill='#1c2c38', font=fonts[16])
    for number, candidate in enumerate(candidates):
        target = np.asarray(candidate['target_world'])
        if candidate['kind'] != 'hold' and in_map(target):
            x, y = pixel(target)
            draw.ellipse((x-10, y-10, x+10, y+10), fill='#225db0', outline='white', width=2)
            draw.text((x-4, y-8), str(number), fill='white', font=fonts[12])
        line_y = 91+number*36
        draw.text((572, line_y), f'{number}: {candidate["id"]}', fill='#153b60', font=fonts[14])
        relative = candidate['relative_body_xyz_m']
        draw.text((582, line_y+17), f'body [{relative[0]:+.1f}, {relative[1]:+.1f}, {relative[2]:+.1f}] m',
                  fill='#52606a', font=fonts[12])
    draw.text((572, 492), 'body: x forward, y left, z up', fill='#3c4a55', font=fonts[12])
    draw.text((572, 512), f'world Z = {position[2]:.2f} m', fill='#3c4a55', font=fonts[12])
    draw.text((572, 534), f'IDs fixed to takeoff yaw {np.rad2deg(anchor_yaw):.0f} deg',
              fill='#3c4a55', font=fonts[12])
    for x, color, label in ((28, '#eaf4eb', 'Observed clear'), (215, '#e89d4a', 'Blocked'), (350, '#aab2b9', 'Unknown')):
        draw.rectangle((x, 555, x+16, 571), fill=color, outline='#73818a')
        draw.text((x+22, 554), label, fill='#344550', font=fonts[14])
    draw.text((28, 577), 'Current-height nominal cells only. Red arrow = heading; blue trail = actual history.',
              fill='#52606a', font=fonts[12])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return dict(schema=SCHEMA, image=str(output.resolve()), image_size=[860, 600],
                axes='world XY; positions relative to recorded home',
                height_world_m=float(position[2]), evidence_timestamp=float(snapshot['timestamp']),
                candidates=len(candidates), occupied_cells=int((~clear).sum()),
                known_clear_cells=int((clear & known).sum()), unknown_cells=int((clear & ~known).sum()))


# Short alias for callers that already use render_* naming.
render_candidates = render_candidate_bev
