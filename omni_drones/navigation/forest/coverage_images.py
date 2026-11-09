"""Two label-only decision images: a work-area overview and a local BEV.

Both are world-aligned (x right, y up), so the BEV is a zoom of the overview
box. Text in the images is limited to aisle numbers and candidate letters; the
legend lives in the prompt. Only online observations are drawn.
"""
from pathlib import Path

import numpy as np

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
UNKNOWN, FREE, BAND = (150, 156, 162), (238, 243, 234), (38, 84, 44)
COVERED, OPEN, TRAIL = (36, 120, 214), (240, 132, 30), (70, 80, 100)
BLOCKED, GRID, CANDIDATE = (236, 150, 60), (205, 210, 205), (200, 20, 120)
DRONE, HOME = (220, 30, 30), (0, 0, 0)
GLOBAL_MAX_PIXELS, GLOBAL_MAX_WIDTH = 262144, 896
BEV_HALF, BEV_SCALE, BEV_CELL = (5., 4.), 50., .2
# A route shorter than this is a local move: drawn on the BEV only.
LOCAL_MOVE = 3.


def _font(size):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def _drone(draw, u, v, yaw, size=1.):
    tri = np.array([[14., 0.], [-8., 9.], [-8., -9.]])*size
    q = tri@np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]).T
    draw.polygon([(u+a, v-b) for a, b in q], fill=DRONE, outline=(255, 255, 255))


def _label(draw, u, v, text, filled=True, radius=10):
    draw.ellipse([u-radius, v-radius, u+radius, v+radius], fill=CANDIDATE if filled else (255, 255, 255),
                 outline=CANDIDATE, width=2)
    draw.text((u, v), text, fill=(255, 255, 255) if filled else CANDIDATE, font=_font(13), anchor='mm')


def candidate_path(candidate, position, corridors):
    """World XY polyline from the vehicle to the candidate's committed end."""
    path = [np.asarray(position, float)[:2]]
    path += [np.asarray(p['target_world'], float)[:2] for p in (candidate.get('route') or [candidate])]
    commitment = candidate.get('commitment') or {}
    corridor = corridors.get(commitment.get('corridor_id'))
    if corridor is not None and candidate.get('role') == 'INSPECT' and commitment.get('direction'):
        d, n = np.asarray(corridor['direction'], float), np.asarray(corridor['normal'], float)
        end = corridor['span'][1 if commitment['direction'] > 0 else 0]
        scope = commitment.get('scope') or {}
        if scope.get('endpoint_s') is not None:
            end = scope['endpoint_s']
        elif commitment.get('until_s') is not None:
            end = commitment['until_s']
        path.append(d*float(end)+n*float(corridor['offset_m']))
    return np.asarray(path)


def render_global(structure, coverage, order, position, yaw, home, trail, candidates, labels, output,
                  max_pixels=GLOBAL_MAX_PIXELS, max_width=GLOBAL_MAX_WIDTH, crop='work_area'):
    """Work-area overview; returns the drawn extent for the record.

    ``crop='structure'`` frames the discovered aisles, vehicle and home instead
    of the whole work area, for more pixels per aisle.
    """
    from PIL import Image, ImageDraw
    corridors = {c['id']: c for c in structure.get('corridors', [])}
    area = (structure.get('public_work_area') or {}).get('xy_bounds')
    if crop == 'structure' and corridors:
        points = [np.asarray(position)[:2], np.asarray(home)[:2]]
        points += [np.asarray(c['span'])[:, None]*np.asarray(c['direction'])+np.asarray(c['normal'])*c['offset_m']
                   for c in corridors.values()]
        points = np.vstack([np.reshape(p, (-1, 2)) for p in points])
        area = [points[:, 0].min()-6, points[:, 0].max()+6, points[:, 1].min()-5, points[:, 1].max()+5]
    if area is None:
        points = [np.asarray(position)[:2], np.asarray(home)[:2]]
        points += [np.asarray(c['span'])[:, None]*np.asarray(c['direction'])+np.asarray(c['normal'])*c['offset_m']
                   for c in corridors.values()]
        points = np.vstack([np.reshape(p, (-1, 2)) for p in points])
        area = [points[:, 0].min()-5, points[:, 0].max()+5, points[:, 1].min()-5, points[:, 1].max()+5]
    x0, x1, y0, y1 = area[0]-2., area[1]+2., area[2]-2., area[3]+2.
    scale = min(np.sqrt(max_pixels/((x1-x0)*(y1-y0))), max_width/(x1-x0))
    W, H = int((x1-x0)*scale), int((y1-y0)*scale)

    def px(p):
        p = np.asarray(p, float)
        return (p[..., 0]-x0)*scale, H-(p[..., 1]-y0)*scale

    pixels = np.empty((H, W, 3), np.uint8)
    pixels[:] = UNKNOWN
    overview = structure.get('overview')
    if overview is not None:
        flags, origin, res = np.asarray(overview['flags']), np.asarray(overview['origin']), overview['resolution']
        for flag, color in ((1, FREE), (2, BAND)):
            i, j = np.nonzero(flags & flag)
            xy = origin+(np.c_[i, j]+.5)*res
            u, v = px(xy)
            half = max(1, int(np.ceil(res*scale/2)))
            for du in range(-half+1, half+1):
                for dv in range(-half+1, half+1):
                    pixels[np.clip((v+dv).astype(int), 0, H-1), np.clip((u+du).astype(int), 0, W-1)] = color
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    a, b = px([area[0], area[3]]), px([area[1], area[2]])
    draw.rectangle([a[0], a[1], b[0], b[1]], outline=(70, 70, 70), width=1)
    for stem in structure.get('stem_candidates', []):
        u, v = px(stem['xy'])
        r = max(1.5, float(stem.get('radius_m', .2))*scale)
        draw.ellipse([u-r, v-r, u+r, v+r], fill=BAND)
    trail = np.asarray(trail, float).reshape(-1, 3)
    if len(trail) > 1:
        u, v = px(trail[:, :2])
        draw.line(list(zip(u.tolist(), v.tolist())), fill=TRAIL, width=1)
    ledger = {c['corridor_id']: c for c in coverage.get('corridors', [])}
    for cid, c in corridors.items():
        d, n, off = np.asarray(c['direction'], float), np.asarray(c['normal'], float), float(c['offset_m'])
        lo, hi = map(float, c['span'])

        def segment(s0, s1):
            return [tuple(map(float, px(d*s0+n*off))), tuple(map(float, px(d*s1+n*off)))]
        draw.line(segment(lo, hi), fill=OPEN, width=4)
        row = ledger.get(cid, {})
        fixed = np.asarray(row.get('along_direction', d), float)
        for s0, s1 in row.get('covered', []):
            # Ledger intervals use their own fixed axis; map back onto this aisle.
            s0, s1 = [(s-off*(n@fixed))/(d@fixed) for s in (s0, s1)]
            if min(s0, s1) < hi and max(s0, s1) > lo:
                draw.line(segment(max(min(s0, s1), lo), min(max(s0, s1), hi)), fill=COVERED, width=4)
        for side, s in (('low', lo), ('high', hi)):
            u, v = px(d*s+n*off)
            confirmed = bool(c['ends'][side].get('confirmed'))
            draw.ellipse([u-4, v-4, u+4, v+4], fill=(20, 20, 20) if confirmed else (255, 255, 255),
                         outline=(20, 20, 20), width=2)
        index = order.get(cid, {}).get('lateral_index')
        if index is not None:
            # Number beyond the aisle end drawn further left.
            end, other = (lo, hi) if px(d*lo+n*off)[0] <= px(d*hi+n*off)[0] else (hi, lo)
            u, v = px(d*(end-2.4*np.sign(other-end))+n*off)
            draw.rectangle([u-8, v-8, u+8, v+8], fill=(255, 255, 255), outline=(20, 20, 20))
            draw.text((u, v), str(index), fill=(0, 0, 0), font=_font(12), anchor='mm')
    u, v = px(np.asarray(home)[:2])
    draw.rectangle([u-5, v-5, u+5, v+5], fill=HOME, outline=(255, 255, 255))
    p = np.asarray(position, float)[:2]
    a, b = px(p+[-BEV_HALF[0], BEV_HALF[1]]), px(p+[BEV_HALF[0], -BEV_HALF[1]])
    draw.rectangle([a[0], a[1], b[0], b[1]], outline=(20, 20, 20), width=1)
    drawn = []
    for c in candidates:
        if c['candidate_id'] not in labels:
            continue
        path = candidate_path(c, position, corridors)
        if np.linalg.norm(path[-1]-p) < LOCAL_MOVE:
            continue
        uv = [tuple(map(float, px(q))) for q in path]
        draw.line(uv, fill=CANDIDATE, width=2)
        drawn.append((uv[-1], labels[c['candidate_id']]))
    for (u, v), text in drawn:
        _label(draw, u, v, text, radius=9)
    u, v = map(float, px(p))
    _drone(draw, u, v, yaw)
    image.save(output)
    return dict(xy_bounds=[x0, x1, y0, y1], size=[W, H], metres_per_pixel=1/scale,
                global_labels=[t for _, t in drawn])


def render_bev(safety, position, yaw, candidates, labels, output, ground=None):
    """World-aligned local slice following the terrain at the current clearance.

    Orange is an observed obstacle within the body clearance; grey lacks free
    evidence. Points outside the planner's local box are NOT marked blocked.
    """
    from PIL import Image, ImageDraw
    from .geometry import center_clearance, collision_free
    hx, hy = BEV_HALF
    W, H = int(2*hx*BEV_SCALE), int(2*hy*BEV_SCALE)
    position = np.asarray(position, float)
    xs = np.arange(-hx+BEV_CELL/2, hx, BEV_CELL)
    ys = np.arange(-hy+BEV_CELL/2, hy, BEV_CELL)
    xx, yy = np.meshgrid(xs, ys, indexing='xy')
    xy = np.c_[xx.ravel(), yy.ravel()]+position[:2]
    z = np.full(len(xy), position[2])
    if ground is not None:
        heights, valid = ground(np.vstack([position[:2], xy]))
        if valid[0]:
            z = np.where(valid[1:], position[2]+heights[1:]-heights[0], position[2])
    centres = np.c_[xy, z]
    clear = collision_free(safety.obstacles, centres, safety.clearance_reserve, center_clearance(safety.flight))
    known = safety.known_points(centres, position)
    pixels = np.empty((len(ys), len(xs), 3), np.uint8)
    pixels[:] = UNKNOWN
    pixels[(clear & known).reshape(len(ys), len(xs))] = FREE
    pixels[(~clear).reshape(len(ys), len(xs))] = BLOCKED
    image = Image.fromarray(pixels[::-1]).resize((W, H), resample=Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)

    def px(p):
        b = np.asarray(p, float)[..., :2]-position[:2]
        return float((b[..., 0]+hx)*BEV_SCALE), float((hy-b[..., 1])*BEV_SCALE)
    for m in np.arange(-np.floor(hx), hx+1e-9):
        u = (m+hx)*BEV_SCALE
        draw.line([(u, 0), (u, H)], fill=GRID, width=1)
    for m in np.arange(-np.floor(hy), hy+1e-9):
        v = (hy-m)*BEV_SCALE
        draw.line([(0, v), (W, v)], fill=GRID, width=1)
    cu, cv = px(position)
    placed = []
    for c in candidates:
        if c['candidate_id'] not in labels or c.get('role') in ('HOLD', 'STOP'):
            continue
        tu, tv = px(c['target_world'])
        inside = 0 <= tu < W and 0 <= tv < H
        if not inside:
            # Clamp onto the border along the bearing from the vehicle.
            du, dv = tu-cu, tv-cv
            k = min(x for x in ((W-12-cu)/du if du > 0 else (12-cu)/du if du < 0 else np.inf,
                                (H-12-cv)/dv if dv > 0 else (12-cv)/dv if dv < 0 else np.inf) if x > 0)
            tu, tv = cu+k*du, cv+k*dv
        u, v = tu, tv
        ray = np.array([u-cu, v-cv])
        ray = ray/np.linalg.norm(ray) if np.linalg.norm(ray) > 1 else np.array([0., -1.])
        # Push overlapping labels outward from the vehicle; a leader marks the point.
        for _ in range(40):
            if all(np.hypot(u-a, v-b) >= 24 for a, b in placed):
                break
            u, v = u+5*ray[0], v+5*ray[1]
        placed.append((u, v))
        if inside and np.hypot(u-tu, v-tv) > 1:
            draw.line([(tu, tv), (u, v)], fill=CANDIDATE, width=1)
            draw.ellipse([tu-3, tv-3, tu+3, tv+3], fill=CANDIDATE)
        _label(draw, u, v, labels[c['candidate_id']], filled=inside, radius=11)
    _drone(draw, cu, cv, yaw)
    image.save(output)
    return dict(half_extent_m=list(BEV_HALF), size=[W, H], cell_m=BEV_CELL)
