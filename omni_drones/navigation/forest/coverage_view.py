"""Compact local scan, observed topology and complete candidate legend for Qwen."""
import math
from pathlib import Path

import numpy as np


def render_coverage_input(snapshot, position, yaw, home, candidates, output, flight,
                          bounds, heights, trail, initial_yaw, structure, ledger):
    """Render only measured/derived inputs; blank topology areas remain unknown."""
    from PIL import Image, ImageDraw, ImageFont
    from .subgoal_inputs import render_candidate_bev

    output = Path(output)
    render_candidate_bev(snapshot, position, yaw, home, candidates, output, flight,
        operating_bounds=bounds, height_bounds=heights, trail=trail, anchor_yaw=initial_yaw)
    with Image.open(output) as original:
        local = original.crop((0, 45, 560, 548)).resize((490, 440))
    rows = math.ceil(len(candidates)/2)
    height = max(700, 535+rows*25)
    image = Image.new('RGB', (1000, height), '#fbfcfa')
    draw = ImageDraw.Draw(image)

    def font(size):
        try:
            return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)
        except OSError:
            return ImageFont.load_default()

    title, text, small = font(21), font(14), font(12)
    draw.text((15, 8), 'ONLINE FOREST COVERAGE | local scan + discovered topology', fill='#193448', font=title)
    draw.text((20, 40), f'Local current-height BEV | world Z={position[2]:.2f} m', fill='#314b5a', font=text)
    image.paste(local, (0, 60))
    draw.text((520, 40), 'Observed topology | blank = unknown', fill='#314b5a', font=text)

    corridors = structure.get('corridors', [])
    geometry = [np.asarray(position)[:2], np.asarray(home)[:2]]
    from .observed_terrain import query_terrain
    hits = np.asarray(snapshot['points']).reshape(-1, 3)
    hits = hits[::max(1, len(hits)//12000)]
    ground, valid = query_terrain(structure.get('terrain', {}), hits[:, :2])
    height_above_ground = hits[:, 2]-ground
    stem_hits = hits[valid & (height_above_ground >= .65) & (height_above_ground <= 3.)]
    geometry.extend(stem_hits[:, :2])
    stem_candidates = structure.get('stem_candidates', [])
    geometry.extend(s['xy'] for s in stem_candidates)
    structure_points = np.asarray(structure.get('structure_points_xy', []), float).reshape(-1, 2)
    geometry.extend(structure_points)
    for row in structure.get('rows', []):
        geometry.extend(np.asarray(row['spans']).reshape(-1, 2))
    for c in corridors:
        d, n = np.asarray(c['direction']), np.asarray(c['normal'])
        geometry.extend(np.asarray(c['span'])[:, None]*d+n*c['offset_m'])
    gaps = structure.get('unresolved_row_gaps', [])
    for gap in gaps:
        geometry.append(gap['observation_target_xy'])
    geometry = np.asarray(geometry)
    low, high = geometry.min(axis=0)-3., geometry.max(axis=0)+3.
    box = (523, 72, 978, 450)
    scale = min((box[2]-box[0])/max(high[0]-low[0], 1.),
                (box[3]-box[1])/max(high[1]-low[1], 1.))
    center = (low+high)/2

    def pixel(p):
        return ((box[0]+box[2])/2+(p[0]-center[0])*scale,
                (box[1]+box[3])/2-(p[1]-center[1])*scale)

    draw.rectangle(box, fill='#e8edf0', outline='#879ba7')
    for point in structure_points:
        x, y = pixel(point)
        draw.point((x, y), fill='#a5b3ad')
    for point in stem_hits:
        x, y = pixel(point)
        draw.rectangle((x-1, y-1, x+1, y+1), fill='#728575')
    for stem in stem_candidates:
        x, y = pixel(stem['xy'])
        radius = max(2., float(stem.get('radius_m', .15))*scale)
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline='#426649', width=2)
    for row in structure.get('rows', []):
        for span in row['spans']:
            draw.line([pixel(p) for p in span], fill='#46734d', width=3)
    for gap in gaps:
        d, n = np.asarray(gap['direction']), np.asarray(gap['normal'])
        a, b = gap['span']
        low_offset, high_offset = gap['lateral_span']
        corners = [d*a+n*low_offset, d*b+n*low_offset,
            d*b+n*high_offset, d*a+n*high_offset]
        draw.line([pixel(p) for p in corners+[corners[0]]], fill='#b68436', width=2)
        x, y = pixel(gap['observation_target_xy'])
        draw.text((x+3, y-6), '? '+gap['id'], fill='#8b5f24', font=small)
    for c in corridors:
        d, n = np.asarray(c['direction']), np.asarray(c['normal'])
        ends = np.asarray(c['span'])[:, None]*d+n*c['offset_m']
        draw.line([pixel(p) for p in ends], fill='#9cafc0', width=2)
        for side, p in zip(('low', 'high'), ends):
            x, y = pixel(p)
            draw.ellipse((x-3, y-3, x+3, y+3), outline='#41617c',
                fill='#41617c' if c['ends'][side].get('confirmed') else '#e8edf0')
        fixed = np.asarray(ledger.directions.get(c['id'], c['direction']))
        frame = np.stack([fixed, n])
        if abs(np.linalg.det(frame)) > .5:
            for a, b in ledger.covered.get(c['id'], []):
                points = np.linalg.solve(frame, np.array([[a, b], [c['offset_m']]*2])).T
                draw.line([pixel(p) for p in points], fill='#2379ac', width=4)
        x, y = pixel(ends.mean(axis=0))
        gap = sum(b-a for a, b in ledger.gaps(c))
        draw.text((x+4, y-17), f'{c["id"]} gap {gap:.1f}m', fill='#24465d', font=small)
    points = np.asarray(trail, float).reshape(-1, 3)
    if len(points) > 1:
        draw.line([pixel(p) for p in points], fill='#8b969a', width=1)
    x, y = pixel(home)
    draw.rectangle((x-4, y-4, x+4, y+4), fill='#da9a26')
    draw.text((x+5, y+3), 'HOME', fill='#725113', font=small)
    x, y = pixel(position)
    draw.ellipse((x-5, y-5, x+5, y+5), fill='#c13e36')
    draw.line([(x, y), (x+14*np.cos(yaw), y-14*np.sin(yaw))], fill='#c13e36', width=3)
    draw.text((520, 458), 'green: stem-band returns/rows | blue: measured coverage | dots: ends',
        fill='#3f5660', font=small)
    draw.text((20, 503), 'Complete candidate legend | numbers match local markers | body XYZ metres',
        fill='#193448', font=text)
    for i, c in enumerate(candidates):
        column, row = divmod(i, rows)
        relative = c['relative_body_xyz_m']
        label = f'{i}: {c["candidate_id"]}  [{relative[0]:+.1f}, {relative[1]:+.1f}, {relative[2]:+.1f}]'
        draw.text((20+column*495, 530+row*25), label, fill='#24465d', font=text)
    image.save(output)
    return dict(image=str(output.resolve()), image_size=list(image.size),
        schema='observed_native_coverage_input_v1', candidates=len(candidates),
        provenance='current LiDAR BEV, online row supports, measured route ledger')
