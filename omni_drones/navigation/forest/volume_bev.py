"""Pixel-preserving atlas of the full observation grid; no teacher routes."""
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import (collision_free, center_clearance, inside_polygon, observed_centers,
                       observed_body_height, RADIUS)
from .waypoint_teacher_xyz import GRID_LOWER, GRID_UPPER, observed_grid, observation_safety

NAME = 'volume_atlas_v1'
HEIGHTS = np.round(np.arange(GRID_LOWER[2], GRID_UPPER[2]+.1, .2), 6)
HEIGHTS[np.abs(HEIGHTS)<1e-6] = 0.
PALETTE = np.array([[140, 140, 140], [232, 244, 249], [255, 149, 48],
                    [48, 53, 62], [142, 188, 255]], dtype=np.uint8)
LABELS = ('Unknown', 'Observed free', 'Blocked', 'Outside corridor', 'Actual body support')
CELL_PIXELS, TILE, HEADER, COLUMNS = 4, 288, 64, 5
SIZE = (COLUMNS*TILE, HEADER+3*TILE)


def grid_states(obs, grid=None):
    if grid is None:
        grid, safety = observed_grid(obs)
    else:
        safety = observation_safety(obs)
    if grid['free'].shape != (61, 61, 15):
        raise ValueError('atlas requires the complete 61x61x15 planning grid')
    cells = np.indices(grid['free'].shape).reshape(3, -1).T
    centers = grid['origin']+cells*float(grid['step'])
    blocked = ~collision_free(safety.obstacles, centers, clearance=center_clearance(safety.flight))
    allowed = (inside_polygon(centers[:, :2], safety.corridor) if safety.corridor is not None
               else np.ones(len(centers), bool))
    free = grid['free'].ravel()
    states = np.zeros(len(centers), np.uint8)
    states[free] = 1
    states[blocked] = 2
    states[~allowed] = 3
    vertical = observed_body_height(safety.flight.max_acceleration)
    # Only centers near the current body can receive additional body evidence.
    near = np.flatnonzero(free & (np.linalg.norm(centers[:, :2], axis=1) <= 2*RADIUS+.05)
                         & (np.abs(centers[:, 2]) <= 2*vertical+.05))
    ray_known = observed_centers(safety.evidence, centers[near], vertical=vertical)
    states[near[~ray_known]] = 4
    if not np.array_equal(np.isin(states, (1, 4)), free):
        raise ValueError('image free space disagrees with shared planning grid')
    return states.reshape(grid['free'].shape)


def panel_origin(index):
    row, column = divmod(index, COLUMNS)
    return column*TILE+32, HEADER+row*TILE+28


def render_atlas(states, output):
    """Every grid cell occupies exactly 4x4 RGB pixels, without overlays."""
    canvas = Image.new('RGB', SIZE, 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('DejaVuSans.ttf', 12)
    title = ImageFont.truetype('DejaVuSans.ttf', 16)
    draw.text((16, 7), 'Local 3D observation | 15 center heights | grid 0.2 m | x forward, y left | origin (0,0,0)', fill='black', font=title)
    for index, (color, label) in enumerate(zip(PALETTE, LABELS)):
        x = 16+index*276
        draw.rectangle((x, 36, x+18, 49), fill=tuple(color))
        draw.text((x+24, 35), label, fill='black', font=font)
    for index, z in enumerate(HEIGHTS):
        x, y = panel_origin(index)
        pixels = PALETTE[states[:, ::-1, index].T]
        layer = Image.fromarray(pixels).resize((244, 244), Image.Resampling.NEAREST)
        canvas.paste(layer, (x, y))
        draw.text((x, y-22), f'{index+1:02d}   center z={z:+.1f} m', fill='black', font=title)
        for cell in range(0, 61, 10):
            x_value, y_value = -2+cell*.2, 6-cell*.2
            draw.text((x+cell*4+2, y+246), f'{x_value:g}', anchor='mt', fill='blue' if x_value==0 else 'black', font=font)
            draw.text((x-4, y+cell*4+2), f'{y_value:g}', anchor='rm', fill='blue' if y_value==0 else 'black', font=font)
    canvas.save(output)


def input_prompt(obs):
    state = {name:np.round(obs[key], 3).tolist() for name,key in
             [('目标XYZ','goal'), ('速度XYZ_m/s','velocity'), ('加速度XYZ_m/s²','acceleration')]}
    flight = json.loads(str(obs['flight_json']))
    limits = {k:flight[k] for k in ('max_speed','max_acceleration','max_jerk')}
    return '\n'.join([
        '无人机局部三维BEV拼图。坐标相对当前机体：x前、y左、z上，单位米；当前位置XYZ=[0,0,0]。',
        '共15层，按从左到右、从上到下排列，每行5层。中心高度z='+str(HEIGHTS.tolist())+'米。',
        '每层是相同61×61格，分辨率0.2米；x从左至右为−2到10米，y从上至下为6到−6米。',
        '橙色=净空不满足，浅色=已观测自由，浅蓝=当前真实机体补充的自由证据，灰色=未知，深灰=任务通道外。',
        '每格表示该机体中心位置的状态，已计入三维0.75米中心净空；不要再次扣除机体半径。',
        '格心自由不保证格间和层间路线可行；输出连续XYZ，整段重建曲线与制动空间仍须通过检查。',
        '允许左右与上下绕行，保留地形跟随。输出点范围：x=[-2,10]，y=[-6,6]，z=[-1.4,1.4]。',
        json.dumps(state,ensure_ascii=False,separators=(',',':')),
        '运动上限='+json.dumps(limits,separators=(',',':'))+'。',
        '观测仅来自当前与过去3秒扫描，不包含未来观测。灰色与通道外不能当作自由。',
        '根据目标与观测提出前方局部路线。仅输出JSON：',
        '{"status":"GO","points":[[x1,y1,z1],[x2,y2,z2],[x3,y3,z3],[x4,y4,z4],[x5,y5,z5],[x6,y6,z6]]}',
        'GO必须恰好6个未来航点。已到达目标且停止动作可验证时，输出{"status":"HOLD","points":[]}；无法确认路线不等于可以安全停止。',
    ])+'\n'


def decoded_processor_image(processor, encoded):
    """Invert Qwen's patch packing to inspect pixels actually sent to vision."""
    p = processor.image_processor
    t, h, w = encoded['image_grid_thw'][0].tolist()
    patch, merge, temporal = p.patch_size, p.merge_size, p.temporal_patch_size
    if t != 1:
        raise ValueError('one static atlas image required')
    packed = encoded['pixel_values'].reshape(h//merge, w//merge, merge, merge, 3, temporal, patch, patch)
    rgb = packed[:, :, :, :, :, 0].permute(4, 0, 2, 5, 1, 3, 6).reshape(3, h*patch, w*patch)
    rgb = rgb.detach().cpu().numpy().transpose(1, 2, 0)
    rgb = rgb*np.asarray(p.image_std)+np.asarray(p.image_mean)
    return np.rint(rgb*255).clip(0, 255).astype(np.uint8)


def decode_states(rgb):
    if rgb.shape != (SIZE[1], SIZE[0], 3):
        raise ValueError('processor resized the atlas; cell visibility must be revalidated')
    states = np.empty((61, 61, 15), np.uint8)
    for index in range(15):
        x, y = panel_origin(index)
        tile = rgb[y:y+244, x:x+244].reshape(61, 4, 61, 4, 3)
        centers = tile[:, 2, :, 2]
        distance = np.abs(centers[:, :, None].astype(int)-PALETTE.astype(int)).max(axis=-1)
        if distance.min(axis=-1).max() > 1:
            raise ValueError('cell colors were lost during image preprocessing')
        if np.abs(tile.astype(int)-centers[:, None, :, None].astype(int)).max() > 1:
            raise ValueError('cell pixels were blended or overdrawn')
        states[:, :, index] = distance.argmin(axis=-1)[::-1].T
    return states
